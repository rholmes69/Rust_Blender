//! Editor state (scene + selection + undo + playback) and the command executor
//! shared by the UI, the AI assistant and voice control.

use crate::editmesh::{self as em, EditState, SelectMode};
use crate::mesh::Primitive;
use crate::scene::{Modifier, Scene};
use glam::Vec3;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Shading {
    Wireframe,
    Solid,
    Material,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ViewPreset {
    Front,
    Back,
    Left,
    Right,
    Top,
    Bottom,
    Iso,
}

/// Requests from the executor to the viewport (which owns the camera).
#[derive(Clone, Debug, PartialEq)]
pub enum ViewRequest {
    Preset(ViewPreset),
    ToggleOrtho,
    FrameSelected,
    FrameAll,
    Zoom(f32),
    Orbit(f32, f32),
    /// Toggle looking through the active camera.
    Camera,
    AlignCameraToView,
}

pub struct Editor {
    pub scene: Scene,
    pub selection: Vec<u64>,
    pub active: Option<u64>,
    pub frame: i32,
    pub playing: bool,
    pub shading: Shading,
    pub undo_stack: Vec<(Scene, Vec<u64>)>,
    pub redo_stack: Vec<(Scene, Vec<u64>)>,
    pub view_requests: Vec<ViewRequest>,
    pub dirty: bool,
    pub file_path: Option<std::path::PathBuf>,
    pub auto_key: bool,
    /// Edit mode (component editing of one mesh object).
    pub edit: Option<EditState>,
    /// Set by the `render` command; the app renders an image (F12) and clears it.
    pub render_request: bool,
    /// Sculpt mode on this mesh object.
    pub sculpt: Option<u64>,
    /// Armature in Pose Mode and its active bone.
    pub pose: Option<u64>,
    pub active_bone: usize,
    /// Last known world matrix per object, used to make children follow their parents.
    parent_cache: std::collections::HashMap<u64, glam::Mat4>,
    /// True while a command batch runs (the batch owns the undo step).
    in_batch: bool,
}

impl Default for Editor {
    fn default() -> Self {
        let scene = Scene::default();
        let first = scene.objects.first().map(|o| o.id);
        Self {
            scene,
            selection: first.into_iter().collect(),
            active: first,
            frame: 1,
            playing: false,
            shading: Shading::Solid,
            undo_stack: vec![],
            redo_stack: vec![],
            view_requests: vec![],
            dirty: false,
            file_path: None,
            auto_key: false,
            edit: None,
            render_request: false,
            parent_cache: Default::default(),
            sculpt: None,
            pose: None,
            active_bone: 0,
            in_batch: false,
        }
    }
}

impl Editor {
    pub fn snapshot(&mut self) {
        self.undo_stack
            .push((self.scene.clone(), self.selection.clone()));
        if self.undo_stack.len() > 200 {
            self.undo_stack.remove(0);
        }
        self.redo_stack.clear();
        self.dirty = true;
    }
    /// Make children follow parents that moved since the last call. Children that were moved
    /// themselves in the same step (e.g. grabbed together with the parent) are left alone.
    /// Bind a mesh to an armature with automatic weights (the mesh's current shape is the bind
    /// pose; the armature's rest position is used).
    pub fn bind_to_armature(&mut self, mesh_id: u64, arm_id: u64) -> Result<(), String> {
        let a = self.scene.get(arm_id).ok_or("no armature")?;
        let arm = a.armature.clone().ok_or("not an armature")?;
        let aw = a.transform.matrix();
        let o = self.scene.get(mesh_id).ok_or("no mesh")?;
        if !o.is_mesh() {
            return Err("only meshes can be bound".into());
        }
        let pm = o.poly();
        let om = o.transform.matrix();
        let (j, w) = crate::armature::auto_weights(&pm, om, &arm, aw);
        let skin = crate::armature::Skin::bind(arm_id, &pm, om, &arm, aw, j, w);
        let o = self.scene.get_mut(mesh_id).unwrap();
        o.mesh = Some(pm);
        o.skin = Some(skin);
        o.posed = None;
        o.parent = Some(arm_id);
        Ok(())
    }

    /// Add an armature fitted to a mesh (a chain along its longest axis) and bind the mesh.
    pub fn rig(&mut self, mesh_id: u64, bones: usize) -> Result<u64, String> {
        let o = self.scene.get(mesh_id).ok_or("no mesh")?;
        let m = o.transform.matrix();
        let pm = o.poly();
        let (lo, hi) = pm.verts.iter().fold((Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)), |(l, h), v| {
            let w = m.transform_point3(*v);
            (l.min(w), h.max(w))
        });
        if lo.x > hi.x {
            return Err("the mesh is empty".into());
        }
        let size = hi - lo;
        let c = (lo + hi) * 0.5;
        let name = format!("{} Rig", o.name);
        let coll = o.collection;
        // upright shapes get a spine from the bottom up; long ones a chain along x or y
        let (start, rot, len) = if size.z >= size.x * 0.8 && size.z >= size.y * 0.8 {
            (Vec3::new(c.x, c.y, lo.z), Vec3::ZERO, size.z)
        } else if size.x >= size.y {
            (Vec3::new(lo.x, c.y, c.z), Vec3::new(0.0, 90.0, 0.0), size.x)
        } else {
            (Vec3::new(c.x, lo.y, c.z), Vec3::new(-90.0, 0.0, 0.0), size.y)
        };
        // coarse meshes can't bend smoothly: add rings first (keeps the shape)
        if pm.verts.len() < 600 {
            let mut dense = pm.clone();
            crate::sculpt::densify(&mut dense, 600, false);
            self.scene.get_mut(mesh_id).unwrap().mesh = Some(dense);
        }
        let arm_id = self.scene.add_armature(crate::armature::Armature::chain(bones, len.max(1e-3)), Some(&name));
        let a = self.scene.get_mut(arm_id).unwrap();
        a.transform.location = start;
        a.transform.rotation = rot;
        a.collection = coll;
        self.bind_to_armature(mesh_id, arm_id)?;
        self.reset_parent_cache();
        Ok(arm_id)
    }

    /// A texture by spec: an image path, an existing texture's name or a procedural pattern
    /// ("none" etc. → 0).
    pub fn resolve_texture(&mut self, spec: &str) -> Result<(u32, String), String> {
        let spec_t = spec.trim();
        let low = spec_t.to_lowercase();
        Ok(if low.is_empty() || matches!(low.as_str(), "none" | "remove" | "off" | "clear") {
            (0, "no texture".to_string())
        } else if std::path::Path::new(spec_t).is_file() {
            let data = std::fs::read(spec_t).map_err(|e| format!("can't read {spec_t}: {e}"))?;
            crate::textures::decode(&crate::textures::TexSource::Image { data: data.clone() })?;
            let name = std::path::Path::new(spec_t).file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or("Image".into());
            (self.scene.add_texture(&name, crate::textures::TexSource::Image { data }), format!("image {name}"))
        } else if let Some(t) = self.scene.textures.iter().find(|t| t.name.eq_ignore_ascii_case(spec_t)) {
            (t.id, t.name.clone())
        } else if let Some(p) = crate::textures::match_pattern(spec_t) {
            let (a, b) = crate::textures::procedural_defaults(p).unwrap();
            (self.scene.add_texture(p, crate::textures::TexSource::Procedural { pattern: p.into(), color_a: a, color_b: b }), format!("{p} texture"))
        } else {
            let list: Vec<&str> = crate::textures::PROCEDURAL.iter().map(|p| p.0).collect();
            return Err(format!("unknown texture '{spec_t}' (use an image path or one of: {})", list.join(", ")));
        })
    }

    pub fn propagate_parents(&mut self) {
        let order = self.scene.hierarchy_order();
        let mut changed: std::collections::HashMap<u64, (glam::Mat4, glam::Mat4)> = Default::default();
        for id in order {
            let parent = self.scene.get(id).and_then(|o| o.parent);
            if let Some(p) = parent {
                if self.scene.get(p).is_none() || self.scene.would_cycle(id, p) {
                    self.scene.get_mut(id).unwrap().parent = None;
                }
            }
            let Some(o) = self.scene.get(id) else { continue };
            let mut cur = o.transform.matrix();
            let old = self.parent_cache.get(&id).copied();
            if let (Some(p), Some(old_m)) = (o.parent, old) {
                // keyed children are placed by their own keys (world space)
                if old_m == cur && o.keyframes.is_empty() {
                    if let Some((po, pn)) = changed.get(&p) {
                        let new = *pn * po.inverse() * cur;
                        let o = self.scene.get_mut(id).unwrap();
                        o.transform = crate::scene::Transform::from_matrix(new);
                        cur = o.transform.matrix();
                    }
                }
            }
            if old != Some(cur) {
                changed.insert(id, (old.unwrap_or(cur), cur));
            }
        }
        self.reset_parent_cache();
    }

    /// Destructive mesh boolean: `main` becomes main ∘ cutters; cutters are hidden in a
    /// "Cutters" collection (so they can be reused).
    pub fn boolean(&mut self, main: u64, cutters: &[u64], op: crate::csg::BoolOp) -> Result<String, String> {
        use crate::csg::Polygon;
        use glam::DVec3;
        let mut cache = crate::scene::MeshCache::default();
        let to_polys = |o: &crate::scene::Object, cache: &mut crate::scene::MeshCache, with_instances: bool| -> Vec<Polygon> {
            let mesh = cache.for_object(o);
            let mats = if with_instances { o.instance_matrices() } else { vec![glam::Mat4::IDENTITY] };
            let mut out = vec![];
            for inst in mats {
                let m = (o.transform.matrix() * inst).as_dmat4();
                let flip = m.determinant() < 0.0;
                for t in &mesh.tris {
                    let mut v: Vec<DVec3> = t.iter().map(|&i| m.transform_point3(mesh.positions[i as usize].as_dvec3())).collect();
                    if flip {
                        v.swap(1, 2);
                    }
                    if let Some(p) = Polygon::new(v) {
                        out.push(p);
                    }
                }
            }
            out
        };
        let mo = self.scene.get(main).ok_or("main object is gone")?;
        if !mo.is_mesh() {
            return Err("booleans work on mesh objects".into());
        }
        let mut acc = to_polys(mo, &mut cache, false);
        let mut names = vec![];
        for &cid in cutters {
            let Some(co) = self.scene.get(cid) else { continue };
            if !co.is_mesh() || cid == main {
                continue;
            }
            names.push(co.name.clone());
            let cp = to_polys(co, &mut cache, true);
            acc = crate::csg::boolean(acc, cp, op);
        }
        if names.is_empty() {
            return Err("no mesh cutter selected".into());
        }
        if acc.is_empty() {
            return Err(format!("the {} left nothing (do the objects overlap?)", op.label().to_lowercase()));
        }
        let inv = self.scene.get(main).unwrap().transform.matrix().as_dmat4().inverse();
        for p in &mut acc {
            for v in &mut p.verts {
                *v = inv.transform_point3(*v);
            }
        }
        let size = acc.iter().flat_map(|p| p.verts.iter()).fold(0.0f64, |m, v| m.max(v.abs().max_element()));
        let pm = crate::csg::to_polymesh(&acc, (size * 1e-6).max(1e-9));
        let nfaces = pm.faces.len();
        let cut_col = self.scene.collection_by_name("Cutters");
        let o = self.scene.get_mut(main).unwrap();
        o.mesh = Some(pm);
        o.modifiers.retain(|m| !matches!(m, Modifier::Subsurf { .. }));
        if o.smooth() && o.auto_smooth == 0.0 {
            o.auto_smooth = 30.0; // keep the new cut edges sharp
        }
        let main_name = o.name.clone();
        for &cid in cutters {
            if let Some(co) = self.scene.get_mut(cid) {
                if co.is_mesh() && cid != main {
                    co.visible = false;
                    co.collection = cut_col;
                }
            }
        }
        if let Some(c) = self.scene.collections.iter_mut().find(|c| c.id == cut_col) {
            c.visible = false;
        }
        self.select_only(main);
        Ok(format!("Boolean {} on \"{main_name}\" with {} ({nfaces} faces; cutter hidden in \"Cutters\")", op.label(), names.join(", ")))
    }

    /// Forget previous positions (after undo, opening a file …) so nothing jumps.
    pub fn reset_parent_cache(&mut self) {
        self.parent_cache = self.scene.objects.iter().map(|o| (o.id, o.transform.matrix())).collect();
    }

    pub fn undo(&mut self) -> bool {
        if let Some((s, sel)) = self.undo_stack.pop() {
            self.redo_stack.push((
                std::mem::replace(&mut self.scene, s),
                std::mem::replace(&mut self.selection, sel),
            ));
            self.fix_active();
            self.reset_parent_cache();
            true
        } else {
            false
        }
    }
    pub fn redo(&mut self) -> bool {
        if let Some((s, sel)) = self.redo_stack.pop() {
            self.undo_stack.push((
                std::mem::replace(&mut self.scene, s),
                std::mem::replace(&mut self.selection, sel),
            ));
            self.fix_active();
            self.reset_parent_cache();
            true
        } else {
            false
        }
    }
    /// Leave edit mode if the edited object no longer exists (deleted, undone, …).
    pub fn validate_edit(&mut self) {
        if let Some(id) = self.pose {
            if !self.scene.get(id).is_some_and(|o| o.armature.is_some()) {
                self.pose = None;
            }
        }
        if let Some(id) = self.sculpt {
            if !self.scene.get(id).is_some_and(|o| o.is_mesh() && o.mesh.is_some()) {
                self.sculpt = None;
            }
        }
        if let Some(st) = &self.edit {
            if !self
                .scene
                .get(st.obj)
                .is_some_and(|o| o.is_mesh() && o.mesh.is_some())
            {
                self.edit = None;
            }
        }
    }

    fn fix_active(&mut self) {
        self.validate_edit();
        self.selection.retain(|id| self.scene.get(*id).is_some());
        if self.active.is_none_or(|a| !self.selection.contains(&a)) {
            self.active = self.selection.last().copied();
        }
    }

    pub fn select_only(&mut self, id: u64) {
        self.selection = vec![id];
        self.active = Some(id);
    }
    pub fn toggle_select(&mut self, id: u64) {
        if let Some(i) = self.selection.iter().position(|&s| s == id) {
            if self.active == Some(id) {
                self.selection.remove(i);
                self.active = self.selection.last().copied();
            } else {
                self.active = Some(id);
            }
        } else {
            self.selection.push(id);
            self.active = Some(id);
        }
    }
    pub fn select_all_toggle(&mut self) {
        if self.selection.len() == self.scene.objects.len() {
            self.selection.clear();
            self.active = None;
        } else {
            self.selection = self.scene.objects.iter().map(|o| o.id).collect();
            self.active = self.selection.last().copied();
        }
    }

    pub fn add_primitive(&mut self, p: Primitive, at: Vec3) -> u64 {
        self.snapshot();
        let id = self.scene.add(p, None);
        if let Some(o) = self.scene.get_mut(id) {
            o.transform.location = at;
        }
        self.select_only(id);
        id
    }

    pub fn delete_selected(&mut self) -> usize {
        if self.selection.is_empty() {
            return 0;
        }
        self.snapshot();
        let n = self.selection.len();
        let sel = std::mem::take(&mut self.selection);
        self.scene.remove(&sel);
        self.active = None;
        self.validate_edit();
        n
    }

    pub fn duplicate_selected(&mut self, offset: Vec3) -> Vec<u64> {
        if self.selection.is_empty() {
            return vec![];
        }
        self.snapshot();
        let mut new_ids = vec![];
        for id in self.selection.clone() {
            if let Some(src) = self.scene.get(id).cloned() {
                let mut o = src;
                o.id = self.scene.fresh_id();
                o.name = self.scene.unique_name(base_name(&o.name));
                o.transform.location += offset;
                new_ids.push(o.id);
                self.scene.objects.push(o);
            }
        }
        self.selection = new_ids.clone();
        self.active = new_ids.last().copied();
        new_ids
    }

    pub fn set_frame(&mut self, f: i32) {
        self.frame = f;
        self.scene.evaluate(f);
        self.propagate_parents();
        self.scene.update_skins();
    }

    pub fn insert_keyframes(&mut self) -> usize {
        let f = self.frame;
        let mut n = 0;
        for id in self.selection.clone() {
            if let Some(o) = self.scene.get_mut(id) {
                o.insert_keyframe(f);
                n += 1;
            }
        }
        if n > 0 {
            self.dirty = true;
        }
        n
    }

    /// Resolve a textual target ("selected", "all", "Cube", "spheres", "last") into object ids.
    pub fn resolve_target(&self, target: Option<&str>) -> Vec<u64> {
        let t = target.unwrap_or("selected").trim();
        let tl = t.to_lowercase();
        match tl.as_str() {
            "" | "selected" | "selection" | "it" | "this" | "that" | "them" | "current"
            | "active" => {
                if self.selection.is_empty() {
                    self.scene
                        .objects
                        .last()
                        .map(|o| vec![o.id])
                        .unwrap_or_default()
                } else {
                    self.selection.clone()
                }
            }
            "all" | "everything" | "scene" | "all objects" => {
                self.scene.objects.iter().map(|o| o.id).collect()
            }
            "last" | "newest" | "latest" => self
                .scene
                .objects
                .last()
                .map(|o| vec![o.id])
                .unwrap_or_default(),
            _ => {
                if let Some(id) = self
                    .scene
                    .objects
                    .iter()
                    .find(|o| o.name.to_lowercase() == tl)
                    .map(|o| o.id)
                {
                    return vec![id];
                }
                // "all spheres" / "cubes" -> by primitive
                let stripped = tl.trim_start_matches("all ").trim_start_matches("the ");
                let singular = stripped
                    .strip_suffix("es")
                    .filter(|s| Primitive::parse(s).is_some())
                    .or_else(|| stripped.strip_suffix('s'))
                    .unwrap_or(stripped);
                if matches!(singular, "light" | "lamp") {
                    return self
                        .scene
                        .objects
                        .iter()
                        .filter(|o| o.light().is_some())
                        .map(|o| o.id)
                        .collect();
                }
                if singular == "camera" {
                    return self
                        .scene
                        .objects
                        .iter()
                        .filter(|o| matches!(o.kind, crate::scene::ObjectKind::Camera(_)))
                        .map(|o| o.id)
                        .collect();
                }
                if let Some(p) = Primitive::parse(singular) {
                    let ids: Vec<u64> = self
                        .scene
                        .objects
                        .iter()
                        .filter(|o| o.primitive == p && o.is_mesh())
                        .map(|o| o.id)
                        .collect();
                    if !ids.is_empty() {
                        return ids;
                    }
                }
                // color match: "the red one"
                if let Some(c) = parse_color_str(
                    stripped
                        .trim_end_matches(" one")
                        .trim_end_matches(" ones")
                        .split_whitespace()
                        .next()
                        .unwrap_or(""),
                ) {
                    let ids: Vec<u64> = self
                        .scene
                        .objects
                        .iter()
                        .filter(|o| color_dist(o.material.color, c) < 0.08)
                        .map(|o| o.id)
                        .collect();
                    if !ids.is_empty() {
                        return ids;
                    }
                }
                self.scene.find_by_name(t).into_iter().collect()
            }
        }
    }

    /// Execute a batch of structured commands (from AI, voice or scripts).
    /// One undo step is recorded for the whole batch. Returns a human-readable log.
    pub fn execute(&mut self, cmds: &[Command]) -> Vec<String> {
        let mutating = cmds.iter().any(Self::is_mutating);
        if mutating {
            self.snapshot();
        }
        self.execute_no_snapshot(cmds)
    }

    pub fn is_mutating(c: &Command) -> bool {
        !matches!(
            c.action.as_str(),
            "undo"
                | "redo"
                | "view"
                | "play"
                | "stop"
                | "select"
                | "deselect_all"
                | "focus"
                | "shading"
                | "set_frame"
                | "chat"
                | "none"
        )
    }

    /// Execute without recording an undo step (the caller manages snapshots, e.g. streaming AI).
    pub fn execute_no_snapshot(&mut self, cmds: &[Command]) -> Vec<String> {
        self.in_batch = true;
        let log = self.execute_batch(cmds);
        self.in_batch = false;
        self.validate_edit();
        self.scene.update_skins();
        log
    }

    fn execute_batch(&mut self, cmds: &[Command]) -> Vec<String> {
        let mut log = vec![];
        for c in cmds {
            let r = self.exec_one(c);
            self.propagate_parents();
            match r {
                Ok(msg) => log.push(format!("✔ {msg}")),
                Err(e) => log.push(format!("✖ {}: {e}", c.action)),
            }
        }
        log
    }

    fn exec_one(&mut self, c: &Command) -> Result<String, String> {
        let action = c.action.to_lowercase().replace([' ', '-'], "_");
        if let Some(r) = self.exec_edit(&action, c) {
            return r;
        }
        let target_ids = self.resolve_target(c.target.as_deref());
        let targets = || target_ids.clone();
        match action.as_str() {
            "add" | "add_object" | "create" => {
                let pname = c.primitive.as_deref().unwrap_or("");
                let (p, special) = match Primitive::parse(pname) {
                    Some(p) => (p, None),
                    None => (
                        Primitive::Cube,
                        Some(
                            parse_special(pname)
                                .ok_or_else(|| format!("unknown primitive {:?}", c.primitive))?,
                        ),
                    ),
                };
                let id = match special {
                    None => self.scene.add(p, c.name.as_deref()),
                    Some(Special::Light(t)) => self.scene.add_light(t, c.name.as_deref()),
                    Some(Special::Camera) => self.scene.add_camera(c.name.as_deref()),
                    Some(Special::Armature) => {
                        let n = c.number.map(|n| n as usize).unwrap_or(1);
                        self.scene.add_armature(crate::armature::Armature::chain(n, 2.0), c.name.as_deref())
                    }
                };
                let is_arm = matches!(special, Some(Special::Armature));
                let o = self.scene.get_mut(id).unwrap();
                if let Some(v) = vec3(&c.location) {
                    o.transform.location = v;
                }
                if let Some(v) = vec3(&c.rotation) {
                    o.transform.rotation = v;
                }
                if let Some(v) = vec3_or_scalar(&c.scale) {
                    o.transform.scale = v;
                }
                if let Some(col) = c.color.as_ref().and_then(parse_color) {
                    o.material.color = col;
                }
                if let Some(d) = c.number.filter(|_| !is_arm) {
                    if (1.0..=6.0).contains(&d) && c.modifier.is_none() {
                        o.detail = d as u32;
                    }
                }
                apply_material_fields(o, c);
                apply_light_fields(o, c);
                let name = o.name.clone();
                let kind_label = o.type_label();
                if let Some(t) = c.look_at.as_ref().and_then(|v| self.resolve_point(v)) {
                    self.scene.get_mut(id).unwrap().look_at(t);
                }
                if let Some(m) = c.modifier.as_deref() {
                    let m = make_modifier(m, c.number, vec3(&c.vector))?;
                    self.scene.get_mut(id).unwrap().modifiers.push(m);
                }
                let mut extra = String::new();
                if let Some(pn) = c.parent.as_deref().filter(|p| !p.is_empty()) {
                    match self.scene.find_by_name(pn) {
                        Some(p) if p != id => {
                            self.scene.get_mut(id).unwrap().parent = Some(p);
                            extra += &format!(" (child of \"{}\")", self.scene.get(p).unwrap().name);
                        }
                        _ => extra += &format!(" (no parent \"{pn}\" found)"),
                    }
                }
                if let Some(cn) = c.collection.as_deref() {
                    let cid = self.scene.collection_by_name(cn);
                    self.scene.get_mut(id).unwrap().collection = cid;
                }
                self.select_only(id);
                Ok(format!("Added {kind_label} \"{name}\"{extra}"))
            }
            "delete" | "remove" => {
                let ids = targets();
                if ids.is_empty() {
                    return Err("nothing to delete".into());
                }
                self.scene.remove(&ids);
                self.selection.retain(|s| !ids.contains(s));
                self.fix_active();
                Ok(format!("Deleted {} object(s)", ids.len()))
            }
            "clear_scene" | "clear" | "new_scene" => {
                self.scene.objects.clear();
                self.selection.clear();
                self.active = None;
                Ok("Cleared the scene".into())
            }
            "select" => {
                let ids = targets();
                if ids.is_empty() {
                    return Err(format!("no object matches {:?}", c.target));
                }
                self.selection = ids.clone();
                self.active = ids.last().copied();
                Ok(format!("Selected {}", self.names(&ids)))
            }
            "deselect_all" | "deselect" => {
                self.selection.clear();
                self.active = None;
                Ok("Deselected all".into())
            }
            "move" | "translate" => {
                let d = vec3(&c.vector)
                    .or_else(|| vec3(&c.location))
                    .ok_or("move needs a vector")?;
                let ids = targets();
                for id in &ids {
                    if let Some(o) = self.scene.get_mut(*id) {
                        o.transform.location += d;
                    }
                }
                Ok(format!(
                    "Moved {} by [{:.2}, {:.2}, {:.2}]",
                    self.names(&ids),
                    d.x,
                    d.y,
                    d.z
                ))
            }
            "rotate" => {
                let d = vec3(&c.vector)
                    .or_else(|| vec3(&c.rotation))
                    .or_else(|| c.number.map(|n| Vec3::new(0.0, 0.0, n as f32)))
                    .ok_or("rotate needs degrees")?;
                let ids = targets();
                for id in &ids {
                    if let Some(o) = self.scene.get_mut(*id) {
                        o.transform.rotation += d;
                    }
                }
                Ok(format!(
                    "Rotated {} by [{:.0}°, {:.0}°, {:.0}°]",
                    self.names(&ids),
                    d.x,
                    d.y,
                    d.z
                ))
            }
            "scale" | "resize" => {
                let f = vec3_or_scalar(&c.vector)
                    .or_else(|| vec3_or_scalar(&c.scale))
                    .or_else(|| c.number.map(|n| Vec3::splat(n as f32)))
                    .ok_or("scale needs a factor")?;
                let ids = targets();
                for id in &ids {
                    if let Some(o) = self.scene.get_mut(*id) {
                        o.transform.scale *= f;
                    }
                }
                Ok(format!(
                    "Scaled {} by [{:.2}, {:.2}, {:.2}]",
                    self.names(&ids),
                    f.x,
                    f.y,
                    f.z
                ))
            }
            "set_transform" | "place" | "set_location" | "set" => {
                let ids = targets();
                if ids.is_empty() {
                    return Err("no target".into());
                }
                for id in &ids {
                    if let Some(o) = self.scene.get_mut(*id) {
                        if let Some(v) = vec3(&c.location) {
                            o.transform.location = v;
                        }
                        if let Some(v) = vec3(&c.rotation) {
                            o.transform.rotation = v;
                        }
                        if let Some(v) = vec3_or_scalar(&c.scale) {
                            o.transform.scale = v;
                        }
                        if let Some(col) = c.color.as_ref().and_then(parse_color) {
                            o.material.color = col;
                        }
                        apply_material_fields(o, c);
                    }
                }
                Ok(format!("Updated {}", self.names(&ids)))
            }
            "set_color" | "color" | "set_material" | "material" | "paint" => {
                let ids = targets();
                let col = c.color.as_ref().and_then(parse_color);
                for id in &ids {
                    if let Some(o) = self.scene.get_mut(*id) {
                        if let Some(col) = col {
                            o.material.color = col;
                        }
                        apply_material_fields(o, c);
                        apply_light_fields(o, c);
                    }
                }
                Ok(format!("Changed material of {}", self.names(&ids)))
            }
            "set_light" | "light" => {
                let ids: Vec<u64> = targets()
                    .into_iter()
                    .filter(|id| self.scene.get(*id).is_some_and(|o| o.light().is_some()))
                    .collect();
                if ids.is_empty() {
                    return Err("no light selected/targeted".into());
                }
                for id in &ids {
                    if let Some(o) = self.scene.get_mut(*id) {
                        apply_light_fields(o, c);
                    }
                }
                Ok(format!("Updated light {}", self.names(&ids)))
            }
            "look_at" | "aim" | "point_at" => {
                let p = c
                    .look_at
                    .as_ref()
                    .or(c.location.as_ref())
                    .and_then(|v| self.resolve_point(v))
                    .ok_or("look_at needs a point [x,y,z] or object name")?;
                let ids = targets();
                for id in &ids {
                    if let Some(o) = self.scene.get_mut(*id) {
                        o.look_at(p);
                    }
                }
                Ok(format!(
                    "Aimed {} at [{:.1}, {:.1}, {:.1}]",
                    self.names(&ids),
                    p.x,
                    p.y,
                    p.z
                ))
            }
            "import" | "import_gltf" => {
                let path = c.name.clone().ok_or("import needs name = the .glb/.gltf file path")?;
                let (st, ids) = crate::gltf_import::import_file(&mut self.scene, std::path::Path::new(path.trim()))?;
                self.exit_edit();
                self.selection = ids.clone();
                self.active = ids.first().copied();
                if !self.scene.textures.is_empty() && st.textures > 0 && self.shading == Shading::Solid {
                    self.shading = Shading::Material;
                }
                self.view_requests.push(ViewRequest::FrameSelected);
                Ok(format!("Imported {}", st.summary()))
            }
            "unwrap" | "uv_unwrap" | "uv_project" => {
                let method = crate::uv::Unwrap::parse(c.name.as_deref().unwrap_or("smart"));
                let (ids, faces): (Vec<u64>, Option<Vec<usize>>) = match &self.edit {
                    Some(st) => {
                        let o = self.scene.get(st.obj).ok_or("edited object is gone")?;
                        let pm = o.mesh.clone().unwrap_or_else(|| o.poly());
                        let f = em::selected_faces(&pm, &st.sel);
                        (vec![st.obj], Some(f))
                    }
                    None => (targets().into_iter().filter(|id| self.scene.get(*id).is_some_and(|o| o.is_mesh())).collect(), None),
                };
                if ids.is_empty() {
                    return Err("no mesh object to unwrap".into());
                }
                for id in &ids {
                    let o = self.scene.get_mut(*id).unwrap();
                    let mut pm = o.mesh.clone().unwrap_or_else(|| o.poly());
                    crate::uv::unwrap(&mut pm, faces.as_deref().unwrap_or(&[]), method);
                    o.mesh = Some(pm);
                }
                Ok(format!("{} on {}", method.label(), self.names(&ids)))
            }
            "boolean" | "bool" | "cut" => {
                let op = c.name.as_deref().or(c.modifier.as_deref()).and_then(crate::csg::BoolOp::parse).unwrap_or(crate::csg::BoolOp::Difference);
                // main object: target, else the active; cutters: `object`, else the other selected
                let main = match c.target.as_deref() {
                    Some(t) => self.scene.find_by_name(t).ok_or(format!("no object named {t:?}"))?,
                    None => self.active.ok_or("select the object to cut (active) and the cutter")?,
                };
                let cutters: Vec<u64> = match c.object.as_deref() {
                    Some(n) => vec![self.scene.find_by_name(n).ok_or(format!("no object named {n:?}"))?],
                    None => self.selection.iter().copied().filter(|i| *i != main).collect(),
                };
                if cutters.is_empty() {
                    return Err("boolean needs a second object (object = the cutter)".into());
                }
                self.boolean(main, &cutters, op)
            }
            "parent" | "set_parent" => {
                let pname = c.parent.clone().or(c.name.clone()).ok_or("parent needs parent = the parent object's name")?;
                let p = self.scene.find_by_name(&pname).ok_or(format!("no object named {pname:?}"))?;
                let ids: Vec<u64> = if c.target.is_some() { targets() } else { self.selection.iter().copied().filter(|i| *i != p).collect() };
                let mut done = vec![];
                for id in ids {
                    if id == p || self.scene.would_cycle(id, p) {
                        continue;
                    }
                    self.scene.get_mut(id).unwrap().parent = Some(p);
                    done.push(id);
                }
                if done.is_empty() {
                    return Err("nothing to parent (an object can't be its own ancestor)".into());
                }
                // parenting meshes to an armature binds them with automatic weights (Blender's
                // Ctrl+P ▸ With Automatic Weights)
                let mut bound = vec![];
                if self.scene.get(p).is_some_and(|o| o.armature.is_some()) {
                    for id in &done {
                        if self.bind_to_armature(*id, p).is_ok() {
                            bound.push(*id);
                        }
                    }
                }
                self.reset_parent_cache();
                Ok(format!(
                    "Parented {} to \"{}\"{}",
                    self.names(&done),
                    self.scene.get(p).unwrap().name,
                    if bound.is_empty() { String::new() } else { " with automatic weights".into() }
                ))
            }
            "clear_parent" | "unparent" => {
                let ids = targets();
                for id in &ids {
                    if let Some(o) = self.scene.get_mut(*id) {
                        o.parent = None;
                    }
                }
                Ok(format!("Cleared parent of {}", self.names(&ids)))
            }
            "move_to_collection" | "collection" => {
                let ids = targets();
                if ids.is_empty() {
                    return Err("nothing selected".into());
                }
                // children go along with their parents
                let mut ids = ids;
                let mut k = 0;
                while k < ids.len() {
                    let p = ids[k];
                    for o in &self.scene.objects {
                        if o.parent == Some(p) && !ids.contains(&o.id) {
                            ids.push(o.id);
                        }
                    }
                    k += 1;
                }
                let cname = c.collection.clone().or(c.name.clone()).unwrap_or_default();
                let cid = self.scene.collection_by_name(&cname);
                let vis = self.scene.collections.iter().find(|x| x.id == cid).map(|x| x.visible).unwrap_or(true);
                for id in &ids {
                    if let Some(o) = self.scene.get_mut(*id) {
                        o.collection = cid;
                        if !vis {
                            o.visible = false;
                        }
                    }
                }
                let cn = self.scene.collections.iter().find(|x| x.id == cid).map(|x| x.name.clone()).unwrap_or("Scene Collection".into());
                Ok(format!("Moved {} to collection \"{cn}\"", self.names(&ids)))
            }
            "delete_collection" => {
                let cname = c.collection.clone().or(c.name.clone()).ok_or("which collection?")?;
                let cid = self.scene.collections.iter().find(|x| x.name.eq_ignore_ascii_case(cname.trim())).map(|x| x.id).ok_or(format!("no collection {cname:?}"))?;
                self.scene.collections.retain(|x| x.id != cid);
                for o in self.scene.objects.iter_mut().filter(|o| o.collection == cid) {
                    o.collection = 0;
                }
                Ok(format!("Deleted collection \"{cname}\" (objects moved to the Scene Collection)"))
            }
            "hide_collection" | "show_collection" => {
                let cname = c.collection.clone().or(c.name.clone()).ok_or("which collection?")?;
                let show = action == "show_collection";
                let col = self.scene.collections.iter_mut().find(|x| x.name.eq_ignore_ascii_case(cname.trim())).ok_or(format!("no collection {cname:?}"))?;
                col.visible = show;
                let cid = col.id;
                for o in self.scene.objects.iter_mut().filter(|o| o.collection == cid) {
                    o.visible = show;
                }
                Ok(format!("{} collection \"{cname}\"", if show { "Showed" } else { "Hid" }))
            }
            "rig" | "auto_rig" | "add_rig" | "add_armature" => {
                let ids: Vec<u64> = targets().into_iter().filter(|id| self.scene.get(*id).is_some_and(|o| o.is_mesh())).collect();
                let &mesh_id = ids.first().ok_or("select the mesh to rig")?;
                let n = c.number.map(|n| n as usize).unwrap_or(4).clamp(1, 32);
                let arm_id = self.rig(mesh_id, n)?;
                let an = self.scene.get(arm_id).unwrap().name.clone();
                self.select_only(arm_id);
                Ok(format!("Rigged \"{}\" with {an} ({n} bones, automatic weights) — Ctrl+Tab for Pose Mode", self.scene.get(mesh_id).unwrap().name))
            }
            "pose_mode" | "pose" => {
                if let Some(t) = c.target.as_deref().and_then(|t| self.scene.find_by_name(t)) {
                    self.active = Some(t);
                    self.pose = None;
                }
                if self.pose.is_some() {
                    return Ok("Already in pose mode".into());
                }
                self.toggle_pose()
            }
            "pose_bone" | "rotate_bone" | "bend" | "bend_bone" => {
                let aid = self
                    .pose
                    .or_else(|| c.target.as_deref().and_then(|t| self.scene.find_by_name(t)).filter(|id| self.scene.get(*id).is_some_and(|o| o.armature.is_some())))
                    .or_else(|| self.active.filter(|id| self.scene.get(*id).is_some_and(|o| o.armature.is_some())))
                    .or_else(|| {
                        // a rigged mesh's armature
                        let t = c.target.as_deref().and_then(|t| self.scene.find_by_name(t)).or(self.active)?;
                        self.scene.get(t)?.skin.as_ref().map(|s| s.armature)
                    })
                    .or_else(|| self.scene.objects.iter().find(|o| o.armature.is_some()).map(|o| o.id))
                    .ok_or("no armature in the scene")?;
                let o = self.scene.get(aid).unwrap();
                let arm = o.armature.as_ref().unwrap();
                let bi = match c.name.as_deref() {
                    Some(n) => arm.find(n).ok_or(format!("no bone {n:?} (bones: {})", arm.bones.iter().map(|b| b.name.as_str()).collect::<Vec<_>>().join(", ")))?,
                    None => self.active_bone.min(arm.bones.len().saturating_sub(1)),
                };
                let rot = vec3(&c.rotation).or_else(|| vec3(&c.vector)).or_else(|| c.number.map(|n| Vec3::new(n as f32, 0.0, 0.0)));
                let loc = vec3(&c.location);
                if rot.is_none() && loc.is_none() {
                    return Err("pose_bone needs rotation = [x,y,z] degrees (or location)".into());
                }
                let a = self.scene.get_mut(aid).unwrap().armature.as_mut().unwrap();
                let b = &mut a.bones[bi];
                if let Some(r) = rot {
                    b.pose.set_euler_deg(r);
                }
                if let Some(l) = loc {
                    b.pose.loc = l;
                }
                let bn = b.name.clone();
                self.active_bone = bi;
                Ok(format!("Posed bone \"{bn}\""))
            }
            "reset_pose" | "rest_pose" | "clear_pose" => {
                let mut n = 0;
                for o in self.scene.objects.iter_mut() {
                    if let Some(a) = o.armature.as_mut() {
                        a.reset_pose();
                        n += 1;
                    }
                }
                if n == 0 {
                    return Err("no armature in the scene".into());
                }
                Ok("Reset the pose".into())
            }
            "bump" | "normal_map" | "set_normal_map" | "bumpy" => {
                let ids: Vec<u64> = targets().into_iter().filter(|id| self.scene.get(*id).is_some_and(|o| o.is_mesh())).collect();
                if ids.is_empty() {
                    return Err("no mesh object for a normal map".into());
                }
                let spec = c.name.clone().or(c.primitive.clone()).unwrap_or_default();
                let low = spec.trim().to_lowercase();
                let strength = c.number.map(|n| (n as f32).clamp(0.0, 10.0));
                if matches!(low.as_str(), "none" | "remove" | "off" | "clear" | "flat") {
                    for id in &ids {
                        if let Some(o) = self.scene.get_mut(*id) {
                            o.material.normal_tex = 0;
                        }
                    }
                    return Ok(format!("Removed the normal map from {}", self.names(&ids)));
                }
                let mut made = vec![];
                for id in &ids {
                    let base = self.scene.get(*id).map(|o| o.material.texture).unwrap_or(0);
                    // the source: given texture / image / pattern, else the object's colour texture
                    let auto = low.replace("invert", "").replace("ed", "");
                    let src_id = if matches!(auto.trim(), "" | "auto" | "texture") {
                        if base == 0 {
                            return Err("give a pattern or image for the bumps (the object has no texture)".into());
                        }
                        base
                    } else {
                        self.resolve_texture(&spec)?.0
                    };
                    let t = self.scene.texture(src_id).cloned().ok_or("texture not found")?;
                    let nid = if crate::textures::decode(&t.source).map(|i| crate::textures::looks_like_normal_map(&i)).unwrap_or(false) {
                        t.id
                    } else {
                        let invert = low.contains("invert");
                        let png = crate::textures::normal_map_png(&t.source, 8.0, invert)?;
                        self.scene.add_texture(&format!("{} normal", t.name), crate::textures::TexSource::Image { data: png })
                    };
                    if let Some(o) = self.scene.get_mut(*id) {
                        o.material.normal_tex = nid;
                        if let Some(s) = strength {
                            o.material.normal_strength = s;
                        }
                    }
                    made.push(self.scene.texture(nid).map(|t| t.name.clone()).unwrap_or_default());
                }
                if self.shading == Shading::Solid {
                    self.shading = Shading::Material;
                }
                made.dedup();
                Ok(format!("Normal map {} on {}", made.join(", "), self.names(&ids)))
            }
            "roughness_map" | "set_roughness_map" => {
                let ids: Vec<u64> = targets().into_iter().filter(|id| self.scene.get(*id).is_some_and(|o| o.is_mesh())).collect();
                if ids.is_empty() {
                    return Err("no mesh object for a roughness map".into());
                }
                let spec = c.name.clone().or(c.primitive.clone()).unwrap_or_default();
                let low = spec.trim().to_lowercase();
                let rid = if matches!(low.as_str(), "none" | "remove" | "off" | "clear") {
                    0
                } else {
                    let src = if low.is_empty() || low == "auto" || low == "texture" {
                        let b = self.scene.get(ids[0]).map(|o| o.material.texture).unwrap_or(0);
                        if b == 0 {
                            return Err("give a pattern or image for the roughness map".into());
                        }
                        b
                    } else {
                        self.resolve_texture(&spec)?.0
                    };
                    let t = self.scene.texture(src).cloned().ok_or("texture not found")?;
                    match &t.source {
                        // a pattern: rough in the grooves (mortar, grout), shinier on top
                        crate::textures::TexSource::Procedural { .. } => {
                            let png = crate::textures::roughness_map_png(&t.source)?;
                            self.scene.add_texture(&format!("{} roughness", t.name), crate::textures::TexSource::Image { data: png })
                        }
                        _ => t.id,
                    }
                };
                for id in &ids {
                    if let Some(o) = self.scene.get_mut(*id) {
                        o.material.rough_tex = rid;
                        if rid != 0 {
                            if let Some(r) = c.number {
                                o.material.roughness = (r as f32).clamp(0.0, 1.0);
                            } else if o.material.roughness < 0.6 {
                                o.material.roughness = 0.8;
                            }
                        }
                    }
                }
                Ok(if rid == 0 { format!("Removed the roughness map from {}", self.names(&ids)) } else { format!("Roughness map on {}", self.names(&ids)) })
            }
            "set_texture" | "texture" => {
                let ids: Vec<u64> = targets().into_iter().filter(|id| self.scene.get(*id).is_some_and(|o| o.is_mesh())).collect();
                if ids.is_empty() {
                    return Err("no mesh object to texture".into());
                }
                let spec = c.name.clone().or(c.primitive.clone()).unwrap_or_default();
                let (tex_id, label) = self.resolve_texture(&spec)?;
                let tint = c.color.as_ref().and_then(parse_color);
                for id in &ids {
                    if let Some(o) = self.scene.get_mut(*id) {
                        o.material.texture = tex_id;
                        if tex_id != 0 {
                            o.material.color = tint.unwrap_or([1.0, 1.0, 1.0]);
                            if let Some(n) = c.number {
                                o.material.tex_scale = (n as f32).clamp(0.05, 100.0);
                            }
                        }
                    }
                }
                if tex_id != 0 && self.shading == Shading::Solid {
                    self.shading = Shading::Material;
                }
                Ok(format!("Applied {label} to {}", self.names(&ids)))
            }
            "render" | "render_image" => {
                match c.name.as_deref().map(|n| n.to_lowercase()) {
                    Some(n) if n.contains("path") || n.contains("cycles") || n.contains("quality") => self.scene.output.engine = crate::scene::Engine::PathTraced,
                    Some(n) if n.contains("fast") || n.contains("preview") => self.scene.output.engine = crate::scene::Engine::Fast,
                    _ => {}
                }
                if let Some(s) = c.number.filter(|s| *s >= 1.0) {
                    self.scene.output.samples = s as u32;
                }
                self.render_request = true;
                Ok("Rendering an image (F12)".into())
            }
            "camera_view" | "look_through_camera" => {
                self.view_requests.push(ViewRequest::Camera);
                Ok("Toggled camera view".into())
            }
            "rename" => {
                let ids = targets();
                let name = c.name.clone().ok_or("rename needs a name")?;
                let id = *ids.first().ok_or("no target")?;
                let unique = self.scene.unique_name(&name);
                self.scene.get_mut(id).unwrap().name = unique.clone();
                Ok(format!("Renamed to \"{unique}\""))
            }
            "duplicate" | "copy" => {
                let ids = targets();
                self.selection = ids;
                let off = vec3(&c.vector).unwrap_or(Vec3::new(2.5, 0.0, 0.0));
                let count = c.number.unwrap_or(1.0).clamp(1.0, 100.0) as usize;
                let mut all = vec![];
                for _ in 0..count {
                    // duplicate_selected snapshots; avoid nested snapshot spam by temporarily stashing stacks
                    let undo = std::mem::take(&mut self.undo_stack);
                    let new = self.duplicate_selected(off);
                    self.undo_stack = undo;
                    all.extend(new);
                }
                Ok(format!("Duplicated into {}", self.names(&all)))
            }
            "hide" => {
                let ids = targets();
                for id in &ids {
                    if let Some(o) = self.scene.get_mut(*id) {
                        o.visible = false;
                    }
                }
                Ok(format!("Hid {}", self.names(&ids)))
            }
            "show" | "unhide" => {
                let ids = if c.target.is_none() {
                    self.scene.objects.iter().map(|o| o.id).collect()
                } else {
                    targets()
                };
                for id in &ids {
                    if let Some(o) = self.scene.get_mut(*id) {
                        o.visible = true;
                    }
                }
                Ok(format!("Revealed {}", self.names(&ids)))
            }
            "add_modifier" | "modifier" => {
                let ids = targets();
                let m = make_modifier(
                    c.modifier.as_deref().unwrap_or("array"),
                    c.number,
                    vec3(&c.vector),
                )?;
                for id in &ids {
                    if let Some(o) = self.scene.get_mut(*id) {
                        if let Modifier::Subsurf { levels } = m {
                            // one Subdivision Surface per object, shaded smooth like Blender's Ctrl+N
                            o.modifiers.retain(|x| !matches!(x, Modifier::Subsurf { .. }));
                            if levels > 0 {
                                o.modifiers.push(m.clone());
                            }
                            o.smooth_override = Some(true);
                            continue;
                        }
                        o.modifiers.push(m.clone());
                    }
                }
                Ok(format!(
                    "Added {} modifier to {}",
                    m.label(),
                    self.names(&ids)
                ))
            }
            "clear_modifiers" | "remove_modifiers" => {
                let ids = targets();
                for id in &ids {
                    if let Some(o) = self.scene.get_mut(*id) {
                        o.modifiers.clear();
                    }
                }
                Ok(format!("Removed modifiers from {}", self.names(&ids)))
            }
            "keyframe" | "insert_keyframe" | "key" => {
                if let Some(f) = c.number {
                    self.frame = f as i32;
                }
                let ids = targets();
                let f = self.frame;
                for id in &ids {
                    if let Some(o) = self.scene.get_mut(*id) {
                        if let Some(v) = vec3(&c.location) {
                            o.transform.location = v;
                        }
                        if let Some(v) = vec3(&c.rotation) {
                            o.transform.rotation = v;
                        }
                        if let Some(v) = vec3_or_scalar(&c.scale) {
                            o.transform.scale = v;
                        }
                        o.insert_keyframe(f);
                    }
                }
                Ok(format!(
                    "Inserted keyframe at frame {f} for {}",
                    self.names(&ids)
                ))
            }
            "clear_keyframes" | "clear_animation" => {
                let ids = targets();
                for id in &ids {
                    if let Some(o) = self.scene.get_mut(*id) {
                        o.keyframes.clear();
                    }
                }
                Ok(format!("Cleared animation on {}", self.names(&ids)))
            }
            "set_frame" | "frame" | "goto_frame" => {
                let f = c.number.ok_or("set_frame needs a number")? as i32;
                self.set_frame(f);
                Ok(format!("Frame {f}"))
            }
            "set_frame_range" | "frame_range" => {
                if let Some(v) = c.vector.as_ref().and_then(|v| v.as_array()) {
                    if let (Some(a), Some(b)) = (
                        v.first().and_then(|x| x.as_f64()),
                        v.get(1).and_then(|x| x.as_f64()),
                    ) {
                        self.scene.frame_start = a as i32;
                        self.scene.frame_end = b as i32;
                    }
                } else if let Some(n) = c.number {
                    self.scene.frame_end = n as i32;
                }
                Ok(format!(
                    "Frame range {}–{}",
                    self.scene.frame_start, self.scene.frame_end
                ))
            }
            "play" => {
                self.playing = true;
                Ok("Playing animation".into())
            }
            "stop" | "pause" => {
                self.playing = false;
                Ok("Stopped playback".into())
            }
            "view" | "set_view" | "camera" => {
                let v = c
                    .view
                    .as_deref()
                    .or(c.target.as_deref())
                    .unwrap_or("iso")
                    .to_lowercase();
                let req = match v.as_str() {
                    "front" => ViewRequest::Preset(ViewPreset::Front),
                    "back" => ViewRequest::Preset(ViewPreset::Back),
                    "left" => ViewRequest::Preset(ViewPreset::Left),
                    "right" | "side" => ViewRequest::Preset(ViewPreset::Right),
                    "top" => ViewRequest::Preset(ViewPreset::Top),
                    "bottom" => ViewRequest::Preset(ViewPreset::Bottom),
                    "ortho" | "orthographic" | "perspective" | "persp" | "toggle_ortho" => {
                        ViewRequest::ToggleOrtho
                    }
                    "selected" | "focus" | "frame_selected" => ViewRequest::FrameSelected,
                    "all" | "frame_all" | "home" => ViewRequest::FrameAll,
                    "camera" | "camera_view" => ViewRequest::Camera,
                    "align_camera" => ViewRequest::AlignCameraToView,
                    "zoom_in" => ViewRequest::Zoom(0.7),
                    "zoom_out" => ViewRequest::Zoom(1.4),
                    "orbit_left" => ViewRequest::Orbit(-30.0, 0.0),
                    "orbit_right" => ViewRequest::Orbit(30.0, 0.0),
                    _ => ViewRequest::Preset(ViewPreset::Iso),
                };
                self.view_requests.push(req);
                Ok(format!("View: {v}"))
            }
            "focus" | "frame_selected" => {
                let ids = targets();
                if !ids.is_empty() {
                    self.selection = ids;
                    self.active = self.selection.last().copied();
                }
                self.view_requests.push(ViewRequest::FrameSelected);
                Ok("Framed selection".into())
            }
            "shading" | "set_shading" => {
                let s = c
                    .view
                    .as_deref()
                    .or(c.name.as_deref())
                    .or(c.target.as_deref())
                    .unwrap_or("solid")
                    .to_lowercase();
                self.shading = if s.contains("wire") {
                    Shading::Wireframe
                } else if s.contains("mat") || s.contains("render") || s.contains("preview") {
                    Shading::Material
                } else {
                    Shading::Solid
                };
                Ok(format!("Shading: {:?}", self.shading))
            }
            "subsurf" | "subdivision_surface" | "subdivision" => {
                let ids: Vec<u64> = targets().into_iter().filter(|id| self.scene.get(*id).is_some_and(|o| o.is_mesh())).collect();
                if ids.is_empty() {
                    return Err("no mesh object selected".into());
                }
                let levels = c.number.unwrap_or(2.0).clamp(0.0, 4.0) as u32;
                for id in &ids {
                    if let Some(o) = self.scene.get_mut(*id) {
                        o.modifiers.retain(|x| !matches!(x, Modifier::Subsurf { .. }));
                        if levels > 0 {
                            o.modifiers.push(Modifier::Subsurf { levels });
                            o.smooth_override = Some(true);
                        }
                    }
                }
                Ok(if levels == 0 { format!("Removed subdivision from {}", self.names(&ids)) } else { format!("Subdivision level {levels} on {}", self.names(&ids)) })
            }
            "auto_smooth" | "shade_auto_smooth" => {
                let ids = targets();
                let ang = c.number.unwrap_or(30.0).clamp(1.0, 179.0) as f32;
                for id in &ids {
                    if let Some(o) = self.scene.get_mut(*id) {
                        o.smooth_override = Some(true);
                        o.auto_smooth = ang;
                    }
                }
                Ok(format!("Shade auto smooth ({ang:.0}°) on {}", self.names(&ids)))
            }
            "smooth" | "shade_smooth" | "flat" | "shade_flat" => {
                let ids = targets();
                let smooth = action.contains("smooth");
                for id in &ids {
                    if let Some(o) = self.scene.get_mut(*id) {
                        o.smooth_override = Some(smooth);
                    }
                }
                Ok(format!(
                    "Shade {} on {}",
                    if smooth { "smooth" } else { "flat" },
                    self.names(&ids)
                ))
            }
            "set_detail" | "subdivide" => {
                let ids = targets();
                for id in &ids {
                    if let Some(o) = self.scene.get_mut(*id) {
                        o.detail = c
                            .number
                            .map(|n| n as u32)
                            .unwrap_or(o.detail + 1)
                            .clamp(1, 6);
                    }
                }
                Ok(format!("Changed resolution of {}", self.names(&ids)))
            }
            "set_background" | "background" | "world" => {
                if let Some(col) = c.color.as_ref().and_then(parse_color) {
                    self.scene.world.background = col;
                }
                if let Some(v) = vec3(&c.vector) {
                    if v.length() > 0.0 {
                        self.scene.world.sun_dir = v.normalize();
                    }
                }
                if let Some(n) = c.number {
                    self.scene.world.sun_strength = n as f32;
                }
                Ok("Updated world settings".into())
            }
            "undo" => {
                if self.undo() {
                    Ok("Undo".into())
                } else {
                    Err("nothing to undo".into())
                }
            }
            "redo" => {
                if self.redo() {
                    Ok("Redo".into())
                } else {
                    Err("nothing to redo".into())
                }
            }
            "chat" | "none" | "answer" | "reply" => Ok("(no scene change)".into()),
            other => Err(format!("unknown action '{other}'")),
        }
    }

    fn names(&self, ids: &[u64]) -> String {
        let n: Vec<String> = ids
            .iter()
            .filter_map(|id| self.scene.get(*id))
            .map(|o| format!("\"{}\"", o.name))
            .collect();
        match n.len() {
            0 => "nothing".into(),
            1..=4 => n.join(", "),
            k => format!("{} and {} more", n[..3].join(", "), k - 3),
        }
    }
}

/// Edit-mode operators (shared by keyboard shortcuts, menus, the command palette and the AI).
#[derive(Clone, Debug, PartialEq)]
pub enum EditOp {
    Extrude(f32),
    Inset(f32),
    /// width, segments
    Bevel(f32, usize),
    Subdivide,
    Delete,
    Merge,
    FlipNormals,
    SelectAll,
    SelectNone,
    SelectInvert,
    SelectNormal(Vec3, bool),
    LoopCut {
        edge: (u32, u32),
        cuts: usize,
    },
    LoopCutAxis {
        axis: Vec3,
        cuts: usize,
    },
    /// world-space translation of selected vertices
    Translate(Vec3),
    Scale(Vec3),
}

impl Editor {
    pub fn editing(&self) -> bool {
        self.edit.is_some()
    }

    pub fn enter_edit(&mut self, id: u64) -> Result<String, String> {
        let o = self.scene.get(id).ok_or("no such object")?;
        if !o.is_mesh() {
            return Err("only mesh objects can be edited".into());
        }
        let name = o.name.clone();
        if o.mesh.is_none() {
            // convert the procedural primitive into an editable mesh (silent: geometry is identical)
            let pm = o.poly();
            self.scene.get_mut(id).unwrap().mesh = Some(pm);
        }
        let n = self
            .scene
            .get(id)
            .and_then(|o| o.mesh.as_ref())
            .map(|m| m.verts.len())
            .unwrap_or(0);
        self.select_only(id);
        self.sculpt = None;
        self.pose = None;
        self.edit = Some(EditState {
            obj: id,
            mode: SelectMode::Vertex,
            sel: vec![true; n],
        });
        Ok(format!("Edit mode: \"{name}\""))
    }

    pub fn exit_edit(&mut self) {
        self.edit = None;
        self.sculpt = None;
        self.pose = None;
    }

    /// Enter / leave Pose Mode on the active armature.
    pub fn toggle_pose(&mut self) -> Result<String, String> {
        if self.pose.take().is_some() {
            return Ok("Object mode".into());
        }
        let id = self
            .active
            .filter(|id| self.scene.get(*id).is_some_and(|o| o.armature.is_some()))
            .or_else(|| self.selection.iter().copied().find(|id| self.scene.get(*id).is_some_and(|o| o.armature.is_some())))
            .or_else(|| self.scene.objects.iter().find(|o| o.armature.is_some()).map(|o| o.id))
            .ok_or("no armature in the scene (add one with Add ▸ Armature, or ask to rig a mesh)")?;
        self.edit = None;
        self.sculpt = None;
        self.select_only(id);
        self.pose = Some(id);
        self.active_bone = 0;
        Ok(format!("Pose mode: \"{}\"", self.scene.get(id).unwrap().name))
    }

    /// Enter sculpt mode on a mesh object (converting it to an editable mesh and adding detail
    /// if it is too coarse to sculpt).
    pub fn enter_sculpt(&mut self, id: u64) -> Result<String, String> {
        let o = self.scene.get(id).ok_or("no such object")?;
        if !o.is_mesh() {
            return Err("only mesh objects can be sculpted".into());
        }
        self.edit = None;
        let name = o.name.clone();
        let mut pm = o.poly();
        let before = pm.verts.len();
        let densified = crate::sculpt::densify(&mut pm, 4000, o.smooth());
        let o = self.scene.get_mut(id).unwrap();
        o.mesh = Some(pm);
        let after = o.mesh.as_ref().map(|m| m.verts.len()).unwrap_or(0);
        self.select_only(id);
        self.sculpt = Some(id);
        Ok(if densified { format!("Sculpt mode: \"{name}\" (subdivided {before} → {after} vertices for detail)") } else { format!("Sculpt mode: \"{name}\"") })
    }

    pub fn toggle_edit(&mut self) -> Result<String, String> {
        if self.pose.take().is_some() {
            return Ok("Object mode".into());
        }
        if self.sculpt.is_some() {
            self.sculpt = None;
            return Ok("Object mode".into());
        }
        if self.edit.is_some() {
            self.exit_edit();
            return Ok("Object mode".into());
        }
        let id = self
            .active
            .or_else(|| self.selection.last().copied())
            .ok_or("select a mesh object first")?;
        self.enter_edit(id)
    }

    /// Mesh being edited + its object's world matrix.
    pub fn edit_mesh(&self) -> Option<(&crate::mesh::PolyMesh, &EditState, glam::Mat4)> {
        let st = self.edit.as_ref()?;
        let o = self.scene.get(st.obj)?;
        Some((o.mesh.as_ref()?, st, o.transform.matrix()))
    }

    /// Run an edit-mode operator as one undo step.
    pub fn edit_op(&mut self, op: EditOp) -> Result<String, String> {
        let selection_only = self.in_batch
            || matches!(
                op,
                EditOp::SelectAll
                    | EditOp::SelectNone
                    | EditOp::SelectInvert
                    | EditOp::SelectNormal(..)
            );
        if !selection_only {
            self.snapshot();
        }
        let r = self.edit_op_inner(op);
        if r.is_err() && !selection_only {
            self.undo_stack.pop();
        }
        r
    }

    fn edit_op_inner(&mut self, op: EditOp) -> Result<String, String> {
        let st = self.edit.as_mut().ok_or("not in edit mode")?;
        let o = self
            .scene
            .objects
            .iter_mut()
            .find(|o| o.id == st.obj)
            .ok_or("edited object is gone")?;
        let m = o.transform.matrix();
        let m3 = glam::Mat3::from_mat4(m);
        let inv3 = m3.inverse();
        let pm = o.mesh.get_or_insert_with(|| o.primitive.poly(o.detail));
        st.fit(pm);
        let sel = &mut st.sel;
        Ok(match op {
            EditOp::Extrude(dist) => {
                let n = em::extrude(pm, sel)?;
                if dist != 0.0 {
                    let world_dir = (m3.inverse().transpose() * n).normalize_or_zero();
                    let local = inv3 * (world_dir * dist);
                    for (v, s) in pm.verts.iter_mut().zip(sel.iter()) {
                        if *s {
                            *v += local;
                        }
                    }
                }
                format!("Extruded {} vertices", sel.iter().filter(|s| **s).count())
            }
            EditOp::Inset(t) => format!("Inset {} face(s)", em::inset(pm, sel, t)?),
            EditOp::Bevel(w, segs) => {
                // the width is given in world units; edit in local space
                let s = (m3 * Vec3::ONE.normalize()).length().max(1e-6);
                format!("Beveled {} edge(s)", em::bevel(pm, sel, w / s, segs)?)
            }
            EditOp::Subdivide => format!("Subdivided {} face(s)", em::subdivide(pm, sel)?),
            EditOp::Delete => format!("Deleted {} face(s)", em::delete(pm, sel, st.mode)),
            EditOp::Merge => format!("Merged {} vertices", em::merge_center(pm, sel)?),
            EditOp::FlipNormals => format!("Flipped {} face(s)", em::flip_normals(pm, sel)),
            EditOp::SelectAll => {
                sel.iter_mut().for_each(|s| *s = true);
                "Selected all".into()
            }
            EditOp::SelectNone => {
                sel.iter_mut().for_each(|s| *s = false);
                "Deselected all".into()
            }
            EditOp::SelectInvert => {
                sel.iter_mut().for_each(|s| *s = !*s);
                "Inverted selection".into()
            }
            EditOp::SelectNormal(dir, extend) => {
                let local = (m3.transpose() * dir).normalize_or_zero();
                let n = em::select_by_normal(pm, local, extend, sel);
                st.mode = SelectMode::Face;
                format!("Selected {n} face(s)")
            }
            EditOp::LoopCut { edge, cuts } => format!(
                "Loop cut through {} faces",
                em::loop_cut(pm, sel, edge, cuts)?
            ),
            EditOp::LoopCutAxis { axis, cuts } => {
                let local = (inv3 * axis).normalize_or_zero();
                let e = em::edge_along(pm, local).ok_or("mesh has no edges")?;
                format!("Loop cut through {} faces", em::loop_cut(pm, sel, e, cuts)?)
            }
            EditOp::Translate(d) => {
                let local = inv3 * d;
                for (v, s) in pm.verts.iter_mut().zip(sel.iter()) {
                    if *s {
                        *v += local;
                    }
                }
                "Moved selection".into()
            }
            EditOp::Scale(f) => {
                let c = em::selection_center(pm, sel).ok_or("nothing selected")?;
                for (v, s) in pm.verts.iter_mut().zip(sel.iter()) {
                    if *s {
                        *v = c + (*v - c) * f;
                    }
                }
                "Scaled selection".into()
            }
        })
    }

    /// Edit-mode aware command handling. Returns None when the command isn't an edit command.
    fn exec_edit(&mut self, action: &str, c: &Command) -> Option<Result<String, String>> {
        let ensure = |ed: &mut Editor| -> Result<(), String> {
            if ed.edit.is_none() {
                let id = ed
                    .resolve_target(c.target.as_deref())
                    .into_iter()
                    .find(|id| ed.scene.get(*id).is_some_and(|o| o.is_mesh()))
                    .ok_or("no mesh object to edit")?;
                ed.enter_edit(id)?;
            }
            Ok(())
        };
        let num = |d: f64| c.number.unwrap_or(d);
        Some(match action {
            "edit_mode" | "enter_edit_mode" => {
                let id = self
                    .resolve_target(c.target.as_deref())
                    .into_iter()
                    .find(|id| self.scene.get(*id).is_some_and(|o| o.is_mesh()));
                match id {
                    Some(id) => self.enter_edit(id),
                    None => Err("no mesh object to edit".into()),
                }
            }
            "object_mode" | "exit_edit_mode" => {
                self.exit_edit();
                Ok("Object mode".into())
            }
            "sculpt_mode" | "sculpt" => {
                match self.resolve_target(c.target.as_deref()).into_iter().find(|id| self.scene.get(*id).is_some_and(|o| o.is_mesh())) {
                    None => Err("no mesh object to sculpt".into()),
                    Some(id) => {
                        if !self.in_batch {
                            self.snapshot();
                        }
                        self.enter_sculpt(id)
                    }
                }
            }
            "select_faces" | "select_by_normal" => ensure(self).and_then(|_| {
                let dir = vec3(&c.vector).unwrap_or(Vec3::Z);
                self.edit_op(EditOp::SelectNormal(
                    dir,
                    c.name.as_deref() == Some("extend"),
                ))
            }),
            "select_all_mesh" => ensure(self).and_then(|_| self.edit_op(EditOp::SelectAll)),
            "extrude" => ensure(self).and_then(|_| {
                let dist = vec3(&c.vector)
                    .map(|v| v.length())
                    .unwrap_or(num(1.0) as f32);
                self.edit_op(EditOp::Extrude(dist))
            }),
            "inset" => ensure(self).and_then(|_| self.edit_op(EditOp::Inset(num(0.25) as f32))),
            "bevel" => {
                let w = num(0.1).clamp(0.001, 100.0) as f32;
                let segs = c.segments.unwrap_or(if c.name.as_deref() == Some("round") { 4.0 } else { 1.0 }).clamp(1.0, 16.0) as usize;
                if self.editing() {
                    self.edit_op(EditOp::Bevel(w, segs))
                } else {
                    // object mode: bevel every edge of the target meshes
                    let ids: Vec<u64> = self.resolve_target(c.target.as_deref()).into_iter().filter(|id| self.scene.get(*id).is_some_and(|o| o.is_mesh())).collect();
                    if ids.is_empty() {
                        Err("no mesh object to bevel".into())
                    } else {
                        if !self.in_batch {
                            self.snapshot();
                        }
                        let mut done = vec![];
                        let mut err = None;
                        for id in &ids {
                            let o = self.scene.get_mut(*id).unwrap();
                            let mut pm = o.mesh.clone().unwrap_or_else(|| o.primitive.poly(o.detail));
                            let mut sel = vec![true; pm.verts.len()];
                            let s = (glam::Mat3::from_mat4(o.transform.matrix()) * Vec3::ONE.normalize()).length().max(1e-6);
                            match em::bevel(&mut pm, &mut sel, w / s, segs) {
                                Ok(_) => {
                                    o.mesh = Some(pm);
                                    if segs > 1 {
                                        // rounded bevels look right with auto smooth
                                        o.smooth_override = Some(true);
                                        if o.auto_smooth == 0.0 {
                                            o.auto_smooth = 30.0;
                                        }
                                    }
                                    done.push(*id);
                                }
                                Err(e) => err = Some(e),
                            }
                        }
                        if done.is_empty() {
                            if !self.in_batch {
                                self.undo_stack.pop();
                            }
                            Err(err.unwrap_or_default())
                        } else {
                            Ok(format!("Beveled {}", self.names(&done)))
                        }
                    }
                }
            }
            "loop_cut" | "loopcut" => ensure(self).and_then(|_| {
                self.edit_op(EditOp::LoopCutAxis {
                    axis: vec3(&c.vector).unwrap_or(Vec3::Z),
                    cuts: num(1.0).max(1.0) as usize,
                })
            }),
            "merge" => ensure(self).and_then(|_| self.edit_op(EditOp::Merge)),
            "flip_normals" => ensure(self).and_then(|_| self.edit_op(EditOp::FlipNormals)),
            "subdivide" if self.editing() => self.edit_op(EditOp::Subdivide),
            "delete" | "remove" if self.editing() && c.target.is_none() => {
                self.edit_op(EditOp::Delete)
            }
            "select" if self.editing() && matches!(c.target.as_deref(), Some("all") | None) => {
                self.edit_op(EditOp::SelectAll)
            }
            "deselect_all" | "deselect" if self.editing() => self.edit_op(EditOp::SelectNone),
            "move" | "translate" if self.editing() && c.target.is_none() => {
                match vec3(&c.vector).or_else(|| vec3(&c.location)) {
                    Some(d) => self.edit_op(EditOp::Translate(d)),
                    None => Err("move needs a vector".into()),
                }
            }
            "scale" | "resize" if self.editing() && c.target.is_none() => {
                match vec3_or_scalar(&c.vector)
                    .or_else(|| vec3_or_scalar(&c.scale))
                    .or_else(|| c.number.map(|n| Vec3::splat(n as f32)))
                {
                    Some(f) => self.edit_op(EditOp::Scale(f)),
                    None => Err("scale needs a factor".into()),
                }
            }
            _ => return None,
        })
    }
}

enum Special {
    Light(crate::scene::LightType),
    Camera,
    Armature,
}

fn parse_special(s: &str) -> Option<Special> {
    use crate::scene::LightType;
    let s = s.to_lowercase().replace(['_', '-', ' '], "");
    Some(match s.as_str() {
        "light" | "pointlight" | "point" | "lamp" | "bulb" => Special::Light(LightType::Point),
        "sun" | "sunlight" | "sunlamp" | "directional" | "directionallight" => {
            Special::Light(LightType::Sun)
        }
        "spot" | "spotlight" | "spotlamp" => Special::Light(LightType::Spot),
        "camera" | "cam" => Special::Camera,
        "armature" | "bone" | "bones" | "skeleton" | "rig" => Special::Armature,
        _ => return None,
    })
}

fn apply_light_fields(o: &mut crate::scene::Object, c: &Command) {
    if let crate::scene::ObjectKind::Light(l) = &mut o.kind {
        if let Some(col) = c.color.as_ref().and_then(parse_color) {
            l.color = col;
        }
        if let Some(p) = c.power {
            l.power = p.max(0.0) as f32;
        }
        if l.light_type == crate::scene::LightType::Spot && c.action.contains("light") {
            if let Some(n) = c.number {
                l.spot_angle = (n as f32).clamp(1.0, 180.0);
            }
        }
    }
}

impl Editor {
    /// A point from `[x,y,z]` or an object name.
    fn resolve_point(&self, v: &Value) -> Option<Vec3> {
        if let Value::String(s) = v {
            if let Some(id) = self.scene.find_by_name(s) {
                return self.scene.get(id).map(|o| o.transform.location);
            }
        }
        vec3(&Some(v.clone()))
    }
}

fn base_name(n: &str) -> &str {
    match n.rsplit_once('.') {
        Some((b, suf)) if suf.len() == 3 && suf.chars().all(|c| c.is_ascii_digit()) => b,
        _ => n,
    }
}

fn apply_material_fields(o: &mut crate::scene::Object, c: &Command) {
    if let Some(v) = c.metallic {
        o.material.metallic = v.clamp(0.0, 1.0) as f32;
    }
    if let Some(v) = c.roughness {
        o.material.roughness = v.clamp(0.0, 1.0) as f32;
    }
    if let Some(v) = c.emission {
        o.material.emission = v.clamp(0.0, 10.0) as f32;
    }
    if let Some(v) = c.alpha {
        o.material.alpha = v.clamp(0.05, 1.0) as f32;
    }
}

fn make_modifier(
    kind: &str,
    number: Option<f64>,
    vector: Option<Vec3>,
) -> Result<Modifier, String> {
    let k = kind.to_lowercase();
    Ok(if k.contains("mirror") {
        let v = vector.unwrap_or(Vec3::X);
        Modifier::Mirror {
            x: v.x != 0.0,
            y: v.y != 0.0,
            z: v.z != 0.0,
        }
    } else if k.contains("radial") || k.contains("circle") || k.contains("ring") {
        Modifier::Radial {
            count: number.unwrap_or(8.0).clamp(1.0, 200.0) as u32,
            radius: vector
                .map(|v| v.length())
                .filter(|r| *r > 0.0)
                .unwrap_or(3.0),
        }
    } else if k.contains("subsurf") || k.contains("subdiv") || k.contains("smooth") || k == "subd" {
        Modifier::Subsurf {
            levels: number.unwrap_or(2.0).clamp(0.0, 4.0) as u32,
        }
    } else if k.contains("array") || k.contains("repeat") || k.contains("row") {
        Modifier::Array {
            count: number.unwrap_or(3.0).clamp(1.0, 200.0) as u32,
            offset: vector.unwrap_or(Vec3::new(2.2, 0.0, 0.0)),
        }
    } else {
        return Err(format!("unknown modifier '{kind}' (array, mirror, radial, subsurf)"));
    })
}

/// A flat, forgiving command record. Small local LLMs are much more reliable with a single
/// flat object + "action" enum than with deeply nested tagged unions.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct Command {
    pub action: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub primitive: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub location: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rotation: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scale: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vector: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub color: Option<Value>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "lenient_f64"
    )]
    pub number: Option<f64>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "lenient_f64"
    )]
    pub metallic: Option<f64>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "lenient_f64"
    )]
    pub roughness: Option<f64>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "lenient_f64"
    )]
    pub emission: Option<f64>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "lenient_f64"
    )]
    pub alpha: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub modifier: Option<String>,
    /// The other object of a two-object operation (boolean cutter).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub object: Option<String>,
    /// Parent object name (for add / parent).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent: Option<String>,
    /// Collection name (for add / move_to_collection).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub collection: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub view: Option<String>,
    /// Light power (W) or sun strength.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "lenient_f64"
    )]
    pub power: Option<f64>,
    /// Bevel profile segments.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "lenient_f64"
    )]
    pub segments: Option<f64>,
    /// Point [x,y,z] or object name to aim a camera/light at.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub look_at: Option<Value>,
}

fn lenient_f64<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Option<f64>, D::Error> {
    let v = Option::<Value>::deserialize(d)?;
    Ok(match v {
        Some(Value::Number(n)) => n.as_f64(),
        Some(Value::String(s)) => s.trim().parse().ok(),
        Some(Value::Array(a)) => a.first().and_then(|x| x.as_f64()),
        _ => None,
    })
}

impl Command {
    pub fn new(action: &str) -> Self {
        Self {
            action: action.into(),
            ..Default::default()
        }
    }
}

pub fn vec3(v: &Option<Value>) -> Option<Vec3> {
    match v.as_ref()? {
        Value::Array(a) if !a.is_empty() => {
            let g = |i: usize| a.get(i).and_then(num).unwrap_or(0.0);
            Some(Vec3::new(g(0), g(1), g(2)))
        }
        Value::Object(m) => {
            let g = |k: &str| m.get(k).and_then(num).unwrap_or(0.0);
            Some(Vec3::new(g("x"), g("y"), g("z")))
        }
        Value::String(s) => {
            let p: Vec<f32> = s
                .split(|c: char| c == ',' || c.is_whitespace())
                .filter_map(|x| x.trim_matches(['[', ']', '(', ')']).parse().ok())
                .collect();
            if p.is_empty() {
                None
            } else {
                Some(Vec3::new(
                    p[0],
                    *p.get(1).unwrap_or(&0.0),
                    *p.get(2).unwrap_or(&0.0),
                ))
            }
        }
        _ => None,
    }
}

pub fn vec3_or_scalar(v: &Option<Value>) -> Option<Vec3> {
    match v.as_ref()? {
        Value::Number(n) => n.as_f64().map(|f| Vec3::splat(f as f32)),
        _ => {
            // A one-element array means uniform scale.
            if let Some(Value::Array(a)) = v.as_ref() {
                if a.len() == 1 {
                    return a[0].as_f64().map(|f| Vec3::splat(f as f32));
                }
            }
            vec3(v)
        }
    }
}

fn num(v: &Value) -> Option<f32> {
    match v {
        Value::Number(n) => n.as_f64().map(|f| f as f32),
        Value::String(s) => s.trim().parse().ok(),
        _ => None,
    }
}

pub fn parse_color(v: &Value) -> Option<[f32; 3]> {
    match v {
        Value::String(s) => parse_color_str(s),
        Value::Array(a) if a.len() >= 3 => {
            let mut c = [0.0f32; 3];
            for (i, x) in a.iter().take(3).enumerate() {
                c[i] = num(x)?;
            }
            if c.iter().any(|x| *x > 1.0) {
                for x in &mut c {
                    *x /= 255.0;
                }
            }
            Some(c)
        }
        _ => None,
    }
}

pub const NAMED_COLORS: &[(&str, [f32; 3])] = &[
    ("red", [0.85, 0.12, 0.1]),
    ("green", [0.15, 0.7, 0.2]),
    ("blue", [0.12, 0.3, 0.9]),
    ("yellow", [0.95, 0.85, 0.15]),
    ("orange", [0.95, 0.45, 0.08]),
    ("purple", [0.5, 0.2, 0.75]),
    ("violet", [0.56, 0.0, 1.0]),
    ("pink", [0.95, 0.45, 0.7]),
    ("magenta", [0.9, 0.1, 0.8]),
    ("cyan", [0.1, 0.8, 0.9]),
    ("teal", [0.0, 0.5, 0.5]),
    ("white", [0.95, 0.95, 0.95]),
    ("black", [0.03, 0.03, 0.03]),
    ("gray", [0.5, 0.5, 0.5]),
    ("grey", [0.5, 0.5, 0.5]),
    ("silver", [0.75, 0.75, 0.78]),
    ("gold", [1.0, 0.77, 0.3]),
    ("brown", [0.45, 0.28, 0.12]),
    ("wood", [0.55, 0.36, 0.2]),
    ("lime", [0.6, 0.95, 0.1]),
    ("navy", [0.05, 0.08, 0.35]),
    ("beige", [0.9, 0.85, 0.7]),
    ("maroon", [0.5, 0.05, 0.08]),
    ("turquoise", [0.25, 0.88, 0.82]),
    ("sky", [0.53, 0.81, 0.98]),
    ("copper", [0.8, 0.45, 0.25]),
    ("bronze", [0.7, 0.45, 0.2]),
    ("grass", [0.3, 0.6, 0.15]),
    ("sand", [0.86, 0.76, 0.55]),
];

pub fn parse_color_str(s: &str) -> Option<[f32; 3]> {
    let s = s.trim().to_lowercase();
    if let Some(h) = s.strip_prefix('#') {
        if h.len() == 6 {
            let p = |i: usize| {
                u8::from_str_radix(&h[i..i + 2], 16)
                    .ok()
                    .map(|v| v as f32 / 255.0)
            };
            return Some([p(0)?, p(2)?, p(4)?]);
        }
    }
    let (light, dark) = (s.starts_with("light "), s.starts_with("dark "));
    let base = s
        .trim_start_matches("light ")
        .trim_start_matches("dark ")
        .trim();
    let c = NAMED_COLORS
        .iter()
        .find(|(n, _)| *n == base)
        .map(|(_, c)| *c)?;
    Some(if light {
        c.map(|x| x + (1.0 - x) * 0.45)
    } else if dark {
        c.map(|x| x * 0.45)
    } else {
        c
    })
}

fn color_dist(a: [f32; 3], b: [f32; 3]) -> f32 {
    (0..3).map(|i| (a[i] - b[i]).powi(2)).sum::<f32>()
}

#[cfg(test)]
mod tests {
    use super::*;
    fn cmd(json: &str) -> Command {
        serde_json::from_str(json).unwrap()
    }
    #[test]
    fn exec_add_move_color_undo() {
        let mut e = Editor::default();
        let log = e.execute(&[
            cmd(r#"{"action":"add","primitive":"sphere","name":"Ball","location":[0,0,3],"color":"red"}"#),
            cmd(r#"{"action":"move","target":"Ball","vector":[1,0,0]}"#),
            cmd(r#"{"action":"scale","target":"Ball","scale":2}"#),
        ]);
        assert!(log.iter().all(|l| l.starts_with('✔')), "{log:?}");
        let b = e.scene.get(e.scene.find_by_name("ball").unwrap()).unwrap();
        assert_eq!(b.transform.location, Vec3::new(1.0, 0.0, 3.0));
        assert_eq!(b.transform.scale, Vec3::splat(2.0));
        assert_eq!(b.material.color, [0.85, 0.12, 0.1]);
        assert!(e.undo());
        assert_eq!(e.scene.objects.len(), 3);
    }
    #[test]
    fn targets_by_kind_and_color() {
        let mut e = Editor::default();
        e.execute(&[
            cmd(r#"{"action":"add","primitive":"sphere","color":"blue"}"#),
            cmd(r#"{"action":"add","primitive":"sphere","color":"red"}"#),
        ]);
        assert_eq!(e.resolve_target(Some("spheres")).len(), 2);
        assert_eq!(e.resolve_target(Some("all")).len(), 5);
        assert_eq!(e.resolve_target(Some("the red one")).len(), 1);
    }
    #[test]
    fn lenient_values() {
        let c = cmd(
            r#"{"action":"add","primitive":"cube","location":"1, 2, 3","number":"2","color":[255,0,0]}"#,
        );
        assert_eq!(vec3(&c.location), Some(Vec3::new(1.0, 2.0, 3.0)));
        assert_eq!(c.number, Some(2.0));
        assert_eq!(
            parse_color(c.color.as_ref().unwrap()),
            Some([1.0, 0.0, 0.0])
        );
    }
    #[test]
    fn lights_and_cameras() {
        let mut e = Editor::default();
        let log = e.execute(&[
            cmd(r#"{"action":"add","primitive":"spot_light","name":"Key","location":[3,-3,4],"look_at":"Cube","color":"orange","power":500}"#),
            cmd(r#"{"action":"add","primitive":"camera","name":"Cam2","location":[0,-10,2],"look_at":[0,0,0]}"#),
            cmd(r#"{"action":"set_light","target":"Key","number":30}"#),
        ]);
        assert!(log.iter().all(|l| l.starts_with('✔')), "{log:?}");
        let key = e.scene.get(e.scene.find_by_name("Key").unwrap()).unwrap();
        let l = key.light().unwrap();
        assert_eq!(l.light_type, crate::scene::LightType::Spot);
        assert_eq!(l.power, 500.0);
        assert_eq!(l.spot_angle, 30.0);
        assert!(
            (key.forward() - (Vec3::ZERO - Vec3::new(3.0, -3.0, 4.0)).normalize()).length() < 1e-3
        );
        assert_eq!(e.resolve_target(Some("lights")).len(), 2);
        assert_eq!(e.resolve_target(Some("cameras")).len(), 2);
        assert_eq!(e.resolve_target(Some("cubes")).len(), 1);
    }

    #[test]
    fn edit_mode_commands() {
        let mut e = Editor::default();
        let log = e.execute(&[
            cmd(r#"{"action":"edit_mode","target":"Cube"}"#),
            cmd(r#"{"action":"select_faces","vector":[0,0,1]}"#),
            cmd(r#"{"action":"extrude","number":2}"#),
            cmd(r#"{"action":"inset","number":0.3}"#),
            cmd(r#"{"action":"loop_cut","vector":[0,0,1],"number":2}"#),
            cmd(r#"{"action":"object_mode"}"#),
        ]);
        assert!(log.iter().all(|l| l.starts_with('✔')), "{log:?}");
        let cube = e.scene.get(e.scene.find_by_name("Cube").unwrap()).unwrap();
        let pm = cube.mesh.as_ref().unwrap();
        let zmax = pm.verts.iter().map(|v| v.z).fold(f32::MIN, f32::max);
        assert!((zmax - 3.0).abs() < 1e-4, "{zmax}");
        assert!(pm.edge_faces().values().all(|f| f.len() == 2));
        assert!(!e.editing());
        // one undo step for the whole batch
        assert!(e.undo());
        assert!(
            e.scene
                .get(e.scene.find_by_name("Cube").unwrap())
                .unwrap()
                .mesh
                .is_none()
        );
    }

    #[test]
    fn duplicate_count() {
        let mut e = Editor::default();
        e.execute(&[cmd(r#"{"action":"duplicate","number":3,"vector":[0,3,0]}"#)]);
        assert_eq!(e.scene.objects.len(), 6);
        assert_eq!(e.undo_stack.len(), 1);
    }

    #[test]
    fn parenting_follows_and_undoes() {
        let mut e = Editor::default();
        e.execute(&[
            cmd(r#"{"action":"add","primitive":"cube","name":"Car","location":[0,0,0]}"#),
            cmd(r#"{"action":"add","primitive":"cylinder","name":"Wheel","location":[1,0,0],"parent":"Car"}"#),
        ]);
        let car = e.scene.find_by_name("Car").unwrap();
        let wheel = e.scene.find_by_name("Wheel").unwrap();
        assert_eq!(e.scene.get(wheel).unwrap().parent, Some(car));
        e.execute(&[cmd(r#"{"action":"move","target":"Car","vector":[0,5,0]}"#)]);
        let w = e.scene.get(wheel).unwrap().transform.location;
        assert!((w - Vec3::new(1.0, 5.0, 0.0)).length() < 1e-5, "{w}");
        e.execute(&[cmd(r#"{"action":"rotate","target":"Car","vector":[0,0,90]}"#)]);
        let w = e.scene.get(wheel).unwrap().transform.location;
        assert!((w - Vec3::new(0.0, 6.0, 0.0)).length() < 1e-4, "{w}");
        // moving both together does not double-move the child
        e.selection = vec![car, wheel];
        e.execute(&[cmd(r#"{"action":"move","vector":[0,0,1]}"#)]);
        let w = e.scene.get(wheel).unwrap().transform.location;
        assert!((w - Vec3::new(0.0, 6.0, 1.0)).length() < 1e-4, "{w}");
        // undo restores without the child jumping
        e.undo();
        e.propagate_parents();
        let w = e.scene.get(wheel).unwrap().transform.location;
        assert!((w - Vec3::new(0.0, 6.0, 0.0)).length() < 1e-4, "{w}");
        // no cycles; clear parent keeps the world position
        assert!(e.execute(&[cmd(r#"{"action":"parent","target":"Car","parent":"Wheel"}"#)])[0].starts_with('✖'));
        e.execute(&[cmd(r#"{"action":"clear_parent","target":"Wheel"}"#)]);
        assert_eq!(e.scene.get(wheel).unwrap().parent, None);
        // collections
        e.execute(&[cmd(r#"{"action":"move_to_collection","target":"Wheel","collection":"Parts"}"#)]);
        assert_eq!(e.scene.collections.len(), 1);
        e.execute(&[cmd(r#"{"action":"hide_collection","collection":"parts"}"#)]);
        assert!(!e.scene.get(wheel).unwrap().visible);
        // deleting the parent leaves no dangling link
        e.execute(&[cmd(r#"{"action":"parent","target":"Wheel","parent":"Car"}"#), cmd(r#"{"action":"delete","target":"Car"}"#)]);
        assert_eq!(e.scene.get(wheel).unwrap().parent, None);
    }
    #[test]
    fn bump_and_roughness_maps() {
        let mut ed = Editor::default();
        ed.execute(&[Command { action: "add".into(), primitive: Some("cube".into()), ..Default::default() }]);
        let name = ed.scene.objects.last().unwrap().name.clone();
        let t = || Some(name.clone());
        let log = ed.execute(&[Command { action: "bump".into(), target: t(), name: Some("auto".into()), ..Default::default() }]);
        assert!(log[0].starts_with('✖'), "no texture yet: {log:?}");
        ed.execute(&[Command { action: "set_texture".into(), target: t(), name: Some("bricks".into()), ..Default::default() }]);
        let log = ed.execute(&[
            Command { action: "bump".into(), target: t(), name: Some("auto".into()), number: Some(2.0), ..Default::default() },
            Command { action: "roughness_map".into(), target: t(), name: Some("auto".into()), ..Default::default() },
        ]);
        assert!(log.iter().all(|l| l.starts_with('✔')), "{log:?}");
        let m = ed.scene.objects.last().unwrap().material;
        assert!(m.normal_tex != 0 && m.rough_tex != 0 && m.normal_strength == 2.0 && m.roughness >= 0.6);
        assert_eq!(ed.scene.texture(m.normal_tex).unwrap().name, "bricks normal");
        // using an existing normal map as-is (no re-generation)
        let n0 = ed.scene.textures.len();
        ed.execute(&[Command { action: "bump".into(), target: t(), name: Some("bricks normal".into()), ..Default::default() }]);
        assert_eq!(ed.scene.textures.len(), n0);
        assert_eq!(ed.scene.objects.last().unwrap().material.normal_tex, m.normal_tex);
        // glTF round trip keeps both maps
        let (glb, _) = crate::gltf::export_glb(&ed.scene, &mut crate::scene::MeshCache::default());
        let mut sc = crate::scene::Scene::default();
        crate::gltf_import::import_bytes(&mut sc, &glb, None).unwrap();
        let o = sc.objects.iter().find(|o| o.name == name).unwrap();
        assert!(o.material.normal_tex != 0 && o.material.rough_tex != 0 && (o.material.normal_strength - 2.0).abs() < 1e-5, "{:?}", o.material);
        ed.execute(&[Command { action: "bump".into(), target: t(), name: Some("none".into()), ..Default::default() }]);
        assert_eq!(ed.scene.objects.last().unwrap().material.normal_tex, 0);
    }
    #[test]
    fn rig_pose_keyframes() {
        let mut ed = Editor::default();
        let log = ed.execute(&[
            cmd(r#"{"action":"add","primitive":"cylinder","name":"Tube","location":[0,0,1.5],"scale":[0.4,0.4,1.5]}"#),
            cmd(r#"{"action":"rig","target":"Tube","number":3}"#),
        ]);
        assert!(log.iter().all(|l| l.starts_with('✔')), "{log:?}");
        let tube = ed.scene.find_by_name("Tube").unwrap();
        let rig = ed.scene.find_by_name("Tube Rig").unwrap();
        assert_eq!(ed.scene.get(tube).unwrap().parent, Some(rig));
        let top = |ed: &Editor| {
            let o = ed.scene.get(tube).unwrap();
            let m = o.transform.matrix();
            o.posed.as_ref().unwrap().verts.iter().map(|v| m.transform_point3(*v)).fold(Vec3::splat(-9.0), |a, b| a.max(b))
        };
        let rest = top(&ed);
        assert!((rest.z - 3.0).abs() < 1e-3, "{rest}");
        // bend, keyframe, animate
        ed.set_frame(1);
        ed.execute(&[cmd(r#"{"action":"keyframe","target":"Tube Rig"}"#)]);
        ed.set_frame(20);
        let log = ed.execute(&[cmd(r#"{"action":"pose_bone","target":"Tube","name":"bone 2","rotation":[60,0,0]}"#), cmd(r#"{"action":"keyframe","target":"Tube Rig"}"#)]);
        assert!(log.iter().all(|l| l.starts_with('✔')), "{log:?}");
        let bent = top(&ed);
        assert!(bent.z < 2.9 && bent.y > 0.3, "the top bends towards +y: {bent}");
        ed.set_frame(1);
        assert!((top(&ed).z - 3.0).abs() < 1e-3, "frame 1 is straight again");
        // moving the armature carries the mesh
        ed.execute(&[cmd(r#"{"action":"move","target":"Tube Rig","vector":[2,0,0]}"#)]);
        assert!((top(&ed).x - (rest.x + 2.0)).abs() < 1e-3);
        // undo / save round trip keeps the rig
        let json = serde_json::to_string(&ed.scene).unwrap();
        let back: crate::scene::Scene = serde_json::from_str(&json).unwrap();
        assert!(back.get(tube).unwrap().skin.is_some() && back.get(rig).unwrap().armature.as_ref().unwrap().bones[1].keys.len() == 2);
        assert!(ed.toggle_pose().is_ok() && ed.pose == Some(rig));
    }
}
