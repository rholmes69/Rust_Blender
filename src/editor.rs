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
    /// Render the timeline (folder, empty = default).
    pub anim_render_request: Option<String>,
    /// Sculpt mode on this mesh object.
    pub sculpt: Option<u64>,
    /// Armature in Pose Mode and its active bone.
    pub pose: Option<u64>,
    pub active_bone: usize,
    /// Rigged mesh in Weight Paint mode (paints `active_bone` of its armature).
    pub weight_paint: Option<u64>,
    /// Armature in Edit Mode (rest bones) and the selected part of `active_bone`.
    pub bone_edit: Option<u64>,
    pub bone_part: crate::armature::BonePart,
    /// Shape key being edited: (object, key index, basis positions to restore).
    pub shape_edit: Option<(u64, usize, Vec<Vec3>)>,
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
            anim_render_request: None,
            parent_cache: Default::default(),
            sculpt: None,
            pose: None,
            active_bone: 0,
            weight_paint: None,
            bone_edit: None,
            bone_part: Default::default(),
            shape_edit: None,
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

    /// Replace an armature's bones after editing their rest positions, keeping bound meshes in
    /// place: `orig` is the armature before the edit, `orig_skins` the skins bound to it then and
    /// `map` sends each original bone to its new index (None = deleted).
    pub fn apply_bone_edit(&mut self, arm_id: u64, new_arm: crate::armature::Armature, orig: &crate::armature::Armature, orig_skins: &[(u64, crate::armature::Skin)], map: &[Option<usize>]) {
        let Some(a) = self.scene.get(arm_id) else { return };
        let aw = a.transform.matrix();
        let g_old = orig.rest_globals();
        let g_new = new_arm.rest_globals();
        let n = new_arm.bones.len();
        // new index of each original bone, deleted ones fall back to their nearest kept ancestor
        let fallback = |mut k: usize| -> u16 {
            loop {
                if let Some(Some(j)) = map.get(k) {
                    return *j as u16;
                }
                match orig.bones.get(k).and_then(|b| b.parent) {
                    Some(p) => k = p,
                    None => return 0,
                }
            }
        };
        for (oid, sk0) in orig_skins {
            let Some(o) = self.scene.get_mut(*oid) else { continue };
            let om = o.transform.matrix();
            let mut ibm = vec![glam::Mat4::IDENTITY; n];
            let mut have = vec![false; n];
            for (k, m) in map.iter().enumerate() {
                if let (Some(j), Some(ib)) = (m, sk0.inverse_bind.get(k)) {
                    ibm[*j] = g_new[*j].inverse() * g_old[k] * *ib;
                    have[*j] = true;
                }
            }
            for j in 0..n {
                if !have[j] {
                    ibm[j] = (aw * g_new[j]).inverse() * om;
                }
            }
            let joints = sk0.joints.iter().map(|q| q.map(|k| fallback(k as usize))).collect();
            o.skin = Some(crate::armature::Skin { armature: arm_id, joints, weights: sk0.weights.clone(), inverse_bind: ibm });
            o.posed_key = 0;
        }
        if let Some(a) = self.scene.get_mut(arm_id) {
            a.armature = Some(new_arm);
        }
        self.dirty = true;
    }

    /// Skins bound to an armature (for edits that must keep meshes in place).
    pub fn skins_of(&self, arm_id: u64) -> Vec<(u64, crate::armature::Skin)> {
        self.scene.objects.iter().filter_map(|o| o.skin.as_ref().filter(|s| s.armature == arm_id).map(|s| (o.id, s.clone()))).collect()
    }

    /// A ready-made object to scatter ("rock", "tree", "grass", "bush" or any primitive),
    /// created hidden at the origin with its base at z = 0.
    fn scatter_preset(&mut self, spec: &str) -> Result<(u64, ScatterPreset), String> {
        use crate::mesh::Primitive;
        let low = spec.to_lowercase();
        let (prim, detail, scale, color, rough, base, preset) = if low.contains("rock") || low.contains("stone") || low.contains("boulder") || low.contains("pebble") {
            (Primitive::IcoSphere, 1, Vec3::new(0.35, 0.3, 0.22), [0.42, 0.4, 0.37], 0.9, 0.3, ScatterPreset { count: 60, scale: (0.4, 1.6), tilt: 30.0 })
        } else if low.contains("tree") || low.contains("pine") || low.contains("forest") {
            (Primitive::Cone, 2, Vec3::new(0.35, 0.35, 0.95), [0.13, 0.36, 0.14], 0.8, 1.0, ScatterPreset { count: 30, scale: (0.7, 1.4), tilt: 4.0 })
        } else if low.contains("grass") || low.contains("blade") {
            (Primitive::Cone, 1, Vec3::new(0.025, 0.025, 0.22), [0.25, 0.55, 0.15], 0.7, 1.0, ScatterPreset { count: 1500, scale: (0.6, 1.4), tilt: 25.0 })
        } else if low.contains("bush") || low.contains("shrub") {
            (Primitive::IcoSphere, 2, Vec3::new(0.45, 0.45, 0.35), [0.18, 0.42, 0.16], 0.9, 0.6, ScatterPreset { count: 40, scale: (0.6, 1.4), tilt: 10.0 })
        } else if low.contains("flower") {
            (Primitive::UvSphere, 1, Vec3::splat(0.06), [0.95, 0.4, 0.6], 0.6, 2.0, ScatterPreset { count: 300, scale: (0.7, 1.3), tilt: 10.0 })
        } else if let Some(p) = Primitive::parse(&low) {
            (p, 2, Vec3::splat(0.25), [0.8, 0.8, 0.8], 0.5, 1.0, ScatterPreset { count: 50, scale: (0.8, 1.2), tilt: 0.0 })
        } else {
            return Err(format!("no object or preset {spec:?} to scatter (rock, tree, grass, bush, flower, or an object name)"));
        };
        let t = spec.trim();
        let label = if t.ends_with("es") && (t.ends_with("shes") || t.ends_with("ches")) { &t[..t.len() - 2] } else if t.ends_with('s') && !t.ends_with("ss") { &t[..t.len() - 1] } else { t };
        let name = if label.is_empty() { "Rock".to_string() } else { format!("{}{}", label[..1].to_uppercase(), &label[1..]) };
        let id = self.scene.add(prim, Some(&name));
        let o = self.scene.get_mut(id).unwrap();
        o.detail = detail;
        // bake the size into the mesh, with the base at the origin (sunk a little: `base`)
        let mut pm = prim.poly(detail);
        let zmin = pm.verts.iter().map(|v| v.z).fold(f32::MAX, f32::min);
        for v in &mut pm.verts {
            *v = Vec3::new(v.x * scale.x, v.y * scale.y, (v.z - zmin) * scale.z - scale.z * (1.0f32 - base).max(0.0) * 0.3);
        }
        if prim == Primitive::IcoSphere {
            // lumpy rocks
            for (i, v) in pm.verts.iter_mut().enumerate() {
                let h = ((i as f32 * 12.9898).sin() * 43758.545).fract().abs();
                *v *= 0.85 + 0.3 * h;
            }
        }
        o.mesh = Some(pm);
        o.material.color = color;
        o.material.roughness = rough;
        o.smooth_override = Some(prim != Primitive::IcoSphere || low.contains("bush"));
        o.visible = false;
        Ok((id, preset))
    }

    /// IK on bone `bi`: reach for `target` (an object name) or a new Empty at the bone's tail.
    pub fn add_ik(&mut self, aid: u64, bi: usize, chain: u32, target: Option<&str>) -> Result<String, String> {
        let a = self.scene.get(aid).ok_or("no armature")?;
        let arm = a.armature.as_ref().ok_or("not an armature")?;
        let tail = a.transform.matrix().transform_point3(arm.segments(&arm.pose_globals())[bi].1);
        let bname = arm.bones[bi].name.clone();
        let coll = a.collection;
        let tid = match target.and_then(|t| self.scene.find_by_name(t)).filter(|t| *t != aid) {
            Some(t) => t,
            None => {
                let t = self.scene.add_empty(Some(&format!("IK {bname}")));
                let o = self.scene.get_mut(t).unwrap();
                o.transform.location = tail;
                o.transform.scale = Vec3::splat(0.4);
                o.collection = coll;
                t
            }
        };
        let b = &mut self.scene.get_mut(aid).unwrap().armature.as_mut().unwrap().bones[bi];
        b.ik = Some(crate::armature::Ik { target: tid, chain });
        self.reset_parent_cache();
        Ok(self.scene.get(tid).unwrap().name.clone())
    }

    /// Add a shape key to a mesh (a copy of its current shape). Returns its index.
    pub fn add_shape_key(&mut self, id: u64, name: &str) -> Result<usize, String> {
        self.finish_shape_edit();
        let o = self.scene.get_mut(id).filter(|o| o.is_mesh()).ok_or("shape keys need a mesh object")?;
        if o.mesh.is_none() {
            o.mesh = Some(o.poly());
        }
        // new keys start from the basis (not from the current mix)
        let verts = o.mesh.as_ref().unwrap().verts.clone();
        let mut nm = name.to_string();
        let mut k = 1;
        while o.shape_keys.iter().any(|s| s.name == nm) {
            nm = format!("{name}.{k:03}");
            k += 1;
        }
        o.shape_keys.push(crate::shapekeys::ShapeKey { name: nm, verts, value: 0.0, keys: vec![] });
        o.posed_key = 0;
        Ok(o.shape_keys.len() - 1)
    }

    /// Show shape key `k` in the mesh for editing / sculpting (the basis is restored after).
    pub fn begin_shape_edit(&mut self, id: u64, k: usize) -> Result<String, String> {
        self.finish_shape_edit();
        let o = self.scene.get_mut(id).ok_or("no object")?;
        let key = o.shape_keys.get(k).ok_or("no such shape key")?;
        let pm = o.mesh.as_mut().ok_or("no mesh")?;
        if key.verts.len() != pm.verts.len() {
            return Err("this shape key doesn't match the mesh any more".into());
        }
        let basis = std::mem::replace(&mut pm.verts, key.verts.clone());
        let name = key.name.clone();
        o.shape_editing = true;
        o.posed = None;
        self.shape_edit = Some((id, k, basis));
        Ok(format!("Editing shape key \"{name}\" (edit or sculpt it; leave the mode to store it)"))
    }

    /// Store the edited shape into its key and bring back the basis.
    pub fn finish_shape_edit(&mut self) -> Option<String> {
        let (id, k, basis) = self.shape_edit.take()?;
        let o = self.scene.get_mut(id)?;
        o.shape_editing = false;
        o.posed_key = 0;
        let pm = o.mesh.as_mut()?;
        if pm.verts.len() != basis.len() {
            // the topology changed: the edited mesh becomes the basis and the keys are dropped
            o.shape_keys.clear();
            return Some("Topology changed: shape keys removed".into());
        }
        let shape = std::mem::replace(&mut pm.verts, basis);
        let key = o.shape_keys.get_mut(k)?;
        key.verts = shape;
        if key.value == 0.0 {
            key.value = 1.0;
        }
        Some(format!("Stored shape key \"{}\"", key.name))
    }

    /// Create a particle system from a preset (snow, sparks, rain, confetti, bubbles, smoke) or an
    /// existing object (`object`), emitted from `emitter` (or a new plane / point above the scene).
    pub fn add_particle_system(&mut self, spec: &str, emitter: Option<u64>, object: Option<&str>) -> Result<u64, String> {
        use crate::mesh::Primitive;
        let low = spec.to_lowercase();
        // (primitive, scale, colour, emission, alpha, rate, life, speed, spread, gravity, drag, shrink, emitter z / size)
        let preset = if low.contains("snow") {
            (Primitive::IcoSphere, Vec3::splat(0.03), [0.97, 0.98, 1.0], 0.0, 1.0, 30.0, 160, 0.8, 0.8, 0.5, 0.6, false, (5.0, 4.5))
        } else if low.contains("rain") {
            (Primitive::Cylinder, Vec3::new(0.008, 0.008, 0.12), [0.6, 0.75, 1.0], 0.0, 0.6, 120.0, 30, 0.0, 0.05, 30.0, 1.0, false, (8.0, 6.0))
        } else if low.contains("confetti") {
            (Primitive::Plane, Vec3::new(0.05, 0.03, 0.05), [1.0, 0.35, 0.6], 0.0, 1.0, 40.0, 120, 3.0, 0.9, 1.5, 0.4, false, (4.0, 1.0))
        } else if low.contains("bubble") {
            (Primitive::UvSphere, Vec3::splat(0.08), [0.8, 0.9, 1.0], 0.0, 0.35, 8.0, 90, 0.5, 0.4, -1.2, 0.7, true, (0.2, 1.0))
        } else if low.contains("smoke") || low.contains("steam") {
            (Primitive::IcoSphere, Vec3::splat(0.18), [0.55, 0.55, 0.58], 0.0, 0.35, 12.0, 70, 0.6, 0.35, -0.8, 0.6, false, (0.2, 0.5))
        } else if low.contains("fire") || low.contains("flame") {
            (Primitive::IcoSphere, Vec3::splat(0.12), [1.0, 0.45, 0.1], 3.0, 0.8, 30.0, 24, 1.0, 0.25, -2.5, 0.5, true, (0.2, 0.6))
        } else {
            // sparks / fountain
            (Primitive::IcoSphere, Vec3::splat(0.03), [1.0, 0.65, 0.2], 4.0, 1.0, 40.0, 30, 5.0, 0.35, 9.81, 0.95, true, (1.0, 0.3))
        };
        let (prim, scale, color, emission, alpha, rate, life, speed, spread, gravity, drag, shrink, (ez, es)) = preset;
        // the particle shape
        let src = match object.and_then(|n| self.scene.find_by_name(n)) {
            Some(id) => id,
            None => {
                let label = spec.trim();
                let nm = format!("{}{} Particle", label.get(..1).unwrap_or("P").to_uppercase(), label.get(1..).unwrap_or(""));
                let id = self.scene.add(prim, Some(&nm));
                let o = self.scene.get_mut(id).unwrap();
                o.detail = 1;
                let mut pm = prim.poly(1);
                for v in &mut pm.verts {
                    *v *= scale;
                }
                o.mesh = Some(pm);
                o.material.color = color;
                o.material.emission = emission;
                o.material.alpha = alpha;
                o.material.roughness = 0.3;
                o.smooth_override = Some(true);
                o.visible = false;
                id
            }
        };
        // where they come from
        let em = match emitter {
            Some(e) => e,
            None => {
                let id = self.scene.add(Primitive::Plane, Some(&format!("{} Emitter", spec.trim())));
                let o = self.scene.get_mut(id).unwrap();
                o.transform.location = Vec3::new(0.0, 0.0, ez);
                o.transform.scale = Vec3::splat(es);
                // a downward-facing sheet for snow/rain, upward for the rest
                if gravity > 3.0 && speed < 1.0 || low.contains("snow") {
                    o.transform.rotation = Vec3::new(180.0, 0.0, 0.0);
                }
                o.visible = false;
                id
            }
        };
        let name = format!("{} Particles", spec.trim());
        let pid = self.scene.add(Primitive::Cube, Some(&name));
        let mut ps = crate::physics::Particles::new(src, em);
        (ps.rate, ps.lifetime, ps.speed, ps.spread, ps.gravity, ps.drag, ps.shrink) = (rate, life, speed, spread, gravity, drag, shrink);
        if emitter.is_none() && (low.contains("snow") || low.contains("rain")) {
            ps.preroll = life;
        }
        let (mat, coll) = (self.scene.get(src).unwrap().material, self.scene.get(em).unwrap().collection);
        let o = self.scene.get_mut(pid).unwrap();
        o.material = mat;
        o.smooth_override = Some(true);
        o.collection = coll;
        o.particles = Some(ps);
        Ok(pid)
    }

    /// Simulate a particle system over the timeline.
    pub fn bake_particle_system(&mut self, pid: u64) -> Result<String, String> {
        let ps = self.scene.get(pid).and_then(|o| o.particles.clone()).ok_or("not a particle system")?;
        let em = self.scene.get(ps.emitter).ok_or("the emitter is gone")?;
        let (epm, ew) = (em.poly(), em.transform.matrix());
        let colliders: Vec<crate::physics::Collider> = if ps.collide {
            self.scene
                .objects
                .iter()
                .filter(|o| o.visible && o.is_mesh() && o.id != ps.emitter && o.id != pid && o.scatter.is_none() && o.particles.is_none())
                .map(|o| {
                    let p = o.poly();
                    let lo = p.verts.iter().fold(Vec3::splat(f32::MAX), |a, b| a.min(*b));
                    let hi = p.verts.iter().fold(Vec3::splat(f32::MIN), |a, b| a.max(*b));
                    crate::physics::collider_for(o, (lo, hi))
                })
                .collect()
        } else {
            vec![]
        };
        let (f0, f1) = (self.scene.frame_start, self.scene.frame_end.max(self.scene.frame_start + 1));
        let frames = ((f1 - f0 + 1) as usize).min(2000);
        let cache = crate::physics::bake_particles(&ps, &epm, ew, &colliders, frames, self.scene.fps.max(1) as f32);
        let peak = cache.iter().map(|f| f.len()).max().unwrap_or(0);
        let o = self.scene.get_mut(pid).unwrap();
        let p = o.particles.as_mut().unwrap();
        p.cache = std::sync::Arc::new(cache);
        p.start = f0;
        o.posed_key = 0;
        let n = o.name.clone();
        self.set_frame(self.frame);
        Ok(format!("\"{n}\": up to {peak} particles over frames {f0}–{f1} — press play"))
    }

    /// Re-run stored simulations (after opening a file).
    pub fn rebake_physics(&mut self) {
        let cloths: Vec<String> = self.scene.objects.iter().filter(|o| o.cloth.is_some()).map(|o| o.name.clone()).collect();
        for n in cloths {
            let _ = self.exec_one(&Command { action: "bake_cloth".into(), target: Some(n), ..Default::default() });
        }
        if self.scene.objects.iter().any(|o| o.rigid.as_ref().is_some_and(|r| !r.passive)) {
            self.bake_rigid_bodies();
        }
        let ps: Vec<u64> = self.scene.objects.iter().filter(|o| o.particles.is_some()).map(|o| o.id).collect();
        for id in ps {
            let _ = self.bake_particle_system(id);
        }
    }

    /// Simulate every active rigid body over the timeline (other meshes are obstacles).
    pub fn bake_rigid_bodies(&mut self) -> String {
        // start positions
        for o in self.scene.objects.iter_mut() {
            if let Some(t) = o.rigid.as_ref().filter(|r| !r.passive).and_then(|r| r.cache.first().copied().or(r.origin)) {
                o.transform = t;
            }
        }
        let active: Vec<u64> = self.scene.objects.iter().filter(|o| o.visible && o.is_mesh() && o.rigid.as_ref().is_some_and(|r| !r.passive)).map(|o| o.id).collect();
        if active.is_empty() {
            return "No active rigid bodies".into();
        }
        let bounds = |o: &crate::scene::Object| {
            let p = o.poly();
            let lo = p.verts.iter().fold(Vec3::splat(f32::MAX), |a, b| a.min(*b));
            let hi = p.verts.iter().fold(Vec3::splat(f32::MIN), |a, b| a.max(*b));
            (lo, hi)
        };
        let bodies: Vec<crate::physics::BodyIn> = active
            .iter()
            .map(|id| {
                let o = self.scene.get(*id).unwrap();
                let r = o.rigid.as_ref().unwrap();
                let s = o.transform.scale.abs();
                let sphere = o.mesh.is_none() && matches!(o.primitive, crate::mesh::Primitive::UvSphere | crate::mesh::Primitive::IcoSphere) && (s.x - s.y).abs() < 1e-3 && (s.y - s.z).abs() < 1e-3;
                crate::physics::BodyIn { transform: o.transform, sphere, bounds: bounds(o), bounce: r.bounce, friction: r.friction }
            })
            .collect();
        let statics: Vec<crate::physics::Collider> = self
            .scene
            .objects
            .iter()
            .filter(|o| o.visible && o.is_mesh() && o.scatter.is_none() && o.cloth.is_none() && !active.contains(&o.id))
            .map(|o| crate::physics::collider_for(o, bounds(o)))
            .collect();
        let (f0, f1) = (self.scene.frame_start, self.scene.frame_end.max(self.scene.frame_start + 1));
        let frames = ((f1 - f0 + 1) as usize).min(2000);
        let out = crate::physics::bake_rigid(&bodies, &statics, frames, self.scene.fps.max(1) as f32, 9.81);
        for (id, tr) in active.iter().zip(out) {
            if let Some(r) = self.scene.get_mut(*id).and_then(|o| o.rigid.as_mut()) {
                r.origin = tr.first().copied();
                r.cache = std::sync::Arc::new(tr);
                r.start = f0;
            }
        }
        self.set_frame(self.frame);
        format!("Simulated {} rigid bodies with {} obstacles over frames {f0}–{f1} — press play", active.len(), statics.len())
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
        if let Some(id) = self.bone_edit {
            if !self.scene.get(id).is_some_and(|o| o.armature.as_ref().is_some_and(|a| !a.bones.is_empty())) {
                self.bone_edit = None;
            }
        }
        if let Some(id) = self.weight_paint {
            if !self.scene.get(id).is_some_and(|o| o.skin.is_some()) {
                self.weight_paint = None;
            }
        }
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
        self.scene.update_derived();
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
        self.scene.update_derived();
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
                    Some(Special::Empty) => self.scene.add_empty(c.name.as_deref()),
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
            "weight_paint" | "weight_paint_mode" | "paint_weights" => {
                if let Some(t) = c.target.as_deref().and_then(|t| self.scene.find_by_name(t)) {
                    self.active = Some(t);
                    self.weight_paint = None;
                }
                if let Some(b) = c.name.as_deref() {
                    if let Some(i) = self.active.and_then(|id| self.scene.get(id)).and_then(|o| o.skin.as_ref()).and_then(|s| self.scene.get(s.armature)).and_then(|a| a.armature.as_ref()).and_then(|a| a.find(b)) {
                        self.active_bone = i;
                    }
                }
                if self.weight_paint.is_some() {
                    return Ok("Already in weight paint".into());
                }
                self.toggle_weight_paint()
            }
            "set_action" | "play_action" | "action" | "new_action" => {
                let aid = c
                    .target
                    .as_deref()
                    .and_then(|t| self.scene.find_by_name(t))
                    .and_then(|t| if self.scene.get(t)?.armature.is_some() { Some(t) } else { self.scene.get(t)?.skin.as_ref().map(|s| s.armature) })
                    .or(self.pose)
                    .or_else(|| self.active.filter(|id| self.scene.get(*id).is_some_and(|o| o.armature.is_some())))
                    .or_else(|| self.scene.objects.iter().find(|o| o.armature.as_ref().is_some_and(|a| !a.actions.is_empty())).map(|o| o.id))
                    .or_else(|| self.scene.objects.iter().find(|o| o.armature.is_some()).map(|o| o.id))
                    .ok_or("no armature in the scene")?;
                let arm = self.scene.get_mut(aid).unwrap().armature.as_mut().unwrap();
                if c.action == "new_action" {
                    let i = arm.new_action(c.name.as_deref().unwrap_or("Action"));
                    return Ok(format!("New action \"{}\" (pose bones and press I to key it)", arm.actions[i].name));
                }
                let list: Vec<String> = arm.actions.iter().map(|a| a.name.clone()).collect();
                let name = c.name.clone().ok_or(format!("which action? ({})", if list.is_empty() { "none stored".into() } else { list.join(", ") }))?;
                let i = arm.find_action(&name).ok_or(format!("no action {name:?} (actions: {})", list.join(", ")))?;
                let playing = self.playing || c.action == "play_action";
                let blend = c.number.map(|n| n as f32).unwrap_or(if playing { 6.0 } else { 0.0 });
                let range = match arm.actions.get(i).and_then(|a| a.range()) {
                    Some((lo, _)) if blend > 0.0 => arm.set_action_blended(i, lo as f32, blend),
                    _ => arm.set_action(i),
                };
                let an = arm.actions[i].name.clone();
                if let Some((lo, hi)) = range {
                    self.scene.frame_start = lo;
                    self.scene.frame_end = hi.max(lo + 1);
                    self.set_frame(lo);
                }
                if c.action == "play_action" || c.primitive.as_deref() == Some("play") {
                    self.playing = true;
                }
                Ok(format!("Action \"{an}\"{}", range.map(|(a, b)| format!(" (frames {a}–{b})")).unwrap_or_default()))
            }
            "add_ik" | "ik" | "remove_ik" => {
                let aid = c
                    .target
                    .as_deref()
                    .and_then(|t| self.scene.find_by_name(t))
                    .and_then(|t| if self.scene.get(t)?.armature.is_some() { Some(t) } else { self.scene.get(t)?.skin.as_ref().map(|s| s.armature) })
                    .or(self.pose)
                    .or_else(|| self.active.filter(|id| self.scene.get(*id).is_some_and(|o| o.armature.is_some())))
                    .or_else(|| self.scene.objects.iter().find(|o| o.armature.is_some()).map(|o| o.id))
                    .ok_or("no armature in the scene")?;
                let arm = self.scene.get(aid).unwrap().armature.as_ref().unwrap();
                let bi = match c.name.as_deref() {
                    Some(n) => arm.find(n).ok_or(format!("no bone {n:?}"))?,
                    None => self.active_bone.min(arm.bones.len().saturating_sub(1)),
                };
                let bname = arm.bones[bi].name.clone();
                if c.action == "remove_ik" || c.primitive.as_deref() == Some("none") {
                    self.scene.get_mut(aid).unwrap().armature.as_mut().unwrap().bones[bi].ik = None;
                    return Ok(format!("Removed IK from \"{bname}\""));
                }
                let chain = c.number.map(|n| n.max(0.0) as u32).unwrap_or(2);
                self.add_ik(aid, bi, chain, c.object.as_deref()).map(|t| format!("IK on \"{bname}\" (chain {chain}) → move \"{t}\" to pose the limb"))
            }
            "add_shape_key" | "shape_key" | "set_shape_key" | "edit_shape_key" | "remove_shape_key" => {
                let id = targets().into_iter().find(|id| self.scene.get(*id).is_some_and(|o| o.is_mesh())).ok_or("shape keys need a mesh object")?;
                let name = c.name.clone().unwrap_or_else(|| "Key".into());
                let find = |o: &crate::scene::Object| o.shape_keys.iter().position(|k| k.name.eq_ignore_ascii_case(&name)).or_else(|| o.shape_keys.iter().position(|k| k.name.to_lowercase().contains(&name.to_lowercase())));
                match c.action.as_str() {
                    "remove_shape_key" => {
                        self.finish_shape_edit();
                        let o = self.scene.get_mut(id).unwrap();
                        let k = find(o).ok_or(format!("no shape key {name:?}"))?;
                        o.shape_keys.remove(k);
                        o.posed_key = 0;
                        Ok(format!("Removed shape key \"{name}\""))
                    }
                    "edit_shape_key" => {
                        let k = find(self.scene.get(id).unwrap()).ok_or(format!("no shape key {name:?}"))?;
                        let msg = self.begin_shape_edit(id, k)?;
                        if self.edit.is_none() {
                            let _ = self.enter_edit(id);
                        }
                        Ok(msg)
                    }
                    _ => {
                        let existing = find(self.scene.get(id).unwrap());
                        let k = match existing {
                            Some(k) if c.action != "add_shape_key" || (c.scale.is_none() && c.vector.is_none()) => k,
                            _ => {
                                let k = self.add_shape_key(id, &name)?;
                                // generated shapes: scale around the centre (squash / stretch) or puff along normals
                                let o = self.scene.get_mut(id).unwrap();
                                let pm = o.mesh.clone().unwrap();
                                let c0 = pm.verts.iter().sum::<Vec3>() / pm.verts.len().max(1) as f32;
                                let normals = crate::sculpt::vertex_normals(&pm);
                                let key = &mut o.shape_keys[k];
                                if let Some(sv) = vec3_or_scalar(&c.scale) {
                                    for v in &mut key.verts {
                                        *v = c0 + (*v - c0) * sv;
                                    }
                                }
                                if let Some(d) = vec3(&c.vector) {
                                    let lo = pm.verts.iter().map(|v| v.z).fold(f32::MAX, f32::min);
                                    let hi = pm.verts.iter().map(|v| v.z).fold(f32::MIN, f32::max);
                                    for (v, _) in key.verts.iter_mut().zip(&normals) {
                                        // shear: more at the top (a lean / wobble)
                                        let t = ((v.z - lo) / (hi - lo).max(1e-6)).clamp(0.0, 1.0);
                                        *v += d * t;
                                    }
                                }
                                if let Some(p) = c.power {
                                    for (v, n) in key.verts.iter_mut().zip(&normals) {
                                        *v += *n * p as f32;
                                    }
                                }
                                k
                            }
                        };
                        let o = self.scene.get_mut(id).unwrap();
                        if let Some(v) = c.number {
                            o.shape_keys[k].value = (v as f32).clamp(-1.0, 2.0);
                        } else if existing.is_none() && (c.scale.is_some() || c.vector.is_some() || c.power.is_some()) {
                            o.shape_keys[k].value = 1.0;
                        }
                        o.posed_key = 0;
                        let (kn, kv) = (o.shape_keys[k].name.clone(), o.shape_keys[k].value);
                        Ok(format!("Shape key \"{kn}\" = {kv:.2}"))
                    }
                }
            }
            "rigid_body" | "add_rigid_body" | "physics" | "bake_physics" | "remove_rigid_body" => {
                if c.action != "bake_physics" {
                    let ids: Vec<u64> = targets().into_iter().filter(|id| self.scene.get(*id).is_some_and(|o| o.is_mesh())).collect();
                    if ids.is_empty() {
                        return Err("select the objects that should fall (meshes)".into());
                    }
                    let mode = c.name.clone().unwrap_or_else(|| if c.action == "remove_rigid_body" { "none".into() } else { "active".into() }).to_lowercase();
                    for id in &ids {
                        let o = self.scene.get_mut(*id).unwrap();
                        // back to where it started before changing anything
                        if let Some(t) = o.rigid.as_ref().and_then(|r| r.cache.first().copied().or(r.origin)) {
                            o.transform = t;
                        }
                        if mode.contains("none") || mode.contains("remove") || mode.contains("off") {
                            o.rigid = None;
                        } else {
                            let mut r = o.rigid.clone().unwrap_or_default();
                            r.passive = mode.contains("passive") || mode.contains("static") || mode.contains("ground");
                            if let Some(b) = c.number {
                                r.bounce = (b as f32).clamp(0.0, 1.0);
                            }
                            r.cache = Default::default();
                            o.rigid = Some(r);
                        }
                    }
                }
                let msg = self.bake_rigid_bodies();
                Ok(msg)
            }
            "particles" | "emit" | "add_particles" | "particle_system" | "bake_particles" => {
                // an existing particle system: re-bake with changed settings
                let existing = c.target.as_deref().and_then(|t| self.scene.find_by_name(t)).filter(|id| self.scene.get(*id).is_some_and(|o| o.particles.is_some()));
                let pid = match existing {
                    Some(id) => id,
                    None => {
                        let spec = c.name.clone().or(c.primitive.clone()).unwrap_or_else(|| "sparks".into());
                        // weather comes from the sky unless an emitter is named
                        let sky = ["snow", "rain", "confetti"].iter().any(|k| spec.to_lowercase().contains(k));
                        let emitter = c
                            .target
                            .as_deref()
                            .and_then(|t| self.scene.find_by_name(t))
                            .or(if sky { None } else { self.active })
                            .filter(|id| self.scene.get(*id).is_some_and(|o| o.is_mesh() && o.particles.is_none()));
                        self.add_particle_system(&spec, emitter, c.object.as_deref())?
                    }
                };
                {
                    let ps = self.scene.get_mut(pid).unwrap().particles.as_mut().unwrap();
                    if let Some(n) = c.number {
                        ps.rate = (n as f32).clamp(0.0, 2000.0);
                    }
                    if let Some(v) = c.power {
                        ps.speed = v as f32;
                    }
                    if let Some(s) = vec3_or_scalar(&c.scale) {
                        ps.size = s.x.max(0.001);
                    }
                    if let Some(sd) = c.seed {
                        ps.seed = sd as u32;
                    }
                }
                self.bake_particle_system(pid)
            }
            "cloth" | "add_cloth" | "simulate_cloth" | "bake_cloth" | "remove_cloth" => {
                let id = targets().into_iter().find(|id| self.scene.get(*id).is_some_and(|o| o.is_mesh())).ok_or("cloth needs a mesh object (a plane works well)")?;
                if c.action == "remove_cloth" || c.name.as_deref() == Some("none") && c.action != "cloth" {
                    let o = self.scene.get_mut(id).unwrap();
                    o.cloth = None;
                    o.posed = None;
                    return Ok("Removed the cloth simulation".into());
                }
                let o = self.scene.get(id).unwrap();
                if o.skin.is_some() || !o.shape_keys.is_empty() {
                    return Err("cloth can't run on rigged meshes or meshes with shape keys".into());
                }
                let mut cloth = o.cloth.clone().unwrap_or_default();
                // enough vertices to fold
                if o.mesh.as_ref().is_none_or(|m| m.verts.len() < 300) && c.action != "bake_cloth" {
                    let mut pm = o.poly();
                    crate::sculpt::densify(&mut pm, 900, false);
                    pm.uvs.clear();
                    self.scene.get_mut(id).unwrap().mesh = Some(pm);
                }
                let o = self.scene.get(id).unwrap();
                let pm = o.mesh.clone().unwrap();
                let om = o.transform.matrix();
                if let Some(how) = c.name.as_deref() {
                    cloth.pins = if how.contains("select") {
                        self.edit.as_ref().filter(|s| s.obj == id).map(|s| s.sel.iter().enumerate().filter(|(_, x)| **x).map(|(i, _)| i as u32).collect()).unwrap_or_default()
                    } else {
                        crate::physics::pick_pins(&pm, om, how)
                    };
                }
                if let Some(k) = c.number {
                    cloth.stiffness = (k as f32).clamp(0.05, 1.0);
                }
                let colliders: Vec<crate::physics::Collider> = self
                    .scene
                    .objects
                    .iter()
                    .filter(|x| x.id != id && x.visible && x.is_mesh() && x.scatter.is_none())
                    .map(|x| {
                        let p = x.posed.as_ref().map(|p| (**p).clone()).unwrap_or_else(|| x.poly());
                        let lo = p.verts.iter().fold(Vec3::splat(f32::MAX), |a, b| a.min(*b));
                        let hi = p.verts.iter().fold(Vec3::splat(f32::MIN), |a, b| a.max(*b));
                        crate::physics::collider_for(x, (lo, hi))
                    })
                    .collect();
                let (f0, f1) = (self.scene.frame_start, self.scene.frame_end.max(self.scene.frame_start + 1));
                let frames = ((f1 - f0 + 1) as usize).min(1000);
                cloth.start = f0;
                cloth.cache = std::sync::Arc::new(crate::physics::bake(&pm, om, &cloth, &colliders, frames, self.scene.fps.max(1) as f32));
                let npins = cloth.pins.len();
                let o = self.scene.get_mut(id).unwrap();
                o.cloth = Some(cloth);
                o.posed_key = 0;
                let n = o.name.clone();
                self.set_frame(self.frame);
                Ok(format!("Baked cloth on \"{n}\": {frames} frames, {} colliders, {npins} pinned vertices — press play", colliders.len()))
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
            "scatter" | "scatter_on" | "distribute" => {
                // surface: target (or the active mesh)
                let surf = c
                    .target
                    .as_deref()
                    .and_then(|t| self.scene.find_by_name(t))
                    .or(self.active)
                    .filter(|id| self.scene.get(*id).is_some_and(|o| o.is_mesh()))
                    .ok_or("scatter needs target = the surface mesh")?;
                let spec = c.object.clone().or(c.name.clone()).or(c.primitive.clone()).unwrap_or_else(|| "rock".into());
                let (src, preset) = match self.scene.find_by_name(&spec).filter(|id| *id != surf && self.scene.get(*id).is_some_and(|o| o.is_mesh() && o.scatter.is_none())) {
                    Some(id) => (id, None),
                    None => {
                        let (id, p) = self.scatter_preset(&spec)?;
                        (id, Some(p))
                    }
                };
                // reuse an existing scatter of the same object on the same surface
                let existing = self.scene.objects.iter().find(|o| o.scatter.as_ref().is_some_and(|sc| sc.source == src && sc.target == surf)).map(|o| o.id);
                let mut sc = existing
                    .and_then(|id| self.scene.get(id)?.scatter.clone())
                    .unwrap_or_else(|| {
                        let mut s = crate::scatter::Scatter::new(src, surf, preset.map(|p| p.count).unwrap_or(50));
                        if let Some(p) = preset {
                            (s.scale_min, s.scale_max, s.tilt) = (p.scale.0, p.scale.1, p.tilt);
                        }
                        s
                    });
                if let Some(n) = c.number {
                    sc.count = (n as u32).clamp(1, 20_000);
                }
                if let Some(v) = c.seed.or(c.segments) {
                    sc.seed = v as u32;
                }
                if let Some(v) = c.slope {
                    sc.max_slope = v as f32;
                }
                if let Some(a) = c.align {
                    sc.align = a;
                }
                match &c.scale {
                    Some(Value::Array(a)) if a.len() == 2 => {
                        let g = |i: usize| a[i].as_f64().unwrap_or(1.0) as f32;
                        (sc.scale_min, sc.scale_max) = (g(0).min(g(1)), g(0).max(g(1)));
                    }
                    Some(Value::Number(n)) => {
                        let k = n.as_f64().unwrap_or(1.0) as f32;
                        (sc.scale_min, sc.scale_max) = (k * 0.8, k * 1.2);
                    }
                    _ => {}
                }
                if let Some(r) = vec3(&c.rotation) {
                    sc.tilt = r.x.abs().max(r.y.abs());
                    if r.z != 0.0 {
                        sc.spin = r.z.abs();
                    }
                }
                let id = match existing {
                    Some(id) => id,
                    None => {
                        let sname = format!("{} Scatter", self.scene.get(src).unwrap().name);
                        let id = self.scene.add(crate::mesh::Primitive::Cube, Some(&sname));
                        let (mat, smooth, auto, coll) = {
                            let so = self.scene.get(src).unwrap();
                            (so.material, so.smooth_override.or(Some(so.smooth())), so.auto_smooth, self.scene.get(surf).unwrap().collection)
                        };
                        let o = self.scene.get_mut(id).unwrap();
                        o.material = mat;
                        o.smooth_override = smooth;
                        o.auto_smooth = auto;
                        o.collection = coll;
                        // the original stays as the editable template, hidden
                        self.scene.get_mut(src).unwrap().visible = false;
                        id
                    }
                };
                let n = sc.count;
                self.scene.get_mut(id).unwrap().scatter = Some(sc);
                self.scene.get_mut(id).unwrap().posed = None;
                self.scene.update_scatter();
                let names = (self.scene.get(src).unwrap().name.clone(), self.scene.get(surf).unwrap().name.clone());
                Ok(format!("Scattered {n} × \"{}\" on \"{}\"{}", names.0, names.1, if existing.is_some() { " (updated)" } else { "" }))
            }
            "realize" | "apply_scatter" | "make_real" => {
                let ids: Vec<u64> = targets().into_iter().filter(|id| self.scene.get(*id).is_some_and(|o| o.scatter.is_some())).collect();
                if ids.is_empty() {
                    return Err("select a scatter object".into());
                }
                for id in &ids {
                    let o = self.scene.get_mut(*id).unwrap();
                    if let Some(p) = o.posed.take() {
                        o.mesh = Some((*p).clone());
                    }
                    o.scatter = None;
                }
                Ok(format!("Made {} a real mesh", self.names(&ids)))
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
            "render_animation" | "render_video" | "render_frames" => {
                match c.name.as_deref().map(|n| n.to_lowercase()) {
                    Some(n) if n.contains("path") || n.contains("cycles") || n.contains("quality") => self.scene.output.engine = crate::scene::Engine::PathTraced,
                    Some(n) if n.contains("fast") || n.contains("preview") => self.scene.output.engine = crate::scene::Engine::Fast,
                    _ => {}
                }
                if let Some(s) = c.number.filter(|s| *s >= 1.0) {
                    self.scene.output.samples = s as u32;
                }
                self.anim_render_request = Some(c.object.clone().unwrap_or_default());
                Ok(format!("Rendering frames {}–{} ({})", self.scene.frame_start, self.scene.frame_end, if self.scene.output.engine == crate::scene::Engine::PathTraced { "path traced" } else { "fast" }))
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
    /// knife cut along these points (in path order)
    Knife(Vec<crate::editmesh::KnifePoint>),
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
        self.weight_paint = None;
        self.edit = Some(EditState {
            obj: id,
            mode: SelectMode::Vertex,
            sel: vec![true; n],
        });
        Ok(format!("Edit mode: \"{name}\""))
    }

    pub fn exit_edit(&mut self) {
        self.edit = None;
        self.finish_shape_edit();
        self.bone_edit = None;
        self.sculpt = None;
        self.pose = None;
        self.weight_paint = None;
    }

    /// Enter / leave Weight Paint on the active rigged mesh.
    pub fn toggle_weight_paint(&mut self) -> Result<String, String> {
        if self.weight_paint.take().is_some() {
            return Ok("Object mode".into());
        }
        let id = self
            .active
            .filter(|id| self.scene.get(*id).is_some_and(|o| o.skin.is_some()))
            .or_else(|| self.selection.iter().copied().find(|id| self.scene.get(*id).is_some_and(|o| o.skin.is_some())))
            .ok_or("select a rigged mesh (Object ▸ Rig, or Ctrl+P to an armature) to paint its weights")?;
        self.edit = None;
        self.sculpt = None;
        self.pose = None;
        self.bone_edit = None;
        self.select_only(id);
        self.weight_paint = Some(id);
        let o = self.scene.get(id).unwrap();
        let nb = o.skin.as_ref().and_then(|s| self.scene.get(s.armature)).and_then(|a| a.armature.as_ref()).map(|a| a.bones.len()).unwrap_or(1);
        self.active_bone = self.active_bone.min(nb.saturating_sub(1));
        Ok(format!("Weight Paint: \"{}\"", o.name))
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
        self.weight_paint = None;
        self.bone_edit = None;
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
        self.pose = None;
        self.weight_paint = None;
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
        if self.pose.take().is_some() || self.weight_paint.take().is_some() || self.bone_edit.take().is_some() {
            return Ok("Object mode".into());
        }
        if let Some(id) = self.active.filter(|id| self.scene.get(*id).is_some_and(|o| o.armature.is_some())) {
            self.sculpt = None;
            self.edit = None;
            self.select_only(id);
            self.bone_edit = Some(id);
            self.bone_part = crate::armature::BonePart::Body;
            let nb = self.scene.get(id).and_then(|o| o.armature.as_ref()).map(|a| a.bones.len()).unwrap_or(1);
            self.active_bone = self.active_bone.min(nb.saturating_sub(1));
            return Ok(format!("Edit mode (bones): \"{}\"", self.scene.get(id).unwrap().name));
        }
        if self.sculpt.is_some() {
            self.sculpt = None;
            let m = self.finish_shape_edit();
            return Ok(m.unwrap_or_else(|| "Object mode".into()));
        }
        if self.edit.is_some() {
            let m = self.shape_edit.is_some();
            self.edit = None;
            let msg = self.finish_shape_edit();
            self.exit_edit();
            return Ok(if m { msg.unwrap_or_default() } else { "Object mode".into() });
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
            EditOp::Knife(pts) => format!("Knife: split {} face(s)", em::knife(pm, &pts, sel)?),
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

#[derive(Clone, Copy)]
struct ScatterPreset {
    count: u32,
    scale: (f32, f32),
    tilt: f32,
}

enum Special {
    Light(crate::scene::LightType),
    Camera,
    Armature,
    Empty,
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
        "empty" | "target" | "null" | "locator" | "iktarget" => Special::Empty,
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
    /// Scatter: random seed, steepest slope in degrees, stand along the surface normal.
    #[serde(default, skip_serializing_if = "Option::is_none", deserialize_with = "lenient_f64")]
    pub seed: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none", deserialize_with = "lenient_f64")]
    pub slope: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub align: Option<bool>,
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
    #[test]
    fn rig_gltf_round_trip() {
        let mut ed = Editor::default();
        ed.execute(&[
            cmd(r#"{"action":"add","primitive":"cylinder","name":"Tube","location":[1,0,1.5],"scale":[0.4,0.4,1.5]}"#),
            cmd(r#"{"action":"rig","target":"Tube","number":3}"#),
        ]);
        ed.set_frame(1);
        ed.execute(&[cmd(r#"{"action":"keyframe","target":"Tube Rig"}"#)]);
        ed.set_frame(10);
        ed.execute(&[cmd(r#"{"action":"pose_bone","name":"bone 1","rotation":[50,0,0]}"#), cmd(r#"{"action":"keyframe","target":"Tube Rig"}"#)]);
        let world = |sc: &crate::scene::Scene, name: &str| -> Vec<Vec3> {
            let o = sc.get(sc.find_by_name(name).unwrap()).unwrap();
            let m = o.transform.matrix();
            o.posed.as_ref().unwrap().verts.iter().map(|v| m.transform_point3(*v)).collect()
        };
        let bbox = |v: &[Vec3]| v.iter().fold((Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)), |(l, h), p| (l.min(*p), h.max(*p)));
        let (glb, _) = crate::gltf::export_glb(&ed.scene, &mut crate::scene::MeshCache::default());
        let mut sc = crate::scene::Scene::default();
        sc.objects.clear();
        let (st, _) = crate::gltf_import::import_bytes(&mut sc, &glb, None).unwrap();
        assert!(st.animation.is_some(), "{}", st.summary());
        for f in [1, 5, 10] {
            ed.set_frame(f);
            sc.evaluate(f);
            sc.update_skins();
            let (a, b) = (bbox(&world(&ed.scene, "Tube")), bbox(&world(&sc, "Tube")));
            assert!(a.0.distance(b.0) < 0.02 && a.1.distance(b.1) < 0.02, "frame {f}: {a:?} vs {b:?}");
        }
    }
    #[test]
    fn bone_edits_keep_mesh() {
        let mut ed = Editor::default();
        ed.execute(&[
            cmd(r#"{"action":"add","primitive":"cylinder","name":"Tube","location":[0,0,1.5],"scale":[0.4,0.4,1.5]}"#),
            cmd(r#"{"action":"rig","target":"Tube","number":3}"#),
        ]);
        let tube = ed.scene.find_by_name("Tube").unwrap();
        let rig = ed.scene.find_by_name("Tube Rig").unwrap();
        assert!(ed.toggle_edit().unwrap().contains("bones") && ed.bone_edit == Some(rig));
        let posed = |ed: &mut Editor| {
            ed.scene.update_derived();
            ed.scene.get(tube).unwrap().posed.as_ref().unwrap().verts.clone()
        };
        let before = posed(&mut ed);
        let orig = ed.scene.get(rig).unwrap().armature.clone().unwrap();
        let skins = ed.skins_of(rig);
        let mut a = orig.clone();
        a.set_rest_points(1, glam::Vec3::new(0.0, 0.0, 1.0), glam::Vec3::new(0.5, 0.0, 2.0));
        let n = a.extrude(2);
        let map: Vec<Option<usize>> = (0..orig.bones.len()).map(Some).collect();
        ed.apply_bone_edit(rig, a.clone(), &orig, &skins, &map);
        let after = posed(&mut ed);
        assert!(before.iter().zip(&after).all(|(p, q)| p.distance(*q) < 1e-4), "rest edits don't move the mesh");
        // deleting a bone hands its weights to the parent and keeps the skin valid
        let skins = ed.skins_of(rig);
        let mut b = a.clone();
        let map = b.delete_bone(1);
        ed.apply_bone_edit(rig, b.clone(), &a, &skins, &map);
        let after2 = posed(&mut ed);
        assert!(before.iter().zip(&after2).all(|(p, q)| p.distance(*q) < 1e-4));
        assert_eq!(ed.scene.get(rig).unwrap().armature.as_ref().unwrap().bones.len(), n);
        assert!(ed.scene.get(tube).unwrap().skin.as_ref().unwrap().joints.iter().flatten().all(|j| (*j as usize) < n));
    }
    #[test]
    fn ik_reaches_target() {
        let mut ed = Editor::default();
        let log = ed.execute(&[
            cmd(r#"{"action":"add","primitive":"armature","name":"Arm","number":3}"#),
            cmd(r#"{"action":"add_ik","target":"Arm","name":"Bone.002","number":0}"#),
        ]);
        assert!(log.iter().all(|l| l.starts_with('✔')), "{log:?}");
        let t = ed.scene.find_by_name("IK Bone.002").unwrap();
        ed.execute(&[cmd(r#"{"action":"set_location","target":"IK Bone.002","location":[1.0,0.5,1.2]}"#)]);
        let arm = ed.scene.get(ed.scene.find_by_name("Arm").unwrap()).unwrap().armature.as_ref().unwrap();
        let tip = arm.segments(&arm.pose_globals())[2].1;
        let goal = ed.scene.get(t).unwrap().transform.location;
        assert!(tip.distance(goal) < 1e-3, "tip {tip} goal {goal}");
        // out of reach: the chain points straight at it
        ed.execute(&[cmd(r#"{"action":"set_location","target":"IK Bone.002","location":[10,0,0]}"#)]);
        let arm = ed.scene.get(ed.scene.find_by_name("Arm").unwrap()).unwrap().armature.as_ref().unwrap();
        let tip = arm.segments(&arm.pose_globals())[2].1;
        assert!((tip - Vec3::new(2.0, 0.0, 0.0)).length() < 0.05, "{tip}");
    }

    #[test]
    fn scatter_command() {
        let mut ed = Editor::default();
        let log = ed.execute(&[
            cmd(r#"{"action":"add","primitive":"plane","name":"Ground","scale":5}"#),
            cmd(r#"{"action":"scatter","target":"Ground","name":"rocks","number":25,"seed":3}"#),
        ]);
        assert!(log.iter().all(|l| l.starts_with('✔')), "{log:?}");
        let sid = ed.scene.find_by_name("Rock Scatter").unwrap();
        let o = ed.scene.get(sid).unwrap();
        let rock = ed.scene.find_by_name("Rock").unwrap();
        assert!(!ed.scene.get(rock).unwrap().visible);
        let n = ed.scene.get(rock).unwrap().mesh.as_ref().unwrap().verts.len();
        assert_eq!(o.posed.as_ref().unwrap().verts.len(), n * 25);
        // the same command updates it; moving the ground carries the rocks
        ed.execute(&[cmd(r#"{"action":"scatter","target":"Ground","name":"Rock","number":40}"#)]);
        assert_eq!(ed.scene.objects.iter().filter(|o| o.scatter.is_some()).count(), 1);
        ed.execute(&[cmd(r#"{"action":"move","target":"Ground","vector":[0,0,2]}"#)]);
        let o = ed.scene.get(sid).unwrap();
        let zmin = o.posed.as_ref().unwrap().verts.iter().map(|v| v.z).fold(f32::MAX, f32::min);
        assert_eq!(o.posed.as_ref().unwrap().verts.len(), n * 40);
        assert!(zmin > 1.5, "rocks follow the ground: {zmin}");
        ed.execute(&[cmd(r#"{"action":"realize","target":"Rock Scatter"}"#)]);
        assert!(ed.scene.get(sid).unwrap().mesh.is_some() && ed.scene.get(sid).unwrap().scatter.is_none());
    }
    #[test]
    fn shape_keys_edit_animate_export() {
        let mut ed = Editor::default();
        let log = ed.execute(&[
            cmd(r#"{"action":"add","primitive":"cube","name":"Blob","location":[0,0,1]}"#),
            cmd(r#"{"action":"add_shape_key","target":"Blob","name":"Squash","scale":[1.4,1.4,0.5]}"#),
        ]);
        assert!(log.iter().all(|l| l.starts_with('✔')), "{log:?}");
        let id = ed.scene.find_by_name("Blob").unwrap();
        let top = |ed: &mut Editor| {
            ed.scene.update_derived();
            let o = ed.scene.get(id).unwrap();
            o.posed.as_ref().map(|p| p.verts.clone()).unwrap_or_else(|| o.mesh.clone().unwrap().verts).iter().map(|v| v.z).fold(f32::MIN, f32::max)
        };
        assert!((top(&mut ed) - 0.5).abs() < 1e-4, "fully squashed");
        ed.execute(&[cmd(r#"{"action":"set_shape_key","target":"Blob","name":"squash","number":0}"#)]);
        assert!((top(&mut ed) - 1.0).abs() < 1e-4);
        // edit a key: the mesh shows it, leaving restores the basis
        let k = ed.add_shape_key(id, "Tall").unwrap();
        ed.begin_shape_edit(id, k).unwrap();
        for v in &mut ed.scene.get_mut(id).unwrap().mesh.as_mut().unwrap().verts {
            v.z *= 3.0;
        }
        ed.finish_shape_edit().unwrap();
        assert!((ed.scene.get(id).unwrap().mesh.as_ref().unwrap().verts.iter().map(|v| v.z).fold(f32::MIN, f32::max) - 1.0).abs() < 1e-5, "basis back");
        assert!((top(&mut ed) - 3.0).abs() < 1e-4, "the new key is on (value 1)");
        // animate and round-trip through glTF
        ed.execute(&[cmd(r#"{"action":"set_shape_key","target":"Blob","name":"Tall","number":0}"#)]);
        ed.set_frame(1);
        ed.execute(&[cmd(r#"{"action":"keyframe","target":"Blob"}"#)]);
        ed.set_frame(10);
        ed.execute(&[cmd(r#"{"action":"set_shape_key","target":"Blob","name":"Tall","number":1}"#), cmd(r#"{"action":"keyframe","target":"Blob"}"#)]);
        let (glb, _) = crate::gltf::export_glb(&ed.scene, &mut crate::scene::MeshCache::default());
        let mut sc = crate::scene::Scene::default();
        sc.objects.clear();
        crate::gltf_import::import_bytes(&mut sc, &glb, None).unwrap();
        let o = sc.objects.iter().find(|o| o.name == "Blob").unwrap();
        assert_eq!(o.shape_keys.iter().map(|k| k.name.as_str()).collect::<Vec<_>>(), ["Squash", "Tall"]);
        sc.evaluate(10);
        sc.update_derived();
        let o = sc.objects.iter().find(|o| o.name == "Blob").unwrap();
        let m = o.transform.matrix();
        let zt = o.posed.as_ref().unwrap().verts.iter().map(|v| m.transform_point3(*v).z).fold(f32::MIN, f32::max);
        assert!((zt - 4.0).abs() < 1e-3, "frame 10 is tall: {zt}");
    }
    #[test]
    fn cloth_command_bakes() {
        let mut ed = Editor::default();
        let log = ed.execute(&[
            cmd(r#"{"action":"delete","target":"Cube"}"#),
            cmd(r#"{"action":"add","primitive":"uv_sphere","name":"Ball","location":[0,0,0]}"#),
            cmd(r#"{"action":"add","primitive":"plane","name":"Sheet","location":[0,0,1.5],"scale":1.5}"#),
            cmd(r#"{"action":"cloth","target":"Sheet"}"#),
        ]);
        assert!(log.iter().all(|l| l.starts_with('✔')), "{log:?}");
        let id = ed.scene.find_by_name("Sheet").unwrap();
        let low = |ed: &mut Editor, f: i32| {
            ed.set_frame(f);
            let o = ed.scene.get(id).unwrap();
            let m = o.transform.matrix();
            o.posed.as_ref().map(|p| p.verts.iter().map(|v| m.transform_point3(*v).z).fold(f32::MAX, f32::min)).unwrap_or(9.0)
        };
        let (a, b) = (low(&mut ed, 1), low(&mut ed, 60));
        assert!((a - 1.5).abs() < 1e-3 && b < 0.5, "falls and drapes: {a} → {b}");
    }
    #[test]
    fn rigid_body_command() {
        let mut ed = Editor::default();
        let log = ed.execute(&[
            cmd(r#"{"action":"delete","target":"Cube"}"#),
            cmd(r#"{"action":"add","primitive":"plane","name":"Ground","scale":6}"#),
            cmd(r#"{"action":"add","primitive":"cube","name":"Crate","location":[0,0,3],"rotation":[30,20,0],"scale":0.5}"#),
            cmd(r#"{"action":"rigid_body","target":"Crate"}"#),
        ]);
        assert!(log.iter().all(|l| l.starts_with('✔')), "{log:?}");
        let id = ed.scene.find_by_name("Crate").unwrap();
        ed.set_frame(1);
        assert!((ed.scene.get(id).unwrap().transform.location.z - 3.0).abs() < 1e-4);
        ed.set_frame(100);
        let z = ed.scene.get(id).unwrap().transform.location.z;
        assert!(z > 0.3 && z < 0.8, "landed on the ground: {z}");
        // saved mid-animation, reopened: re-simulated from the start position
        let json = serde_json::to_string(&ed.scene).unwrap();
        let mut ed2 = Editor::default();
        ed2.scene = serde_json::from_str(&json).unwrap();
        ed2.rebake_physics();
        ed2.set_frame(1);
        assert!((ed2.scene.get(id).unwrap().transform.location.z - 3.0).abs() < 1e-4);
        ed2.set_frame(100);
        assert!((ed2.scene.get(id).unwrap().transform.location.z - z).abs() < 1e-3);
        // turning it off puts it back
        ed.execute(&[cmd(r#"{"action":"rigid_body","target":"Crate","name":"none"}"#)]);
        assert!((ed.scene.get(id).unwrap().transform.location.z - 3.0).abs() < 1e-4);
    }
    #[test]
    fn particle_presets() {
        let mut ed = Editor::default();
        let log = ed.execute(&[
            cmd(r#"{"action":"delete","target":"Cube"}"#),
            cmd(r#"{"action":"add","primitive":"plane","name":"Ground","scale":6}"#),
            cmd(r#"{"action":"particles","name":"snow"}"#),
            cmd(r#"{"action":"add","primitive":"uv_sphere","name":"Orb","location":[0,0,2]}"#),
            cmd(r#"{"action":"particles","target":"Orb","name":"sparks","number":60}"#),
        ]);
        assert!(log.iter().all(|l| l.starts_with('✔')), "{log:?}");
        let snow = ed.scene.find_by_name("snow Particles").unwrap();
        let sparks = ed.scene.find_by_name("sparks Particles").unwrap();
        ed.set_frame(60);
        let count = |ed: &Editor, id| ed.scene.get(id).unwrap().posed.as_ref().map(|p| p.verts.len()).unwrap_or(0);
        assert!(count(&ed, snow) > 0 && count(&ed, sparks) > 0);
        // sparks never sink into the ground
        let o = ed.scene.get(sparks).unwrap();
        let zmin = o.posed.as_ref().unwrap().verts.iter().filter(|v| v.x.abs() < 5.5 && v.y.abs() < 5.5).map(|v| v.z).fold(f32::MAX, f32::min);
        assert!(zmin > -0.1, "{zmin}");
        assert_eq!(ed.scene.get(sparks).unwrap().particles.as_ref().unwrap().rate, 60.0);
    }
}
