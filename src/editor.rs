//! Editor state (scene + selection + undo + playback) and the command executor
//! shared by the UI, the AI assistant and voice control.

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
        }
    }
}

impl Editor {
    pub fn snapshot(&mut self) {
        self.undo_stack.push((self.scene.clone(), self.selection.clone()));
        if self.undo_stack.len() > 200 {
            self.undo_stack.remove(0);
        }
        self.redo_stack.clear();
        self.dirty = true;
    }
    pub fn undo(&mut self) -> bool {
        if let Some((s, sel)) = self.undo_stack.pop() {
            self.redo_stack.push((std::mem::replace(&mut self.scene, s), std::mem::replace(&mut self.selection, sel)));
            self.fix_active();
            true
        } else {
            false
        }
    }
    pub fn redo(&mut self) -> bool {
        if let Some((s, sel)) = self.redo_stack.pop() {
            self.undo_stack.push((std::mem::replace(&mut self.scene, s), std::mem::replace(&mut self.selection, sel)));
            self.fix_active();
            true
        } else {
            false
        }
    }
    fn fix_active(&mut self) {
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
            "" | "selected" | "selection" | "it" | "this" | "that" | "them" | "current" | "active" => {
                if self.selection.is_empty() {
                    self.scene.objects.last().map(|o| vec![o.id]).unwrap_or_default()
                } else {
                    self.selection.clone()
                }
            }
            "all" | "everything" | "scene" | "all objects" => self.scene.objects.iter().map(|o| o.id).collect(),
            "last" | "newest" | "latest" => self.scene.objects.last().map(|o| vec![o.id]).unwrap_or_default(),
            _ => {
                if let Some(id) = self.scene.objects.iter().find(|o| o.name.to_lowercase() == tl).map(|o| o.id) {
                    return vec![id];
                }
                // "all spheres" / "cubes" -> by primitive
                let stripped = tl.trim_start_matches("all ").trim_start_matches("the ");
                let singular = stripped.strip_suffix("es").filter(|s| Primitive::parse(s).is_some()).or_else(|| stripped.strip_suffix('s')).unwrap_or(stripped);
                if let Some(p) = Primitive::parse(singular) {
                    let ids: Vec<u64> = self.scene.objects.iter().filter(|o| o.primitive == p).map(|o| o.id).collect();
                    if !ids.is_empty() {
                        return ids;
                    }
                }
                // color match: "the red one"
                if let Some(c) = parse_color_str(stripped.trim_end_matches(" one").trim_end_matches(" ones").split_whitespace().next().unwrap_or("")) {
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
        let mut log = vec![];
        let mutating = cmds.iter().any(|c| !matches!(c.action.as_str(), "undo" | "redo" | "view" | "play" | "stop" | "select" | "deselect_all" | "focus" | "shading" | "set_frame" | "chat" | "none"));
        if mutating {
            self.snapshot();
        }
        for c in cmds {
            match self.exec_one(c) {
                Ok(msg) => log.push(format!("✔ {msg}")),
                Err(e) => log.push(format!("✖ {}: {e}", c.action)),
            }
        }
        log
    }

    fn exec_one(&mut self, c: &Command) -> Result<String, String> {
        let action = c.action.to_lowercase().replace([' ', '-'], "_");
        let target_ids = self.resolve_target(c.target.as_deref());
        let targets = || target_ids.clone();
        match action.as_str() {
            "add" | "add_object" | "create" => {
                let p = c
                    .primitive
                    .as_deref()
                    .and_then(Primitive::parse)
                    .ok_or_else(|| format!("unknown primitive {:?}", c.primitive))?;
                let id = self.scene.add(p, c.name.as_deref());
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
                if let Some(d) = c.number {
                    if (1.0..=6.0).contains(&d) && c.modifier.is_none() {
                        o.detail = d as u32;
                    }
                }
                apply_material_fields(o, c);
                let name = o.name.clone();
                if let Some(m) = c.modifier.as_deref() {
                    let m = make_modifier(m, c.number, vec3(&c.vector))?;
                    self.scene.get_mut(id).unwrap().modifiers.push(m);
                }
                self.select_only(id);
                Ok(format!("Added {} \"{}\"", p.label(), name))
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
                let d = vec3(&c.vector).or_else(|| vec3(&c.location)).ok_or("move needs a vector")?;
                let ids = targets();
                for id in &ids {
                    if let Some(o) = self.scene.get_mut(*id) {
                        o.transform.location += d;
                    }
                }
                Ok(format!("Moved {} by [{:.2}, {:.2}, {:.2}]", self.names(&ids), d.x, d.y, d.z))
            }
            "rotate" => {
                let d = vec3(&c.vector).or_else(|| vec3(&c.rotation)).or_else(|| c.number.map(|n| Vec3::new(0.0, 0.0, n as f32))).ok_or("rotate needs degrees")?;
                let ids = targets();
                for id in &ids {
                    if let Some(o) = self.scene.get_mut(*id) {
                        o.transform.rotation += d;
                    }
                }
                Ok(format!("Rotated {} by [{:.0}°, {:.0}°, {:.0}°]", self.names(&ids), d.x, d.y, d.z))
            }
            "scale" | "resize" => {
                let f = vec3_or_scalar(&c.vector).or_else(|| vec3_or_scalar(&c.scale)).or_else(|| c.number.map(|n| Vec3::splat(n as f32))).ok_or("scale needs a factor")?;
                let ids = targets();
                for id in &ids {
                    if let Some(o) = self.scene.get_mut(*id) {
                        o.transform.scale *= f;
                    }
                }
                Ok(format!("Scaled {} by [{:.2}, {:.2}, {:.2}]", self.names(&ids), f.x, f.y, f.z))
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
                    }
                }
                Ok(format!("Changed material of {}", self.names(&ids)))
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
                let ids = if c.target.is_none() { self.scene.objects.iter().map(|o| o.id).collect() } else { targets() };
                for id in &ids {
                    if let Some(o) = self.scene.get_mut(*id) {
                        o.visible = true;
                    }
                }
                Ok(format!("Revealed {}", self.names(&ids)))
            }
            "add_modifier" | "modifier" => {
                let ids = targets();
                let m = make_modifier(c.modifier.as_deref().unwrap_or("array"), c.number, vec3(&c.vector))?;
                for id in &ids {
                    if let Some(o) = self.scene.get_mut(*id) {
                        o.modifiers.push(m.clone());
                    }
                }
                Ok(format!("Added {} modifier to {}", m.label(), self.names(&ids)))
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
                Ok(format!("Inserted keyframe at frame {f} for {}", self.names(&ids)))
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
                    if let (Some(a), Some(b)) = (v.first().and_then(|x| x.as_f64()), v.get(1).and_then(|x| x.as_f64())) {
                        self.scene.frame_start = a as i32;
                        self.scene.frame_end = b as i32;
                    }
                } else if let Some(n) = c.number {
                    self.scene.frame_end = n as i32;
                }
                Ok(format!("Frame range {}–{}", self.scene.frame_start, self.scene.frame_end))
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
                let v = c.view.as_deref().or(c.target.as_deref()).unwrap_or("iso").to_lowercase();
                let req = match v.as_str() {
                    "front" => ViewRequest::Preset(ViewPreset::Front),
                    "back" => ViewRequest::Preset(ViewPreset::Back),
                    "left" => ViewRequest::Preset(ViewPreset::Left),
                    "right" | "side" => ViewRequest::Preset(ViewPreset::Right),
                    "top" => ViewRequest::Preset(ViewPreset::Top),
                    "bottom" => ViewRequest::Preset(ViewPreset::Bottom),
                    "ortho" | "orthographic" | "perspective" | "persp" | "toggle_ortho" => ViewRequest::ToggleOrtho,
                    "selected" | "focus" | "frame_selected" => ViewRequest::FrameSelected,
                    "all" | "frame_all" | "home" => ViewRequest::FrameAll,
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
                let s = c.view.as_deref().or(c.name.as_deref()).or(c.target.as_deref()).unwrap_or("solid").to_lowercase();
                self.shading = if s.contains("wire") {
                    Shading::Wireframe
                } else if s.contains("mat") || s.contains("render") || s.contains("preview") {
                    Shading::Material
                } else {
                    Shading::Solid
                };
                Ok(format!("Shading: {:?}", self.shading))
            }
            "smooth" | "shade_smooth" | "flat" | "shade_flat" => {
                let ids = targets();
                let smooth = action.contains("smooth");
                for id in &ids {
                    if let Some(o) = self.scene.get_mut(*id) {
                        o.smooth_override = Some(smooth);
                    }
                }
                Ok(format!("Shade {} on {}", if smooth { "smooth" } else { "flat" }, self.names(&ids)))
            }
            "set_detail" | "subdivide" => {
                let ids = targets();
                for id in &ids {
                    if let Some(o) = self.scene.get_mut(*id) {
                        o.detail = c.number.map(|n| n as u32).unwrap_or(o.detail + 1).clamp(1, 6);
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
        let n: Vec<String> = ids.iter().filter_map(|id| self.scene.get(*id)).map(|o| format!("\"{}\"", o.name)).collect();
        match n.len() {
            0 => "nothing".into(),
            1..=4 => n.join(", "),
            k => format!("{} and {} more", n[..3].join(", "), k - 3),
        }
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

fn make_modifier(kind: &str, number: Option<f64>, vector: Option<Vec3>) -> Result<Modifier, String> {
    let k = kind.to_lowercase();
    Ok(if k.contains("mirror") {
        let v = vector.unwrap_or(Vec3::X);
        Modifier::Mirror { x: v.x != 0.0, y: v.y != 0.0, z: v.z != 0.0 }
    } else if k.contains("radial") || k.contains("circle") || k.contains("ring") {
        Modifier::Radial { count: number.unwrap_or(8.0).clamp(1.0, 200.0) as u32, radius: vector.map(|v| v.length()).filter(|r| *r > 0.0).unwrap_or(3.0) }
    } else if k.contains("array") || k.contains("repeat") || k.contains("row") {
        Modifier::Array { count: number.unwrap_or(3.0).clamp(1.0, 200.0) as u32, offset: vector.unwrap_or(Vec3::new(2.2, 0.0, 0.0)) }
    } else {
        return Err(format!("unknown modifier '{kind}' (array, mirror, radial)"));
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
    #[serde(default, skip_serializing_if = "Option::is_none", deserialize_with = "lenient_f64")]
    pub number: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none", deserialize_with = "lenient_f64")]
    pub metallic: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none", deserialize_with = "lenient_f64")]
    pub roughness: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none", deserialize_with = "lenient_f64")]
    pub emission: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none", deserialize_with = "lenient_f64")]
    pub alpha: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub modifier: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub view: Option<String>,
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
        Self { action: action.into(), ..Default::default() }
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
            let p: Vec<f32> = s.split(|c: char| c == ',' || c.is_whitespace()).filter_map(|x| x.trim_matches(['[', ']', '(', ')']).parse().ok()).collect();
            if p.is_empty() { None } else { Some(Vec3::new(p[0], *p.get(1).unwrap_or(&0.0), *p.get(2).unwrap_or(&0.0))) }
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
            let p = |i: usize| u8::from_str_radix(&h[i..i + 2], 16).ok().map(|v| v as f32 / 255.0);
            return Some([p(0)?, p(2)?, p(4)?]);
        }
    }
    let (light, dark) = (s.starts_with("light "), s.starts_with("dark "));
    let base = s.trim_start_matches("light ").trim_start_matches("dark ").trim();
    let c = NAMED_COLORS.iter().find(|(n, _)| *n == base).map(|(_, c)| *c)?;
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
        assert_eq!(e.scene.objects.len(), 1);
    }
    #[test]
    fn targets_by_kind_and_color() {
        let mut e = Editor::default();
        e.execute(&[
            cmd(r#"{"action":"add","primitive":"sphere","color":"blue"}"#),
            cmd(r#"{"action":"add","primitive":"sphere","color":"red"}"#),
        ]);
        assert_eq!(e.resolve_target(Some("spheres")).len(), 2);
        assert_eq!(e.resolve_target(Some("all")).len(), 3);
        assert_eq!(e.resolve_target(Some("the red one")).len(), 1);
    }
    #[test]
    fn lenient_values() {
        let c = cmd(r#"{"action":"add","primitive":"cube","location":"1, 2, 3","number":"2","color":[255,0,0]}"#);
        assert_eq!(vec3(&c.location), Some(Vec3::new(1.0, 2.0, 3.0)));
        assert_eq!(c.number, Some(2.0));
        assert_eq!(parse_color(c.color.as_ref().unwrap()), Some([1.0, 0.0, 0.0]));
    }
    #[test]
    fn duplicate_count() {
        let mut e = Editor::default();
        e.execute(&[cmd(r#"{"action":"duplicate","number":3,"vector":[0,3,0]}"#)]);
        assert_eq!(e.scene.objects.len(), 4);
        assert_eq!(e.undo_stack.len(), 1);
    }
}
