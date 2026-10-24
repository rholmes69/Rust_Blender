//! Scene graph: objects, transforms, materials, modifiers, keyframes.

use crate::mesh::{Mesh, Primitive};
use glam::{EulerRot, Mat4, Quat, Vec3};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Transform {
    pub location: Vec3,
    /// Euler XYZ rotation in degrees (Blender convention).
    pub rotation: Vec3,
    pub scale: Vec3,
}

impl Default for Transform {
    fn default() -> Self {
        Self {
            location: Vec3::ZERO,
            rotation: Vec3::ZERO,
            scale: Vec3::ONE,
        }
    }
}

impl Transform {
    /// Decompose a matrix (shear is lost).
    pub fn from_matrix(m: Mat4) -> Transform {
        let (s, r, t) = m.to_scale_rotation_translation();
        let mut tr = Transform { location: t, rotation: Vec3::ZERO, scale: s };
        tr.set_quat(r);
        tr
    }
    pub fn quat(&self) -> Quat {
        let r = self.rotation * (std::f32::consts::PI / 180.0);
        Quat::from_euler(EulerRot::ZYX, r.z, r.y, r.x)
    }
    pub fn matrix(&self) -> Mat4 {
        Mat4::from_scale_rotation_translation(self.scale, self.quat(), self.location)
    }
    pub fn set_quat(&mut self, q: Quat) {
        let (z, y, x) = q.to_euler(EulerRot::ZYX);
        self.rotation = Vec3::new(x, y, z) * (180.0 / std::f32::consts::PI);
    }
    pub fn lerp(&self, o: &Transform, t: f32) -> Transform {
        Transform {
            location: self.location.lerp(o.location, t),
            rotation: self.rotation.lerp(o.rotation, t),
            scale: self.scale.lerp(o.scale, t),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Material {
    pub color: [f32; 3],
    pub roughness: f32,
    pub metallic: f32,
    pub emission: f32,
    pub alpha: f32,
    /// Base-colour texture (`Scene::textures` id; 0 = none), multiplied with `color`.
    #[serde(default)]
    pub texture: u32,
    /// UV repeats of the texture.
    #[serde(default = "one")]
    pub tex_scale: f32,
    /// Tangent-space normal map (texture id; 0 = none), same UVs and repeats as `texture`.
    #[serde(default)]
    pub normal_tex: u32,
    /// Bump strength of the normal map.
    #[serde(default = "one")]
    pub normal_strength: f32,
    /// Roughness / metallic map (glTF layout: G = roughness, B = metallic; a grey image works
    /// as a roughness map), multiplied with `roughness` and `metallic`.
    #[serde(default)]
    pub rough_tex: u32,
}

fn one() -> f32 {
    1.0
}

impl Default for Material {
    fn default() -> Self {
        Self {
            color: [0.8, 0.8, 0.8],
            roughness: 0.5,
            metallic: 0.0,
            emission: 0.0,
            alpha: 1.0,
            texture: 0,
            tex_scale: 1.0,
            normal_tex: 0,
            normal_strength: 1.0,
            rough_tex: 0,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Modifier {
    Array { count: u32, offset: Vec3 },
    Mirror { x: bool, y: bool, z: bool },
    Radial { count: u32, radius: f32 },
    /// Catmull–Clark subdivision surface.
    Subsurf { levels: u32 },
}

impl Modifier {
    pub fn label(&self) -> &'static str {
        match self {
            Modifier::Array { .. } => "Array",
            Modifier::Mirror { .. } => "Mirror",
            Modifier::Radial { .. } => "Radial Array",
            Modifier::Subsurf { .. } => "Subdivision Surface",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Keyframe {
    pub frame: i32,
    pub transform: Transform,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LightType {
    Point,
    Sun,
    Spot,
}

impl LightType {
    pub fn label(self) -> &'static str {
        match self {
            LightType::Point => "Point",
            LightType::Sun => "Sun",
            LightType::Spot => "Spot",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Light {
    pub light_type: LightType,
    pub color: [f32; 3],
    /// Watts for point/spot lights, irradiance strength for sun.
    pub power: f32,
    /// Full cone angle in degrees (spot only).
    pub spot_angle: f32,
    /// Spot edge softness 0..1.
    pub spot_blend: f32,
}

impl Light {
    pub fn new(t: LightType) -> Self {
        Self {
            light_type: t,
            color: [1.0, 1.0, 1.0],
            power: if t == LightType::Sun { 3.0 } else { 1000.0 },
            spot_angle: 45.0,
            spot_blend: 0.15,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct CameraData {
    /// Vertical field of view in degrees.
    pub fov_deg: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Default, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ObjectKind {
    #[default]
    Mesh,
    Light(Light),
    Camera(CameraData),
    /// Bones live in `Object::armature`.
    Armature,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Object {
    pub id: u64,
    pub name: String,
    #[serde(default)]
    pub kind: ObjectKind,
    pub primitive: Primitive,
    /// Custom (edited) mesh; when `None` the procedural primitive is used.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mesh: Option<crate::mesh::PolyMesh>,
    #[serde(default = "default_detail")]
    pub detail: u32,
    pub transform: Transform,
    pub material: Material,
    #[serde(default)]
    pub modifiers: Vec<Modifier>,
    #[serde(default)]
    pub keyframes: Vec<Keyframe>,
    #[serde(default = "yes")]
    pub visible: bool,
    #[serde(default)]
    pub smooth_override: Option<bool>,
    /// Shade Auto Smooth angle in degrees (0 = off).
    #[serde(default)]
    pub auto_smooth: f32,
    /// Parent object: this object follows the parent's movements (transforms stay in world space).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent: Option<u64>,
    /// Collection id (0 = the Scene Collection itself).
    #[serde(default)]
    pub collection: u32,
    /// Bones (armature objects only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub armature: Option<crate::armature::Armature>,
    /// Skin weights: the mesh is deformed by an armature.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub skin: Option<crate::armature::Skin>,
    /// Deformed mesh for the current pose (derived, see `Scene::update_skins`).
    #[serde(skip)]
    pub posed: Option<std::sync::Arc<crate::mesh::PolyMesh>>,
    #[serde(skip)]
    pub posed_key: u64,
}

/// A named group of objects (like Blender's collections), shown as a folder in the outliner.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Collection {
    pub id: u32,
    pub name: String,
    #[serde(default = "yes")]
    pub visible: bool,
}

fn default_detail() -> u32 {
    2
}
fn yes() -> bool {
    true
}

impl Object {
    pub fn is_mesh(&self) -> bool {
        matches!(self.kind, ObjectKind::Mesh)
    }
    pub fn light(&self) -> Option<&Light> {
        if let ObjectKind::Light(l) = &self.kind {
            Some(l)
        } else {
            None
        }
    }
    /// Local -Z axis in world space (the direction cameras look and lights shine, like Blender).
    pub fn forward(&self) -> Vec3 {
        (self.transform.quat() * -Vec3::Z).normalize()
    }
    /// Rotate so the local -Z axis points at `target` (local +Y stays as close to world +Z as possible).
    pub fn look_at(&mut self, target: Vec3) {
        let f = (target - self.transform.location).normalize_or_zero();
        if f == Vec3::ZERO {
            return;
        }
        let world_up = if f.z.abs() > 0.999 { Vec3::Y } else { Vec3::Z };
        let x = f.cross(world_up).normalize();
        let y = x.cross(f);
        let m = glam::Mat3::from_cols(x, y, -f);
        self.transform.set_quat(Quat::from_mat3(&m));
    }
    /// The editable polygon mesh (custom mesh or the primitive's).
    pub fn poly(&self) -> crate::mesh::PolyMesh {
        self.mesh
            .clone()
            .unwrap_or_else(|| self.primitive.poly(self.detail))
    }
    /// Total Catmull–Clark levels from Subdivision Surface modifiers.
    pub fn subsurf_levels(&self) -> u32 {
        self.modifiers.iter().map(|m| if let Modifier::Subsurf { levels } = m { *levels } else { 0 }).sum::<u32>().min(6)
    }
    pub fn smooth(&self) -> bool {
        self.smooth_override
            .unwrap_or(self.primitive.default_smooth())
    }

    pub fn type_label(&self) -> String {
        match &self.kind {
            ObjectKind::Mesh => self.primitive.label().to_string(),
            ObjectKind::Light(l) => format!("{} Light", l.light_type.label()),
            ObjectKind::Camera(_) => "Camera".into(),
            ObjectKind::Armature => format!("Armature, {} bones", self.armature.as_ref().map(|a| a.bones.len()).unwrap_or(0)),
        }
    }

    /// Local matrices produced by the modifier stack (instances).
    pub fn instance_matrices(&self) -> Vec<Mat4> {
        let mut out = vec![Mat4::IDENTITY];
        for m in &self.modifiers {
            match m {
                Modifier::Array { count, offset } => {
                    let mut next = vec![];
                    for i in 0..(*count).clamp(1, 200) {
                        let t = Mat4::from_translation(*offset * i as f32);
                        next.extend(out.iter().map(|m| t * *m));
                    }
                    out = next;
                }
                Modifier::Mirror { x, y, z } => {
                    for (on, s) in [
                        (*x, Vec3::new(-1., 1., 1.)),
                        (*y, Vec3::new(1., -1., 1.)),
                        (*z, Vec3::new(1., 1., -1.)),
                    ] {
                        if on {
                            let mm = Mat4::from_scale(s);
                            let mut extra: Vec<Mat4> = out.iter().map(|m| mm * *m).collect();
                            out.append(&mut extra);
                        }
                    }
                }
                Modifier::Radial { count, radius } => {
                    let mut next = vec![];
                    let n = (*count).clamp(1, 200);
                    for i in 0..n {
                        let a = std::f32::consts::TAU * i as f32 / n as f32;
                        let t = Mat4::from_rotation_z(a)
                            * Mat4::from_translation(Vec3::new(*radius, 0.0, 0.0));
                        next.extend(out.iter().map(|m| t * *m));
                    }
                    out = next;
                }
                Modifier::Subsurf { .. } => {}
            }
        }
        out.truncate(2000);
        out
    }

    pub fn evaluate(&mut self, frame: i32) {
        if let Some(t) = sample_keys(&self.keyframes, frame) {
            self.transform = t;
        }
        if let Some(a) = self.armature.as_mut() {
            a.evaluate(frame as f32);
        }
    }

    pub fn insert_keyframe(&mut self, frame: i32) {
        if let Some(a) = self.armature.as_mut() {
            a.insert_keys(frame);
        }
        self.keyframes.retain(|k| k.frame != frame);
        self.keyframes.push(Keyframe {
            frame,
            transform: self.transform,
        });
        self.keyframes.sort_by_key(|k| k.frame);
    }
}

pub fn sample_keys(keys: &[Keyframe], frame: i32) -> Option<Transform> {
    let first = keys.first()?;
    if frame <= first.frame {
        return Some(first.transform);
    }
    let last = keys.last()?;
    if frame >= last.frame {
        return Some(last.transform);
    }
    for w in keys.windows(2) {
        if frame >= w[0].frame && frame <= w[1].frame {
            let span = (w[1].frame - w[0].frame).max(1) as f32;
            let t = (frame - w[0].frame) as f32 / span;
            // smoothstep easing like Blender's default Bezier interpolation
            let t = t * t * (3.0 - 2.0 * t);
            return Some(w[0].transform.lerp(&w[1].transform, t));
        }
    }
    None
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct World {
    pub background: [f32; 3],
    /// Sun direction (points *towards* the light), Z-up.
    pub sun_dir: Vec3,
    pub sun_strength: f32,
    pub ambient: f32,
}

impl Default for World {
    fn default() -> Self {
        Self {
            background: [0.224, 0.224, 0.224],
            sun_dir: Vec3::new(0.4, -0.6, 0.8).normalize(),
            sun_strength: 1.0,
            ambient: 0.28,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Scene {
    pub objects: Vec<Object>,
    pub world: World,
    pub frame_start: i32,
    pub frame_end: i32,
    pub fps: u32,
    /// Render output settings (F12).
    #[serde(default)]
    pub output: Output,
    /// Image and procedural textures used by materials.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub textures: Vec<crate::textures::Texture>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub collections: Vec<Collection>,
    next_id: u64,
}

/// Render output settings, saved with the scene like Blender's Output properties.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct Output {
    pub width: u32,
    pub height: u32,
    /// Transparent background (alpha) instead of the world colour.
    pub transparent: bool,
    /// Render engine for F12.
    pub engine: Engine,
    /// Path tracer samples per pixel.
    pub samples: u32,
    /// Denoise path traced renders.
    pub denoise: bool,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum Engine {
    /// Rasterized with ray-traced shadows and AO (fast).
    #[default]
    Fast,
    /// Progressive path tracing: bounced light, glossy reflections, glass.
    PathTraced,
}

impl Default for Output {
    fn default() -> Self {
        Self { width: 1920, height: 1080, transparent: false, engine: Engine::Fast, samples: 32, denoise: true }
    }
}

impl Default for Scene {
    fn default() -> Self {
        // Blender's default scene: cube, light and camera.
        let mut s = Self {
            objects: vec![],
            world: World::default(),
            frame_start: 1,
            frame_end: 120,
            fps: 24,
            output: Output::default(),
            textures: vec![],
            collections: vec![],
            next_id: 1,
        };
        s.add(Primitive::Cube, None);
        let l = s.add_light(LightType::Point, None);
        if let Some(o) = s.get_mut(l) {
            o.transform.location = Vec3::new(4.08, 1.0, 5.9);
        }
        let c = s.add_camera(None);
        if let Some(o) = s.get_mut(c) {
            o.transform.location = Vec3::new(7.36, -6.93, 4.96);
            o.look_at(Vec3::ZERO);
        }
        s
    }
}

impl Scene {
    /// Find or create a collection by name (case-insensitive); returns its id.
    pub fn collection_by_name(&mut self, name: &str) -> u32 {
        let name = name.trim();
        if name.is_empty() || name.eq_ignore_ascii_case("scene collection") || name.eq_ignore_ascii_case("scene") {
            return 0;
        }
        if let Some(c) = self.collections.iter().find(|c| c.name.eq_ignore_ascii_case(name)) {
            return c.id;
        }
        let id = self.collections.iter().map(|c| c.id).max().unwrap_or(0) + 1;
        self.collections.push(Collection { id, name: name.to_string(), visible: true });
        id
    }

    /// Would making `child` a child of `parent` create a loop?
    pub fn would_cycle(&self, child: u64, parent: u64) -> bool {
        let mut p = Some(parent);
        let mut guard = 0;
        while let Some(id) = p {
            if id == child {
                return true;
            }
            guard += 1;
            if guard > 10_000 {
                return true;
            }
            p = self.get(id).and_then(|o| o.parent);
        }
        false
    }

    /// Objects ordered so that every parent comes before its children.
    pub fn hierarchy_order(&self) -> Vec<u64> {
        let mut out = Vec::with_capacity(self.objects.len());
        let mut done: std::collections::HashSet<u64> = std::collections::HashSet::new();
        fn visit(s: &Scene, id: u64, out: &mut Vec<u64>, done: &mut std::collections::HashSet<u64>, depth: usize) {
            if done.contains(&id) || depth > 256 {
                return;
            }
            if let Some(p) = s.get(id).and_then(|o| o.parent) {
                if s.get(p).is_some() {
                    visit(s, p, out, done, depth + 1);
                }
            }
            if done.insert(id) {
                out.push(id);
            }
        }
        for o in &self.objects {
            visit(self, o.id, &mut out, &mut done, 0);
        }
        out
    }

    /// Add (or reuse an identical) texture and return its id.
    pub fn add_texture(&mut self, name: &str, source: crate::textures::TexSource) -> u32 {
        if let Some(t) = self.textures.iter().find(|t| t.source == source) {
            return t.id;
        }
        let id = self.textures.iter().map(|t| t.id).max().unwrap_or(0) + 1;
        let mut nm = name.to_string();
        let mut k = 1;
        while self.textures.iter().any(|t| t.name == nm) {
            nm = format!("{name}.{k:03}");
            k += 1;
        }
        self.textures.push(crate::textures::Texture { id, name: nm, source });
        id
    }

    pub fn texture(&self, id: u32) -> Option<&crate::textures::Texture> {
        (id != 0).then(|| self.textures.iter().find(|t| t.id == id)).flatten()
    }

    pub fn fresh_id(&mut self) -> u64 {
        self.next_id = self
            .next_id
            .max(self.objects.iter().map(|o| o.id).max().unwrap_or(0) + 1);
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    pub fn unique_name(&self, base: &str) -> String {
        let base = base.trim();
        let base = if base.is_empty() { "Object" } else { base };
        if !self.objects.iter().any(|o| o.name == base) {
            return base.to_string();
        }
        for i in 1.. {
            let n = format!("{base}.{i:03}");
            if !self.objects.iter().any(|o| o.name == n) {
                return n;
            }
        }
        unreachable!()
    }

    pub fn add(&mut self, prim: Primitive, name: Option<&str>) -> u64 {
        let id = self.fresh_id();
        let name = self.unique_name(name.unwrap_or(prim.label()));
        let mut transform = Transform::default();
        if prim == Primitive::Torus {
            transform.scale = Vec3::ONE;
        }
        self.objects.push(Object {
            id,
            name,
            kind: ObjectKind::Mesh,
            primitive: prim,
            mesh: None,
            detail: 2,
            transform,
            material: Material::default(),
            modifiers: vec![],
            keyframes: vec![],
            visible: true,
            smooth_override: None,
            auto_smooth: 0.0,
            parent: None,
            collection: 0,
            armature: None,
            skin: None,
            posed: None,
            posed_key: 0,
        });
        id
    }

    pub fn add_armature(&mut self, arm: crate::armature::Armature, name: Option<&str>) -> u64 {
        let id = self.add(Primitive::Cube, Some(name.unwrap_or("Armature")));
        let o = self.get_mut(id).unwrap();
        o.kind = ObjectKind::Armature;
        o.armature = Some(arm);
        id
    }

    /// Recompute the deformed meshes of skinned objects whose armature pose (or transforms)
    /// changed. Cheap when nothing moved.
    pub fn update_skins(&mut self) {
        use std::hash::{Hash, Hasher};
        if !self.objects.iter().any(|o| o.skin.is_some()) {
            return;
        }
        let arms: Vec<(u64, Mat4, crate::armature::Armature)> =
            self.objects.iter().filter_map(|o| o.armature.as_ref().map(|a| (o.id, o.transform.matrix(), a.clone()))).collect();
        for o in self.objects.iter_mut().filter(|o| o.skin.is_some()) {
            let skin = o.skin.as_ref().unwrap();
            let Some((_, aw, arm)) = arms.iter().find(|(id, _, _)| *id == skin.armature) else {
                o.posed = None;
                continue;
            };
            let Some(pm) = o.mesh.as_ref() else {
                o.posed = None;
                continue;
            };
            if !skin.valid_for(pm, arm) {
                // topology changed (edit mode) or bones changed: the weights no longer apply
                o.posed = None;
                continue;
            }
            let om = o.transform.matrix();
            let mut h = std::collections::hash_map::DefaultHasher::new();
            for m in arm.pose_globals().iter().chain([aw, &om]) {
                for f in m.to_cols_array() {
                    f.to_bits().hash(&mut h);
                }
            }
            pm.verts.len().hash(&mut h);
            for v in pm.verts.iter().step_by((pm.verts.len() / 64).max(1)) {
                (v.x.to_bits(), v.y.to_bits(), v.z.to_bits()).hash(&mut h);
            }
            let key = h.finish();
            if o.posed.is_some() && o.posed_key == key {
                continue;
            }
            o.posed = Some(std::sync::Arc::new(crate::armature::deform(pm, skin, arm, *aw, om)));
            o.posed_key = key;
        }
    }

    pub fn add_light(&mut self, t: LightType, name: Option<&str>) -> u64 {
        let id = self.add(
            Primitive::Cube,
            Some(name.unwrap_or(match t {
                LightType::Point => "Light",
                LightType::Sun => "Sun",
                LightType::Spot => "Spot",
            })),
        );
        let o = self.get_mut(id).unwrap();
        o.kind = ObjectKind::Light(Light::new(t));
        o.material.color = [1.0, 0.95, 0.8];
        id
    }

    pub fn add_camera(&mut self, name: Option<&str>) -> u64 {
        let id = self.add(Primitive::Cube, Some(name.unwrap_or("Camera")));
        self.get_mut(id).unwrap().kind = ObjectKind::Camera(CameraData { fov_deg: 39.6 });
        id
    }

    pub fn active_camera(&self) -> Option<&Object> {
        self.objects
            .iter()
            .find(|o| matches!(o.kind, ObjectKind::Camera(_)))
    }

    pub fn get(&self, id: u64) -> Option<&Object> {
        self.objects.iter().find(|o| o.id == id)
    }
    pub fn get_mut(&mut self, id: u64) -> Option<&mut Object> {
        self.objects.iter_mut().find(|o| o.id == id)
    }

    pub fn find_by_name(&self, name: &str) -> Option<u64> {
        let n = name.trim().to_lowercase();
        self.objects
            .iter()
            .find(|o| o.name.to_lowercase() == n)
            .or_else(|| {
                self.objects
                    .iter()
                    .find(|o| o.name.to_lowercase().starts_with(&n))
            })
            .or_else(|| {
                self.objects
                    .iter()
                    .find(|o| o.name.to_lowercase().contains(&n))
            })
            .map(|o| o.id)
    }

    pub fn remove(&mut self, ids: &[u64]) {
        self.objects.retain(|o| !ids.contains(&o.id));
    }

    pub fn evaluate(&mut self, frame: i32) {
        for o in &mut self.objects {
            o.evaluate(frame);
        }
    }

    pub fn summary(&self, selected: &[u64]) -> String {
        if self.objects.is_empty() {
            return "The scene is empty.".into();
        }
        let mut s = String::new();
        for o in &self.objects {
            let t = &o.transform;
            let c = o.material.color;
            s += &format!(
                "- \"{}\" ({}){}: location=[{:.2},{:.2},{:.2}] rotation=[{:.0},{:.0},{:.0}] scale=[{:.2},{:.2},{:.2}] color=[{:.2},{:.2},{:.2}]{}{}\n",
                o.name,
                o.type_label(),
                if selected.contains(&o.id) {
                    " [SELECTED]"
                } else {
                    ""
                },
                t.location.x,
                t.location.y,
                t.location.z,
                t.rotation.x,
                t.rotation.y,
                t.rotation.z,
                t.scale.x,
                t.scale.y,
                t.scale.z,
                c[0],
                c[1],
                c[2],
                {
                    let mut x = String::new();
                    if !o.visible {
                        x += " hidden";
                    }
                    if let Some(p) = o.parent.and_then(|p| self.get(p)) {
                        x += &format!(" parent=\"{}\"", p.name);
                    }
                    if let Some(c) = self.collections.iter().find(|c| c.id == o.collection) {
                        x += &format!(" collection=\"{}\"", c.name);
                    }
                    if let Some(sk) = &o.skin {
                        if let Some(a) = self.get(sk.armature) {
                            x += &format!(" rigged_to=\"{}\"", a.name);
                        }
                    }
                    if let Some(a) = &o.armature {
                        let names: Vec<&str> = a.bones.iter().take(16).map(|b| b.name.as_str()).collect();
                        x += &format!(" bones=[{}{}]", names.join(", "), if a.bones.len() > 16 { ", …" } else { "" });
                        if a.has_keys() {
                            x += " (animated)";
                        }
                    }
                    x
                },
                if o.keyframes.is_empty() {
                    String::new()
                } else {
                    format!(" keyframes={}", o.keyframes.len())
                }
            );
        }
        s
    }

    /// Wavefront OBJ export (applies transforms + modifiers).
    pub fn to_obj(&self, cache: &mut MeshCache) -> String {
        let mut out = String::from("# Exported from Rust Blender\n");
        let mut base = 1u32;
        for o in self.objects.iter().filter(|o| o.visible && o.is_mesh()) {
            let mesh = cache.for_object(o);
            for (ii, inst) in o.instance_matrices().iter().enumerate() {
                let m = o.transform.matrix() * *inst;
                out += &format!(
                    "o {}{}\n",
                    o.name.replace(' ', "_"),
                    if ii > 0 {
                        format!("_{ii}")
                    } else {
                        String::new()
                    }
                );
                for p in &mesh.positions {
                    let w = m.transform_point3(*p);
                    out += &format!("v {:.5} {:.5} {:.5}\n", w.x, w.y, w.z);
                }
                for t in &mesh.tris {
                    out += &format!("f {} {} {}\n", t[0] + base, t[1] + base, t[2] + base);
                }
                base += mesh.positions.len() as u32;
            }
        }
        out
    }
}

#[derive(Default)]
pub struct MeshCache {
    map: HashMap<(Primitive, u32, bool), Mesh>,
    /// object id (and whether it's the cage) -> (source mesh, smooth, subsurf levels, auto smooth, result)
    custom: HashMap<(u64, u8), (Option<crate::mesh::PolyMesh>, (Primitive, u32), bool, u32, u32, u64, Mesh)>,
}

impl MeshCache {
    pub fn get(&mut self, p: Primitive, detail: u32, smooth: bool) -> &Mesh {
        self.map
            .entry((p, detail, smooth))
            .or_insert_with(|| p.poly(detail).to_mesh(smooth))
    }

    /// Render mesh for an object (custom edited mesh or cached primitive), with its
    /// Subdivision Surface modifiers and auto smooth applied.
    pub fn for_object(&mut self, o: &Object) -> Mesh {
        self.build(o, 0)
    }

    /// The editable cage (no subdivision), as drawn in edit mode.
    pub fn for_object_cage(&mut self, o: &Object) -> Mesh {
        self.build(o, 1)
    }

    /// Posed (armature-deformed) but not subdivided: vertices match the mesh's own (weight paint).
    pub fn for_object_posed_base(&mut self, o: &Object) -> Mesh {
        self.build(o, 2)
    }

    /// mode 0: final, 1: edit cage (rest, no subdivision), 2: posed without subdivision
    fn build(&mut self, o: &Object, mode: u8) -> Mesh {
        let cage = mode == 1;
        let smooth = o.smooth();
        let levels = if mode != 0 { 0 } else { o.subsurf_levels() };
        let auto = (o.auto_smooth * 100.0).round() as u32;
        if o.mesh.is_none() && levels == 0 && auto == 0 && o.posed.is_none() {
            return self.get(o.primitive, o.detail, smooth).clone();
        }
        let prim = (o.primitive, o.detail);
        let key2 = if !cage && o.posed.is_some() { o.posed_key } else { 0 };
        if let Some((src, p, s, l, a, k, m)) = self.custom.get(&(o.id, mode)) {
            if *src == o.mesh && *p == prim && *s == smooth && *l == levels && *a == auto && *k == key2 {
                return m.clone();
            }
        }
        let base = match (&o.posed, cage) {
            (Some(p), false) => (**p).clone(),
            _ => o.poly(),
        };
        let pm = if levels > 0 { crate::editmesh::subdivide_surface(&base, levels) } else { base };
        let m = pm.to_mesh_opts(smooth, o.auto_smooth);
        self.custom.insert((o.id, mode), (o.mesh.clone(), prim, smooth, levels, auto, key2, m.clone()));
        m
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn keyframe_interp() {
        let mut o = Scene::default().objects[0].clone();
        o.insert_keyframe(1);
        o.transform.location = Vec3::new(10.0, 0.0, 0.0);
        o.insert_keyframe(11);
        o.evaluate(6);
        assert!((o.transform.location.x - 5.0).abs() < 1e-4);
        o.evaluate(100);
        assert_eq!(o.transform.location.x, 10.0);
    }
    #[test]
    fn unique_names() {
        let mut s = Scene::default();
        s.add(Primitive::Cube, None);
        s.add(Primitive::Cube, None);
        let names: Vec<_> = s
            .objects
            .iter()
            .filter(|o| o.is_mesh())
            .map(|o| o.name.clone())
            .collect();
        assert_eq!(names, vec!["Cube", "Cube.001", "Cube.002"]);
    }
    #[test]
    fn euler_roundtrip() {
        let mut t = Transform {
            rotation: Vec3::new(30.0, 20.0, 70.0),
            ..Default::default()
        };
        let q = t.quat();
        t.set_quat(q);
        assert!((t.rotation - Vec3::new(30.0, 20.0, 70.0)).length() < 1e-3);
    }
    #[test]
    fn look_at_points_minus_z() {
        let s = Scene::default();
        let cam = s.active_camera().unwrap();
        let f = cam.forward();
        let want = (Vec3::ZERO - cam.transform.location).normalize();
        assert!((f - want).length() < 1e-3, "{f} {want}");
        // camera up (+Y local) should point roughly up
        assert!((cam.transform.quat() * Vec3::Y).z > 0.5);
    }

    #[test]
    fn modifiers() {
        let mut s = Scene::default();
        s.objects[0].modifiers.push(Modifier::Array {
            count: 3,
            offset: Vec3::X * 2.0,
        });
        s.objects[0].modifiers.push(Modifier::Mirror {
            x: false,
            y: true,
            z: false,
        });
        assert_eq!(s.objects[0].instance_matrices().len(), 6);
        let json = serde_json::to_string(&s).unwrap();
        let back: Scene = serde_json::from_str(&json).unwrap();
        assert_eq!(back.objects, s.objects);
    }
}
