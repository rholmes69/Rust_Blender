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
}

impl Default for Material {
    fn default() -> Self {
        Self {
            color: [0.8, 0.8, 0.8],
            roughness: 0.5,
            metallic: 0.0,
            emission: 0.0,
            alpha: 1.0,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Modifier {
    Array { count: u32, offset: Vec3 },
    Mirror { x: bool, y: bool, z: bool },
    Radial { count: u32, radius: f32 },
}

impl Modifier {
    pub fn label(&self) -> &'static str {
        match self {
            Modifier::Array { .. } => "Array",
            Modifier::Mirror { .. } => "Mirror",
            Modifier::Radial { .. } => "Radial Array",
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
    pub fn smooth(&self) -> bool {
        self.smooth_override
            .unwrap_or(self.primitive.default_smooth())
    }

    pub fn type_label(&self) -> String {
        match &self.kind {
            ObjectKind::Mesh => self.primitive.label().to_string(),
            ObjectKind::Light(l) => format!("{} Light", l.light_type.label()),
            ObjectKind::Camera(_) => "Camera".into(),
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
            }
        }
        out.truncate(2000);
        out
    }

    pub fn evaluate(&mut self, frame: i32) {
        if let Some(t) = sample_keys(&self.keyframes, frame) {
            self.transform = t;
        }
    }

    pub fn insert_keyframe(&mut self, frame: i32) {
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
    next_id: u64,
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
        });
        id
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
                if o.visible { "" } else { " hidden" },
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
    map: HashMap<(Primitive, u32), Mesh>,
    custom: HashMap<u64, (crate::mesh::PolyMesh, bool, Mesh)>,
}

impl MeshCache {
    pub fn get(&mut self, p: Primitive, detail: u32) -> &Mesh {
        self.map
            .entry((p, detail))
            .or_insert_with(|| p.build(detail))
    }

    /// Render mesh for an object (custom edited mesh or cached primitive).
    pub fn for_object(&mut self, o: &Object) -> Mesh {
        let smooth = o.smooth();
        match &o.mesh {
            None => {
                let mut m = self.get(o.primitive, o.detail).clone();
                m.smooth = smooth;
                m
            }
            Some(pm) => {
                if let Some((cached, s, m)) = self.custom.get(&o.id) {
                    if cached == pm && *s == smooth {
                        return m.clone();
                    }
                }
                let m = pm.to_mesh(smooth);
                self.custom.insert(o.id, (pm.clone(), smooth, m.clone()));
                m
            }
        }
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
