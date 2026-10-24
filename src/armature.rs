//! Armatures (bone hierarchies), pose mode data, bone keyframes and linear-blend skinning.
//!
//! Bones follow Blender's convention: a bone points along its local +Y axis from its head.
//! `rest` places a bone relative to its parent (or the armature object); `pose` is applied on
//! top of the rest pose, so an identity pose shows the rest position.

use crate::mesh::PolyMesh;
use glam::{Mat4, Quat, Vec3};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct BonePose {
    pub loc: Vec3,
    pub rot: Quat,
    pub scale: Vec3,
}

impl Default for BonePose {
    fn default() -> Self {
        Self { loc: Vec3::ZERO, rot: Quat::IDENTITY, scale: Vec3::ONE }
    }
}

impl BonePose {
    pub fn matrix(&self) -> Mat4 {
        Mat4::from_scale_rotation_translation(self.scale, self.rot, self.loc)
    }
    pub fn from_matrix(m: Mat4) -> BonePose {
        let (scale, rot, loc) = m.to_scale_rotation_translation();
        BonePose { loc, rot: rot.normalize(), scale }
    }
    pub fn lerp(&self, o: &BonePose, t: f32) -> BonePose {
        BonePose { loc: self.loc.lerp(o.loc, t), rot: self.rot.slerp(o.rot, t).normalize(), scale: self.scale.lerp(o.scale, t) }
    }
    /// Rotation as Euler XYZ degrees (for the UI and the AI).
    pub fn euler_deg(&self) -> Vec3 {
        let (z, y, x) = self.rot.to_euler(glam::EulerRot::ZYX);
        Vec3::new(x, y, z) * (180.0 / std::f32::consts::PI)
    }
    pub fn set_euler_deg(&mut self, e: Vec3) {
        let r = e * (std::f32::consts::PI / 180.0);
        self.rot = Quat::from_euler(glam::EulerRot::ZYX, r.z, r.y, r.x);
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct BoneKey {
    pub frame: i32,
    pub pose: BonePose,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Bone {
    pub name: String,
    /// Parent bone index (always lower than this bone's index).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent: Option<usize>,
    pub rest: BonePose,
    pub length: f32,
    /// Tail in the bone's own space when it doesn't point along +Y (imported skeletons).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tail: Option<Vec3>,
    #[serde(default)]
    pub pose: BonePose,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub keys: Vec<BoneKey>,
}

impl Bone {
    pub fn tail_local(&self) -> Vec3 {
        self.tail.unwrap_or(Vec3::new(0.0, self.length.max(1e-4), 0.0))
    }
}

#[derive(Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
pub struct Armature {
    pub bones: Vec<Bone>,
    /// Show (and deform with) the rest position instead of the pose.
    #[serde(default)]
    pub rest_position: bool,
}

impl Armature {
    /// A chain of `n` bones of total `height`, standing on the armature origin along +Z.
    pub fn chain(n: usize, height: f32) -> Armature {
        let n = n.clamp(1, 64);
        let len = height / n as f32;
        let bones = (0..n)
            .map(|i| Bone {
                name: if n == 1 { "Bone".into() } else { format!("Bone.{:03}", i) },
                parent: (i > 0).then(|| i - 1),
                // the root turns +Y (bone direction) to +Z; children continue along their parent
                rest: if i == 0 { BonePose { rot: Quat::from_rotation_x(std::f32::consts::FRAC_PI_2), ..Default::default() } } else { BonePose { loc: Vec3::new(0.0, len, 0.0), ..Default::default() } },
                length: len,
                tail: None,
                pose: BonePose::default(),
                keys: vec![],
            })
            .collect();
        Armature { bones, rest_position: false }
    }

    fn globals(&self, posed: bool) -> Vec<Mat4> {
        let mut g: Vec<Mat4> = Vec::with_capacity(self.bones.len());
        for b in &self.bones {
            let local = if posed { b.rest.matrix() * b.pose.matrix() } else { b.rest.matrix() };
            let m = match b.parent.and_then(|p| g.get(p)) {
                Some(p) => *p * local,
                None => local,
            };
            g.push(m);
        }
        g
    }

    /// Bone matrices in armature space at rest.
    pub fn rest_globals(&self) -> Vec<Mat4> {
        self.globals(false)
    }

    /// Bone matrices in armature space in the current pose (rest if `rest_position`).
    pub fn pose_globals(&self) -> Vec<Mat4> {
        self.globals(!self.rest_position)
    }

    /// The armature-space matrix of a bone's frame *before* its own pose (parent pose × rest).
    pub fn pre_pose(&self, i: usize) -> Mat4 {
        let g = self.pose_globals();
        let b = &self.bones[i];
        match b.parent {
            Some(p) => g[p] * b.rest.matrix(),
            None => b.rest.matrix(),
        }
    }

    /// Head and tail of every bone (armature space) for drawing.
    pub fn segments(&self, globals: &[Mat4]) -> Vec<(Vec3, Vec3)> {
        globals
            .iter()
            .zip(&self.bones)
            .map(|(g, b)| {
                let head = g.transform_point3(Vec3::ZERO);
                (head, g.transform_point3(b.tail_local()))
            })
            .collect()
    }

    pub fn find(&self, name: &str) -> Option<usize> {
        let n = name.trim().to_lowercase();
        self.bones
            .iter()
            .position(|b| b.name.to_lowercase() == n)
            .or_else(|| self.bones.iter().position(|b| b.name.to_lowercase().contains(&n)))
            .or_else(|| {
                // "bone 2" / "2" → index
                n.split_whitespace().last().and_then(|w| w.parse::<usize>().ok()).filter(|i| *i < self.bones.len())
            })
    }

    pub fn insert_keys(&mut self, frame: i32) {
        for b in &mut self.bones {
            b.keys.retain(|k| k.frame != frame);
            b.keys.push(BoneKey { frame, pose: b.pose });
            b.keys.sort_by_key(|k| k.frame);
        }
    }

    pub fn has_keys(&self) -> bool {
        self.bones.iter().any(|b| !b.keys.is_empty())
    }

    pub fn evaluate(&mut self, frame: f32) {
        for b in &mut self.bones {
            if let Some(p) = sample(&b.keys, frame) {
                b.pose = p;
            }
        }
    }

    pub fn reset_pose(&mut self) {
        for b in &mut self.bones {
            b.pose = BonePose::default();
        }
    }
}

/// Linear between keys (imported animation is baked per frame; hand keys get an ease).
fn sample(keys: &[BoneKey], frame: f32) -> Option<BonePose> {
    let first = keys.first()?;
    if frame <= first.frame as f32 {
        return Some(first.pose);
    }
    let last = keys.last()?;
    if frame >= last.frame as f32 {
        return Some(last.pose);
    }
    let i = keys.partition_point(|k| (k.frame as f32) <= frame);
    let (a, b) = (&keys[i - 1], &keys[i]);
    let span = (b.frame - a.frame).max(1) as f32;
    let mut t = (frame - a.frame as f32) / span;
    if span > 1.5 {
        t = t * t * (3.0 - 2.0 * t);
    }
    Some(a.pose.lerp(&b.pose, t))
}

/// Skin weights of a mesh: up to 4 bones per vertex.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Skin {
    /// The armature object deforming the mesh.
    pub armature: u64,
    pub joints: Vec<[u16; 4]>,
    pub weights: Vec<[f32; 4]>,
    /// Per bone: mesh-local space → bone space at bind time.
    pub inverse_bind: Vec<Mat4>,
}

impl Skin {
    pub fn valid_for(&self, pm: &PolyMesh, arm: &Armature) -> bool {
        self.joints.len() == pm.verts.len() && self.weights.len() == pm.verts.len() && self.inverse_bind.len() == arm.bones.len()
    }

    /// Bind `pm` (with object matrix `obj`) to the armature in its current rest position.
    pub fn bind(armature: u64, pm: &PolyMesh, obj: Mat4, arm: &Armature, arm_world: Mat4, joints: Vec<[u16; 4]>, weights: Vec<[f32; 4]>) -> Skin {
        let inverse_bind = arm.rest_globals().iter().map(|g| (arm_world * *g).inverse() * obj).collect();
        let _ = pm;
        Skin { armature, joints, weights, inverse_bind }
    }
}

/// Deformed copy of `pm` (local space of the object with matrix `obj`).
pub fn deform(pm: &PolyMesh, skin: &Skin, arm: &Armature, arm_world: Mat4, obj: Mat4) -> PolyMesh {
    let g = arm.pose_globals();
    let to_local = obj.inverse() * arm_world;
    let mats: Vec<Mat4> = g.iter().zip(&skin.inverse_bind).map(|(g, ib)| to_local * *g * *ib).collect();
    let mut out = pm.clone();
    for (i, v) in out.verts.iter_mut().enumerate() {
        let (j, w) = (skin.joints[i], skin.weights[i]);
        let mut acc = Vec3::ZERO;
        let mut ws = 0.0;
        for k in 0..4 {
            if w[k] > 0.0 {
                if let Some(m) = mats.get(j[k] as usize) {
                    acc += m.transform_point3(pm.verts[i]) * w[k];
                    ws += w[k];
                }
            }
        }
        if ws > 1e-6 {
            *v = acc / ws;
        }
    }
    out
}

fn seg_dist(p: Vec3, a: Vec3, b: Vec3) -> f32 {
    let ab = b - a;
    let t = ((p - a).dot(ab) / ab.length_squared().max(1e-12)).clamp(0.0, 1.0);
    p.distance(a + ab * t)
}

/// Automatic weights: each vertex follows the nearest bones (inverse distance to the bone
/// segments, sharpened, best 4 kept).
pub fn auto_weights(pm: &PolyMesh, obj: Mat4, arm: &Armature, arm_world: Mat4) -> (Vec<[u16; 4]>, Vec<[f32; 4]>) {
    let segs: Vec<(Vec3, Vec3)> = arm.segments(&arm.rest_globals()).into_iter().map(|(h, t)| (arm_world.transform_point3(h), arm_world.transform_point3(t))).collect();
    let size = segs.iter().map(|(h, t)| h.distance(*t)).fold(0.0, f32::max).max(1e-3);
    let mut joints = Vec::with_capacity(pm.verts.len());
    let mut weights = Vec::with_capacity(pm.verts.len());
    for v in &pm.verts {
        let p = obj.transform_point3(*v);
        let mut best: Vec<(f32, u16)> = segs.iter().enumerate().map(|(i, (a, b))| (seg_dist(p, *a, *b), i as u16)).collect();
        best.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        best.truncate(4);
        let d0 = best[0].0;
        let mut j = [0u16; 4];
        let mut w = [0f32; 4];
        let mut sum = 0.0;
        for (k, (d, i)) in best.iter().enumerate() {
            // bones much further away than the nearest one don't pull
            let x = 1.0 / (d - d0 * 0.5 + size * 0.08).powi(4);
            j[k] = *i;
            w[k] = x;
            sum += x;
        }
        for x in &mut w {
            *x /= sum;
        }
        // drop tiny influences
        for x in &mut w {
            if *x < 0.02 {
                *x = 0.0;
            }
        }
        let s: f32 = w.iter().sum();
        for x in &mut w {
            *x /= s.max(1e-6);
        }
        joints.push(j);
        weights.push(w);
    }
    (joints, weights)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn chain_pose_and_skinning() {
        let mut arm = Armature::chain(2, 2.0);
        let seg = arm.segments(&arm.rest_globals());
        assert!(seg[0].0.distance(Vec3::ZERO) < 1e-5 && seg[0].1.distance(Vec3::Z) < 1e-5, "{seg:?}");
        assert!(seg[1].1.distance(Vec3::Z * 2.0) < 1e-5);
        // a column of vertices along the chain
        let pm = PolyMesh { verts: (0..=8).map(|i| Vec3::new(0.1, 0.0, i as f32 * 0.25)).collect(), faces: vec![vec![0, 1, 2]], uvs: vec![] };
        let (j, w) = auto_weights(&pm, Mat4::IDENTITY, &arm, Mat4::IDENTITY);
        assert_eq!(j[0][0], 0, "bottom follows the first bone");
        assert_eq!(j[8][0], 1, "top follows the second bone");
        let skin = Skin::bind(1, &pm, Mat4::IDENTITY, &arm, Mat4::IDENTITY, j, w);
        // rest pose: no change
        let d = deform(&pm, &skin, &arm, Mat4::IDENTITY, Mat4::IDENTITY);
        assert!(d.verts.iter().zip(&pm.verts).all(|(a, b)| a.distance(*b) < 1e-4));
        // bend the upper bone 90° about its local X: the top swings sideways, the base stays
        arm.bones[1].pose.rot = Quat::from_rotation_x(std::f32::consts::FRAC_PI_2);
        let d = deform(&pm, &skin, &arm, Mat4::IDENTITY, Mat4::IDENTITY);
        assert!(d.verts[0].distance(pm.verts[0]) < 1e-3);
        assert!((d.verts[8].z - 1.0).abs() < 0.1 && d.verts[8].y.abs() > 0.8, "{:?}", d.verts[8]);
        // moving armature and mesh together keeps the mesh in place locally
        let t = Mat4::from_translation(Vec3::new(5.0, 0.0, 0.0));
        let d2 = deform(&pm, &skin, &arm, t, t);
        assert!(d2.verts.iter().zip(&d.verts).all(|(a, b)| a.distance(*b) < 1e-3));
        // keys
        arm.insert_keys(10);
        arm.bones[1].pose = BonePose::default();
        arm.insert_keys(1);
        arm.evaluate(10.0);
        assert!(arm.bones[1].pose.rot.angle_between(Quat::from_rotation_x(std::f32::consts::FRAC_PI_2)) < 1e-3);
        arm.evaluate(5.5);
        let a = arm.bones[1].pose.rot.angle_between(Quat::IDENTITY);
        assert!(a > 0.3 && a < 1.3, "{a}");
        assert_eq!(arm.find("bone 1"), Some(1));
        assert_eq!(arm.find("Bone.000"), Some(0));
    }
}
