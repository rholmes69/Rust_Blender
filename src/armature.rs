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
    /// Inverse kinematics: this bone's tail reaches for an object.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ik: Option<Ik>,
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
    /// Stored animations (walk, run …). The active one's keys live in the bones.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub actions: Vec<Action>,
    #[serde(default)]
    pub active_action: usize,
    /// Hash of the last IK solve's inputs+result (skip re-solving when nothing moved).
    #[serde(skip)]
    pub ik_key: u64,
    /// Cross-fade after switching actions: the pose we came from, start frame, length.
    #[serde(skip)]
    pub blend: Option<(Vec<BonePose>, f32, f32)>,
    #[serde(skip)]
    pub blend_last: f32,
}

/// IK constraint: rotate this bone and `chain - 1` parents (0 = up to the root) so the bone's
/// tail touches the `target` object.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Ik {
    pub target: u64,
    pub chain: u32,
}

/// A named set of bone keys (Blender's Action).
#[derive(Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
pub struct Action {
    pub name: String,
    /// Keys per bone (same order as `Armature::bones`).
    pub keys: Vec<Vec<BoneKey>>,
}

impl Action {
    pub fn range(&self) -> Option<(i32, i32)> {
        let lo = self.keys.iter().filter_map(|k| k.first()).map(|k| k.frame).min()?;
        let hi = self.keys.iter().filter_map(|k| k.last()).map(|k| k.frame).max()?;
        Some((lo, hi))
    }
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
                ik: None,
                pose: BonePose::default(),
                keys: vec![],
            })
            .collect();
        Armature { bones, ..Default::default() }
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

    // ---- inverse kinematics ----

    /// Bones of an IK chain from the tip up.
    pub fn ik_chain(&self, tip: usize, chain: u32) -> Vec<usize> {
        let mut ids = vec![tip];
        let mut b = tip;
        while let Some(p) = self.bones[b].parent {
            if chain > 0 && ids.len() >= chain as usize {
                break;
            }
            ids.push(p);
            b = p;
        }
        ids
    }

    /// Cyclic coordinate descent: turn the chain so the tail of `tip` reaches `goal`
    /// (armature space). Returns the remaining distance.
    pub fn solve_ik(&mut self, tip: usize, goal: Vec3, chain: u32) -> f32 {
        let ids = self.ik_chain(tip, chain);
        let tip_pos = |a: &Armature| a.pose_globals()[tip].transform_point3(a.bones[tip].tail_local());
        for _ in 0..32 {
            for &j in &ids {
                let g = self.pose_globals();
                let t = g[tip].transform_point3(self.bones[tip].tail_local());
                if t.distance(goal) < 1e-5 {
                    return 0.0;
                }
                let head = g[j].w_axis.truncate();
                let (a, b) = (t - head, goal - head);
                if a.length() < 1e-6 || b.length() < 1e-6 {
                    continue;
                }
                let q = Quat::from_rotation_arc(a.normalize(), b.normalize());
                let pre = match self.bones[j].parent {
                    Some(p) => g[p] * self.bones[j].rest.matrix(),
                    None => self.bones[j].rest.matrix(),
                };
                let br = pre.to_scale_rotation_translation().1;
                self.bones[j].pose.rot = (br.inverse() * q * br * self.bones[j].pose.rot).normalize();
            }
        }
        tip_pos(self).distance(goal)
    }

    // ---- actions ----

    /// Copy the bones' keys into the active action (if actions are in use).
    pub fn stash_action(&mut self) {
        let keys: Vec<Vec<BoneKey>> = self.bones.iter().map(|b| b.keys.clone()).collect();
        if let Some(a) = self.actions.get_mut(self.active_action) {
            a.keys = keys;
        }
    }

    /// Make action `i` active: its keys go into the bones. Returns its frame range.
    pub fn set_action(&mut self, i: usize) -> Option<(i32, i32)> {
        if i >= self.actions.len() {
            return None;
        }
        self.stash_action();
        self.active_action = i;
        let a = &self.actions[i];
        for (k, b) in self.bones.iter_mut().enumerate() {
            b.keys = a.keys.get(k).cloned().unwrap_or_default();
        }
        if !self.has_keys() {
            self.reset_pose();
        }
        let r = self.actions[i].range();
        if let Some((lo, _)) = r {
            self.evaluate(lo as f32);
        }
        r
    }

    /// Start a new empty action (the current keys are kept in the previous one).
    pub fn new_action(&mut self, name: &str) -> usize {
        if self.actions.is_empty() {
            self.actions.push(Action { name: "Action".into(), keys: vec![] });
            self.active_action = 0;
        }
        self.stash_action();
        let mut nm = name.to_string();
        let mut k = 1;
        while self.actions.iter().any(|a| a.name == nm) {
            nm = format!("{name}.{k:03}");
            k += 1;
        }
        self.actions.push(Action { name: nm, keys: vec![vec![]; self.bones.len()] });
        self.active_action = self.actions.len() - 1;
        for b in &mut self.bones {
            b.keys.clear();
        }
        self.active_action
    }

    pub fn find_action(&self, name: &str) -> Option<usize> {
        let n = name.trim().to_lowercase();
        self.actions.iter().position(|a| a.name.to_lowercase() == n).or_else(|| self.actions.iter().position(|a| a.name.to_lowercase().contains(&n)))
    }

    // ---- edit mode (rest pose) ----

    /// Rebuild the parent-relative rest poses from armature-space matrices.
    fn set_rest_globals(&mut self, g: &[Mat4]) {
        for i in 0..self.bones.len() {
            let local = match self.bones[i].parent {
                Some(p) => g[p].inverse() * g[i],
                None => g[i],
            };
            self.bones[i].rest = BonePose::from_matrix(local);
        }
    }

    /// Head and tail (armature space, rest) of bone `i`.
    pub fn rest_points(&self, i: usize) -> (Vec3, Vec3) {
        self.segments(&self.rest_globals())[i]
    }

    /// Move bone `i` so its rest head/tail are `head`/`tail` (armature space). Other bones keep
    /// their place, except connected neighbours: children whose head sat on the old tail follow
    /// it, and a parent whose tail sat on the old head follows that.
    pub fn set_rest_points(&mut self, i: usize, head: Vec3, tail: Vec3) {
        let mut g = self.rest_globals();
        let segs = self.segments(&g);
        let (oh, ot) = segs[i];
        let len = head.distance(tail).max(1e-4);
        let eps = (oh.distance(ot) * 1e-3).max(1e-6);
        let place = |g: &Mat4, b: &Bone, h: Vec3, t: Vec3| -> (Mat4, f32) {
            let (_, r, oh2) = g.to_scale_rotation_translation();
            let old_t = g.transform_point3(b.tail_local());
            let od = (old_t - oh2).normalize_or(Vec3::Y);
            let nd = (t - h).normalize_or(od);
            let r2 = (Quat::from_rotation_arc(od, nd) * r).normalize();
            (Mat4::from_rotation_translation(r2, h), h.distance(t).max(1e-4))
        };
        let mut new_len = vec![None; self.bones.len()];
        let (m, l) = place(&g[i], &self.bones[i], head, tail);
        g[i] = m;
        new_len[i] = Some(l);
        let _ = len;
        for k in 0..self.bones.len() {
            if k == i {
                continue;
            }
            let (kh, kt) = segs[k];
            if self.bones[k].parent == Some(i) && kh.distance(ot) < eps {
                let (m, l) = place(&g[k], &self.bones[k], tail, kt);
                g[k] = m;
                new_len[k] = Some(l);
            } else if self.bones[i].parent == Some(k) && kt.distance(oh) < eps {
                let (m, l) = place(&g[k], &self.bones[k], kh, head);
                g[k] = m;
                new_len[k] = Some(l);
            }
        }
        for (k, l) in new_len.iter().enumerate() {
            if let Some(l) = l {
                let b = &mut self.bones[k];
                b.tail = b.tail.map(|t| t.normalize_or(Vec3::Y) * *l);
                b.length = *l;
            }
        }
        self.set_rest_globals(&g);
    }

    /// New bone continuing from bone `i`'s tail (Blender's E). Returns its index.
    pub fn extrude(&mut self, i: usize) -> usize {
        let (h, t) = self.rest_points(i);
        let dir = (t - h).normalize_or(Vec3::Z);
        let len = h.distance(t).max(0.05) * 0.6;
        self.add_bone(t, t + dir * len, Some(i))
    }

    /// Add a bone between `head` and `tail` (armature space).
    pub fn add_bone(&mut self, head: Vec3, tail: Vec3, parent: Option<usize>) -> usize {
        let mut g = self.rest_globals();
        let dir = (tail - head).normalize_or(Vec3::Z);
        let m = Mat4::from_rotation_translation(Quat::from_rotation_arc(Vec3::Y, dir), head);
        let mut name = format!("Bone.{:03}", self.bones.len());
        let mut k = self.bones.len();
        while self.bones.iter().any(|b| b.name == name) {
            k += 1;
            name = format!("Bone.{k:03}");
        }
        self.bones.push(Bone { name, parent, rest: BonePose::default(), length: head.distance(tail).max(1e-4), tail: None, ik: None, pose: BonePose::default(), keys: vec![] });
        for a in &mut self.actions {
            a.keys.resize(self.bones.len(), vec![]);
        }
        g.push(m);
        self.set_rest_globals(&g);
        self.bones.len() - 1
    }

    /// Delete bone `i`; its children move to its parent. Returns old index → new index.
    pub fn delete_bone(&mut self, i: usize) -> Vec<Option<usize>> {
        let g = self.rest_globals();
        let parent = self.bones[i].parent;
        let map: Vec<Option<usize>> = (0..self.bones.len()).map(|k| if k == i { None } else if k > i { Some(k - 1) } else { Some(k) }).collect();
        let mut g2 = vec![];
        let mut bones = vec![];
        for (k, b) in self.bones.iter().enumerate() {
            if k == i {
                continue;
            }
            let mut b = b.clone();
            b.parent = match b.parent {
                Some(p) if p == i => parent.and_then(|q| map[q]),
                Some(p) => map[p],
                None => None,
            };
            bones.push(b);
            g2.push(g[k]);
        }
        self.bones = bones;
        for a in &mut self.actions {
            if i < a.keys.len() {
                a.keys.remove(i);
            }
        }
        self.set_rest_globals(&g2);
        map
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
        let back = frame < self.blend_last;
        self.blend_last = frame;
        if let Some((from, start, len)) = &self.blend {
            let t = (frame - start) / len.max(1.0);
            if !back && (0.0..1.0).contains(&t) && from.len() == self.bones.len() {
                let t = t * t * (3.0 - 2.0 * t);
                for (b, f) in self.bones.iter_mut().zip(from) {
                    b.pose = f.lerp(&b.pose, t);
                }
            } else {
                self.blend = None;
            }
        }
    }

    /// Switch action with a cross-fade of `frames` from the current pose (starting at `start`).
    pub fn set_action_blended(&mut self, i: usize, start: f32, frames: f32) -> Option<(i32, i32)> {
        let from: Vec<BonePose> = self.bones.iter().map(|b| b.pose).collect();
        let r = self.set_action(i);
        if frames >= 1.0 {
            self.blend = Some((from, start, frames));
            self.blend_last = start;
            self.evaluate(start);
        }
        r
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

/// Which part of a bone is selected in armature edit mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum BonePart {
    Head,
    Tail,
    #[default]
    Body,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PaintMode {
    Add,
    Subtract,
    Blur,
}

impl Skin {
    /// Weight of `bone` on vertex `i`.
    pub fn weight(&self, i: usize, bone: usize) -> f32 {
        let (j, w) = (self.joints[i], self.weights[i]);
        (0..4).filter(|k| j[*k] as usize == bone).map(|k| w[k]).sum()
    }

    /// Set the weight of `bone` on vertex `i`; the other influences share the rest (auto
    /// normalize). A vertex keeps its four strongest bones.
    pub fn set_weight(&mut self, i: usize, bone: usize, w: f32) {
        let w = w.clamp(0.0, 1.0);
        let (j, ws) = (&mut self.joints[i], &mut self.weights[i]);
        let slot = match (0..4).find(|k| j[*k] as usize == bone && ws[*k] > 0.0).or_else(|| (0..4).find(|k| j[*k] as usize == bone)) {
            Some(k) => k,
            None => {
                if w <= 0.0 {
                    return;
                }
                let k = (0..4).min_by(|a, b| ws[*a].partial_cmp(&ws[*b]).unwrap_or(std::cmp::Ordering::Equal)).unwrap();
                j[k] = bone as u16;
                ws[k] = 0.0;
                k
            }
        };
        // duplicates of the same bone in other slots fold into this one
        for k in 0..4 {
            if k != slot && j[k] as usize == bone {
                ws[k] = 0.0;
            }
        }
        let others: f32 = (0..4).filter(|k| *k != slot).map(|k| ws[k]).sum();
        ws[slot] = w;
        if others > 1e-6 {
            let s = (1.0 - w) / others;
            for k in 0..4 {
                if k != slot {
                    ws[k] *= s;
                }
            }
        }
    }
}

/// One weight-paint dab on the posed mesh (`posed`, local space; `to_world` places it) around
/// the world point `c` with world radius `r`. Returns how many vertices changed.
#[allow(clippy::too_many_arguments)]
pub fn paint(posed: &PolyMesh, to_world: Mat4, adj: &[Vec<u32>], skin: &mut Skin, c: Vec3, r: f32, bone: usize, strength: f32, mode: PaintMode) -> usize {
    let inside: Vec<(usize, f32)> = posed
        .verts
        .iter()
        .enumerate()
        .filter_map(|(i, v)| {
            let d = to_world.transform_point3(*v).distance(c);
            (d < r).then(|| (i, crate::editmesh::falloff(d, r)))
        })
        .collect();
    let old: Vec<f32> = match mode {
        PaintMode::Blur => (0..posed.verts.len()).map(|i| skin.weight(i, bone)).collect(),
        _ => vec![],
    };
    let mut n = 0;
    for (i, f) in inside {
        if i >= skin.joints.len() {
            continue;
        }
        let f = (f * strength).clamp(0.0, 1.0);
        let w = skin.weight(i, bone);
        let nw = match mode {
            PaintMode::Add => w + (1.0 - w) * f,
            PaintMode::Subtract => w * (1.0 - f),
            PaintMode::Blur => {
                let nb = &adj[i];
                if nb.is_empty() {
                    w
                } else {
                    let m = nb.iter().map(|&k| old[k as usize]).sum::<f32>() / nb.len() as f32;
                    w + (m - w) * f
                }
            }
        };
        if (nw - w).abs() > 1e-5 {
            skin.set_weight(i, bone, nw);
            n += 1;
        }
    }
    n
}

/// Weight → colour (blue 0 · green 0.5 · red 1), like Blender's weight display.
pub fn weight_color(w: f32) -> [f32; 3] {
    let w = w.clamp(0.0, 1.0);
    let stops = [[0.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [1.0, 0.0, 0.0]];
    let x = w * 4.0;
    let i = (x.floor() as usize).min(3);
    let t = x - i as f32;
    let (a, b) = (stops[i], stops[i + 1]);
    [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]
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
        // weight painting keeps weights normalised
        let mut sk = skin.clone();
        sk.set_weight(0, 1, 0.75);
        assert!((sk.weight(0, 1) - 0.75).abs() < 1e-5 && (sk.weight(0, 0) - 0.25).abs() < 1e-3, "{:?}", sk.weights[0]);
        let adj = crate::sculpt::adjacency(&pm);
        let n = paint(&pm, Mat4::IDENTITY, &adj, &mut sk, pm.verts[8], 0.6, 0, 1.0, PaintMode::Add);
        assert!(n > 0 && sk.weight(8, 0) > 0.5, "painting bone 0 at the top");
        paint(&pm, Mat4::IDENTITY, &adj, &mut sk, pm.verts[8], 0.6, 0, 1.0, PaintMode::Subtract);
        assert!(sk.weight(8, 0) < 0.3);
        assert_eq!(weight_color(0.0), [0.0, 0.0, 1.0]);
        // edit bones
        let mut e = Armature::chain(2, 2.0);
        e.set_rest_points(0, Vec3::ZERO, Vec3::new(0.0, 1.0, 1.0));
        let (h1, t1) = e.rest_points(1);
        assert!(h1.distance(Vec3::new(0.0, 1.0, 1.0)) < 1e-4, "connected child follows: {h1}");
        assert!(t1.distance(Vec3::Z * 2.0) < 1e-4, "its tail stays: {t1}");
        let n = e.extrude(1);
        assert_eq!(e.bones[n].parent, Some(1));
        assert!(e.rest_points(n).0.distance(t1) < 1e-4);
        let map = e.delete_bone(1);
        assert_eq!(map, vec![Some(0), None, Some(1)]);
        assert_eq!(e.bones[1].parent, Some(0));
        assert!(e.rest_points(1).0.distance(t1) < 1e-4, "orphans keep their place");
        // actions
        let mut a = Armature::chain(2, 2.0);
        a.bones[1].pose.rot = Quat::from_rotation_x(0.5);
        a.insert_keys(1);
        let walk = a.new_action("Walk");
        assert_eq!((walk, a.actions.len()), (1, 2));
        assert!(!a.has_keys());
        a.bones[1].pose.rot = Quat::from_rotation_x(-0.5);
        a.insert_keys(5);
        assert_eq!(a.set_action(0), Some((1, 1)));
        assert!(a.bones[1].pose.rot.angle_between(Quat::from_rotation_x(0.5)) < 1e-4);
        assert_eq!(a.set_action(a.find_action("walk").unwrap()), Some((5, 5)));
        // cross-fade: halfway through the blend the pose is between the two actions
        a.set_action(0);
        a.set_action_blended(1, 1.0, 8.0);
        a.evaluate(5.0);
        let mid = a.bones[1].pose.rot.angle_between(Quat::IDENTITY);
        assert!(mid < 0.3, "blending passes through the middle: {mid}");
        a.evaluate(20.0);
        assert!(a.blend.is_none());
        assert!(a.bones[1].pose.rot.angle_between(Quat::from_rotation_x(-0.5)) < 1e-4);
        assert_eq!(weight_color(1.0), [1.0, 0.0, 0.0]);
        assert_eq!(arm.find("Bone.000"), Some(0));
    }
}
