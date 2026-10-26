//! Character starter kit: a stylised mannequin (one smooth mesh, about 1.75 m tall, facing −Y)
//! with a named humanoid armature, clean per-part skin weights and ready-made actions (Idle,
//! Walk, Run, Wave, Jump). Used by the asset library and the Character template.

use crate::armature::{Armature, BoneKey, BonePose};
use crate::mesh::PolyMesh;
use glam::{Mat4, Quat, Vec2, Vec3};

/// Bone names, in armature order (parents first).
pub const BONES: &[(&str, Option<usize>, [f32; 3], [f32; 3])] = &[
    ("hips", None, [0.0, 0.0, 0.95], [0.0, 0.0, 1.1]),
    ("spine", Some(0), [0.0, 0.0, 1.1], [0.0, 0.0, 1.3]),
    ("chest", Some(1), [0.0, 0.0, 1.3], [0.0, 0.0, 1.47]),
    ("neck", Some(2), [0.0, 0.0, 1.47], [0.0, 0.0, 1.56]),
    ("head", Some(3), [0.0, 0.0, 1.56], [0.0, 0.0, 1.8]),
    ("upper_arm.L", Some(2), [0.17, 0.0, 1.42], [0.34, 0.0, 1.17]),
    ("forearm.L", Some(5), [0.34, 0.0, 1.17], [0.46, -0.02, 0.95]),
    ("hand.L", Some(6), [0.46, -0.02, 0.95], [0.51, -0.03, 0.84]),
    ("upper_arm.R", Some(2), [-0.17, 0.0, 1.42], [-0.34, 0.0, 1.17]),
    ("forearm.R", Some(8), [-0.34, 0.0, 1.17], [-0.46, -0.02, 0.95]),
    ("hand.R", Some(9), [-0.46, -0.02, 0.95], [-0.51, -0.03, 0.84]),
    ("thigh.L", Some(0), [0.1, 0.0, 0.95], [0.11, -0.01, 0.52]),
    ("shin.L", Some(11), [0.11, -0.01, 0.52], [0.12, 0.02, 0.09]),
    ("foot.L", Some(12), [0.12, 0.02, 0.09], [0.12, -0.13, 0.03]),
    ("thigh.R", Some(0), [-0.1, 0.0, 0.95], [-0.11, -0.01, 0.52]),
    ("shin.R", Some(14), [-0.11, -0.01, 0.52], [-0.12, 0.02, 0.09]),
    ("foot.R", Some(15), [-0.12, 0.02, 0.09], [-0.12, -0.13, 0.03]),
];

fn v(a: [f32; 3]) -> Vec3 {
    Vec3::from(a)
}

/// The humanoid armature (armature space = the character standing on its origin).
pub fn armature() -> Armature {
    let mut arm = Armature::default();
    for (name, parent, head, tail) in BONES {
        let i = arm.add_bone(v(*head), v(*tail), *parent);
        arm.bones[i].name = name.to_string();
    }
    arm
}

/// A tube along a path of (centre, half-width x, half-depth y) sections; the ends close to a
/// point. `side` is the axis the half-width runs along (the depth runs along side × direction).
fn tube(pm: &mut PolyMesh, path: &[(Vec3, Vec2)], side: Vec3, ring: usize) -> std::ops::Range<usize> {
    let start = pm.verts.len();
    let n = path.len();
    let mut rings: Vec<Option<u32>> = vec![];
    for i in 0..n {
        let (c, r) = path[i];
        let dir = if i + 1 < n { path[i + 1].0 - c } else { c - path[i - 1].0 };
        let dir = if i > 0 && i + 1 < n { path[i + 1].0 - path[i - 1].0 } else { dir }.normalize_or(Vec3::Z);
        if r.x <= 1e-5 && r.y <= 1e-5 {
            pm.verts.push(c);
            rings.push(None);
            continue;
        }
        let sx = (side - dir * side.dot(dir)).normalize_or(Vec3::X);
        let sy = dir.cross(sx).normalize_or(Vec3::Y);
        let base = pm.verts.len() as u32;
        for k in 0..ring {
            let a = k as f32 / ring as f32 * std::f32::consts::TAU;
            pm.verts.push(c + sx * (a.cos() * r.x) + sy * (a.sin() * r.y));
        }
        rings.push(Some(base));
    }
    let pole = |i: usize, pm: &PolyMesh| -> u32 { (start + (0..i).map(|j| if rings[j].is_some() { ring } else { 1 }).sum::<usize>()) as u32 + 0 * pm.verts.len() as u32 };
    for i in 0..n - 1 {
        let r = ring as u32;
        match (rings[i], rings[i + 1]) {
            (Some(a), Some(b)) => {
                for k in 0..r {
                    let k1 = (k + 1) % r;
                    pm.faces.push(vec![a + k, b + k, b + k1, a + k1]);
                }
            }
            (None, Some(b)) => {
                let p = pole(i, pm);
                for k in 0..r {
                    pm.faces.push(vec![p, b + k, b + (k + 1) % r]);
                }
            }
            (Some(a), None) => {
                let p = pole(i + 1, pm);
                for k in 0..r {
                    pm.faces.push(vec![a + k, p, a + (k + 1) % r]);
                }
            }
            (None, None) => {}
        }
    }
    start..pm.verts.len()
}

/// Make each face wind counter-clockwise seen from outside (away from the part's centre line).
fn orient(pm: &mut PolyMesh, faces_from: usize, axis: &[Vec3]) {
    for f in &mut pm.faces[faces_from..] {
        let p: Vec<Vec3> = f.iter().map(|&i| pm.verts[i as usize]).collect();
        let c = p.iter().copied().sum::<Vec3>() / p.len() as f32;
        let n = (p[1] - p[0]).cross(p[2] - p[0]);
        // nearest point on the part's axis polyline
        let mut best = (f32::MAX, Vec3::ZERO);
        for w in axis.windows(2) {
            let (a, b) = (w[0], w[1]);
            let ab = b - a;
            let t = ((c - a).dot(ab) / ab.length_squared().max(1e-9)).clamp(0.0, 1.0);
            let q = a + ab * t;
            let d = q.distance_squared(c);
            if d < best.0 {
                best = (d, q);
            }
        }
        if n.dot(c - best.1) < 0.0 {
            f.reverse();
        }
    }
}

/// Which bones may pull a part's vertices.
struct Part {
    range: std::ops::Range<usize>,
    bones: &'static [usize],
}

/// The mannequin mesh and its parts (vertex ranges with candidate bones).
fn body() -> (PolyMesh, Vec<Part>) {
    let mut pm = PolyMesh::default();
    let mut parts = vec![];
    let p = |x: f32, y: f32, z: f32, rx: f32, ry: f32| (Vec3::new(x, y, z), Vec2::new(rx, ry));
    let mut add = |pm: &mut PolyMesh, path: Vec<(Vec3, Vec2)>, side: Vec3, ring: usize, bones: &'static [usize]| {
        let f0 = pm.faces.len();
        let range = tube(pm, &path, side, ring);
        let axis: Vec<Vec3> = path.iter().map(|q| q.0).collect();
        orient(pm, f0, &axis);
        parts.push(Part { range, bones });
    };
    // torso: pelvis up to the shoulders
    add(&mut pm, vec![p(0.0, 0.0, 0.82, 0.0, 0.0), p(0.0, 0.0, 0.84, 0.1, 0.07), p(0.0, 0.0, 0.9, 0.16, 0.1), p(0.0, 0.0, 1.0, 0.17, 0.11), p(0.0, 0.0, 1.1, 0.15, 0.1), p(0.0, 0.005, 1.22, 0.155, 0.1), p(0.0, 0.0, 1.34, 0.18, 0.115), p(0.0, 0.0, 1.42, 0.195, 0.11), p(0.0, 0.0, 1.47, 0.14, 0.08), p(0.0, 0.0, 1.49, 0.06, 0.05), p(0.0, 0.0, 1.5, 0.0, 0.0)], Vec3::X, 16, &[0, 1, 2, 3, 5, 8, 11, 14]);
    // neck and head
    add(&mut pm, vec![p(0.0, 0.0, 1.44, 0.0, 0.0), p(0.0, 0.0, 1.45, 0.05, 0.05), p(0.0, 0.005, 1.54, 0.048, 0.048), p(0.0, 0.0, 1.57, 0.07, 0.08), p(0.0, -0.005, 1.63, 0.095, 0.11), p(0.0, 0.0, 1.7, 0.1, 0.115), p(0.0, 0.005, 1.77, 0.085, 0.1), p(0.0, 0.005, 1.81, 0.05, 0.06), p(0.0, 0.005, 1.825, 0.0, 0.0)], Vec3::X, 14, &[2, 3, 4]);
    // nose, so the front is readable
    add(&mut pm, vec![p(0.0, -0.09, 1.665, 0.0, 0.0), p(0.0, -0.1, 1.665, 0.016, 0.014), p(0.0, -0.125, 1.655, 0.012, 0.01), p(0.0, -0.13, 1.65, 0.0, 0.0)], Vec3::X, 8, &[4]);
    for s in [1.0f32, -1.0] {
        let ft: &'static [usize] = if s > 0.0 { &[12, 13] } else { &[15, 16] };
        // arm: shoulder → wrist in one tube, hand separately (flat)
        add(&mut pm, vec![p(0.13 * s, 0.0, 1.44, 0.0, 0.0), p(0.15 * s, 0.0, 1.43, 0.05, 0.05), p(0.2 * s, 0.0, 1.38, 0.056, 0.056), p(0.28 * s, 0.0, 1.27, 0.045, 0.045), p(0.34 * s, 0.0, 1.17, 0.038, 0.038), p(0.4 * s, -0.01, 1.06, 0.04, 0.038), p(0.455 * s, -0.02, 0.96, 0.028, 0.025), p(0.465 * s, -0.02, 0.94, 0.0, 0.0)], Vec3::Y, 12, ua_fa(s));
        add(&mut pm, vec![p(0.455 * s, -0.02, 0.97, 0.0, 0.0), p(0.46 * s, -0.02, 0.955, 0.03, 0.016), p(0.485 * s, -0.025, 0.9, 0.04, 0.018), p(0.505 * s, -0.03, 0.85, 0.032, 0.014), p(0.515 * s, -0.03, 0.82, 0.0, 0.0)], Vec3::new(0.0, -1.0, 0.0).cross(Vec3::new(s, 0.0, -2.0)).normalize(), 10, ha_of(s));
        // leg: hip → ankle
        add(&mut pm, vec![p(0.095 * s, 0.0, 0.98, 0.0, 0.0), p(0.1 * s, 0.0, 0.95, 0.08, 0.085), p(0.105 * s, -0.005, 0.8, 0.078, 0.08), p(0.11 * s, -0.01, 0.62, 0.058, 0.06), p(0.11 * s, -0.01, 0.52, 0.05, 0.052), p(0.115 * s, 0.0, 0.38, 0.052, 0.055), p(0.12 * s, 0.015, 0.18, 0.036, 0.038), p(0.12 * s, 0.02, 0.09, 0.035, 0.036), p(0.12 * s, 0.02, 0.06, 0.0, 0.0)], Vec3::X, 12, th_sh(s));
        // foot: heel → toe
        add(&mut pm, vec![p(0.12 * s, 0.06, 0.05, 0.0, 0.0), p(0.12 * s, 0.045, 0.05, 0.035, 0.035), p(0.12 * s, 0.0, 0.045, 0.042, 0.04), p(0.12 * s, -0.08, 0.035, 0.045, 0.03), p(0.12 * s, -0.14, 0.03, 0.035, 0.022), p(0.12 * s, -0.16, 0.03, 0.0, 0.0)], Vec3::X, 10, ft);
    }
    (pm, parts)
}

fn ua_fa(s: f32) -> &'static [usize] {
    if s > 0.0 { &[2, 5, 6, 7] } else { &[2, 8, 9, 10] }
}
fn ha_of(s: f32) -> &'static [usize] {
    if s > 0.0 { &[6, 7] } else { &[9, 10] }
}
fn th_sh(s: f32) -> &'static [usize] {
    if s > 0.0 { &[0, 11, 12, 13] } else { &[0, 14, 15, 16] }
}

fn seg_dist(p: Vec3, a: Vec3, b: Vec3) -> f32 {
    let ab = b - a;
    let t = ((p - a).dot(ab) / ab.length_squared().max(1e-9)).clamp(0.0, 1.0);
    (a + ab * t).distance(p)
}

/// Skin weights: each vertex blends only the bones its part allows, by distance.
fn weights(pm: &PolyMesh, parts: &[Part]) -> (Vec<[u16; 4]>, Vec<[f32; 4]>) {
    let segs: Vec<(Vec3, Vec3)> = BONES.iter().map(|b| (v(b.2), v(b.3))).collect();
    let mut j = vec![[0u16; 4]; pm.verts.len()];
    let mut w = vec![[0f32; 4]; pm.verts.len()];
    for part in parts {
        for i in part.range.clone() {
            let p = pm.verts[i];
            let mut c: Vec<(f32, usize)> = part.bones.iter().map(|&b| (seg_dist(p, segs[b].0, segs[b].1), b)).collect();
            c.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            c.truncate(4);
            let d0 = c[0].0;
            let mut sum = 0.0;
            for (k, (d, b)) in c.iter().enumerate() {
                // a soft blend over ~4 cm around joints
                let x = 1.0 / (d - d0 * 0.6 + 0.012).powi(4);
                j[i][k] = *b as u16;
                w[i][k] = x;
                sum += x;
            }
            for x in &mut w[i] {
                *x /= sum;
            }
        }
    }
    (j, w)
}

/// Mesh + skin data for a character (armature at the origin, mesh at the origin).
pub fn build() -> (PolyMesh, Armature, Vec<[u16; 4]>, Vec<[f32; 4]>) {
    let (pm, parts) = body();
    let (j, w) = weights(&pm, &parts);
    let mut arm = armature();
    add_actions(&mut arm);
    (pm, arm, j, w)
}

// ---------------------------------------------------------------------------------------------
// Actions
// ---------------------------------------------------------------------------------------------

/// A pose rotation for bone `i` that turns it by `deg` about a world axis (at rest).
fn turn(arm: &Armature, g: &[Mat4], i: usize, axis: Vec3, deg: f32) -> Quat {
    let _ = arm;
    let br = g[i].to_scale_rotation_translation().1;
    (br.inverse() * Quat::from_axis_angle(axis.normalize(), deg.to_radians()) * br).normalize()
}

/// A pose rotation that aims bone `i` along a world direction (at rest).
fn aim(arm: &Armature, g: &[Mat4], i: usize, dir: Vec3) -> Quat {
    let (h, t) = (g[i].transform_point3(Vec3::ZERO), g[i].transform_point3(arm.bones[i].tail_local()));
    let q = Quat::from_rotation_arc((t - h).normalize(), dir.normalize());
    let br = g[i].to_scale_rotation_translation().1;
    (br.inverse() * q * br).normalize()
}

fn idx(name: &str) -> usize {
    BONES.iter().position(|b| b.0 == name).expect(name)
}

/// Key every bone at every `step` frames from `f(frame t 0..1)` → per-bone poses.
fn make_action(arm: &mut Armature, name: &str, frames: i32, step: i32, f: impl Fn(&Armature, &[Mat4], f32) -> Vec<(usize, BonePose)>) {
    let g = arm.rest_globals();
    let n = arm.bones.len();
    let mut keys: Vec<Vec<BoneKey>> = vec![vec![]; n];
    let mut fr = 1;
    loop {
        let t = (fr - 1) as f32 / (frames - 1).max(1) as f32;
        let poses = f(arm, &g, t);
        for (b, keyset) in keys.iter_mut().enumerate() {
            let pose = poses.iter().find(|(i, _)| *i == b).map(|x| x.1).unwrap_or_default();
            keyset.push(BoneKey { frame: fr, pose });
        }
        if fr >= frames {
            break;
        }
        fr = (fr + step).min(frames);
    }
    arm.actions.push(crate::armature::Action { name: name.into(), keys });
}

fn rot(q: Quat) -> BonePose {
    BonePose { rot: q, ..Default::default() }
}

pub const ACTIONS: &[&str] = &["Idle", "Walk", "Run", "Wave", "Jump"];

fn add_actions(arm: &mut Armature) {
    use std::f32::consts::TAU;
    let x = Vec3::X;
    let (hips, spine, chest, neck, head) = (idx("hips"), idx("spine"), idx("chest"), idx("neck"), idx("head"));
    let (ual, fal, hal, uar, far, har) = (idx("upper_arm.L"), idx("forearm.L"), idx("hand.L"), idx("upper_arm.R"), idx("forearm.R"), idx("hand.R"));
    let (thl, shl, ftl, thr, shr, ftr) = (idx("thigh.L"), idx("shin.L"), idx("foot.L"), idx("thigh.R"), idx("shin.R"), idx("foot.R"));
    // hips move along their own +Y = world up (the root bone points up)
    let lift = |dz: f32, dy: f32| Vec3::new(0.0, dz, dy);
    // arms hang down at the sides in every action except where they move
    let arms_down = |a: &Armature, g: &[Mat4], swing_l: f32, swing_r: f32, bend: f32| -> Vec<(usize, BonePose)> {
        vec![
            (ual, rot(turn(a, g, ual, x, swing_l) * aim(a, g, ual, Vec3::new(0.12, 0.0, -1.0)))),
            (uar, rot(turn(a, g, uar, x, swing_r) * aim(a, g, uar, Vec3::new(-0.12, 0.0, -1.0)))),
            (fal, rot(turn(a, g, fal, x, -bend))),
            (far, rot(turn(a, g, far, x, -bend))),
        ]
    };
    make_action(arm, "Idle", 73, 3, |a, g, t| {
        let s = (t * TAU).sin();
        let mut v = arms_down(a, g, 2.0 * s, -2.0 * s, 10.0);
        v.push((chest, rot(turn(a, g, chest, x, 1.5 * s))));
        v.push((head, rot(turn(a, g, head, Vec3::Z, 6.0 * (t * TAU * 0.5).sin()))));
        v.push((hips, BonePose { loc: lift(0.004 * s, 0.0), ..Default::default() }));
        v
    });
    let gait = |a: &Armature, g: &[Mat4], t: f32, stride: f32, knee: f32, arm: f32, bob: f32, lean: f32| -> Vec<(usize, BonePose)> {
        let ph = t * TAU;
        let (s, c) = ph.sin_cos();
        // L leg forward at s = 1 (rotate −X swings the foot forward, −Y)
        let mut v = arms_down(a, g, arm * s, -arm * s, 15.0 + arm * 0.5);
        v.push((thl, rot(turn(a, g, thl, x, -stride * s))));
        v.push((thr, rot(turn(a, g, thr, x, stride * s))));
        // knees bend while the leg swings forward
        let kl = knee * (0.5 + 0.5 * (ph + 1.2).sin()).powi(2) + 4.0;
        let kr = knee * (0.5 + 0.5 * (ph + 1.2 + std::f32::consts::PI).sin()).powi(2) + 4.0;
        v.push((shl, rot(turn(a, g, shl, x, kl))));
        v.push((shr, rot(turn(a, g, shr, x, kr))));
        v.push((ftl, rot(turn(a, g, ftl, x, -kl * 0.3 + stride * 0.2 * s))));
        v.push((ftr, rot(turn(a, g, ftr, x, -kr * 0.3 - stride * 0.2 * s))));
        v.push((hips, BonePose { loc: lift(bob * (2.0 * ph).cos() - bob, 0.0), rot: turn(a, g, hips, Vec3::Z, 4.0 * s), ..Default::default() }));
        v.push((spine, rot(turn(a, g, spine, x, -lean))));
        v.push((chest, rot(turn(a, g, chest, Vec3::Z, -7.0 * s))));
        v.push((neck, rot(turn(a, g, neck, Vec3::Z, 3.0 * s))));
        let _ = c;
        v
    };
    make_action(arm, "Walk", 33, 2, |a, g, t| gait(a, g, t, 24.0, 35.0, 18.0, 0.018, 2.0));
    make_action(arm, "Run", 19, 1, |a, g, t| {
        let mut v = gait(a, g, t, 38.0, 75.0, 40.0, 0.04, 10.0);
        // elbows bent while running
        v.retain(|(i, _)| *i != fal && *i != far);
        v.push((fal, rot(turn(a, g, fal, x, -80.0))));
        v.push((far, rot(turn(a, g, far, x, -80.0))));
        v
    });
    make_action(arm, "Wave", 49, 2, |a, g, t| {
        let w = (t * TAU * 3.0).sin();
        // up over 6 frames, wave, down at the end
        let up = (t * 8.0).min(1.0).min((1.0 - t) * 8.0).clamp(0.0, 1.0);
        let up = up * up * (3.0 - 2.0 * up);
        let mut v = arms_down(a, g, 0.0, 0.0, 10.0);
        v.retain(|(i, _)| *i != uar && *i != far);
        let down = aim(a, g, uar, Vec3::new(-0.12, 0.0, -1.0));
        let raised = aim(a, g, uar, Vec3::new(-0.95, -0.1, 0.4));
        v.push((uar, rot(down.slerp(raised, up))));
        v.push((far, rot(Quat::IDENTITY.slerp(turn(a, g, far, Vec3::new(1.0, 0.0, -0.3), -70.0) * turn(a, g, far, Vec3::Y, 25.0 * w), up))));
        v.push((har, rot(turn(a, g, har, Vec3::Y, 15.0 * w * up))));
        v.push((head, rot(turn(a, g, head, Vec3::Z, -12.0 * up))));
        v.push((hal, rot(Quat::IDENTITY)));
        v
    });
    make_action(arm, "Jump", 37, 1, |a, g, t| {
        // crouch (0–0.3), push off and fly (0.3–0.7), land and recover
        let crouch = if t < 0.3 { (t / 0.3).powi(2) } else if t < 0.4 { 1.0 - (t - 0.3) / 0.1 } else if t < 0.7 { 0.0 } else if t < 0.8 { (t - 0.7) / 0.1 } else { 1.0 - (t - 0.8) / 0.2 };
        let air = if (0.38..0.72).contains(&t) { ((t - 0.38) / 0.34 * std::f32::consts::PI).sin() } else { 0.0 };
        let k = crouch * 70.0;
        let swing = if t < 0.35 { -30.0 * crouch } else { 60.0 * air };
        let mut v = arms_down(a, g, swing, swing, 20.0);
        for (th, sh, ft) in [(thl, shl, ftl), (thr, shr, ftr)] {
            v.push((th, rot(turn(a, g, th, x, -k * 0.8))));
            v.push((sh, rot(turn(a, g, sh, x, k * 1.4))));
            v.push((ft, rot(turn(a, g, ft, x, -k * 0.55 + air * 25.0))));
        }
        v.push((spine, rot(turn(a, g, spine, x, -crouch * 20.0))));
        v.push((hips, BonePose { loc: lift(-crouch * 0.22 + air * 0.45, 0.0), ..Default::default() }));
        v
    });
    // start on Idle
    arm.active_action = 0;
    let keys = arm.actions[0].keys.clone();
    for (b, k) in arm.bones.iter_mut().zip(keys) {
        b.keys = k;
    }
    arm.evaluate(1.0);
}

/// Add a character to the scene at `at` (armature + skinned body, optionally hair on a scalp).
/// Returns (armature id, body id).
pub fn add(scene: &mut crate::scene::Scene, name: &str, at: Vec3, color: [f32; 3]) -> (u64, u64) {
    let (pm, arm, j, w) = build();
    let arm_id = scene.add_armature(arm.clone(), Some(&format!("{name} Rig")));
    scene.get_mut(arm_id).unwrap().transform.location = at;
    let body = scene.add(crate::mesh::Primitive::Cube, Some(name));
    let aw = Mat4::from_translation(at);
    let om = Mat4::from_translation(at);
    {
        let o = scene.get_mut(body).unwrap();
        o.transform.location = at;
        o.mesh = Some(pm.clone());
        o.smooth_override = Some(true);
        o.material.color = color;
        o.material.roughness = 0.55;
        o.modifiers.push(crate::scene::Modifier::Subsurf { levels: 1 });
        o.skin = Some(crate::armature::Skin::bind(arm_id, &pm, om, &arm, aw, j, w));
        o.parent = Some(arm_id);
    }
    scene.update_derived();
    (arm_id, body)
}

/// Give a character (its body object) a scalp with long hair, skinned to the head bone so the
/// hair follows the head. Returns (scalp id, hair id).
pub fn add_hair(scene: &mut crate::scene::Scene, body: u64, color: [f32; 3]) -> (u64, u64) {
    let (at, arm_id, name) = {
        let o = scene.get(body).unwrap();
        (o.transform.location, o.skin.as_ref().map(|s| s.armature).unwrap_or(0), o.name.clone())
    };
    // a cap over the top and back of the head, just outside the skin
    let mut cap = PolyMesh::default();
    let (c, r) = (Vec3::new(0.0, 0.012, 1.685), Vec3::new(0.106, 0.122, 0.142));
    let (rings, segs) = (7usize, 18usize);
    cap.verts.push(c + Vec3::new(0.0, 0.0, r.z));
    for i in 1..=rings {
        let th = i as f32 / rings as f32 * 1.75; // down past the ears
        for k in 0..segs {
            let a = k as f32 / segs as f32 * std::f32::consts::TAU;
            // the hairline sits higher at the front (−Y)
            let front = (-a.sin()).max(0.0);
            let th = th * (1.0 - 0.38 * front);
            cap.verts.push(c + Vec3::new(r.x * th.sin() * a.cos(), r.y * th.sin() * a.sin(), r.z * th.cos()));
        }
    }
    for k in 0..segs as u32 {
        cap.faces.push(vec![0, 1 + k, 1 + (k + 1) % segs as u32]);
    }
    for i in 0..rings as u32 - 1 {
        for k in 0..segs as u32 {
            let k1 = (k + 1) % segs as u32;
            let (a, b) = (1 + i * segs as u32, 1 + (i + 1) * segs as u32);
            cap.faces.push(vec![a + k, b + k, b + k1, a + k1]);
        }
    }
    let scalp = scene.add(crate::mesh::Primitive::UvSphere, Some(&format!("{name} Scalp")));
    let head = idx("head") as u16;
    let n = cap.verts.len();
    if let Some(arm) = scene.get(arm_id).and_then(|a| a.armature.clone()) {
        let aw = scene.get(arm_id).unwrap().transform.matrix();
        let om = Mat4::from_translation(at);
        let o = scene.get_mut(scalp).unwrap();
        o.skin = Some(crate::armature::Skin::bind(arm_id, &cap, om, &arm, aw, vec![[head, 0, 0, 0]; n], vec![[1.0, 0.0, 0.0, 0.0]; n]));
        o.parent = Some(arm_id);
    }
    let o = scene.get_mut(scalp).unwrap();
    o.transform.location = at;
    o.mesh = Some(cap);
    o.smooth_override = Some(true);
    o.material.color = [color[0] * 0.5, color[1] * 0.5, color[2] * 0.5];
    o.material.roughness = 0.6;
    let (mut h, _, _) = crate::hair::Hair::preset(scalp, "hair");
    (h.count, h.children, h.length, h.gravity, h.child_radius, h.clump) = (420, 10, 0.32, 0.9, 0.025, 0.5);
    // combed so it lies on the head: back from the hairline, down and a little outward elsewhere
    let cap_now = scene.get(scalp).unwrap().mesh.clone().unwrap();
    let guides = crate::hair::guides(&h, &cap_now, &cap_now, Mat4::from_translation(at));
    h.groom = guides
        .iter()
        .map(|g| {
            let n = g.as_ref().map(|g| g.normal).unwrap_or(Vec3::Z);
            let front = (-n.y).max(0.0);
            let out = Vec3::new(n.x, n.y.max(0.0), 0.0) * 0.25;
            let c = out + Vec3::new(0.0, 1.3 * front + 0.25, -1.3 * (1.0 - 0.7 * front));
            [c.x, c.y, c.z, 1.0 - 0.4 * front]
        })
        .collect();
    let hid = scene.add(crate::mesh::Primitive::Cube, Some(&format!("{name} Hair")));
    let mut g = crate::nodes::NodeGraph::preset(0, "hair strands").expect("hair preset");
    g.set_rgb(color);
    let gid = scene.add_graph(g);
    let o = scene.get_mut(hid).unwrap();
    o.hair = Some(h);
    o.smooth_override = Some(true);
    o.material.color = color;
    o.material.roughness = 0.45;
    o.material.graph = gid;
    o.parent = Some(arm_id);
    scene.update_derived();
    (scalp, hid)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mannequin_mesh_rig_and_actions() {
        let (pm, arm, j, w) = build();
        assert_eq!(arm.bones.len(), BONES.len());
        assert_eq!(j.len(), pm.verts.len());
        assert!(w.iter().all(|x| (x.iter().sum::<f32>() - 1.0).abs() < 1e-3));
        assert!(pm.faces.iter().all(|f| f.iter().all(|&i| (i as usize) < pm.verts.len())));
        // standing on the floor, about 1.8 m tall, facing −Y (the nose)
        let (lo, hi) = pm.verts.iter().fold((Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)), |(l, h), p| (l.min(*p), h.max(*p)));
        assert!(lo.z.abs() < 0.04 && (hi.z - 1.82).abs() < 0.05, "{lo} {hi}");
        let nose = pm.verts.iter().filter(|p| p.z > 1.6).map(|p| p.y).fold(f32::MAX, f32::min);
        assert!(nose < -0.12);
        // normals point outward: the hand's faces face away from the forearm axis roughly; check
        // the whole mesh encloses volume with consistent winding (positive signed volume)
        let mut vol = 0.0;
        for f in &pm.faces {
            for k in 1..f.len() - 1 {
                let (a, b, c) = (pm.verts[f[0] as usize], pm.verts[f[k] as usize], pm.verts[f[k + 1] as usize]);
                vol += a.dot(b.cross(c)) / 6.0;
            }
        }
        assert!(vol > 0.03, "outward faces enclose positive volume: {vol}");
        // the right hand's vertices only follow right-arm bones
        let hr = idx("hand.R");
        let near: Vec<usize> = (0..pm.verts.len()).filter(|&i| pm.verts[i].x < -0.48 && pm.verts[i].z < 0.88).collect();
        assert!(!near.is_empty());
        for i in near {
            assert!(j[i].iter().zip(w[i]).all(|(b, x)| x < 1e-4 || [idx("forearm.R") as u16, hr as u16].contains(b)), "{:?} {:?}", j[i], w[i]);
        }
        // actions: all there, each has keys and moves something
        assert_eq!(arm.actions.iter().map(|a| a.name.as_str()).collect::<Vec<_>>(), ACTIONS);
        for a in &arm.actions {
            let (lo, hi) = a.range().unwrap();
            assert!(hi > lo + 10, "{}", a.name);
            assert!(a.keys.iter().all(|k| k.iter().all(|k| k.pose.rot.is_finite() && k.pose.loc.is_finite())));
        }
    }

    #[test]
    fn walk_swings_the_legs_and_the_character_poses() {
        let mut s = crate::scene::Scene::default();
        s.objects.clear();
        let (arm_id, body) = add(&mut s, "Hero", Vec3::new(1.0, 0.0, 0.0), [0.8, 0.6, 0.5]);
        let a = s.get_mut(arm_id).unwrap().armature.as_mut().unwrap();
        let wi = a.find_action("walk").unwrap();
        a.set_action(wi);
        let foot = |s: &crate::scene::Scene, f: i32| -> f32 {
            let mut s = s.clone();
            s.evaluate(f);
            s.update_derived();
            let o = s.get(body).unwrap();
            // left foot: lowest vertices with x > 1
            let p = o.posed.as_ref().unwrap();
            let pts: Vec<Vec3> = p.verts.iter().filter(|v| v.x > 0.05 && v.z < 0.12).copied().collect();
            pts.iter().map(|v| v.y).sum::<f32>() / pts.len().max(1) as f32
        };
        // the left foot is in front (−Y) a quarter cycle in and behind three quarters in
        let (a1, a2) = (foot(&s, 9), foot(&s, 25));
        assert!(a1 < a2 - 0.15, "left foot forward then back: {a1} {a2}");
        // hair on a scalp that follows the head
        let (scalp, hair) = add_hair(&mut s, body, [0.3, 0.2, 0.1]);
        s.update_derived();
        let hp = s.get(hair).unwrap().posed.clone().unwrap();
        assert!(!hp.faces.is_empty());
        let top = hp.verts.iter().map(|v| v.z).fold(f32::MIN, f32::max);
        assert!(top > 1.7 && top < 1.95, "hair on the head: {top}");
        assert!(s.get(scalp).unwrap().skin.is_some());
    }
}
