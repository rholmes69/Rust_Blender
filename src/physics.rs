//! Cloth simulation (position-based dynamics), baked per frame. Other visible meshes act as
//! colliders using simple shapes: spheres for sphere primitives, boxes for everything else
//! (planes become thin boxes, i.e. floors).

use crate::mesh::PolyMesh;
use glam::{Mat4, Vec3};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Cloth {
    /// Pinned vertices stay where they are ("top", "corners", "left", "selected" chose them).
    pub pins: Vec<u32>,
    /// 0..1: how much the cloth resists stretching / bending.
    pub stiffness: f32,
    pub bending: f32,
    /// Velocity kept per step (air resistance).
    pub damping: f32,
    pub gravity: f32,
    /// Collision margin (thickness).
    pub thickness: f32,
    /// Baked local-space positions per frame, starting at `start`.
    #[serde(skip)]
    pub cache: std::sync::Arc<Vec<Vec<Vec3>>>,
    #[serde(default)]
    pub start: i32,
}

impl Default for Cloth {
    fn default() -> Self {
        Cloth { pins: vec![], stiffness: 0.9, bending: 0.15, damping: 0.99, gravity: 9.81, thickness: 0.02, cache: Default::default(), start: 1 }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum Collider {
    Sphere { c: Vec3, r: f32 },
    /// Oriented box: world→box matrix and half extents.
    Box { to_local: Mat4, to_world: Mat4, half: Vec3 },
}

impl Collider {
    /// Where a moving point a→b first enters the collider (boxes; thin floors included).
    fn sweep(&self, a: Vec3, b: Vec3) -> Option<(Vec3, Vec3)> {
        let Collider::Box { to_local, to_world, half } = *self else { return None };
        let (la, lb) = (to_local.transform_point3(a), to_local.transform_point3(b));
        let d = lb - la;
        let (mut t0, mut t1) = (0.0f32, 1.0f32);
        let mut axis = None;
        for k in 0..3 {
            if d[k].abs() < 1e-12 {
                if la[k].abs() > half[k] {
                    return None;
                }
                continue;
            }
            let (mut e, mut x) = ((-half[k] - la[k]) / d[k], (half[k] - la[k]) / d[k]);
            if e > x {
                std::mem::swap(&mut e, &mut x);
            }
            if e > t0 {
                t0 = e;
                axis = Some(k);
            }
            t1 = t1.min(x);
            if t0 > t1 {
                return None;
            }
        }
        let k = axis?;
        if t0 <= 0.0 {
            return None;
        }
        let mut nl = Vec3::ZERO;
        nl[k] = -d[k].signum();
        let hit = la + d * t0 + nl * 1e-4;
        Some((to_world.transform_point3(hit), to_world.transform_vector3(nl).normalize_or_zero()))
    }

    /// Push `p` out of the collider; returns the surface normal when it was inside.
    fn resolve(&self, p: &mut Vec3, margin: f32) -> Option<Vec3> {
        match *self {
            Collider::Sphere { c, r } => {
                let d = *p - c;
                let l = d.length();
                if l < r + margin {
                    let n = if l > 1e-6 { d / l } else { Vec3::Z };
                    *p = c + n * (r + margin);
                    return Some(n);
                }
                None
            }
            Collider::Box { to_local, to_world, half } => {
                let lp = to_local.transform_point3(*p);
                // margin in box space (approximately, using the box scale)
                let sc = to_world.transform_vector3(Vec3::ONE).abs().max(Vec3::splat(1e-6)) / 3f32.sqrt();
                let m = Vec3::splat(margin) / sc;
                let h = half + m;
                if lp.x.abs() < h.x && lp.y.abs() < h.y && lp.z.abs() < h.z {
                    // leave through the nearest face
                    let pen = h - lp.abs();
                    let (axis, _) = [(0usize, pen.x), (1, pen.y), (2, pen.z)].into_iter().min_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap();
                    let mut q = lp;
                    q[axis] = h[axis] * lp[axis].signum().max(if lp[axis] == 0.0 { 1.0 } else { -1.0 });
                    let mut nl = Vec3::ZERO;
                    nl[axis] = if lp[axis] >= 0.0 { 1.0 } else { -1.0 };
                    *p = to_world.transform_point3(q);
                    return Some(to_world.transform_vector3(nl).normalize_or_zero());
                }
                None
            }
        }
    }
}

/// Colliders from other objects: sphere primitives → spheres, the rest → their (local) bounds.
pub fn collider_for(o: &crate::scene::Object, mesh_bounds: (Vec3, Vec3)) -> Collider {
    let m = o.transform.matrix();
    use crate::mesh::Primitive;
    if o.mesh.is_none() && matches!(o.primitive, Primitive::UvSphere | Primitive::IcoSphere) {
        let s = o.transform.scale.abs();
        if (s.x - s.y).abs() < 1e-3 && (s.y - s.z).abs() < 1e-3 {
            return Collider::Sphere { c: o.transform.location, r: s.x * 0.99 };
        }
    }
    let (lo, hi) = mesh_bounds;
    let c = (lo + hi) * 0.5;
    let mut half = (hi - lo) * 0.5;
    // flat things (planes) get a little thickness
    half = half.max(Vec3::splat(1e-3));
    let to_world = m * Mat4::from_translation(c);
    Collider::Box { to_local: to_world.inverse(), to_world, half }
}

/// Bake the cloth: returns local-space positions for `frames` frames (the first is the rest).
pub fn bake(pm: &PolyMesh, obj: Mat4, cloth: &Cloth, colliders: &[Collider], frames: usize, fps: f32) -> Vec<Vec<Vec3>> {
    let n = pm.verts.len();
    let mut x: Vec<Vec3> = pm.verts.iter().map(|v| obj.transform_point3(*v)).collect();
    let rest_pos = x.clone();
    let mut prev = x.clone();
    let pinned: HashSet<u32> = cloth.pins.iter().copied().collect();
    let inv_mass: Vec<f32> = (0..n).map(|i| if pinned.contains(&(i as u32)) { 0.0 } else { 1.0 }).collect();
    // distance constraints: edges (stretch) and across adjacent faces (bending)
    let mut edges: HashSet<(u32, u32)> = HashSet::new();
    for f in &pm.faces {
        for i in 0..f.len() {
            let (a, b) = (f[i], f[(i + 1) % f.len()]);
            edges.insert((a.min(b), a.max(b)));
        }
        // quads: shear diagonals keep them from collapsing
        if f.len() == 4 {
            edges.insert((f[0].min(f[2]), f[0].max(f[2])));
            edges.insert((f[1].min(f[3]), f[1].max(f[3])));
        }
    }
    let mut bend: HashSet<(u32, u32)> = HashSet::new();
    let ef = pm.edge_faces();
    for ((a, b), fs) in &ef {
        if fs.len() == 2 {
            let opp = |fi: usize| pm.faces[fi].iter().copied().find(|v| v != a && v != b);
            if let (Some(p), Some(q)) = (opp(fs[0]), opp(fs[1])) {
                if p != q {
                    bend.insert((p.min(q), p.max(q)));
                }
            }
        }
    }
    let mk = |set: &HashSet<(u32, u32)>| -> Vec<(usize, usize, f32)> { set.iter().map(|&(a, b)| (a as usize, b as usize, rest_pos[a as usize].distance(rest_pos[b as usize]))).collect() };
    let stretch = mk(&edges);
    let bendc = mk(&bend);
    let sub = 6;
    let dt = 1.0 / fps.max(1.0) / sub as f32;
    let g = Vec3::new(0.0, 0.0, -cloth.gravity) * dt * dt;
    let iters = 12;
    let ks = 1.0 - (1.0 - cloth.stiffness.clamp(0.0, 1.0)).powf(1.0 / iters as f32);
    let kb = 1.0 - (1.0 - cloth.bending.clamp(0.0, 1.0)).powf(1.0 / iters as f32);
    let inv = obj.inverse();
    let mut out = Vec::with_capacity(frames);
    out.push(pm.verts.clone());
    for _f in 1..frames {
        for _s in 0..sub {
            for i in 0..n {
                if inv_mass[i] == 0.0 {
                    continue;
                }
                let v = (x[i] - prev[i]) * cloth.damping;
                prev[i] = x[i];
                x[i] += v + g;
            }
            for _ in 0..iters {
                for (cons, k) in [(&stretch, ks), (&bendc, kb)] {
                    if k <= 0.0 {
                        continue;
                    }
                    for &(a, b, l0) in cons.iter() {
                        let (wa, wb) = (inv_mass[a], inv_mass[b]);
                        let w = wa + wb;
                        if w == 0.0 {
                            continue;
                        }
                        let d = x[b] - x[a];
                        let l = d.length();
                        if l < 1e-9 {
                            continue;
                        }
                        let corr = d * ((l - l0) / (l * w)) * k;
                        x[a] += corr * wa;
                        x[b] -= corr * wb;
                    }
                }
                for i in 0..n {
                    if inv_mass[i] == 0.0 {
                        continue;
                    }
                    for c in colliders {
                        if let Some(nrm) = c.resolve(&mut x[i], cloth.thickness) {
                            // friction: drop most tangential motion
                            let v = x[i] - prev[i];
                            let vt = v - nrm * v.dot(nrm);
                            prev[i] = x[i] - vt * 0.3;
                        }
                    }
                }
            }
        }
        out.push(x.iter().map(|p| inv.transform_point3(*p)).collect());
    }
    out
}

/// Vertices to pin by a name: "top", "corners", "left" / "right" / "front" / "back" (edges), or none.
pub fn pick_pins(pm: &PolyMesh, obj: Mat4, how: &str) -> Vec<u32> {
    let w: Vec<Vec3> = pm.verts.iter().map(|v| obj.transform_point3(*v)).collect();
    if w.is_empty() {
        return vec![];
    }
    let lo = w.iter().fold(Vec3::splat(f32::MAX), |a, b| a.min(*b));
    let hi = w.iter().fold(Vec3::splat(f32::MIN), |a, b| a.max(*b));
    let size = (hi - lo).max_element().max(1e-6);
    let eps = size * 0.02;
    let h = how.to_lowercase();
    let sel = |f: &dyn Fn(Vec3) -> bool| -> Vec<u32> { (0..w.len()).filter(|i| f(w[*i])).map(|i| i as u32).collect() };
    if h.contains("corner") {
        let corners = [(lo.x, lo.y), (hi.x, lo.y), (lo.x, hi.y), (hi.x, hi.y)];
        let mut out = vec![];
        for (cx, cy) in corners {
            if let Some(i) = (0..w.len()).min_by(|a, b| {
                let da = (w[*a].x - cx).powi(2) + (w[*a].y - cy).powi(2) - w[*a].z * 1e-3;
                let db = (w[*b].x - cx).powi(2) + (w[*b].y - cy).powi(2) - w[*b].z * 1e-3;
                da.partial_cmp(&db).unwrap()
            }) {
                out.push(i as u32);
            }
        }
        out.sort();
        out.dedup();
        out
    } else if h.contains("top") {
        sel(&|p| p.z > hi.z - eps)
    } else if h.contains("left") {
        sel(&|p| p.x < lo.x + eps)
    } else if h.contains("right") {
        sel(&|p| p.x > hi.x - eps)
    } else if h.contains("back") {
        sel(&|p| p.y > hi.y - eps)
    } else if h.contains("front") {
        sel(&|p| p.y < lo.y + eps)
    } else {
        vec![]
    }
}

// ---- rigid bodies ---------------------------------------------------------------------------

/// A rigid body: simulated (active) or an obstacle that other bodies hit (passive).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RigidBody {
    pub passive: bool,
    /// 0..1 bounciness, 0..1 friction.
    pub bounce: f32,
    pub friction: f32,
    /// Baked transforms per frame, from `start`.
    #[serde(skip)]
    pub cache: std::sync::Arc<Vec<crate::scene::Transform>>,
    #[serde(default)]
    pub start: i32,
    /// Where the body starts (saved, so re-simulating after opening a file starts right).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub origin: Option<crate::scene::Transform>,
}

impl Default for RigidBody {
    fn default() -> Self {
        RigidBody { passive: false, bounce: 0.3, friction: 0.5, cache: Default::default(), start: 1, origin: None }
    }
}

/// Input for a rigid-body bake: one active body.
pub struct BodyIn {
    pub transform: crate::scene::Transform,
    /// sphere primitive with uniform scale (simulated as one round particle)
    pub sphere: bool,
    /// local-space bounds of its mesh
    pub bounds: (Vec3, Vec3),
    pub bounce: f32,
    pub friction: f32,
}

/// Rotation nearest to `a` (Müller et al. 2016), warm-started from `q`.
fn extract_rotation(a: glam::Mat3, mut q: glam::Quat) -> glam::Quat {
    for _ in 0..24 {
        let r = glam::Mat3::from_quat(q);
        let omega = r.x_axis.cross(a.x_axis) + r.y_axis.cross(a.y_axis) + r.z_axis.cross(a.z_axis);
        let denom = (r.x_axis.dot(a.x_axis) + r.y_axis.dot(a.y_axis) + r.z_axis.dot(a.z_axis)).abs() + 1e-9;
        let w = omega / denom;
        let angle = w.length();
        if angle < 1e-9 {
            break;
        }
        q = (glam::Quat::from_axis_angle(w / angle, angle) * q).normalize();
    }
    q
}

/// Bake active bodies (particle clusters with shape matching) against static colliders.
/// Returns per body the transform of every frame.
pub fn bake_rigid(bodies: &[BodyIn], statics: &[Collider], frames: usize, fps: f32, gravity: f32) -> Vec<Vec<crate::scene::Transform>> {
    struct Body {
        idx: Vec<usize>,
        rest: Vec<Vec3>,
        c0: Vec3,
        m0: Mat4,
        q: glam::Quat,
        bounce: f32,
        friction: f32,
    }
    let mut x: Vec<Vec3> = vec![];
    let mut rad: Vec<f32> = vec![];
    let mut owner: Vec<usize> = vec![];
    let mut bs: Vec<Body> = vec![];
    for (bi, b) in bodies.iter().enumerate() {
        let m = b.transform.matrix();
        let mut pts = vec![];
        if b.sphere {
            pts.push((m.transform_point3(Vec3::ZERO), b.transform.scale.x.abs().max(1e-3)));
        } else {
            let (lo, hi) = b.bounds;
            let ext = (hi - lo).max(Vec3::splat(1e-4));
            let world_ext = m.transform_vector3(Vec3::X).length() * ext.x;
            let world_ey = m.transform_vector3(Vec3::Y).length() * ext.y;
            let world_ez = m.transform_vector3(Vec3::Z).length() * ext.z;
            let we = Vec3::new(world_ext, world_ey, world_ez);
            // particles across the box, spacing from the smallest side
            let spacing = (we.min_element() / 3.0).max(we.max_element() / 7.0).max(1e-3);
            let n = (we / spacing).ceil().max(Vec3::ONE);
            let (nx, ny, nz) = (n.x as usize, n.y as usize, n.z as usize);
            let r = spacing * 0.5;
            for i in 0..nx {
                for j in 0..ny {
                    for k in 0..nz {
                        let f = Vec3::new((i as f32 + 0.5) / nx as f32, (j as f32 + 0.5) / ny as f32, (k as f32 + 0.5) / nz as f32);
                        // pull the outer particles in so their spheres stay inside the box
                        let local = lo + ext * f;
                        pts.push((m.transform_point3(local), r.min(we.min_element() * 0.5)));
                    }
                }
            }
        }
        let start = x.len();
        let c0 = pts.iter().map(|p| p.0).sum::<Vec3>() / pts.len() as f32;
        let mut rest = vec![];
        for (p, r) in pts {
            x.push(p);
            rad.push(r);
            owner.push(bi);
            rest.push(p - c0);
        }
        bs.push(Body { idx: (start..x.len()).collect(), rest, c0, m0: m, q: glam::Quat::IDENTITY, bounce: b.bounce, friction: b.friction });
    }
    let n = x.len();
    let mut v = vec![Vec3::ZERO; n];
    let sub = 10;
    let dt = 1.0 / fps.max(1.0) / sub as f32;
    let g = Vec3::new(0.0, 0.0, -gravity);
    let rmax = rad.iter().copied().fold(0.01, f32::max);
    let cell = rmax * 2.0;
    let mut out: Vec<Vec<crate::scene::Transform>> = bodies.iter().map(|b| vec![b.transform]).collect();
    for _f in 1..frames {
        for _s in 0..sub {
            let x0 = x.clone();
            for i in 0..n {
                v[i] += g * dt;
                x[i] += v[i] * dt;
            }
            let v_pre = v.clone();
            let mut contact = vec![None::<Vec3>; n];
            for _it in 0..6 {
                // static colliders
                for i in 0..n {
                    for c in statics {
                        let before = x[i];
                        if let Some(nrm) = c.resolve(&mut x[i], rad[i]) {
                            contact[i] = Some(nrm);
                            // position-based friction: cancel sliding up to mu × penetration
                            let pen = (x[i] - before).length();
                            let dp = x[i] - x0[i];
                            let dpt = dp - nrm * dp.dot(nrm);
                            let mu = bs[owner[i]].friction;
                            let l = dpt.length();
                            if l > 1e-9 {
                                x[i] -= dpt * (mu * pen * 2.0 / l).min(1.0);
                            }
                        }
                    }
                }
                // body–body: spatial hash
                let mut grid: std::collections::HashMap<(i32, i32, i32), Vec<usize>> = std::collections::HashMap::new();
                let key = |p: Vec3| ((p.x / cell).floor() as i32, (p.y / cell).floor() as i32, (p.z / cell).floor() as i32);
                for i in 0..n {
                    grid.entry(key(x[i])).or_default().push(i);
                }
                for i in 0..n {
                    let (kx, ky, kz) = key(x[i]);
                    for dx in -1..=1 {
                        for dy in -1..=1 {
                            for dz in -1..=1 {
                                let Some(list) = grid.get(&(kx + dx, ky + dy, kz + dz)) else { continue };
                                for &j in list {
                                    if j <= i || owner[j] == owner[i] {
                                        continue;
                                    }
                                    let d = x[j] - x[i];
                                    let l = d.length();
                                    let min = rad[i] + rad[j];
                                    if l < min && l > 1e-7 {
                                        let nrm = d / l;
                                        let corr = nrm * (min - l) * 0.5;
                                        x[i] -= corr;
                                        x[j] += corr;
                                        // friction between the two bodies
                                        let dp = (x[j] - x0[j]) - (x[i] - x0[i]);
                                        let dpt = dp - nrm * dp.dot(nrm);
                                        let mu = (bs[owner[i]].friction + bs[owner[j]].friction) * 0.5;
                                        let lt = dpt.length();
                                        if lt > 1e-9 {
                                            let k = (mu * (min - l) * 2.0 / lt).min(1.0) * 0.5;
                                            x[i] += dpt * k;
                                            x[j] -= dpt * k;
                                        }
                                        contact[i] = Some(-nrm);
                                        contact[j] = Some(nrm);
                                    }
                                }
                            }
                        }
                    }
                }
                // shape matching: every body stays rigid
                for b in bs.iter_mut() {
                    let c = b.idx.iter().map(|&i| x[i]).sum::<Vec3>() / b.idx.len() as f32;
                    if b.idx.len() > 1 {
                        let mut a = glam::Mat3::ZERO;
                        for (k, &i) in b.idx.iter().enumerate() {
                            let p = x[i] - c;
                            let q = b.rest[k];
                            a += glam::Mat3::from_cols(p * q.x, p * q.y, p * q.z);
                        }
                        b.q = extract_rotation(a, b.q);
                    }
                    let r = glam::Mat3::from_quat(b.q);
                    for (k, &i) in b.idx.iter().enumerate() {
                        x[i] = c + r * b.rest[k];
                    }
                }
            }
            // velocities from positions, with bounce and friction at contacts
            for b in &bs {
                for &i in &b.idx {
                    let mut vel = (x[i] - x0[i]) / dt;
                    if let Some(nrm) = contact[i] {
                        // position corrections must not launch things: the normal speed after a
                        // contact is the bounce of the approach speed (resting contacts stay put)
                        let approach = v_pre[i].dot(nrm);
                        let vt = vel - nrm * vel.dot(nrm);
                        let vn = if approach < -0.3 { -approach * b.bounce } else { vel.dot(nrm).clamp(-0.5, 0.0) };
                        vel = nrm * vn + vt * (1.0 - b.friction * 0.15);
                    }
                    v[i] = vel * 0.998;
                }
            }
            // keep each body's velocity field rigid (average linear + rotation implied by positions)
            for b in &bs {
                let vc = b.idx.iter().map(|&i| v[i]).sum::<Vec3>() / b.idx.len() as f32;
                for &i in &b.idx {
                    v[i] = v[i] * 0.5 + vc * 0.5;
                }
            }
        }
        for (bi, b) in bs.iter().enumerate() {
            let c = b.idx.iter().map(|&i| x[i]).sum::<Vec3>() / b.idx.len() as f32;
            let m = Mat4::from_rotation_translation(b.q, c) * Mat4::from_translation(-b.c0) * b.m0;
            let mut t = crate::scene::Transform::from_matrix(m);
            t.scale = bodies[bi].transform.scale;
            out[bi].push(t);
        }
    }
    out
}

// ---- particles --------------------------------------------------------------------------------

/// A particle system: copies of `source` emitted from the surface of `emitter` over time.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Particles {
    pub source: u64,
    pub emitter: u64,
    /// New particles per frame.
    pub rate: f32,
    /// Life in frames.
    pub lifetime: u32,
    /// Launch speed along the emitter's normal (m/s) and random spread (0..1).
    pub speed: f32,
    pub spread: f32,
    pub gravity: f32,
    /// Velocity kept per second (air drag; 1 = none).
    pub drag: f32,
    pub size: f32,
    /// Shrink to nothing at the end of life.
    pub shrink: bool,
    /// Bounce off the other meshes.
    pub collide: bool,
    pub seed: u32,
    /// Frames simulated before the start, so it is already going at frame 1 (weather).
    #[serde(default)]
    pub preroll: u32,
    /// Baked per frame: (position, size) of each living particle, world space, from `start`.
    #[serde(skip)]
    pub cache: std::sync::Arc<Vec<Vec<(Vec3, glam::Quat, f32)>>>,
    #[serde(default)]
    pub start: i32,
}

impl Particles {
    pub fn new(source: u64, emitter: u64) -> Particles {
        Particles { source, emitter, rate: 10.0, lifetime: 48, speed: 2.0, spread: 0.3, gravity: 9.81, drag: 0.9, size: 1.0, shrink: true, collide: true, seed: 1, preroll: 0, cache: Default::default(), start: 1 }
    }
}

/// Simulate a particle system for `frames` frames.
pub fn bake_particles(p: &Particles, emitter: &PolyMesh, emitter_world: Mat4, colliders: &[Collider], frames: usize, fps: f32) -> Vec<Vec<(Vec3, glam::Quat, f32)>> {
    struct P {
        x: Vec3,
        v: Vec3,
        rot: glam::Quat,
        spin: Vec3,
        age: f32,
        life: f32,
        size: f32,
    }
    let mut rng = {
        let mut st = (p.seed as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        move || {
            st ^= st << 13;
            st ^= st >> 7;
            st ^= st << 17;
            (st >> 40) as f32 / (1u64 << 24) as f32
        }
    };
    let dt = 1.0 / fps.max(1.0);
    let drag = p.drag.clamp(0.0, 1.0).powf(dt);
    let g = Vec3::new(0.0, 0.0, -p.gravity);
    let mut alive: Vec<P> = vec![];
    let mut out = Vec::with_capacity(frames);
    let mut carry = 0.0f32;
    let pre = p.preroll.min(2000) as usize;
    for f in 0..frames + pre {
        // emit (spawn points sampled on the surface, area-weighted)
        carry += p.rate.max(0.0);
        let n = carry.floor() as usize;
        carry -= n as f32;
        if n > 0 {
            let pts = crate::scatter::sample_surface(emitter, emitter_world, n, p.seed.wrapping_add(f as u32 * 7919), 180.0);
            for (pos, nrm) in pts {
                let rand_dir = Vec3::new(rng() - 0.5, rng() - 0.5, rng() - 0.5) * 2.0;
                let dir = (nrm + rand_dir * p.spread * 2.0).normalize_or(nrm);
                let sub = rng(); // spread spawns over the frame
                let mut x = pos;
                let v = dir * p.speed * (0.75 + 0.5 * rng());
                x += v * dt * sub;
                alive.push(P {
                    x,
                    v,
                    rot: glam::Quat::from_euler(glam::EulerRot::XYZ, rng() * 6.283, rng() * 6.283, rng() * 6.283),
                    spin: Vec3::new(rng() - 0.5, rng() - 0.5, rng() - 0.5) * 6.0,
                    age: 0.0,
                    life: p.lifetime.max(1) as f32 * (0.8 + 0.4 * rng()),
                    size: p.size * (0.7 + 0.6 * rng()),
                });
            }
        }
        // move
        for q in alive.iter_mut() {
            q.v = (q.v + g * dt) * drag;
            let before = q.x;
            q.x += q.v * dt;
            if p.collide {
                for c in colliders {
                    if let Some((hit, nrm)) = c.sweep(before, q.x) {
                        q.x = hit;
                        let vn = q.v.dot(nrm);
                        if vn < 0.0 {
                            q.v -= nrm * vn * 1.5;
                            q.v *= 0.7;
                        }
                        continue;
                    }
                    if let Some(nrm) = c.resolve(&mut q.x, 0.0) {
                        let vn = q.v.dot(nrm);
                        if vn < 0.0 {
                            q.v -= nrm * vn * 1.5; // bounce a little
                            q.v *= 0.7;
                        }
                        let _ = before;
                    }
                }
            }
            let w = q.spin * dt;
            if w.length() > 1e-6 {
                q.rot = (glam::Quat::from_axis_angle(w.normalize(), w.length()) * q.rot).normalize();
            }
            q.age += 1.0;
        }
        alive.retain(|q| q.age < q.life);
        if f < pre {
            continue;
        }
        out.push(
            alive
                .iter()
                .map(|q| {
                    let k = if p.shrink { (1.0 - q.age / q.life).clamp(0.0, 1.0).sqrt() } else { 1.0 };
                    (q.x, q.rot, q.size * k)
                })
                .collect(),
        );
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cloth_drapes_over_sphere() {
        // a 2x2 grid cloth above a unit sphere
        let mut pm = crate::mesh::Primitive::Plane.poly(1);
        crate::sculpt::densify(&mut pm, 400, false);
        let obj = Mat4::from_scale_rotation_translation(Vec3::splat(1.5), glam::Quat::IDENTITY, Vec3::new(0.0, 0.0, 1.5));
        let cloth = Cloth::default();
        let col = [Collider::Sphere { c: Vec3::ZERO, r: 1.0 }];
        let frames = bake(&pm, obj, &cloth, &col, 48, 24.0);
        assert_eq!(frames.len(), 48);
        let last: Vec<Vec3> = frames[47].iter().map(|v| obj.transform_point3(*v)).collect();
        // nothing went through the sphere, the middle rests on top, the edges hang lower
        assert!(last.iter().all(|p| p.length() > 0.98), "penetration");
        let centre = last.iter().min_by(|a, b| (a.x * a.x + a.y * a.y).partial_cmp(&(b.x * b.x + b.y * b.y)).unwrap()).unwrap();
        assert!((centre.z - 1.02).abs() < 0.08, "rests on the sphere: {centre}");
        let lowest = last.iter().map(|p| p.z).fold(f32::MAX, f32::min);
        assert!(lowest < 0.6, "edges drape down: {lowest}");
        // pinned cloth: the pins never move
        let pins = pick_pins(&pm, obj, "corners");
        assert_eq!(pins.len(), 4);
        let c2 = Cloth { pins: pins.clone(), ..Cloth::default() };
        let fr = bake(&pm, obj, &c2, &[], 24, 24.0);
        for p in pins {
            assert!(fr[23][p as usize].distance(pm.verts[p as usize]) < 1e-5);
        }
    }
    #[test]
    fn rigid_bodies_fall_and_stack() {
        let t = |z: f32| crate::scene::Transform { location: Vec3::new(0.0, 0.0, z), rotation: Vec3::new(20.0, 10.0, 0.0), scale: Vec3::splat(0.5) };
        let cube = (Vec3::splat(-1.0), Vec3::splat(1.0));
        let bodies = vec![
            BodyIn { transform: t(2.0), sphere: false, bounds: cube, bounce: 0.1, friction: 0.6 },
            BodyIn { transform: t(4.0), sphere: false, bounds: cube, bounce: 0.1, friction: 0.6 },
        ];
        // a floor: a thin box at z = 0
        let floor = Collider::Box { to_local: Mat4::IDENTITY, to_world: Mat4::IDENTITY, half: Vec3::new(10.0, 10.0, 0.001) };
        let out = bake_rigid(&bodies, &[floor], 96, 24.0, 9.81);
        let (a, b) = (out[0].last().unwrap(), out[1].last().unwrap());
        assert!(a.location.z > 0.3 && a.location.z < 0.8, "first box rests on the floor: {:?}", a.location);
        assert!(b.location.z > a.location.z + 0.6, "second box lands on top: {:?} vs {:?}", b.location, a.location);
        // it settled: nearly still at the end
        let p = out[1][out[1].len() - 2].location;
        assert!(p.distance(b.location) < 0.02);
    }
    #[test]
    fn particles_emit_fall_and_die() {
        let plane = crate::mesh::Primitive::Plane.poly(1);
        let w = Mat4::from_translation(Vec3::new(0.0, 0.0, 3.0));
        let mut p = Particles::new(1, 2);
        p.rate = 20.0;
        p.lifetime = 24;
        p.speed = 0.0;
        let floor = Collider::Box { to_local: Mat4::IDENTITY, to_world: Mat4::IDENTITY, half: Vec3::new(10.0, 10.0, 0.001) };
        let fr = bake_particles(&p, &plane, w, &[floor], 60, 24.0);
        assert_eq!(fr.len(), 60);
        assert!(fr[0].len() >= 19 && fr[0].len() <= 20);
        // steady state: about rate × lifetime alive, nothing below the floor
        let n = fr[59].len();
        assert!(n > 350 && n < 600, "{n}");
        assert!(fr.iter().flatten().all(|(x, _, _)| x.z >= -1e-3));
        // falling
        let z0 = fr[2].iter().map(|q| q.0.z).sum::<f32>() / fr[2].len() as f32;
        let z1 = fr[20].iter().map(|q| q.0.z).sum::<f32>() / fr[20].len() as f32;
        assert!(z1 < z0);
    }
}
