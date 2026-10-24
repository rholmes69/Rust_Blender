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
}
