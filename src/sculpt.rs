//! Sculpt mode brushes working directly on a `PolyMesh` (local object space).

use crate::editmesh::falloff;
use crate::mesh::PolyMesh;
use glam::Vec3;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Brush {
    /// Push the surface out along its average normal (Ctrl: in).
    Draw,
    /// Inflate along each vertex normal.
    Inflate,
    /// Relax towards the neighbours.
    Smooth,
    /// Move a region with the mouse.
    Grab,
    /// Pinch towards the stroke and cut in: sharp creases.
    Crease,
    /// Flatten towards the average plane.
    Flatten,
}

impl Brush {
    pub const ALL: [Brush; 6] = [Brush::Draw, Brush::Inflate, Brush::Smooth, Brush::Grab, Brush::Crease, Brush::Flatten];
    pub fn label(self) -> &'static str {
        match self {
            Brush::Draw => "Draw",
            Brush::Inflate => "Inflate",
            Brush::Smooth => "Smooth",
            Brush::Grab => "Grab",
            Brush::Crease => "Crease",
            Brush::Flatten => "Flatten",
        }
    }
    pub fn key(self) -> &'static str {
        match self {
            Brush::Draw => "X",
            Brush::Inflate => "I",
            Brush::Smooth => "S",
            Brush::Grab => "G",
            Brush::Crease => "C",
            Brush::Flatten => "T",
        }
    }
}

/// Vertex neighbours (from face edges).
pub fn adjacency(pm: &PolyMesh) -> Vec<Vec<u32>> {
    let mut adj: Vec<Vec<u32>> = vec![vec![]; pm.verts.len()];
    for f in &pm.faces {
        for i in 0..f.len() {
            let (a, b) = (f[i], f[(i + 1) % f.len()]);
            if !adj[a as usize].contains(&b) {
                adj[a as usize].push(b);
            }
            if !adj[b as usize].contains(&a) {
                adj[b as usize].push(a);
            }
        }
    }
    adj
}

pub fn vertex_normals(pm: &PolyMesh) -> Vec<Vec3> {
    let mut n = vec![Vec3::ZERO; pm.verts.len()];
    for f in &pm.faces {
        let p0 = pm.verts[f[0] as usize];
        let mut fnrm = Vec3::ZERO;
        for k in 1..f.len().saturating_sub(1) {
            fnrm += (pm.verts[f[k] as usize] - p0).cross(pm.verts[f[k + 1] as usize] - p0);
        }
        for &v in f {
            n[v as usize] += fnrm;
        }
    }
    n.into_iter().map(|v| v.normalize_or_zero()).collect()
}

/// Closest hit of a (local-space) ray with the mesh: (distance, point, face normal).
pub fn ray_hit(pm: &PolyMesh, o: Vec3, d: Vec3) -> Option<(f32, Vec3, Vec3)> {
    let mut best: Option<(f32, Vec3, Vec3)> = None;
    for f in &pm.faces {
        let a = pm.verts[f[0] as usize];
        for k in 1..f.len().saturating_sub(1) {
            let (b, c) = (pm.verts[f[k] as usize], pm.verts[f[k + 1] as usize]);
            let e1 = b - a;
            let e2 = c - a;
            let p = d.cross(e2);
            let det = e1.dot(p);
            if det.abs() < 1e-12 {
                continue;
            }
            let inv = 1.0 / det;
            let s = o - a;
            let u = s.dot(p) * inv;
            if !(0.0..=1.0).contains(&u) {
                continue;
            }
            let q = s.cross(e1);
            let v = d.dot(q) * inv;
            if v < 0.0 || u + v > 1.0 {
                continue;
            }
            let t = e2.dot(q) * inv;
            if t > 1e-5 && best.is_none_or(|b| t < b.0) {
                best = Some((t, o + d * t, e1.cross(e2).normalize_or_zero()));
            }
        }
    }
    best
}

/// One brush dab at `c` (local space) with radius `r` and strength 0..1.
/// `invert` flips Draw/Inflate/Crease. `along` is used as the push direction when the brush
/// region has no clear normal.
pub fn dab(pm: &mut PolyMesh, adj: &[Vec<u32>], c: Vec3, r: f32, brush: Brush, strength: f32, invert: bool, along: Vec3) {
    if r <= 0.0 || brush == Brush::Grab {
        return;
    }
    let normals = vertex_normals(pm);
    let r2 = r * r;
    let inside: Vec<(usize, f32)> = pm
        .verts
        .iter()
        .enumerate()
        .filter_map(|(i, v)| {
            let d2 = v.distance_squared(c);
            (d2 < r2).then(|| (i, falloff(d2.sqrt(), r)))
        })
        .collect();
    if inside.is_empty() {
        return;
    }
    let sign = if invert { -1.0 } else { 1.0 };
    let mut avg_n = Vec3::ZERO;
    let mut avg_p = Vec3::ZERO;
    let mut wsum = 0.0;
    for (i, w) in &inside {
        avg_n += normals[*i] * *w;
        avg_p += pm.verts[*i] * *w;
        wsum += *w;
    }
    let avg_n = avg_n.try_normalize().unwrap_or(along);
    let avg_p = avg_p / wsum.max(1e-6);
    let step = r * 0.12 * strength;
    let old = pm.verts.clone();
    for (i, w) in inside {
        let v = old[i];
        pm.verts[i] = match brush {
            Brush::Draw => v + avg_n * step * w * sign,
            Brush::Inflate => v + normals[i] * step * w * sign,
            Brush::Smooth => {
                let nb = &adj[i];
                if nb.is_empty() {
                    v
                } else {
                    let m = nb.iter().map(|&j| old[j as usize]).sum::<Vec3>() / nb.len() as f32;
                    v.lerp(m, (strength * w * 0.8).min(1.0))
                }
            }
            Brush::Crease => {
                let to_c = c - v;
                let tangential = to_c - avg_n * to_c.dot(avg_n);
                v + tangential * (0.25 * strength * w) - avg_n * step * 0.6 * w * sign
            }
            Brush::Flatten => v - avg_n * (v - avg_p).dot(avg_n) * (strength * w * 0.5),
            Brush::Grab => v,
        };
    }
}

/// Vertices a grab stroke moves, with their weights.
pub fn grab_set(pm: &PolyMesh, c: Vec3, r: f32) -> Vec<(u32, Vec3, f32)> {
    pm.verts
        .iter()
        .enumerate()
        .filter_map(|(i, v)| {
            let d = v.distance(c);
            (d < r).then(|| (i as u32, *v, falloff(d, r)))
        })
        .collect()
}

/// Make a mesh dense enough to sculpt (keeps its shape), up to about `target` vertices.
/// Smooth shapes are refined with Catmull–Clark (rounder), hard ones linearly (keeps corners).
pub fn densify(pm: &mut PolyMesh, target: usize, smooth: bool) -> bool {
    let mut changed = false;
    for _ in 0..4 {
        if pm.verts.len() >= target {
            break;
        }
        if smooth {
            *pm = crate::editmesh::catmull_clark(pm);
        } else {
            let mut sel = vec![true; pm.verts.len()];
            if crate::editmesh::subdivide(pm, &mut sel).is_err() {
                break;
            }
            pm.weld(1e-6);
        }
        changed = true;
    }
    changed
}

/// Dynamic topology: turn the mesh into triangles (UVs are dropped).
pub fn to_triangles(pm: &mut PolyMesh) {
    let mut faces = Vec::with_capacity(pm.faces.len() * 2);
    for f in &pm.faces {
        if f.len() == 3 {
            faces.push(f.clone());
        } else {
            faces.extend(crate::mesh::triangulate(&pm.verts, f).into_iter().map(|t| t.to_vec()));
        }
    }
    pm.faces = faces;
    pm.uvs.clear();
}

/// Dynamic topology refinement around `c`: split triangle edges longer than `max_len` that touch
/// the brush (radius `r`), so detail appears where you sculpt. Returns whether anything changed.
pub fn refine(pm: &mut PolyMesh, c: Vec3, r: f32, max_len: f32) -> bool {
    if max_len <= 0.0 {
        return false;
    }
    let r2 = r * r;
    let reach = (r + max_len * 2.0).powi(2);
    let mut changed = false;
    for _pass in 0..8 {
        // triangles near the brush
        let cand: Vec<usize> = (0..pm.faces.len())
            .filter(|&fi| pm.faces[fi].len() == 3 && pm.faces[fi].iter().any(|&v| pm.verts[v as usize].distance_squared(c) < reach))
            .collect();
        let key = |a: u32, b: u32| (a.min(b), a.max(b));
        let mut edge_faces: std::collections::HashMap<(u32, u32), Vec<usize>> = std::collections::HashMap::new();
        for &fi in &cand {
            let f = &pm.faces[fi];
            for i in 0..3 {
                edge_faces.entry(key(f[i], f[(i + 1) % 3])).or_default().push(fi);
            }
        }
        let ml2 = max_len * max_len;
        let mut edges: Vec<((u32, u32), f32)> = edge_faces
            .keys()
            .filter_map(|&(a, b)| {
                let (pa, pb) = (pm.verts[a as usize], pm.verts[b as usize]);
                let l2 = pa.distance_squared(pb);
                (l2 > ml2 && (pa.distance_squared(c) < r2 || pb.distance_squared(c) < r2 || ((pa + pb) * 0.5).distance_squared(c) < r2)).then_some(((a, b), l2))
            })
            .collect();
        if edges.is_empty() {
            break;
        }
        edges.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap_or(std::cmp::Ordering::Equal));
        let mut touched = vec![false; pm.faces.len()];
        let mut split_any = false;
        for ((a, b), _) in edges {
            let fs = &edge_faces[&(a, b)];
            // an edge is split only if none of its faces changed in this pass (keeps it conforming)
            if fs.iter().any(|&fi| touched[fi]) {
                continue;
            }
            let m = (pm.verts[a as usize] + pm.verts[b as usize]) * 0.5;
            pm.verts.push(m);
            let mv = (pm.verts.len() - 1) as u32;
            for &fi in fs.clone().iter() {
                let f = pm.faces[fi].clone();
                let i = (0..3).find(|&i| key(f[i], f[(i + 1) % 3]) == (a, b)).unwrap();
                let (p, q, x) = (f[i], f[(i + 1) % 3], f[(i + 2) % 3]);
                pm.faces[fi] = vec![p, mv, x];
                pm.faces.push(vec![mv, q, x]);
                touched[fi] = true;
                touched.push(true);
            }
            split_any = true;
        }
        if !split_any {
            break;
        }
        changed = true;
    }
    changed
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn brushes() {
        let mut pm = crate::mesh::cube();
        assert!(densify(&mut pm, 1000, false));
        assert!(pm.verts.len() >= 1000);
        let adj = adjacency(&pm);
        // ray from above hits the top face
        let (t, p, n) = ray_hit(&pm, Vec3::new(0.1, 0.1, 5.0), -Vec3::Z).unwrap();
        assert!((t - 4.0).abs() < 1e-4 && (p.z - 1.0).abs() < 1e-4 && n.z > 0.9);
        let top = |pm: &PolyMesh| pm.verts.iter().map(|v| v.z).fold(f32::MIN, f32::max);
        let before = top(&pm);
        dab(&mut pm, &adj, p, 0.5, Brush::Draw, 1.0, false, Vec3::Z);
        assert!(top(&pm) > before, "draw raises the surface");
        let raised = top(&pm);
        for _ in 0..20 {
            dab(&mut pm, &adj, p, 0.6, Brush::Smooth, 1.0, false, Vec3::Z);
        }
        assert!(top(&pm) < raised, "smooth relaxes the bump");
        dab(&mut pm, &adj, p, 0.5, Brush::Draw, 1.0, true, Vec3::Z);
        // far vertices never move
        let far = pm.verts.iter().filter(|v| v.z < -0.99).count();
        assert!(far > 100);
        let g = grab_set(&pm, p, 0.5);
        assert!(!g.is_empty() && g.iter().all(|(_, _, w)| *w >= 0.0 && *w <= 1.0));
    }
}
#[cfg(test)]
mod densify_tests {
    #[test]
    fn densify_sphere_valid() {
        let mut pm = crate::mesh::Primitive::UvSphere.poly(2);
        let n0 = pm.verts.len();
        super::densify(&mut pm, 6000, true);
        let c = pm.verts.iter().sum::<glam::Vec3>() / pm.verts.len() as f32;
        let bad = (0..pm.faces.len()).filter(|&f| pm.face_normal(f).dot(pm.face_center(f) - c) <= 0.0).count();
        let open = pm.edge_faces().values().filter(|f| f.len() != 2).count();
        let m = pm.to_mesh(true);
        assert!(pm.verts.len() > n0 && pm.verts.len() >= 4000);
        assert_eq!((bad, open), (0, 0), "closed and outward");
        assert!(m.tris.len() > 1000 && pm.verts.iter().all(|v| v.is_finite()));
    }
    #[test]
    fn dyntopo_refines_locally() {
        let mut pm = crate::mesh::Primitive::IcoSphere.poly(1);
        super::to_triangles(&mut pm);
        let (v0, f0) = (pm.verts.len(), pm.faces.len());
        let c = pm.verts[0];
        assert!(super::refine(&mut pm, c, 0.4, 0.1));
        assert!(pm.verts.len() > v0 + 10 && pm.faces.len() > f0, "{} -> {}", v0, pm.verts.len());
        // still closed, and far away nothing changed
        assert!(pm.edge_faces().values().all(|f| f.len() == 2), "watertight");
        let far = pm.verts.iter().filter(|v| v.distance(-c) < 0.3).count();
        assert!(far < 10);
    }
}
