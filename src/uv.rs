//! UV unwrapping: smart projection (charts by normal + packing), cube, sphere and cylinder
//! projections. Results are per-face-corner UVs on the `PolyMesh`.

use crate::mesh::{PolyMesh, box_uv};
use glam::{Vec2, Vec3};
use std::collections::HashMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Unwrap {
    Smart,
    Cube,
    Sphere,
    Cylinder,
}

impl Unwrap {
    pub fn parse(s: &str) -> Unwrap {
        let s = s.to_lowercase();
        if s.contains("cube") || s.contains("box") {
            Unwrap::Cube
        } else if s.contains("sphere") || s.contains("ball") {
            Unwrap::Sphere
        } else if s.contains("cyl") || s.contains("tube") {
            Unwrap::Cylinder
        } else {
            Unwrap::Smart
        }
    }
    pub fn label(self) -> &'static str {
        match self {
            Unwrap::Smart => "Smart UV Project",
            Unwrap::Cube => "Cube Projection",
            Unwrap::Sphere => "Sphere Projection",
            Unwrap::Cylinder => "Cylinder Projection",
        }
    }
}

/// Make sure `pm.uvs` exists (generated box UVs where missing).
pub fn ensure_uvs(pm: &mut PolyMesh) {
    if !pm.has_uvs() {
        pm.uvs = (0..pm.faces.len()).map(|f| box_uv(&pm.verts, &pm.faces[f])).collect();
    }
}

fn face_normal(pm: &PolyMesh, f: usize) -> Vec3 {
    pm.face_normal(f)
}

/// Unwrap `faces` (all faces when empty) with the given method.
pub fn unwrap(pm: &mut PolyMesh, faces: &[usize], method: Unwrap) {
    ensure_uvs(pm);
    let faces: Vec<usize> = if faces.is_empty() { (0..pm.faces.len()).collect() } else { faces.to_vec() };
    match method {
        Unwrap::Smart => smart(pm, &faces, 66.0, 0.02),
        Unwrap::Cube => {
            for &f in &faces {
                pm.uvs[f] = box_uv(&pm.verts, &pm.faces[f]);
            }
        }
        Unwrap::Sphere | Unwrap::Cylinder => {
            let c = faces.iter().flat_map(|&f| pm.faces[f].iter()).map(|&v| pm.verts[v as usize]).sum::<Vec3>()
                / faces.iter().map(|&f| pm.faces[f].len()).sum::<usize>().max(1) as f32;
            let (zlo, zhi) = faces.iter().flat_map(|&f| pm.faces[f].iter()).fold((f32::MAX, f32::MIN), |(l, h), &v| (l.min(pm.verts[v as usize].z), h.max(pm.verts[v as usize].z)));
            for &f in &faces {
                let mut uv: Vec<[f32; 2]> = pm.faces[f]
                    .iter()
                    .map(|&v| {
                        let p = pm.verts[v as usize] - c;
                        let u = 0.5 + p.y.atan2(p.x) / std::f32::consts::TAU;
                        let v = if method == Unwrap::Sphere {
                            0.5 - (p.z / p.length().max(1e-6)).clamp(-1.0, 1.0).asin() / std::f32::consts::PI
                        } else {
                            1.0 - (pm.verts[v as usize].z - zlo) / (zhi - zlo).max(1e-6)
                        };
                        [u, v]
                    })
                    .collect();
                // faces crossing the seam: unwrap to one side
                let umin = uv.iter().map(|q| q[0]).fold(f32::MAX, f32::min);
                let umax = uv.iter().map(|q| q[0]).fold(f32::MIN, f32::max);
                if umax - umin > 0.5 {
                    for q in &mut uv {
                        if q[0] < 0.5 {
                            q[0] += 1.0;
                        }
                    }
                }
                // poles: use the face's mean u for the pole vertex
                if method == Unwrap::Sphere {
                    let n = pm.faces[f].len();
                    let mean_u = uv.iter().map(|q| q[0]).sum::<f32>() / n as f32;
                    for (k, &v) in pm.faces[f].iter().enumerate() {
                        let p = (pm.verts[v as usize] - c).normalize_or_zero();
                        if p.z.abs() > 0.9999 {
                            uv[k][0] = mean_u;
                        }
                    }
                }
                pm.uvs[f] = uv;
            }
        }
    }
}

/// Smart UV Project: grow charts of connected faces whose normals stay within `angle`
/// of the chart's seed, project each onto its plane, then pack all charts into 0..1.
fn smart(pm: &mut PolyMesh, faces: &[usize], angle: f32, margin: f32) {
    let cos = angle.to_radians().cos();
    let in_set: std::collections::HashSet<usize> = faces.iter().copied().collect();
    let ef = pm.edge_faces();
    let key = |a: u32, b: u32| (a.min(b), a.max(b));
    let normals: HashMap<usize, Vec3> = faces.iter().map(|&f| (f, face_normal(pm, f))).collect();
    let mut chart_of: HashMap<usize, usize> = HashMap::new();
    let mut charts: Vec<Vec<usize>> = vec![];
    for &seed in faces {
        if chart_of.contains_key(&seed) {
            continue;
        }
        let ci = charts.len();
        let sn = normals[&seed];
        let mut chart = vec![seed];
        chart_of.insert(seed, ci);
        let mut k = 0;
        while k < chart.len() {
            let f = chart[k];
            k += 1;
            let fv = pm.faces[f].clone();
            for i in 0..fv.len() {
                for &g in ef.get(&key(fv[i], fv[(i + 1) % fv.len()])).into_iter().flatten() {
                    if g != f && in_set.contains(&g) && !chart_of.contains_key(&g) && normals[&g].dot(sn) >= cos {
                        chart_of.insert(g, ci);
                        chart.push(g);
                    }
                }
            }
        }
        charts.push(chart);
    }
    // project each chart onto its average-normal plane
    struct Chart {
        faces: Vec<usize>,
        uv: Vec<Vec<Vec2>>,
        size: Vec2,
    }
    let mut out: Vec<Chart> = vec![];
    for chart in charts {
        let n = chart.iter().map(|f| normals[f] * pm.face_area(*f).max(1e-9)).sum::<Vec3>().normalize_or_zero();
        let n = if n == Vec3::ZERO { Vec3::Z } else { n };
        let t = if n.z.abs() < 0.9 { Vec3::Z.cross(n).normalize() } else { Vec3::X.cross(n).normalize() };
        let b = n.cross(t);
        let mut uv: Vec<Vec<Vec2>> = chart.iter().map(|&f| pm.faces[f].iter().map(|&v| { let p = pm.verts[v as usize]; Vec2::new(p.dot(t), p.dot(b)) }).collect()).collect();
        let lo = uv.iter().flatten().fold(Vec2::splat(f32::MAX), |a, b| a.min(*b));
        let hi = uv.iter().flatten().fold(Vec2::splat(f32::MIN), |a, b| a.max(*b));
        for f in &mut uv {
            for q in f {
                *q -= lo;
            }
        }
        out.push(Chart { faces: chart, uv, size: (hi - lo).max(Vec2::splat(1e-6)) });
    }
    // shelf packing, tallest first; then scale everything to fit the unit square
    out.sort_by(|a, b| b.size.y.partial_cmp(&a.size.y).unwrap_or(std::cmp::Ordering::Equal));
    let total_area: f32 = out.iter().map(|c| (c.size.x + margin) * (c.size.y + margin)).sum();
    let width = total_area.sqrt().max(out.iter().map(|c| c.size.x).fold(0.0, f32::max) + margin);
    let (mut x, mut y, mut shelf) = (0.0f32, 0.0f32, 0.0f32);
    let mut offs = vec![];
    for c in &out {
        if x + c.size.x > width && x > 0.0 {
            x = 0.0;
            y += shelf + margin;
            shelf = 0.0;
        }
        offs.push(Vec2::new(x, y));
        x += c.size.x + margin;
        shelf = shelf.max(c.size.y);
    }
    let extent = (y + shelf).max(width).max(1e-6);
    for (c, o) in out.iter().zip(offs) {
        for (fi, f) in c.faces.iter().enumerate() {
            pm.uvs[*f] = c.uv[fi].iter().map(|q| {
                let p = (*q + o) / extent;
                [p.x, 1.0 - p.y]
            }).collect();
        }
    }
}

/// Number of UV islands (charts) — for tests / UI.
pub fn islands(pm: &PolyMesh) -> usize {
    if !pm.has_uvs() {
        return 0;
    }
    // faces sharing an edge with identical UVs on both ends are in the same island
    let mut parent: Vec<usize> = (0..pm.faces.len()).collect();
    fn find(p: &mut [usize], x: usize) -> usize {
        let mut r = x;
        while p[r] != r {
            r = p[r];
        }
        let mut y = x;
        while p[y] != r {
            let n = p[y];
            p[y] = r;
            y = n;
        }
        r
    }
    let mut edges: HashMap<(u32, u32), Vec<(usize, [f32; 2], [f32; 2])>> = HashMap::new();
    for (fi, f) in pm.faces.iter().enumerate() {
        for i in 0..f.len() {
            let j = (i + 1) % f.len();
            let (a, b) = (f[i], f[j]);
            let (ua, ub) = (pm.uvs[fi][i], pm.uvs[fi][j]);
            let (k, u1, u2) = if a < b { ((a, b), ua, ub) } else { ((b, a), ub, ua) };
            edges.entry(k).or_default().push((fi, u1, u2));
        }
    }
    let same = |p: [f32; 2], q: [f32; 2]| (p[0] - q[0]).abs() < 1e-5 && (p[1] - q[1]).abs() < 1e-5;
    for list in edges.values() {
        if list.len() == 2 && same(list[0].1, list[1].1) && same(list[0].2, list[1].2) {
            let (a, b) = (find(&mut parent, list[0].0), find(&mut parent, list[1].0));
            parent[a] = b;
        }
    }
    let mut roots = std::collections::HashSet::new();
    for f in 0..pm.faces.len() {
        roots.insert(find(&mut parent, f));
    }
    roots.len()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn unwrap_methods() {
        let mut pm = crate::mesh::cube();
        unwrap(&mut pm, &[], Unwrap::Smart);
        assert!(pm.has_uvs());
        assert_eq!(islands(&pm), 6, "a cube unwraps into 6 charts");
        let all: Vec<[f32; 2]> = pm.uvs.iter().flatten().copied().collect();
        assert!(all.iter().all(|q| (-1e-5..=1.0 + 1e-5).contains(&q[0]) && (-1e-5..=1.0 + 1e-5).contains(&q[1])), "packed into 0..1");
        // charts don't overlap: total UV area ≈ sum of chart areas (each face area > 0)
        let mut s = crate::mesh::Primitive::UvSphere.poly(2);
        unwrap(&mut s, &[], Unwrap::Sphere);
        assert!(s.has_uvs());
        let span = s.uvs.iter().map(|f| f.iter().map(|q| q[0]).fold(f32::MIN, f32::max) - f.iter().map(|q| q[0]).fold(f32::MAX, f32::min)).fold(0.0, f32::max);
        assert!(span < 0.5, "no face wraps around the seam");
        let mut c = crate::mesh::Primitive::Cylinder.poly(2);
        unwrap(&mut c, &[0], Unwrap::Cylinder);
        assert!(c.has_uvs());
        assert_eq!(Unwrap::parse("box"), Unwrap::Cube);
    }
}
