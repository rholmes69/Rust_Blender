//! Procedural mesh primitives (Z-up, like Blender).

use glam::Vec3;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::f32::consts::{PI, TAU};

#[derive(Clone, Debug, Default)]
pub struct Mesh {
    pub positions: Vec<Vec3>,
    pub tris: Vec<[u32; 3]>,
    /// Per-vertex smooth normals (same length as positions).
    pub normals: Vec<Vec3>,
    /// Unique edges, each with the (up to two) triangles that share it.
    pub edges: Vec<Edge>,
    pub smooth: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct Edge {
    pub a: u32,
    pub b: u32,
    pub faces: [u32; 2], // u32::MAX = boundary
    /// "Hard" edges are drawn in wireframe mode (quad borders, not diagonals).
    pub feature: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Primitive {
    Cube,
    Plane,
    UvSphere,
    IcoSphere,
    Cylinder,
    Cone,
    Torus,
    Monkey,
}

impl Primitive {
    pub const ALL: [Primitive; 8] = [
        Primitive::Cube,
        Primitive::Plane,
        Primitive::UvSphere,
        Primitive::IcoSphere,
        Primitive::Cylinder,
        Primitive::Cone,
        Primitive::Torus,
        Primitive::Monkey,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Primitive::Cube => "Cube",
            Primitive::Plane => "Plane",
            Primitive::UvSphere => "UV Sphere",
            Primitive::IcoSphere => "Ico Sphere",
            Primitive::Cylinder => "Cylinder",
            Primitive::Cone => "Cone",
            Primitive::Torus => "Torus",
            Primitive::Monkey => "Blob",
        }
    }

    pub fn icon(self) -> &'static str {
        match self {
            Primitive::Cube => "■",
            Primitive::Plane => "▱",
            Primitive::UvSphere => "●",
            Primitive::IcoSphere => "⬢",
            Primitive::Cylinder => "▮",
            Primitive::Cone => "▲",
            Primitive::Torus => "◎",
            Primitive::Monkey => "✿",
        }
    }

    /// Lenient parser used by the AI / voice layer.
    pub fn parse(s: &str) -> Option<Primitive> {
        let s = s.to_lowercase().replace(['_', '-', ' '], "");
        Some(match s.as_str() {
            "cube" | "box" | "block" | "crate" => Primitive::Cube,
            "plane" | "floor" | "ground" | "grid" | "square" => Primitive::Plane,
            "sphere" | "uvsphere" | "ball" | "orb" | "globe" => Primitive::UvSphere,
            "icosphere" | "ico" | "icosahedron" | "geosphere" => Primitive::IcoSphere,
            "cylinder" | "tube" | "pillar" | "column" | "can" => Primitive::Cylinder,
            "cone" | "pyramid" | "spike" => Primitive::Cone,
            "torus" | "donut" | "doughnut" | "ring" => Primitive::Torus,
            "monkey" | "suzanne" | "blob" | "organic" => Primitive::Monkey,
            _ => return None,
        })
    }

    pub fn build(self, detail: u32) -> Mesh {
        let d = detail.clamp(1, 6);
        match self {
            Primitive::Cube => cube(),
            Primitive::Plane => plane(),
            Primitive::UvSphere => uv_sphere(8 * d + 8, 4 * d + 4),
            Primitive::IcoSphere => ico_sphere(d.min(5)),
            Primitive::Cylinder => cylinder(8 * d + 8, 1.0, 1.0),
            Primitive::Cone => cylinder(8 * d + 8, 1.0, 0.0),
            Primitive::Torus => torus(12 * d + 12, 6 * d + 6, 1.0, 0.25),
            Primitive::Monkey => blob(d.min(5)),
        }
    }
}

struct Builder {
    p: Vec<Vec3>,
    t: Vec<[u32; 3]>,
    /// Diagonal edges created by triangulating quads (hidden in wireframe).
    diag: Vec<(u32, u32)>,
}

impl Builder {
    fn new() -> Self {
        Self { p: vec![], t: vec![], diag: vec![] }
    }
    fn v(&mut self, p: Vec3) -> u32 {
        self.p.push(p);
        (self.p.len() - 1) as u32
    }
    fn tri(&mut self, a: u32, b: u32, c: u32) {
        self.t.push([a, b, c]);
    }
    fn quad(&mut self, a: u32, b: u32, c: u32, d: u32) {
        self.t.push([a, b, c]);
        self.t.push([a, c, d]);
        self.diag.push((a.min(c), a.max(c)));
    }
    fn finish(self, smooth: bool) -> Mesh {
        let mut m = Mesh { positions: self.p, tris: self.t, smooth, ..Default::default() };
        let remap = m.weld();
        let diag: std::collections::HashSet<(u32, u32)> = self.diag.into_iter().collect();
        m.compute(&diag, remap.as_deref());
        m
    }
}

impl Mesh {
    /// Merge coincident vertices so edges / normals are shared.
    fn weld(&mut self) -> Option<Vec<u32>> {
        let mut map: HashMap<(i64, i64, i64), u32> = HashMap::new();
        let mut remap = vec![0u32; self.positions.len()];
        let mut out = Vec::new();
        let q = |f: f32| (f * 10000.0).round() as i64;
        for (i, p) in self.positions.iter().enumerate() {
            let k = (q(p.x), q(p.y), q(p.z));
            let id = *map.entry(k).or_insert_with(|| {
                out.push(*p);
                (out.len() - 1) as u32
            });
            remap[i] = id;
        }
        // If no welding happened we keep indices (diag set is index based on original).
        if out.len() == self.positions.len() {
            return None;
        }
        self.positions = out;
        for t in &mut self.tris {
            for i in t.iter_mut() {
                *i = remap[*i as usize];
            }
        }
        self.tris.retain(|t| t[0] != t[1] && t[1] != t[2] && t[0] != t[2]);
        Some(remap)
    }

    fn compute(&mut self, diag_orig: &std::collections::HashSet<(u32, u32)>, remap: Option<&[u32]>) {
        let diag: std::collections::HashSet<(u32, u32)> = diag_orig
            .iter()
            .map(|&(a, b)| match remap {
                Some(r) => {
                    let (a, b) = (r[a as usize], r[b as usize]);
                    (a.min(b), a.max(b))
                }
                None => (a, b),
            })
            .collect();

        // normals
        let mut n = vec![Vec3::ZERO; self.positions.len()];
        for t in &self.tris {
            let [a, b, c] = t.map(|i| self.positions[i as usize]);
            let fnrm = (b - a).cross(c - a); // area weighted
            for &i in t {
                n[i as usize] += fnrm;
            }
        }
        self.normals = n.into_iter().map(|v| v.normalize_or_zero()).collect();

        // edges
        let mut emap: HashMap<(u32, u32), usize> = HashMap::new();
        let mut edges: Vec<Edge> = Vec::new();
        for (fi, t) in self.tris.iter().enumerate() {
            for k in 0..3 {
                let (a, b) = (t[k], t[(k + 1) % 3]);
                let key = (a.min(b), a.max(b));
                match emap.get(&key) {
                    Some(&ei) => {
                        if edges[ei].faces[1] == u32::MAX {
                            edges[ei].faces[1] = fi as u32;
                        }
                    }
                    None => {
                        emap.insert(key, edges.len());
                        edges.push(Edge {
                            a: key.0,
                            b: key.1,
                            faces: [fi as u32, u32::MAX],
                            feature: !diag.contains(&key),
                        });
                    }
                }
            }
        }
        // Also hide edges between coplanar faces (e.g. fan triangulated caps).
        for e in &mut edges {
            if e.feature && e.faces[1] != u32::MAX {
                let f0 = self.face_normal(e.faces[0] as usize);
                let f1 = self.face_normal(e.faces[1] as usize);
                if f0.dot(f1) > 0.99999 && !self.smooth {
                    e.feature = false;
                }
            }
        }
        self.edges = edges;
    }

    pub fn face_normal(&self, f: usize) -> Vec3 {
        let [a, b, c] = self.tris[f].map(|i| self.positions[i as usize]);
        (b - a).cross(c - a).normalize_or_zero()
    }

    pub fn bounds(&self) -> (Vec3, Vec3) {
        let mut lo = Vec3::splat(f32::MAX);
        let mut hi = Vec3::splat(f32::MIN);
        for p in &self.positions {
            lo = lo.min(*p);
            hi = hi.max(*p);
        }
        (lo, hi)
    }
}

pub fn cube() -> Mesh {
    let mut b = Builder::new();
    // 6 faces, outward CCW winding
    let faces: [(Vec3, Vec3, Vec3); 6] = [
        (Vec3::X, Vec3::Y, Vec3::Z),
        (-Vec3::X, -Vec3::Y, Vec3::Z),
        (Vec3::Y, -Vec3::X, Vec3::Z),
        (-Vec3::Y, Vec3::X, Vec3::Z),
        (Vec3::Z, Vec3::X, Vec3::Y),
        (-Vec3::Z, Vec3::X, -Vec3::Y),
    ];
    for (n, u, v) in faces {
        let a = b.v(n - u - v);
        let bb = b.v(n + u - v);
        let c = b.v(n + u + v);
        let d = b.v(n - u + v);
        b.quad(a, bb, c, d);
    }
    b.finish(false)
}

pub fn plane() -> Mesh {
    let mut b = Builder::new();
    let a = b.v(Vec3::new(-1.0, -1.0, 0.0));
    let c = b.v(Vec3::new(1.0, -1.0, 0.0));
    let d = b.v(Vec3::new(1.0, 1.0, 0.0));
    let e = b.v(Vec3::new(-1.0, 1.0, 0.0));
    b.quad(a, c, d, e);
    b.finish(false)
}

pub fn uv_sphere(seg: u32, rings: u32) -> Mesh {
    let mut b = Builder::new();
    let mut idx = vec![];
    for r in 0..=rings {
        let th = PI * r as f32 / rings as f32;
        for s in 0..seg {
            let ph = TAU * s as f32 / seg as f32;
            idx.push(b.v(Vec3::new(th.sin() * ph.cos(), th.sin() * ph.sin(), th.cos())));
        }
    }
    let at = |r: u32, s: u32| idx[(r * seg + s % seg) as usize];
    for r in 0..rings {
        for s in 0..seg {
            let (a, bb, c, d) = (at(r, s), at(r + 1, s), at(r + 1, s + 1), at(r, s + 1));
            if r == 0 {
                b.tri(a, bb, c);
            } else if r == rings - 1 {
                b.tri(a, bb, d);
            } else {
                b.quad(a, bb, c, d);
            }
        }
    }
    b.finish(true)
}

pub fn ico_sphere(sub: u32) -> Mesh {
    let t = (1.0 + 5f32.sqrt()) / 2.0;
    let mut p: Vec<Vec3> = [
        (-1., t, 0.), (1., t, 0.), (-1., -t, 0.), (1., -t, 0.),
        (0., -1., t), (0., 1., t), (0., -1., -t), (0., 1., -t),
        (t, 0., -1.), (t, 0., 1.), (-t, 0., -1.), (-t, 0., 1.),
    ]
    .iter()
    .map(|&(x, y, z)| Vec3::new(x, y, z).normalize())
    .collect();
    let mut f: Vec<[u32; 3]> = vec![
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ];
    for _ in 1..sub {
        let mut cache: HashMap<(u32, u32), u32> = HashMap::new();
        let mut mid = |a: u32, b: u32, p: &mut Vec<Vec3>| -> u32 {
            let k = (a.min(b), a.max(b));
            *cache.entry(k).or_insert_with(|| {
                p.push(((p[a as usize] + p[b as usize]) * 0.5).normalize());
                (p.len() - 1) as u32
            })
        };
        let mut nf = vec![];
        for [a, b, c] in f {
            let ab = mid(a, b, &mut p);
            let bc = mid(b, c, &mut p);
            let ca = mid(c, a, &mut p);
            nf.extend([[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]);
        }
        f = nf;
    }
    let mut m = Mesh { positions: p, tris: f, smooth: sub > 1, ..Default::default() };
    m.compute(&Default::default(), None);
    // Ico spheres show every edge in Blender.
    for e in &mut m.edges {
        e.feature = true;
    }
    m
}

pub fn cylinder(seg: u32, r_bottom: f32, r_top: f32) -> Mesh {
    let mut b = Builder::new();
    let mut bot = vec![];
    let mut top = vec![];
    for s in 0..seg {
        let a = TAU * s as f32 / seg as f32;
        let (c, sn) = (a.cos(), a.sin());
        bot.push(b.v(Vec3::new(c * r_bottom, sn * r_bottom, -1.0)));
        top.push(b.v(Vec3::new(c * r_top, sn * r_top, 1.0)));
    }
    let cb = b.v(Vec3::new(0.0, 0.0, -1.0));
    let ct = b.v(Vec3::new(0.0, 0.0, 1.0));
    for s in 0..seg as usize {
        let n = (s + 1) % seg as usize;
        if r_top > 0.0 {
            b.quad(bot[s], bot[n], top[n], top[s]);
            b.tri(ct, top[s], top[n]);
        } else {
            b.tri(bot[s], bot[n], ct);
        }
        b.tri(cb, bot[n], bot[s]);
    }
    let mut m = b.finish(false);
    // smooth the sides but keep caps flat: approximate by marking smooth only for shading of sides
    m.smooth = false;
    m
}

pub fn torus(seg: u32, sides: u32, r: f32, tube: f32) -> Mesh {
    let mut b = Builder::new();
    let mut idx = vec![];
    for i in 0..seg {
        let u = TAU * i as f32 / seg as f32;
        for j in 0..sides {
            let v = TAU * j as f32 / sides as f32;
            let rr = r + tube * v.cos();
            idx.push(b.v(Vec3::new(rr * u.cos(), rr * u.sin(), tube * v.sin())));
        }
    }
    let at = |i: u32, j: u32| idx[((i % seg) * sides + j % sides) as usize];
    for i in 0..seg {
        for j in 0..sides {
            b.quad(at(i, j), at(i + 1, j), at(i + 1, j + 1), at(i, j + 1));
        }
    }
    b.finish(true)
}

/// An organic "blob" head shape (a stand-in for Blender's Suzanne).
pub fn blob(sub: u32) -> Mesh {
    let mut m = ico_sphere(sub.max(3));
    for p in &mut m.positions {
        let d = *p;
        let mut s = 1.0;
        // ears
        for sx in [-1.0f32, 1.0] {
            let ear = Vec3::new(sx * 0.9, 0.0, 0.35).normalize();
            s += 0.45 * d.dot(ear).max(0.0).powf(12.0);
        }
        // snout
        let snout = Vec3::new(0.0, -1.0, -0.25).normalize();
        s += 0.25 * d.dot(snout).max(0.0).powf(6.0);
        // brow
        s += 0.08 * (d.z * 3.0).sin() * d.y.min(0.0).abs();
        *p = Vec3::new(d.x * 1.05, d.y * 0.85, d.z * 0.9) * s;
    }
    let mut m2 = Mesh { positions: m.positions, tris: m.tris, smooth: true, ..Default::default() };
    m2.compute(&Default::default(), None);
    for e in &mut m2.edges {
        e.feature = true;
    }
    m2
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn primitives_are_closed_or_valid() {
        for p in Primitive::ALL {
            let m = p.build(2);
            assert!(!m.tris.is_empty(), "{p:?}");
            assert_eq!(m.normals.len(), m.positions.len());
            for t in &m.tris {
                for &i in t {
                    assert!((i as usize) < m.positions.len());
                }
            }
        }
        // cube: 8 welded verts, 12 tris, 12 visible edges
        let c = cube();
        assert_eq!(c.positions.len(), 8);
        assert_eq!(c.tris.len(), 12);
        assert_eq!(c.edges.iter().filter(|e| e.feature).count(), 12);
    }
    #[test]
    fn cube_normals_point_outward() {
        let c = cube();
        for f in 0..c.tris.len() {
            let n = c.face_normal(f);
            let ctr = c.tris[f].iter().map(|&i| c.positions[i as usize]).sum::<Vec3>() / 3.0;
            assert!(n.dot(ctr) > 0.0);
        }
    }
    #[test]
    fn parse_names() {
        assert_eq!(Primitive::parse("Donut"), Some(Primitive::Torus));
        assert_eq!(Primitive::parse("uv_sphere"), Some(Primitive::UvSphere));
        assert_eq!(Primitive::parse("xyz"), None);
    }
}
