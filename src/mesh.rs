//! Meshes: an editable polygon mesh (`PolyMesh`, n-gon faces) plus the derived render mesh
//! (`Mesh`: triangles, normals, edge adjacency). Procedural primitives are Z-up like Blender.

use glam::Vec3;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::f32::consts::{PI, TAU};

/// Editable polygon mesh (what edit mode works on and what is saved in .rblend files).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct PolyMesh {
    pub verts: Vec<Vec3>,
    /// Faces as counter-clockwise vertex loops (outward normal by right-hand rule).
    pub faces: Vec<Vec<u32>>,
    /// Optional texture coordinates per face corner (parallel to `faces`). Empty = generated
    /// box-projection UVs. Topology-changing edit operators drop them.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub uvs: Vec<Vec<[f32; 2]>>,
}

/// Render data derived from a `PolyMesh`.
#[derive(Clone, Debug, Default)]
pub struct Mesh {
    pub positions: Vec<Vec3>,
    pub tris: Vec<[u32; 3]>,
    /// Polygon index for each triangle.
    pub tri_face: Vec<u32>,
    /// Per-vertex smooth normals (same length as positions).
    pub normals: Vec<Vec3>,
    /// Texture coordinates for each triangle corner.
    pub tri_uv: Vec<[[f32; 2]; 3]>,
    /// Shading normal for each triangle corner (smooth, flat or split by the auto-smooth angle).
    pub corner_normals: Vec<[Vec3; 3]>,
    /// Auto smooth was applied: `corner_normals` differ from the shared vertex normals.
    pub split_normals: bool,
    /// Unique edges, each with the (up to two) triangles that share it.
    pub edges: Vec<Edge>,
    pub smooth: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct Edge {
    pub a: u32,
    pub b: u32,
    pub faces: [u32; 2], // triangle indices, u32::MAX = boundary
    /// Polygon edges (drawn in wireframe); triangulation diagonals are not.
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


    pub fn default_smooth(self) -> bool {
        matches!(
            self,
            Primitive::UvSphere | Primitive::IcoSphere | Primitive::Torus | Primitive::Monkey
        )
    }

    pub fn poly(self, detail: u32) -> PolyMesh {
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

impl PolyMesh {
    /// Polygon area (Newell vector length / 2).
    pub fn face_area(&self, f: usize) -> f32 {
        let face = &self.faces[f];
        let mut n = Vec3::ZERO;
        for i in 0..face.len() {
            let a = self.verts[face[i] as usize];
            let b = self.verts[face[(i + 1) % face.len()] as usize];
            n += a.cross(b);
        }
        n.length() * 0.5
    }

    pub fn face_normal(&self, f: usize) -> Vec3 {
        // Newell's method (robust for n-gons)
        let face = &self.faces[f];
        let mut n = Vec3::ZERO;
        for i in 0..face.len() {
            let a = self.verts[face[i] as usize];
            let b = self.verts[face[(i + 1) % face.len()] as usize];
            n += Vec3::new(
                (a.y - b.y) * (a.z + b.z),
                (a.z - b.z) * (a.x + b.x),
                (a.x - b.x) * (a.y + b.y),
            );
        }
        n.normalize_or_zero()
    }

    pub fn face_center(&self, f: usize) -> Vec3 {
        let face = &self.faces[f];
        face.iter().map(|&i| self.verts[i as usize]).sum::<Vec3>() / face.len().max(1) as f32
    }

    /// Unique undirected edges.
    pub fn edges(&self) -> Vec<(u32, u32)> {
        let mut set = HashSet::new();
        let mut out = vec![];
        for f in &self.faces {
            for i in 0..f.len() {
                let (a, b) = (f[i], f[(i + 1) % f.len()]);
                let k = (a.min(b), a.max(b));
                if set.insert(k) {
                    out.push(k);
                }
            }
        }
        out
    }

    /// Map from edge -> faces using it.
    pub fn edge_faces(&self) -> HashMap<(u32, u32), Vec<usize>> {
        let mut m: HashMap<(u32, u32), Vec<usize>> = HashMap::new();
        for (fi, f) in self.faces.iter().enumerate() {
            for i in 0..f.len() {
                let (a, b) = (f[i], f[(i + 1) % f.len()]);
                m.entry((a.min(b), a.max(b))).or_default().push(fi);
            }
        }
        m
    }

    /// Merge coincident vertices, drop degenerate faces and unused vertices.
    pub fn weld(&mut self, eps: f32) {
        let mut map: HashMap<(i64, i64, i64), u32> = HashMap::new();
        let mut remap = vec![0u32; self.verts.len()];
        let mut out = Vec::new();
        let q = |f: f32| (f / eps).round() as i64;
        for (i, p) in self.verts.iter().enumerate() {
            let k = (q(p.x), q(p.y), q(p.z));
            remap[i] = *map.entry(k).or_insert_with(|| {
                out.push(*p);
                (out.len() - 1) as u32
            });
        }
        self.verts = out;
        for f in &mut self.faces {
            for i in f.iter_mut() {
                *i = remap[*i as usize];
            }
        }
        self.cleanup();
    }

    /// Are the per-corner UVs present and consistent with the faces?
    pub fn has_uvs(&self) -> bool {
        !self.uvs.is_empty() && self.uvs.len() == self.faces.len() && self.uvs.iter().zip(&self.faces).all(|(u, f)| u.len() == f.len())
    }

    /// Remove consecutive duplicate indices, faces with < 3 verts, and unreferenced vertices.
    pub fn cleanup(&mut self) {
        let keep_uvs = self.has_uvs();
        let mut uvs = std::mem::take(&mut self.uvs);
        if !keep_uvs {
            uvs = vec![vec![]; self.faces.len()];
        }
        for (f, uv) in self.faces.iter_mut().zip(uvs.iter_mut()) {
            // dedup consecutive indices, keeping the UV list aligned
            let mut nf = Vec::with_capacity(f.len());
            let mut nu = Vec::with_capacity(f.len());
            for (k, &v) in f.iter().enumerate() {
                if nf.last() != Some(&v) {
                    nf.push(v);
                    if keep_uvs {
                        nu.push(uv[k]);
                    }
                }
            }
            while nf.len() > 1 && nf.first() == nf.last() {
                nf.pop();
                if keep_uvs {
                    nu.pop();
                }
            }
            *f = nf;
            *uv = nu;
        }
        let keep: Vec<bool> = self.faces.iter().map(|f| f.len() >= 3).collect();
        let mut k = keep.iter();
        self.faces.retain(|_| *k.next().unwrap());
        if keep_uvs {
            let mut k = keep.iter();
            uvs.retain(|_| *k.next().unwrap());
            self.uvs = uvs;
        }
        let mut used = vec![false; self.verts.len()];
        for f in &self.faces {
            for &i in f {
                used[i as usize] = true;
            }
        }
        if used.iter().all(|u| *u) {
            return;
        }
        let mut remap = vec![u32::MAX; self.verts.len()];
        let mut nv = vec![];
        for (i, u) in used.iter().enumerate() {
            if *u {
                remap[i] = nv.len() as u32;
                nv.push(self.verts[i]);
            }
        }
        self.verts = nv;
        for f in &mut self.faces {
            for i in f.iter_mut() {
                *i = remap[*i as usize];
            }
        }
    }

    pub fn to_mesh(&self, smooth: bool) -> Mesh {
        self.to_mesh_opts(smooth, 0.0)
    }

    /// Render mesh; with `smooth` and `auto_smooth_deg` > 0, faces meeting at a sharper
    /// angle keep hard edges (Blender's "Shade Auto Smooth").
    pub fn to_mesh_opts(&self, smooth: bool, auto_smooth_deg: f32) -> Mesh {
        let mut m = self.to_mesh_inner(smooth);
        m.corner_normals(smooth, auto_smooth_deg);
        m
    }

    fn to_mesh_inner(&self, smooth: bool) -> Mesh {
        let mut tris = vec![];
        let mut tri_face = vec![];
        let mut tri_uv = vec![];
        let has_uvs = self.has_uvs();
        let mut poly_edges: HashSet<(u32, u32)> = HashSet::new();
        for (fi, f) in self.faces.iter().enumerate() {
            for i in 0..f.len() {
                let (a, b) = (f[i], f[(i + 1) % f.len()]);
                poly_edges.insert((a.min(b), a.max(b)));
            }
            let corner_uv: Vec<[f32; 2]> = if has_uvs { self.uvs[fi].clone() } else { box_uv(&self.verts, f) };
            for t in triangulate(&self.verts, f) {
                tri_uv.push(t.map(|v| f.iter().position(|&x| x == v).map(|k| corner_uv[k]).unwrap_or([0.0, 0.0])));
                tris.push(t);
                tri_face.push(fi as u32);
            }
        }
        let mut m = Mesh {
            positions: self.verts.clone(),
            tris,
            tri_face,
            tri_uv,
            smooth,
            ..Default::default()
        };
        m.compute(&poly_edges);
        m
    }
}

/// Generated UVs: project each face onto the plane of its dominant normal axis
/// (2 units of geometry = one texture repeat, like a default cube face).
pub fn box_uv(verts: &[Vec3], f: &[u32]) -> Vec<[f32; 2]> {
    let mut n = Vec3::ZERO;
    for i in 0..f.len() {
        let (a, b) = (verts[f[i] as usize], verts[f[(i + 1) % f.len()] as usize]);
        n += Vec3::new((a.y - b.y) * (a.z + b.z), (a.z - b.z) * (a.x + b.x), (a.x - b.x) * (a.y + b.y));
    }
    let a = n.abs();
    f.iter()
        .map(|&i| {
            let p = verts[i as usize];
            let (u, v) = if a.z >= a.x && a.z >= a.y {
                (p.x * n.z.signum(), p.y)
            } else if a.x >= a.y {
                (p.y * n.x.signum(), p.z)
            } else {
                (-p.x * n.y.signum(), p.z)
            };
            [u * 0.5 + 0.5, 1.0 - (v * 0.5 + 0.5)]
        })
        .collect()
}

/// Triangulate a polygon (ear clipping in the polygon's plane; fan for convex quads/tris).
pub fn triangulate(verts: &[Vec3], f: &[u32]) -> Vec<[u32; 3]> {
    if f.len() == 3 {
        return vec![[f[0], f[1], f[2]]];
    }
    // normal
    let mut n = Vec3::ZERO;
    for i in 0..f.len() {
        let a = verts[f[i] as usize];
        let b = verts[f[(i + 1) % f.len()] as usize];
        n += Vec3::new(
            (a.y - b.y) * (a.z + b.z),
            (a.z - b.z) * (a.x + b.x),
            (a.x - b.x) * (a.y + b.y),
        );
    }
    let n = n.normalize_or_zero();
    let mut idx: Vec<u32> = f.to_vec();
    let mut out = vec![];
    let mut guard = 0;
    while idx.len() > 3 && guard < 10_000 {
        guard += 1;
        let k = idx.len();
        let mut clipped = false;
        for i in 0..k {
            let (ia, ib, ic) = (idx[(i + k - 1) % k], idx[i], idx[(i + 1) % k]);
            let (a, b, c) = (verts[ia as usize], verts[ib as usize], verts[ic as usize]);
            if (b - a).cross(c - b).dot(n) <= 1e-9 {
                continue; // reflex or degenerate
            }
            let inside = idx.iter().any(|&j| {
                if j == ia || j == ib || j == ic {
                    return false;
                }
                let p = verts[j as usize];
                let s1 = (b - a).cross(p - a).dot(n);
                let s2 = (c - b).cross(p - b).dot(n);
                let s3 = (a - c).cross(p - c).dot(n);
                s1 > 0.0 && s2 > 0.0 && s3 > 0.0
            });
            if inside {
                continue;
            }
            out.push([ia, ib, ic]);
            idx.remove(i);
            clipped = true;
            break;
        }
        if !clipped {
            // fall back to a fan for weird / non-planar polygons
            for i in 1..idx.len() - 1 {
                out.push([idx[0], idx[i], idx[i + 1]]);
            }
            return out;
        }
    }
    if idx.len() == 3 {
        out.push([idx[0], idx[1], idx[2]]);
    }
    out
}

struct Builder {
    p: Vec<Vec3>,
    faces: Vec<Vec<u32>>,
}

impl Builder {
    fn new() -> Self {
        Self {
            p: vec![],
            faces: vec![],
        }
    }
    fn v(&mut self, p: Vec3) -> u32 {
        self.p.push(p);
        (self.p.len() - 1) as u32
    }
    fn tri(&mut self, a: u32, b: u32, c: u32) {
        self.faces.push(vec![a, b, c]);
    }
    fn quad(&mut self, a: u32, b: u32, c: u32, d: u32) {
        self.faces.push(vec![a, b, c, d]);
    }
    fn ngon(&mut self, f: Vec<u32>) {
        self.faces.push(f);
    }
    fn finish(self) -> PolyMesh {
        let mut m = PolyMesh {
            verts: self.p,
            faces: self.faces,
            ..Default::default()
        };
        m.weld(1e-4);
        m
    }
}

impl Mesh {
    fn corner_normals(&mut self, smooth: bool, auto_deg: f32) {
        let fnrm: Vec<Vec3> = self
            .tris
            .iter()
            .map(|t| {
                let [a, b, c] = t.map(|i| self.positions[i as usize]);
                (b - a).cross(c - a)
            })
            .collect();
        if !smooth {
            self.corner_normals = fnrm.iter().map(|n| [n.normalize_or_zero(); 3]).collect();
            return;
        }
        if auto_deg <= 0.0 || auto_deg >= 180.0 {
            self.corner_normals = self.tris.iter().map(|t| t.map(|i| self.normals[i as usize])).collect();
            return;
        }
        let cos = auto_deg.to_radians().cos();
        let mut around: Vec<Vec<u32>> = vec![vec![]; self.positions.len()];
        for (ti, t) in self.tris.iter().enumerate() {
            for &i in t {
                around[i as usize].push(ti as u32);
            }
        }
        let unit: Vec<Vec3> = fnrm.iter().map(|n| n.normalize_or_zero()).collect();
        let mut split = false;
        self.corner_normals = self
            .tris
            .iter()
            .enumerate()
            .map(|(ti, t)| {
                t.map(|v| {
                    let mut n = Vec3::ZERO;
                    for &o in &around[v as usize] {
                        if unit[o as usize].dot(unit[ti]) >= cos {
                            n += fnrm[o as usize];
                        }
                    }
                    let n = n.normalize_or_zero();
                    if n.dot(self.normals[v as usize]) < 0.9999 {
                        split = true;
                    }
                    n
                })
            })
            .collect();
        self.split_normals = split;
    }

    fn compute(&mut self, poly_edges: &HashSet<(u32, u32)>) {
        let mut n = vec![Vec3::ZERO; self.positions.len()];
        for t in &self.tris {
            let [a, b, c] = t.map(|i| self.positions[i as usize]);
            let fnrm = (b - a).cross(c - a); // area weighted
            for &i in t {
                n[i as usize] += fnrm;
            }
        }
        self.normals = n.into_iter().map(|v| v.normalize_or_zero()).collect();
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
                            feature: poly_edges.contains(&key),
                        });
                    }
                }
            }
        }
        self.edges = edges;
    }

    pub fn bounds(&self) -> (Vec3, Vec3) {
        let mut lo = Vec3::splat(f32::MAX);
        let mut hi = Vec3::splat(f32::MIN);
        for p in &self.positions {
            lo = lo.min(*p);
            hi = hi.max(*p);
        }
        if self.positions.is_empty() {
            return (Vec3::ZERO, Vec3::ZERO);
        }
        (lo, hi)
    }
}

pub fn cube() -> PolyMesh {
    let mut b = Builder::new();
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
    b.finish()
}

pub fn plane() -> PolyMesh {
    let mut b = Builder::new();
    let a = b.v(Vec3::new(-1.0, -1.0, 0.0));
    let c = b.v(Vec3::new(1.0, -1.0, 0.0));
    let d = b.v(Vec3::new(1.0, 1.0, 0.0));
    let e = b.v(Vec3::new(-1.0, 1.0, 0.0));
    b.quad(a, c, d, e);
    b.finish()
}

pub fn uv_sphere(seg: u32, rings: u32) -> PolyMesh {
    let mut b = Builder::new();
    let mut idx = vec![];
    for r in 0..=rings {
        let th = PI * r as f32 / rings as f32;
        for s in 0..seg {
            let ph = TAU * s as f32 / seg as f32;
            idx.push(b.v(Vec3::new(
                th.sin() * ph.cos(),
                th.sin() * ph.sin(),
                th.cos(),
            )));
        }
    }
    let at = |r: u32, s: u32| idx[(r * seg + s % seg) as usize];
    for r in 0..rings {
        for s in 0..seg {
            let (a, bb, c, d) = (at(r, s), at(r + 1, s), at(r + 1, s + 1), at(r, s + 1));
            b.quad(a, bb, c, d); // poles collapse to triangles when welded
        }
    }
    b.finish()
}

pub fn ico_sphere(sub: u32) -> PolyMesh {
    let t = (1.0 + 5f32.sqrt()) / 2.0;
    let mut p: Vec<Vec3> = [
        (-1., t, 0.),
        (1., t, 0.),
        (-1., -t, 0.),
        (1., -t, 0.),
        (0., -1., t),
        (0., 1., t),
        (0., -1., -t),
        (0., 1., -t),
        (t, 0., -1.),
        (t, 0., 1.),
        (-t, 0., -1.),
        (-t, 0., 1.),
    ]
    .iter()
    .map(|&(x, y, z)| Vec3::new(x, y, z).normalize())
    .collect();
    let mut f: Vec<[u32; 3]> = vec![
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
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
    PolyMesh {
        verts: p,
        faces: f.into_iter().map(|t| t.to_vec()).collect(),
        ..Default::default()
    }
}

pub fn cylinder(seg: u32, r_bottom: f32, r_top: f32) -> PolyMesh {
    let mut b = Builder::new();
    let mut bot = vec![];
    let mut top = vec![];
    for s in 0..seg {
        let a = TAU * s as f32 / seg as f32;
        let (c, sn) = (a.cos(), a.sin());
        bot.push(b.v(Vec3::new(c * r_bottom, sn * r_bottom, -1.0)));
        top.push(b.v(Vec3::new(c * r_top, sn * r_top, 1.0)));
    }
    let apex = (r_top <= 0.0).then(|| b.v(Vec3::new(0.0, 0.0, 1.0)));
    for s in 0..seg as usize {
        let n = (s + 1) % seg as usize;
        match apex {
            None => b.quad(bot[s], bot[n], top[n], top[s]),
            Some(ap) => b.tri(bot[s], bot[n], ap),
        }
    }
    // n-gon caps, like Blender
    b.ngon(bot.iter().rev().copied().collect());
    if apex.is_none() {
        b.ngon(top.clone());
    }
    b.finish()
}

pub fn torus(seg: u32, sides: u32, r: f32, tube: f32) -> PolyMesh {
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
    b.finish()
}

/// An organic "blob" head shape (a stand-in for Blender's Suzanne).
pub fn blob(sub: u32) -> PolyMesh {
    let mut m = ico_sphere(sub.max(3));
    for p in &mut m.verts {
        let d = *p;
        let mut s = 1.0;
        for sx in [-1.0f32, 1.0] {
            let ear = Vec3::new(sx * 0.9, 0.0, 0.35).normalize();
            s += 0.45 * d.dot(ear).max(0.0).powf(12.0);
        }
        let snout = Vec3::new(0.0, -1.0, -0.25).normalize();
        s += 0.25 * d.dot(snout).max(0.0).powf(6.0);
        s += 0.08 * (d.z * 3.0).sin() * d.y.min(0.0).abs();
        *p = Vec3::new(d.x * 1.05, d.y * 0.85, d.z * 0.9) * s;
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn primitives_are_valid() {
        for p in Primitive::ALL {
            let pm = p.poly(2);
            let m = pm.to_mesh(true);
            assert!(!m.tris.is_empty(), "{p:?}");
            assert_eq!(m.normals.len(), m.positions.len());
            assert_eq!(m.tri_face.len(), m.tris.len());
            for t in &m.tris {
                for &i in t {
                    assert!((i as usize) < m.positions.len());
                }
            }
            // closed meshes: every edge shared by exactly two faces (except plane)
            if p != Primitive::Plane {
                for (e, f) in pm.edge_faces() {
                    assert_eq!(f.len(), 2, "{p:?} edge {e:?}");
                }
            }
        }
        let c = cube();
        assert_eq!(c.verts.len(), 8);
        assert_eq!(c.faces.len(), 6);
        let m = c.to_mesh(false);
        assert_eq!(m.tris.len(), 12);
        assert_eq!(m.edges.iter().filter(|e| e.feature).count(), 12);
        let cyl = cylinder(16, 1.0, 1.0);
        assert_eq!(cyl.faces.len(), 18);
    }
    #[test]
    fn normals_point_outward() {
        for p in [
            Primitive::Cube,
            Primitive::Cylinder,
            Primitive::Cone,
            Primitive::UvSphere,
        ] {
            let c = p.poly(2);
            for f in 0..c.faces.len() {
                let n = c.face_normal(f);
                assert!(n.dot(c.face_center(f)) > 0.0, "{p:?} face {f}");
            }
        }
    }
    #[test]
    fn triangulate_concave() {
        // L-shape
        let v = vec![
            Vec3::new(0., 0., 0.),
            Vec3::new(2., 0., 0.),
            Vec3::new(2., 1., 0.),
            Vec3::new(1., 1., 0.),
            Vec3::new(1., 2., 0.),
            Vec3::new(0., 2., 0.),
        ];
        let t = triangulate(&v, &[0, 1, 2, 3, 4, 5]);
        assert_eq!(t.len(), 4);
        let area: f32 = t
            .iter()
            .map(|t| {
                (v[t[1] as usize] - v[t[0] as usize])
                    .cross(v[t[2] as usize] - v[t[0] as usize])
                    .z
                    * 0.5
            })
            .sum();
        assert!((area - 3.0).abs() < 1e-4, "{area}");
    }
    #[test]
    fn parse_names() {
        assert_eq!(Primitive::parse("Donut"), Some(Primitive::Torus));
        assert_eq!(Primitive::parse("uv_sphere"), Some(Primitive::UvSphere));
        assert_eq!(Primitive::parse("xyz"), None);
    }
}
