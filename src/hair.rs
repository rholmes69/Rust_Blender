//! Hair and fur (particle-hair lite): strands grown from a surface as thin tapered tubes, with
//! guide strands plus children around them, length randomness, gravity droop, curl, clumping and
//! frizz. The strands are a derived mesh on the hair object (like a scatter), so every renderer
//! and the glTF export see real geometry. Roots are fixed on the surface's faces (face + barycentric),
//! so hair rides along when the surface is skinned, shape-keyed or edited.
//!
//! UVs run along each strand: U = 0 at the root … 1 at the tip, V = a random value per strand,
//! so a Gradient node (or the Hair node preset) gives root-to-tip colour.

use crate::mesh::PolyMesh;
use glam::{Mat4, Vec3};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Hair {
    /// Surface the hair grows from.
    pub target: u64,
    /// Guide strands.
    pub count: u32,
    /// Child strands around each guide (render count = count × (1 + children)).
    pub children: u32,
    /// Radius (m) children spread around their guide at the root.
    pub child_radius: f32,
    pub length: f32,
    /// 0 = all the same length, 1 = anywhere from 0 to `length`.
    pub random_length: f32,
    pub segments: u32,
    /// Strand radius at the root (m); tapers to `tip` × that at the tip.
    pub thickness: f32,
    pub tip: f32,
    /// How much the strands bend down (0 = stand along the normal, 1 = hang).
    pub gravity: f32,
    /// Curl turns along the strand and their radius (m).
    pub curl: f32,
    pub curl_radius: f32,
    /// Children pulled toward their guide at the tip (0 = parallel, 1 = meet at the tip).
    pub clump: f32,
    /// Random kinks (m).
    pub frizz: f32,
    pub seed: u32,
}

impl Hair {
    pub fn new(target: u64) -> Hair {
        Hair { target, count: 400, children: 5, child_radius: 0.04, length: 0.25, random_length: 0.3, segments: 5, thickness: 0.004, tip: 0.15, gravity: 0.2, curl: 0.0, curl_radius: 0.02, clump: 0.3, frizz: 0.005, seed: 1 }
    }

    /// A named style: fur, hair, long hair, curly, grass, shaggy (unknown names → fur).
    pub fn preset(target: u64, name: &str) -> (Hair, &'static str, [f32; 3]) {
        let low = name.to_lowercase();
        let mut h = Hair::new(target);
        if low.contains("grass") {
            (h.count, h.children, h.child_radius, h.length, h.random_length, h.segments, h.thickness, h.tip, h.gravity, h.clump, h.frizz) = (2500, 4, 0.05, 0.18, 0.5, 3, 0.008, 0.05, 0.08, 0.0, 0.0);
            (h, "Grass", [0.22, 0.45, 0.12])
        } else if low.contains("curl") {
            (h.count, h.children, h.length, h.gravity, h.curl, h.curl_radius, h.clump, h.segments) = (250, 6, 0.4, 0.35, 3.0, 0.035, 0.6, 12);
            (h, "Curly Hair", [0.18, 0.1, 0.05])
        } else if low.contains("long") || low == "hair" || low.contains("hair") {
            (h.count, h.children, h.child_radius, h.length, h.random_length, h.segments, h.thickness, h.gravity, h.clump) = (500, 12, 0.05, 0.7, 0.2, 8, 0.004, 0.75, 0.55);
            (h, "Hair", [0.25, 0.14, 0.07])
        } else if low.contains("shag") {
            (h.count, h.children, h.length, h.random_length, h.gravity, h.clump, h.frizz, h.segments) = (500, 4, 0.35, 0.5, 0.5, 0.4, 0.02, 6);
            (h, "Shaggy Fur", [0.55, 0.45, 0.35])
        } else {
            (h.count, h.children, h.child_radius, h.length, h.random_length, h.segments, h.thickness, h.tip, h.gravity, h.clump) = (1500, 8, 0.03, 0.1, 0.35, 3, 0.006, 0.25, 0.3, 0.2);
            (h, "Fur", [0.45, 0.3, 0.18])
        }
    }

    pub fn strands(&self) -> usize {
        self.count as usize * (1 + self.children as usize)
    }
}

struct Rng(u64);
impl Rng {
    fn new(seed: u32) -> Rng {
        Rng((seed as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0x2545_F491_4F6C_DD1D)
    }
    fn f(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / (1u64 << 24) as f32
    }
    fn s(&mut self) -> f32 {
        self.f() * 2.0 - 1.0
    }
}

/// A root on the surface: face, fan triangle (k) and barycentric (u, v).
#[derive(Clone, Copy)]
struct Root {
    face: u32,
    k: u32,
    u: f32,
    v: f32,
}

/// Area-weighted roots on the rest mesh (deterministic for a seed and topology).
fn sample_roots(rest: &PolyMesh, count: usize, seed: u32) -> Vec<Root> {
    let mut tris: Vec<(u32, u32, f32)> = vec![];
    for (fi, f) in rest.faces.iter().enumerate() {
        for k in 1..f.len().saturating_sub(1) {
            let (a, b, c) = (rest.verts[f[0] as usize], rest.verts[f[k] as usize], rest.verts[f[k + 1] as usize]);
            let area = (b - a).cross(c - a).length() * 0.5;
            if area > 1e-12 {
                tris.push((fi as u32, k as u32, area));
            }
        }
    }
    let total: f32 = tris.iter().map(|t| t.2).sum();
    if tris.is_empty() || total <= 0.0 {
        return vec![];
    }
    let mut cdf = Vec::with_capacity(tris.len());
    let mut acc = 0.0;
    for t in &tris {
        acc += t.2 / total;
        cdf.push(acc);
    }
    let mut r = Rng::new(seed);
    (0..count)
        .map(|_| {
            let x = r.f();
            let i = cdf.partition_point(|c| *c < x).min(tris.len() - 1);
            let (mut u, mut v) = (r.f(), r.f());
            if u + v > 1.0 {
                u = 1.0 - u;
                v = 1.0 - v;
            }
            Root { face: tris[i].0, k: tris[i].1, u, v }
        })
        .collect()
}

/// Root position and normal on the (possibly deformed) surface, in world space.
fn eval_root(m: &PolyMesh, r: Root, to_world: Mat4, nm: glam::Mat3) -> Option<(Vec3, Vec3)> {
    let f = m.faces.get(r.face as usize)?;
    let k = r.k as usize;
    if k + 1 >= f.len() {
        return None;
    }
    let (a, b, c) = (*m.verts.get(f[0] as usize)?, *m.verts.get(f[k] as usize)?, *m.verts.get(f[k + 1] as usize)?);
    let p = a + (b - a) * r.u + (c - a) * r.v;
    // the face normal (whole polygon, Newell) so roots on a quad agree
    let mut n = Vec3::ZERO;
    for i in 0..f.len() {
        let (p0, p1) = (m.verts[f[i] as usize], m.verts[f[(i + 1) % f.len()] as usize]);
        n += Vec3::new((p0.y - p1.y) * (p0.z + p1.z), (p0.z - p1.z) * (p0.x + p1.x), (p0.x - p1.x) * (p0.y + p1.y));
    }
    let n = (nm * n).normalize_or_zero();
    (n != Vec3::ZERO).then(|| (to_world.transform_point3(p), n))
}

fn perp(n: Vec3) -> (Vec3, Vec3) {
    let a = if n.z.abs() < 0.9 { Vec3::Z } else { Vec3::X };
    let t = a.cross(n).normalize();
    (t, n.cross(t))
}

/// The strand points (world): a guide's shape. `t` along the strand 0..1.
fn guide_points(h: &Hair, p: Vec3, n: Vec3, len: f32, r: &mut Rng) -> Vec<Vec3> {
    let segs = h.segments.clamp(1, 32) as usize;
    let step = len / segs as f32;
    let down = Vec3::NEG_Z;
    let phase = r.f() * std::f32::consts::TAU;
    let (t1, t2) = perp(n);
    let mut pts = Vec::with_capacity(segs + 1);
    let mut cur = p;
    pts.push(cur);
    let mut dir = n;
    // a little random lean so fur isn't combed perfectly
    let lean = (t1 * r.s() + t2 * r.s()) * 0.15;
    for i in 1..=segs {
        let t = i as f32 / segs as f32;
        let g = h.gravity.clamp(0.0, 1.0) * t * 1.6;
        dir = (n + lean + down * g).normalize_or(dir);
        cur += dir * step;
        pts.push(cur);
    }
    // curl: rotate around the strand axis, radius growing from the root
    if h.curl.abs() > 1e-4 && h.curl_radius > 0.0 {
        let base: Vec<Vec3> = pts.clone();
        for (i, q) in pts.iter_mut().enumerate().skip(1) {
            let t = i as f32 / segs as f32;
            let tan = (base[i] - base[i - 1]).normalize_or(n);
            let (a, b) = perp(tan);
            let ang = phase + t * h.curl * std::f32::consts::TAU;
            *q += (a * ang.cos() + b * ang.sin()) * h.curl_radius * t.sqrt();
        }
    }
    if h.frizz > 0.0 {
        for (i, q) in pts.iter_mut().enumerate().skip(1) {
            let t = i as f32 / segs as f32;
            *q += Vec3::new(r.s(), r.s(), r.s()) * h.frizz * t;
        }
    }
    pts
}

/// Build the hair mesh in the hair object's local space (`to_local`).
/// `rest` fixes where the roots are; `surface` is the current (maybe deformed) mesh.
pub fn build(h: &Hair, rest: &PolyMesh, surface: &PolyMesh, surface_world: Mat4, to_local: Mat4) -> PolyMesh {
    let roots = sample_roots(rest, h.count.min(20_000) as usize, h.seed);
    let nm = glam::Mat3::from_mat4(surface_world).inverse().transpose();
    let mut r = Rng::new(h.seed.wrapping_mul(2654435761).wrapping_add(7));
    let segs = h.segments.clamp(1, 32) as usize;
    let children = h.children.min(64) as usize;
    let total = roots.len() * (1 + children);
    // 3 sides per ring, rings 0..segs, the tip collapses to one vertex
    let mut out = PolyMesh { verts: Vec::with_capacity(total * (segs * 3 + 1)), faces: Vec::with_capacity(total * segs * 3), uvs: Vec::with_capacity(total * segs * 3) };
    let strand = |pts: &[Vec3], radius: f32, v: f32, out: &mut PolyMesh| {
        let base = out.verts.len() as u32;
        let n = pts.len();
        let mut frame_n = perp((pts[1] - pts[0]).normalize_or(Vec3::Z)).0;
        for i in 0..n - 1 {
            let tan = if i == 0 { pts[1] - pts[0] } else { pts[i + 1] - pts[i - 1] }.normalize_or(Vec3::Z);
            // parallel transport keeps the tube from twisting
            frame_n = (frame_n - tan * tan.dot(frame_n)).normalize_or(perp(tan).0);
            let b = tan.cross(frame_n);
            let t = i as f32 / (n - 1) as f32;
            let rad = radius * (1.0 + (h.tip.clamp(0.0, 1.0) - 1.0) * t);
            for s in 0..3 {
                let a = s as f32 / 3.0 * std::f32::consts::TAU;
                out.verts.push(to_local.transform_point3(pts[i] + (frame_n * a.cos() + b * a.sin()) * rad));
            }
        }
        out.verts.push(to_local.transform_point3(pts[n - 1]));
        let tip = out.verts.len() as u32 - 1;
        for i in 0..n - 1 {
            let (t0, t1) = (i as f32 / (n - 1) as f32, (i + 1) as f32 / (n - 1) as f32);
            for s in 0..3u32 {
                let s1 = (s + 1) % 3;
                let a = base + i as u32 * 3 + s;
                let b = base + i as u32 * 3 + s1;
                if i + 1 < n - 1 {
                    let (c, d) = (base + (i as u32 + 1) * 3 + s1, base + (i as u32 + 1) * 3 + s);
                    out.faces.push(vec![a, b, c, d]);
                    out.uvs.push(vec![[t0, v], [t0, v], [t1, v], [t1, v]]);
                } else {
                    out.faces.push(vec![a, b, tip]);
                    out.uvs.push(vec![[t0, v], [t0, v], [t1, v]]);
                }
            }
        }
    };
    for root in roots {
        let Some((p, n)) = eval_root(surface, root, surface_world, nm) else {
            // keep the random sequence stable
            for _ in 0..(4 + segs * 3) * (1 + children) {
                r.f();
            }
            continue;
        };
        let len = h.length.max(0.0) * (1.0 - h.random_length.clamp(0.0, 1.0) * r.f());
        if len < 1e-5 {
            continue;
        }
        let guide = guide_points(h, p, n, len, &mut r);
        strand(&guide, h.thickness, r.f(), &mut out);
        let (t1, t2) = perp(n);
        for _ in 0..children {
            // a random offset in the surface's tangent plane, pulled in toward the tip
            let ang = r.f() * std::f32::consts::TAU;
            let rad = h.child_radius * r.f().sqrt();
            let off = (t1 * ang.cos() + t2 * ang.sin()) * rad;
            let k = 1.0 - h.random_length.clamp(0.0, 1.0) * 0.5 * r.f();
            let mut pts: Vec<Vec3> = guide
                .iter()
                .enumerate()
                .map(|(i, q)| {
                    let t = i as f32 / (guide.len() - 1) as f32;
                    p + (*q - p) * k + off * (1.0 - h.clump.clamp(0.0, 1.0) * t)
                })
                .collect();
            if h.frizz > 0.0 {
                for (i, q) in pts.iter_mut().enumerate().skip(1) {
                    *q += Vec3::new(r.s(), r.s(), r.s()) * h.frizz * (i as f32 / segs as f32);
                }
            }
            strand(&pts, h.thickness * 0.9, r.f(), &mut out);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grows_strands_on_the_surface() {
        let plane = crate::mesh::Primitive::Plane.poly(0);
        let mut h = Hair::new(1);
        (h.count, h.children, h.segments, h.length, h.random_length, h.gravity, h.frizz) = (50, 2, 4, 0.5, 0.0, 0.0, 0.0);
        let m = build(&h, &plane, &plane, Mat4::IDENTITY, Mat4::IDENTITY);
        let strands = 50 * 3;
        assert_eq!(m.verts.len(), strands * (4 * 3 + 1));
        assert_eq!(m.faces.len(), strands * 4 * 3);
        assert_eq!(m.uvs.len(), m.faces.len());
        // roots on the plane (z ≈ 0), tips up to the length, inside the plane's extent
        let zmax = m.verts.iter().map(|v| v.z).fold(f32::MIN, f32::max);
        assert!((zmax - 0.5).abs() < 0.06, "tips at the length: {zmax}");
        assert!(m.verts.iter().all(|v| v.z > -0.01 && v.x.abs() < 1.2 && v.y.abs() < 1.2));
        // gravity bends them down; deterministic for a seed
        h.gravity = 1.0;
        let hung = build(&h, &plane, &plane, Mat4::IDENTITY, Mat4::IDENTITY);
        let zmax2 = hung.verts.iter().map(|v| v.z).fold(f32::MIN, f32::max);
        assert!(zmax2 < zmax - 0.1, "{zmax2} vs {zmax}");
        assert_eq!(hung.verts, build(&h, &plane, &plane, Mat4::IDENTITY, Mat4::IDENTITY).verts);
        // roots follow a moved / deformed surface
        let mut lifted = plane.clone();
        for v in &mut lifted.verts {
            v.z += 2.0;
        }
        h.gravity = 0.0;
        let up = build(&h, &plane, &lifted, Mat4::IDENTITY, Mat4::IDENTITY);
        assert!(up.verts.iter().all(|v| v.z > 1.99));
        for name in ["fur", "hair", "curly", "grass", "shaggy"] {
            let (h, _, _) = Hair::preset(1, name);
            let m = build(&h, &plane, &plane, Mat4::IDENTITY, Mat4::IDENTITY);
            assert!(!m.faces.is_empty() && m.verts.iter().all(|v| v.is_finite()), "{name}");
        }
    }
}
