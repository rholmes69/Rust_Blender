//! Scatter (a small "geometry nodes" setup): copies of one object spread over the faces of
//! another, with random scale / rotation, optional normal alignment and a slope limit.
//! The result is a derived mesh on the scatter object, regenerated when its inputs change.

use crate::mesh::PolyMesh;
use glam::{Mat4, Quat, Vec3};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Scatter {
    /// Object whose mesh is copied.
    pub source: u64,
    /// Surface the copies sit on.
    pub target: u64,
    pub count: u32,
    pub seed: u32,
    /// Random uniform scale between these factors (times the source's own scale).
    pub scale_min: f32,
    pub scale_max: f32,
    /// Random spin around the up axis, degrees (360 = any direction).
    #[serde(default = "full_turn")]
    pub spin: f32,
    /// Random tilt away from up, degrees.
    #[serde(default)]
    pub tilt: f32,
    /// Stand along the surface normal instead of world up.
    #[serde(default)]
    pub align: bool,
    /// Only faces up to this slope (degrees from flat; 180 = everywhere, even overhangs).
    #[serde(default = "everywhere")]
    pub max_slope: f32,
}

fn full_turn() -> f32 {
    360.0
}
fn everywhere() -> f32 {
    180.0
}

impl Scatter {
    pub fn new(source: u64, target: u64, count: u32) -> Scatter {
        Scatter { source, target, count, seed: 1, scale_min: 0.8, scale_max: 1.2, spin: 360.0, tilt: 0.0, align: false, max_slope: 180.0 }
    }
}

struct Rng(u64);
impl Rng {
    fn new(seed: u32) -> Rng {
        Rng((seed as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xD1B5_4A32_D192_ED03)
    }
    fn f(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// Points on the surface (world space): position, normal. Area-weighted, deterministic.
pub fn sample_surface(target: &PolyMesh, to_world: Mat4, count: usize, seed: u32, max_slope: f32) -> Vec<(Vec3, Vec3)> {
    let nm = glam::Mat3::from_mat4(to_world).inverse().transpose();
    let cos_max = max_slope.clamp(0.0, 180.0).to_radians().cos();
    let mut tris: Vec<([Vec3; 3], Vec3, f32)> = vec![];
    for f in &target.faces {
        if f.len() < 3 {
            continue;
        }
        let p: Vec<Vec3> = f.iter().map(|&i| to_world.transform_point3(target.verts[i as usize])).collect();
        for k in 1..p.len() - 1 {
            let (a, b, c) = (p[0], p[k], p[k + 1]);
            let n = (b - a).cross(c - a);
            let area = n.length() * 0.5;
            if area < 1e-9 {
                continue;
            }
            // normals from the local cross product mapped by the normal matrix (handles mirroring)
            let ln = (target.verts[f[k] as usize] - target.verts[f[0] as usize]).cross(target.verts[f[k + 1] as usize] - target.verts[f[0] as usize]);
            let wn = (nm * ln).normalize_or_zero();
            if wn.z < cos_max - 1e-6 {
                continue;
            }
            tris.push(([a, b, c], wn, area));
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
            let ([a, b, c], n, _) = tris[i];
            let (mut u, mut v) = (r.f(), r.f());
            if u + v > 1.0 {
                u = 1.0 - u;
                v = 1.0 - v;
            }
            (a + (b - a) * u + (c - a) * v, n)
        })
        .collect()
}

/// The instance matrices (world) for a scatter.
pub fn instances(s: &Scatter, target: &PolyMesh, target_world: Mat4, source_rot_scale: Mat4) -> Vec<Mat4> {
    let pts = sample_surface(target, target_world, s.count.min(20_000) as usize, s.seed, s.max_slope);
    let mut r = Rng::new(s.seed.wrapping_mul(7919).wrapping_add(13));
    pts.into_iter()
        .map(|(p, n)| {
            let up = if s.align { n } else { Vec3::Z };
            let base = Quat::from_rotation_arc(Vec3::Z, up.normalize_or(Vec3::Z));
            let spin = Quat::from_rotation_z((r.f() - 0.5) * s.spin.to_radians());
            let tilt = Quat::from_axis_angle(Vec3::new(r.f() - 0.5, r.f() - 0.5, 0.0).normalize_or(Vec3::X), r.f() * s.tilt.to_radians());
            let k = s.scale_min + (s.scale_max - s.scale_min) * r.f();
            Mat4::from_scale_rotation_translation(Vec3::splat(k.max(1e-4)), base * tilt * spin, p) * source_rot_scale
        })
        .collect()
}

/// All copies merged into one mesh in the local space `to_local`.
pub fn build(source: &PolyMesh, mats: &[Mat4], to_local: Mat4) -> PolyMesh {
    let has_uv = source.has_uvs();
    let nv = source.verts.len();
    let mut out = PolyMesh { verts: Vec::with_capacity(nv * mats.len()), faces: Vec::with_capacity(source.faces.len() * mats.len()), uvs: vec![] };
    for (k, m) in mats.iter().enumerate() {
        let m = to_local * *m;
        let flip = m.determinant() < 0.0;
        out.verts.extend(source.verts.iter().map(|v| m.transform_point3(*v)));
        let base = (k * nv) as u32;
        for (fi, f) in source.faces.iter().enumerate() {
            let mut nf: Vec<u32> = f.iter().map(|i| i + base).collect();
            let mut uv = if has_uv { source.uvs[fi].clone() } else { vec![] };
            if flip {
                nf.reverse();
                uv.reverse();
            }
            out.faces.push(nf);
            if has_uv {
                out.uvs.push(uv);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn scatter_on_plane() {
        let plane = crate::mesh::Primitive::Plane.poly(1);
        let w = Mat4::from_scale(Vec3::new(5.0, 5.0, 1.0));
        let s = Scatter { count: 50, ..Scatter::new(1, 2, 50) };
        let m = instances(&s, &plane, w, Mat4::IDENTITY);
        assert_eq!(m.len(), 50);
        // every copy sits on the plane, inside it
        for x in &m {
            let p = x.w_axis.truncate();
            assert!(p.z.abs() < 1e-5 && p.x.abs() <= 5.0 + 1e-4 && p.y.abs() <= 5.0 + 1e-4);
        }
        // deterministic and seed-dependent
        assert_eq!(m, instances(&s, &plane, w, Mat4::IDENTITY));
        assert_ne!(m, instances(&Scatter { seed: 2, ..s.clone() }, &plane, w, Mat4::IDENTITY));
        let cube = crate::mesh::cube();
        let merged = build(&cube, &m, Mat4::IDENTITY);
        assert_eq!((merged.verts.len(), merged.faces.len()), (8 * 50, 6 * 50));
        // slope limit: on a cube only the top faces qualify
        let pts = sample_surface(&cube, Mat4::IDENTITY, 100, 3, 10.0);
        assert!(pts.iter().all(|(p, n)| n.z > 0.99 && (p.z - 1.0).abs() < 1e-5));
    }
}
