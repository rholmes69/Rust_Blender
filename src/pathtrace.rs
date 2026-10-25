//! Progressive path tracer ("Cycles-lite") for F12 renders: global illumination, soft shadows
//! from area-like lights, GGX glossy reflections, glass, emissive surfaces and textures.
//! Runs on all CPU cores in a background thread and streams refined images.

use crate::raytrace::Bvh;
use crate::scene::{LightType, MeshCache, Scene};
use crate::textures::{TexImage, TextureCache};
use crate::viewport::Camera;
use crossbeam_channel::{Receiver, unbounded};
use glam::{Mat3, Vec3};
use std::f32::consts::PI;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Clone, Copy)]
struct PtMat {
    albedo: Vec3,
    metallic: f32,
    roughness: f32,
    emission: Vec3,
    /// 1 = opaque, < 1 = glass-like transmission
    alpha: f32,
    tex: Option<u16>,
    normal_tex: Option<u16>,
    normal_strength: f32,
    rough_tex: Option<u16>,
    /// shader node graph (index into `PtScene::graphs`) and the material's emission strength
    graph: Option<u16>,
    emit: f32,
}

struct Light {
    kind: LightType,
    pos: Vec3,
    dir: Vec3,
    /// irradiance scale (matches the raster renderer's units)
    radiance: Vec3,
    cos_outer: f32,
    cos_inner: f32,
    radius: f32,
}

/// Everything the tracer needs, gathered on the UI thread.
pub struct PtScene {
    bvh: Bvh,
    shadow_bvh: Bvh,
    /// per triangle: corner normals, corner UVs, material index
    normals: Vec<[Vec3; 3]>,
    uvs: Vec<[[f32; 2]; 3]>,
    /// per triangle: UV tangent frame (dP/du, -dP/dv), for normal maps
    tangents: Vec<[Vec3; 2]>,
    /// world → object matrix per triangle (index into `invs`), for Object texture coordinates
    inv_of: Vec<u32>,
    invs: Vec<glam::Mat4>,
    mat_of: Vec<u16>,
    mats: Vec<PtMat>,
    textures: Vec<Arc<TexImage>>,
    graphs: Vec<Arc<crate::nodes::NodeGraph>>,
    /// scene texture id → slot, for Image nodes
    graph_tex: std::collections::HashMap<u32, u16>,
    lights: Vec<Light>,
    ambient: f32,
    bg_top: Vec3,
    bg_bot: Vec3,
    cam: Camera,
    #[allow(dead_code)]
    pub tris: usize,
}

/// Tangent and bitangent of a triangle's UV mapping: directions of +u and of image-up (−v).
fn uv_frame(p: &[Vec3; 3], uv: &[[f32; 2]; 3]) -> [Vec3; 2] {
    let (e1, e2) = (p[1] - p[0], p[2] - p[0]);
    let (du1, dv1) = (uv[1][0] - uv[0][0], uv[1][1] - uv[0][1]);
    let (du2, dv2) = (uv[2][0] - uv[0][0], uv[2][1] - uv[0][1]);
    let det = du1 * dv2 - du2 * dv1;
    if det.abs() < 1e-12 {
        return [Vec3::ZERO; 2];
    }
    let r = 1.0 / det;
    let t = (e1 * dv2 - e2 * dv1) * r;
    let b = (e2 * du1 - e1 * du2) * r;
    [t, -b]
}

fn lin(c: [f32; 3]) -> Vec3 {
    Vec3::new(c[0].powf(2.2), c[1].powf(2.2), c[2].powf(2.2))
}

impl PtScene {
    pub fn gather(scene: &Scene, cache: &mut MeshCache, tex_cache: &mut TextureCache, cam: Camera) -> PtScene {
        let mut tris = vec![];
        let mut normals = vec![];
        let mut uvs = vec![];
        let mut tangents = vec![];
        let mut inv_of = vec![];
        let mut invs = vec![];
        let mut mat_of = vec![];
        let mut mats = vec![];
        let mut textures: Vec<Arc<TexImage>> = vec![];
        let mut graphs: Vec<Arc<crate::nodes::NodeGraph>> = vec![];
        let mut graph_tex: std::collections::HashMap<u32, u16> = std::collections::HashMap::new();
        let mut tex_slot: Vec<(u32, u16)> = vec![];
        let mut opaque = vec![];
        for o in scene.objects.iter().filter(|o| o.visible && o.is_mesh()) {
            let mesh = cache.for_object(o);
            let m = o.material;
            let mut slot = |id: u32| -> Option<u16> {
                if id == 0 {
                    return None;
                }
                match tex_slot.iter().find(|(i, _)| *i == id) {
                    Some((_, s)) => Some(*s),
                    None => scene.texture(id).and_then(|t| tex_cache.get(t)).map(|img| {
                        textures.push(img);
                        let s = (textures.len() - 1) as u16;
                        tex_slot.push((id, s));
                        s
                    }),
                }
            };
            let g = scene.graph(m.graph);
            let (tex, normal_tex, rough_tex) = if g.is_some() { (None, None, None) } else { (slot(m.texture), slot(m.normal_tex), slot(m.rough_tex)) };
            let graph = g.map(|g| {
                for n in &g.nodes {
                    if let crate::nodes::NodeKind::Image { texture } = n.kind {
                        if let Some(s) = slot(texture) {
                            graph_tex.insert(texture, s);
                        }
                    }
                }
                match graphs.iter().position(|x: &Arc<crate::nodes::NodeGraph>| x.id == g.id) {
                    Some(i) => i as u16,
                    None => {
                        graphs.push(Arc::new(g.clone()));
                        (graphs.len() - 1) as u16
                    }
                }
            });
            let needs_frame = normal_tex.is_some() || g.is_some_and(|g| g.has_bump() || g.drives(5));
            let base = lin(m.color);
            mats.push(PtMat {
                albedo: base,
                metallic: m.metallic.clamp(0.0, 1.0),
                roughness: m.roughness.clamp(0.02, 1.0),
                emission: base * m.emission,
                alpha: m.alpha.clamp(0.0, 1.0),
                tex,
                normal_tex,
                normal_strength: m.normal_strength.clamp(0.0, 10.0),
                rough_tex,
                graph,
                emit: m.emission,
            });
            let mi = (mats.len() - 1) as u16;
            let tsc = m.tex_scale;
            for inst in o.instance_matrices() {
                let w = o.transform.matrix() * inst;
                let nm = Mat3::from_mat4(w).inverse().transpose();
                let flip = w.determinant() < 0.0;
                let wp: Vec<Vec3> = mesh.positions.iter().map(|p| w.transform_point3(*p)).collect();
                invs.push(w.inverse());
                let ii = (invs.len() - 1) as u32;
                for (ti, t) in mesh.tris.iter().enumerate() {
                    let tri = t.map(|i| wp[i as usize]);
                    let fnrm = (tri[1] - tri[0]).cross(tri[2] - tri[0]).normalize_or_zero();
                    let cn = mesh.corner_normals.get(ti).copied().unwrap_or([fnrm; 3]);
                    normals.push(if mesh.smooth { cn.map(|n| (nm * n).normalize_or_zero() * if flip { -1.0 } else { 1.0 }) } else { [fnrm * if flip { -1.0 } else { 1.0 }; 3] });
                    let uv = mesh.tri_uv.get(ti).map(|u| u.map(|c| [c[0] * tsc, c[1] * tsc])).unwrap_or([[0.0; 2]; 3]);
                    inv_of.push(ii);
                    tangents.push(if needs_frame { uv_frame(&tri, &uv) } else { [Vec3::ZERO; 2] });
                    uvs.push(uv);
                    mat_of.push(mi);
                    if m.alpha > 0.5 {
                        opaque.push(tri);
                    }
                    tris.push(tri);
                }
            }
        }
        // lights (same units as the Material Preview / raster renderer)
        let mut lights = vec![];
        for o in scene.objects.iter().filter(|o| o.visible) {
            let Some(l) = o.light() else { continue };
            let half = (l.spot_angle * 0.5).to_radians();
            lights.push(Light {
                kind: l.light_type,
                pos: o.transform.location,
                dir: o.forward(),
                radiance: lin(l.color) * l.power,
                cos_outer: half.cos(),
                cos_inner: (half * (1.0 - l.spot_blend.clamp(0.0, 1.0))).cos(),
                radius: if l.light_type == LightType::Sun { 0.02 } else { 0.1 },
            });
        }
        if lights.is_empty() {
            let w = &scene.world;
            lights.push(Light { kind: LightType::Sun, pos: Vec3::ZERO, dir: -w.sun_dir.normalize(), radiance: Vec3::splat(w.sun_strength * 3.0), cos_outer: 0.0, cos_inner: 0.0, radius: 0.02 });
        }
        let bg = scene.world.background;
        let n = tris.len();
        PtScene {
            bvh: Bvh::new(tris),
            shadow_bvh: Bvh::new(opaque),
            normals,
            uvs,
            tangents,
            inv_of,
            invs,
            mat_of,
            mats,
            textures,
            graphs,
            graph_tex,
            lights,
            ambient: scene.world.ambient,
            // the viewport background is a display colour: undo gamma (tone mapping is close to neutral there)
            bg_top: lin([(bg[0] * 1.12).min(1.0), (bg[1] * 1.12).min(1.0), (bg[2] * 1.12).min(1.0)]) * 1.15,
            bg_bot: lin([bg[0] * 0.82, bg[1] * 0.82, bg[2] * 0.82]) * 1.15,
            cam,
            tris: n,
        }
    }

    /// Sky radiance used for indirect light (same hemisphere colours as Material Preview).
    fn env(&self, d: Vec3) -> Vec3 {
        let sky = Vec3::new(0.55, 0.65, 0.85);
        let ground = Vec3::new(0.25, 0.22, 0.2);
        ground.lerp(sky, 0.5 + 0.5 * d.z) * self.ambient * 1.6
    }
}

// ---------------------------------------------------------------------------------------------

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed.wrapping_mul(0x9E3779B97F4A7C15) ^ 0xD1B54A32D192ED03)
    }
    #[inline]
    fn f(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / (1u64 << 24) as f32
    }
}

fn basis(n: Vec3) -> (Vec3, Vec3) {
    let a = if n.z.abs() < 0.9 { Vec3::Z } else { Vec3::X };
    let t = a.cross(n).normalize();
    (t, n.cross(t))
}

fn cosine_dir(n: Vec3, r: &mut Rng) -> Vec3 {
    let (u1, u2) = (r.f(), r.f());
    let (rr, phi) = (u1.sqrt(), 2.0 * PI * u2);
    let (t, b) = basis(n);
    (t * (rr * phi.cos()) + b * (rr * phi.sin()) + n * (1.0 - u1).max(0.0).sqrt()).normalize()
}

/// GGX half-vector sample (alpha = roughness²).
fn ggx_half(n: Vec3, alpha: f32, r: &mut Rng) -> Vec3 {
    let (u1, u2) = (r.f(), r.f());
    let a2 = alpha * alpha;
    let cos_t = ((1.0 - u1) / (1.0 + (a2 - 1.0) * u1)).max(0.0).sqrt();
    let sin_t = (1.0 - cos_t * cos_t).max(0.0).sqrt();
    let phi = 2.0 * PI * u2;
    let (t, b) = basis(n);
    (t * (sin_t * phi.cos()) + b * (sin_t * phi.sin()) + n * cos_t).normalize()
}

fn reflect(d: Vec3, n: Vec3) -> Vec3 {
    d - n * 2.0 * d.dot(n)
}

fn fresnel_dielectric(cos_i: f32, eta: f32) -> f32 {
    let sin_t2 = eta * eta * (1.0 - cos_i * cos_i);
    if sin_t2 >= 1.0 {
        return 1.0;
    }
    let cos_t = (1.0 - sin_t2).sqrt();
    let rs = (cos_i - eta * cos_t) / (cos_i + eta * cos_t);
    let rp = (eta * cos_i - cos_t) / (eta * cos_i + cos_t);
    0.5 * (rs * rs + rp * rp)
}

impl PtScene {
    fn trace(&self, mut o: Vec3, mut d: Vec3, r: &mut Rng, aux: &mut Option<(Vec3, Vec3)>) -> Vec3 {
        let mut through = Vec3::ONE;
        let mut out = Vec3::ZERO;
        // denoiser guides come from the first non-mirror, non-glass surface (seen through
        // reflections/refractions), like production denoisers do
        let mut aux_tint = Vec3::ONE;
        let mut aux_pending = true;
        for bounce in 0..8 {
            let Some(h) = self.bvh.intersect(o, d, 1e6) else {
                if aux_pending {
                    *aux = Some((aux_tint.max(Vec3::splat(0.02)), Vec3::ZERO));
                }
                // camera rays see the viewport-style background; bounces see the sky
                out += through * if bounce == 0 { self.bg_bot.lerp(self.bg_top, (d.z * 0.5 + 0.5).clamp(0.0, 1.0)) } else { self.env(d) };
                break;
            };
            let ti = h.tri as usize;
            let p = o + d * h.t;
            let w0 = 1.0 - h.u - h.v;
            let ns = self.normals[ti];
            let mut n = (ns[0] * w0 + ns[1] * h.u + ns[2] * h.v).normalize_or_zero();
            let mut m = self.mats[self.mat_of[ti] as usize];
            let mut albedo = m.albedo;
            let mut alpha = m.alpha;
            if m.tex.is_some() || m.normal_tex.is_some() || m.rough_tex.is_some() || m.graph.is_some() {
                let uv = self.uvs[ti];
                let tu = uv[0][0] * w0 + uv[1][0] * h.u + uv[2][0] * h.v;
                let tv = uv[0][1] * w0 + uv[1][1] * h.u + uv[2][1] * h.v;
                if let Some(gi) = m.graph {
                    let g = &self.graphs[gi as usize];
                    let tex = |id: u32, q: glam::Vec2| self.graph_tex.get(&id).map(|s| self.textures[*s as usize].sample(q.x, q.y));
                    let at = glam::Vec2::new(tu, tv);
                    let inv = self.invs[self.inv_of[ti] as usize];
                    let lp = |wp: Vec3| inv.transform_point3(wp);
                    let sh = g.eval(at, lp(p), &tex);
                    if let Some(c) = sh.color {
                        let c = c.max(Vec3::ZERO);
                        albedo = Vec3::new(c.x.powf(2.2), c.y.powf(2.2), c.z.powf(2.2));
                    }
                    if let Some(r) = sh.roughness {
                        m.roughness = r.clamp(0.02, 1.0);
                    }
                    if let Some(v) = sh.metallic {
                        m.metallic = v;
                    }
                    m.emission = albedo * sh.emission.unwrap_or(m.emit);
                    // the surface frame: dP/du, dP/dv from the UVs, or any frame when they're missing
                    let [tt, bb] = self.tangents[ti];
                    let (fu, fv, has_uv) = if tt.length_squared() > 1e-16 && bb.length_squared() > 1e-16 {
                        (tt, -bb, true)
                    } else {
                        let (a, b) = basis(n);
                        (a, b, false)
                    };
                    let side = n.dot(d).signum();
                    let keep_side = |pn: Vec3, n: Vec3| if pn.dot(d).signum() == side { pn } else { (pn - d * (pn.dot(d) - side * 0.01)).normalize_or(n) };
                    if let Some(h0) = sh.height {
                        // bump: height differences across the surface frame tilt the normal; each
                        // step moves both the UV and the 3D point so UV and Object coords agree
                        let (eps, step) = if has_uv { (1.0 / 1024.0, 1.0 / 1024.0) } else { (0.0, 1e-3) };
                        let du = (g.height(at + glam::Vec2::new(eps, 0.0), lp(p + fu * step), &tex) - h0) / step;
                        let dv = (g.height(at + glam::Vec2::new(0.0, eps), lp(p + fv * step), &tex) - h0) / step;
                        let grad = fu * (du / fu.length_squared().max(1e-12)) + fv * (dv / fv.length_squared().max(1e-12));
                        let gs = grad - n * n.dot(grad);
                        n = keep_side((n - gs * (sh.bump * 0.05)).normalize_or(n), n);
                    }
                    if let Some(tn) = sh.normal {
                        let t1 = (fu - n * n.dot(fu)).normalize_or_zero();
                        let bv = -fv;
                        let b1 = (bv - n * n.dot(bv) - t1 * t1.dot(bv)).normalize_or_zero();
                        if t1 != Vec3::ZERO && b1 != Vec3::ZERO {
                            n = keep_side((t1 * tn.x + b1 * tn.y + n * tn.z).normalize_or(n), n);
                        }
                    }
                }
                if let Some(t) = m.tex {
                    let c = self.textures[t as usize].sample(tu, tv);
                    albedo *= Vec3::new(c[0].powf(2.2), c[1].powf(2.2), c[2].powf(2.2));
                    alpha *= c[3];
                }
                if let Some(t) = m.rough_tex {
                    // linear data: G = roughness, B = metallic
                    let c = self.textures[t as usize].sample(tu, tv);
                    m.roughness = (m.roughness * c[1]).clamp(0.02, 1.0);
                    m.metallic = (m.metallic * c[2]).clamp(0.0, 1.0);
                }
                if let Some(t) = m.normal_tex {
                    let [tt, bb] = self.tangents[ti];
                    let tn = crate::textures::texel_normal(self.textures[t as usize].sample(tu, tv), m.normal_strength);
                    // Gram–Schmidt against the shading normal; keep the bitangent's handedness
                    let t1 = (tt - n * n.dot(tt)).normalize_or_zero();
                    let b1 = (bb - n * n.dot(bb) - t1 * t1.dot(bb)).normalize_or_zero();
                    if t1 != Vec3::ZERO && b1 != Vec3::ZERO {
                        let pn = (t1 * tn.x + b1 * tn.y + n * tn.z).normalize_or_zero();
                        // never tilt past the view direction (black speckles)
                        let side = n.dot(d).signum();
                        n = if pn.dot(d).signum() == side { pn } else { (pn - d * (pn.dot(d) - side * 0.01)).normalize_or(n) };
                    }
                }
            }
            let glassy = alpha < 0.999;
            let mirror = m.metallic > 0.5 && m.roughness < 0.3;
            if aux_pending && bounce < 4 {
                if glassy {
                    aux_tint *= albedo.lerp(Vec3::ONE, 0.6);
                } else if mirror {
                    aux_tint *= albedo;
                } else {
                    *aux = Some(((albedo * aux_tint).max(Vec3::splat(0.02)), n));
                    aux_pending = false;
                }
            }
            // emitters aren't light-sampled, so they always count when hit
            out += through * m.emission;
            let entering = n.dot(d) < 0.0;
            if !entering {
                n = -n;
            }
            // glass / transparency
            if alpha < 0.999 && r.f() > alpha {
                let eta = if entering { 1.0 / 1.45 } else { 1.45 };
                let cos_i = (-d).dot(n).clamp(0.0, 1.0);
                let f = fresnel_dielectric(cos_i, eta);
                if r.f() < f {
                    d = reflect(d, n);
                    o = p + n * 1e-3;
                } else {
                    let k = 1.0 - eta * eta * (1.0 - cos_i * cos_i);
                    d = (d * eta + n * (eta * cos_i - k.max(0.0).sqrt())).normalize();
                    o = p - n * 1e-3;
                    through *= albedo.lerp(Vec3::ONE, 0.6);
                }
                continue;
            }
            let v = -d;
            let f0 = Vec3::splat(0.04).lerp(albedo, m.metallic);
            let nv = n.dot(v).max(1e-4);
            let fres = f0 + (Vec3::ONE - f0) * (1.0 - nv).powf(5.0);
            let spec_w = fres.max_element().clamp(0.05, 1.0);
            let diff_w = (1.0 - m.metallic) * (1.0 - spec_w);
            let p_spec = spec_w / (spec_w + diff_w).max(1e-6);
            let a = m.roughness * m.roughness;
            // next-event estimation: diffuse + a glossy highlight from each light
            {
                let origin = p + n * 1e-3;
                for l in &self.lights {
                    let (ldir, dist, rad) = match l.kind {
                        LightType::Sun => {
                            let (t, b) = basis(-l.dir);
                            let j = (t * (r.f() - 0.5) + b * (r.f() - 0.5)) * l.radius;
                            ((-l.dir + j).normalize(), 1e6, l.radiance / 3.0)
                        }
                        _ => {
                            let jitter = Vec3::new(r.f() - 0.5, r.f() - 0.5, r.f() - 0.5) * 2.0 * l.radius;
                            let to = l.pos + jitter - origin;
                            let d2 = to.length_squared().max(0.01);
                            let ld = to / d2.sqrt();
                            let mut rad = l.radiance / (4.0 * PI * d2) * 0.9;
                            if l.kind == LightType::Spot {
                                let c = (-ld).dot(l.dir);
                                let t = ((c - l.cos_outer) / (l.cos_inner - l.cos_outer).max(1e-4)).clamp(0.0, 1.0);
                                rad *= t * t * (3.0 - 2.0 * t);
                            }
                            (ld, d2.sqrt() - 0.05, rad)
                        }
                    };
                    let ndl = n.dot(ldir);
                    if ndl <= 0.0 || rad.max_element() <= 0.0 {
                        continue;
                    }
                    if self.shadow_bvh.occluded(origin, ldir, dist) {
                        continue;
                    }
                    // raster units: diffuse = albedo * irradiance * cos
                    out += through * albedo * diff_w * rad * ndl;
                    // sharp highlights for glossy surfaces from the same light sample
                    let hh = (ldir + v).normalize();
                    let shin = 2.0 / (a * a).max(1e-4) + 2.0;
                    out += through * fres * rad * n.dot(hh).max(0.0).powf(shin) * (shin + 8.0) / 25.0 * ndl;
                }
            }
            // choose the next direction
            if r.f() < p_spec {
                let hv = ggx_half(n, a.max(0.002), r);
                let nd = reflect(d, hv);
                if nd.dot(n) <= 0.0 {
                    break;
                }
                d = nd;
                through *= fres / p_spec;
            } else {
                d = cosine_dir(n, r);
                through *= albedo * diff_w / (1.0 - p_spec).max(1e-3);
            }
            o = p + n * 1e-3;
            // Russian roulette
            if bounce >= 3 {
                let q = through.max_element().clamp(0.05, 0.95);
                if r.f() > q {
                    break;
                }
                through /= q;
            }
        }
        // clamp fireflies
        let mx = out.max_element();
        if mx > 12.0 { out * (12.0 / mx) } else { out }
    }

    /// One sample per pixel, added to `acc`.
    fn pass(&self, w: usize, h: usize, acc: &mut [Px], pass_idx: u32) {
        let aspect = w as f32 / h as f32;
        let cam = &self.cam;
        let (eye, fwd, right, up) = (cam.eye(), cam.forward(), cam.right(), cam.up());
        let th = (cam.fov_deg.to_radians() * 0.5).tan();
        let ortho_h = cam.half_height();
        let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).clamp(1, 32);
        let rows_per = h.div_ceil(threads * 4).max(1);
        let chunks: Vec<(usize, &mut [Px])> = acc.chunks_mut(rows_per * w).enumerate().collect();
        let next = std::sync::atomic::AtomicUsize::new(0);
        let chunks = std::sync::Mutex::new(chunks.into_iter().map(Some).collect::<Vec<_>>());
        std::thread::scope(|s| {
            for _ in 0..threads {
                s.spawn(|| loop {
                    let k = next.fetch_add(1, Ordering::Relaxed);
                    let item = {
                        let mut g = chunks.lock().unwrap();
                        if k >= g.len() {
                            break;
                        }
                        g[k].take()
                    };
                    let Some((ci, rows)) = item else { break };
                    for (i, px) in rows.iter_mut().enumerate() {
                        let idx = ci * rows_per * w + i;
                        let (x, y) = (idx % w, idx / w);
                        let mut r = Rng::new((idx as u64) << 20 ^ pass_idx as u64 ^ 0x5bd1e995);
                        let (sx, sy) = (x as f32 + r.f(), y as f32 + r.f());
                        let nx = sx / w as f32 * 2.0 - 1.0;
                        let ny = 1.0 - sy / h as f32 * 2.0;
                        let (o, d) = if cam.ortho {
                            (eye + right * (nx * ortho_h * aspect) + up * (ny * ortho_h) - fwd * 1000.0, fwd)
                        } else {
                            (eye, (fwd + right * (nx * th * aspect) + up * (ny * th)).normalize())
                        };
                        let mut aux = None;
                        px.c += self.trace(o, d, &mut r, &mut aux);
                        let (a, n) = aux.unwrap_or((Vec3::ONE, Vec3::ZERO));
                        px.a += a;
                        px.n += n;
                    }
                });
            }
        });
    }
}

fn aces(x: f32) -> u8 {
    let x = (x * 0.8).max(0.0);
    let y = (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14);
    (y.clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0 + 0.5) as u8
}

/// Accumulated samples of one pixel: radiance, first-hit albedo and normal (denoiser guides).
#[derive(Clone, Copy, Default)]
pub struct Px {
    c: Vec3,
    a: Vec3,
    n: Vec3,
}

/// Edge-aware à-trous wavelet denoiser on demodulated illumination, guided by albedo and normals.
pub fn denoise(acc: &[Px], w: usize, h: usize, n: u32) -> Vec<Vec3> {
    let inv = 1.0 / n.max(1) as f32;
    let alb: Vec<Vec3> = acc.iter().map(|p| p.a * inv).collect();
    let nrm: Vec<Vec3> = acc.iter().map(|p| p.n.normalize_or_zero()).collect();
    let mut ill: Vec<Vec3> = acc.iter().zip(&alb).map(|(p, a)| p.c * inv / a.max(Vec3::splat(0.02))).collect();
    let lum = |c: Vec3| c.dot(Vec3::new(0.2126, 0.7152, 0.0722));
    let kern = [1.0 / 16.0, 1.0 / 4.0, 3.0 / 8.0, 1.0 / 4.0, 1.0 / 16.0];
    // fewer samples → stronger filtering
    let sigma_l = (4.0 / (n as f32).sqrt()).clamp(0.08, 2.0);
    let iters = if n >= 256 { 3 } else { 5 };
    for it in 0..iters {
        let step = 1i32 << it;
        let src = ill.clone();
        let rows: Vec<usize> = (0..h).collect();
        let threads = std::thread::available_parallelism().map(|t| t.get()).unwrap_or(4).clamp(1, 32);
        let chunk = h.div_ceil(threads).max(1);
        let mut out = vec![Vec3::ZERO; w * h];
        std::thread::scope(|s| {
            for (rows, dst) in rows.chunks(chunk).zip(out.chunks_mut(chunk * w)) {
                let (src, alb, nrm) = (&src, &alb, &nrm);
                s.spawn(move || {
                    for (ri, &y) in rows.iter().enumerate() {
                        for x in 0..w {
                            let i = y * w + x;
                            let (cp, np, ap) = (src[i], nrm[i], alb[i]);
                            let lp = lum(cp);
                            let (mut sum, mut wsum) = (Vec3::ZERO, 0.0f32);
                            for (ky, kvy) in kern.iter().enumerate() {
                                let yy = y as i32 + (ky as i32 - 2) * step;
                                if yy < 0 || yy >= h as i32 {
                                    continue;
                                }
                                for (kx, kvx) in kern.iter().enumerate() {
                                    let xx = x as i32 + (kx as i32 - 2) * step;
                                    if xx < 0 || xx >= w as i32 {
                                        continue;
                                    }
                                    let j = yy as usize * w + xx as usize;
                                    let wn = if np == Vec3::ZERO || nrm[j] == Vec3::ZERO {
                                        if np == nrm[j] { 1.0 } else { 0.0 }
                                    } else {
                                        np.dot(nrm[j]).max(0.0).powi(32)
                                    };
                                    let da = ap - alb[j];
                                    let wa = (-da.length_squared() * 60.0).exp();
                                    let wl = (-(lp - lum(src[j])).abs() / (sigma_l * (lp + 0.05) * step as f32 * 0.5 + 1e-4)).exp();
                                    let wt = kvy * kvx * wn * wa * wl;
                                    sum += src[j] * wt;
                                    wsum += wt;
                                }
                            }
                            dst[ri * w + x] = if wsum > 1e-8 { sum / wsum } else { cp };
                        }
                    }
                });
            }
        });
        ill = out;
    }
    ill.iter().zip(&alb).map(|(l, a)| *l * a.max(Vec3::splat(0.02))).collect()
}

/// Tone-map accumulated radiance to RGBA8 (optionally denoised).
pub fn develop(acc: &[Px], w: usize, h: usize, n: u32, denoised: bool) -> Vec<u8> {
    let inv = 1.0 / n.max(1) as f32;
    let cols: Vec<Vec3> = if denoised && n >= 2 { denoise(acc, w, h, n) } else { acc.iter().map(|p| p.c * inv).collect() };
    let mut out = Vec::with_capacity(acc.len() * 4);
    for c in cols {
        out.extend_from_slice(&[aces(c.x), aces(c.y), aces(c.z), 255]);
    }
    out
}

pub struct PtUpdate {
    pub rgba: Vec<u8>,
    pub width: usize,
    pub height: usize,
    pub samples: u32,
    pub total: u32,
    pub secs: f32,
}

/// A running progressive render.
pub struct PtJob {
    pub rx: Receiver<PtUpdate>,
    cancel: Arc<AtomicBool>,
}

impl PtJob {
    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }
}

impl Drop for PtJob {
    fn drop(&mut self) {
        self.cancel();
    }
}

/// Start rendering in the background; an update is sent after each refinement step.
pub fn start(scene: PtScene, w: usize, h: usize, samples: u32, denoised: bool, repaint: impl Fn() + Send + 'static) -> PtJob {
    let (tx, rx) = unbounded();
    let cancel = Arc::new(AtomicBool::new(false));
    let c2 = cancel.clone();
    std::thread::Builder::new()
        .name("pathtrace".into())
        .spawn(move || {
            let t0 = std::time::Instant::now();
            let mut acc = vec![Px::default(); w * h];
            let total = samples.max(1);
            let mut last_denoise = std::time::Instant::now();
            let mut done = 0;
            let mut last_send = std::time::Instant::now();
            while done < total {
                if c2.load(Ordering::Relaxed) {
                    return;
                }
                scene.pass(w, h, &mut acc, done);
                done += 1;
                // stream often at first, then about twice a second
                if done <= 4 || done == total || last_send.elapsed().as_secs_f32() > 0.5 {
                    last_send = std::time::Instant::now();
                    // denoise the final image and, while refining, every couple of seconds
                    let dn = denoised && (done == total || last_denoise.elapsed().as_secs_f32() > 2.0 || done == 4);
                    if dn {
                        last_denoise = std::time::Instant::now();
                    }
                    if tx.send(PtUpdate { rgba: develop(&acc, w, h, done, dn), width: w, height: h, samples: done, total, secs: t0.elapsed().as_secs_f32() }).is_err() {
                        return;
                    }
                    repaint();
                }
            }
        })
        .expect("spawn path tracer");
    PtJob { rx, cancel }
}

/// Blocking render (for MCP / tests).
pub fn render_blocking(scene: &PtScene, w: usize, h: usize, samples: u32, denoised: bool) -> Vec<u8> {
    let mut acc = vec![Px::default(); w * h];
    for s in 0..samples.max(1) {
        scene.pass(w, h, &mut acc, s);
    }
    develop(&acc, w, h, samples.max(1), denoised)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn renders_default_scene() {
        let ed = crate::editor::Editor::default();
        let vp = crate::viewport::Viewport::default();
        let (cam, _) = vp.render_view(&ed, 16.0 / 9.0);
        let sc = PtScene::gather(&ed.scene, &mut MeshCache::default(), &mut TextureCache::default(), cam);
        assert_eq!(sc.tris, 12);
        let img = render_blocking(&sc, 64, 36, 4, true);
        let px = |x: usize, y: usize| &img[(y * 64 + x) * 4..(y * 64 + x) * 4 + 3];
        // the cube in the centre is lit, the corner is background
        assert_ne!(px(32, 18), px(1, 1));
        assert!(px(32, 18).iter().map(|c| *c as u32).sum::<u32>() > 60);
    }
}
