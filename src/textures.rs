//! Image and procedural textures: storage in the scene, decoding, procedural generators and
//! (bilinear, repeating) sampling for the renderers.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// A texture stored in the scene (and in .rblend files). Materials refer to it by `id`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Texture {
    pub id: u32,
    pub name: String,
    pub source: TexSource,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TexSource {
    /// Original PNG/JPEG file bytes (base64 in the saved file).
    Image {
        #[serde(with = "b64")]
        data: Vec<u8>,
    },
    /// Generated pattern (see `PROCEDURAL`), tinted with two colours.
    Procedural { pattern: String, color_a: [f32; 3], color_b: [f32; 3] },
}

mod b64 {
    use base64::Engine;
    use serde::{Deserialize, Deserializer, Serializer};
    pub fn serialize<S: Serializer>(v: &[u8], s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&base64::engine::general_purpose::STANDARD.encode(v))
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<u8>, D::Error> {
        let s = String::deserialize(d)?;
        base64::engine::general_purpose::STANDARD.decode(s.as_bytes()).map_err(serde::de::Error::custom)
    }
}

/// Built-in procedural patterns: (name, default colour A, default colour B).
pub const PROCEDURAL: &[(&str, [f32; 3], [f32; 3])] = &[
    ("checker", [0.9, 0.9, 0.9], [0.15, 0.15, 0.15]),
    ("bricks", [0.62, 0.25, 0.16], [0.78, 0.76, 0.72]),
    ("wood", [0.55, 0.33, 0.15], [0.33, 0.18, 0.07]),
    ("marble", [0.93, 0.93, 0.92], [0.35, 0.37, 0.42]),
    ("tiles", [0.85, 0.87, 0.9], [0.4, 0.42, 0.45]),
    ("grid", [0.8, 0.8, 0.82], [0.2, 0.45, 0.85]),
    ("noise", [0.8, 0.8, 0.8], [0.25, 0.25, 0.25]),
    ("stripes", [0.95, 0.95, 0.95], [0.85, 0.15, 0.15]),
    ("concrete", [0.62, 0.62, 0.6], [0.45, 0.45, 0.44]),
    ("grass", [0.28, 0.55, 0.16], [0.13, 0.32, 0.07]),
];

pub fn procedural_defaults(pattern: &str) -> Option<([f32; 3], [f32; 3])> {
    PROCEDURAL.iter().find(|(n, _, _)| *n == pattern).map(|(_, a, b)| (*a, *b))
}

/// Find a pattern by a loose name ("brick wall" → bricks, "wooden" → wood).
pub fn match_pattern(text: &str) -> Option<&'static str> {
    let t = text.to_lowercase();
    let aliases: &[(&str, &str)] = &[
        ("check", "checker"),
        ("brick", "bricks"),
        ("wood", "wood"),
        ("plank", "wood"),
        ("marble", "marble"),
        ("stone", "marble"),
        ("tile", "tiles"),
        ("grid", "grid"),
        ("noise", "noise"),
        ("stripe", "stripes"),
        ("concrete", "concrete"),
        ("cement", "concrete"),
        ("asphalt", "concrete"),
        ("grass", "grass"),
        ("lawn", "grass"),
    ];
    aliases.iter().find(|(k, _)| t.contains(k)).map(|(_, v)| *v)
}

/// Decoded texture, straight sRGB RGBA8.
pub struct TexImage {
    pub w: usize,
    pub h: usize,
    pub rgba: Vec<[u8; 4]>,
}

impl TexImage {
    /// Bilinear, repeating lookup; returns sRGB colour 0..1 (premultiplication not applied).
    #[inline]
    pub fn sample(&self, u: f32, v: f32) -> [f32; 4] {
        let (w, h) = (self.w as f32, self.h as f32);
        let x = (u - u.floor()) * w - 0.5;
        let y = (v - v.floor()) * h - 0.5;
        let (x0, y0) = (x.floor(), y.floor());
        let (fx, fy) = (x - x0, y - y0);
        let wrap = |i: f32, n: usize| -> usize { (i as i64).rem_euclid(n as i64) as usize };
        let (xa, xb) = (wrap(x0, self.w), wrap(x0 + 1.0, self.w));
        let (ya, yb) = (wrap(y0, self.h), wrap(y0 + 1.0, self.h));
        let p = |x: usize, y: usize| self.rgba[y * self.w + x];
        let (a, b, c, d) = (p(xa, ya), p(xb, ya), p(xa, yb), p(xb, yb));
        let mut out = [0.0; 4];
        for k in 0..4 {
            let top = a[k] as f32 * (1.0 - fx) + b[k] as f32 * fx;
            let bot = c[k] as f32 * (1.0 - fx) + d[k] as f32 * fx;
            out[k] = (top * (1.0 - fy) + bot * fy) / 255.0;
        }
        out
    }

    /// Box-filtered copy at a smaller size (for the GPU atlas).
    pub fn resized(&self, w: usize, h: usize) -> TexImage {
        let mut rgba = vec![[0u8; 4]; w * h];
        for y in 0..h {
            for x in 0..w {
                let (sx0, sx1) = (x * self.w / w, ((x + 1) * self.w / w).max(x * self.w / w + 1));
                let (sy0, sy1) = (y * self.h / h, ((y + 1) * self.h / h).max(y * self.h / h + 1));
                let mut acc = [0u32; 4];
                let mut n = 0;
                for sy in sy0..sy1.min(self.h) {
                    for sx in sx0..sx1.min(self.w) {
                        let p = self.rgba[sy * self.w + sx];
                        for k in 0..4 {
                            acc[k] += p[k] as u32;
                        }
                        n += 1;
                    }
                }
                let n = n.max(1);
                rgba[y * w + x] = [(acc[0] / n) as u8, (acc[1] / n) as u8, (acc[2] / n) as u8, (acc[3] / n) as u8];
            }
        }
        TexImage { w, h, rgba }
    }
}

pub fn decode(src: &TexSource) -> Result<TexImage, String> {
    match src {
        TexSource::Image { data } => {
            let img = image::load_from_memory(data).map_err(|e| format!("can't decode image: {e}"))?.to_rgba8();
            let (w, h) = (img.width() as usize, img.height() as usize);
            // keep memory and render time sane
            let img = TexImage { w, h, rgba: img.pixels().map(|p| p.0).collect() };
            Ok(if w > 2048 || h > 2048 {
                let s = 2048.0 / w.max(h) as f32;
                img.resized(((w as f32 * s) as usize).max(1), ((h as f32 * s) as usize).max(1))
            } else {
                img
            })
        }
        TexSource::Procedural { pattern, color_a, color_b } => Ok(procedural(pattern, *color_a, *color_b, 512)),
    }
}

fn hash(x: i32, y: i32, seed: u32) -> f32 {
    let mut h = (x as u32).wrapping_mul(374761393) ^ (y as u32).wrapping_mul(668265263) ^ seed.wrapping_mul(2246822519);
    h = (h ^ (h >> 13)).wrapping_mul(1274126177);
    ((h ^ (h >> 16)) & 0xffff) as f32 / 65535.0
}

/// Tileable value noise on a `period`-cell lattice.
fn vnoise(x: f32, y: f32, period: i32, seed: u32) -> f32 {
    let (x0, y0) = (x.floor(), y.floor());
    let (fx, fy) = (x - x0, y - y0);
    let s = |t: f32| t * t * (3.0 - 2.0 * t);
    let (sx, sy) = (s(fx), s(fy));
    let w = |i: i32| i.rem_euclid(period);
    let (xi, yi) = (x0 as i32, y0 as i32);
    let a = hash(w(xi), w(yi), seed);
    let b = hash(w(xi + 1), w(yi), seed);
    let c = hash(w(xi), w(yi + 1), seed);
    let d = hash(w(xi + 1), w(yi + 1), seed);
    (a * (1.0 - sx) + b * sx) * (1.0 - sy) + (c * (1.0 - sx) + d * sx) * sy
}

fn fbm(u: f32, v: f32, base: i32, octaves: u32, seed: u32) -> f32 {
    let (mut sum, mut amp, mut norm) = (0.0, 0.5, 0.0);
    let mut p = base;
    for o in 0..octaves {
        sum += vnoise(u * p as f32, v * p as f32, p, seed + o) * amp;
        norm += amp;
        amp *= 0.5;
        p *= 2;
    }
    sum / norm
}

/// Generate a tileable `size`×`size` pattern.
pub fn procedural(pattern: &str, a: [f32; 3], b: [f32; 3], size: usize) -> TexImage {
    let mix = |t: f32| -> [u8; 4] {
        let t = t.clamp(0.0, 1.0);
        let c = |k: usize| ((a[k] * (1.0 - t) + b[k] * t).clamp(0.0, 1.0) * 255.0) as u8;
        [c(0), c(1), c(2), 255]
    };
    let mut rgba = Vec::with_capacity(size * size);
    for y in 0..size {
        for x in 0..size {
            let (u, v) = ((x as f32 + 0.5) / size as f32, (y as f32 + 0.5) / size as f32);
            let t = match pattern {
                "checker" => (((u * 8.0) as i32 + (v * 8.0) as i32) % 2) as f32,
                "bricks" => {
                    let rows = 8.0;
                    let row = (v * rows).floor();
                    let off = if row as i32 % 2 == 0 { 0.0 } else { 0.5 };
                    let bu = (u * 4.0 + off).fract();
                    let bv = (v * rows).fract();
                    let mortar = bu < 0.04 || bu > 0.96 || bv < 0.08 || bv > 0.92;
                    if mortar { 1.0 } else { fbm(u, v, 16, 3, row as u32) * 0.35 }
                }
                "wood" => {
                    // planks running along v, with end joints, grain lines and per-plank tone
                    let planks = 4.0;
                    let pi = (u * planks).floor();
                    let pu = (u * planks).fract();
                    let pv = v * 2.0 + hash(pi as i32, 1, 9);
                    let joint = pv.fract() < 0.006 || pv.fract() > 0.994;
                    let gap = !(0.012..=0.988).contains(&pu) || joint;
                    let seg = pv.floor();
                    let warp = fbm(u * 0.5, v * 0.25, 4, 3, 7 + pi as u32) * 0.6;
                    let rings = ((pu * 1.6 + warp + seg * 0.31) * 38.0).sin() * 0.5 + 0.5;
                    let grain = fbm(u * 2.0, v * 0.125, 16, 3, 11);
                    let tone = hash(pi as i32, seg as i32, 5);
                    if gap { 1.0 } else { (rings.powf(1.5) * 0.4 + grain * 0.35 + tone * 0.3).min(0.95) }
                }
                "marble" => {
                    let n = fbm(u, v, 4, 6, 3);
                    let s = ((u * 0.6 + v) * 9.0 + n * 7.0).sin().abs();
                    let veins = (1.0 - s).powf(10.0);
                    (veins * 0.85 + fbm(u, v, 8, 4, 13) * 0.18).min(1.0)
                }
                "tiles" => {
                    let (tu, tv) = ((u * 6.0).fract(), (v * 6.0).fract());
                    if !(0.04..=0.96).contains(&tu) || !(0.04..=0.96).contains(&tv) { 1.0 } else { fbm(u, v, 32, 2, 9) * 0.15 }
                }
                "grid" => {
                    let (gu, gv) = ((u * 8.0).fract(), (v * 8.0).fract());
                    let line = !(0.03..=0.97).contains(&gu) || !(0.03..=0.97).contains(&gv);
                    let major = !(0.012..=0.988).contains(&u) || !(0.012..=0.988).contains(&v);
                    if line || major { 1.0 } else { 0.0 }
                }
                "noise" => fbm(u, v, 8, 5, 1),
                "stripes" => {
                    if (v * 10.0).fract() < 0.5 { 0.0 } else { 1.0 }
                }
                "concrete" => (fbm(u, v, 16, 5, 21) * 1.3 - 0.15 + if hash(x as i32, y as i32, 3) > 0.985 { 0.6 } else { 0.0 }).clamp(0.0, 1.0),
                "grass" => {
                    let n = fbm(u, v, 32, 3, 4);
                    let blades = fbm(u * 4.0, v * 0.5, 64, 2, 8);
                    (n * 0.6 + blades * 0.6).clamp(0.0, 1.0)
                }
                _ => (((u * 8.0) as i32 + (v * 8.0) as i32) % 2) as f32,
            };
            rgba.push(mix(t));
        }
    }
    TexImage { w: size, h: size, rgba }
}

/// Encode straight RGBA8 as PNG.
pub fn encode_png(w: usize, h: usize, rgba: &[u8]) -> Result<Vec<u8>, String> {
    let mut buf = Vec::new();
    {
        let mut enc = png::Encoder::new(&mut buf, w as u32, h as u32);
        enc.set_color(png::ColorType::Rgba);
        enc.set_depth(png::BitDepth::Eight);
        let mut wr = enc.write_header().map_err(|e| e.to_string())?;
        wr.write_image_data(rgba).map_err(|e| e.to_string())?;
    }
    Ok(buf)
}

/// Cheap change signature of a texture source (for the viewport's redraw check).
pub fn signature(src: &TexSource) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    match src {
        TexSource::Image { data } => {
            data.len().hash(&mut h);
            for i in (0..data.len()).step_by((data.len() / 64).max(1)) {
                data[i].hash(&mut h);
            }
        }
        TexSource::Procedural { pattern, color_a, color_b } => {
            pattern.hash(&mut h);
            for c in color_a.iter().chain(color_b) {
                c.to_bits().hash(&mut h);
            }
        }
    }
    h.finish()
}

/// Decoded textures by id, invalidated when a texture's source changes.
#[derive(Default)]
pub struct TextureCache {
    map: HashMap<u32, (TexSource, Arc<TexImage>)>,
    baked_more: Vec<(BakeKey, Arc<TexImage>)>,
}

type BakeKey = (Option<(u32, u64)>, u32, u64, u32);

impl TextureCache {
    pub fn get(&mut self, t: &Texture) -> Option<Arc<TexImage>> {
        if let Some((src, img)) = self.map.get(&t.id) {
            if *src == t.source {
                return Some(img.clone());
            }
        }
        let img = Arc::new(decode(&t.source).ok()?);
        self.map.insert(t.id, (t.source.clone(), img.clone()));
        Some(img)
    }
}

// ---- normal / bump maps ----------------------------------------------------------------------

/// Tangent-space normal from a normal-map texel (OpenGL / glTF convention: +Y = image up),
/// with the bumps scaled by `strength`.
#[inline]
pub fn texel_normal(c: [f32; 4], strength: f32) -> glam::Vec3 {
    let x = (c[0] * 2.0 - 1.0) * strength;
    let y = (c[1] * 2.0 - 1.0) * strength;
    let z = (c[2] * 2.0 - 1.0).max(0.05);
    glam::Vec3::new(x, y, z).normalize_or(glam::Vec3::Z)
}

/// Height (0..1) per pixel: procedural patterns use their own mask (mortar / grout low),
/// images their luminance.
pub fn height_of(src: &TexSource) -> Result<TexImage, String> {
    match src {
        TexSource::Procedural { pattern, .. } => Ok(procedural(pattern, [1.0; 3], [0.0; 3], 512)),
        TexSource::Image { .. } => {
            let img = decode(src)?;
            let s = 1024.0 / img.w.max(img.h) as f32;
            let img = if s < 1.0 { img.resized(((img.w as f32 * s) as usize).max(1), ((img.h as f32 * s) as usize).max(1)) } else { img };
            let rgba = img
                .rgba
                .iter()
                .map(|p| {
                    let l = (0.2126 * p[0] as f32 + 0.7152 * p[1] as f32 + 0.0722 * p[2] as f32) as u8;
                    [l, l, l, 255]
                })
                .collect();
            Ok(TexImage { w: img.w, h: img.h, rgba })
        }
    }
}

/// Tileable normal map from a height map (red channel). `depth` is the bump height in pixels.
pub fn height_to_normal(h: &TexImage, depth: f32, invert: bool) -> TexImage {
    let (w, hh) = (h.w, h.h);
    // soften hard steps (mortar lines) into bevels a few pixels wide, so the bumps read at
    // render size: two passes of a tileable box blur
    let mut f: Vec<f32> = h.rgba.iter().map(|p| { let v = p[0] as f32 / 255.0; if invert { 1.0 - v } else { v } }).collect();
    let r = (w.max(hh) / 170).max(1) as i64;
    for _ in 0..2 {
        let mut t = vec![0.0; f.len()];
        for y in 0..hh {
            for x in 0..w as i64 {
                t[y * w + x as usize] = (-r..=r).map(|k| f[y * w + (x + k).rem_euclid(w as i64) as usize]).sum::<f32>() / (2 * r + 1) as f32;
            }
        }
        for y in 0..hh as i64 {
            for x in 0..w {
                f[y as usize * w + x] = (-r..=r).map(|k| t[(y + k).rem_euclid(hh as i64) as usize * w + x]).sum::<f32>() / (2 * r + 1) as f32;
            }
        }
    }
    let at = |x: i64, y: i64| -> f32 { f[(y.rem_euclid(hh as i64) as usize) * w + x.rem_euclid(w as i64) as usize] };
    let mut rgba = Vec::with_capacity(w * hh);
    for y in 0..hh as i64 {
        for x in 0..w as i64 {
            // Sobel gradient
            let dx = (at(x + 1, y - 1) + 2.0 * at(x + 1, y) + at(x + 1, y + 1)) - (at(x - 1, y - 1) + 2.0 * at(x - 1, y) + at(x - 1, y + 1));
            let dy = (at(x - 1, y + 1) + 2.0 * at(x, y + 1) + at(x + 1, y + 1)) - (at(x - 1, y - 1) + 2.0 * at(x, y - 1) + at(x + 1, y - 1));
            // image y runs down; tangent-space +Y is up
            let n = glam::Vec3::new(-dx * depth / 8.0, dy * depth / 8.0, 1.0).normalize();
            let q = |c: f32| ((c * 0.5 + 0.5) * 255.0).round().clamp(0.0, 255.0) as u8;
            rgba.push([q(n.x), q(n.y), q(n.z), 255]);
        }
    }
    TexImage { w, h: hh, rgba }
}

/// Generate a normal map (PNG bytes) for a texture: "make it bumpy".
pub fn normal_map_png(src: &TexSource, depth: f32, invert: bool) -> Result<Vec<u8>, String> {
    let n = height_to_normal(&height_of(src)?, depth, invert);
    let flat: Vec<u8> = n.rgba.iter().flatten().copied().collect();
    encode_png(n.w, n.h, &flat)
}

/// Normal maps are mostly blue-violet: x, y around the middle and z high.
pub fn looks_like_normal_map(img: &TexImage) -> bool {
    let n = img.rgba.len().max(1) as f32;
    let step = (img.rgba.len() / 4096).max(1);
    let (mut r, mut g, mut b, mut k) = (0.0, 0.0, 0.0, 0.0);
    for p in img.rgba.iter().step_by(step) {
        r += p[0] as f32;
        g += p[1] as f32;
        b += p[2] as f32;
        k += 1.0;
    }
    let _ = n;
    let (r, g, b) = (r / k / 255.0, g / k / 255.0, b / k / 255.0);
    (r - 0.5).abs() < 0.12 && (g - 0.5).abs() < 0.12 && b > 0.75
}

/// Roughness/metallic map (glTF layout) from a pattern: rough in the low parts.
pub fn roughness_map_png(src: &TexSource) -> Result<Vec<u8>, String> {
    let h = height_of(src)?;
    let flat: Vec<u8> = h.rgba.iter().flat_map(|p| [255, (255.0 * (1.0 - 0.45 * p[0] as f32 / 255.0)) as u8, 255, 255]).collect();
    encode_png(h.w, h.h, &flat)
}

/// Bake the look of a normal map into a colour texture for the raster renderers (which shade
/// per vertex): base × lighting of the bumps by a soft key light from the upper left.
pub fn bake_bumps(base: Option<&TexImage>, normal: &TexImage, strength: f32) -> TexImage {
    let (w, h) = match base {
        Some(b) => (b.w.max(normal.w).min(1024), b.h.max(normal.h).min(1024)),
        None => (normal.w, normal.h),
    };
    let l = glam::Vec3::new(-0.4, 0.45, 0.8).normalize();
    let flat = 0.45 + 0.55 * l.z;
    let mut rgba = Vec::with_capacity(w * h);
    for y in 0..h {
        for x in 0..w {
            let (u, v) = ((x as f32 + 0.5) / w as f32, (y as f32 + 0.5) / h as f32);
            let n = texel_normal(normal.sample(u, v), strength);
            let k = (0.45 + 0.55 * n.dot(l).max(0.0)) / flat;
            let c = base.map(|b| b.sample(u, v)).unwrap_or([1.0; 4]);
            let q = |c: f32| (c * k * 255.0).round().clamp(0.0, 255.0) as u8;
            rgba.push([q(c[0]), q(c[1]), q(c[2]), (c[3] * 255.0) as u8]);
        }
    }
    TexImage { w, h, rgba }
}

impl TextureCache {
    /// Base texture with a normal map's shading baked in (cached).
    pub fn baked(&mut self, base: Option<&Texture>, normal: &Texture, strength: f32) -> Option<Arc<TexImage>> {
        let key = (base.map(|b| (b.id, signature(&b.source))), normal.id, signature(&normal.source), strength.to_bits());
        if let Some((_, img)) = self.baked_more.iter().find(|(k, _)| *k == key) {
            return Some(img.clone());
        }
        let b = match base {
            Some(t) => Some(self.get(t)?),
            None => None,
        };
        let n = self.get(normal)?;
        let img = Arc::new(bake_bumps(b.as_deref(), &n, strength));
        self.baked_more.push((key, img.clone()));
        if self.baked_more.len() > 16 {
            self.baked_more.remove(0);
        }
        Some(img)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn patterns_and_sampling() {
        for (name, a, b) in PROCEDURAL {
            let t = procedural(name, *a, *b, 64);
            assert_eq!(t.rgba.len(), 64 * 64, "{name}");
            // not a flat colour
            let first = t.rgba[0];
            assert!(t.rgba.iter().any(|p| *p != first), "{name} should vary");
        }
        let c = procedural("checker", [1.0; 3], [0.0; 3], 64);
        assert!(c.sample(0.01, 0.01)[0] > 0.9);
        assert!(c.sample(0.18, 0.01)[0] < 0.1);
        assert!((c.sample(1.01, 2.01)[0] - c.sample(0.01, 0.01)[0]).abs() < 1e-5, "repeats");
        assert_eq!(match_pattern("an old brick wall"), Some("bricks"));
        assert_eq!(match_pattern("wooden floor"), Some("wood"));
        assert_eq!(match_pattern("glass"), None);
        // PNG round trip through the scene format
        let mut png = vec![];
        {
            let mut e = png::Encoder::new(&mut png, 2, 1);
            e.set_color(png::ColorType::Rgba);
            e.set_depth(png::BitDepth::Eight);
            e.write_header().unwrap().write_image_data(&[255, 0, 0, 255, 0, 0, 255, 255]).unwrap();
        }
        let t = Texture { id: 1, name: "img".into(), source: TexSource::Image { data: png } };
        let js = serde_json::to_string(&t).unwrap();
        let back: Texture = serde_json::from_str(&js).unwrap();
        assert_eq!(back, t);
        let img = decode(&back.source).unwrap();
        assert_eq!((img.w, img.h), (2, 1));
        assert_eq!(img.rgba[0], [255, 0, 0, 255]);
    }
    #[test]
    fn normal_maps() {
        // a height ramp rising to the right tilts normals to the left (-x)
        let w = 32;
        let rgba = (0..w * w).map(|i| { let l = ((i % w) * 255 / (w - 1)) as u8; [l, l, l, 255] }).collect();
        let n = height_to_normal(&TexImage { w, h: w, rgba }, 4.0, false);
        let c = n.rgba[w * 10 + 10];
        assert!(c[0] < 120 && (c[1] as i32 - 128).abs() < 3 && c[2] > 200, "{c:?}");
        assert!(looks_like_normal_map(&n));
        assert!(!looks_like_normal_map(&procedural("bricks", [0.6, 0.2, 0.1], [0.8; 3], 64)));
        let t = texel_normal([c[0] as f32 / 255.0, c[1] as f32 / 255.0, c[2] as f32 / 255.0, 1.0], 1.0);
        assert!(t.x < -0.1 && t.z > 0.5);
        // bricks: generated map decodes, baked shading darkens one side of the mortar lines
        let src = TexSource::Procedural { pattern: "bricks".into(), color_a: [0.6, 0.2, 0.1], color_b: [0.8; 3] };
        let png = normal_map_png(&src, 8.0, false).unwrap();
        let nm = decode(&TexSource::Image { data: png }).unwrap();
        assert!(looks_like_normal_map(&nm));
        let baked = bake_bumps(None, &nm, 1.0);
        let (lo, hi) = baked.rgba.iter().fold((255u8, 0u8), |(l, h), p| (l.min(p[0]), h.max(p[0])));
        assert!(lo < 200 && hi > 240, "{lo} {hi}");
        let r = decode(&TexSource::Image { data: roughness_map_png(&src).unwrap() }).unwrap();
        assert!(r.rgba.iter().any(|p| p[1] > 250) && r.rgba.iter().any(|p| p[1] < 160));
    }
}
