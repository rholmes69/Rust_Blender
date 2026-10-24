//! Multithreaded software rasterizer with a real depth buffer, object-ID buffer
//! (pixel-exact picking) and Blender-style selection outlines computed from the ID buffer.

use egui::Color32;

#[derive(Clone, Copy)]
pub struct RPrim {
    /// Pixel-space positions (x, y), NDC depth z in [0, 1].
    pub p: [[f32; 3]; 4],
    pub c: [[f32; 4]; 4],
    /// 3 = triangle, 4 = quad (thick line)
    pub n: u8,
    /// object index + 1 (0 = not an object, e.g. grid line)
    pub id: u32,
    /// alpha-blended, depth-tested but no depth write
    pub blend: bool,
    /// ignore the depth test (x-ray wire)
    pub no_depth: bool,
    /// Texture coordinates per corner and texture (index + 1 into `Raster::textures`; 0 = none).
    pub uv: [[f32; 2]; 4],
    pub tex: u32,
    /// 1 / clip-space w per corner, for perspective-correct texturing (1 = affine).
    pub iw: [f32; 4],
}

impl RPrim {
    pub const NO_UV: [[f32; 2]; 4] = [[0.0; 2]; 4];
}

pub struct Raster {
    pub w: usize,
    pub h: usize,
    pub color: Vec<Color32>,
    pub depth: Vec<f32>,
    pub id: Vec<u32>,
    /// Textures referenced by `RPrim::tex`.
    pub textures: Vec<std::sync::Arc<crate::textures::TexImage>>,
}

impl Default for Raster {
    fn default() -> Self {
        Self {
            w: 0,
            h: 0,
            color: vec![],
            depth: vec![],
            id: vec![],
            textures: vec![],
        }
    }
}

pub struct Outline {
    /// per object index+1 -> 0 none, 1 selected, 2 active
    pub sel: Vec<u8>,
    pub selected: Color32,
    pub active: Color32,
    pub width: i32,
}

impl Raster {
    pub fn resize(&mut self, w: usize, h: usize) {
        if self.w != w || self.h != h {
            self.w = w;
            self.h = h;
            self.color = vec![Color32::BLACK; w * h];
            self.depth = vec![1.0; w * h];
            self.id = vec![0; w * h];
        }
    }

    pub fn id_at(&self, x: usize, y: usize) -> u32 {
        if x < self.w && y < self.h {
            self.id[y * self.w + x]
        } else {
            0
        }
    }

    /// Render all primitives. Opaque first, then blended (in given order).
    pub fn render(
        &mut self,
        bg_top: [f32; 3],
        bg_bot: [f32; 3],
        prims: &[RPrim],
        outline: Option<&Outline>,
    ) {
        let (w, h) = (self.w, self.h);
        if w == 0 || h == 0 {
            return;
        }
        let threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
            .clamp(1, 16);
        let band = h.div_ceil(threads).max(8);
        let opaque: Vec<&RPrim> = prims.iter().filter(|p| !p.blend).collect();
        let blended: Vec<&RPrim> = prims.iter().filter(|p| p.blend).collect();
        let color_chunks = self.color.chunks_mut(band * w);
        let depth_chunks = self.depth.chunks_mut(band * w);
        let id_chunks = self.id.chunks_mut(band * w);
        std::thread::scope(|s| {
            for (bi, ((col, dep), ids)) in color_chunks.zip(depth_chunks).zip(id_chunks).enumerate()
            {
                let y0 = bi * band;
                let rows = col.len() / w;
                let opaque = &opaque;
                let blended = &blended;
                let textures = &self.textures;
                s.spawn(move || {
                    // clear with vertical gradient
                    for r in 0..rows {
                        let t = (y0 + r) as f32 / h as f32;
                        let c = [
                            bg_top[0] + (bg_bot[0] - bg_top[0]) * t,
                            bg_top[1] + (bg_bot[1] - bg_top[1]) * t,
                            bg_top[2] + (bg_bot[2] - bg_top[2]) * t,
                        ];
                        let c = Color32::from_rgb(
                            (c[0] * 255.0) as u8,
                            (c[1] * 255.0) as u8,
                            (c[2] * 255.0) as u8,
                        );
                        col[r * w..(r + 1) * w].fill(c);
                    }
                    dep.fill(1.0);
                    ids.fill(0);
                    let mut band = Band {
                        w,
                        y0,
                        y1: y0 + rows,
                        col,
                        dep,
                        ids,
                        textures,
                    };
                    for p in opaque.iter() {
                        band.prim(p);
                    }
                    for p in blended.iter() {
                        band.prim(p);
                    }
                });
            }
        });
        if let Some(o) = outline {
            self.outline(o);
        }
    }

    fn outline(&mut self, o: &Outline) {
        let (w, h) = (self.w as i32, self.h as i32);
        let sel = |id: u32| -> u8 { o.sel.get(id as usize).copied().unwrap_or(0) };
        if !o.sel.iter().any(|s| *s > 0) {
            return;
        }
        let r = o.width;
        let mut offs = vec![];
        for dy in -r..=r {
            for dx in -r..=r {
                if (dx != 0 || dy != 0) && dx * dx + dy * dy <= r * r + 1 {
                    offs.push((dx, dy));
                }
            }
        }
        let ids = &self.id;
        // restrict work to the bounding box of selected pixels
        let (mut bx0, mut by0, mut bx1, mut by1) = (w, h, -1, -1);
        for y in 0..h {
            let row = &ids[(y * w) as usize..((y + 1) * w) as usize];
            for (x, id) in row.iter().enumerate() {
                if sel(*id) > 0 {
                    let x = x as i32;
                    bx0 = bx0.min(x);
                    bx1 = bx1.max(x);
                    by0 = by0.min(y);
                    by1 = by1.max(y);
                }
            }
        }
        if bx1 < 0 {
            return;
        }
        let mut out: Vec<(usize, Color32)> = vec![];
        for y in (by0 - r).max(0)..=(by1 + r).min(h - 1) {
            for x in (bx0 - r).max(0)..=(bx1 + r).min(w - 1) {
                let i = (y * w + x) as usize;
                let me = ids[i];
                if sel(me) > 0 && x > 0 && y > 0 && x < w - 1 && y < h - 1 {
                    continue;
                }
                let mut best = 0u8;
                for (dx, dy) in &offs {
                    let (nx, ny) = (x + dx, y + dy);
                    if nx < 0 || ny < 0 || nx >= w || ny >= h {
                        continue;
                    }
                    let nid = ids[(ny * w + nx) as usize];
                    if nid != me {
                        best = best.max(sel(nid));
                        if best == 2 {
                            break;
                        }
                    }
                }
                if best > 0 {
                    out.push((i, if best == 2 { o.active } else { o.selected }));
                }
            }
        }
        for (i, c) in out {
            self.color[i] = c;
        }
    }
}

struct Band<'a> {
    w: usize,
    y0: usize,
    y1: usize,
    col: &'a mut [Color32],
    dep: &'a mut [f32],
    ids: &'a mut [u32],
    textures: &'a [std::sync::Arc<crate::textures::TexImage>],
}

#[inline]
fn edge(a: [f32; 3], b: [f32; 3], x: f32, y: f32) -> f32 {
    (b[0] - a[0]) * (y - a[1]) - (b[1] - a[1]) * (x - a[0])
}

impl Band<'_> {
    fn prim(&mut self, p: &RPrim) {
        self.tri(p, 0, 1, 2);
        if p.n == 4 {
            self.tri(p, 0, 2, 3);
        }
    }

    fn tri(&mut self, p: &RPrim, i0: usize, i1: usize, i2: usize) {
        let (a, b, c) = (p.p[i0], p.p[i1], p.p[i2]);
        let area = edge(a, b, c[0], c[1]);
        if area.abs() < 1e-8 {
            return;
        }
        let minx = a[0].min(b[0]).min(c[0]).floor().max(0.0) as i64;
        let maxx = (a[0].max(b[0]).max(c[0]).ceil() as i64).min(self.w as i64 - 1);
        let miny = (a[1].min(b[1]).min(c[1]).floor() as i64).max(self.y0 as i64);
        let maxy = (a[1].max(b[1]).max(c[1]).ceil() as i64).min(self.y1 as i64 - 1);
        if minx > maxx || miny > maxy {
            return;
        }
        let inv = 1.0 / area;
        let (ca, cb, cc) = (p.c[i0], p.c[i1], p.c[i2]);
        let tex = if p.tex > 0 { self.textures.get(p.tex as usize - 1) } else { None };
        let (ua, ub, uc) = (p.uv[i0], p.uv[i1], p.uv[i2]);
        let (wa, wb, wc) = (p.iw[i0], p.iw[i1], p.iw[i2]);
        let flat = ca == cb && cb == cc && tex.is_none();
        let flat_px = if flat {
            to_px(ca)
        } else {
            Color32::TRANSPARENT
        };
        let blend = p.blend;
        for y in miny..=maxy {
            let py = y as f32 + 0.5;
            let row = (y as usize - self.y0) * self.w;
            for x in minx..=maxx {
                let px = x as f32 + 0.5;
                let w0 = edge(b, c, px, py) * inv;
                let w1 = edge(c, a, px, py) * inv;
                let w2 = 1.0 - w0 - w1;
                if w0 < 0.0 || w1 < 0.0 || w2 < 0.0 {
                    continue;
                }
                let z = w0 * a[2] + w1 * b[2] + w2 * c[2];
                let i = row + x as usize;
                if !p.no_depth && (z > self.dep[i] || z < 0.0) {
                    continue;
                }
                // texture modulation (perspective-correct UVs)
                let texel = tex.map(|t| {
                    let (q0, q1, q2) = (w0 * wa, w1 * wb, w2 * wc);
                    let d = 1.0 / (q0 + q1 + q2).max(1e-12);
                    t.sample((q0 * ua[0] + q1 * ub[0] + q2 * uc[0]) * d, (q0 * ua[1] + q1 * ub[1] + q2 * uc[1]) * d)
                });
                if blend {
                    let mut s = if flat {
                        ca
                    } else {
                        [
                            ca[0] * w0 + cb[0] * w1 + cc[0] * w2,
                            ca[1] * w0 + cb[1] * w1 + cc[1] * w2,
                            ca[2] * w0 + cb[2] * w1 + cc[2] * w2,
                            ca[3] * w0 + cb[3] * w1 + cc[3] * w2,
                        ]
                    };
                    if let Some(t) = texel {
                        for k in 0..4 {
                            s[k] *= t[k];
                        }
                    }
                    let d = self.col[i];
                    let al = s[3].clamp(0.0, 1.0);
                    let mix = |sv: f32, dv: u8| {
                        ((sv * al + dv as f32 / 255.0 * (1.0 - al)) * 255.0).clamp(0.0, 255.0) as u8
                    };
                    self.col[i] =
                        Color32::from_rgb(mix(s[0], d.r()), mix(s[1], d.g()), mix(s[2], d.b()));
                    if p.id != 0 && al > 0.3 {
                        self.ids[i] = p.id;
                    }
                } else {
                    self.dep[i] = z;
                    self.ids[i] = p.id;
                    self.col[i] = if flat {
                        flat_px
                    } else {
                        let t = texel.unwrap_or([1.0; 4]);
                        to_px([
                            (ca[0] * w0 + cb[0] * w1 + cc[0] * w2) * t[0],
                            (ca[1] * w0 + cb[1] * w1 + cc[1] * w2) * t[1],
                            (ca[2] * w0 + cb[2] * w1 + cc[2] * w2) * t[2],
                            1.0,
                        ])
                    };
                }
            }
        }
    }
}

#[inline]
fn to_px(c: [f32; 4]) -> Color32 {
    Color32::from_rgb(
        (c[0].clamp(0.0, 1.0) * 255.0) as u8,
        (c[1].clamp(0.0, 1.0) * 255.0) as u8,
        (c[2].clamp(0.0, 1.0) * 255.0) as u8,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn depth_test_and_ids() {
        let mut r = Raster::default();
        r.resize(64, 64);
        let tri = |z: f32, id: u32, col: f32| RPrim {
            p: [
                [0.0, 0.0, z],
                [64.0, 0.0, z],
                [0.0, 64.0, z],
                [0.0, 64.0, z],
            ],
            c: [[col, 0.0, 0.0, 1.0]; 4],
            n: 3,
            id,
            blend: false,
            no_depth: false,
            uv: RPrim::NO_UV,
            tex: 0,
            iw: [1.0; 4],
        };
        // far drawn after near must not overwrite
        r.render(
            [0.0; 3],
            [0.0; 3],
            &[tri(0.2, 1, 1.0), tri(0.8, 2, 0.5)],
            None,
        );
        assert_eq!(r.id_at(5, 5), 1);
        assert_eq!(r.color[5 * 64 + 5].r(), 255);
        assert_eq!(r.id_at(60, 60), 0);
        let o = Outline {
            sel: vec![0, 1, 0],
            selected: Color32::YELLOW,
            active: Color32::RED,
            width: 2,
        };
        r.render([0.0; 3], [0.0; 3], &[tri(0.2, 1, 1.0)], Some(&o));
        // a pixel just outside the hypotenuse gets outlined
        assert_eq!(r.color[33 * 64 + 33], Color32::YELLOW);
    }
}
