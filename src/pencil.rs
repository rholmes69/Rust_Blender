//! Grease Pencil-lite: freehand strokes drawn in 3D (on a plane facing the view or on surfaces),
//! with colour and thickness, frame-by-frame drawings for 2D animation (each drawing holds until
//! the next) and onion skinning. Strokes are drawn flat on top of the scene in the viewport and
//! composited into final renders the same way, like Blender's Grease Pencil.

use glam::Vec3;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Stroke {
    /// Object-local points.
    pub points: Vec<Vec3>,
    /// sRGB colour and opacity.
    pub color: [f32; 4],
    /// World-space thickness (m).
    pub width: f32,
}

/// One drawing, shown from `frame` until the next drawing.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Drawing {
    pub frame: i32,
    pub strokes: Vec<Stroke>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Pencil {
    pub drawings: Vec<Drawing>,
    /// Show the previous and next drawings faintly.
    #[serde(default)]
    pub onion: bool,
}

impl Default for Pencil {
    fn default() -> Self {
        Pencil { drawings: vec![Drawing { frame: 1, strokes: vec![] }], onion: false }
    }
}

impl Pencil {
    /// Index of the drawing shown at `frame` (the last one starting at or before it; else the first).
    pub fn shown(&self, frame: i32) -> Option<usize> {
        if self.drawings.is_empty() {
            return None;
        }
        Some(self.drawings.iter().rposition(|d| d.frame <= frame).unwrap_or(0))
    }

    /// The drawing to add strokes to at `frame`: a new one at `frame` when `new_per_frame` and
    /// there isn't one exactly there, else the one shown.
    pub fn target(&mut self, frame: i32, new_per_frame: bool) -> &mut Drawing {
        if self.drawings.is_empty() || (new_per_frame && !self.drawings.iter().any(|d| d.frame == frame)) {
            self.drawings.push(Drawing { frame, strokes: vec![] });
            self.drawings.sort_by_key(|d| d.frame);
        }
        let i = self.shown(frame).unwrap();
        &mut self.drawings[i]
    }

    pub fn stroke_count(&self) -> usize {
        self.drawings.iter().map(|d| d.strokes.len()).sum()
    }

    /// Remove the parts of strokes (in drawing `di`) whose points `inside` says to erase, splitting
    /// strokes where needed. Returns how many points went.
    pub fn erase(&mut self, di: usize, inside: impl Fn(Vec3) -> bool) -> usize {
        let Some(d) = self.drawings.get_mut(di) else { return 0 };
        let mut removed = 0;
        let mut out = vec![];
        for s in d.strokes.drain(..) {
            let mut cur: Vec<Vec3> = vec![];
            for p in &s.points {
                if inside(*p) {
                    removed += 1;
                    if cur.len() >= 2 {
                        out.push(Stroke { points: std::mem::take(&mut cur), ..s.clone() });
                    }
                    cur.clear();
                } else {
                    cur.push(*p);
                }
            }
            if cur.len() >= 2 {
                out.push(Stroke { points: cur, ..s.clone() });
            }
        }
        d.strokes = out;
        removed
    }
}

/// Light smoothing and thinning of a freshly drawn stroke.
pub fn smooth(points: &[Vec3], min_step: f32) -> Vec<Vec3> {
    if points.len() < 3 {
        return points.to_vec();
    }
    let mut thin = vec![points[0]];
    for p in &points[1..] {
        if p.distance(*thin.last().unwrap()) >= min_step {
            thin.push(*p);
        }
    }
    if thin.last() != points.last() {
        thin.push(*points.last().unwrap());
    }
    let mut out = thin.clone();
    for _ in 0..2 {
        let prev = out.clone();
        for i in 1..prev.len().saturating_sub(1) {
            out[i] = prev[i] * 0.5 + (prev[i - 1] + prev[i + 1]) * 0.25;
        }
    }
    out
}

/// A stroke ready to draw: world points, width, colour (with any onion fade applied).
#[derive(Clone, Debug)]
pub struct WStroke {
    pub points: Vec<Vec3>,
    pub width: f32,
    pub color: [f32; 4],
}

/// All visible strokes in world space at each pencil object's current frame (`sim_frame`).
pub fn world_strokes(scene: &crate::scene::Scene, onion: bool) -> Vec<WStroke> {
    let mut out = vec![];
    for o in scene.objects.iter().filter(|o| o.visible) {
        let Some(p) = &o.pencil else { continue };
        let m = o.transform.matrix();
        let k = m.transform_vector3(Vec3::ONE).length() / 3f32.sqrt();
        let Some(i) = p.shown(o.sim_frame) else { continue };
        let mut emit = |d: &Drawing, tint: Option<[f32; 3]>, fade: f32| {
            for s in &d.strokes {
                let c = match tint {
                    Some(t) => [t[0], t[1], t[2], s.color[3] * fade],
                    None => s.color,
                };
                out.push(WStroke { points: s.points.iter().map(|q| m.transform_point3(*q)).collect(), width: s.width * k, color: c });
            }
        };
        if onion && p.onion {
            if i > 0 {
                emit(&p.drawings[i - 1], Some([1.0, 0.35, 0.35]), 0.3);
            }
            if i + 1 < p.drawings.len() {
                emit(&p.drawings[i + 1], Some([0.35, 0.8, 1.0]), 0.3);
            }
        }
        emit(&p.drawings[i], None, 1.0);
    }
    out
}

/// Screen projection for a render camera: world → (pixel x, pixel y, world size of one pixel).
pub fn project(cam: &crate::viewport::Camera, w: usize, h: usize, p: Vec3) -> Option<(f32, f32, f32)> {
    let aspect = w as f32 / h as f32;
    let (eye, fwd, right, up) = (cam.eye(), cam.forward(), cam.right(), cam.up());
    let d = p - eye;
    let (x, y, wpp) = if cam.ortho {
        let oh = cam.half_height();
        (d.dot(right) / (oh * aspect), d.dot(up) / oh, 2.0 * oh / h as f32)
    } else {
        let z = d.dot(fwd);
        if z < 1e-3 {
            return None;
        }
        let th = (cam.fov_deg.to_radians() * 0.5).tan();
        (d.dot(right) / (z * th * aspect), d.dot(up) / (z * th), 2.0 * z * th / h as f32)
    };
    Some(((x + 1.0) * 0.5 * w as f32, (1.0 - y) * 0.5 * h as f32, wpp))
}

fn seg_dist(px: f32, py: f32, a: (f32, f32), b: (f32, f32)) -> f32 {
    let (dx, dy) = (b.0 - a.0, b.1 - a.1);
    let l2 = dx * dx + dy * dy;
    let t = if l2 > 0.0 { (((px - a.0) * dx + (py - a.1) * dy) / l2).clamp(0.0, 1.0) } else { 0.0 };
    let (qx, qy) = (a.0 + dx * t, a.1 + dy * t);
    ((px - qx).powi(2) + (py - qy).powi(2)).sqrt()
}

/// Draw strokes over a rendered RGBA8 image (anti-aliased round-capped lines, each stroke blended
/// once so it doesn't darken where its segments overlap).
pub fn composite(rgba: &mut [u8], w: usize, h: usize, cam: &crate::viewport::Camera, strokes: &[WStroke]) {
    for s in strokes {
        let pts: Vec<(f32, f32, f32)> = s.points.iter().filter_map(|p| project(cam, w, h, *p)).map(|(x, y, wpp)| (x, y, (s.width / wpp * 0.5).max(0.5))).collect();
        if pts.is_empty() {
            continue;
        }
        let (mut x0, mut y0, mut x1, mut y1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
        for (x, y, r) in &pts {
            x0 = x0.min(x - r - 1.0);
            y0 = y0.min(y - r - 1.0);
            x1 = x1.max(x + r + 1.0);
            y1 = y1.max(y + r + 1.0);
        }
        let (bx0, by0) = (x0.floor().max(0.0) as usize, y0.floor().max(0.0) as usize);
        let (bx1, by1) = ((x1.ceil().max(0.0) as usize).min(w), (y1.ceil().max(0.0) as usize).min(h));
        if bx0 >= bx1 || by0 >= by1 {
            continue;
        }
        let bw = bx1 - bx0;
        let mut cov = vec![0f32; bw * (by1 - by0)];
        let segs: Vec<((f32, f32, f32), (f32, f32, f32))> = if pts.len() == 1 { vec![(pts[0], pts[0])] } else { pts.windows(2).map(|p| (p[0], p[1])).collect() };
        for (a, b) in segs {
            let r = (a.2 + b.2) * 0.5;
            let sx0 = ((a.0.min(b.0) - r - 1.0).floor().max(bx0 as f32)) as usize;
            let sx1 = ((a.0.max(b.0) + r + 1.0).ceil().min(bx1 as f32)) as usize;
            let sy0 = ((a.1.min(b.1) - r - 1.0).floor().max(by0 as f32)) as usize;
            let sy1 = ((a.1.max(b.1) + r + 1.0).ceil().min(by1 as f32)) as usize;
            for y in sy0..sy1 {
                for x in sx0..sx1 {
                    let d = seg_dist(x as f32 + 0.5, y as f32 + 0.5, (a.0, a.1), (b.0, b.1));
                    let c = (r - d + 0.5).clamp(0.0, 1.0);
                    let k = (y - by0) * bw + (x - bx0);
                    if c > cov[k] {
                        cov[k] = c;
                    }
                }
            }
        }
        let col = [s.color[0], s.color[1], s.color[2]];
        for y in by0..by1 {
            for x in bx0..bx1 {
                let a = cov[(y - by0) * bw + (x - bx0)] * s.color[3].clamp(0.0, 1.0);
                if a <= 0.0 {
                    continue;
                }
                let o = (y * w + x) * 4;
                for k in 0..3 {
                    rgba[o + k] = (rgba[o + k] as f32 * (1.0 - a) + col[k].clamp(0.0, 1.0) * 255.0 * a).round() as u8;
                }
                rgba[o + 3] = (rgba[o + 3] as f32 + (255.0 - rgba[o + 3] as f32) * a).round() as u8;
            }
        }
    }
}

/// A straight stroke (and optional arrow head) in object-local space, for the AI and tools.
/// `facing` = towards the viewer, so the arrow head opens flat to the camera.
pub fn line(points: &[Vec3], color: [f32; 4], width: f32, arrow: bool, facing: Vec3) -> Vec<Stroke> {
    let mut out = vec![Stroke { points: points.to_vec(), color, width }];
    if arrow && points.len() >= 2 {
        let (a, b) = (points[points.len() - 2], points[points.len() - 1]);
        let d = (b - a).normalize_or(Vec3::X);
        let len = ((b - a).length() * 0.12).clamp(width * 6.0, 0.6);
        let f = facing.normalize_or(Vec3::NEG_Y);
        let side = d.cross(if d.dot(f).abs() < 0.95 { f } else { Vec3::Z }).normalize_or(Vec3::Y);
        for s in [1.0, -1.0] {
            out.push(Stroke { points: vec![b, b - d * len + side * len * 0.5 * s], color, width });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drawings_hold_until_the_next_and_erase_splits() {
        let mut p = Pencil::default();
        p.target(1, false).strokes.push(Stroke { points: vec![Vec3::ZERO, Vec3::X, Vec3::X * 2.0, Vec3::X * 3.0], color: [0.0, 0.0, 0.0, 1.0], width: 0.02 });
        p.target(10, true).strokes.push(Stroke { points: vec![Vec3::Y, Vec3::Y * 2.0], color: [1.0, 0.0, 0.0, 1.0], width: 0.02 });
        assert_eq!(p.drawings.len(), 2);
        assert_eq!(p.shown(0), Some(0));
        assert_eq!(p.shown(5), Some(0));
        assert_eq!(p.shown(10), Some(1));
        assert_eq!(p.shown(99), Some(1));
        // not new_per_frame: adds to the drawing shown
        p.target(12, false).strokes.push(Stroke { points: vec![Vec3::Z, Vec3::Z * 2.0], color: [0.0; 4], width: 0.01 });
        assert_eq!(p.drawings.len(), 2);
        assert_eq!(p.drawings[1].strokes.len(), 2);
        // erase the middle of the first stroke → two pieces
        let n = p.erase(0, |q| (q.x - 1.5).abs() < 0.6);
        assert_eq!(n, 2);
        assert_eq!(p.drawings[0].strokes.len(), 0, "both pieces are single points, dropped");
        p.drawings[0].strokes.push(Stroke { points: (0..10).map(|i| Vec3::X * i as f32).collect(), color: [0.0; 4], width: 0.01 });
        p.erase(0, |q| (q.x - 5.0).abs() < 0.5);
        assert_eq!(p.drawings[0].strokes.len(), 2);
        let s = smooth(&(0..50).map(|i| Vec3::new(i as f32 * 0.01, 0.0, 0.0)).collect::<Vec<_>>(), 0.05);
        assert!(s.len() < 15 && s.len() >= 2);
    }

    #[test]
    fn composite_draws_where_the_camera_sees_the_stroke() {
        let cam = crate::viewport::Camera { target: Vec3::ZERO, yaw: 0.0, pitch: 0.0, dist: 10.0, ortho: false, fov_deg: 40.0 };
        let (w, h) = (64, 48);
        let mut img = vec![255u8; w * h * 4];
        // a horizontal red line through the centre
        let s = WStroke { points: vec![Vec3::new(-2.0, 0.0, 0.0), Vec3::new(2.0, 0.0, 0.0)], width: 0.8, color: [1.0, 0.0, 0.0, 1.0] };
        composite(&mut img, w, h, &cam, &[s]);
        let px = |x: usize, y: usize| &img[(y * w + x) * 4..(y * w + x) * 4 + 3];
        assert_eq!(px(32, 24), &[255, 0, 0]);
        assert_eq!(px(32, 2), &[255, 255, 255]);
        assert_eq!(px(2, 24), &[255, 255, 255], "the line ends inside the frame");
        let c = project(&cam, w, h, Vec3::ZERO).unwrap();
        assert!((c.0 - 32.0).abs() < 0.01 && (c.1 - 24.0).abs() < 0.01);
    }
}
