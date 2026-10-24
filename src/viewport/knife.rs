//! Knife tool (K in edit mode): click a path across the mesh; Enter cuts it in.

use super::*;
use crate::editmesh::KnifePoint;
use crate::editor::EditOp;

#[derive(Clone, Copy)]
pub struct KnifeClick {
    screen: Pos2,
    snap: Option<KnifePoint>,
}

fn seg_hit(p0: Pos2, p1: Pos2, q0: Pos2, q1: Pos2) -> Option<(f32, f32)> {
    let r = p1 - p0;
    let s = q1 - q0;
    let den = r.x * s.y - r.y * s.x;
    if den.abs() < 1e-9 {
        return None;
    }
    let w = q0 - p0;
    let u = (w.x * s.y - w.y * s.x) / den;
    let v = (w.x * r.y - w.y * r.x) / den;
    ((0.0..=1.0).contains(&u) && (0.0..=1.0).contains(&v)).then_some((u, v))
}

impl Viewport {
    /// Parameter along the world edge a→b whose projection is closest to screen point `s`.
    fn edge_param_at(&self, a: Vec3, b: Vec3, s: Pos2) -> f32 {
        let d = |t: f32| self.project(a.lerp(b, t)).map(|p| (p - s).length()).unwrap_or(f32::MAX);
        let mut best = (0.0, f32::MAX);
        for k in 0..=32 {
            let t = k as f32 / 32.0;
            let v = d(t);
            if v < best.1 {
                best = (t, v);
            }
        }
        let (mut lo, mut hi) = ((best.0 - 1.0 / 32.0).max(0.0), (best.0 + 1.0 / 32.0).min(1.0));
        for _ in 0..30 {
            let (m1, m2) = (lo + (hi - lo) / 3.0, hi - (hi - lo) / 3.0);
            if d(m1) < d(m2) {
                hi = m2;
            } else {
                lo = m1;
            }
        }
        (lo + hi) * 0.5
    }

    fn knife_snap(&self, ed: &Editor, mouse: Pos2) -> Option<KnifePoint> {
        let d = self.edit_draw.as_ref()?;
        let xray = self.overlays.xray || ed.shading == Shading::Wireframe;
        let mut best: Option<(f32, u32)> = None;
        for (i, p) in d.verts.iter().enumerate() {
            if let Some(s) = self.project(*p) {
                let dist = (s - mouse).length();
                if dist < 10.0 && best.is_none_or(|(b, _)| dist < b) && self.visible(*p, xray) {
                    best = Some((dist, i as u32));
                }
            }
        }
        if let Some((_, v)) = best {
            return Some(KnifePoint::Vert(v));
        }
        let (a, b) = self.nearest_edge(ed, mouse, 8.0)?;
        let t = self.edge_param_at(d.verts[a as usize], d.verts[b as usize], mouse);
        Some(KnifePoint::Edge { a, b, t })
    }

    /// The cut: clicked snaps plus every visible edge crossed between clicks, in path order.
    fn knife_points(&self, ed: &Editor, clicks: &[KnifeClick]) -> Vec<(KnifePoint, Vec3)> {
        let Some(d) = self.edit_draw.as_ref() else { return vec![] };
        let xray = self.overlays.xray || ed.shading == Shading::Wireframe;
        let world = |p: &KnifePoint| match *p {
            KnifePoint::Vert(v) => d.verts[v as usize],
            KnifePoint::Edge { a, b, t } => d.verts[a as usize].lerp(d.verts[b as usize], t),
        };
        let touches = |p: Option<KnifePoint>, a: u32, b: u32| match p {
            Some(KnifePoint::Vert(v)) => v == a || v == b,
            Some(KnifePoint::Edge { a: x, b: y, .. }) => (x == a && y == b) || (x == b && y == a),
            None => false,
        };
        let mut out = vec![];
        for k in 0..clicks.len() {
            if let Some(sp) = clicks[k].snap {
                out.push((sp, world(&sp)));
            }
            let Some(next) = clicks.get(k + 1) else { break };
            let (s0, s1) = (clicks[k].screen, next.screen);
            let mut hits: Vec<(f32, KnifePoint, Vec3)> = vec![];
            for &(a, b) in &d.edges {
                if touches(clicks[k].snap, a, b) || touches(next.snap, a, b) {
                    continue;
                }
                let (pa, pb) = (d.verts[a as usize], d.verts[b as usize]);
                let (Some(sa), Some(sb)) = (self.project(pa), self.project(pb)) else { continue };
                let Some((u, v)) = seg_hit(s0, s1, sa, sb) else { continue };
                let t = self.edge_param_at(pa, pb, sa + (sb - sa) * v);
                let p = pa.lerp(pb, t);
                if !self.visible(p, xray) {
                    continue;
                }
                hits.push((u, KnifePoint::Edge { a, b, t }, p));
            }
            hits.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            out.extend(hits.into_iter().map(|h| (h.1, h.2)));
        }
        out
    }

    pub(super) fn knife_input(&mut self, ctx: &egui::Context, resp: &egui::Response, ed: &mut Editor, mouse: Pos2) -> bool {
        let Some(clicks) = self.knife.clone() else { return false };
        if !ed.editing() {
            self.knife = None;
            return false;
        }
        let cancel = ctx.input(|i| i.key_pressed(Key::Escape)) || resp.clicked_by(PointerButton::Secondary);
        let confirm = ctx.input(|i| i.key_pressed(Key::Enter) || i.key_pressed(Key::Space));
        if cancel {
            self.knife = None;
            return true;
        }
        if ctx.input(|i| i.key_pressed(Key::Backspace)) {
            if let Some(k) = self.knife.as_mut() {
                k.pop();
            }
        }
        if confirm {
            let pts: Vec<KnifePoint> = self.knife_points(ed, &clicks).into_iter().map(|p| p.0).collect();
            self.knife = None;
            let r = ed.edit_op(EditOp::Knife(pts));
            self.report(r);
            return true;
        }
        if resp.clicked_by(PointerButton::Primary) {
            let snap = self.knife_snap(ed, mouse);
            let screen = snap.and_then(|s| self.knife_screen(&s)).unwrap_or(mouse);
            if let Some(k) = self.knife.as_mut() {
                k.push(KnifeClick { screen, snap });
            }
        }
        self.status_hint = format!("Knife · {} point(s) · LMB add a point (snaps to vertices / edges) · Backspace undo point · Enter/Space cut · Esc cancel", clicks.len());
        ctx.request_repaint();
        // navigation (MMB) stays available
        !resp.dragged_by(PointerButton::Middle) && !ctx.input(|i| i.smooth_scroll_delta.y != 0.0)
    }

    fn knife_screen(&self, p: &KnifePoint) -> Option<Pos2> {
        let d = self.edit_draw.as_ref()?;
        let w = match *p {
            KnifePoint::Vert(v) => *d.verts.get(v as usize)?,
            KnifePoint::Edge { a, b, t } => d.verts.get(a as usize)?.lerp(*d.verts.get(b as usize)?, t),
        };
        self.project(w)
    }

    pub(super) fn draw_knife_preview(&self, painter: &egui::Painter, ed: &Editor) {
        let Some(clicks) = &self.knife else { return };
        let col = Color32::from_rgb(255, 230, 40);
        let mouse = painter.ctx().pointer_hover_pos();
        let mut path: Vec<KnifeClick> = clicks.clone();
        if let Some(m) = mouse.filter(|m| self.rect.contains(*m)) {
            let snap = self.knife_snap(ed, m);
            path.push(KnifeClick { screen: snap.and_then(|s| self.knife_screen(&s)).unwrap_or(m), snap });
            if let Some(s) = snap.and_then(|s| self.knife_screen(&s)) {
                painter.circle_stroke(s, 6.0, Stroke::new(1.5, Color32::WHITE));
            }
        }
        for w in path.windows(2) {
            painter.line_segment([w[0].screen, w[1].screen], Stroke::new(2.0, col));
        }
        for c in clicks {
            painter.circle_filled(c.screen, 3.5, col);
        }
        for (_, p) in self.knife_points(ed, &path) {
            if let Some(s) = self.project(p) {
                painter.circle_filled(s, 2.5, Color32::from_rgb(80, 255, 120));
            }
        }
    }
}
