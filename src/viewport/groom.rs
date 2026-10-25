//! Hair Edit mode: groom the guide strands with brushes — comb, length, puff, cut, smooth.

use super::*;
use std::collections::HashSet;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GroomTool {
    Comb,
    Length,
    Puff,
    Cut,
    Smooth,
}

pub struct GroomUi {
    pub tool: GroomTool,
    pub radius_px: f32,
    pub strength: f32,
    stroke: Option<(Pos2, Instant)>,
    cut_done: HashSet<usize>,
}

impl Default for GroomUi {
    fn default() -> Self {
        Self { tool: GroomTool::Comb, radius_px: 60.0, strength: 0.5, stroke: None, cut_done: HashSet::new() }
    }
}

impl Viewport {
    /// The groomed guides of the hair being edited, on its surface now.
    fn groom_guides(ed: &Editor) -> Option<(u64, crate::hair::Hair, glam::Mat4, Vec<Option<crate::hair::Guide>>)> {
        let id = ed.hair_edit?;
        let h = ed.scene.get(id)?.hair.clone()?;
        let t = ed.scene.get(h.target)?;
        let rest = t.poly();
        let surface = match &t.posed {
            Some(p) if p.faces.len() == rest.faces.len() && p.verts.len() == rest.verts.len() => (**p).clone(),
            _ => rest.clone(),
        };
        let m = t.transform.matrix();
        let g = crate::hair::guides(&h, &rest, &surface, m);
        Some((id, h, m, g))
    }

    pub(super) fn groom_input(&mut self, ctx: &egui::Context, resp: &egui::Response, ed: &mut Editor, mouse: Pos2, shift: bool, ctrl: bool) -> bool {
        if ed.hair_edit.is_none() {
            self.groom_ui.stroke = None;
            return false;
        }
        let typing = ctx.egui_wants_keyboard_input();
        use egui::Modifiers as M;
        let c = |m: M, k: Key| ctx.input_mut(|i| i.consume_key(m, k));
        if !typing {
            if c(M::NONE, Key::CloseBracket) {
                self.groom_ui.radius_px = (self.groom_ui.radius_px * 1.15).min(600.0);
            }
            if c(M::NONE, Key::OpenBracket) {
                self.groom_ui.radius_px = (self.groom_ui.radius_px / 1.15).max(4.0);
            }
            if c(M::SHIFT, Key::CloseBracket) {
                self.groom_ui.strength = (self.groom_ui.strength + 0.1).min(1.0);
            }
            if c(M::SHIFT, Key::OpenBracket) {
                self.groom_ui.strength = (self.groom_ui.strength - 0.1).max(0.05);
            }
            for (k, t) in [(Key::Num1, GroomTool::Comb), (Key::Num2, GroomTool::Length), (Key::Num3, GroomTool::Puff), (Key::Num4, GroomTool::Cut), (Key::Num5, GroomTool::Smooth)] {
                if c(M::NONE, k) {
                    self.groom_ui.tool = t;
                }
            }
        }
        self.status_hint = "Hair Edit · LMB brush · 1 Comb · 2 Length (Ctrl shorter) · 3 Puff (Ctrl flatten) · 4 Cut · 5 Smooth · Shift smooth · [ ] radius · Shift+[ ] strength · Tab exit".into();
        let down = ctx.input(|i| i.pointer.primary_down() && !i.modifiers.alt) && (resp.hovered() || self.groom_ui.stroke.is_some());
        if !down {
            self.groom_ui.stroke = None;
            self.groom_ui.cut_done.clear();
            return resp.hovered();
        }
        let Some((id, h, surf_m, guides)) = Self::groom_guides(ed) else { return true };
        let first = self.groom_ui.stroke.is_none();
        if first {
            ed.snapshot();
            self.groom_ui.stroke = Some((mouse, Instant::now()));
        }
        let (last, _) = self.groom_ui.stroke.unwrap();
        let delta = mouse - last;
        let tool = if shift { GroomTool::Smooth } else { self.groom_ui.tool };
        // the comb needs movement; the others work while held
        if tool == GroomTool::Comb && delta.length() < 2.0 {
            return true;
        }
        if tool != GroomTool::Comb && !first && delta.length() < 1.0 && self.groom_ui.stroke.unwrap().1.elapsed().as_secs_f32() < 0.03 {
            return true;
        }
        self.groom_ui.stroke = Some((mouse, Instant::now()));
        let r = self.groom_ui.radius_px;
        let strength = self.groom_ui.strength;
        let inv_rot = glam::Mat3::from_mat4(surf_m).inverse();
        let to_local = |v: Vec3| -> Vec3 {
            let l = inv_rot * v;
            l.normalize_or_zero() * v.length()
        };
        let eye = self.cam.eye();
        let fwd = self.cam.forward();
        let segs = h.segments.clamp(1, 32) as usize;
        // guides under the brush: (index, weight, nearest point index)
        let mut hits: Vec<(usize, f32, usize)> = vec![];
        for (i, g) in guides.iter().enumerate() {
            let Some(g) = g else { continue };
            let view = if self.cam.ortho { -fwd } else { (eye - g.root).normalize_or_zero() };
            if g.normal.dot(view) < -0.05 {
                continue; // the far side
            }
            let mut best = (f32::MAX, 0usize);
            for (k, p) in g.points.iter().enumerate() {
                if let Some(s) = self.project(*p) {
                    let d = (s - mouse).length();
                    if d < best.0 {
                        best = (d, k);
                    }
                }
            }
            if best.0 < r {
                let f = 1.0 - best.0 / r;
                hits.push((i, f * f * strength, best.1));
            }
        }
        if hits.is_empty() {
            return true;
        }
        let mut changed = 0;
        let mean = if tool == GroomTool::Smooth {
            let (mut s, mut n) = (Vec3::ZERO, 0.0);
            for (i, _, _) in &hits {
                s += h.groom_of(*i).0;
                n += 1.0;
            }
            s / n
        } else {
            Vec3::ZERO
        };
        let Some(hm) = ed.scene.get_mut(id).and_then(|o| o.hair.as_mut()) else { return true };
        for (i, w, k) in hits {
            let g = guides[i].as_ref().unwrap();
            let len: f32 = g.points.windows(2).map(|p| (p[1] - p[0]).length()).sum::<f32>().max(1e-4);
            match tool {
                GroomTool::Comb => {
                    let wpp = self.world_per_pixel(g.root);
                    let mut dw = (self.cam.right() * delta.x - self.cam.up() * delta.y) * wpp;
                    dw -= g.normal * g.normal.dot(dw) * 0.7;
                    let add = to_local(dw / len * 3.0 * w);
                    let e = hm.groom_mut(i);
                    let mut v = Vec3::new(e[0], e[1], e[2]) + add;
                    if v.length() > 3.0 {
                        v = v.normalize() * 3.0;
                    }
                    (e[0], e[1], e[2]) = (v.x, v.y, v.z);
                }
                GroomTool::Length => {
                    let e = hm.groom_mut(i);
                    let k = 1.0 + 0.08 * w;
                    e[3] = (if ctrl { e[3] / k } else { e[3] * k }).clamp(0.02, 6.0);
                }
                GroomTool::Puff => {
                    let n = to_local(g.normal * 0.12 * w * if ctrl { -1.0 } else { 1.0 });
                    let e = hm.groom_mut(i);
                    let v = Vec3::new(e[0], e[1], e[2]) + n;
                    (e[0], e[1], e[2]) = (v.x, v.y, v.z);
                }
                GroomTool::Cut => {
                    if k == 0 || self.groom_ui.cut_done.contains(&i) {
                        continue;
                    }
                    let e = hm.groom_mut(i);
                    e[3] = (e[3] * ((k as f32 - 0.5) / segs as f32)).max(0.02);
                    self.groom_ui.cut_done.insert(i);
                }
                GroomTool::Smooth => {
                    let e = hm.groom_mut(i);
                    let v = Vec3::new(e[0], e[1], e[2]).lerp(mean, (0.35 * w / strength.max(0.05)).min(1.0) * strength);
                    (e[0], e[1], e[2]) = (v.x, v.y, v.z);
                }
            }
            changed += 1;
        }
        if changed > 0 {
            // grooming changes the rest shape: a baked simulation is out of date
            if hm.dynamics {
                hm.sim = Default::default();
            }
            if let Some(o) = ed.scene.get_mut(id) {
                o.posed_key = 0;
            }
            ed.scene.update_hair();
            ed.dirty = true;
        }
        ctx.request_repaint();
        true
    }

    pub(super) fn draw_groom_overlay(&self, painter: &egui::Painter, ui: &Ui, ed: &Editor) {
        if ed.hair_edit.is_none() {
            return;
        }
        // the guides, so it's clear what the brush moves
        if let Some((_, _, _, guides)) = Self::groom_guides(ed) {
            let col = Color32::from_rgba_unmultiplied(255, 200, 90, 150);
            for g in guides.iter().flatten().take(4000) {
                let pts: Vec<Pos2> = g.points.iter().filter_map(|p| self.project(*p)).collect();
                if pts.len() >= 2 {
                    painter.add(egui::Shape::line(pts, Stroke::new(1.0, col)));
                }
            }
        }
        let Some(m) = ui.ctx().pointer_hover_pos().filter(|p| self.rect.contains(*p)) else { return };
        let col = if ui.input(|i| i.modifiers.command) { Color32::from_rgb(120, 170, 255) } else { Color32::from_white_alpha(220) };
        painter.circle_stroke(m, self.groom_ui.radius_px, Stroke::new(1.3, col));
        painter.circle_stroke(m, self.groom_ui.radius_px * (1.0 - self.groom_ui.strength * 0.5), Stroke::new(0.6, col.gamma_multiply(0.5)));
    }

    pub(super) fn groom_header(&mut self, ui: &mut Ui, _ed: &mut Editor) {
        for (t, l) in [(GroomTool::Comb, "Comb"), (GroomTool::Length, "Length"), (GroomTool::Puff, "Puff"), (GroomTool::Cut, "Cut"), (GroomTool::Smooth, "Smooth")] {
            if ui.add(egui::Button::selectable(self.groom_ui.tool == t, egui::RichText::new(l).size(12.0))).clicked() {
                self.groom_ui.tool = t;
            }
        }
        ui.add(egui::DragValue::new(&mut self.groom_ui.radius_px).range(4.0..=600.0).speed(1.0).prefix("r ").suffix(" px"));
        ui.add(egui::DragValue::new(&mut self.groom_ui.strength).range(0.05..=1.0).speed(0.01).prefix("s "));
    }
}
