//! Sculpt mode: brush strokes, brush keys, the brush cursor and the sculpt header.

use super::*;
use crate::sculpt::{self as sc, Brush};

pub struct SculptUi {
    pub brush: Brush,
    /// Brush radius on screen (pixels).
    pub radius_px: f32,
    pub strength: f32,
    /// Mirror strokes across the object's local X axis.
    pub symmetry: bool,
    stroke: Option<StrokeState>,
    /// (faces, verts) the adjacency was built for
    adj: Option<((usize, usize), Vec<Vec<u32>>)>,
    /// Last surface point under the mouse (world) for the cursor.
    hover: Option<Vec3>,
}

impl Default for SculptUi {
    fn default() -> Self {
        Self { brush: Brush::Draw, radius_px: 60.0, strength: 0.5, symmetry: true, stroke: None, adj: None, hover: None }
    }
}

struct StrokeState {
    last: Pos2,
    last_time: Instant,
    start_mouse: Pos2,
    /// grab: vertices with original positions and weights (+ mirrored side), and the depth point
    grab: Vec<(u32, Vec3, f32, bool)>,
    grab_at: Vec3,
}

impl Viewport {
    /// Sculpt mode input. Returns true when the viewport input was consumed.
    pub(super) fn sculpt_input(&mut self, ctx: &egui::Context, resp: &egui::Response, ed: &mut Editor, mouse: Pos2, shift: bool, ctrl: bool) -> bool {
        let Some(id) = ed.sculpt else {
            self.sculpt_ui.stroke = None;
            return false;
        };
        let typing = ctx.egui_wants_keyboard_input();
        use egui::Modifiers as M;
        let c = |m: M, k: Key| ctx.input_mut(|i| i.consume_key(m, k));
        if !typing {
            for b in Brush::ALL {
                let k = match b {
                    Brush::Draw => Key::X,
                    Brush::Inflate => Key::I,
                    Brush::Smooth => Key::S,
                    Brush::Grab => Key::G,
                    Brush::Crease => Key::C,
                    Brush::Flatten => Key::T,
                };
                if c(M::NONE, k) {
                    self.sculpt_ui.brush = b;
                }
            }
            if c(M::NONE, Key::CloseBracket) {
                self.sculpt_ui.radius_px = (self.sculpt_ui.radius_px * 1.15).min(600.0);
            }
            if c(M::NONE, Key::OpenBracket) {
                self.sculpt_ui.radius_px = (self.sculpt_ui.radius_px / 1.15).max(4.0);
            }
            if c(M::SHIFT, Key::CloseBracket) {
                self.sculpt_ui.strength = (self.sculpt_ui.strength + 0.1).min(1.0);
            }
            if c(M::SHIFT, Key::OpenBracket) {
                self.sculpt_ui.strength = (self.sculpt_ui.strength - 0.1).max(0.05);
            }
        }
        let invert = ctrl;
        self.status_hint = format!(
            "Sculpt Mode · {} (X draw · I inflate · S smooth · G grab · C crease · T flatten) · [ ] radius · Shift+[ ] strength · Shift = smooth · Ctrl = invert · Tab exit",
            self.sculpt_ui.brush.label()
        );
        let Some(o) = ed.scene.get(id) else { return false };
        let mat = o.transform.matrix();
        let inv = mat.inverse();
        let scale = (glam::Mat3::from_mat4(mat) * Vec3::ONE.normalize()).length().max(1e-6);
        let Some(pm) = o.mesh.as_ref() else { return false };
        let key = (pm.faces.len(), pm.verts.len());
        if self.sculpt_ui.adj.as_ref().is_none_or(|(k, _)| *k != key) {
            self.sculpt_ui.adj = Some((key, sc::adjacency(pm)));
        }
        // surface under the mouse
        let (ro, rd) = self.ray(mouse);
        let lo = inv.transform_point3(ro);
        let ld = inv.transform_vector3(rd).normalize_or_zero();
        let hit = if resp.hovered() || self.sculpt_ui.stroke.is_some() { sc::ray_hit(pm, lo, ld) } else { None };
        self.sculpt_ui.hover = hit.map(|h| mat.transform_point3(h.1));
        let down = ctx.input(|i| i.pointer.primary_down()) && (resp.hovered() || self.sculpt_ui.stroke.is_some());
        let started = resp.drag_started_by(PointerButton::Primary) || (resp.is_pointer_button_down_on() && self.sculpt_ui.stroke.is_none() && ctx.input(|i| i.pointer.primary_pressed()));
        if !down {
            self.sculpt_ui.stroke = None;
            return resp.hovered();
        }
        let brush = if shift { Brush::Smooth } else { self.sculpt_ui.brush };
        if started && self.sculpt_ui.stroke.is_none() {
            let Some((_, p, _)) = hit else { return true };
            let r_local = self.sculpt_ui.radius_px * self.world_per_pixel(mat.transform_point3(p)) / scale;
            let mut grab = vec![];
            if brush == Brush::Grab {
                grab.extend(sc::grab_set(pm, p, r_local).into_iter().map(|(i, v, w)| (i, v, w, false)));
                if self.sculpt_ui.symmetry && p.x.abs() > r_local * 0.05 {
                    let mp = Vec3::new(-p.x, p.y, p.z);
                    for (i, v, w) in sc::grab_set(pm, mp, r_local) {
                        if !grab.iter().any(|g| g.0 == i) {
                            grab.push((i, v, w, true));
                        }
                    }
                }
            }
            ed.snapshot();
            self.sculpt_ui.stroke = Some(StrokeState { last: mouse - vec2(1000.0, 0.0), last_time: Instant::now(), start_mouse: mouse, grab, grab_at: mat.transform_point3(p) });
        }
        let Some((grab_at, start_mouse, last, last_time, has_grab)) = self.sculpt_ui.stroke.as_ref().map(|st| (st.grab_at, st.start_mouse, st.last, st.last_time, !st.grab.is_empty())) else { return true };
        if brush == Brush::Grab && has_grab {
            let wpp = self.world_per_pixel(grab_at);
            let d = mouse - start_mouse;
            let world = (self.cam.right() * d.x - self.cam.up() * d.y) * wpp;
            let local = inv.transform_vector3(world);
            let grab = self.sculpt_ui.stroke.as_ref().map(|st| st.grab.clone()).unwrap_or_default();
            if let Some(pm) = ed.scene.get_mut(id).and_then(|o| o.mesh.as_mut()) {
                for (i, v0, w, mirrored) in grab {
                    let dl = if mirrored { Vec3::new(-local.x, local.y, local.z) } else { local };
                    if let Some(v) = pm.verts.get_mut(i as usize) {
                        *v = v0 + dl * w;
                    }
                }
            }
            ed.dirty = true;
            return true;
        }
        // other brushes: dab when the mouse moved enough, or periodically while held
        let moved = (mouse - last).length();
        let spacing = (self.sculpt_ui.radius_px * 0.12).max(2.0);
        let held = last_time.elapsed().as_secs_f32() > 0.05 && matches!(brush, Brush::Inflate | Brush::Smooth | Brush::Draw);
        if moved < spacing && !held {
            return true;
        }
        let Some((_, p, fnrm)) = hit else { return true };
        if let Some(st) = self.sculpt_ui.stroke.as_mut() {
            st.last = mouse;
            st.last_time = Instant::now();
        }
        let r_local = self.sculpt_ui.radius_px * self.world_per_pixel(mat.transform_point3(p)) / scale;
        let strength = self.sculpt_ui.strength * if held && moved < spacing { 0.35 } else { 1.0 };
        let sym = self.sculpt_ui.symmetry;
        let adj = &self.sculpt_ui.adj.as_ref().unwrap().1;
        if let Some(pm) = ed.scene.get_mut(id).and_then(|o| o.mesh.as_mut()) {
            sc::dab(pm, adj, p, r_local, brush, strength, invert, fnrm);
            if sym && p.x.abs() > r_local * 0.05 {
                sc::dab(pm, adj, Vec3::new(-p.x, p.y, p.z), r_local, brush, strength, invert, Vec3::new(-fnrm.x, fnrm.y, fnrm.z));
            }
        }
        ed.dirty = true;
        ctx.request_repaint();
        true
    }

    pub(super) fn draw_sculpt_cursor(&self, painter: &egui::Painter, ui: &Ui, ed: &Editor) {
        if ed.sculpt.is_none() {
            return;
        }
        let Some(m) = ui.ctx().pointer_hover_pos().filter(|p| self.rect.contains(*p)) else { return };
        let ctrl = ui.input(|i| i.modifiers.command);
        let col = if ctrl { Color32::from_rgb(120, 170, 255) } else { Color32::from_white_alpha(220) };
        let on_surface = self.sculpt_ui.hover.is_some();
        painter.circle_stroke(m, self.sculpt_ui.radius_px, Stroke::new(if on_surface { 1.5 } else { 0.8 }, col));
        painter.circle_stroke(m, self.sculpt_ui.radius_px * (1.0 - self.sculpt_ui.strength * 0.5), Stroke::new(0.6, col.gamma_multiply(0.5)));
    }

    /// Brush pickers and settings for the viewport header.
    pub(super) fn sculpt_header(&mut self, ui: &mut Ui) {
        for b in Brush::ALL {
            if ui
                .add(egui::Button::selectable(self.sculpt_ui.brush == b, egui::RichText::new(b.label()).size(12.0)))
                .on_hover_text(format!("{} brush ({})", b.label(), b.key()))
                .clicked()
            {
                self.sculpt_ui.brush = b;
            }
        }
        ui.separator();
        ui.add(egui::DragValue::new(&mut self.sculpt_ui.radius_px).range(4.0..=600.0).speed(1.0).prefix("r ").suffix(" px")).on_hover_text("Radius ([ and ])");
        ui.add(egui::DragValue::new(&mut self.sculpt_ui.strength).range(0.05..=1.0).speed(0.01).prefix("s ")).on_hover_text("Strength (Shift+[ and ])");
        ui.toggle_value(&mut self.sculpt_ui.symmetry, "X").on_hover_text("Mirror strokes across X");
    }
}
