//! Weight Paint mode: paint how strongly the active bone moves each vertex of a rigged mesh.

use super::*;
use crate::armature::PaintMode;

pub struct WeightUi {
    pub mode: PaintMode,
    pub radius_px: f32,
    pub strength: f32,
    stroke: Option<(Pos2, Instant)>,
    adj: Option<((usize, usize), Vec<Vec<u32>>)>,
}

impl Default for WeightUi {
    fn default() -> Self {
        Self { mode: PaintMode::Add, radius_px: 50.0, strength: 0.5, stroke: None, adj: None }
    }
}

impl Viewport {
    /// The armature deforming the weight-painted mesh.
    pub(super) fn weight_armature(ed: &Editor) -> Option<(u64, u64)> {
        let id = ed.weight_paint?;
        let a = ed.scene.get(id)?.skin.as_ref()?.armature;
        ed.scene.get(a)?.armature.as_ref()?;
        Some((id, a))
    }

    pub(super) fn weight_input(&mut self, ctx: &egui::Context, resp: &egui::Response, ed: &mut Editor, mouse: Pos2, shift: bool, ctrl: bool) -> bool {
        let Some((id, arm_id)) = Self::weight_armature(ed) else {
            self.weight_ui.stroke = None;
            return false;
        };
        let nb = ed.scene.get(arm_id).and_then(|a| a.armature.as_ref()).map(|a| a.bones.len()).unwrap_or(0);
        if nb == 0 {
            return false;
        }
        ed.active_bone = ed.active_bone.min(nb - 1);
        let typing = ctx.egui_wants_keyboard_input();
        use egui::Modifiers as M;
        let c = |m: M, k: Key| ctx.input_mut(|i| i.consume_key(m, k));
        if !typing {
            if c(M::NONE, Key::CloseBracket) {
                self.weight_ui.radius_px = (self.weight_ui.radius_px * 1.15).min(600.0);
            }
            if c(M::NONE, Key::OpenBracket) {
                self.weight_ui.radius_px = (self.weight_ui.radius_px / 1.15).max(4.0);
            }
            if c(M::SHIFT, Key::CloseBracket) {
                self.weight_ui.strength = (self.weight_ui.strength + 0.1).min(1.0);
            }
            if c(M::SHIFT, Key::OpenBracket) {
                self.weight_ui.strength = (self.weight_ui.strength - 0.1).max(0.05);
            }
        }
        let bname = ed.scene.get(arm_id).and_then(|a| a.armature.as_ref()).map(|a| a.bones[ed.active_bone].name.clone()).unwrap_or_default();
        self.status_hint = format!("Weight Paint · bone \"{bname}\" · LMB paint · Ctrl subtract · Shift blur · Alt+click a bone to pick it · [ ] radius · Shift+[ ] strength · Tab exit");
        // Alt+click: pick the bone under the mouse
        if resp.clicked_by(PointerButton::Primary) && ctx.input(|i| i.modifiers.alt) {
            if let Some(b) = self.bone_at(ed, arm_id, mouse) {
                ed.active_bone = b;
            }
            return true;
        }
        let Some(o) = ed.scene.get(id) else { return false };
        let Some(posed) = o.posed.clone() else { return resp.hovered() };
        let mat = o.transform.matrix();
        let inv = mat.inverse();
        let key = (posed.faces.len(), posed.verts.len());
        if self.weight_ui.adj.as_ref().is_none_or(|(k, _)| *k != key) {
            self.weight_ui.adj = Some((key, crate::sculpt::adjacency(&posed)));
        }
        let down = ctx.input(|i| i.pointer.primary_down() && !i.modifiers.alt) && (resp.hovered() || self.weight_ui.stroke.is_some());
        if !down {
            self.weight_ui.stroke = None;
            return resp.hovered();
        }
        let (ro, rd) = self.ray(mouse);
        let Some((_, p, _)) = crate::sculpt::ray_hit(&posed, inv.transform_point3(ro), inv.transform_vector3(rd).normalize_or_zero()) else { return true };
        if self.weight_ui.stroke.is_none() {
            ed.snapshot();
            self.weight_ui.stroke = Some((mouse - vec2(1000.0, 0.0), Instant::now()));
        }
        let (last, t) = self.weight_ui.stroke.unwrap();
        let moved = (mouse - last).length();
        if moved < (self.weight_ui.radius_px * 0.15).max(2.0) && t.elapsed().as_secs_f32() < 0.06 {
            return true;
        }
        self.weight_ui.stroke = Some((mouse, Instant::now()));
        let pw = mat.transform_point3(p);
        let r_world = self.weight_ui.radius_px * self.world_per_pixel(pw);
        let mode = if shift { PaintMode::Blur } else if ctrl { PaintMode::Subtract } else { self.weight_ui.mode };
        let bone = ed.active_bone;
        let strength = self.weight_ui.strength;
        let adj = &self.weight_ui.adj.as_ref().unwrap().1;
        if let Some(o) = ed.scene.get_mut(id) {
            if let Some(sk) = o.skin.as_mut() {
                if crate::armature::paint(&posed, mat, adj, sk, pw, r_world, bone, strength, mode) > 0 {
                    // weights changed: re-skin
                    o.posed_key = 0;
                }
            }
        }
        ed.dirty = true;
        ctx.request_repaint();
        true
    }

    pub(super) fn draw_weight_cursor(&self, painter: &egui::Painter, ui: &Ui, ed: &Editor) {
        if ed.weight_paint.is_none() {
            return;
        }
        let Some(m) = ui.ctx().pointer_hover_pos().filter(|p| self.rect.contains(*p)) else { return };
        let col = if ui.input(|i| i.modifiers.command) { Color32::from_rgb(120, 170, 255) } else { Color32::from_white_alpha(220) };
        painter.circle_stroke(m, self.weight_ui.radius_px, Stroke::new(1.3, col));
        painter.circle_stroke(m, self.weight_ui.radius_px * (1.0 - self.weight_ui.strength * 0.5), Stroke::new(0.6, col.gamma_multiply(0.5)));
    }

    pub(super) fn weight_header(&mut self, ui: &mut Ui, ed: &mut Editor) {
        let Some((_, arm_id)) = Self::weight_armature(ed) else { return };
        let names: Vec<String> = ed.scene.get(arm_id).and_then(|a| a.armature.as_ref()).map(|a| a.bones.iter().map(|b| b.name.clone()).collect()).unwrap_or_default();
        let mut sel = ed.active_bone.min(names.len().saturating_sub(1));
        egui::ComboBox::from_id_salt("wp_bone").selected_text(egui::RichText::new(names.get(sel).cloned().unwrap_or_default()).size(12.0)).width(110.0).show_ui(ui, |ui| {
            for (i, n) in names.iter().enumerate() {
                ui.selectable_value(&mut sel, i, n);
            }
        });
        ed.active_bone = sel;
        for (m, l) in [(PaintMode::Add, "Add"), (PaintMode::Subtract, "Subtract"), (PaintMode::Blur, "Blur")] {
            if ui.add(egui::Button::selectable(self.weight_ui.mode == m, egui::RichText::new(l).size(12.0))).clicked() {
                self.weight_ui.mode = m;
            }
        }
        ui.add(egui::DragValue::new(&mut self.weight_ui.radius_px).range(4.0..=600.0).speed(1.0).prefix("r ").suffix(" px"));
        ui.add(egui::DragValue::new(&mut self.weight_ui.strength).range(0.05..=1.0).speed(0.01).prefix("s "));
    }
}
