//! Draw mode (Grease Pencil): freehand strokes, straight lines and the eraser; strokes of every
//! drawing object are drawn on top of the scene.

use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PencilTool {
    Draw,
    Line,
    Erase,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Placement {
    /// on a plane through the 3D cursor facing the view
    View,
    /// on whatever surface is under the pen (else the view plane)
    Surface,
}

pub struct PencilUi {
    pub tool: PencilTool,
    pub color: [f32; 3],
    pub opacity: f32,
    /// pen thickness in pixels (stored in metres at the depth drawn)
    pub thickness: f32,
    pub erase_px: f32,
    pub placement: Placement,
    /// each new frame you draw on gets its own drawing (2D animation)
    pub new_per_frame: bool,
    current: Vec<Vec3>,
    line_start: Option<Vec3>,
    plane: Option<(Vec3, Vec3)>,
    bvh: Option<crate::raytrace::Bvh>,
    width_world: f32,
    erasing: bool,
}

impl Default for PencilUi {
    fn default() -> Self {
        Self { tool: PencilTool::Draw, color: [0.07, 0.07, 0.08], opacity: 1.0, thickness: 4.0, erase_px: 18.0, placement: Placement::View, new_per_frame: false, current: vec![], line_start: None, plane: None, bvh: None, width_world: 0.01, erasing: false }
    }
}

pub const PALETTE: &[[f32; 3]] = &[[0.07, 0.07, 0.08], [0.95, 0.95, 0.95], [0.9, 0.2, 0.15], [0.98, 0.6, 0.1], [0.98, 0.85, 0.15], [0.25, 0.75, 0.3], [0.2, 0.55, 0.95], [0.6, 0.35, 0.85]];

impl Viewport {
    /// The point under the pen: on a surface (Surface placement) or the stroke's plane.
    fn pen_point(&mut self, mouse: Pos2) -> Option<Vec3> {
        let (o, d) = self.ray(mouse);
        if self.pencil_ui.placement == Placement::Surface {
            if let Some(h) = self.pencil_ui.bvh.as_ref().and_then(|b| b.intersect(o, d, 1e5)) {
                // a hair in front of the surface so the stroke isn't hidden in it
                return Some(o + d * (h.t * 0.997));
            }
        }
        let (p0, n) = self.pencil_ui.plane?;
        let den = d.dot(n);
        if den.abs() < 1e-6 {
            return None;
        }
        let t = (p0 - o).dot(n) / den;
        (t > 0.0).then(|| o + d * t)
    }

    pub(super) fn pencil_input(&mut self, ctx: &egui::Context, resp: &egui::Response, ed: &mut Editor, mouse: Pos2, ctrl: bool) -> bool {
        let Some(id) = ed.pencil_draw else {
            self.pencil_ui.current.clear();
            return false;
        };
        let typing = ctx.egui_wants_keyboard_input();
        use egui::Modifiers as M;
        let c = |m: M, k: Key| ctx.input_mut(|i| i.consume_key(m, k));
        if !typing {
            if c(M::NONE, Key::CloseBracket) {
                self.pencil_ui.thickness = (self.pencil_ui.thickness * 1.2).min(80.0);
            }
            if c(M::NONE, Key::OpenBracket) {
                self.pencil_ui.thickness = (self.pencil_ui.thickness / 1.2).max(1.0);
            }
            for (k, t) in [(Key::D, PencilTool::Draw), (Key::L, PencilTool::Line), (Key::E, PencilTool::Erase)] {
                if c(M::NONE, k) {
                    self.pencil_ui.tool = t;
                }
            }
        }
        let tool = if ctrl { PencilTool::Erase } else { self.pencil_ui.tool };
        self.status_hint = "Draw · LMB draw · D draw · L line · E (or Ctrl) erase · [ ] thickness · Tab exit".into();
        let (pressed, down, released) = ctx.input(|i| (i.pointer.primary_pressed(), i.pointer.primary_down() && !i.modifiers.alt, i.pointer.primary_released()));
        let over = resp.hovered();
        let drawing = !self.pencil_ui.current.is_empty() || self.pencil_ui.line_start.is_some() || self.pencil_ui.erasing;
        if pressed && over {
            ed.snapshot();
            match tool {
                PencilTool::Erase => self.pencil_ui.erasing = true,
                _ => {
                    // the plane: through what's under the pen (or the 3D cursor), facing the view
                    self.pencil_ui.bvh = (self.pencil_ui.placement == Placement::Surface).then(|| scene_bvh(ed, &mut self.mesh_cache));
                    let (o, d) = self.ray(mouse);
                    let hit = self.pencil_ui.bvh.as_ref().and_then(|b| b.intersect(o, d, 1e5)).map(|h| o + d * h.t);
                    let anchor = hit.unwrap_or(self.cursor);
                    self.pencil_ui.plane = Some((anchor, -self.cam.forward()));
                    if let Some(p) = self.pen_point(mouse) {
                        self.pencil_ui.width_world = self.pencil_ui.thickness * self.world_per_pixel(p);
                        if tool == PencilTool::Line {
                            self.pencil_ui.line_start = Some(p);
                        }
                        self.pencil_ui.current = vec![p];
                    }
                }
            }
        }
        if !drawing && !(pressed && over) {
            return over;
        }
        if down {
            match tool {
                PencilTool::Erase if self.pencil_ui.erasing => {
                    let r = self.pencil_ui.erase_px;
                    if let Some(o) = ed.scene.get(id) {
                        let m = o.transform.matrix();
                        let frame = o.sim_frame;
                        let vp = self.clone_projector();
                        let gone = ed.scene.get_mut(id).and_then(|o| o.pencil.as_mut()).map(|p| match p.shown(frame) {
                            Some(di) => p.erase(di, |q| vp(m.transform_point3(q)).is_some_and(|s| (s - mouse).length() < r)),
                            None => 0,
                        });
                        if gone.unwrap_or(0) > 0 {
                            ed.dirty = true;
                        }
                    }
                }
                PencilTool::Draw => {
                    if let Some(p) = self.pen_point(mouse) {
                        let far_enough = self.pencil_ui.current.last().is_none_or(|q| self.project(*q).is_none_or(|s| (s - mouse).length() >= 2.0));
                        if far_enough {
                            self.pencil_ui.current.push(p);
                        }
                    }
                }
                PencilTool::Line => {
                    if let (Some(a), Some(p)) = (self.pencil_ui.line_start, self.pen_point(mouse)) {
                        self.pencil_ui.current = vec![a, p];
                    }
                }
                PencilTool::Erase => {}
            }
            ctx.request_repaint();
        }
        if released || !down {
            self.pencil_ui.erasing = false;
            let pts = std::mem::take(&mut self.pencil_ui.current);
            self.pencil_ui.line_start = None;
            self.pencil_ui.bvh = None;
            if pts.len() >= 2 {
                let step = self.pencil_ui.width_world * 0.6;
                let pts = if tool == PencilTool::Draw { crate::pencil::smooth(&pts, step) } else { pts };
                let frame = ed.frame;
                let new_per_frame = self.pencil_ui.new_per_frame;
                let (col, op, width) = (self.pencil_ui.color, self.pencil_ui.opacity, self.pencil_ui.width_world);
                if let Some(o) = ed.scene.get_mut(id) {
                    let inv = o.transform.matrix().inverse();
                    let k = o.transform.scale.abs().max_element().max(1e-6);
                    let local: Vec<Vec3> = pts.iter().map(|p| inv.transform_point3(*p)).collect();
                    o.sim_frame = frame;
                    if let Some(p) = o.pencil.as_mut() {
                        p.target(frame, new_per_frame).strokes.push(crate::pencil::Stroke { points: local, color: [col[0], col[1], col[2], op], width: width / k });
                    }
                    ed.dirty = true;
                }
            }
        }
        true
    }

    /// A copy of the current projection, usable while the editor is borrowed.
    fn clone_projector(&self) -> impl Fn(Vec3) -> Option<Pos2> + use<> {
        let (vp, rect) = (self.vp, self.rect);
        move |p: Vec3| {
            let c = vp * p.extend(1.0);
            if c.w <= 1e-6 {
                return None;
            }
            let n = c.xyz() / c.w;
            Some(pos2(rect.center().x + n.x * rect.width() * 0.5, rect.center().y - n.y * rect.height() * 0.5))
        }
    }

    /// Every drawing's strokes, on top of the scene (onion skins in Draw mode).
    pub(super) fn draw_pencil(&self, painter: &egui::Painter, ui: &Ui, ed: &Editor) {
        let strokes = crate::pencil::world_strokes(&ed.scene, ed.pencil_draw.is_some());
        let draw = |pts: &[Vec3], width: f32, color: [f32; 4]| {
            let s: Vec<Pos2> = pts.iter().filter_map(|p| self.project(*p)).collect();
            if s.is_empty() {
                return;
            }
            let mid = pts[pts.len() / 2];
            let px = (width / self.world_per_pixel(mid)).clamp(0.6, 200.0);
            let c = Color32::from_rgba_unmultiplied((color[0].clamp(0.0, 1.0) * 255.0) as u8, (color[1].clamp(0.0, 1.0) * 255.0) as u8, (color[2].clamp(0.0, 1.0) * 255.0) as u8, (color[3].clamp(0.0, 1.0) * 255.0) as u8);
            if s.len() == 1 {
                painter.circle_filled(s[0], px * 0.5, c);
                return;
            }
            painter.add(egui::Shape::line(s.clone(), Stroke::new(px, c)));
            // round ends
            painter.circle_filled(s[0], px * 0.5, c);
            painter.circle_filled(*s.last().unwrap(), px * 0.5, c);
        };
        for s in &strokes {
            draw(&s.points, s.width, s.color);
        }
        if ed.pencil_draw.is_none() {
            return;
        }
        if !self.pencil_ui.current.is_empty() {
            let c = self.pencil_ui.color;
            draw(&self.pencil_ui.current, self.pencil_ui.width_world, [c[0], c[1], c[2], self.pencil_ui.opacity]);
        }
        if let Some(m) = ui.ctx().pointer_hover_pos().filter(|p| self.rect.contains(*p)) {
            let erase = self.pencil_ui.tool == PencilTool::Erase || ui.input(|i| i.modifiers.command);
            if erase {
                painter.circle_stroke(m, self.pencil_ui.erase_px, Stroke::new(1.2, Color32::from_rgb(255, 120, 120)));
            } else {
                let c = self.pencil_ui.color;
                painter.circle_stroke(m, self.pencil_ui.thickness * 0.5 + 1.5, Stroke::new(1.0, Color32::from_rgb((c[0] * 255.0) as u8, (c[1] * 255.0) as u8, (c[2] * 255.0) as u8)));
            }
        }
    }

    pub(super) fn pencil_header(&mut self, ui: &mut Ui, ed: &mut Editor) {
        for (t, l, tip) in [(PencilTool::Draw, "✏", "Draw (D)"), (PencilTool::Line, "╱", "Line (L)"), (PencilTool::Erase, "⌫", "Erase (E, or hold Ctrl)")] {
            if ui.add(egui::Button::selectable(self.pencil_ui.tool == t, egui::RichText::new(l).size(13.0))).on_hover_text(tip).clicked() {
                self.pencil_ui.tool = t;
            }
        }
        ui.color_edit_button_rgb(&mut self.pencil_ui.color);
        for c in &PALETTE[..6] {
            let col = Color32::from_rgb((c[0] * 255.0) as u8, (c[1] * 255.0) as u8, (c[2] * 255.0) as u8);
            let (r, resp) = ui.allocate_exact_size(vec2(14.0, 14.0), egui::Sense::click());
            ui.painter().circle_filled(r.center(), 6.0, col);
            if self.pencil_ui.color == *c {
                ui.painter().circle_stroke(r.center(), 7.0, Stroke::new(1.5, Color32::WHITE));
            }
            if resp.clicked() {
                self.pencil_ui.color = *c;
                if self.pencil_ui.tool == PencilTool::Erase {
                    self.pencil_ui.tool = PencilTool::Draw;
                }
            }
        }
        ui.add(egui::DragValue::new(&mut self.pencil_ui.thickness).range(1.0..=80.0).speed(0.2).prefix("w ").suffix(" px"));
        egui::ComboBox::from_id_salt("gp_place").selected_text(egui::RichText::new(match self.pencil_ui.placement {
            Placement::View => "View",
            Placement::Surface => "Surface",
        }).size(12.0)).width(58.0).show_ui(ui, |ui| {
            ui.add(egui::Slider::new(&mut self.pencil_ui.opacity, 0.05..=1.0).text("Opacity"));
            ui.selectable_value(&mut self.pencil_ui.placement, Placement::View, "View plane").on_hover_text("On a plane through the 3D cursor (or what you start on), facing you");
            ui.selectable_value(&mut self.pencil_ui.placement, Placement::Surface, "Surface").on_hover_text("Stick to the objects under the pen");
        });
        ui.toggle_value(&mut self.pencil_ui.new_per_frame, egui::RichText::new("🎞").size(12.0)).on_hover_text("Frame by frame: each frame you draw on gets a new drawing (2D animation)");
        if let Some(p) = ed.pencil_draw.and_then(|id| ed.scene.get_mut(id)).and_then(|o| o.pencil.as_mut()) {
            ui.toggle_value(&mut p.onion, egui::RichText::new("🧅").size(12.0)).on_hover_text("Onion skin: show the previous (red) and next (blue) drawings");
        }
    }
}
