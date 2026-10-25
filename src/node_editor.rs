//! The Shader Editor window: node boxes with sockets, bezier links, add / delete / connect.

use crate::editor::Editor;
use crate::nodes::{MathOp, MixMode, NodeKind};
use crate::theme;
use egui::{Color32, Pos2, Rect, Sense, Stroke, Vec2, pos2, vec2};

#[derive(Default)]
pub struct NodeEditor {
    pub open: bool,
    pub graph: u32,
    pan: Vec2,
    /// dragging a new link from (node, output)
    drag_link: Option<(u32, usize)>,
    selected: Option<u32>,
    snapshotted: bool,
}

const W: f32 = 176.0;
const HEAD: f32 = 22.0;
const ROW: f32 = 18.0;
const PROW: f32 = 22.0;

fn node_height(k: &NodeKind) -> f32 {
    crate::nodes::node_height_estimate(k)
}

fn param_rows(k: &NodeKind) -> usize {
    crate::nodes::param_rows(k)
}

fn header_color(k: &NodeKind) -> Color32 {
    match k {
        NodeKind::Output { .. } => Color32::from_rgb(150, 60, 60),
        NodeKind::Coords { .. } => Color32::from_rgb(110, 60, 150),
        NodeKind::Image { .. } | NodeKind::Noise { .. } | NodeKind::Voronoi { .. } | NodeKind::Checker { .. } | NodeKind::Bricks { .. } | NodeKind::Waves { .. } => Color32::from_rgb(190, 120, 40),
        NodeKind::Gradient { .. } => Color32::from_rgb(190, 120, 40),
        NodeKind::Ramp { .. } | NodeKind::Mix { .. } | NodeKind::Invert | NodeKind::Rgb { .. } | NodeKind::HueSat { .. } => Color32::from_rgb(70, 110, 150),
        NodeKind::NormalMap { .. } => Color32::from_rgb(90, 90, 170),
        NodeKind::Math { .. } | NodeKind::Value { .. } => Color32::from_rgb(70, 120, 90),
    }
}

impl NodeEditor {
    pub fn open_graph(&mut self, id: u32) {
        self.open = true;
        self.graph = id;
        self.pan = Vec2::ZERO;
        self.selected = None;
    }

    pub fn show(&mut self, ctx: &egui::Context, ed: &mut Editor) {
        if !self.open {
            return;
        }
        if ed.scene.graph(self.graph).is_none() {
            self.graph = ed.scene.node_graphs.first().map(|g| g.id).unwrap_or(0);
        }
        let textures: Vec<(u32, String)> = ed.scene.textures.iter().map(|t| (t.id, t.name.clone())).collect();
        let graphs: Vec<(u32, String)> = ed.scene.node_graphs.iter().map(|g| (g.id, g.name.clone())).collect();
        let users = |ed: &Editor, gid: u32| ed.scene.objects.iter().filter(|o| o.material.graph == gid).count();
        let mut open = true;
        let title = ed.scene.graph(self.graph).map(|g| format!("Shader Editor — {}", g.name)).unwrap_or_else(|| "Shader Editor".into());
        let before = ed.scene.graph(self.graph).cloned();
        let mut add_at: Option<(NodeKind, [f32; 2])> = None;
        egui::Window::new(title)
            .id(egui::Id::unique("shader_editor"))
            .open(&mut open)
            .default_size(vec2(940.0, 540.0))
            .resizable(true)
            .collapsible(false)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    egui::ComboBox::from_id_salt("graph_pick").selected_text(graphs.iter().find(|g| g.0 == self.graph).map(|g| g.1.clone()).unwrap_or_else(|| "—".into())).show_ui(ui, |ui| {
                        for (id, n) in &graphs {
                            ui.selectable_value(&mut self.graph, *id, format!("{n}  ({} users)", users(ed, *id)));
                        }
                    });
                    ui.menu_button("Add ▾", |ui| {
                        for k in NodeKind::catalog() {
                            if ui.button(k.label()).clicked() {
                                let c = -self.pan + vec2(260.0, 160.0);
                                add_at = Some((k, [c.x, c.y]));
                                ui.close();
                            }
                        }
                    });
                    let mut arrange = false;
                    if ui.button("Arrange").on_hover_text("Lay the nodes out in columns").clicked() {
                        arrange = true;
                    }
                    if arrange {
                        if let Some(g) = ed.scene.graph_mut(self.graph) {
                            g.auto_layout();
                            self.pan = Vec2::ZERO;
                        }
                    }
                    ui.label(egui::RichText::new("Drag from an output dot to an input · drag a header to move · right-click to add · Del removes · middle-drag pans").small().color(theme::TEXT_DIM));
                });
                ui.separator();
                let Some(g) = ed.scene.graph_mut(self.graph) else {
                    ui.label("No node material. Pick one in the Material tab (Use Nodes or a preset).");
                    return;
                };
                let (canvas, resp) = ui.allocate_exact_size(ui.available_size(), Sense::click_and_drag());
                let painter = ui.painter_at(canvas);
                painter.rect_filled(canvas, 4.0, Color32::from_rgb(30, 31, 34));
                // grid
                let step = 24.0;
                let off = vec2(self.pan.x.rem_euclid(step), self.pan.y.rem_euclid(step));
                let mut x = canvas.left() + off.x;
                while x < canvas.right() {
                    painter.line_segment([pos2(x, canvas.top()), pos2(x, canvas.bottom())], Stroke::new(1.0, Color32::from_rgb(38, 39, 43)));
                    x += step;
                }
                let mut y = canvas.top() + off.y;
                while y < canvas.bottom() {
                    painter.line_segment([pos2(canvas.left(), y), pos2(canvas.right(), y)], Stroke::new(1.0, Color32::from_rgb(38, 39, 43)));
                    y += step;
                }
                if resp.dragged_by(egui::PointerButton::Middle) {
                    self.pan += resp.drag_delta();
                }
                if resp.hovered() {
                    let s = ui.input(|i| i.smooth_scroll_delta);
                    self.pan += s;
                }
                let origin = canvas.min.to_vec2() + self.pan;
                let rect_of = |n: &crate::nodes::Node| Rect::from_min_size(pos2(n.pos[0], n.pos[1]) + origin, vec2(W, node_height(&n.kind)));
                let out_pos = |n: &crate::nodes::Node, i: usize| rect_of(n).min + vec2(W, HEAD + ROW * (i as f32 + 0.5));
                let in_pos = |n: &crate::nodes::Node, i: usize| {
                    let r = rect_of(n);
                    r.min + vec2(0.0, HEAD + n.kind.outputs().len() as f32 * ROW + param_rows(&n.kind) as f32 * PROW + ROW * (i as f32 + 0.5))
                };
                let bez = |a: Pos2, b: Pos2, col: Color32| {
                    let dx = ((b.x - a.x).abs() * 0.5).max(40.0);
                    egui::epaint::CubicBezierShape::from_points_stroke([a, a + vec2(dx, 0.0), b - vec2(dx, 0.0), b], false, Color32::TRANSPARENT, Stroke::new(2.0, col))
                };
                // links
                for l in &g.links {
                    if let (Some(a), Some(b)) = (g.node(l.from.0), g.node(l.to.0)) {
                        painter.add(bez(out_pos(a, l.from.1), in_pos(b, l.to.1), Color32::from_rgb(170, 170, 180)));
                    }
                }
                let pointer = ui.input(|i| i.pointer.hover_pos());
                let pressed = ui.input(|i| i.pointer.primary_pressed());
                let released = ui.input(|i| i.pointer.primary_released());
                // start / finish link drags
                let press_at = ui.input(|i| i.pointer.press_origin()).or(pointer);
                if pressed {
                    if let Some(p) = press_at.filter(|p| canvas.contains(*p)) {
                        'find: for n in &g.nodes {
                            for i in 0..n.kind.outputs().len() {
                                if out_pos(n, i).distance(p) < 8.0 {
                                    self.drag_link = Some((n.id, i));
                                    break 'find;
                                }
                            }
                            for i in 0..n.kind.inputs().len() {
                                if in_pos(n, i).distance(p) < 8.0 {
                                    // pick up an existing link from its input
                                    if let Some(l) = g.input_link(n.id, i).copied() {
                                        self.drag_link = Some(l.from);
                                        g.links.retain(|x| *x != l);
                                    }
                                    break 'find;
                                }
                            }
                        }
                    }
                }
                if let (Some(from), Some(p)) = (self.drag_link, pointer) {
                    if let Some(a) = g.node(from.0) {
                        painter.add(bez(out_pos(a, from.1), p, Color32::from_rgb(255, 210, 90)));
                    }
                    if released {
                        let target = g.nodes.iter().find_map(|n| (0..n.kind.inputs().len()).find(|&i| in_pos(n, i).distance(p) < 10.0).map(|i| (n.id, i)));
                        if let Some(to) = target {
                            g.connect(from, to);
                        }
                        self.drag_link = None;
                    }
                }
                if !ui.input(|i| i.pointer.primary_down()) {
                    self.drag_link = None;
                }
                // nodes
                let mut remove = None;
                let ids: Vec<u32> = g.nodes.iter().map(|n| n.id).collect();
                for id in ids {
                    let Some(idx) = g.nodes.iter().position(|n| n.id == id) else { continue };
                    let r = rect_of(&g.nodes[idx]);
                    if !r.intersects(canvas) {
                        continue;
                    }
                    let sel = self.selected == Some(id);
                    let kind_label = g.nodes[idx].kind.label();
                    let hc = header_color(&g.nodes[idx].kind);
                    painter.rect_filled(r, 5.0, Color32::from_rgb(52, 53, 58));
                    painter.rect_filled(Rect::from_min_size(r.min, vec2(W, HEAD)), egui::CornerRadius { nw: 5, ne: 5, sw: 0, se: 0 }, hc);
                    painter.rect_stroke(r, 5.0, Stroke::new(if sel { 2.0 } else { 1.0 }, if sel { Color32::WHITE } else { Color32::from_rgb(20, 20, 22) }), egui::StrokeKind::Outside);
                    painter.text(r.min + vec2(8.0, HEAD * 0.5), egui::Align2::LEFT_CENTER, kind_label, egui::FontId::proportional(12.5), Color32::WHITE);
                    // header drag moves the node
                    let hr = ui.interact(Rect::from_min_size(r.min, vec2(W, HEAD)), egui::Id::unique(("node_head", id)), Sense::click_and_drag());
                    if hr.dragged() && self.drag_link.is_none() {
                        let d = hr.drag_delta();
                        g.nodes[idx].pos[0] += d.x;
                        g.nodes[idx].pos[1] += d.y;
                    }
                    if hr.clicked() || hr.drag_started() {
                        self.selected = Some(id);
                    }
                    // outputs
                    let n = &g.nodes[idx];
                    for (i, name) in n.kind.outputs().iter().enumerate() {
                        let p = out_pos(n, i);
                        painter.text(p - vec2(10.0, 0.0), egui::Align2::RIGHT_CENTER, *name, egui::FontId::proportional(11.5), theme::TEXT);
                        painter.circle(p, 5.0, Color32::from_rgb(200, 200, 90), Stroke::new(1.0, Color32::BLACK));
                    }
                    // inputs
                    for (i, name) in n.kind.inputs().iter().enumerate() {
                        let p = in_pos(n, i);
                        let linked = g.input_link(id, i).is_some();
                        painter.text(p + vec2(10.0, 0.0), egui::Align2::LEFT_CENTER, *name, egui::FontId::proportional(11.5), if linked { theme::TEXT } else { theme::TEXT_DIM });
                        painter.circle(p, 5.0, if linked { Color32::from_rgb(160, 160, 220) } else { Color32::from_rgb(90, 90, 110) }, Stroke::new(1.0, Color32::BLACK));
                    }
                    // parameters
                    let prect = Rect::from_min_size(r.min + vec2(8.0, HEAD + n.kind.outputs().len() as f32 * ROW), vec2(W - 16.0, param_rows(&n.kind) as f32 * PROW));
                    let node = &mut g.nodes[idx];
                    ui.scope_builder(egui::UiBuilder::new().max_rect(prect).id_salt(("node_params", id)), |ui| {
                        ui.spacing_mut().item_spacing.y = 2.0;
                        ui.style_mut().spacing.interact_size.y = 18.0;
                        params(ui, &mut node.kind, &textures, id);
                    });
                    if (hr.clicked() || r.contains(press_at.unwrap_or(Pos2::ZERO)) && pressed) && self.drag_link.is_none() {
                        self.selected = Some(id);
                    }
                    if sel && ui.input(|i| i.key_pressed(egui::Key::Delete) || i.key_pressed(egui::Key::X)) && !ui.ctx().egui_wants_keyboard_input() {
                        remove = Some(id);
                    }
                }
                if let Some(id) = remove {
                    g.remove_node(id);
                    self.selected = None;
                }
                // right-click: add menu at the pointer
                resp.context_menu(|ui| {
                    let at = ui.ctx().input(|i| i.pointer.interact_pos()).unwrap_or(canvas.center()) - origin;
                    for k in NodeKind::catalog() {
                        if ui.button(k.label()).clicked() {
                            add_at = Some((k, [at.x, at.y]));
                            ui.close();
                        }
                    }
                });
                if resp.clicked() {
                    self.selected = None;
                }
            });
        if let Some((k, p)) = add_at {
            if let Some(g) = ed.scene.graph_mut(self.graph) {
                let id = g.add(k, p);
                self.selected = Some(id);
            }
        }
        // one undo step per interaction
        let after = ed.scene.graph(self.graph).cloned();
        if before != after {
            if !self.snapshotted {
                if let (Some(b), Some(gi)) = (before, ed.scene.node_graphs.iter().position(|g| g.id == self.graph)) {
                    let cur = std::mem::replace(&mut ed.scene.node_graphs[gi], b);
                    ed.snapshot();
                    ed.scene.node_graphs[gi] = cur;
                }
                self.snapshotted = true;
            }
            ed.dirty = true;
        }
        if !ctx.input(|i| i.pointer.any_down()) {
            self.snapshotted = false;
        }
        if !open {
            self.open = false;
        }
    }
}

fn color_edit(ui: &mut egui::Ui, c: &mut [f32; 3]) {
    ui.color_edit_button_rgb(c);
}

fn params(ui: &mut egui::Ui, k: &mut NodeKind, textures: &[(u32, String)], id: u32) {
    let dv = |ui: &mut egui::Ui, v: &mut f32, label: &str, speed: f64| {
        ui.add(egui::DragValue::new(v).speed(speed).prefix(format!("{label} ")).max_decimals(3));
    };
    match k {
        NodeKind::Output { bump } => dv(ui, bump, "Bump", 0.02),
        NodeKind::Coords { scale, rotate, object } => {
            ui.horizontal(|ui| {
                ui.selectable_value(object, false, "UV");
                ui.selectable_value(object, true, "Object");
            });
            dv(ui, scale, "Scale", 0.05);
            dv(ui, rotate, "Rotate°", 1.0);
        }
        NodeKind::Gradient { mode } => {
            egui::ComboBox::from_id_salt(("grad", id)).width(110.0).selected_text(format!("{mode:?}")).show_ui(ui, |ui| {
                for m in [crate::nodes::GradMode::Linear, crate::nodes::GradMode::Radial, crate::nodes::GradMode::Spherical] {
                    ui.selectable_value(mode, m, format!("{m:?}"));
                }
            });
        }
        NodeKind::HueSat { hue, saturation, value } => {
            ui.add(egui::DragValue::new(hue).range(-0.5..=0.5).speed(0.005).prefix("Hue ").max_decimals(3));
            ui.add(egui::DragValue::new(saturation).range(0.0..=4.0).speed(0.01).prefix("Saturation ").max_decimals(3));
            ui.add(egui::DragValue::new(value).range(0.0..=4.0).speed(0.01).prefix("Value ").max_decimals(3));
        }
        NodeKind::NormalMap { strength } => {
            ui.add(egui::DragValue::new(strength).range(0.0..=10.0).speed(0.02).prefix("Strength ").max_decimals(3));
        }
        NodeKind::Image { texture } => {
            let cur = textures.iter().find(|t| t.0 == *texture).map(|t| t.1.clone()).unwrap_or_else(|| "None".into());
            egui::ComboBox::from_id_salt(("img", id)).width(150.0).selected_text(cur).show_ui(ui, |ui| {
                ui.selectable_value(texture, 0, "None");
                for (tid, n) in textures {
                    ui.selectable_value(texture, *tid, n);
                }
            });
        }
        NodeKind::Noise { scale, detail, distortion } => {
            dv(ui, scale, "Scale", 0.1);
            ui.add(egui::DragValue::new(detail).range(1..=8).prefix("Detail "));
            dv(ui, distortion, "Distortion", 0.05);
        }
        NodeKind::Voronoi { scale, randomness } => {
            dv(ui, scale, "Scale", 0.1);
            ui.add(egui::Slider::new(randomness, 0.0..=1.0).text("Random"));
        }
        NodeKind::Checker { scale } => dv(ui, scale, "Scale", 0.1),
        NodeKind::Bricks { scale, mortar } => {
            dv(ui, scale, "Scale", 0.1);
            ui.add(egui::Slider::new(mortar, 0.0..=0.3).text("Mortar"));
        }
        NodeKind::Waves { scale, distortion } => {
            dv(ui, scale, "Scale", 0.1);
            dv(ui, distortion, "Distortion", 0.05);
        }
        NodeKind::Ramp { stops } => {
            // gradient preview
            let (r, _) = ui.allocate_exact_size(vec2(ui.available_width(), 14.0), Sense::hover());
            let n = 40;
            for i in 0..n {
                let t = i as f32 / n as f32;
                let mut s: Vec<&(f32, [f32; 3])> = stops.iter().collect();
                s.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
                let c = sample_stops(&s, t);
                let x0 = r.left() + r.width() * t;
                ui.painter().rect_filled(Rect::from_min_max(pos2(x0, r.top()), pos2(x0 + r.width() / n as f32 + 0.5, r.bottom())), 0.0, Color32::from_rgb((c[0] * 255.0) as u8, (c[1] * 255.0) as u8, (c[2] * 255.0) as u8));
            }
            let mut del = None;
            for (i, st) in stops.iter_mut().enumerate() {
                ui.horizontal(|ui| {
                    ui.add(egui::DragValue::new(&mut st.0).range(0.0..=1.0).speed(0.005).max_decimals(3));
                    color_edit(ui, &mut st.1);
                    if ui.small_button("✖").clicked() {
                        del = Some(i);
                    }
                });
            }
            if let Some(i) = del.filter(|_| stops.len() > 1) {
                stops.remove(i);
            }
            if ui.small_button("+ stop").clicked() {
                stops.push((0.5, [0.5; 3]));
            }
        }
        NodeKind::Mix { mode, fac, a, b } => {
            egui::ComboBox::from_id_salt(("mix", id)).width(150.0).selected_text(format!("{mode:?}")).show_ui(ui, |ui| {
                for m in [MixMode::Mix, MixMode::Multiply, MixMode::Add, MixMode::Overlay] {
                    ui.selectable_value(mode, m, format!("{m:?}"));
                }
            });
            ui.add(egui::Slider::new(fac, 0.0..=1.0).text("Fac"));
            ui.horizontal(|ui| {
                ui.label("A");
                color_edit(ui, a);
            });
            ui.horizontal(|ui| {
                ui.label("B");
                color_edit(ui, b);
            });
        }
        NodeKind::Math { op, a, b } => {
            egui::ComboBox::from_id_salt(("math", id)).width(150.0).selected_text(format!("{op:?}")).show_ui(ui, |ui| {
                for o in [MathOp::Add, MathOp::Multiply, MathOp::Subtract, MathOp::Power, MathOp::Greater, MathOp::Smooth] {
                    ui.selectable_value(op, o, format!("{o:?}"));
                }
            });
            dv(ui, a, "A", 0.01);
            dv(ui, b, "B", 0.01);
        }
        NodeKind::Invert => {}
        NodeKind::Rgb { color } => color_edit(ui, color),
        NodeKind::Value { value } => dv(ui, value, "Value", 0.01),
    }
}

fn sample_stops(s: &[&(f32, [f32; 3])], t: f32) -> [f32; 3] {
    if s.is_empty() {
        return [t; 3];
    }
    if t <= s[0].0 {
        return s[0].1;
    }
    for w in s.windows(2) {
        if t <= w[1].0 {
            let k = ((t - w[0].0) / (w[1].0 - w[0].0).max(1e-6)).clamp(0.0, 1.0);
            return [0, 1, 2].map(|i| w[0].1[i] + (w[1].1[i] - w[0].1[i]) * k);
        }
    }
    s.last().unwrap().1
}
