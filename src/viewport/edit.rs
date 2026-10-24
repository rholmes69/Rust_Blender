//! Edit-mode drawing, component picking and tools (extrude, inset, loop cut …).

use super::*;
use crate::editmesh::{self as em, SelectMode};
use crate::editor::EditOp;

/// World-space data of the mesh in edit mode, captured while drawing (used for picking).
pub struct EditDraw {
    pub verts: Vec<Vec3>,
    pub edges: Vec<(u32, u32)>,
    pub face_centers: Vec<Vec3>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Pick {
    Vert(u32),
    Edge(u32, u32),
    Face(usize),
}

const ORANGE: [f32; 4] = [1.0, 0.55, 0.12, 1.0];
const WIRE: [f32; 4] = [0.02, 0.02, 0.02, 0.85];

impl Viewport {
    /// Screen-space square dot at a world point (depth tested, slightly biased).
    fn push_dot(&mut self, p: Vec3, size: f32, col: [f32; 4], view: &Mat4, no_depth: bool) {
        let mut v = view.transform_point3(p);
        if !self.cam.ortho {
            if v.z > -0.06 {
                return;
            }
            v *= 0.994;
        } else {
            v.z += 0.05;
        }
        let proj = self.proj;
        let (s, z) = self.vproj(&proj, v);
        let (x, y) = self.to_px(s);
        let h = size * 0.5;
        let p4 = [
            [x - h, y - h, z],
            [x + h, y - h, z],
            [x + h, y + h, z],
            [x - h, y + h, z],
        ];
        self.prims.push(RPrim {
            p: p4,
            c: [col; 4],
            n: 4,
            id: 0,
            blend: true,
            no_depth,
            uv: RPrim::NO_UV,
            tex: 0,
            iw: [1.0; 4],
        });
    }

    pub(super) fn edit_overlay(
        &mut self,
        ed: &Editor,
        mesh: &crate::mesh::Mesh,
        wp: &[Vec3],
        view: &Mat4,
    ) {
        let Some(st) = ed.edit.as_ref() else { return };
        let Some(pm) = ed.scene.get(st.obj).and_then(|o| o.mesh.as_ref()) else {
            return;
        };
        let lw = self.px_scale;
        let xray = self.overlays.xray || ed.shading == Shading::Wireframe;
        let sel = |i: u32| st.sel.get(i as usize).copied().unwrap_or(false);
        let mut edges = vec![];
        for e in &mesh.edges {
            if !e.feature {
                continue;
            }
            let s = sel(e.a) && sel(e.b);
            let col = if s { ORANGE } else { WIRE };
            self.push_line(
                wp[e.a as usize],
                wp[e.b as usize],
                if s { 1.8 * lw } else { 1.2 * lw },
                col,
                0.006,
                view,
                xray,
            );
            edges.push((e.a, e.b));
        }
        let face_centers: Vec<Vec3> = (0..pm.faces.len())
            .map(|f| {
                let fv = &pm.faces[f];
                fv.iter().map(|&i| wp[i as usize]).sum::<Vec3>() / fv.len().max(1) as f32
            })
            .collect();
        match st.mode {
            SelectMode::Vertex => {
                for (i, p) in wp.iter().enumerate() {
                    let s = sel(i as u32);
                    self.push_dot(
                        *p,
                        if s { 6.0 } else { 5.0 } * lw,
                        if s { ORANGE } else { [0.0, 0.0, 0.0, 1.0] },
                        view,
                        xray,
                    );
                }
            }
            SelectMode::Face => {
                for (f, c) in face_centers.iter().enumerate() {
                    let s = em::face_selected(pm, &st.sel, f);
                    self.push_dot(
                        *c,
                        4.0 * lw,
                        if s { ORANGE } else { [0.0, 0.0, 0.0, 0.8] },
                        view,
                        xray,
                    );
                }
            }
            SelectMode::Edge => {}
        }
        self.edit_draw = Some(EditDraw {
            verts: wp.to_vec(),
            edges,
            face_centers,
        });
    }

    /// Is a world point visible (not hidden behind geometry) in the last rendered frame?
    pub(super) fn visible(&self, p: Vec3, xray: bool) -> bool {
        if xray {
            return true;
        }
        let Some(s) = self.project(p) else {
            return false;
        };
        let (x, y) = self.to_px(s);
        let (w, h) = (self.raster.w as i64, self.raster.h as i64);
        let view = self.cam.view();
        let vz = -view.transform_point3(p).z;
        let inv = self.proj.inverse();
        let lin_depth = |z: f32| {
            let v = inv * Vec4::new(0.0, 0.0, z, 1.0);
            -(v.z / v.w)
        };
        // accept if any nearby pixel is not in front of the point
        for dy in -2i64..=2 {
            for dx in -2i64..=2 {
                let (px, py) = (x as i64 + dx, y as i64 + dy);
                if px < 0 || py < 0 || px >= w || py >= h {
                    continue;
                }
                let d = self.raster.depth[(py * w + px) as usize];
                if d >= 1.0 {
                    return true;
                }
                let ld = lin_depth(d);
                if ld >= vz - (0.02 + vz * 0.01) {
                    return true;
                }
            }
        }
        false
    }

    pub(super) fn edit_pick(&self, ed: &Editor, mouse: Pos2) -> Option<Pick> {
        let st = ed.edit.as_ref()?;
        let d = self.edit_draw.as_ref()?;
        let xray = self.overlays.xray || ed.shading == Shading::Wireframe;
        match st.mode {
            SelectMode::Vertex => {
                let mut best: Option<(f32, u32)> = None;
                for (i, p) in d.verts.iter().enumerate() {
                    if let Some(s) = self.project(*p) {
                        let dist = (s - mouse).length();
                        if dist < 14.0
                            && best.is_none_or(|(b, _)| dist < b)
                            && self.visible(*p, xray)
                        {
                            best = Some((dist, i as u32));
                        }
                    }
                }
                best.map(|(_, i)| Pick::Vert(i))
            }
            SelectMode::Edge => self
                .nearest_edge(ed, mouse, 12.0)
                .map(|(a, b)| Pick::Edge(a, b)),
            SelectMode::Face => {
                let (x, y) = self.to_px(mouse);
                let id = self.raster.id_at(x.max(0.0) as usize, y.max(0.0) as usize);
                if id >= FACE_BASE {
                    return Some(Pick::Face((id - FACE_BASE) as usize));
                }
                // fall back to the nearest face dot (x-ray / thin faces)
                let mut best: Option<(f32, usize)> = None;
                for (f, c) in d.face_centers.iter().enumerate() {
                    if let Some(s) = self.project(*c) {
                        let dist = (s - mouse).length();
                        if dist < 12.0 && best.is_none_or(|(b, _)| dist < b) {
                            best = Some((dist, f));
                        }
                    }
                }
                best.map(|(_, f)| Pick::Face(f))
            }
        }
    }

    pub(super) fn nearest_edge(&self, ed: &Editor, mouse: Pos2, max: f32) -> Option<(u32, u32)> {
        let d = self.edit_draw.as_ref()?;
        let xray = self.overlays.xray || ed.shading == Shading::Wireframe;
        let mut best: Option<(f32, (u32, u32))> = None;
        for &(a, b) in &d.edges {
            let (pa, pb) = (d.verts[a as usize], d.verts[b as usize]);
            let (Some(sa), Some(sb)) = (self.project(pa), self.project(pb)) else {
                continue;
            };
            let dist = seg_dist(mouse, sa, sb);
            if dist < max
                && best.is_none_or(|(bd, _)| dist < bd)
                && self.visible((pa + pb) * 0.5, xray)
            {
                best = Some((dist, (a, b)));
            }
        }
        best.map(|(_, e)| e)
    }

    pub(super) fn edit_click(&mut self, ed: &mut Editor, mouse: Pos2, extend: bool) {
        let pick = self.edit_pick(ed, mouse);
        let Some(st) = ed.edit.as_mut() else { return };
        let Some(pm) = ed.scene.get(st.obj).and_then(|o| o.mesh.as_ref()) else {
            return;
        };
        st.fit(pm);
        let verts: Vec<u32> = match pick {
            Some(Pick::Vert(v)) => vec![v],
            Some(Pick::Edge(a, b)) => vec![a, b],
            Some(Pick::Face(f)) => pm.faces.get(f).cloned().unwrap_or_default(),
            None => vec![],
        };
        if !extend {
            st.sel.iter_mut().for_each(|s| *s = false);
        }
        if verts.is_empty() {
            return;
        }
        let all_on = verts.iter().all(|&v| st.sel[v as usize]);
        for v in verts {
            st.sel[v as usize] = !(extend && all_on);
        }
    }

    pub(super) fn edit_box(&mut self, ed: &mut Editor, r: Rect, extend: bool, subtract: bool) {
        let xray = self.overlays.xray || ed.shading == Shading::Wireframe;
        let Some(d) = self.edit_draw.as_ref() else {
            return;
        };
        let inside: Vec<bool> = d
            .verts
            .iter()
            .map(|p| self.project(*p).is_some_and(|s| r.contains(s)) && self.visible(*p, xray))
            .collect();
        let Some(st) = ed.edit.as_mut() else { return };
        let Some(pm) = ed.scene.get(st.obj).and_then(|o| o.mesh.as_ref()) else {
            return;
        };
        st.fit(pm);
        if !extend && !subtract {
            st.sel.iter_mut().for_each(|s| *s = false);
        }
        let mut mark = |v: u32| {
            if let Some(s) = st.sel.get_mut(v as usize) {
                *s = !subtract;
            }
        };
        match st.mode {
            SelectMode::Vertex => {
                for (i, ins) in inside.iter().enumerate() {
                    if *ins {
                        mark(i as u32);
                    }
                }
            }
            SelectMode::Edge => {
                for (a, b) in pm.edges() {
                    if inside.get(a as usize).copied().unwrap_or(false)
                        && inside.get(b as usize).copied().unwrap_or(false)
                    {
                        mark(a);
                        mark(b);
                    }
                }
            }
            SelectMode::Face => {
                for f in &pm.faces {
                    if f.iter()
                        .all(|&v| inside.get(v as usize).copied().unwrap_or(false))
                    {
                        for &v in f {
                            mark(v);
                        }
                    }
                }
            }
        }
    }

    /// Edit-mode keyboard handling. Returns true when the key was consumed.
    pub(super) fn edit_keys(&mut self, ctx: &egui::Context, ed: &mut Editor, mouse: Pos2) -> bool {
        use egui::Modifiers as M;
        let c = |m: M, k: Key| ctx.input_mut(|i| i.consume_key(m, k));
        for (k, mode) in [
            (Key::Num1, SelectMode::Vertex),
            (Key::Num2, SelectMode::Edge),
            (Key::Num3, SelectMode::Face),
        ] {
            if c(M::NONE, k) {
                if let Some(st) = ed.edit.as_mut() {
                    st.mode = mode;
                }
                return true;
            }
        }
        if c(M::ALT, Key::A) {
            let _ = ed.edit_op(EditOp::SelectNone);
        } else if c(M::COMMAND, Key::I) {
            let _ = ed.edit_op(EditOp::SelectInvert);
        } else if c(M::NONE, Key::A) {
            let all = ed.edit.as_ref().is_some_and(|s| s.sel.iter().all(|x| *x));
            let _ = ed.edit_op(if all {
                EditOp::SelectNone
            } else {
                EditOp::SelectAll
            });
        } else if c(M::NONE, Key::E) {
            self.begin_extrude(ed, mouse);
        } else if c(M::NONE, Key::I) {
            self.begin_inset(ed, mouse);
        } else if c(M::COMMAND, Key::R) {
            self.loopcut = Some(1);
        } else if c(M::NONE, Key::K) {
            self.knife = Some(vec![]);
        } else if c(M::NONE, Key::U) {
            self.report(ed.execute(&[crate::editor::Command { action: "unwrap".into(), name: Some("smart".into()), ..Default::default() }]).first().cloned().map(|l| if l.starts_with('✔') { Ok(l) } else { Err(l) }).unwrap_or(Ok(String::new())));
            self.flash = Some(("Smart UV Project".into(), std::time::Instant::now()));
        } else if c(M::COMMAND, Key::B) {
            self.begin_bevel(ed, mouse);
        } else if c(M::NONE, Key::O) {
            self.proportional = !self.proportional;
            self.flash = Some((
                format!("Proportional editing {}", if self.proportional { "on — mouse wheel changes the radius during G/R/S" } else { "off" }),
                std::time::Instant::now(),
            ));
        } else if c(M::NONE, Key::M) {
            self.report(ed.edit_op(EditOp::Merge));
        } else if c(M::NONE, Key::X) || c(M::NONE, Key::Delete) {
            self.report(ed.edit_op(EditOp::Delete));
        } else if c(M::ALT, Key::N) {
            self.report(ed.edit_op(EditOp::FlipNormals));
        } else {
            return false;
        }
        true
    }

    pub(super) fn report(&mut self, r: Result<String, String>) {
        if let Err(e) = r {
            self.status_hint = format!("⚠ {e}");
            self.flash = Some((e, std::time::Instant::now()));
        }
    }

    pub fn begin_extrude(&mut self, ed: &mut Editor, mouse: Pos2) {
        ed.snapshot();
        let before = ed.undo_stack.len();
        // extrude without its own undo step, then grab along the normal
        let r = {
            let undo = std::mem::take(&mut ed.undo_stack);
            let r = ed.edit_op(EditOp::Extrude(0.0));
            ed.undo_stack = undo;
            r
        };
        if let Err(e) = r {
            ed.undo_stack.truncate(before - 1);
            self.report(Err(e));
            return;
        }
        let normal = ed.edit_mesh().map(|(pm, st, m)| {
            let n: Vec3 = em::selected_faces(pm, &st.sel)
                .iter()
                .map(|&f| pm.face_normal(f))
                .sum();
            (Mat3::from_mat4(m).inverse().transpose() * n).normalize_or_zero()
        });
        self.begin_vert_modal(ed, ModalKind::Grab, mouse, false);
        if let Some(m) = &mut self.modal {
            m.axis_vec = normal.filter(|n| n.length() > 0.5);
            m.own_snapshot = false;
        }
    }

    pub fn begin_inset(&mut self, ed: &mut Editor, mouse: Pos2) {
        let Some((pm, st, _)) = ed.edit_mesh() else {
            return;
        };
        if em::selected_faces(pm, &st.sel).is_empty() {
            self.report(Err("select faces to inset".into()));
            return;
        }
        let base = (pm.clone(), st.sel.clone());
        self.begin_vert_modal(ed, ModalKind::Inset, mouse, false);
        if let Some(m) = &mut self.modal {
            m.edit_base = Some(base);
        }
    }

    pub fn begin_bevel(&mut self, ed: &mut Editor, mouse: Pos2) {
        let Some((pm, st, _)) = ed.edit_mesh() else {
            return;
        };
        if em::selected_edges(pm, &st.sel).is_empty() {
            self.report(Err("select edges to bevel".into()));
            return;
        }
        let base = (pm.clone(), st.sel.clone());
        self.begin_vert_modal(ed, ModalKind::Bevel, mouse, false);
        if let Some(m) = &mut self.modal {
            m.edit_base = Some(base);
            m.prop.clear();
        }
        self.bevel_scroll = 0.0;
    }

    pub(super) fn apply_bevel(&mut self, ed: &mut Editor, m: &Modal, mouse: Pos2) {
        let Some((base, base_sel)) = &m.edit_base else {
            return;
        };
        let center = self.project(m.pivot).unwrap_or(self.rect.center());
        let d0 = (m.start_mouse - center).length();
        let d = (mouse - center).length();
        let mut w = (d - d0 * 0.8).max(0.0) * 0.5 * self.world_per_pixel(m.pivot);
        if let Ok(v) = m.typed.parse::<f32>() {
            w = v.max(0.0);
        }
        let Some(st) = ed.edit.as_ref() else { return };
        let Some(o) = ed.scene.get(st.obj) else { return };
        let scale = (glam::Mat3::from_mat4(o.transform.matrix()) * Vec3::ONE.normalize()).length().max(1e-6);
        let mut pm = base.clone();
        let mut sel = base_sel.clone();
        if w > 1e-4 && em::bevel(&mut pm, &mut sel, w / scale, self.bevel_segments).is_ok() {
            if let Some(st) = ed.edit.as_mut() {
                st.sel = sel;
                if let Some(o) = ed.scene.get_mut(st.obj) {
                    o.mesh = Some(pm);
                }
            }
        } else if let Some(st) = ed.edit.as_mut() {
            st.sel = base_sel.clone();
            if let Some(o) = ed.scene.get_mut(st.obj) {
                o.mesh = Some(base.clone());
            }
        }
        self.bevel_width = w;
    }

    /// Start a modal transform of the selected vertices.
    pub(super) fn begin_vert_modal(
        &mut self,
        ed: &mut Editor,
        kind: ModalKind,
        mouse: Pos2,
        from_gizmo: bool,
    ) {
        let Some((pm, st, m)) = ed.edit_mesh() else {
            return;
        };
        let Some(c) = em::selection_center(pm, &st.sel) else {
            self.report(Err("nothing selected".into()));
            return;
        };
        let verts: Vec<(u32, Vec3)> = pm
            .verts
            .iter()
            .enumerate()
            .filter(|(i, _)| st.sel.get(*i).copied().unwrap_or(false))
            .map(|(i, v)| (i as u32, *v))
            .collect();
        let pivot = m.transform_point3(c);
        // proportional editing: distances from every unselected vertex to the selection
        let mut prop = vec![];
        if self.proportional && matches!(kind, ModalKind::Grab | ModalKind::Rotate | ModalKind::Scale) {
            let sel_w: Vec<Vec3> = verts.iter().map(|(_, p)| m.transform_point3(*p)).collect();
            let n_unsel = pm.verts.len() - verts.len();
            if n_unsel * sel_w.len() <= 40_000_000 {
                for (i, p) in pm.verts.iter().enumerate() {
                    if st.sel.get(i).copied().unwrap_or(false) {
                        continue;
                    }
                    let w = m.transform_point3(*p);
                    let d = sel_w.iter().map(|s| s.distance_squared(w)).fold(f32::MAX, f32::min).sqrt();
                    prop.push((i as u32, *p, d));
                }
            }
        }
        ed.snapshot();
        self.modal = Some(Modal {
            kind,
            axis: None,
            axis_vec: None,
            start_mouse: mouse,
            originals: vec![],
            verts,
            edit_base: None,
            prop,
            pivot,
            typed: String::new(),
            from_gizmo,
            own_snapshot: true,
        });
    }

    /// Apply a world-space point mapping to the selected vertices of the modal.
    pub(super) fn apply_vert_modal(
        &mut self,
        ed: &mut Editor,
        m: &Modal,
        f: &dyn Fn(Vec3) -> Vec3,
    ) {
        let Some(st) = ed.edit.as_ref() else { return };
        let Some(o) = ed.scene.get_mut(st.obj) else {
            return;
        };
        let mat = o.transform.matrix();
        let inv = mat.inverse();
        if let Some(pm) = o.mesh.as_mut() {
            for (i, p0) in &m.verts {
                if let Some(v) = pm.verts.get_mut(*i as usize) {
                    *v = inv.transform_point3(f(mat.transform_point3(*p0)));
                }
            }
            for (i, p0, d) in &m.prop {
                if let Some(v) = pm.verts.get_mut(*i as usize) {
                    let w = em::falloff(*d, self.prop_radius);
                    *v = if w > 0.0 {
                        let a = mat.transform_point3(*p0);
                        inv.transform_point3(a + (f(a) - a) * w)
                    } else {
                        *p0
                    };
                }
            }
        }
    }

    pub(super) fn apply_inset(&mut self, ed: &mut Editor, m: &Modal, mouse: Pos2) {
        let Some((base, base_sel)) = &m.edit_base else {
            return;
        };
        let center = self.project(m.pivot).unwrap_or(self.rect.center());
        let d0 = (m.start_mouse - center).length().max(20.0);
        let mut t = (1.0 - (mouse - center).length() / d0).clamp(0.0, 0.95);
        if let Ok(v) = m.typed.parse::<f32>() {
            t = v.clamp(0.0, 0.95);
        }
        let mut pm = base.clone();
        let mut sel = base_sel.clone();
        if em::inset(&mut pm, &mut sel, t).is_ok() {
            if let Some(st) = ed.edit.as_mut() {
                st.sel = sel;
                if let Some(o) = ed.scene.get_mut(st.obj) {
                    o.mesh = Some(pm);
                }
            }
        }
        self.inset_amount = t;
    }

    pub(super) fn cancel_vert_modal(&mut self, ed: &mut Editor, m: &Modal) {
        if let Some((base, sel)) = &m.edit_base {
            if let Some(st) = ed.edit.as_mut() {
                st.sel = sel.clone();
                if let Some(o) = ed.scene.get_mut(st.obj) {
                    o.mesh = Some(base.clone());
                }
            }
        } else {
            self.apply_vert_modal(ed, m, &|p| p);
        }
    }

    /// Loop-cut tool: hover preview, wheel = number of cuts, click = apply.
    pub(super) fn loopcut_input(
        &mut self,
        ctx: &egui::Context,
        resp: &egui::Response,
        ed: &mut Editor,
        mouse: Pos2,
    ) -> bool {
        let Some(cuts) = self.loopcut else {
            return false;
        };
        if !ed.editing() {
            self.loopcut = None;
            return false;
        }
        self.loopcut_edge = self.nearest_edge(ed, mouse, 40.0);
        let scroll = ctx.input(|i| i.smooth_scroll_delta.y);
        self.loopcut_scroll += scroll;
        if self.loopcut_scroll.abs() > 30.0 {
            self.loopcut =
                Some(((cuts as i32) + self.loopcut_scroll.signum() as i32).clamp(1, 16) as usize);
            self.loopcut_scroll = 0.0;
        }
        for k in [(Key::Plus, 1i32), (Key::Equals, 1), (Key::Minus, -1)] {
            if ctx.input(|i| i.key_pressed(k.0)) {
                self.loopcut =
                    Some(((self.loopcut.unwrap_or(1) as i32) + k.1).clamp(1, 16) as usize);
            }
        }
        if ctx.input(|i| i.key_pressed(Key::Escape)) || resp.clicked_by(PointerButton::Secondary) {
            self.loopcut = None;
        } else if resp.clicked_by(PointerButton::Primary) {
            if let Some(e) = self.loopcut_edge {
                let r = ed.edit_op(EditOp::LoopCut {
                    edge: e,
                    cuts: self.loopcut.unwrap_or(1),
                });
                self.report(r);
            }
            self.loopcut = None;
        }
        self.status_hint = format!(
            "Loop Cut: {} cut(s)   [Wheel] number of cuts   [LMB] apply   [RMB/Esc] cancel",
            self.loopcut.unwrap_or(cuts)
        );
        ctx.request_repaint();
        true
    }

    pub(super) fn draw_loopcut_preview(&self, painter: &egui::Painter, ed: &Editor) {
        let (Some(cuts), Some(edge)) = (self.loopcut, self.loopcut_edge) else {
            return;
        };
        let Some((pm, _, m)) = ed.edit_mesh() else {
            return;
        };
        let (faces, _) = em::edge_ring(pm, edge);
        let key = |a: u32, b: u32| (a.min(b), a.max(b));
        let (_, ring_edges) = em::edge_ring(pm, edge);
        let rset: std::collections::HashSet<(u32, u32)> = ring_edges.into_iter().collect();
        let col = Color32::from_rgb(255, 230, 40);
        for f in faces {
            let q = &pm.faces[f];
            let Some(k) = (0..4).find(|&i| {
                rset.contains(&key(q[i], q[(i + 1) % 4]))
                    && rset.contains(&key(q[(i + 2) % 4], q[(i + 3) % 4]))
            }) else {
                continue;
            };
            let (a, b, c, d) = (q[k], q[(k + 1) % 4], q[(k + 2) % 4], q[(k + 3) % 4]);
            let w = |i: u32| m.transform_point3(pm.verts[i as usize]);
            for j in 1..=cuts {
                let t = j as f32 / (cuts + 1) as f32;
                let p1 = w(a).lerp(w(b), t);
                let p2 = w(d).lerp(w(c), t);
                if let (Some(s1), Some(s2)) = (self.project(p1), self.project(p2)) {
                    painter.line_segment([s1, s2], Stroke::new(2.0, col));
                }
            }
        }
        if let Some(d) = &self.edit_draw {
            if let (Some(sa), Some(sb)) = (
                d.verts.get(edge.0 as usize).and_then(|p| self.project(*p)),
                d.verts.get(edge.1 as usize).and_then(|p| self.project(*p)),
            ) {
                painter.line_segment([sa, sb], Stroke::new(3.0, Color32::from_rgb(255, 255, 255)));
            }
        }
    }
}
