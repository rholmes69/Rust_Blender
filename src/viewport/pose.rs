//! Armatures in the viewport: bone drawing and Pose Mode (select bones, R rotate, G move,
//! Alt+R / Alt+G clear, I keyframe).

use super::*;
use glam::Quat;

#[derive(Default)]
pub struct PoseUi {
    modal: Option<PoseModal>,
}

struct PoseModal {
    bone: usize,
    rotate: bool,
    orig: crate::armature::BonePose,
    start: Pos2,
    /// head of the bone on screen (rotation pivot)
    pivot: Pos2,
    head_world: Vec3,
}

impl Viewport {
    /// Octahedral bones, drawn in front of meshes.
    pub(super) fn draw_bones(&mut self, o: &crate::scene::Object, pid: u32, col: [f32; 4], view: &Mat4) {
        let Some(arm) = o.armature.as_ref() else { return };
        let aw = o.transform.matrix();
        let editing = self.bone_edit_draw.filter(|(id, _, _)| *id == o.id);
        let g = if editing.is_some() { arm.rest_globals() } else { arm.pose_globals() };
        let posing = self.pose_draw.filter(|(id, _)| *id == o.id).or(editing.map(|e| (e.0, e.1)));
        let lw = self.px_scale;
        // in pose mode bones aren't object-pickable (and get no selection outline)
        if posing.is_some() {
            self.line_id = 0;
        }
        for (i, (b, m)) in arm.bones.iter().zip(&g).enumerate() {
            let m = aw * *m;
            let head = m.transform_point3(Vec3::ZERO);
            let tail = m.transform_point3(b.tail_local());
            let len = head.distance(tail);
            // cross-section axes perpendicular to the bone
            let dir = (tail - head).normalize_or(Vec3::Z);
            let ref_ax = m.transform_vector3(Vec3::X).normalize_or(Vec3::X);
            let ref_ax = if ref_ax.dot(dir).abs() > 0.9 { m.transform_vector3(Vec3::Z).normalize_or(Vec3::Y) } else { ref_ax };
            let x = (ref_ax - dir * ref_ax.dot(dir)).normalize_or_zero() * len * 0.1;
            let z = dir.cross(x);
            let mid = head.lerp(tail, 0.1);
            let ring = [mid + x, mid + z, mid - x, mid - z];
            let c = match posing {
                Some((_, a)) if a == i => [0.45, 0.75, 1.0, 1.0],
                Some(_) => [0.55, 0.6, 0.68, 1.0],
                None => col,
            };
            let w = if posing.is_some_and(|(_, a)| a == i) { 2.2 * lw } else { 1.3 * lw };
            for k in 0..4 {
                self.push_line(head, ring[k], w, c, 0.0, view, true);
                self.push_line(ring[k], tail, w, c, 0.0, view, true);
                self.push_line(ring[k], ring[(k + 1) % 4], w, c, 0.0, view, true);
            }
            // relationship line to the parent's tail when the bone isn't connected
            if let Some(p) = b.parent {
                let pm = aw * g[p];
                let ptail = pm.transform_point3(arm.bones[p].tail_local());
                if ptail.distance(head) > len * 0.05 {
                    self.push_line(ptail, head, lw, [c[0], c[1], c[2], 0.35], 0.0, view, true);
                }
            }
        }
        if editing.is_some() {
            self.draw_bone_joints(o, view);
        }
        self.line_id = pid;
    }

    /// Bone under the mouse (screen distance to the bone segment).
    pub(super) fn bone_at(&self, ed: &Editor, id: u64, mouse: Pos2) -> Option<usize> {
        let o = ed.scene.get(id)?;
        let arm = o.armature.as_ref()?;
        let aw = o.transform.matrix();
        let mut best: Option<(f32, usize)> = None;
        for (i, (h, t)) in arm.segments(&arm.pose_globals()).into_iter().enumerate() {
            let (Some(a), Some(b)) = (self.project(aw.transform_point3(h)), self.project(aw.transform_point3(t))) else { continue };
            let ab = b - a;
            let tt = ((mouse - a).dot(ab) / ab.length_sq().max(1e-6)).clamp(0.0, 1.0);
            let d = (a + ab * tt - mouse).length();
            if d < 14.0 && best.is_none_or(|(bd, _)| d < bd) {
                best = Some((d, i));
            }
        }
        best.map(|b| b.1)
    }

    /// Pose mode input. Returns true when consumed.
    pub(super) fn pose_input(&mut self, ctx: &egui::Context, resp: &egui::Response, ed: &mut Editor, mouse: Pos2) -> bool {
        let Some(id) = ed.pose else {
            self.pose_ui.modal = None;
            return false;
        };
        let Some(o) = ed.scene.get(id) else { return false };
        let Some(arm) = o.armature.as_ref() else { return false };
        let nb = arm.bones.len();
        if nb == 0 {
            return false;
        }
        ed.active_bone = ed.active_bone.min(nb - 1);
        let aw = o.transform.matrix();
        let typing = ctx.egui_wants_keyboard_input();
        use egui::Modifiers as M;
        let c = |m: M, k: Key| ctx.input_mut(|i| i.consume_key(m, k));

        // ---- modal rotate / grab ----
        if let Some(m) = &self.pose_ui.modal {
            let (bone, orig) = (m.bone, m.orig);
            let cancel = ctx.input(|i| i.key_pressed(Key::Escape) || i.pointer.secondary_pressed());
            if cancel {
                if let Some(b) = ed.scene.get_mut(id).and_then(|o| o.armature.as_mut()).and_then(|a| a.bones.get_mut(bone)) {
                    b.pose = orig;
                }
                self.pose_ui.modal = None;
                ed.dirty = true;
                return true;
            }
            let pre = arm.pre_pose(bone);
            let mut pose = orig;
            if m.rotate {
                let a0 = m.start - m.pivot;
                let a1 = mouse - m.pivot;
                // screen y runs down: flip for a counter-clockwise-positive angle
                let ang = (-a1.y).atan2(a1.x) - (-a0.y).atan2(a0.x);
                let toward_viewer = -self.cam.forward();
                let qa_world = Quat::from_axis_angle(toward_viewer, ang);
                let arot = aw.to_scale_rotation_translation().1;
                let qa_arm = arot.inverse() * qa_world * arot;
                let b = pre.to_scale_rotation_translation().1;
                pose.rot = (b.inverse() * qa_arm * b * orig.rot).normalize();
                self.status_hint = format!("Rotate bone {:.0}° · LMB/Enter confirm · RMB/Esc cancel", ang.to_degrees());
            } else {
                let wpp = self.world_per_pixel(m.head_world);
                let d = mouse - m.start;
                let world = (self.cam.right() * d.x - self.cam.up() * d.y) * wpp;
                let local = (aw * pre).inverse().transform_vector3(world);
                pose.loc = orig.loc + local;
                self.status_hint = "Move bone · LMB/Enter confirm · RMB/Esc cancel".into();
            }
            if let Some(b) = ed.scene.get_mut(id).and_then(|o| o.armature.as_mut()).and_then(|a| a.bones.get_mut(bone)) {
                b.pose = pose;
            }
            ed.dirty = true;
            ctx.request_repaint();
            if ctx.input(|i| i.pointer.primary_pressed() || i.key_pressed(Key::Enter)) {
                self.pose_ui.modal = None;
            }
            return true;
        }

        let bone = ed.active_bone;
        let bname = arm.bones[bone].name.clone();
        self.status_hint = format!("Pose Mode · bone \"{bname}\" · click a bone to select · R rotate · G move · Alt+R / Alt+G clear · I keyframe · Ctrl+Tab exit");
        if !typing && resp.hovered() {
            let start = |rotate: bool, this: &mut Self, ed: &mut Editor| {
                let o = ed.scene.get(id).unwrap();
                let arm = o.armature.as_ref().unwrap();
                let head = (aw * arm.pose_globals()[bone]).transform_point3(Vec3::ZERO);
                let pivot = this.project(head).unwrap_or(mouse);
                let orig = arm.bones[bone].pose;
                ed.snapshot();
                this.pose_ui.modal = Some(PoseModal { bone, rotate, orig, start: mouse, pivot, head_world: head });
            };
            let (clear_r, clear_g) = (c(M::ALT, Key::R), c(M::ALT, Key::G));
            if clear_r || clear_g {
                ed.snapshot();
                if let Some(b) = ed.scene.get_mut(id).and_then(|o| o.armature.as_mut()).and_then(|a| a.bones.get_mut(bone)) {
                    if clear_r {
                        b.pose.rot = Quat::IDENTITY;
                    }
                    if clear_g {
                        b.pose.loc = Vec3::ZERO;
                    }
                }
                ed.dirty = true;
                return true;
            }
            if c(M::NONE, Key::R) {
                start(true, self, ed);
                return true;
            }
            if c(M::NONE, Key::G) {
                start(false, self, ed);
                return true;
            }
        }
        if resp.clicked_by(PointerButton::Primary) {
            if let Some(i) = self.bone_at(ed, id, mouse) {
                ed.active_bone = i;
            }
            return true;
        }
        // LMB drags shouldn't box-select objects in pose mode
        resp.dragged_by(PointerButton::Primary) || resp.hovered() && ctx.input(|i| i.pointer.primary_down())
    }

    /// Bone picker and pose buttons for the viewport header.
    pub(super) fn pose_header(&mut self, ui: &mut Ui, ed: &mut Editor) {
        let Some(id) = ed.pose else { return };
        let Some(arm) = ed.scene.get(id).and_then(|o| o.armature.as_ref()) else { return };
        let names: Vec<String> = arm.bones.iter().map(|b| b.name.clone()).collect();
        let mut sel = ed.active_bone.min(names.len().saturating_sub(1));
        egui::ComboBox::from_id_salt("bone_pick").selected_text(egui::RichText::new(names.get(sel).cloned().unwrap_or_default()).size(12.0)).width(110.0).show_ui(ui, |ui| {
            for (i, n) in names.iter().enumerate() {
                ui.selectable_value(&mut sel, i, n);
            }
        });
        ed.active_bone = sel;
        if ui.small_button("Reset Pose").on_hover_text("Clear the pose of all bones").clicked() {
            ed.snapshot();
            if let Some(a) = ed.scene.get_mut(id).and_then(|o| o.armature.as_mut()) {
                a.reset_pose();
            }
            ed.dirty = true;
        }
        if let Some(a) = ed.scene.get_mut(id).and_then(|o| o.armature.as_mut()) {
            ui.toggle_value(&mut a.rest_position, "Rest").on_hover_text("Show the rest position");
        }
    }
}
