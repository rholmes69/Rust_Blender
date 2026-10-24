//! Armature Edit Mode: select bone heads, tails or bodies; G move, E extrude, X delete.
//! Bound meshes keep their shape (their bind matrices follow the edit).

use super::*;
use crate::armature::{Armature, BonePart, Skin};

#[derive(Default)]
pub struct BoneEditUi {
    modal: Option<BoneModal>,
}

struct BoneModal {
    bone: usize,
    part: BonePart,
    /// armature before the operation (cancel restores it) and the one being moved
    orig: Armature,
    base: Armature,
    skins: Vec<(u64, Skin)>,
    map: Vec<Option<usize>>,
    start: Pos2,
    at: Vec3,
    head: Vec3,
    tail: Vec3,
}

impl Viewport {
    /// Bone part under the mouse: head/tail dots first, then the bone body.
    fn bone_part_at(&self, ed: &Editor, id: u64, mouse: Pos2) -> Option<(usize, BonePart)> {
        let o = ed.scene.get(id)?;
        let arm = o.armature.as_ref()?;
        let aw = o.transform.matrix();
        let segs = arm.segments(&arm.rest_globals());
        let mut best: Option<(f32, usize, BonePart)> = None;
        for (i, (h, t)) in segs.iter().enumerate() {
            for (p, part) in [(*h, BonePart::Head), (*t, BonePart::Tail)] {
                if let Some(s) = self.project(aw.transform_point3(p)) {
                    let d = (s - mouse).length();
                    if d < 9.0 && best.is_none_or(|b| d < b.0) {
                        best = Some((d, i, part));
                    }
                }
            }
        }
        if let Some(b) = best {
            return Some((b.1, b.2));
        }
        self.bone_at_rest(ed, id, mouse).map(|i| (i, BonePart::Body))
    }

    fn bone_at_rest(&self, ed: &Editor, id: u64, mouse: Pos2) -> Option<usize> {
        let o = ed.scene.get(id)?;
        let arm = o.armature.as_ref()?;
        let aw = o.transform.matrix();
        let mut best: Option<(f32, usize)> = None;
        for (i, (h, t)) in arm.segments(&arm.rest_globals()).into_iter().enumerate() {
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

    pub(super) fn bone_edit_input(&mut self, ctx: &egui::Context, resp: &egui::Response, ed: &mut Editor, mouse: Pos2) -> bool {
        let Some(id) = ed.bone_edit else {
            self.bone_ui.modal = None;
            return false;
        };
        let Some(o) = ed.scene.get(id) else { return false };
        let Some(arm) = o.armature.clone() else { return false };
        if arm.bones.is_empty() {
            return false;
        }
        let aw = o.transform.matrix();
        ed.active_bone = ed.active_bone.min(arm.bones.len() - 1);
        let typing = ctx.egui_wants_keyboard_input();
        use egui::Modifiers as M;
        let c = |m: M, k: Key| ctx.input_mut(|i| i.consume_key(m, k));

        if let Some(m) = &self.bone_ui.modal {
            if ctx.input(|i| i.key_pressed(Key::Escape) || i.pointer.secondary_pressed()) {
                let (orig, skins) = (m.orig.clone(), m.skins.clone());
                let ident: Vec<Option<usize>> = (0..orig.bones.len()).map(Some).collect();
                let o2 = orig.clone();
                ed.apply_bone_edit(id, o2, &orig, &skins, &ident);
                self.bone_ui.modal = None;
                return true;
            }
            let wpp = self.world_per_pixel(m.at);
            let d = mouse - m.start;
            let world = (self.cam.right() * d.x - self.cam.up() * d.y) * wpp;
            let dl = aw.inverse().transform_vector3(world);
            let (h, t) = match m.part {
                BonePart::Head => (m.head + dl, m.tail),
                BonePart::Tail => (m.head, m.tail + dl),
                BonePart::Body => (m.head + dl, m.tail + dl),
            };
            let mut a = m.base.clone();
            a.set_rest_points(m.bone, h, t);
            let (orig, skins, map) = (m.orig.clone(), m.skins.clone(), m.map.clone());
            ed.apply_bone_edit(id, a, &orig, &skins, &map);
            self.status_hint = format!("Move · {:.2} m · LMB/Enter confirm · RMB/Esc cancel", dl.length());
            ctx.request_repaint();
            if ctx.input(|i| i.pointer.primary_pressed() || i.key_pressed(Key::Enter)) {
                self.bone_ui.modal = None;
            }
            return true;
        }

        let bone = ed.active_bone;
        self.status_hint = format!(
            "Edit Mode (bones) · \"{}\" {:?} · click a bone, head or tail · G move · E extrude · X delete · Shift+A add bone · Tab exit",
            arm.bones[bone].name, ed.bone_part
        );
        let start_grab = |this: &mut Self, ed: &mut Editor, orig: Armature, base: Armature, map: Vec<Option<usize>>, bone: usize, part: BonePart| {
            let (head, tail) = base.rest_points(bone);
            let at = aw.transform_point3(match part {
                BonePart::Head => head,
                BonePart::Tail => tail,
                BonePart::Body => (head + tail) * 0.5,
            });
            let skins = ed.skins_of(id);
            this.bone_ui.modal = Some(BoneModal { bone, part, orig, base, skins, map, start: mouse, at, head, tail });
        };
        if !typing && resp.hovered() {
            if c(M::NONE, Key::G) {
                ed.snapshot();
                let ident: Vec<Option<usize>> = (0..arm.bones.len()).map(Some).collect();
                start_grab(self, ed, arm.clone(), arm.clone(), ident, bone, ed.bone_part);
                return true;
            }
            if c(M::NONE, Key::E) {
                ed.snapshot();
                let skins = ed.skins_of(id);
                let mut a = arm.clone();
                let nb = a.extrude(bone);
                let mut map: Vec<Option<usize>> = (0..arm.bones.len()).map(Some).collect();
                ed.apply_bone_edit(id, a.clone(), &arm, &skins, &map);
                // the new bone is moved relative to the original armature
                map.truncate(arm.bones.len());
                ed.active_bone = nb;
                ed.bone_part = BonePart::Tail;
                start_grab(self, ed, arm.clone(), a, map, nb, BonePart::Tail);
                if let Some(m) = self.bone_ui.modal.as_mut() {
                    m.skins = skins;
                }
                return true;
            }
            if c(M::SHIFT, Key::A) {
                ed.snapshot();
                let skins = ed.skins_of(id);
                let cur = aw.inverse().transform_point3(self.cursor);
                let mut a = arm.clone();
                let nb = a.add_bone(cur, cur + Vec3::Z * 0.5, None);
                let map: Vec<Option<usize>> = (0..arm.bones.len()).map(Some).collect();
                ed.apply_bone_edit(id, a, &arm, &skins, &map);
                ed.active_bone = nb;
                ed.bone_part = BonePart::Body;
                return true;
            }
            if (c(M::NONE, Key::X) || c(M::NONE, Key::Delete)) && arm.bones.len() > 1 {
                ed.snapshot();
                let skins = ed.skins_of(id);
                let mut a = arm.clone();
                let map = a.delete_bone(bone);
                ed.apply_bone_edit(id, a, &arm, &skins, &map);
                ed.active_bone = arm.bones[bone].parent.and_then(|p| map[p]).unwrap_or(0);
                return true;
            }
        }
        if resp.clicked_by(PointerButton::Primary) {
            if let Some((b, part)) = self.bone_part_at(ed, id, mouse) {
                ed.active_bone = b;
                ed.bone_part = part;
            }
            return true;
        }
        resp.hovered() && ctx.input(|i| i.pointer.primary_down())
    }

    /// Head / tail dots in edit mode.
    pub(super) fn draw_bone_joints(&mut self, o: &crate::scene::Object, view: &Mat4) {
        let Some((_, active, part)) = self.bone_edit_draw.filter(|(id, _, _)| *id == o.id) else { return };
        let Some(arm) = o.armature.as_ref() else { return };
        let aw = o.transform.matrix();
        let (r, u) = (self.cam.right(), self.cam.up());
        for (i, (h, t)) in arm.segments(&arm.rest_globals()).into_iter().enumerate() {
            for (p, pp) in [(h, BonePart::Head), (t, BonePart::Tail)] {
                let w = aw.transform_point3(p);
                let rad = 4.0 * self.world_per_pixel(w);
                let sel = i == active && (part == pp || part == BonePart::Body);
                let col = if sel { [1.0, 0.85, 0.35, 1.0] } else { [0.15, 0.15, 0.18, 1.0] };
                for k in 0..8 {
                    let a0 = k as f32 / 8.0 * std::f32::consts::TAU;
                    let a1 = (k + 1) as f32 / 8.0 * std::f32::consts::TAU;
                    self.push_line(w + (r * a0.cos() + u * a0.sin()) * rad, w + (r * a1.cos() + u * a1.sin()) * rad, 1.6 * self.px_scale, col, 0.0, view, true);
                }
            }
        }
    }
}
