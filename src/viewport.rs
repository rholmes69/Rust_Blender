//! 3D viewport: camera, software rasterized (depth-sorted) renderer on top of egui meshes,
//! picking, Blender-style navigation, modal G/R/S transforms and transform gizmos.

use crate::editor::{Editor, Shading, ViewPreset, ViewRequest};
use crate::render::{Outline, RPrim, Raster};
use crate::scene::{MeshCache, Transform};
use crate::theme;
use egui::{
    Align2, Color32, FontId, Key, PointerButton, Pos2, Rect, Sense, Stroke, Ui, Vec2, pos2, vec2,
};
use glam::{Mat3, Mat4, Quat, Vec3, Vec4, Vec4Swizzles};
use std::time::Instant;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Camera {
    pub target: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub dist: f32,
    pub ortho: bool,
    pub fov_deg: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            target: Vec3::ZERO,
            yaw: 38.0,
            pitch: 24.0,
            dist: 17.0,
            ortho: false,
            fov_deg: 40.0,
        }
    }
}

impl Camera {
    pub fn offset_dir(&self) -> Vec3 {
        let (y, p) = (self.yaw.to_radians(), self.pitch.to_radians());
        Vec3::new(y.sin() * p.cos(), -y.cos() * p.cos(), p.sin())
    }
    pub fn eye(&self) -> Vec3 {
        self.target + self.offset_dir() * self.dist
    }
    pub fn forward(&self) -> Vec3 {
        -self.offset_dir()
    }
    pub fn right(&self) -> Vec3 {
        let y = self.yaw.to_radians();
        Vec3::new(y.cos(), y.sin(), 0.0)
    }
    pub fn up(&self) -> Vec3 {
        self.right().cross(self.forward()).normalize()
    }
    pub fn view(&self) -> Mat4 {
        glam::camera::rh::view::look_to_mat4(self.eye(), self.forward(), self.up())
    }
    pub fn half_height(&self) -> f32 {
        self.dist * (self.fov_deg.to_radians() * 0.5).tan()
    }
    pub fn proj(&self, aspect: f32) -> Mat4 {
        if self.ortho {
            let h = self.half_height();
            let w = h * aspect;
            glam::camera::rh::proj::directx::orthographic(-w, w, -h, h, -2000.0, 2000.0)
        } else {
            glam::camera::rh::proj::directx::perspective(
                self.fov_deg.to_radians(),
                aspect,
                0.05,
                2000.0,
            )
        }
    }
    pub fn preset(&mut self, p: ViewPreset) {
        let (y, pt) = match p {
            ViewPreset::Front => (0.0, 0.0),
            ViewPreset::Back => (180.0, 0.0),
            ViewPreset::Right => (90.0, 0.0),
            ViewPreset::Left => (-90.0, 0.0),
            ViewPreset::Top => (0.0, 90.0),
            ViewPreset::Bottom => (0.0, -90.0),
            ViewPreset::Iso => (38.0, 24.0),
        };
        self.yaw = y;
        self.pitch = pt;
        if p != ViewPreset::Iso {
            self.ortho = true;
        }
    }
    pub fn view_name(&self) -> String {
        let snapped =
            |a: f32, b: f32| (a - b).rem_euclid(360.0).min((b - a).rem_euclid(360.0)) < 0.01;
        let name = if snapped(self.pitch, 90.0) {
            Some("Top")
        } else if snapped(self.pitch, -90.0) {
            Some("Bottom")
        } else if snapped(self.pitch, 0.0) {
            if snapped(self.yaw, 0.0) {
                Some("Front")
            } else if snapped(self.yaw, 180.0) {
                Some("Back")
            } else if snapped(self.yaw, 90.0) {
                Some("Right")
            } else if snapped(self.yaw, 270.0) {
                Some("Left")
            } else {
                None
            }
        } else {
            None
        };
        let proj = if self.ortho {
            "Orthographic"
        } else {
            "Perspective"
        };
        match name {
            Some(n) => format!("{n} {proj}"),
            None => format!("User {proj}"),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tool {
    Select,
    Move,
    Rotate,
    Scale,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModalKind {
    Grab,
    Rotate,
    Scale,
    /// Edit mode: interactive inset of the selected faces.
    Inset,
}

#[derive(Clone, Debug)]
pub struct Modal {
    pub kind: ModalKind,
    pub axis: Option<usize>,
    /// Arbitrary constraint axis (e.g. a face normal when extruding).
    pub axis_vec: Option<Vec3>,
    pub start_mouse: Pos2,
    pub originals: Vec<(u64, Transform)>,
    /// Edit mode: original local positions of the transformed vertices.
    pub verts: Vec<(u32, Vec3)>,
    /// Edit mode: mesh + selection before a topology tool (inset).
    pub edit_base: Option<(crate::mesh::PolyMesh, Vec<bool>)>,
    pub pivot: Vec3,
    pub typed: String,
    /// Started from a gizmo drag: releasing the mouse confirms.
    pub from_gizmo: bool,
    /// Whether this modal pushed its own undo snapshot (popped on cancel).
    pub own_snapshot: bool,
}

pub struct Overlays {
    pub grid: bool,
    pub wire_overlay: bool,
    pub stats: bool,
    pub xray: bool,
}

impl Default for Overlays {
    fn default() -> Self {
        Self {
            grid: true,
            wire_overlay: false,
            stats: true,
            xray: false,
        }
    }
}

#[derive(Clone, PartialEq)]
struct RenderKey {
    objects: Vec<crate::scene::Object>,
    world: crate::scene::World,
    selection: Vec<u64>,
    active: Option<u64>,
    shading: Shading,
    cam: Camera,
    rect: Rect,
    px: (usize, usize),
    overlays: (bool, bool, bool),
    edit: Option<crate::editmesh::EditState>,
    loopcut: Option<(usize, Option<(u32, u32)>)>,
    gpu: bool,
}

pub struct Viewport {
    last_key: Option<RenderKey>,
    /// GPU renderer (wgpu) when available; falls back to the software rasterizer.
    pub gpu: Option<crate::gpu::GpuHandle>,
    pub use_gpu: bool,
    frame_version: u64,
    gpu_frame: Option<std::sync::Arc<crate::gpu::FrameData>>,
    /// Transient error message shown in the HUD.
    pub flash: Option<(String, Instant)>,
    inset_amount: f32,
    /// Object currently drawn in edit mode + world-space data for component picking.
    edit_obj: Option<u64>,
    edit_draw: Option<edit::EditDraw>,
    /// Active loop-cut tool (number of cuts) and hovered edge.
    pub loopcut: Option<usize>,
    loopcut_scroll: f32,
    loopcut_edge: Option<(u32, u32)>,
    line_id: u32,
    /// Looking through this camera object (Numpad 0).
    pub camera_view: Option<u64>,
    pub cam: Camera,
    target_cam: Option<Camera>,
    pub tool: Tool,
    pub modal: Option<Modal>,
    pub cursor: Vec3,
    pub overlays: Overlays,
    pub mesh_cache: MeshCache,
    prims: Vec<RPrim>,
    raster: Raster,
    id_map: Vec<u64>,
    texture: Option<egui::TextureHandle>,
    proj: Mat4,
    px_scale: f32,
    pub render_ms: f32,
    box_select: Option<(Pos2, Pos2)>,
    vp: Mat4,
    rect: Rect,
    pub status_hint: String,
    pub stats: (usize, usize, usize),
    pub add_menu_at: Option<Pos2>,
    gizmo_hover: Option<GizmoHandle>,
    last_frame_time: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum GizmoHandle {
    Axis(usize),
    Free,
}

impl Default for Viewport {
    fn default() -> Self {
        Self {
            last_key: None,
            gpu: None,
            use_gpu: true,
            frame_version: 0,
            gpu_frame: None,
            flash: None,
            inset_amount: 0.0,
            edit_obj: None,
            edit_draw: None,
            loopcut: None,
            loopcut_scroll: 0.0,
            loopcut_edge: None,
            line_id: 0,
            camera_view: None,
            cam: Camera::default(),
            target_cam: None,
            tool: Tool::Move,
            modal: None,
            cursor: Vec3::ZERO,
            overlays: Overlays::default(),
            mesh_cache: MeshCache::default(),
            prims: vec![],
            raster: Raster::default(),
            id_map: vec![],
            texture: None,
            proj: Mat4::IDENTITY,
            px_scale: 1.0,
            render_ms: 0.0,
            box_select: None,
            vp: Mat4::IDENTITY,
            rect: Rect::NOTHING,
            status_hint: String::new(),
            stats: (0, 0, 0),
            add_menu_at: None,
            gizmo_hover: None,
            last_frame_time: 0.0,
        }
    }
}

mod edit;

/// Triangle ids at or above this value encode a face index of the mesh in edit mode.
pub const FACE_BASE: u32 = 1 << 24;

const AXIS_COLORS: [Color32; 3] = [
    Color32::from_rgb(255, 54, 83),
    Color32::from_rgb(138, 219, 0),
    Color32::from_rgb(44, 143, 255),
];
const AXES: [Vec3; 3] = [Vec3::X, Vec3::Y, Vec3::Z];

/// A light evaluated for Material Preview shading.
struct SceneLight {
    kind: crate::scene::LightType,
    pos: Vec3,
    dir: Vec3,
    radiance: Vec3,
    cos_outer: f32,
    cos_inner: f32,
}

impl SceneLight {
    fn gather(scene: &crate::scene::Scene) -> Vec<SceneLight> {
        use crate::scene::LightType;
        let mut v: Vec<SceneLight> = scene
            .objects
            .iter()
            .filter(|o| o.visible)
            .filter_map(|o| {
                let l = o.light()?;
                let half = (l.spot_angle * 0.5).to_radians();
                Some(SceneLight {
                    kind: l.light_type,
                    pos: o.transform.location,
                    dir: o.forward(),
                    radiance: lin(l.color) * l.power,
                    cos_outer: half.cos(),
                    cos_inner: (half * (1.0 - l.spot_blend.clamp(0.0, 1.0))).cos(),
                })
            })
            .collect();
        if v.is_empty() {
            // no lights in the scene: fall back to the World sun
            let w = &scene.world;
            v.push(SceneLight {
                kind: LightType::Sun,
                pos: Vec3::ZERO,
                dir: -w.sun_dir.normalize(),
                radiance: Vec3::splat(w.sun_strength * 3.0),
                cos_outer: 0.0,
                cos_inner: 0.0,
            });
        }
        v
    }

    /// (direction to light, incident radiance) at point p.
    fn incident(&self, p: Vec3) -> (Vec3, Vec3) {
        use crate::scene::LightType;
        match self.kind {
            LightType::Sun => (-self.dir, self.radiance / 3.0),
            LightType::Point | LightType::Spot => {
                let d = self.pos - p;
                let d2 = d.length_squared().max(0.01);
                let l = d / d2.sqrt();
                let mut r = self.radiance / (4.0 * std::f32::consts::PI * d2) * 0.9;
                if self.kind == LightType::Spot {
                    let c = (-l).dot(self.dir);
                    let t = ((c - self.cos_outer) / (self.cos_inner - self.cos_outer).max(1e-4))
                        .clamp(0.0, 1.0);
                    r *= t * t * (3.0 - 2.0 * t);
                }
                (l, r)
            }
        }
    }
}

fn srgb4(c: Vec3, a: f32) -> [f32; 4] {
    let t = |x: f32| {
        // gentle filmic-ish shoulder then gamma
        let x = x.max(0.0);
        let x = x / (1.0 + x * 0.15);
        x.powf(1.0 / 2.2).min(1.0)
    };
    [t(c.x), t(c.y), t(c.z), a.clamp(0.0, 1.0)]
}
fn lin(c: [f32; 3]) -> Vec3 {
    Vec3::new(c[0].powf(2.2), c[1].powf(2.2), c[2].powf(2.2))
}

impl Viewport {
    pub fn rect(&self) -> Rect {
        self.rect
    }

    pub fn project(&self, p: Vec3) -> Option<Pos2> {
        let c = self.vp * p.extend(1.0);
        if c.w <= 1e-5 {
            return None;
        }
        let n = c.xyz() / c.w;
        Some(pos2(
            self.rect.center().x + n.x * self.rect.width() * 0.5,
            self.rect.center().y - n.y * self.rect.height() * 0.5,
        ))
    }

    pub fn ray(&self, s: Pos2) -> (Vec3, Vec3) {
        let inv = self.vp.inverse();
        let nx = (s.x - self.rect.center().x) / (self.rect.width() * 0.5);
        let ny = -(s.y - self.rect.center().y) / (self.rect.height() * 0.5);
        let a = inv * Vec4::new(nx, ny, 0.0, 1.0);
        let b = inv * Vec4::new(nx, ny, 0.9, 1.0);
        let (a, b) = (a.xyz() / a.w, b.xyz() / b.w);
        (a, (b - a).normalize())
    }

    fn ground_point(&self, s: Pos2) -> Option<Vec3> {
        let (o, d) = self.ray(s);
        if d.z.abs() < 1e-5 {
            return None;
        }
        let t = -o.z / d.z;
        (t > 0.0).then(|| o + d * t)
    }

    fn world_per_pixel(&self, at: Vec3) -> f32 {
        let h = self.rect.height().max(1.0);
        if self.cam.ortho {
            2.0 * self.cam.half_height() / h
        } else {
            let depth = (at - self.cam.eye()).dot(self.cam.forward()).max(0.05);
            2.0 * depth * (self.cam.fov_deg.to_radians() * 0.5).tan() / h
        }
    }

    fn handle_view_requests(&mut self, ed: &mut Editor) {
        for r in std::mem::take(&mut ed.view_requests) {
            if !matches!(r, ViewRequest::Camera | ViewRequest::AlignCameraToView)
                && self.camera_view.is_some()
            {
                self.leave_camera_view();
            }
            let mut c = self.target_cam.unwrap_or(self.cam);
            match r {
                ViewRequest::Preset(p) => c.preset(p),
                ViewRequest::ToggleOrtho => c.ortho = !c.ortho,
                ViewRequest::FrameSelected => {
                    if let Some((ctr, rad)) = self.bounds_of(ed, true) {
                        c.target = ctr;
                        c.dist = (rad * 2.6).max(2.0);
                    }
                }
                ViewRequest::FrameAll => {
                    if let Some((ctr, rad)) = self.bounds_of(ed, false) {
                        c.target = ctr;
                        c.dist = (rad * 2.4).max(3.0);
                    }
                }
                ViewRequest::Camera => {
                    if self.camera_view.is_some() {
                        self.camera_view = None;
                        c.fov_deg = Camera::default().fov_deg;
                    } else if let Some(cam) = ed.scene.active_camera() {
                        self.camera_view = Some(cam.id);
                        c = self.camera_from_object(cam);
                    }
                }
                ViewRequest::AlignCameraToView => {
                    let eye = self.cam.eye();
                    let tgt = self.cam.target;
                    let id = ed.scene.active_camera().map(|o| o.id);
                    if let Some(id) = id {
                        ed.snapshot();
                        if let Some(o) = ed.scene.get_mut(id) {
                            o.transform.location = eye;
                            o.look_at(tgt);
                        }
                        self.camera_view = Some(id);
                        if let Some(o) = ed.scene.get(id) {
                            c = self.camera_from_object(o);
                        }
                    }
                }
                ViewRequest::Zoom(f) => c.dist = (c.dist * f).clamp(0.2, 900.0),
                ViewRequest::Orbit(dy, dp) => {
                    c.yaw += dy;
                    c.pitch = (c.pitch + dp).clamp(-90.0, 90.0);
                }
            }
            self.target_cam = Some(c);
        }
    }

    fn leave_camera_view(&mut self) {
        if self.camera_view.take().is_some() {
            self.cam.fov_deg = Camera::default().fov_deg;
            if let Some(t) = &mut self.target_cam {
                t.fov_deg = Camera::default().fov_deg;
            }
        }
    }

    /// Orbit-camera parameters that reproduce the view through a camera object.
    fn camera_from_object(&self, o: &crate::scene::Object) -> Camera {
        let f = o.forward();
        let off = -f;
        let pitch = off.z.clamp(-1.0, 1.0).asin().to_degrees();
        let yaw = off.x.atan2(-off.y).to_degrees();
        let fov = match o.kind {
            crate::scene::ObjectKind::Camera(c) => c.fov_deg,
            _ => 40.0,
        };
        // widen the viewport FOV a little so the 16:9 camera frame fits with a margin
        let aspect = self.rect.width().max(1.0) / self.rect.height().max(1.0);
        let s_h = (0.86f32).min(0.86 * aspect / (16.0 / 9.0));
        let vfov = 2.0 * ((fov.to_radians() * 0.5).tan() / s_h).atan();
        let dist = 10.0;
        Camera {
            target: o.transform.location + f * dist,
            yaw,
            pitch,
            dist,
            ortho: false,
            fov_deg: vfov.to_degrees(),
        }
    }

    /// The camera frame rectangle on screen while looking through a camera.
    fn camera_frame(&self) -> Option<Rect> {
        self.camera_view?;
        let aspect = self.rect.width().max(1.0) / self.rect.height().max(1.0);
        let s_h = (0.86f32).min(0.86 * aspect / (16.0 / 9.0));
        let h = self.rect.height() * s_h;
        Some(Rect::from_center_size(
            self.rect.center(),
            vec2(h * 16.0 / 9.0, h),
        ))
    }

    fn bounds_of(&mut self, ed: &Editor, selected_only: bool) -> Option<(Vec3, f32)> {
        if selected_only {
            if let Some((pm, st, m)) = ed.edit_mesh() {
                let pts: Vec<Vec3> = pm
                    .verts
                    .iter()
                    .zip(&st.sel)
                    .filter(|(_, s)| **s)
                    .map(|(p, _)| m.transform_point3(*p))
                    .collect();
                if !pts.is_empty() {
                    let lo = pts.iter().fold(Vec3::splat(f32::MAX), |a, b| a.min(*b));
                    let hi = pts.iter().fold(Vec3::splat(f32::MIN), |a, b| a.max(*b));
                    return Some(((lo + hi) * 0.5, ((hi - lo).length() * 0.5).max(0.3)));
                }
            }
        }
        let mut lo = Vec3::splat(f32::MAX);
        let mut hi = Vec3::splat(f32::MIN);
        let mut any = false;
        for o in &ed.scene.objects {
            if !o.visible || (selected_only && !ed.selection.contains(&o.id)) {
                continue;
            }
            if !o.is_mesh() {
                lo = lo.min(o.transform.location - Vec3::splat(0.5));
                hi = hi.max(o.transform.location + Vec3::splat(0.5));
                any = true;
                continue;
            }
            let (blo, bhi) = self.mesh_cache.for_object(o).bounds();
            for inst in o.instance_matrices() {
                let m = o.transform.matrix() * inst;
                for i in 0..8 {
                    let p = Vec3::new(
                        if i & 1 == 0 { blo.x } else { bhi.x },
                        if i & 2 == 0 { blo.y } else { bhi.y },
                        if i & 4 == 0 { blo.z } else { bhi.z },
                    );
                    let w = m.transform_point3(p);
                    lo = lo.min(w);
                    hi = hi.max(w);
                    any = true;
                }
            }
        }
        any.then(|| ((lo + hi) * 0.5, ((hi - lo).length() * 0.5).max(0.5)))
    }

    fn animate_camera(&mut self, ui: &Ui) {
        if let Some(t) = self.target_cam {
            let now = ui.input(|i| i.time);
            let dt = (now - self.last_frame_time).clamp(0.0, 0.1) as f32;
            let k = 1.0 - (-dt * 14.0).exp();
            let k = if dt == 0.0 { 0.2 } else { k };
            let mut c = self.cam;
            // shortest yaw path
            let mut dy = (t.yaw - c.yaw).rem_euclid(360.0);
            if dy > 180.0 {
                dy -= 360.0;
            }
            c.yaw += dy * k;
            c.pitch += (t.pitch - c.pitch) * k;
            c.dist += (t.dist - c.dist) * k;
            c.target = c.target.lerp(t.target, k);
            c.fov_deg += (t.fov_deg - c.fov_deg) * k;
            c.ortho = t.ortho;
            let done = dy.abs() < 0.05
                && (t.pitch - c.pitch).abs() < 0.05
                && (t.dist - c.dist).abs() < 0.002
                && (t.target - c.target).length() < 0.0005;
            if done {
                self.cam = Camera {
                    yaw: t.yaw.rem_euclid(360.0),
                    ..t
                };
                self.target_cam = None;
            } else {
                self.cam = c;
                ui.ctx().request_repaint();
            }
        }
        self.last_frame_time = ui.input(|i| i.time);
    }

    // ------------------------------------------------------------------------------------------
    // Main entry
    // ------------------------------------------------------------------------------------------
    pub fn ui(&mut self, ui: &mut Ui, ed: &mut Editor) {
        let (rect, resp) = ui.allocate_exact_size(ui.available_size(), Sense::click_and_drag());
        self.rect = rect;
        self.handle_view_requests(ed);
        self.animate_camera(ui);
        if let Some(cid) = self.camera_view {
            match ed.scene.get(cid) {
                Some(o) if self.target_cam.is_none() => self.cam = self.camera_from_object(o),
                None => self.camera_view = None,
                _ => {}
            }
        }
        let aspect = rect.width().max(1.0) / rect.height().max(1.0);
        self.vp = self.cam.proj(aspect) * self.cam.view();

        self.sync_gpu_readback(ui);
        self.handle_input(ui, &resp, ed);
        self.vp = self.cam.proj(aspect) * self.cam.view();

        let painter = ui.painter_at(rect);
        // render target size in physical pixels (capped for performance)
        let ppp = ui.ctx().pixels_per_point();
        let max_px = if self.gpu_active() {
            6_000_000.0f32
        } else {
            2_600_000.0f32
        };
        let area = rect.width() * rect.height();
        self.px_scale = ppp.min((max_px / area.max(1.0)).sqrt()).max(0.5);
        let (w, h) = (
            (rect.width() * self.px_scale).round().max(1.0) as usize,
            (rect.height() * self.px_scale).round().max(1.0) as usize,
        );
        self.raster.resize(w, h);
        self.proj = self.cam.proj(aspect);

        let key = RenderKey {
            objects: ed.scene.objects.clone(),
            world: ed.scene.world.clone(),
            selection: ed.selection.clone(),
            active: ed.active,
            shading: ed.shading,
            cam: self.cam,
            rect,
            px: (w, h),
            overlays: (
                self.overlays.grid,
                self.overlays.wire_overlay,
                self.overlays.xray,
            ),
            edit: ed.edit.clone(),
            loopcut: self.loopcut.map(|n| (n, self.loopcut_edge)),
            gpu: self.gpu_active(),
        };
        let have_image = if self.gpu_active() {
            self.gpu_frame
                .as_ref()
                .is_some_and(|f| f.w as usize == w && f.h as usize == h)
        } else {
            self.texture.is_some()
        };
        if !have_image || self.last_key.as_ref() != Some(&key) {
            self.build_scene(ed);
            self.flush(ui, &painter, ed);
            self.last_key = Some(key);
        } else if self.gpu_active() {
            self.paint_gpu(&painter);
        } else if let Some(t) = &self.texture {
            painter.image(
                t.id(),
                rect,
                Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)),
                Color32::WHITE,
            );
        }

        if let Some(fr) = self.camera_frame() {
            let dark = Color32::from_black_alpha(110);
            let r = self.rect;
            painter.rect_filled(
                Rect::from_min_max(r.min, pos2(r.max.x, fr.min.y)),
                0.0,
                dark,
            );
            painter.rect_filled(
                Rect::from_min_max(pos2(r.min.x, fr.max.y), r.max),
                0.0,
                dark,
            );
            painter.rect_filled(
                Rect::from_min_max(pos2(r.min.x, fr.min.y), pos2(fr.min.x, fr.max.y)),
                0.0,
                dark,
            );
            painter.rect_filled(
                Rect::from_min_max(pos2(fr.max.x, fr.min.y), pos2(r.max.x, fr.max.y)),
                0.0,
                dark,
            );
            painter.rect_stroke(
                fr,
                0.0,
                Stroke::new(1.0, Color32::from_gray(10)),
                egui::StrokeKind::Outside,
            );
        }
        self.draw_cursor(&painter);
        self.draw_gizmo(&painter, ed);
        self.draw_loopcut_preview(&painter, ed);
        self.draw_modal_overlay(&painter, ed);
        if let Some((a, b)) = self.box_select {
            let r = Rect::from_two_pos(a, b);
            painter.rect_filled(r, 0.0, Color32::from_rgba_unmultiplied(255, 255, 255, 12));
            painter.rect_stroke(
                r,
                0.0,
                Stroke::new(1.0, Color32::from_gray(220)),
                egui::StrokeKind::Inside,
            );
        }
        self.draw_hud(ui, &painter, ed);
        self.nav_gizmo(ui, ed);
        self.header_overlay(ui, ed);
    }

    // ------------------------------------------------------------------------------------------
    // Rendering
    // ------------------------------------------------------------------------------------------
    fn to_px(&self, s: Pos2) -> (f32, f32) {
        (
            (s.x - self.rect.min.x) * self.px_scale,
            (s.y - self.rect.min.y) * self.px_scale,
        )
    }

    /// Project a view-space point to (screen point, ndc depth).
    fn vproj(&self, proj: &Mat4, v: Vec3) -> (Pos2, f32) {
        let c = *proj * v.extend(1.0);
        let n = c.xyz() / c.w;
        (
            pos2(
                self.rect.center().x + n.x * self.rect.width() * 0.5,
                self.rect.center().y - n.y * self.rect.height() * 0.5,
            ),
            n.z,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn push_line(
        &mut self,
        a: Vec3,
        b: Vec3,
        w: f32,
        col: [f32; 4],
        bias: f32,
        view: &Mat4,
        no_depth: bool,
    ) {
        let (mut va, mut vb) = (view.transform_point3(a), view.transform_point3(b));
        let near = -0.06;
        if !self.cam.ortho {
            if va.z > near && vb.z > near {
                return;
            }
            if va.z > near {
                let t = (near - vb.z) / (va.z - vb.z);
                va = vb + (va - vb) * t;
            } else if vb.z > near {
                let t = (near - va.z) / (vb.z - va.z);
                vb = va + (vb - va) * t;
            }
        }
        // depth bias: pull towards camera
        let pull = |v: Vec3| {
            if self.cam.ortho {
                v + Vec3::Z * bias * 10.0
            } else {
                v * (1.0 - bias)
            }
        };
        let proj = self.proj;
        let (sa, za) = self.vproj(&proj, pull(va));
        let (sb, zb) = self.vproj(&proj, pull(vb));
        let d = sb - sa;
        if d.length_sq() < 1e-6 {
            return;
        }
        let n = vec2(-d.y, d.x).normalized() * (w * 0.5);
        let pts = [sa + n, sb + n, sb - n, sa - n];
        let zs = [za, zb, zb, za];
        let mut p = [[0.0f32; 3]; 4];
        for i in 0..4 {
            let (x, y) = self.to_px(pts[i]);
            p[i] = [x, y, zs[i].clamp(0.0, 1.0)];
        }
        self.prims.push(RPrim {
            p,
            c: [col; 4],
            n: 4,
            id: self.line_id,
            blend: true,
            no_depth,
        });
    }

    /// Wire gizmos for lights and cameras (pickable via the ID buffer).
    fn draw_object_gizmo(
        &mut self,
        o: &crate::scene::Object,
        pid: u32,
        selected: bool,
        active: bool,
        view: &Mat4,
    ) {
        use crate::scene::{LightType, ObjectKind};
        let lw = self.px_scale;
        let col = if active {
            let c = theme::ACTIVE_OUTLINE;
            [
                c.r() as f32 / 255.0,
                c.g() as f32 / 255.0,
                c.b() as f32 / 255.0,
                1.0,
            ]
        } else if selected {
            let c = theme::SELECT_OUTLINE;
            [
                c.r() as f32 / 255.0,
                c.g() as f32 / 255.0,
                c.b() as f32 / 255.0,
                1.0,
            ]
        } else {
            [0.05, 0.05, 0.05, 0.95]
        };
        let p = o.transform.location;
        let q = o.transform.quat();
        let fwd = o.forward();
        let (r, u) = (self.cam.right(), self.cam.up());
        self.line_id = pid;
        let circle = |this: &mut Self, c: Vec3, ax: Vec3, ay: Vec3, rad: f32, w: f32| {
            let n = 24;
            for i in 0..n {
                let a0 = i as f32 / n as f32 * std::f32::consts::TAU;
                let a1 = (i + 1) as f32 / n as f32 * std::f32::consts::TAU;
                this.push_line(
                    c + (ax * a0.cos() + ay * a0.sin()) * rad,
                    c + (ax * a1.cos() + ay * a1.sin()) * rad,
                    w,
                    col,
                    0.0,
                    view,
                    false,
                );
            }
        };
        match &o.kind {
            ObjectKind::Light(l) => {
                let wpp = self.world_per_pixel(p);
                let rad = 9.0 * wpp;
                circle(self, p, r, u, rad, 1.6 * lw);
                circle(self, p, r, u, rad * 1.6, 1.0 * lw);
                // filled core, light-coloured, for picking
                let core = [
                    l.color[0].min(1.0),
                    l.color[1].min(1.0),
                    l.color[2].min(1.0),
                    1.0,
                ];
                let vp = view.transform_point3(p);
                let (vr, vu) = (
                    view.transform_vector3(r) * rad * 0.8,
                    view.transform_vector3(u) * rad * 0.8,
                );
                for i in 0..12 {
                    let a0 = i as f32 / 12.0 * std::f32::consts::TAU;
                    let a1 = (i + 1) as f32 / 12.0 * std::f32::consts::TAU;
                    self.push_tri(
                        [
                            vp,
                            vp + vr * a0.cos() + vu * a0.sin(),
                            vp + vr * a1.cos() + vu * a1.sin(),
                        ],
                        [core; 3],
                        pid,
                        false,
                    );
                }
                match l.light_type {
                    LightType::Point => {
                        // dashed drop line to the floor
                        let g = Vec3::new(p.x, p.y, 0.0);
                        let n = ((p.z.abs() / 0.25) as i32).clamp(1, 200);
                        for i in (0..n).step_by(2) {
                            let a = p.lerp(g, i as f32 / n as f32);
                            let b = p.lerp(g, (i + 1) as f32 / n as f32);
                            self.push_line(
                                a,
                                b,
                                lw,
                                [col[0], col[1], col[2], 0.5],
                                0.0,
                                view,
                                false,
                            );
                        }
                    }
                    LightType::Sun => {
                        self.push_line(p, p + fwd * 3.0, 1.4 * lw, col, 0.0, view, false);
                        for i in 0..8 {
                            let a = i as f32 / 8.0 * std::f32::consts::TAU;
                            let d = r * a.cos() + u * a.sin();
                            self.push_line(
                                p + d * rad * 2.0,
                                p + d * rad * 3.0,
                                lw,
                                col,
                                0.0,
                                view,
                                false,
                            );
                        }
                    }
                    LightType::Spot => {
                        let len = 2.5;
                        let rr = (l.spot_angle.to_radians() * 0.5).tan() * len;
                        let (ax, ay) = (q * Vec3::X, q * Vec3::Y);
                        let c = p + fwd * len;
                        circle(self, c, ax, ay, rr, lw);
                        for i in 0..4 {
                            let a = i as f32 / 4.0 * std::f32::consts::TAU;
                            self.push_line(
                                p,
                                c + (ax * a.cos() + ay * a.sin()) * rr,
                                lw,
                                col,
                                0.0,
                                view,
                                false,
                            );
                        }
                        self.push_line(p, c, lw, [col[0], col[1], col[2], 0.5], 0.0, view, false);
                    }
                }
            }
            ObjectKind::Camera(cd) => {
                let s = o.transform.scale.x.abs().max(0.01);
                let d = 1.0 * s;
                let h = (cd.fov_deg.to_radians() * 0.5).tan() * d;
                let w = h * 16.0 / 9.0;
                let (ax, ay) = (q * Vec3::X, q * Vec3::Y);
                let c = p + fwd * d;
                let corners = [
                    c + ax * w + ay * h,
                    c - ax * w + ay * h,
                    c - ax * w - ay * h,
                    c + ax * w - ay * h,
                ];
                for i in 0..4 {
                    self.push_line(p, corners[i], 1.3 * lw, col, 0.0, view, false);
                    self.push_line(
                        corners[i],
                        corners[(i + 1) % 4],
                        1.3 * lw,
                        col,
                        0.0,
                        view,
                        false,
                    );
                }
                // "up" triangle
                let t0 = c + ay * h * 1.15 + ax * w * 0.4;
                let t1 = c + ay * h * 1.15 - ax * w * 0.4;
                let t2 = c + ay * h * 1.6;
                let fill = if selected {
                    [col[0], col[1], col[2], 1.0]
                } else {
                    [0.12, 0.12, 0.12, 1.0]
                };
                self.push_tri(
                    [
                        view.transform_point3(t0),
                        view.transform_point3(t2),
                        view.transform_point3(t1),
                    ],
                    [fill; 3],
                    pid,
                    false,
                );
                self.push_tri(
                    [
                        view.transform_point3(t0),
                        view.transform_point3(t1),
                        view.transform_point3(t2),
                    ],
                    [fill; 3],
                    pid,
                    false,
                );
                // a faint filled frame so the camera is easy to click
                let vc = corners.map(|k| view.transform_point3(k));
                let faint = [col[0], col[1], col[2], 0.06];
                self.push_tri([vc[0], vc[1], vc[2]], [faint; 3], pid, true);
                self.push_tri([vc[0], vc[2], vc[3]], [faint; 3], pid, true);
            }
            ObjectKind::Mesh => {}
        }
        self.line_id = 0;
    }

    fn build_scene(&mut self, ed: &Editor) {
        self.prims.clear();
        self.id_map.clear();
        self.edit_obj = ed.edit.as_ref().map(|s| s.obj);
        self.edit_draw = None;
        let view = self.cam.view();
        let eye = self.cam.eye();
        let fwd = self.cam.forward();
        let (mut nverts, mut ntris, mut nobj) = (0, 0, 0);
        let lw = self.px_scale; // line width scale

        // ---- grid ----
        if self.overlays.grid {
            let side_view = self.cam.ortho && self.cam.pitch.abs() < 0.01;
            let mut step = 1.0f32;
            if self.cam.dist > 60.0 {
                step = 10.0;
            } else if self.cam.dist < 4.0 {
                step = 0.1;
            }
            let ext_n = 30i32;
            let center = (self.cam.target / step).round() * step;
            let ext = ext_n as f32 * step;
            if !side_view {
                for i in -ext_n..=ext_n {
                    let x = center.x + i as f32 * step;
                    let y = center.y + i as f32 * step;
                    let major = |v: f32| {
                        ((v / (step * 10.0)).round() * step * 10.0 - v).abs() < step * 0.01
                    };
                    for (k, line) in [(0, x), (1, y)] {
                        let is_axis = line.abs() < step * 0.01;
                        let segs = 12;
                        for s in 0..segs {
                            let t0 = -ext + 2.0 * ext * s as f32 / segs as f32;
                            let t1 = -ext + 2.0 * ext * (s + 1) as f32 / segs as f32;
                            let (a, b) = if k == 0 {
                                (
                                    Vec3::new(line, center.y + t0, 0.0),
                                    Vec3::new(line, center.y + t1, 0.0),
                                )
                            } else {
                                (
                                    Vec3::new(center.x + t0, line, 0.0),
                                    Vec3::new(center.x + t1, line, 0.0),
                                )
                            };
                            let mid = (a + b) * 0.5;
                            let fade = (1.0 - ((mid - center).truncate().length() / ext))
                                .clamp(0.0, 1.0)
                                .powf(0.8);
                            let col = if is_axis {
                                let c = if k == 0 {
                                    AXIS_COLORS[1]
                                } else {
                                    AXIS_COLORS[0]
                                };
                                [
                                    c.r() as f32 / 255.0,
                                    c.g() as f32 / 255.0,
                                    c.b() as f32 / 255.0,
                                    0.9 * fade.max(0.25),
                                ]
                            } else {
                                let a = if major(line) { 0.2 } else { 0.1 };
                                [1.0, 1.0, 1.0, a * fade]
                            };
                            if fade > 0.01 {
                                self.push_line(
                                    a,
                                    b,
                                    if is_axis { 1.6 * lw } else { lw },
                                    col,
                                    0.002,
                                    &view,
                                    false,
                                );
                            }
                        }
                    }
                }
            } else {
                let r = self.cam.right();
                for i in -ext_n..=ext_n {
                    let o = i as f32 * step;
                    let a = center + r * o - Vec3::Z * ext;
                    let b = center + r * o + Vec3::Z * ext;
                    self.push_line(a, b, lw, [1.0, 1.0, 1.0, 0.08], 0.0, &view, true);
                    let a = center - r * ext + Vec3::Z * o;
                    let b = center + r * ext + Vec3::Z * o;
                    let col = if (o + center.z).abs() < 1e-4 {
                        [1.0, 0.25, 0.3, 0.8]
                    } else {
                        [1.0, 1.0, 1.0, 0.08]
                    };
                    self.push_line(a, b, lw, col, 0.0, &view, true);
                }
            }
        }

        // ---- objects ----
        let world = &ed.scene.world;
        let studio_key = (self.cam.right() * -0.35 + self.cam.up() * 0.75 - fwd * 0.55).normalize();
        let studio_fill = (self.cam.right() * 0.7 - self.cam.up() * 0.2 - fwd * 0.4).normalize();
        let ortho = self.cam.ortho;
        let shading = ed.shading;
        let scene_lights = SceneLight::gather(&ed.scene);
        for o in ed.scene.objects.iter().filter(|o| o.visible) {
            nobj += 1;
            self.id_map.push(o.id);
            let pid = self.id_map.len() as u32;
            let selected = ed.selection.contains(&o.id);
            let active = ed.active == Some(o.id);
            if !o.is_mesh() {
                if self.camera_view != Some(o.id) {
                    self.draw_object_gizmo(o, pid, selected, active, &view);
                }
                continue;
            }
            let mesh = self.mesh_cache.for_object(o);
            let smooth = mesh.smooth;
            let base = lin(o.material.color);
            let mat = o.material;
            let alpha = if self.overlays.xray { 0.4 } else { mat.alpha };
            let blend = alpha < 0.999;
            let shade = |p: Vec3, n: Vec3| -> Vec3 {
                let v = if ortho { -fwd } else { (eye - p).normalize() };
                match shading {
                    Shading::Solid => {
                        let d1 = n.dot(studio_key).max(0.0);
                        let d2 = n.dot(studio_fill).max(0.0);
                        let hemi = 0.5 + 0.5 * n.z;
                        let h = (studio_key + v).normalize();
                        let spec = n.dot(h).max(0.0).powf(40.0) * 0.25;
                        let rim = (1.0 - n.dot(v).max(0.0)).powf(3.0) * 0.12;
                        base * (0.18 + 0.12 * hemi + d1 * 0.85 + d2 * 0.25)
                            + Vec3::splat(spec + rim)
                            + base * mat.emission * 0.5
                    }
                    _ => {
                        let sky = Vec3::new(0.55, 0.65, 0.85);
                        let ground = Vec3::new(0.25, 0.22, 0.2);
                        let hemi = ground.lerp(sky, 0.5 + 0.5 * n.z) * world.ambient * 1.6;
                        let rough = mat.roughness.clamp(0.04, 1.0);
                        let shin = 2.0 / (rough * rough) + 2.0;
                        let f0 = Vec3::splat(0.04).lerp(base, mat.metallic);
                        let nv = n.dot(v).max(0.0);
                        let fres = f0 + (Vec3::ONE - f0) * (1.0 - nv).powf(5.0);
                        let mut diffuse_light = hemi;
                        let mut spec = Vec3::ZERO;
                        for sl in &scene_lights {
                            let (l, rad) = sl.incident(p);
                            let ndl = n.dot(l).max(0.0);
                            if ndl <= 0.0 || rad == Vec3::ZERO {
                                continue;
                            }
                            let h = (l + v).normalize();
                            diffuse_light += rad * ndl;
                            spec += fres * rad * n.dot(h).max(0.0).powf(shin) * (shin + 8.0) / 25.0
                                * ndl;
                        }
                        let r = (2.0 * n.dot(v) * n - v).normalize();
                        let env = ground.lerp(sky, 0.5 + 0.5 * r.z) * 0.9;
                        let diffuse = base * (1.0 - mat.metallic) * diffuse_light;
                        diffuse + spec + env * fres * (1.0 - rough * 0.7) + base * mat.emission
                    }
                }
            };
            let editing_this = ed.edit.as_ref().filter(|s| s.obj == o.id);
            let face_sel: Vec<bool> = match (editing_this, &o.mesh) {
                (Some(st), Some(pm)) => (0..pm.faces.len())
                    .map(|f| crate::editmesh::face_selected(pm, &st.sel, f))
                    .collect(),
                _ => vec![],
            };
            for (inst_i, inst) in o.instance_matrices().into_iter().enumerate() {
                let m = o.transform.matrix() * inst;
                let flip = m.determinant() < 0.0;
                let nm = Mat3::from_mat4(m).inverse().transpose();
                let wp: Vec<Vec3> = mesh
                    .positions
                    .iter()
                    .map(|p| m.transform_point3(*p))
                    .collect();
                let wn: Vec<Vec3> = mesh
                    .normals
                    .iter()
                    .map(|n| (nm * *n).normalize_or_zero() * if flip { -1.0 } else { 1.0 })
                    .collect();
                let vpos: Vec<Vec3> = wp.iter().map(|p| view.transform_point3(*p)).collect();
                nverts += wp.len();
                ntris += mesh.tris.len();

                let face_n: Vec<Vec3> = mesh
                    .tris
                    .iter()
                    .map(|t| {
                        let [a, b, c] = t.map(|i| wp[i as usize]);
                        let n = (b - a).cross(c - a).normalize_or_zero();
                        if flip { -n } else { n }
                    })
                    .collect();
                let facing: Vec<bool> = mesh
                    .tris
                    .iter()
                    .enumerate()
                    .map(|(fi, t)| {
                        let ctr = (wp[t[0] as usize] + wp[t[1] as usize] + wp[t[2] as usize]) / 3.0;
                        let to_eye = if ortho { -fwd } else { (eye - ctr).normalize() };
                        face_n[fi].dot(to_eye) > 0.0
                    })
                    .collect();

                if shading != Shading::Wireframe {
                    let vcol: Vec<[f32; 4]> = if smooth {
                        (0..wp.len())
                            .map(|i| srgb4(shade(wp[i], wn[i]), alpha))
                            .collect()
                    } else {
                        vec![]
                    };
                    for (fi, t) in mesh.tris.iter().enumerate() {
                        if !blend && !facing[fi] {
                            continue;
                        }
                        let cols: [[f32; 4]; 3] = if smooth && facing[fi] {
                            t.map(|i| vcol[i as usize])
                        } else {
                            let ctr =
                                (wp[t[0] as usize] + wp[t[1] as usize] + wp[t[2] as usize]) / 3.0;
                            let n = if facing[fi] { face_n[fi] } else { -face_n[fi] };
                            let k = if facing[fi] { 1.0 } else { 0.6 };
                            [srgb4(shade(ctr, n) * k, alpha); 3]
                        };
                        let (tid, cols) = if editing_this.is_some() && inst_i == 0 {
                            let f = mesh.tri_face[fi] as usize;
                            let cols = if face_sel.get(f).copied().unwrap_or(false) {
                                cols.map(|c| {
                                    [
                                        c[0] * 0.6 + 0.4 * 1.0,
                                        c[1] * 0.6 + 0.4 * 0.55,
                                        c[2] * 0.6 + 0.4 * 0.15,
                                        c[3],
                                    ]
                                })
                            } else {
                                cols
                            };
                            (FACE_BASE + f as u32, cols)
                        } else {
                            (pid, cols)
                        };
                        self.push_tri(
                            [
                                vpos[t[0] as usize],
                                vpos[t[1] as usize],
                                vpos[t[2] as usize],
                            ],
                            cols,
                            tid,
                            blend,
                        );
                    }
                }
                if editing_this.is_some() {
                    if inst_i == 0 {
                        self.edit_overlay(ed, &mesh, &wp, &view);
                    }
                    continue;
                }

                // edges: wireframe mode / wire overlay
                let sel_col = if active {
                    theme::ACTIVE_OUTLINE
                } else {
                    theme::SELECT_OUTLINE
                };
                let sc = [
                    sel_col.r() as f32 / 255.0,
                    sel_col.g() as f32 / 255.0,
                    sel_col.b() as f32 / 255.0,
                    1.0,
                ];
                for e in &mesh.edges {
                    let (a, b) = (wp[e.a as usize], wp[e.b as usize]);
                    let boundary = e.faces[1] == u32::MAX;
                    match shading {
                        Shading::Wireframe => {
                            let silhouette = boundary
                                || facing[e.faces[0] as usize] != facing[e.faces[1] as usize];
                            if e.feature || silhouette {
                                let wc = if selected {
                                    sc
                                } else {
                                    [0.06, 0.06, 0.06, 0.9]
                                };
                                self.push_line(
                                    a,
                                    b,
                                    if selected { 1.4 * lw } else { lw },
                                    wc,
                                    0.0,
                                    &view,
                                    true,
                                );
                            }
                        }
                        _ => {
                            if self.overlays.wire_overlay && e.feature {
                                let wc = if selected {
                                    [sc[0], sc[1], sc[2], 0.6]
                                } else {
                                    [0.0, 0.0, 0.0, 0.4]
                                };
                                self.push_line(a, b, lw, wc, 0.004, &view, false);
                            }
                        }
                    }
                }
            }
        }
        self.stats = (nobj, nverts, ntris);
    }

    fn push_tri(&mut self, v: [Vec3; 3], c: [[f32; 4]; 3], id: u32, blend: bool) {
        let near = if self.cam.ortho { -1e9 } else { -0.05 };
        let inside = |p: &Vec3| p.z <= near;
        let proj = self.proj;
        let emit = |this: &mut Self, pts: [Vec3; 3], cols: [[f32; 4]; 3]| {
            let mut p = [[0.0f32; 3]; 4];
            for i in 0..3 {
                let (s, z) = this.vproj(&proj, pts[i]);
                let (x, y) = this.to_px(s);
                p[i] = [x, y, z];
            }
            p[3] = p[2];
            // reject fully off-screen
            let (w, h) = (this.raster.w as f32, this.raster.h as f32);
            if p[..3].iter().all(|q| q[0] < 0.0)
                || p[..3].iter().all(|q| q[0] > w)
                || p[..3].iter().all(|q| q[1] < 0.0)
                || p[..3].iter().all(|q| q[1] > h)
            {
                return;
            }
            this.prims.push(RPrim {
                p,
                c: [cols[0], cols[1], cols[2], cols[2]],
                n: 3,
                id,
                blend,
                no_depth: false,
            });
        };
        if v.iter().all(inside) {
            emit(self, v, c);
            return;
        }
        if !v.iter().any(inside) {
            return;
        }
        let mut poly: Vec<(Vec3, [f32; 4])> = vec![];
        for i in 0..3 {
            let (a, ca) = (v[i], c[i]);
            let (b, cb) = (v[(i + 1) % 3], c[(i + 1) % 3]);
            let (ia, ib) = (inside(&a), inside(&b));
            if ia {
                poly.push((a, ca));
            }
            if ia != ib {
                let t = (near - a.z) / (b.z - a.z);
                poly.push((a + (b - a) * t, if t < 0.5 { ca } else { cb }));
            }
        }
        for k in 1..poly.len().saturating_sub(1) {
            emit(
                self,
                [poly[0].0, poly[k].0, poly[k + 1].0],
                [poly[0].1, poly[k].1, poly[k + 1].1],
            );
        }
    }

    fn flush(&mut self, ui: &Ui, painter: &egui::Painter, ed: &Editor) {
        let bg = ed.scene.world.background;
        let top = [
            (bg[0] * 1.12).min(1.0),
            (bg[1] * 1.12).min(1.0),
            (bg[2] * 1.12).min(1.0),
        ];
        let bot = [bg[0] * 0.82, bg[1] * 0.82, bg[2] * 0.82];
        // blended prims: draw far-to-near
        let key = |p: &RPrim| p.p[0][2] + p.p[1][2] + p.p[2][2];
        let (mut opaque, mut blended): (Vec<RPrim>, Vec<RPrim>) =
            self.prims.drain(..).partition(|p| !p.blend);
        blended.sort_by(|a, b| {
            key(b)
                .partial_cmp(&key(a))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        opaque.append(&mut blended);
        self.prims = opaque;
        let mut sel = vec![0u8; self.id_map.len() + 1];
        if ed.shading != Shading::Wireframe {
            for (i, id) in self.id_map.iter().enumerate() {
                if ed.selection.contains(id) {
                    sel[i + 1] = if ed.active == Some(*id) { 2 } else { 1 };
                }
            }
        }
        let width = (1.5 * self.px_scale).round().max(1.0) as i32;
        if self.gpu_active() {
            let t0 = Instant::now();
            self.frame_version += 1;
            let fd = crate::gpu::FrameData::new(
                self.frame_version,
                self.raster.w as u32,
                self.raster.h as u32,
                top,
                bot,
                &self.prims,
                &sel,
                width as u32,
                theme::SELECT_OUTLINE,
                theme::ACTIVE_OUTLINE,
            );
            self.gpu_frame = Some(std::sync::Arc::new(fd));
            self.render_ms = self.render_ms * 0.9 + t0.elapsed().as_secs_f32() * 1000.0 * 0.1;
            self.paint_gpu(painter);
            return;
        }
        let _ = ui;
        let outline = Outline {
            sel,
            selected: theme::SELECT_OUTLINE,
            active: theme::ACTIVE_OUTLINE,
            width,
        };
        let t0 = Instant::now();
        self.raster.render(top, bot, &self.prims, Some(&outline));
        self.render_ms = self.render_ms * 0.9 + t0.elapsed().as_secs_f32() * 1000.0 * 0.1;
        let img = egui::ColorImage {
            size: [self.raster.w, self.raster.h],
            source_size: vec2(self.raster.w as f32, self.raster.h as f32),
            pixels: self.raster.color.clone(),
        };
        let opts = egui::TextureOptions::LINEAR;
        match &mut self.texture {
            Some(t) => t.set(img, opts),
            None => self.texture = Some(ui.ctx().load_texture("viewport", img, opts)),
        }
        if let Some(t) = &self.texture {
            painter.image(
                t.id(),
                self.rect,
                Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)),
                Color32::WHITE,
            );
        }
    }

    pub fn gpu_active(&self) -> bool {
        self.use_gpu && self.gpu.is_some()
    }

    fn paint_gpu(&self, painter: &egui::Painter) {
        if let Some(fd) = &self.gpu_frame {
            painter.add(eframe::egui_wgpu::Callback::new_paint_callback(
                self.rect,
                crate::gpu::ViewportCallback { frame: fd.clone() },
            ));
        }
    }

    /// Pull the latest GPU ID/depth readback into the CPU buffers used for picking.
    fn sync_gpu_readback(&mut self, ui: &Ui) {
        let Some(g) = &self.gpu else { return };
        if !self.use_gpu {
            return;
        }
        g.poll();
        let mut rb = g.readback.lock().unwrap();
        if rb.fresh && rb.w as usize == self.raster.w && rb.h as usize == self.raster.h {
            std::mem::swap(&mut self.raster.id, &mut rb.ids);
            std::mem::swap(&mut self.raster.depth, &mut rb.depth);
        }
        rb.fresh = false;
        if rb.version != self.frame_version {
            ui.ctx()
                .request_repaint_after(std::time::Duration::from_millis(8));
        }
    }

    pub fn pick(&self, s: Pos2) -> Option<u64> {
        let (x, y) = self.to_px(s);
        if x < 0.0 || y < 0.0 {
            return None;
        }
        let id = self.raster.id_at(x as usize, y as usize);
        if id >= FACE_BASE {
            return self.edit_obj;
        }
        if id == 0 {
            None
        } else {
            self.id_map.get(id as usize - 1).copied()
        }
    }

    fn draw_cursor(&self, painter: &egui::Painter) {
        if let Some(p) = self.project(self.cursor) {
            if !self.rect.contains(p) {
                return;
            }
            let r = 9.0;
            for i in 0..8 {
                let a0 = i as f32 / 8.0 * std::f32::consts::TAU;
                let a1 = (i as f32 + 1.0) / 8.0 * std::f32::consts::TAU;
                let col = if i % 2 == 0 {
                    Color32::from_rgb(230, 50, 50)
                } else {
                    Color32::WHITE
                };
                painter.line_segment(
                    [
                        p + vec2(a0.cos(), a0.sin()) * r,
                        p + vec2(a1.cos(), a1.sin()) * r,
                    ],
                    Stroke::new(1.6, col),
                );
            }
            let c = Color32::from_black_alpha(160);
            for d in [
                vec2(1.0, 0.0),
                vec2(-1.0, 0.0),
                vec2(0.0, 1.0),
                vec2(0.0, -1.0),
            ] {
                painter.line_segment([p + d * 4.0, p + d * 14.0], Stroke::new(1.0, c));
            }
        }
    }

    // ------------------------------------------------------------------------------------------
    // Gizmos
    // ------------------------------------------------------------------------------------------
    fn selection_pivot(ed: &Editor) -> Option<Vec3> {
        if let Some((pm, st, m)) = ed.edit_mesh() {
            return crate::editmesh::selection_center(pm, &st.sel).map(|c| m.transform_point3(c));
        }
        let pts: Vec<Vec3> = ed
            .selection
            .iter()
            .filter_map(|id| ed.scene.get(*id))
            .map(|o| o.transform.location)
            .collect();
        if pts.is_empty() {
            None
        } else {
            Some(pts.iter().sum::<Vec3>() / pts.len() as f32)
        }
    }

    fn gizmo_geom(&self, pivot: Vec3) -> Option<(Pos2, [Pos2; 3], f32)> {
        let c = self.project(pivot)?;
        let wpp = self.world_per_pixel(pivot);
        let len = 85.0 * wpp;
        let ends = [0, 1, 2].map(|i| self.project(pivot + AXES[i] * len).unwrap_or(c));
        Some((c, ends, len))
    }

    fn gizmo_hit(&self, ed: &Editor, mouse: Pos2) -> Option<GizmoHandle> {
        if self.tool == Tool::Select || self.modal.is_some() {
            return None;
        }
        let pivot = Self::selection_pivot(ed)?;
        let (c, ends, len) = self.gizmo_geom(pivot)?;
        if self.tool == Tool::Rotate {
            for i in 0..3 {
                let ring = self.ring_points(pivot, i, len * 0.9);
                for w in ring.windows(2) {
                    if seg_dist(mouse, w[0], w[1]) < 7.0 {
                        return Some(GizmoHandle::Axis(i));
                    }
                }
            }
            if (mouse - c).length() < len / self.world_per_pixel(pivot) * 1.05 {
                return Some(GizmoHandle::Free);
            }
            return None;
        }
        if (mouse - c).length() < 11.0 {
            return Some(GizmoHandle::Free);
        }
        let mut best = None;
        for i in 0..3 {
            let d = seg_dist(mouse, c + (ends[i] - c) * 0.2, ends[i]);
            if d < 8.0 && best.is_none_or(|(bd, _)| d < bd) {
                best = Some((d, i));
            }
        }
        best.map(|(_, i)| GizmoHandle::Axis(i))
    }

    fn ring_points(&self, pivot: Vec3, axis: usize, r: f32) -> Vec<Pos2> {
        let (u, v) = match axis {
            0 => (Vec3::Y, Vec3::Z),
            1 => (Vec3::X, Vec3::Z),
            _ => (Vec3::X, Vec3::Y),
        };
        (0..=48)
            .filter_map(|i| {
                let a = i as f32 / 48.0 * std::f32::consts::TAU;
                self.project(pivot + (u * a.cos() + v * a.sin()) * r)
            })
            .collect()
    }

    fn draw_gizmo(&self, painter: &egui::Painter, ed: &Editor) {
        if self.tool == Tool::Select {
            return;
        }
        let Some(pivot) = Self::selection_pivot(ed) else {
            return;
        };
        let Some((c, ends, len)) = self.gizmo_geom(pivot) else {
            return;
        };
        let active_axis = self
            .modal
            .as_ref()
            .filter(|m| m.from_gizmo)
            .and_then(|m| m.axis);
        if self.modal.as_ref().is_some_and(|m| !m.from_gizmo) {
            return;
        }
        let hov = self.gizmo_hover;
        match self.tool {
            Tool::Rotate => {
                // view-aligned outer ring
                painter.circle_stroke(
                    c,
                    len / self.world_per_pixel(pivot) * 1.05,
                    Stroke::new(
                        1.5,
                        Color32::from_white_alpha(if hov == Some(GizmoHandle::Free) {
                            200
                        } else {
                            90
                        }),
                    ),
                );
                for i in 0..3 {
                    let ring = self.ring_points(pivot, i, len * 0.9);
                    let hot = hov == Some(GizmoHandle::Axis(i)) || active_axis == Some(i);
                    let col = if hot { Color32::WHITE } else { AXIS_COLORS[i] };
                    painter.add(egui::Shape::line(
                        ring,
                        Stroke::new(if hot { 3.5 } else { 2.5 }, col),
                    ));
                }
            }
            _ => {
                let mut order = [0usize, 1, 2];
                order.sort_by(|a, b| {
                    (ends[*b] - c)
                        .length()
                        .partial_cmp(&(ends[*a] - c).length())
                        .unwrap()
                });
                for i in order {
                    let hot = hov == Some(GizmoHandle::Axis(i)) || active_axis == Some(i);
                    let col = if hot { Color32::WHITE } else { AXIS_COLORS[i] };
                    let dir = ends[i] - c;
                    if dir.length() < 4.0 {
                        continue;
                    }
                    painter.line_segment(
                        [c + dir * 0.2, ends[i]],
                        Stroke::new(if hot { 3.5 } else { 2.5 }, col),
                    );
                    if self.tool == Tool::Scale {
                        let r = Rect::from_center_size(ends[i], vec2(10.0, 10.0));
                        painter.rect_filled(r, 1.0, col);
                    } else {
                        let d = dir.normalized();
                        let n = vec2(-d.y, d.x);
                        let tip = ends[i] + d * 14.0;
                        painter.add(egui::Shape::convex_polygon(
                            vec![tip, ends[i] + n * 6.0, ends[i] - n * 6.0],
                            col,
                            Stroke::NONE,
                        ));
                    }
                }
                let hot = hov == Some(GizmoHandle::Free);
                painter.circle_stroke(
                    c,
                    9.0,
                    Stroke::new(
                        if hot { 2.5 } else { 1.5 },
                        Color32::from_white_alpha(if hot { 255 } else { 170 }),
                    ),
                );
            }
        }
    }

    fn draw_modal_overlay(&self, painter: &egui::Painter, _ed: &Editor) {
        let Some(m) = &self.modal else { return };
        if let Some(ax) = m.axis {
            let a = m.pivot - AXES[ax] * 1000.0;
            let b = m.pivot + AXES[ax] * 1000.0;
            // project long line by clipping to screen via many samples
            let pts: Vec<Pos2> = (0..=200)
                .filter_map(|i| self.project(a.lerp(b, i as f32 / 200.0)))
                .filter(|p| p.x.abs() < 1e5 && p.y.abs() < 1e5)
                .collect();
            let view = self.cam.view();
            let pts: Vec<Pos2> = pts
                .into_iter()
                .zip((0..=200).map(|i| view.transform_point3(a.lerp(b, i as f32 / 200.0)).z))
                .filter(|(_, z)| self.cam.ortho || *z < -0.05)
                .map(|(p, _)| p)
                .collect();
            if pts.len() > 1 {
                painter.add(egui::Shape::line(pts, Stroke::new(1.2, AXIS_COLORS[ax])));
            }
        }
        if m.kind != ModalKind::Grab {
            if let (Some(c), Some(mouse)) =
                (self.project(m.pivot), painter.ctx().pointer_latest_pos())
            {
                dashed(painter, c, mouse, Color32::from_gray(10));
            }
        }
    }

    fn draw_hud(&self, ui: &Ui, painter: &egui::Painter, ed: &Editor) {
        let r = self.rect;
        let font = FontId::proportional(13.0);
        let tl = r.left_top() + vec2(12.0, 40.0);
        let vname = if self.camera_view.is_some() {
            "Camera Perspective".to_string()
        } else {
            self.cam.view_name()
        };
        painter.text(
            tl,
            Align2::LEFT_TOP,
            vname,
            font.clone(),
            Color32::from_gray(225),
        );
        let active_name = ed
            .active
            .and_then(|a| ed.scene.get(a))
            .map(|o| o.name.clone())
            .unwrap_or_default();
        painter.text(
            tl + vec2(0.0, 18.0),
            Align2::LEFT_TOP,
            format!("({}) Scene Collection | {}", ed.frame, active_name),
            font.clone(),
            Color32::from_gray(200),
        );
        if self.overlays.stats {
            let (o, v, t) = self.stats;
            let sel = ed.selection.len();
            let txt = match ed.edit_mesh() {
                Some((pm, st, _)) => {
                    let fs = crate::editmesh::selected_faces(pm, &st.sel).len();
                    format!(
                        "Vertices {}/{}\nFaces    {fs}/{}\nRender   {:.1} ms {}",
                        st.count(),
                        pm.verts.len(),
                        pm.faces.len(),
                        self.render_ms,
                        if self.gpu_active() { "GPU" } else { "CPU" }
                    )
                }
                None => format!(
                    "Objects  {sel}/{o}\nVertices {v}\nTriangles {t}\nRender   {:.1} ms {}",
                    self.render_ms,
                    if self.gpu_active() { "GPU" } else { "CPU" }
                ),
            };
            painter.text(
                tl + vec2(0.0, 44.0),
                Align2::LEFT_TOP,
                txt,
                FontId::monospace(11.5),
                Color32::from_gray(190),
            );
        }
        if let Some((msg, t0)) = &self.flash {
            let age = t0.elapsed().as_secs_f32();
            if age < 3.0 {
                let a = (3.0 - age).min(1.0);
                let pos = r.center_bottom() - vec2(0.0, 40.0);
                let rr = Rect::from_center_size(pos, vec2(msg.len() as f32 * 7.2 + 30.0, 26.0));
                painter.rect_filled(
                    rr,
                    6.0,
                    Color32::from_rgba_unmultiplied(120, 30, 30, (200.0 * a) as u8),
                );
                painter.text(
                    pos,
                    Align2::CENTER_CENTER,
                    msg,
                    FontId::proportional(13.0),
                    Color32::WHITE.gamma_multiply(a),
                );
                ui.ctx().request_repaint();
            }
        }
        if let Some(m) = &self.modal {
            let ax = m.axis.map(|a| ["X", "Y", "Z"][a]).unwrap_or("free");
            let s = match m.kind {
                ModalKind::Inset => format!("Inset  {:.3}", self.inset_amount),
                ModalKind::Grab => {
                    let d = ed
                        .selection
                        .first()
                        .and_then(|id| ed.scene.get(*id))
                        .zip(m.originals.first())
                        .map(|(o, (_, t0))| o.transform.location - t0.location)
                        .unwrap_or_default();
                    format!("Move  D: {:.3}  {:.3}  {:.3}  ({ax})", d.x, d.y, d.z)
                }
                ModalKind::Rotate => {
                    let a = ed
                        .selection
                        .first()
                        .and_then(|id| ed.scene.get(*id))
                        .zip(m.originals.first())
                        .map(|(o, (_, t0))| (o.transform.rotation - t0.rotation).length())
                        .unwrap_or(0.0);
                    format!("Rotate  {a:.1}°  ({ax})")
                }
                ModalKind::Scale => {
                    let s = ed
                        .selection
                        .first()
                        .and_then(|id| ed.scene.get(*id))
                        .zip(m.originals.first())
                        .map(|(o, (_, t0))| o.transform.scale / t0.scale.max(Vec3::splat(1e-6)))
                        .unwrap_or(Vec3::ONE);
                    format!("Scale  {:.3}  {:.3}  {:.3}  ({ax})", s.x, s.y, s.z)
                }
            };
            let s = if m.typed.is_empty() {
                s
            } else {
                format!("{s}   [{}]", m.typed)
            };
            let galley_pos = r.center_top() + vec2(0.0, 40.0);
            let rr = Rect::from_center_size(
                galley_pos + vec2(0.0, 9.0),
                vec2(s.len() as f32 * 7.5 + 30.0, 26.0),
            );
            painter.rect_filled(rr, 6.0, Color32::from_black_alpha(170));
            painter.text(
                galley_pos + vec2(0.0, 9.0),
                Align2::CENTER_CENTER,
                s,
                FontId::proportional(13.0),
                Color32::WHITE,
            );
        }
        let _ = ui;
    }

    fn header_overlay(&mut self, ui: &mut Ui, ed: &mut Editor) {
        let r = self.rect;
        let bar = Rect::from_min_size(r.left_top() + vec2(8.0, 6.0), vec2(r.width() - 16.0, 26.0));
        let mut child = ui.new_child(
            egui::UiBuilder::new()
                .max_rect(bar)
                .layout(egui::Layout::left_to_right(egui::Align::Center)),
        );
        let ui = &mut child;
        ui.spacing_mut().item_spacing.x = 2.0;
        egui::Frame::new()
            .fill(theme::HEADER_PILL)
            .corner_radius(6.0)
            .inner_margin(egui::Margin::symmetric(4, 2))
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    let editing = ed.editing();
                    let label = if editing {
                        "✎ Edit Mode"
                    } else {
                        "▣ Object Mode"
                    };
                    let b =
                        egui::Button::new(egui::RichText::new(label).size(12.5)).fill(if editing {
                            Color32::from_rgb(120, 70, 20)
                        } else {
                            Color32::TRANSPARENT
                        });
                    if ui
                        .add(b)
                        .on_hover_text("Toggle Object / Edit mode (Tab)")
                        .clicked()
                    {
                        let r = ed.toggle_edit();
                        self.report(r);
                    }
                    if let Some(st) = ed.edit.as_mut() {
                        ui.separator();
                        for (m, icon) in [
                            (crate::editmesh::SelectMode::Vertex, "⬝"),
                            (crate::editmesh::SelectMode::Edge, "╱"),
                            (crate::editmesh::SelectMode::Face, "▰"),
                        ] {
                            if ui
                                .add(egui::Button::selectable(
                                    st.mode == m,
                                    egui::RichText::new(icon).size(15.0),
                                ))
                                .on_hover_text(format!(
                                    "{} select ({})",
                                    m.label(),
                                    match m {
                                        crate::editmesh::SelectMode::Vertex => 1,
                                        crate::editmesh::SelectMode::Edge => 2,
                                        _ => 3,
                                    }
                                ))
                                .clicked()
                            {
                                st.mode = m;
                            }
                        }
                    }
                });
            });
        egui::Frame::new()
            .fill(theme::HEADER_PILL)
            .corner_radius(6.0)
            .inner_margin(egui::Margin::symmetric(4, 2))
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    for (t, icon, tip) in [
                        (Tool::Select, "⬚", "Select Box (W)"),
                        (Tool::Move, "✥", "Move (G)"),
                        (Tool::Rotate, "⟳", "Rotate (R)"),
                        (Tool::Scale, "⤡", "Scale (S)"),
                    ] {
                        if ui
                            .add(egui::Button::selectable(
                                self.tool == t,
                                egui::RichText::new(icon).size(15.0),
                            ))
                            .on_hover_text(tip)
                            .clicked()
                        {
                            self.tool = t;
                        }
                    }
                });
            });
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            egui::Frame::new()
                .fill(theme::HEADER_PILL)
                .corner_radius(6.0)
                .inner_margin(egui::Margin::symmetric(4, 2))
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        for (s, icon, tip) in [
                            (Shading::Wireframe, "◯", "Wireframe (Z)"),
                            (Shading::Solid, "●", "Solid"),
                            (Shading::Material, "◕", "Material Preview"),
                        ] {
                            if ui
                                .add(egui::Button::selectable(
                                    ed.shading == s,
                                    egui::RichText::new(icon).size(14.0),
                                ))
                                .on_hover_text(tip)
                                .clicked()
                            {
                                ed.shading = s;
                            }
                        }
                    });
                });
            egui::Frame::new()
                .fill(theme::HEADER_PILL)
                .corner_radius(6.0)
                .inner_margin(egui::Margin::symmetric(4, 2))
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.toggle_value(&mut self.overlays.xray, "X-Ray")
                            .on_hover_text("Alt+Z");
                        ui.menu_button("Overlays ▾", |ui| {
                            ui.checkbox(&mut self.overlays.grid, "Floor grid");
                            ui.checkbox(&mut self.overlays.wire_overlay, "Wireframe overlay");
                            ui.checkbox(&mut self.overlays.stats, "Statistics");
                            ui.add_enabled(
                                self.gpu.is_some(),
                                egui::Checkbox::new(&mut self.use_gpu, "GPU renderer (4× MSAA)"),
                            )
                            .on_disabled_hover_text("Not available with this graphics backend");
                        });
                    });
                });
        });
    }

    fn nav_gizmo(&mut self, ui: &mut Ui, ed: &mut Editor) {
        let r = self.rect;
        let c = pos2(r.right() - 62.0, r.top() + 100.0);
        let rad = 44.0;
        let resp = ui.interact(
            Rect::from_center_size(c, vec2(rad * 2.2, rad * 2.2)),
            ui.scope_id().with("navgizmo"),
            Sense::click_and_drag(),
        );
        let painter = ui.painter_at(r);
        if resp.hovered() || resp.dragged() {
            painter.circle_filled(c, rad * 1.1, Color32::from_white_alpha(18));
        }
        let view = self.cam.view();
        let mut balls = vec![];
        for i in 0..3 {
            for sgn in [1.0f32, -1.0] {
                let d = view.transform_vector3(AXES[i] * sgn);
                balls.push((i, sgn, d));
            }
        }
        balls.sort_by(|a, b| a.2.z.partial_cmp(&b.2.z).unwrap());
        let mut clicked_axis = None;
        let hover = resp.hover_pos();
        for (i, sgn, d) in &balls {
            let p = c + vec2(d.x, -d.y) * rad * 0.8;
            let pos_ax = *sgn > 0.0;
            if pos_ax {
                painter.line_segment([c, p], Stroke::new(2.0, AXIS_COLORS[*i]));
            }
            let br = if pos_ax { 9.0 } else { 7.5 };
            let hot = hover.is_some_and(|h| (h - p).length() < br + 1.0);
            let col = if pos_ax {
                AXIS_COLORS[*i]
            } else {
                AXIS_COLORS[*i].gamma_multiply(0.45)
            };
            painter.circle_filled(p, br, col);
            if hot {
                painter.circle_stroke(p, br, Stroke::new(1.5, Color32::WHITE));
                if resp.clicked() {
                    clicked_axis = Some((*i, *sgn));
                }
            }
            if pos_ax {
                painter.text(
                    p,
                    Align2::CENTER_CENTER,
                    ["X", "Y", "Z"][*i],
                    FontId::proportional(11.0),
                    Color32::from_gray(20),
                );
            }
        }
        if let Some((i, s)) = clicked_axis {
            let p = match (i, s > 0.0) {
                (0, true) => ViewPreset::Right,
                (0, false) => ViewPreset::Left,
                (1, true) => ViewPreset::Back,
                (1, false) => ViewPreset::Front,
                (2, true) => ViewPreset::Top,
                _ => ViewPreset::Bottom,
            };
            ed.view_requests.push(ViewRequest::Preset(p));
        }
        if resp.dragged() {
            let d = resp.drag_delta();
            self.target_cam = None;
            self.leave_camera_view();
            self.cam.yaw -= d.x * 0.5;
            self.cam.pitch = (self.cam.pitch + d.y * 0.5).clamp(-90.0, 90.0);
        }
        // zoom / pan / ortho buttons
        let mut y = c.y + rad + 22.0;
        for (icon, id, tip) in [
            ("🔍", "zoom", "Drag to zoom"),
            ("✋", "pan", "Drag to pan"),
            (
                if self.cam.ortho { "▦" } else { "🎥" },
                "ortho",
                "Toggle perspective/orthographic (Numpad 5)",
            ),
            ("⛶", "frame", "Frame all (Home)"),
        ] {
            let br = Rect::from_center_size(pos2(c.x, y), vec2(28.0, 28.0));
            let b = ui
                .interact(br, ui.scope_id().with(id), Sense::click_and_drag())
                .on_hover_text(tip);
            painter.circle_filled(
                br.center(),
                14.0,
                if b.hovered() || b.dragged() {
                    Color32::from_white_alpha(45)
                } else {
                    Color32::from_white_alpha(18)
                },
            );
            painter.text(
                br.center(),
                Align2::CENTER_CENTER,
                icon,
                FontId::proportional(13.0),
                Color32::from_gray(225),
            );
            if b.dragged() {
                let d = b.drag_delta();
                self.target_cam = None;
                if id == "zoom" {
                    self.cam.dist = (self.cam.dist * (1.0 + d.y * 0.01)).clamp(0.2, 900.0);
                } else if id == "pan" {
                    let wpp = self.world_per_pixel(self.cam.target);
                    self.cam.target += (-self.cam.right() * d.x + self.cam.up() * d.y) * wpp;
                }
            }
            if b.clicked() {
                if id == "ortho" {
                    ed.view_requests.push(ViewRequest::ToggleOrtho);
                } else if id == "frame" {
                    ed.view_requests.push(ViewRequest::FrameAll);
                }
            }
            y += 34.0;
        }
    }

    // ------------------------------------------------------------------------------------------
    // Input
    // ------------------------------------------------------------------------------------------
    fn begin_modal(
        &mut self,
        ed: &mut Editor,
        kind: ModalKind,
        mouse: Pos2,
        axis: Option<usize>,
        from_gizmo: bool,
    ) {
        if ed.editing() {
            self.begin_vert_modal(ed, kind, mouse, from_gizmo);
            if let Some(m) = &mut self.modal {
                m.axis = axis;
            }
            return;
        }
        let Some(pivot) = Self::selection_pivot(ed) else {
            return;
        };
        ed.snapshot();
        let originals = ed
            .selection
            .iter()
            .filter_map(|id| ed.scene.get(*id).map(|o| (*id, o.transform)))
            .collect();
        self.modal = Some(Modal {
            kind,
            axis,
            axis_vec: None,
            start_mouse: mouse,
            originals,
            verts: vec![],
            edit_base: None,
            pivot,
            typed: String::new(),
            from_gizmo,
            own_snapshot: true,
        });
    }

    fn apply_modal(&mut self, ed: &mut Editor, mouse: Pos2, ctrl_snap: bool) {
        let Some(m) = self.modal.clone() else { return };
        if m.kind == ModalKind::Inset {
            self.apply_inset(ed, &m, mouse);
            return;
        }
        let edit = ed.editing();
        let typed: Option<f32> = m.typed.parse().ok();
        let center = self.project(m.pivot).unwrap_or(self.rect.center());
        let to_viewer = if self.cam.ortho {
            -self.cam.forward()
        } else {
            (self.cam.eye() - m.pivot).normalize()
        };
        match m.kind {
            ModalKind::Inset => {}
            ModalKind::Grab => {
                let constraint = m.axis.map(|a| AXES[a]).or(m.axis_vec);
                let delta_w = if let Some(axis) = constraint {
                    let amount = if let Some(t) = typed {
                        t
                    } else {
                        let s0 = center;
                        let s1 = self
                            .project(m.pivot + axis)
                            .unwrap_or(center + vec2(1.0, 0.0));
                        let sd = s1 - s0;
                        let ppu = sd.length().max(1e-3);
                        (mouse - m.start_mouse).dot(sd / ppu) / ppu
                    };
                    let amount = if ctrl_snap { amount.round() } else { amount };
                    axis * amount
                } else {
                    let wpp = self.world_per_pixel(m.pivot);
                    let d = mouse - m.start_mouse;
                    let mut v = (self.cam.right() * d.x - self.cam.up() * d.y) * wpp;
                    if ctrl_snap {
                        v = v.round();
                    }
                    v
                };
                if edit {
                    self.apply_vert_modal(ed, &m, &|p| p + delta_w);
                }
                for (id, t0) in &m.originals {
                    if let Some(o) = ed.scene.get_mut(*id) {
                        o.transform.location = t0.location + delta_w;
                    }
                }
            }
            ModalKind::Rotate => {
                let a0 = (m.start_mouse - center).angle();
                let a1 = (mouse - center).angle();
                // screen y points down -> negate for counter-clockwise positive
                let mut ang = -(a1 - a0).to_degrees();
                let axis = match m.axis {
                    Some(ax) => {
                        let a = AXES[ax];
                        if a.dot(to_viewer) < 0.0 {
                            ang = -ang;
                        }
                        a
                    }
                    None => to_viewer,
                };
                if let Some(t) = typed {
                    ang = t;
                }
                if ctrl_snap {
                    ang = (ang / 5.0).round() * 5.0;
                }
                let q = Quat::from_axis_angle(axis.normalize(), ang.to_radians());
                if edit {
                    let pv = m.pivot;
                    self.apply_vert_modal(ed, &m, &|p| pv + q * (p - pv));
                }
                for (id, t0) in &m.originals {
                    if let Some(o) = ed.scene.get_mut(*id) {
                        o.transform.location = m.pivot + q * (t0.location - m.pivot);
                        o.transform.set_quat(q * t0.quat());
                    }
                }
            }
            ModalKind::Scale => {
                let d0 = (m.start_mouse - center).length().max(1.0);
                let mut f = (mouse - center).length() / d0;
                if let Some(t) = typed {
                    f = t;
                }
                if ctrl_snap {
                    f = (f * 10.0).round() / 10.0;
                }
                let fv = match m.axis {
                    Some(ax) => {
                        let mut v = Vec3::ONE;
                        v[ax] = f;
                        v
                    }
                    None => Vec3::splat(f),
                };
                if edit {
                    let pv = m.pivot;
                    self.apply_vert_modal(ed, &m, &|p| pv + (p - pv) * fv);
                }
                for (id, t0) in &m.originals {
                    if let Some(o) = ed.scene.get_mut(*id) {
                        o.transform.scale = t0.scale * fv;
                        o.transform.location = m.pivot + (t0.location - m.pivot) * fv;
                    }
                }
            }
        }
    }

    fn end_modal(&mut self, ed: &mut Editor, confirm: bool) {
        if let Some(m) = self.modal.take() {
            if !confirm {
                for (id, t0) in &m.originals {
                    if let Some(o) = ed.scene.get_mut(*id) {
                        o.transform = *t0;
                    }
                }
                if ed.editing() {
                    self.cancel_vert_modal(ed, &m);
                }
                if m.own_snapshot {
                    ed.undo_stack.pop();
                }
            } else if ed.auto_key && !ed.editing() {
                ed.insert_keyframes();
            }
        }
    }

    fn handle_input(&mut self, ui: &mut Ui, resp: &egui::Response, ed: &mut Editor) {
        let ctx = ui.ctx().clone();
        let typing = ctx.egui_wants_keyboard_input();
        let hovered = resp.hovered() || resp.dragged() || self.modal.is_some();
        let mouse = ctx.pointer_latest_pos().unwrap_or(self.rect.center());
        let (shift, ctrl, alt) =
            ctx.input(|i| (i.modifiers.shift, i.modifiers.command, i.modifiers.alt));

        // ---------- modal transform ----------
        if self.modal.is_some() {
            let mut confirm = None;
            ctx.input_mut(|i| {
                for (k, ax) in [(Key::X, 0), (Key::Y, 1), (Key::Z, 2)] {
                    if i.consume_key(egui::Modifiers::NONE, k)
                        || i.consume_key(egui::Modifiers::SHIFT, k)
                    {
                        if let Some(m) = &mut self.modal {
                            m.axis = if m.axis == Some(ax) { None } else { Some(ax) };
                            m.axis_vec = None;
                        }
                    }
                }
                for e in &i.events {
                    if let egui::Event::Text(t) = e {
                        if let Some(m) = &mut self.modal {
                            for ch in t.chars() {
                                if ch.is_ascii_digit()
                                    || ch == '.'
                                    || (ch == '-' && m.typed.is_empty())
                                {
                                    m.typed.push(ch);
                                }
                            }
                        }
                    }
                }
                if i.key_pressed(Key::Backspace) {
                    if let Some(m) = &mut self.modal {
                        m.typed.pop();
                    }
                }
                if i.key_pressed(Key::Minus) {
                    if let Some(m) = &mut self.modal {
                        if m.typed.starts_with('-') {
                            m.typed.remove(0);
                        } else if !m.typed.contains('-') {
                            m.typed.insert(0, '-');
                        }
                    }
                }
                if i.key_pressed(Key::Enter) || i.pointer.button_pressed(PointerButton::Primary) {
                    confirm = Some(true);
                }
                if i.key_pressed(Key::Escape) || i.pointer.button_pressed(PointerButton::Secondary)
                {
                    confirm = Some(false);
                }
                if self.modal.as_ref().is_some_and(|m| m.from_gizmo)
                    && i.pointer.button_released(PointerButton::Primary)
                {
                    confirm = Some(true);
                }
            });
            // typed '-' was already handled via Text event; avoid double-handling Minus
            self.apply_modal(ed, mouse, ctrl);
            if let Some(c) = confirm {
                self.end_modal(ed, c);
            }
            ctx.request_repaint();
            self.status_hint = "[LMB/Enter] Confirm   [RMB/Esc] Cancel   [X/Y/Z] Axis   [Ctrl] Snap   type a number for exact value".into();
            return;
        }
        self.status_hint = String::new();
        if ed.editing() {
            self.status_hint = "Edit Mode · 1/2/3 vertex/edge/face · E extrude · I inset · Ctrl+R loop cut · M merge · X delete · G/R/S · Tab exit".into();
        }
        if self.loopcut_input(&ctx, resp, ed, mouse) {
            // still allow navigation while choosing the loop
            if resp.dragged_by(PointerButton::Middle) {
                let d = resp.drag_delta();
                self.cam.yaw -= d.x * 0.4;
                self.cam.pitch = (self.cam.pitch + d.y * 0.4).clamp(-90.0, 90.0);
            }
            return;
        }

        // ---------- navigation ----------
        let mmb = resp.dragged_by(PointerButton::Middle);
        let alt_lmb = alt && resp.dragged_by(PointerButton::Primary);
        if mmb || alt_lmb {
            self.target_cam = None;
            self.leave_camera_view();
            let d = resp.drag_delta();
            if shift {
                let wpp = self.world_per_pixel(self.cam.target);
                self.cam.target += (-self.cam.right() * d.x + self.cam.up() * d.y) * wpp;
            } else if ctrl {
                self.cam.dist = (self.cam.dist * (1.0 + d.y * 0.01)).clamp(0.2, 900.0);
            } else {
                self.cam.yaw -= d.x * 0.4;
                self.cam.pitch = (self.cam.pitch + d.y * 0.4).clamp(-90.0, 90.0);
                if self.cam.ortho && !alt_lmb {
                    // Blender auto-perspective
                    self.cam.ortho = false;
                }
            }
        }
        if resp.hovered() {
            let (scroll, zoom) = ctx.input(|i| (i.smooth_scroll_delta, i.zoom_delta()));
            if scroll.y.abs() > 0.0 || zoom != 1.0 {
                self.target_cam = None;
                self.leave_camera_view();
                let f = (1.0 - scroll.y * 0.0015).clamp(0.5, 1.5) / zoom;
                self.cam.dist = (self.cam.dist * f).clamp(0.2, 900.0);
            }
            if scroll.x.abs() > 0.0 {
                self.cam.yaw -= scroll.x * 0.2;
            }
        }

        // ---------- gizmo ----------
        self.gizmo_hover = if resp.hovered() {
            self.gizmo_hit(ed, mouse)
        } else {
            None
        };
        if resp.drag_started_by(PointerButton::Primary) && !alt {
            let press = ctx.input(|i| i.pointer.press_origin()).unwrap_or(mouse);
            if let Some(h) = self.gizmo_hit(ed, press) {
                let kind = match self.tool {
                    Tool::Rotate => ModalKind::Rotate,
                    Tool::Scale => ModalKind::Scale,
                    _ => ModalKind::Grab,
                };
                let axis = match h {
                    GizmoHandle::Axis(a) => Some(a),
                    GizmoHandle::Free => None,
                };
                self.begin_modal(ed, kind, press, axis, true);
                return;
            }
            self.box_select = Some((press, mouse));
        }
        if let Some((a, _)) = self.box_select {
            if resp.dragged_by(PointerButton::Primary) {
                self.box_select = Some((a, mouse));
            }
            if resp.drag_stopped() && ed.editing() {
                self.edit_box(ed, Rect::from_two_pos(a, mouse), shift, ctrl);
                self.box_select = None;
            } else if resp.drag_stopped() {
                let r = Rect::from_two_pos(a, mouse);
                let hits: Vec<u64> = ed
                    .scene
                    .objects
                    .iter()
                    .filter(|o| {
                        o.visible
                            && self
                                .project(o.transform.location)
                                .is_some_and(|p| r.contains(p))
                    })
                    .map(|o| o.id)
                    .collect();
                if !shift && !ctrl {
                    ed.selection.clear();
                }
                for h in hits {
                    if ctrl {
                        ed.selection.retain(|s| *s != h);
                    } else if !ed.selection.contains(&h) {
                        ed.selection.push(h);
                    }
                }
                ed.active = ed.selection.last().copied();
                self.box_select = None;
            }
        }

        // ---------- click select ----------
        if resp.clicked_by(PointerButton::Primary) && !alt && ed.editing() {
            self.edit_click(ed, mouse, shift);
        } else if resp.clicked_by(PointerButton::Primary) && !alt {
            let hit = self.pick(mouse);
            match (hit, shift) {
                (Some(id), true) => ed.toggle_select(id),
                (Some(id), false) => ed.select_only(id),
                (None, false) => {
                    ed.selection.clear();
                    ed.active = None;
                }
                _ => {}
            }
        }
        if resp.clicked_by(PointerButton::Secondary) && shift {
            if let Some(p) = self.ground_point(mouse) {
                self.cursor = p;
            }
        }

        // ---------- keyboard ----------
        if typing || !hovered {
            return;
        }
        if ed.editing() && self.edit_keys(&ctx, ed, mouse) {
            return;
        }
        let pressed = |k: Key| ctx.input(|i| i.key_pressed(k));
        let shortcut = |m: egui::Modifiers, k: Key| ctx.input_mut(|i| i.consume_key(m, k));
        if shortcut(egui::Modifiers::SHIFT, Key::A) {
            self.add_menu_at = Some(mouse);
        } else if shortcut(egui::Modifiers::SHIFT, Key::D) {
            if !ed.duplicate_selected(Vec3::ZERO).is_empty() {
                // duplicate then immediately grab, like Blender
                ed.undo_stack.pop();
                let snapshot_count = ed.undo_stack.len();
                self.begin_modal(ed, ModalKind::Grab, mouse, None, false);
                // keep single undo step for dup+move
                while ed.undo_stack.len() > snapshot_count + 1 {
                    ed.undo_stack.pop();
                }
            }
        } else if shortcut(egui::Modifiers::ALT, Key::Z) {
            self.overlays.xray = !self.overlays.xray;
        } else if shortcut(egui::Modifiers::ALT, Key::H) || shortcut(egui::Modifiers::ALT, Key::H) {
            for o in &mut ed.scene.objects {
                o.visible = true;
            }
        } else if pressed(Key::G) && !ctrl {
            self.begin_modal(ed, ModalKind::Grab, mouse, None, false);
        } else if pressed(Key::R) && !ctrl {
            self.begin_modal(ed, ModalKind::Rotate, mouse, None, false);
        } else if pressed(Key::S) && !ctrl {
            self.begin_modal(ed, ModalKind::Scale, mouse, None, false);
        } else if (pressed(Key::X) && !ctrl) || pressed(Key::Delete) {
            ed.delete_selected();
        } else if pressed(Key::A) && !ctrl && !alt {
            ed.select_all_toggle();
        } else if pressed(Key::A) && alt {
            ed.selection.clear();
            ed.active = None;
        } else if pressed(Key::H) && !alt {
            for id in ed.selection.clone() {
                if let Some(o) = ed.scene.get_mut(id) {
                    o.visible = false;
                }
            }
        } else if pressed(Key::Z) && !ctrl && !shift {
            ed.shading = match ed.shading {
                Shading::Wireframe => Shading::Solid,
                Shading::Solid => Shading::Material,
                Shading::Material => Shading::Wireframe,
            };
        } else if pressed(Key::W) {
            self.tool = match self.tool {
                Tool::Select => Tool::Move,
                Tool::Move => Tool::Rotate,
                Tool::Rotate => Tool::Scale,
                Tool::Scale => Tool::Select,
            };
        } else if pressed(Key::I) {
            ed.insert_keyframes();
        } else if pressed(Key::Home) {
            ed.view_requests.push(ViewRequest::FrameAll);
        } else if pressed(Key::Period) && !ctrl {
            ed.view_requests.push(ViewRequest::FrameSelected);
        } else if pressed(Key::F) && !ctrl {
            ed.view_requests.push(ViewRequest::FrameSelected);
        }
        // numpad-ish views (egui reports numpad digits as regular Num keys)
        let cpre = ctrl;
        for (k, p, alt_p) in [
            (Key::Num1, ViewPreset::Front, ViewPreset::Back),
            (Key::Num3, ViewPreset::Right, ViewPreset::Left),
            (Key::Num7, ViewPreset::Top, ViewPreset::Bottom),
        ] {
            if ctx.input(|i| i.key_pressed(k)) {
                ed.view_requests
                    .push(ViewRequest::Preset(if cpre { alt_p } else { p }));
            }
        }
        if pressed(Key::Num5) {
            ed.view_requests.push(ViewRequest::ToggleOrtho);
        }
        if pressed(Key::Num0) {
            ed.view_requests.push(if ctrl && alt {
                ViewRequest::AlignCameraToView
            } else {
                ViewRequest::Camera
            });
        }
        if pressed(Key::Num4) {
            ed.view_requests.push(ViewRequest::Orbit(-15.0, 0.0));
        }
        if pressed(Key::Num6) {
            ed.view_requests.push(ViewRequest::Orbit(15.0, 0.0));
        }
        if pressed(Key::Num8) {
            ed.view_requests.push(ViewRequest::Orbit(0.0, 15.0));
        }
        if pressed(Key::Num2) {
            ed.view_requests.push(ViewRequest::Orbit(0.0, -15.0));
        }
        if pressed(Key::Plus) || pressed(Key::Equals) {
            ed.view_requests.push(ViewRequest::Zoom(0.8));
        }
        if pressed(Key::Minus) {
            ed.view_requests.push(ViewRequest::Zoom(1.25));
        }
    }
}

fn seg_dist(p: Pos2, a: Pos2, b: Pos2) -> f32 {
    let ab = b - a;
    let t = ((p - a).dot(ab) / ab.length_sq().max(1e-6)).clamp(0.0, 1.0);
    (p - (a + ab * t)).length()
}

fn dashed(painter: &egui::Painter, a: Pos2, b: Pos2, col: Color32) {
    let d = b - a;
    let len = d.length();
    if len < 1.0 {
        return;
    }
    let dir = d / len;
    let mut t = 0.0;
    while t < len {
        let e = (t + 6.0).min(len);
        painter.line_segment([a + dir * t, a + dir * e], Stroke::new(1.2, col));
        t += 11.0;
    }
}

#[allow(dead_code)]
pub fn vec_to_pos(v: Vec2) -> Pos2 {
    pos2(v.x, v.y)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn camera_basis_is_orthonormal() {
        for (y, p) in [(0.0, 0.0), (48.0, 28.0), (90.0, 90.0), (200.0, -90.0)] {
            let c = Camera {
                yaw: y,
                pitch: p,
                ..Default::default()
            };
            let (f, r, u) = (c.forward(), c.right(), c.up());
            assert!(
                f.dot(r).abs() < 1e-4 && f.dot(u).abs() < 1e-4 && r.dot(u).abs() < 1e-4,
                "{y} {p}"
            );
        }
        let mut c = Camera::default();
        c.preset(ViewPreset::Front);
        assert!((c.eye() - Vec3::new(0.0, -17.0, 0.0)).length() < 1e-3);
        assert_eq!(c.view_name(), "Front Orthographic");
    }
    #[test]
    fn project_and_pick() {
        let mut v = Viewport {
            rect: Rect::from_min_size(Pos2::ZERO, vec2(800.0, 600.0)),
            ..Default::default()
        };
        v.vp = v.cam.proj(800.0 / 600.0) * v.cam.view();
        let p = v.project(Vec3::ZERO).unwrap();
        assert!((p - pos2(400.0, 300.0)).length() < 1.0);
        let ed = Editor::default();
        v.raster.resize(800, 600);
        v.proj = v.cam.proj(800.0 / 600.0);
        v.build_scene(&ed);
        let prims = v.prims.clone();
        v.raster.render([0.2; 3], [0.2; 3], &prims, None);
        assert_eq!(
            v.pick(pos2(400.0, 300.0)),
            ed.scene.objects.first().map(|o| o.id)
        );
        assert_eq!(v.pick(pos2(5.0, 5.0)), None);
        let (o, d) = v.ray(pos2(400.0, 300.0));
        assert!((d - v.cam.forward()).length() < 1e-3, "{o} {d}");
    }
}
