//! The start screen (templates + recent files) and the Asset Library window, with background
//! thumbnail rendering and drag-and-drop of assets into the viewport.

use super::App;
use crate::assets::{self, Asset, AssetKind, Entry, ThumbSource};
use crate::theme;
use crossbeam_channel::{Receiver, Sender, unbounded};
use egui::{Color32, RichText, Sense, vec2};
use std::collections::HashMap;
use std::path::PathBuf;

enum Job {
    Template(String),
    Builtin(String),
    User(PathBuf, String),
}

struct Done {
    key: String,
    w: usize,
    h: usize,
    rgba: Vec<u8>,
}

/// Dragged from the library onto the viewport.
#[derive(Clone)]
pub struct AssetDrag {
    entry: Entry,
}

pub struct Library {
    pub show_start: bool,
    pub show_assets: bool,
    tab: AssetKind,
    search: String,
    save_name: String,
    /// what `save_name` was filled in from (refilled when the selection changes, unless edited)
    save_auto: Option<(u64, AssetKind, String)>,
    thumbs: HashMap<String, Option<egui::TextureHandle>>,
    jobs: Sender<Job>,
    done: Receiver<Done>,
    user: Vec<Entry>,
    user_loaded: bool,
    /// a template being built in the background
    building: Option<(String, Receiver<Option<crate::scene::Scene>>)>,
    /// asset whose thumbnail should be written next to it when it arrives
    status: Option<String>,
}

pub const TEMPLATE_THUMB: (usize, usize) = (224, 126);
pub const ASSET_THUMB: usize = 112;

fn template_key(name: &str) -> String {
    format!("template:{name}:v4")
}

impl Library {
    pub fn new(show_start: bool) -> Library {
        let (jobs, rx) = unbounded::<Job>();
        let (tx, done) = unbounded::<Done>();
        std::thread::Builder::new()
            .name("thumbnails".into())
            .spawn(move || {
                while let Ok(job) = rx.recv() {
                    let (key, src, w, h, write_next_to) = match job {
                        Job::Template(n) => {
                            let key = template_key(&n);
                            if let Some((w, h, rgba)) = assets::cached_thumb(&key) {
                                let _ = tx.send(Done { key, w, h, rgba });
                                continue;
                            }
                            let Some(s) = crate::templates::build(&n) else { continue };
                            (key, ThumbSource::Scene(s), TEMPLATE_THUMB.0, TEMPLATE_THUMB.1, None)
                        }
                        Job::Builtin(n) => {
                            let key = assets::builtin_entries().into_iter().find(|e| e.name == n).map(|e| e.key).unwrap_or_default();
                            if let Some((w, h, rgba)) = assets::cached_thumb(&key) {
                                let _ = tx.send(Done { key, w, h, rgba });
                                continue;
                            }
                            let Some(a) = assets::builtin(&n) else { continue };
                            (key, ThumbSource::Asset(a), ASSET_THUMB, ASSET_THUMB, None)
                        }
                        Job::User(p, key) => {
                            let png = p.with_extension("png");
                            if let Some(img) = std::fs::read(&png).ok().and_then(|b| crate::textures::decode(&crate::textures::TexSource::Image { data: b }).ok()) {
                                let _ = tx.send(Done { key, w: img.w, h: img.h, rgba: img.rgba.into_iter().flatten().collect() });
                                continue;
                            }
                            let Ok(a) = assets::load(&p) else { continue };
                            (key, ThumbSource::Asset(a), ASSET_THUMB, ASSET_THUMB, Some(png))
                        }
                    };
                    let rgba = assets::render_thumb(&src, w, h, 24);
                    match write_next_to {
                        Some(p) => {
                            if let Ok(b) = crate::textures::encode_png(w, h, &rgba) {
                                let _ = std::fs::write(p, b);
                            }
                        }
                        None => assets::store_thumb(&key, w, h, &rgba),
                    }
                    let _ = tx.send(Done { key, w, h, rgba });
                }
            })
            .expect("thumbnail thread");
        Library { show_start, show_assets: false, tab: AssetKind::Object, search: String::new(), save_name: String::new(), save_auto: None, thumbs: HashMap::new(), jobs, done, user: vec![], user_loaded: false, building: None, status: None }
    }

    fn thumb(&mut self, key: &str, job: impl FnOnce() -> Job) -> Option<egui::TextureHandle> {
        match self.thumbs.get(key) {
            Some(t) => t.clone(),
            None => {
                self.thumbs.insert(key.to_string(), None);
                let _ = self.jobs.send(job());
                None
            }
        }
    }

    fn refresh_user(&mut self) {
        self.user = assets::list(&assets::library_dir());
        self.user_loaded = true;
    }

    fn load_entry(e: &Entry) -> Result<Asset, String> {
        match &e.path {
            Some(p) => assets::load(p),
            None => assets::builtin(&e.name).ok_or_else(|| format!("no built-in asset {}", e.name)),
        }
    }
}

/// A thumbnail card: image (or a placeholder while it renders) and a caption.
fn card(ui: &mut egui::Ui, tex: Option<&egui::TextureHandle>, size: egui::Vec2, title: &str, sub: Option<&str>, hovered_outline: bool) -> egui::Response {
    let h = size.y + 20.0 + if sub.is_some() { 30.0 } else { 0.0 };
    let (rect, resp) = ui.allocate_exact_size(vec2(size.x, h), Sense::click_and_drag());
    let p = ui.painter_at(rect.expand(2.0));
    let img = egui::Rect::from_min_size(rect.min, size);
    p.rect_filled(img, 6.0, Color32::from_rgb(30, 30, 32));
    match tex {
        Some(t) => {
            p.image(t.id(), img.shrink(1.0), egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)), Color32::WHITE);
        }
        None => {
            let t = ui.input(|i| i.time) as f32;
            let c = img.center();
            for k in 0..3 {
                let a = t * 3.0 + k as f32 * 2.1;
                p.circle_filled(c + vec2(a.cos(), a.sin()) * 10.0, 3.0, theme::TEXT_DIM);
            }
            ui.ctx().request_repaint_after(std::time::Duration::from_millis(60));
        }
    }
    if resp.hovered() || hovered_outline {
        p.rect_stroke(img, 6.0, egui::Stroke::new(2.0, theme::ACCENT_BRIGHT), egui::StrokeKind::Inside);
    }
    p.text(img.left_bottom() + vec2(2.0, 3.0), egui::Align2::LEFT_TOP, title, egui::FontId::proportional(13.0), Color32::from_gray(225));
    if let Some(s) = sub {
        let galley = ui.painter().layout(s.to_string(), egui::FontId::proportional(11.0), theme::TEXT_DIM, size.x);
        p.galley(img.left_bottom() + vec2(2.0, 20.0), galley, theme::TEXT_DIM);
    }
    resp
}

impl App {
    pub(super) fn library_ui(&mut self, ctx: &egui::Context) {
        // finished thumbnails → textures
        while let Ok(d) = self.lib.done.try_recv() {
            let img = egui::ColorImage::from_rgba_unmultiplied([d.w, d.h], &d.rgba);
            let t = ctx.load_texture(d.key.clone(), img, egui::TextureOptions::LINEAR);
            self.lib.thumbs.insert(d.key, Some(t));
        }
        // a template finished building
        if let Some((name, rx)) = &self.lib.building {
            if let Ok(s) = rx.try_recv() {
                let name = name.clone();
                self.lib.building = None;
                match s {
                    Some(scene) => {
                        self.ed.snapshot();
                        let msg = self.ed.use_template_scene(&name, scene);
                        self.toast(msg, theme::OK);
                        self.lib.show_start = false;
                    }
                    None => self.toast(format!("No template {name}"), theme::ERR),
                }
            } else {
                ctx.request_repaint_after(std::time::Duration::from_millis(50));
            }
        }
        self.start_screen(ctx);
        self.asset_window(ctx);
        self.asset_drop(ctx);
    }

    pub(super) fn start_template(&mut self, name: &str) {
        if self.lib.building.is_some() {
            return;
        }
        let (tx, rx) = unbounded();
        let n = name.to_string();
        std::thread::spawn(move || {
            let _ = tx.send(crate::templates::build(&n));
        });
        self.lib.building = Some((name.to_string(), rx));
    }

    fn start_screen(&mut self, ctx: &egui::Context) {
        if !self.lib.show_start {
            return;
        }
        let mut open = true;
        let mut pick: Option<String> = None;
        let mut open_file: Option<PathBuf> = None;
        let mut browse = false;
        let screen = ctx.content_rect();
        let wsize = vec2((screen.width() - 80.0).min(1010.0), (screen.height() - 140.0).min(600.0));
        egui::Window::new(RichText::new("◆  Rust Blender").strong())
            .id(egui::Id::unique("start_screen"))
            .collapsible(false)
            .resizable(false)
            .open(&mut open)
            .fixed_pos(screen.center() - wsize * 0.5 - vec2(0.0, 20.0))
            .fixed_size(wsize)
            .show(ctx, |ui| {
                ui.horizontal_top(|ui| {
                    ui.vertical(|ui| {
                        ui.label(RichText::new("New file").size(16.0).strong());
                        ui.add_space(4.0);
                        let size = vec2(TEMPLATE_THUMB.0 as f32 * 0.82, TEMPLATE_THUMB.1 as f32 * 0.82);
                        let cols = 4;
                        egui::ScrollArea::vertical().id_salt("tmpl").max_height(ui.available_height() - 4.0).show(ui, |ui| {
                            egui::Grid::new("templates").spacing(vec2(12.0, 10.0)).show(ui, |ui| {
                                for (i, t) in crate::templates::TEMPLATES.iter().enumerate() {
                                    let key = template_key(t.name);
                                    let n = t.name.to_string();
                                    let tex = self.lib.thumb(&key, || Job::Template(n));
                                    let busy = self.lib.building.as_ref().is_some_and(|b| b.0 == t.name);
                                    let r = card(ui, tex.as_ref(), size, if busy { "Building…" } else { t.name }, Some(t.blurb), busy);
                                    if r.clicked() {
                                        pick = Some(t.name.to_string());
                                    }
                                    if (i + 1) % cols == 0 {
                                        ui.end_row();
                                    }
                                }
                            });
                        });
                    });
                    ui.separator();
                    ui.vertical(|ui| {
                        ui.set_width(200.0);
                        ui.label(RichText::new("Open").size(16.0).strong());
                        if ui.button("📂  Open…").clicked() {
                            browse = true;
                        }
                        ui.add_space(8.0);
                        ui.label(RichText::new("Recent").color(theme::TEXT_DIM));
                        if self.settings.recent_files.is_empty() {
                            ui.label(RichText::new("No recent files yet").small().color(theme::TEXT_DIM));
                        }
                        for f in self.settings.recent_files.iter().take(10) {
                            let p = PathBuf::from(f);
                            let name = p.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_else(|| f.clone());
                            let exists = p.exists();
                            let r = ui.add_enabled(exists, egui::Button::new(format!("🗋 {name}")).frame(false)).on_hover_text(f);
                            if r.clicked() {
                                open_file = Some(p);
                            }
                        }
                        ui.add_space(12.0);
                        ui.label(RichText::new("Tip: ask the Copilot \"start a product shot\" or \"new landscape scene\".").small().color(theme::TEXT_DIM));
                        ui.add_space(8.0);
                        ui.checkbox(&mut self.settings.show_start_screen, "Show at startup");
                    });
                });
            });
        if let Some(n) = pick {
            self.start_template(&n);
        }
        if browse {
            self.lib.show_start = false;
            self.open();
        }
        if let Some(p) = open_file {
            self.lib.show_start = false;
            self.open_path(&p);
        }
        if !open {
            self.lib.show_start = false;
        }
    }

    fn asset_window(&mut self, ctx: &egui::Context) {
        if !self.lib.show_assets {
            return;
        }
        if !self.lib.user_loaded {
            self.lib.refresh_user();
        }
        let mut open = true;
        let mut clicked: Option<Entry> = None;
        let mut delete: Option<Entry> = None;
        let mut save = false;
        egui::Window::new("Asset Library").open(&mut open).default_size(vec2(470.0, 560.0)).default_pos(egui::pos2(380.0, 90.0)).show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.lib.tab, AssetKind::Object, "🧊 Objects");
                ui.selectable_value(&mut self.lib.tab, AssetKind::Material, "🎨 Materials");
                ui.add(egui::TextEdit::singleline(&mut self.lib.search).hint_text("🔍 search").desired_width(140.0));
            });
            ui.label(
                RichText::new(match self.lib.tab {
                    AssetKind::Object => "Click to place at the 3D cursor, or drag into the viewport.",
                    AssetKind::Material => "Click to apply to the selection, or drag onto an object.",
                })
                .small()
                .color(theme::TEXT_DIM),
            );
            ui.separator();
            let q = self.lib.search.to_lowercase();
            let tab = self.lib.tab;
            let user: Vec<Entry> = self.lib.user.iter().filter(|e| e.kind == tab && (q.is_empty() || e.name.to_lowercase().contains(&q))).cloned().collect();
            let builtin: Vec<Entry> = assets::builtin_entries().into_iter().filter(|e| e.kind == tab && (q.is_empty() || e.name.to_lowercase().contains(&q))).collect();
            let size = vec2(ASSET_THUMB as f32 * 0.85, ASSET_THUMB as f32 * 0.85);
            let per_row = ((ui.available_width() + 8.0) / (size.x + 8.0)).floor().max(1.0) as usize;
            egui::ScrollArea::vertical().id_salt("assets").max_height(ui.available_height() - 70.0).show(ui, |ui| {
                for (title, list) in [("My Library", &user), ("Built-in", &builtin)] {
                    if list.is_empty() && title == "My Library" {
                        ui.label(RichText::new("My Library — empty. Select something and press Save below.").small().color(theme::TEXT_DIM));
                        ui.add_space(6.0);
                        continue;
                    }
                    if list.is_empty() {
                        continue;
                    }
                    ui.label(RichText::new(title).strong());
                    for row in list.chunks(per_row) {
                        ui.horizontal(|ui| {
                            for e in row {
                                let job_e = e.clone();
                                let tex = self.lib.thumb(&e.key, move || match &job_e.path {
                                    Some(p) => Job::User(p.clone(), job_e.key.clone()),
                                    None => Job::Builtin(job_e.name.clone()),
                                });
                                let r = card(ui, tex.as_ref(), size, &e.name, None, false);
                                if r.drag_started() {
                                    egui::DragAndDrop::set_payload(ui.ctx(), AssetDrag { entry: e.clone() });
                                }
                                if r.clicked() {
                                    clicked = Some(e.clone());
                                }
                                let r = r.on_hover_text(match (&e.path, e.kind) {
                                    (Some(p), _) => p.display().to_string(),
                                    (None, AssetKind::Object) => "Built-in object".into(),
                                    (None, AssetKind::Material) => "Built-in material".into(),
                                });
                                if e.path.is_some() {
                                    r.context_menu(|ui| {
                                        if ui.button("🗑 Delete from library").clicked() {
                                            delete = Some(e.clone());
                                            ui.close();
                                        }
                                    });
                                }
                            }
                        });
                        ui.add_space(4.0);
                    }
                    ui.add_space(6.0);
                }
            });
            ui.separator();
            if let Some(o) = self.ed.active.and_then(|a| self.ed.scene.get(a)) {
                let auto = match tab {
                    AssetKind::Object => o.name.clone(),
                    AssetKind::Material => self.ed.scene.graph(o.material.graph).map(|g| g.name.clone()).unwrap_or_else(|| format!("{} Material", o.name)),
                };
                let untouched = self.lib.save_name.is_empty() || self.lib.save_auto.as_ref().is_some_and(|a| a.2 == self.lib.save_name);
                if untouched && self.lib.save_auto.as_ref().is_none_or(|a| a.0 != o.id || a.1 != tab || a.2 != auto) {
                    self.lib.save_name = auto.clone();
                    self.lib.save_auto = Some((o.id, tab, auto));
                }
            }
            ui.horizontal(|ui| {
                ui.add(egui::TextEdit::singleline(&mut self.lib.save_name).hint_text("asset name").desired_width(170.0));
                let label = match tab {
                    AssetKind::Object => "💾 Save Selected",
                    AssetKind::Material => "💾 Save Active Material",
                };
                if ui.add_enabled(self.ed.active.is_some(), egui::Button::new(label)).clicked() {
                    save = true;
                }
            });
            match &self.lib.status {
                Some(s) => ui.label(RichText::new(s).small().color(theme::TEXT_DIM)),
                None => ui.label(RichText::new(format!("Saved to {}", assets::library_dir().display())).small().color(theme::TEXT_DIM)),
            };
        });
        if !open {
            self.lib.show_assets = false;
        }
        if let Some(e) = clicked {
            match e.kind {
                AssetKind::Object => {
                    let at = self.vp.cursor;
                    self.place_entry(&e, at);
                }
                AssetKind::Material => {
                    let ids: Vec<u64> = self.ed.selection.clone();
                    self.apply_entry(&e, &ids);
                }
            }
        }
        if let Some(e) = delete {
            if let Some(p) = &e.path {
                match assets::delete(p) {
                    Ok(()) => self.toast(format!("Deleted {} from the library", e.name), theme::OK),
                    Err(err) => self.toast(err, theme::ERR),
                }
            }
            self.lib.thumbs.remove(&e.key);
            self.lib.refresh_user();
        }
        if save {
            self.save_selected_asset();
        }
    }

    fn save_selected_asset(&mut self) {
        let name = if self.lib.save_name.trim().is_empty() { "Asset".to_string() } else { self.lib.save_name.trim().to_string() };
        let a = match self.lib.tab {
            AssetKind::Object => {
                let ids = self.ed.selection.clone();
                assets::from_objects(&self.ed.scene, &ids, &name)
            }
            AssetKind::Material => self.ed.active.and_then(|a| self.ed.scene.get(a)).map(|o| assets::from_material(&self.ed.scene, &o.material, &name)),
        };
        let Some(a) = a else {
            self.toast("Select something to save", theme::WARN);
            return;
        };
        match assets::save(&assets::library_dir(), &a, None) {
            Ok(p) => {
                self.toast(format!("Saved \"{name}\" to the asset library"), theme::OK);
                self.lib.status = Some(format!("Saved {}", p.display()));
                self.lib.save_name.clear();
                self.lib.save_auto = None;
                self.lib.refresh_user();
            }
            Err(e) => self.toast(format!("Save failed: {e}"), theme::ERR),
        }
    }

    fn place_entry(&mut self, e: &Entry, at: glam::Vec3) {
        match Library::load_entry(e) {
            Ok(a) => {
                self.ed.snapshot();
                let ids = self.ed.place_asset(&a, at);
                self.ed.dirty = true;
                self.toast(format!("Placed {} ({} object{})", a.name, ids.len(), if ids.len() == 1 { "" } else { "s" }), theme::OK);
            }
            Err(err) => self.toast(err, theme::ERR),
        }
    }

    fn apply_entry(&mut self, e: &Entry, ids: &[u64]) {
        if ids.is_empty() {
            self.toast("Select the objects to give the material (or drag it onto one)", theme::WARN);
            return;
        }
        match Library::load_entry(e) {
            Ok(a) => {
                self.ed.snapshot();
                let n = assets::apply_material(&mut self.ed.scene, &a, ids);
                self.ed.dirty = true;
                if self.ed.shading == crate::editor::Shading::Solid {
                    self.ed.shading = crate::editor::Shading::Material;
                }
                self.toast(format!("{} on {n} object{}", a.name, if n == 1 { "" } else { "s" }), theme::OK);
            }
            Err(err) => self.toast(err, theme::ERR),
        }
    }

    /// Dropping a dragged asset: objects land on the surface under the pointer, materials go on
    /// the object under it.
    fn asset_drop(&mut self, ctx: &egui::Context) {
        let Some(drag) = egui::DragAndDrop::payload::<AssetDrag>(ctx) else { return };
        let pos = ctx.pointer_latest_pos();
        // the dragged card follows the pointer
        if let Some(p) = pos {
            let tex = self.lib.thumbs.get(&drag.entry.key).cloned().flatten();
            egui::Area::new(egui::Id::unique("asset_drag_ghost")).order(egui::Order::Tooltip).fixed_pos(p + vec2(10.0, 10.0)).interactable(false).show(ctx, |ui| {
                egui::Frame::popup(ui.style()).show(ui, |ui| {
                    if let Some(t) = &tex {
                        ui.add(egui::Image::new(t).fit_to_exact_size(vec2(56.0, 56.0)));
                    }
                    ui.label(RichText::new(&drag.entry.name).small());
                });
            });
        }
        if !ctx.input(|i| i.pointer.any_released()) {
            return;
        }
        egui::DragAndDrop::clear_payload(ctx);
        let Some(p) = pos.filter(|p| self.vp.screen_rect().contains(*p)) else { return };
        // not when released over a window on top of the viewport
        if ctx.layer_id_at(p).is_some_and(|l| l.order != egui::Order::Background) {
            return;
        }
        match drag.entry.kind {
            AssetKind::Object => {
                let at = self.vp.surface_point(&self.ed, p);
                self.place_entry(&drag.entry, at);
            }
            AssetKind::Material => {
                let ids: Vec<u64> = self.vp.pick(p).map(|id| vec![id]).unwrap_or_else(|| self.ed.selection.clone());
                self.apply_entry(&drag.entry, &ids);
            }
        }
    }
}
