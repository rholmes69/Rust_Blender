#![cfg_attr(all(not(debug_assertions), target_os = "windows"), windows_subsystem = "windows")]

mod ai;
mod app;
mod editor;
mod mesh;
mod render;
mod scene;
mod theme;
mod viewport;
mod voice;

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_title("Rust Blender").with_inner_size([1600.0, 950.0]).with_min_inner_size([900.0, 560.0]),
        multisampling: 4,
        ..Default::default()
    };
    eframe::run_native("Rust Blender", options, Box::new(|cc| Ok(Box::new(app::App::new(cc)))))
}
