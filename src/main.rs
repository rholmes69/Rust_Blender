#![cfg_attr(
    all(not(debug_assertions), target_os = "windows"),
    windows_subsystem = "windows"
)]

mod agent;
mod armature;
mod scatter;
mod shapekeys;
mod physics;
mod ai;
mod app;
mod editmesh;
mod csg;
mod editor;
mod gltf;
mod gltf_import;
mod gpu;
mod mcp;
mod pathtrace;
mod raytrace;
mod mesh;
mod render;
mod router;
mod scene;
mod sculpt;
mod textures;
mod uv;
mod theme;
mod viewport;
mod voice;
mod voiceflow;

fn main() -> eframe::Result<()> {
    // Headless helper: `rust_blender --export-demo out.glb` builds a demo scene and exports it.
    let args: Vec<String> = std::env::args().collect();
    // MCP stdio shim: `rust_blender --mcp-stdio` (launched by Claude Desktop, Cursor, …).
    if args.get(1).map(String::as_str) == Some("--mcp-stdio") {
        mcp::run_stdio_bridge();
        return Ok(());
    }
    if args.len() >= 3 && args[1] == "--export-demo" {
        let mut ed = editor::Editor::default();
        let cmds: Vec<editor::Command> =
            serde_json::from_str(include_str!("../docs/demo_scene.json")).expect("demo json");
        for l in ed.execute(&cmds) {
            println!("{l}");
        }
        let (bytes, st) = gltf::export_glb(&ed.scene, &mut scene::MeshCache::default());
        std::fs::write(&args[2], bytes).expect("write");
        println!(
            "wrote {} ({} meshes, {} lights, {} cameras, {} animations, {} bytes)",
            args[2], st.meshes, st.lights, st.cameras, st.animations, st.bytes
        );
        return Ok(());
    }

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("Rust Blender")
            .with_inner_size([1600.0, 950.0])
            .with_min_inner_size([900.0, 560.0]),
        // egui needs no MSAA; the GPU viewport renderer does its own 4x MSAA offscreen.
        multisampling: 1,
        ..Default::default()
    };
    eframe::run_native(
        "Rust Blender",
        options,
        Box::new(|cc| Ok(Box::new(app::App::new(cc)))),
    )
}
