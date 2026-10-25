//! Start-up templates: ready-made scenes (studio product shot, turntable, landscape, interior,
//! physics playground, hair study …) built from ordinary objects, so everything in them stays
//! editable. Also the procedural meshes they (and the built-in asset library) use.

use crate::editor::{Command, Editor};
use crate::mesh::PolyMesh;
use crate::scene::Scene;
use glam::Vec3;

pub struct Template {
    pub name: &'static str,
    pub blurb: &'static str,
}

pub const TEMPLATES: &[Template] = &[
    Template { name: "General", blurb: "The default cube, light and camera." },
    Template { name: "Empty", blurb: "Nothing but a camera and the sun." },
    Template { name: "Product Shot", blurb: "Seamless studio backdrop, three-point lighting, a ceramic vase." },
    Template { name: "Turntable", blurb: "The product spinning 360° over 120 frames — render it to an MP4." },
    Template { name: "Landscape", blurb: "Rolling hills with trees, rocks and a low sun." },
    Template { name: "Interior", blurb: "A sunlit room with a wooden floor, a table and a lamp." },
    Template { name: "Physics Playground", blurb: "Dominoes, a block tower and a ball — press Play." },
    Template { name: "Hair & Fur", blurb: "A furry ball, long hair and grass in the studio." },
];

fn cmd(json: &str) -> Command {
    serde_json::from_str(json).unwrap_or_else(|e| panic!("template command {json}: {e}"))
}

thread_local! {
    /// Steps that failed while building (checked by the tests).
    static FAILED: std::cell::RefCell<Vec<String>> = const { std::cell::RefCell::new(vec![]) };
}

fn run(ed: &mut Editor, cmds: &[&str]) {
    let cs: Vec<Command> = cmds.iter().map(|c| cmd(c)).collect();
    for l in ed.execute(&cs) {
        if !l.starts_with('✔') {
            FAILED.with(|f| f.borrow_mut().push(l));
        }
    }
}

/// Revolve a profile of (radius, height) points around Z (smooth, capped where the radius is 0
/// or at the ends).
pub fn lathe(profile: &[(f32, f32)], segments: usize) -> PolyMesh {
    let seg = segments.max(3);
    let mut pm = PolyMesh::default();
    let mut rings: Vec<Option<u32>> = vec![]; // start index per ring, None = a pole vertex
    let mut poles: Vec<u32> = vec![];
    for &(r, z) in profile {
        if r <= 1e-5 {
            pm.verts.push(Vec3::new(0.0, 0.0, z));
            poles.push(pm.verts.len() as u32 - 1);
            rings.push(None);
        } else {
            let start = pm.verts.len() as u32;
            for k in 0..seg {
                let a = k as f32 / seg as f32 * std::f32::consts::TAU;
                pm.verts.push(Vec3::new(r * a.cos(), r * a.sin(), z));
            }
            poles.push(u32::MAX);
            rings.push(Some(start));
        }
    }
    for i in 0..profile.len().saturating_sub(1) {
        let up = profile[i + 1].1 >= profile[i].1;
        match (rings[i], rings[i + 1]) {
            (Some(a), Some(b)) => {
                for k in 0..seg as u32 {
                    let k1 = (k + 1) % seg as u32;
                    let f = vec![a + k, a + k1, b + k1, b + k];
                    pm.faces.push(if up { f } else { f.into_iter().rev().collect() });
                }
            }
            (None, Some(b)) => {
                let p = poles[i];
                for k in 0..seg as u32 {
                    let k1 = (k + 1) % seg as u32;
                    pm.faces.push(if up { vec![p, b + k1, b + k] } else { vec![p, b + k, b + k1] });
                }
            }
            (Some(a), None) => {
                let p = poles[i + 1];
                for k in 0..seg as u32 {
                    let k1 = (k + 1) % seg as u32;
                    pm.faces.push(if up { vec![a + k, a + k1, p] } else { vec![a + k1, a + k, p] });
                }
            }
            (None, None) => {}
        }
    }
    pm
}

/// A ceramic-vase profile (closed at the bottom, open lip).
pub fn vase() -> PolyMesh {
    let prof = [
        (0.0, 0.0),
        (0.32, 0.0),
        (0.42, 0.08),
        (0.55, 0.35),
        (0.6, 0.6),
        (0.52, 0.95),
        (0.3, 1.25),
        (0.22, 1.45),
        (0.26, 1.62),
        (0.3, 1.68),
        (0.26, 1.7),
        (0.2, 1.6),
        (0.16, 1.4),
        (0.0, 1.35),
    ];
    lathe(&prof, 40)
}

/// A seamless studio backdrop ("cyc"): floor in front, a smooth curve up into the back wall.
pub fn cyclorama(width: f32, depth: f32, height: f32, radius: f32) -> PolyMesh {
    let mut prof: Vec<(f32, f32)> = vec![(-depth, 0.0), (-depth * 0.5, 0.0), (-radius, 0.0)];
    let n = 10;
    for i in 1..n {
        let a = i as f32 / n as f32 * std::f32::consts::FRAC_PI_2;
        prof.push((-radius + radius * a.sin(), radius - radius * a.cos()));
    }
    prof.push((0.0, radius));
    prof.push((0.0, height));
    let cols = 12;
    let mut pm = PolyMesh::default();
    for (y, z) in &prof {
        for c in 0..=cols {
            let x = (c as f32 / cols as f32 - 0.5) * width;
            // the profile's y is towards the back (+Y) from the curve
            pm.verts.push(Vec3::new(x, depth * 0.5 + y, *z));
        }
    }
    let w = cols as u32 + 1;
    for r in 0..prof.len() as u32 - 1 {
        for c in 0..cols as u32 {
            let (a, b) = (r * w + c, (r + 1) * w + c);
            pm.faces.push(vec![a, a + 1, b + 1, b]);
        }
    }
    pm
}

fn hash(x: i32, y: i32, s: u32) -> f32 {
    let mut h = (x as u32).wrapping_mul(0x8da6_b343) ^ (y as u32).wrapping_mul(0xd816_3841) ^ s.wrapping_mul(0xcb1a_b31f);
    h = (h ^ (h >> 13)).wrapping_mul(0x5bd1_e995);
    ((h ^ (h >> 15)) & 0x00ff_ffff) as f32 / 16_777_215.0
}

fn vnoise(x: f32, y: f32, s: u32) -> f32 {
    let (xi, yi) = (x.floor() as i32, y.floor() as i32);
    let (fx, fy) = (x - xi as f32, y - yi as f32);
    let (ux, uy) = (fx * fx * (3.0 - 2.0 * fx), fy * fy * (3.0 - 2.0 * fy));
    let a = hash(xi, yi, s) + (hash(xi + 1, yi, s) - hash(xi, yi, s)) * ux;
    let b = hash(xi, yi + 1, s) + (hash(xi + 1, yi + 1, s) - hash(xi, yi + 1, s)) * ux;
    a + (b - a) * uy
}

/// Rolling terrain: a `size`-metre square grid with fractal hills, flatter towards the middle.
pub fn terrain(size: f32, res: usize, height: f32, seed: u32) -> PolyMesh {
    let n = res.max(2);
    let mut pm = PolyMesh::default();
    for j in 0..=n {
        for i in 0..=n {
            let (u, v) = (i as f32 / n as f32 - 0.5, j as f32 / n as f32 - 0.5);
            let (x, y) = (u * size, v * size);
            let mut h = 0.0;
            let (mut amp, mut f) = (1.0, 0.08);
            for o in 0..5 {
                h += (vnoise(x * f, y * f, seed + o) - 0.5) * amp;
                amp *= 0.5;
                f *= 2.1;
            }
            // a gentle bowl: flat-ish centre, hills at the edges
            let r = (u * u + v * v).sqrt() * 2.0;
            let k = 0.35 + 0.65 * r.min(1.0).powf(1.5);
            pm.verts.push(Vec3::new(x, y, h * height * k + r * r * height * 0.35));
        }
    }
    let w = n as u32 + 1;
    for j in 0..n as u32 {
        for i in 0..n as u32 {
            let a = j * w + i;
            pm.faces.push(vec![a, a + 1, a + w + 1, a + w]);
        }
    }
    pm
}

fn set_mesh(ed: &mut Editor, name: &str, pm: PolyMesh, smooth: bool) {
    if let Some(id) = ed.scene.find_by_name(name) {
        let o = ed.scene.get_mut(id).unwrap();
        o.mesh = Some(pm);
        o.smooth_override = Some(smooth);
    }
}

fn studio(ed: &mut Editor) {
    run(ed, &[
        r#"{"action":"add","primitive":"plane","name":"Backdrop"}"#,
        r#"{"action":"set_color","target":"Backdrop","color":[0.82,0.82,0.84],"roughness":0.8}"#,
        r#"{"action":"add","primitive":"spot_light","name":"Key Light","location":[-3.5,-3.5,4.5]}"#,
        r#"{"action":"look_at","target":"Key Light","look_at":[0,0,0.8]}"#,
        r#"{"action":"add","primitive":"point_light","name":"Fill Light","location":[4,-3,2]}"#,
        r#"{"action":"add","primitive":"spot_light","name":"Rim Light","location":[2.5,3.5,3.5]}"#,
        r#"{"action":"look_at","target":"Rim Light","look_at":[0,0,1]}"#,
        r#"{"action":"add","primitive":"camera","name":"Camera","location":[0,-6.2,2.0]}"#,
        r#"{"action":"look_at","target":"Camera","look_at":[0,0,0.85]}"#,
        r#"{"action":"set_background","color":[0.78,0.79,0.82]}"#,
    ]);
    set_mesh(ed, "Backdrop", cyclorama(16.0, 12.0, 7.0, 2.5), true);
    let lights = [("Key Light", 1600.0, [1.0, 0.95, 0.88], 50.0), ("Fill Light", 300.0, [0.85, 0.9, 1.0], 0.0), ("Rim Light", 1200.0, [1.0, 1.0, 1.0], 40.0)];
    for (n, p, c, ang) in lights {
        if let Some(id) = ed.scene.find_by_name(n) {
            if let crate::scene::ObjectKind::Light(l) = &mut ed.scene.get_mut(id).unwrap().kind {
                l.power = p;
                l.color = c;
                if ang > 0.0 {
                    l.spot_angle = ang;
                    l.spot_blend = 0.5;
                }
            }
        }
    }
    if let Some(id) = ed.scene.find_by_name("Camera") {
        if let crate::scene::ObjectKind::Camera(c) = &mut ed.scene.get_mut(id).unwrap().kind {
            c.fov_deg = 28.0;
        }
    }
    ed.scene.world.ambient = 0.35;
}

fn add_vase(ed: &mut Editor, name: &str, at: Vec3) {
    run(ed, &[&format!(r#"{{"action":"add","primitive":"cylinder","name":"{name}","location":[{},{},{}]}}"#, at.x, at.y, at.z)]);
    set_mesh(ed, name, vase(), true);
    if let Some(id) = ed.scene.find_by_name(name) {
        let o = ed.scene.get_mut(id).unwrap();
        o.material.color = [0.12, 0.32, 0.55];
        o.material.roughness = 0.18;
        o.auto_smooth = 0.0;
    }
}

/// Build a template scene by name (case-insensitive). The result is a fresh scene.
pub fn build(name: &str) -> Option<Scene> {
    let low = name.to_lowercase();
    let mut ed = Editor::default();
    if low == "general" || low == "default" {
        return Some(Scene::default());
    }
    run(&mut ed, &[r#"{"action":"clear_scene"}"#]);
    if low == "empty" || low == "blank" {
        run(&mut ed, &[r#"{"action":"add","primitive":"camera","name":"Camera","location":[7.36,-6.93,4.96]}"#, r#"{"action":"look_at","target":"Camera","look_at":[0,0,0]}"#, r#"{"action":"add","primitive":"sun","name":"Sun","location":[0,0,6]}"#]);
    } else if low.contains("product") || low.contains("studio") {
        studio(&mut ed);
        add_vase(&mut ed, "Vase", Vec3::new(-0.15, 0.4, 0.0));
        run(&mut ed, &[
            r#"{"action":"add","primitive":"uv_sphere","name":"Accent Ball","location":[1.25,-0.3,0.3],"scale":0.3}"#,
            r#"{"action":"set_color","target":"Accent Ball","color":[0.95,0.75,0.3],"metallic":1.0,"roughness":0.25}"#,
            r#"{"action":"smooth","target":"Accent Ball"}"#,
            r#"{"action":"add","primitive":"cube","name":"Plinth","location":[-1.4,0.6,0.25],"scale":[0.35,0.35,0.25]}"#,
            r#"{"action":"set_color","target":"Plinth","color":[0.9,0.9,0.88],"roughness":0.6}"#,
        ]);
        ed.scene.output.engine = crate::scene::Engine::PathTraced;
    } else if low.contains("turntable") || low.contains("spin") {
        studio(&mut ed);
        run(&mut ed, &[
            r#"{"action":"add","primitive":"cylinder","name":"Turntable","location":[0,0,0.06],"scale":[1.3,1.3,0.06]}"#,
            r#"{"action":"set_color","target":"Turntable","color":[0.15,0.15,0.16],"roughness":0.35}"#,
            r#"{"action":"smooth","target":"Turntable"}"#,
        ]);
        add_vase(&mut ed, "Vase", Vec3::new(0.0, 0.0, 0.12));
        let (vid, tid) = (ed.scene.find_by_name("Vase").unwrap(), ed.scene.find_by_name("Turntable").unwrap());
        ed.scene.get_mut(vid).unwrap().parent = Some(tid);
        // a key every frame: constant speed, and frame 121 = frame 1 so it loops
        let o = ed.scene.get_mut(tid).unwrap();
        let base = o.transform;
        o.keyframes = (1..=121)
            .map(|f| {
                let mut t = base;
                t.rotation.z = (f - 1) as f32 * 3.0;
                crate::scene::Keyframe { frame: f, transform: t }
            })
            .collect();
        ed.scene.frame_start = 1;
        ed.scene.frame_end = 120;
    } else if low.contains("landscape") || low.contains("terrain") || low.contains("outdoor") {
        run(&mut ed, &[
            r#"{"action":"add","primitive":"plane","name":"Terrain"}"#,
            r#"{"action":"set_color","target":"Terrain","color":[0.32,0.5,0.2],"roughness":0.95}"#,
            r#"{"action":"add","primitive":"sun","name":"Sun","location":[0,0,10]}"#,
            r#"{"action":"add","primitive":"camera","name":"Camera","location":[0,-19,3.2]}"#,
            r#"{"action":"look_at","target":"Camera","look_at":[0,4,1.5]}"#,
        ]);
        set_mesh(&mut ed, "Terrain", terrain(40.0, 80, 5.0, 7), true);
        run(&mut ed, &[
            r#"{"action":"set_texture","target":"Terrain","name":"grass","number":12}"#,
            r#"{"action":"scatter","target":"Terrain","name":"tree","number":70,"seed":4,"slope":28}"#,
            r#"{"action":"scatter","target":"Terrain","name":"rock","number":90,"seed":9}"#,
            r#"{"action":"scatter","target":"Terrain","name":"bush","number":60,"seed":2,"slope":30}"#,
            r#"{"action":"set_background","color":[0.55,0.7,0.9],"vector":[-0.6,0.5,0.35],"number":3.2}"#,
        ]);
        if let Some(id) = ed.scene.find_by_name("Sun") {
            let o = ed.scene.get_mut(id).unwrap();
            o.look_at(Vec3::new(4.0, -3.0, 0.0) + Vec3::new(0.0, 0.0, 10.0) - Vec3::new(-6.0, 5.0, 3.5));
            if let crate::scene::ObjectKind::Light(l) = &mut o.kind {
                l.color = [1.0, 0.88, 0.7];
                l.power = 3.5;
            }
        }
        ed.scene.world.ambient = 0.4;
    } else if low.contains("interior") || low.contains("room") {
        run(&mut ed, &[
            r#"{"action":"add","primitive":"plane","name":"Floor","scale":[4,4,1]}"#,
            r#"{"action":"node_material","target":"Floor","name":"wood floor","number":3}"#,
            r#"{"action":"add","primitive":"plane","name":"Back Wall","location":[0,4,2],"rotation":[90,0,0],"scale":[4,2,1]}"#,
            r#"{"action":"set_color","target":"Back Wall","color":[0.86,0.83,0.78],"roughness":0.9}"#,
            r#"{"action":"add","primitive":"plane","name":"Side Wall","location":[-4,0,2],"rotation":[90,0,90],"scale":[4,2,1]}"#,
            r#"{"action":"set_color","target":"Side Wall","color":[0.62,0.7,0.66],"roughness":0.9}"#,
            r#"{"action":"add","primitive":"cube","name":"Table Top","location":[-0.6,1.2,0.78],"scale":[1.0,0.6,0.04]}"#,
            r#"{"action":"node_material","target":"Table Top","name":"marble"}"#,
            r#"{"action":"add","primitive":"cylinder","name":"Table Leg","location":[-1.45,0.72,0.37],"scale":[0.04,0.04,0.37]}"#,
            r#"{"action":"set_color","target":"last","color":[0.1,0.1,0.1],"metallic":1.0,"roughness":0.3}"#,
            r#"{"action":"add","primitive":"cylinder","name":"Table Leg","location":[0.25,0.72,0.37],"scale":[0.04,0.04,0.37]}"#,
            r#"{"action":"set_color","target":"last","color":[0.1,0.1,0.1],"metallic":1.0,"roughness":0.3}"#,
            r#"{"action":"add","primitive":"cylinder","name":"Table Leg","location":[-1.45,1.68,0.37],"scale":[0.04,0.04,0.37]}"#,
            r#"{"action":"set_color","target":"last","color":[0.1,0.1,0.1],"metallic":1.0,"roughness":0.3}"#,
            r#"{"action":"add","primitive":"cylinder","name":"Table Leg","location":[0.25,1.68,0.37],"scale":[0.04,0.04,0.37]}"#,
            r#"{"action":"set_color","target":"last","color":[0.1,0.1,0.1],"metallic":1.0,"roughness":0.3}"#,
            r#"{"action":"add","primitive":"plane","name":"Rug","location":[0.3,-0.6,0.005],"scale":[1.6,1.1,1]}"#,
            r#"{"action":"set_texture","target":"Rug","name":"stripes","number":4,"color":[0.55,0.25,0.22]}"#,
            r#"{"action":"add","primitive":"uv_sphere","name":"Lamp Shade","location":[-3.2,3.2,1.9],"scale":[0.3,0.3,0.22]}"#,
            r#"{"action":"set_color","target":"Lamp Shade","color":[1.0,0.85,0.6],"emission":3.0}"#,
            r#"{"action":"add","primitive":"cylinder","name":"Lamp Pole","location":[-3.2,3.2,0.85],"scale":[0.025,0.025,0.85]}"#,
            r#"{"action":"set_color","target":"Lamp Pole","color":[0.08,0.08,0.08],"metallic":1.0,"roughness":0.4}"#,
            r#"{"action":"add","primitive":"point_light","name":"Lamp Light","location":[-3.2,3.2,1.6]}"#,
            r#"{"action":"add","primitive":"sun","name":"Sun","location":[4,-2,5]}"#,
            r#"{"action":"add","primitive":"camera","name":"Camera","location":[3.4,-3.6,1.6]}"#,
            r#"{"action":"look_at","target":"Camera","look_at":[-1.2,1.6,0.8]}"#,
            r#"{"action":"set_background","color":[0.75,0.82,0.92],"vector":[0.8,-0.5,0.45],"number":3}"#,
        ]);
        add_vase(&mut ed, "Vase", Vec3::new(-0.9, 1.3, 0.82));
        if let Some(id) = ed.scene.find_by_name("Vase") {
            ed.scene.get_mut(id).unwrap().transform.scale = Vec3::splat(0.22);
        }
        for (n, p, c) in [("Lamp Light", 250.0, [1.0, 0.75, 0.45]), ("Sun", 3.5, [1.0, 0.92, 0.8])] {
            if let Some(id) = ed.scene.find_by_name(n) {
                let o = ed.scene.get_mut(id).unwrap();
                if n == "Sun" {
                    o.look_at(Vec3::new(4.0, -2.0, 5.0) - Vec3::new(0.8, -0.5, 0.45));
                }
                if let crate::scene::ObjectKind::Light(l) = &mut o.kind {
                    l.power = p;
                    l.color = c;
                }
            }
        }
        if let Some(id) = ed.scene.find_by_name("Camera") {
            if let crate::scene::ObjectKind::Camera(c) = &mut ed.scene.get_mut(id).unwrap().kind {
                c.fov_deg = 50.0;
            }
        }
        ed.scene.world.ambient = 0.3;
        ed.scene.output.engine = crate::scene::Engine::PathTraced;
    } else if low.contains("physics") || low.contains("domino") || low.contains("playground") {
        run(&mut ed, &[
            r#"{"action":"add","primitive":"plane","name":"Ground","scale":10}"#,
            r#"{"action":"set_texture","target":"Ground","name":"checker","number":5,"color":[0.75,0.75,0.78]}"#,
            r#"{"action":"add","primitive":"sun","name":"Sun","location":[0,0,8]}"#,
            r#"{"action":"add","primitive":"camera","name":"Camera","location":[6.5,-8.5,4.2]}"#,
            r#"{"action":"look_at","target":"Camera","look_at":[0,0,0.8]}"#,
        ]);
        // a curve of dominoes leading to a tower
        let mut cmds: Vec<String> = vec![];
        let n = 16;
        for i in 0..n {
            let t = i as f32 / (n - 1) as f32;
            let ang = t * 1.6 - 0.8;
            let (x, y) = (-5.0 + t * 6.0, -2.5 + ang.sin() * 2.2);
            let dir = (-0.8f32 + (t + 0.02) * 1.6).sin() * 2.2 - ang.sin() * 2.2;
            let yaw = dir.atan2(6.0 * 0.02).to_degrees();
            let tilt = if i == 0 { 18.0 } else { 0.0 };
            cmds.push(format!(r#"{{"action":"add","primitive":"cube","name":"Domino","location":[{x:.3},{y:.3},0.5],"rotation":[0,{tilt},{yaw:.1}],"scale":[0.06,0.25,0.5]}}"#));
            let hue = [[0.9, 0.25, 0.2], [0.95, 0.75, 0.2], [0.25, 0.55, 0.9]][i % 3];
            cmds.push(format!(r#"{{"action":"set_color","target":"last","color":[{},{},{}],"roughness":0.4}}"#, hue[0], hue[1], hue[2]));
        }
        for (lvl, count) in [(0, 3), (1, 2), (2, 1)] {
            for k in 0..count {
                let x = 2.2 + (k as f32 - (count - 1) as f32 * 0.5) * 0.62;
                cmds.push(format!(r#"{{"action":"add","primitive":"cube","name":"Block","location":[{x:.3},1.2,{:.3}],"scale":0.3}}"#, 0.3 + lvl as f32 * 0.6));
                cmds.push(r#"{"action":"set_color","target":"last","color":[0.85,0.85,0.8],"roughness":0.7}"#.into());
            }
        }
        cmds.push(r#"{"action":"add","primitive":"uv_sphere","name":"Ball","location":[2.0,-1.2,4.5],"scale":0.45}"#.into());
        cmds.push(r#"{"action":"set_color","target":"Ball","color":[0.2,0.75,0.35],"roughness":0.3}"#.into());
        cmds.push(r#"{"action":"smooth","target":"Ball"}"#.into());
        let refs: Vec<&str> = cmds.iter().map(|s| s.as_str()).collect();
        run(&mut ed, &refs);
        let movers: Vec<u64> = ed.scene.objects.iter().filter(|o| o.name.starts_with("Domino") || o.name.starts_with("Block") || o.name == "Ball").map(|o| o.id).collect();
        for id in movers {
            let o = ed.scene.get_mut(id).unwrap();
            o.rigid = Some(crate::physics::RigidBody { bounce: if o.name == "Ball" { 0.55 } else { 0.15 }, friction: 0.6, ..Default::default() });
        }
        if let Some(id) = ed.scene.find_by_name("Ground") {
            ed.scene.get_mut(id).unwrap().rigid = Some(crate::physics::RigidBody { passive: true, ..Default::default() });
        }
        run(&mut ed, &[r#"{"action":"bake_physics"}"#]);
    } else if low.contains("hair") || low.contains("fur") {
        studio(&mut ed);
        run(&mut ed, &[
            r#"{"action":"add","primitive":"uv_sphere","name":"Fuzzball","location":[-1.4,0,0.75],"scale":0.75}"#,
            r#"{"action":"smooth","target":"Fuzzball"}"#,
            r#"{"action":"hair","target":"Fuzzball","name":"fur","color":[0.85,0.55,0.25]}"#,
            r#"{"action":"add","primitive":"uv_sphere","name":"Head","location":[1.0,0.3,1.1],"scale":[0.8,0.85,0.95]}"#,
            r#"{"action":"smooth","target":"Head"}"#,
            r#"{"action":"set_color","target":"Head","color":[0.85,0.65,0.55],"roughness":0.5}"#,
            r#"{"action":"hair","target":"Head","name":"long hair","color":[0.3,0.17,0.08]}"#,
            r#"{"action":"add","primitive":"plane","name":"Lawn","location":[0.2,-1.6,0.01],"scale":[1.2,0.5,1]}"#,
            r#"{"action":"hair","target":"Lawn","name":"grass"}"#,
        ]);
    } else {
        return None;
    }
    ed.scene.update_derived();
    ed.selection.clear();
    Some(ed.scene)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_template_builds() {
        for t in TEMPLATES {
            let s = build(t.name).unwrap_or_else(|| panic!("{}", t.name));
            let failed = FAILED.with(|f| std::mem::take(&mut *f.borrow_mut()));
            assert!(failed.is_empty(), "{}: {failed:?}", t.name);
            if t.name != "Empty" && t.name != "General" {
                assert!(s.objects.iter().filter(|o| o.is_mesh()).count() >= 2, "{} has content", t.name);
            }
            assert!(s.objects.iter().any(|o| matches!(o.kind, crate::scene::ObjectKind::Camera(_))), "{} has a camera", t.name);
            // round-trips through the file format
            let j = serde_json::to_string(&s).unwrap();
            let back: Scene = serde_json::from_str(&j).unwrap();
            assert_eq!(back.objects.len(), s.objects.len());
        }
        assert!(build("nope").is_none());
        // the turntable turns a full circle and loops
        let s = build("turntable").unwrap();
        let t = s.objects.iter().find(|o| o.name == "Turntable").unwrap();
        assert_eq!(t.keyframes.last().unwrap().transform.rotation.z, 360.0);
        // physics is baked
        let s = build("physics playground").unwrap();
        assert!(s.objects.iter().filter(|o| o.rigid.as_ref().is_some_and(|r| !r.cache.is_empty())).count() > 10);
    }

    #[test]
    fn lathe_and_cyc_are_closed_and_sane() {
        let v = vase();
        assert!(v.faces.iter().all(|f| f.iter().all(|&i| (i as usize) < v.verts.len())));
        let c = cyclorama(10.0, 8.0, 5.0, 2.0);
        let zmax = c.verts.iter().map(|p| p.z).fold(f32::MIN, f32::max);
        assert!((zmax - 5.0).abs() < 1e-4);
        // the floor faces up
        let f = &c.faces[0];
        let (a, b, d) = (c.verts[f[0] as usize], c.verts[f[1] as usize], c.verts[f[3] as usize]);
        assert!((b - a).cross(d - a).z > 0.0, "cyc floor normal up");
        let t = terrain(10.0, 8, 2.0, 1);
        assert_eq!(t.verts.len(), 81);
    }
}
