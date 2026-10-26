//! Asset library: objects (with their children, materials, node graphs, textures and whatever
//! they depend on — the armature of a skinned mesh, the surface of a hair system …) and
//! materials saved as `.rbasset` files, plus built-in assets, and small thumbnails for all of
//! them and for the templates (path traced in the background, cached on disk).

use crate::mesh::Primitive;
use crate::scene::{Material, Object, Scene};
use glam::Vec3;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

pub const VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AssetKind {
    Object,
    Material,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Asset {
    pub version: u32,
    pub name: String,
    pub kind: AssetKind,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub objects: Vec<Object>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub material: Option<Material>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub textures: Vec<crate::textures::Texture>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub node_graphs: Vec<crate::nodes::NodeGraph>,
    /// Bottom centre of the saved objects (where they sit when placed).
    #[serde(default)]
    pub pivot: Vec3,
}

fn refs(o: &Object) -> Vec<u64> {
    let mut v = vec![];
    if let Some(s) = &o.skin {
        v.push(s.armature);
    }
    if let Some(s) = &o.scatter {
        v.extend([s.source, s.target]);
    }
    if let Some(h) = &o.hair {
        v.push(h.target);
    }
    if let Some(p) = &o.particles {
        v.extend([p.source, p.emitter]);
    }
    if let Some(a) = &o.armature {
        v.extend(a.bones.iter().filter_map(|b| b.ik.map(|k| k.target)));
    }
    v
}

fn graph_textures(g: &crate::nodes::NodeGraph) -> Vec<u32> {
    g.nodes.iter().filter_map(|n| if let crate::nodes::NodeKind::Image { texture } = n.kind { Some(texture) } else { None }).collect()
}

/// Material + the textures and graph it uses.
fn material_deps(scene: &Scene, m: &Material, textures: &mut Vec<crate::textures::Texture>, graphs: &mut Vec<crate::nodes::NodeGraph>) {
    let mut tex_ids = vec![m.texture, m.normal_tex, m.rough_tex];
    if let Some(g) = scene.graph(m.graph) {
        if !graphs.iter().any(|x| x.id == g.id) {
            graphs.push(g.clone());
        }
        tex_ids.extend(graph_textures(g));
    }
    for t in tex_ids {
        if let Some(t) = scene.texture(t) {
            if !textures.iter().any(|x| x.id == t.id) {
                textures.push(t.clone());
            }
        }
    }
}

/// World-space bottom centre of some objects.
pub fn pivot_of(scene: &Scene, ids: &[u64]) -> Vec3 {
    let (mut lo, mut hi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
    for id in ids {
        let Some(o) = scene.get(*id) else { continue };
        if o.is_mesh() {
            let m = o.transform.matrix();
            let pm = o.posed.as_ref().map(|p| (**p).clone()).unwrap_or_else(|| o.poly());
            for v in &pm.verts {
                let w = m.transform_point3(*v);
                lo = lo.min(w);
                hi = hi.max(w);
            }
        } else {
            lo = lo.min(o.transform.location);
            hi = hi.max(o.transform.location);
        }
    }
    if lo.x > hi.x {
        return Vec3::ZERO;
    }
    Vec3::new((lo.x + hi.x) * 0.5, (lo.y + hi.y) * 0.5, lo.z)
}

/// Package objects (plus their children and everything they depend on) as an asset.
pub fn from_objects(scene: &Scene, roots: &[u64], name: &str) -> Option<Asset> {
    let mut set: Vec<u64> = vec![];
    let mut seen: HashSet<u64> = HashSet::new();
    let mut stack: Vec<u64> = roots.to_vec();
    while let Some(id) = stack.pop() {
        if !seen.insert(id) {
            continue;
        }
        let Some(o) = scene.get(id) else { continue };
        set.push(id);
        stack.extend(refs(o));
        // children, and generated objects that hang off this one (its hair, its scatter)
        for c in &scene.objects {
            let hangs = c.parent == Some(id) || c.hair.as_ref().is_some_and(|h| h.target == id) || c.scatter.as_ref().is_some_and(|s| s.target == id && roots.contains(&id)) || c.skin.as_ref().is_some_and(|s| s.armature == id && roots.contains(&id));
            if hangs {
                stack.push(c.id);
            }
        }
    }
    if set.is_empty() {
        return None;
    }
    // keep scene order
    let objects: Vec<Object> = scene.objects.iter().filter(|o| set.contains(&o.id)).cloned().collect();
    let (mut textures, mut graphs) = (vec![], vec![]);
    for o in &objects {
        material_deps(scene, &o.material, &mut textures, &mut graphs);
    }
    let visible_roots: Vec<u64> = objects.iter().filter(|o| o.visible && o.is_mesh()).map(|o| o.id).collect();
    let pivot = pivot_of(scene, if visible_roots.is_empty() { roots } else { &visible_roots });
    let objects = objects
        .into_iter()
        .map(|mut o| {
            o.posed = None;
            o.posed_key = 0;
            o.collection = 0;
            if o.parent.is_some_and(|p| !set.contains(&p)) {
                o.parent = None;
            }
            if let Some(r) = &mut o.rigid {
                r.cache = Default::default();
            }
            o
        })
        .collect();
    Some(Asset { version: VERSION, name: name.to_string(), kind: AssetKind::Object, objects, material: None, textures, node_graphs: graphs, pivot })
}

pub fn from_material(scene: &Scene, m: &Material, name: &str) -> Asset {
    let (mut textures, mut graphs) = (vec![], vec![]);
    material_deps(scene, m, &mut textures, &mut graphs);
    Asset { version: VERSION, name: name.to_string(), kind: AssetKind::Material, objects: vec![], material: Some(*m), textures, node_graphs: graphs, pivot: Vec3::ZERO }
}

/// Bring an asset's textures and graphs into the scene; returns the material with remapped ids.
fn import_material(scene: &mut Scene, a: &Asset, m: &Material, tex_map: &mut HashMap<u32, u32>, graph_map: &mut HashMap<u32, u32>) -> Material {
    let mut tex = |scene: &mut Scene, id: u32| -> u32 {
        if id == 0 {
            return 0;
        }
        if let Some(n) = tex_map.get(&id) {
            return *n;
        }
        let Some(t) = a.textures.iter().find(|t| t.id == id) else { return 0 };
        let n = scene.add_texture(&t.name, t.source.clone());
        tex_map.insert(id, n);
        n
    };
    let mut out = *m;
    out.texture = tex(scene, m.texture);
    out.normal_tex = tex(scene, m.normal_tex);
    out.rough_tex = tex(scene, m.rough_tex);
    if m.graph != 0 {
        out.graph = match graph_map.get(&m.graph) {
            Some(g) => *g,
            None => match a.node_graphs.iter().find(|g| g.id == m.graph) {
                Some(g) => {
                    let mut g = g.clone();
                    for n in &mut g.nodes {
                        if let crate::nodes::NodeKind::Image { texture } = &mut n.kind {
                            *texture = tex(scene, *texture);
                        }
                    }
                    let id = scene.add_graph(g);
                    graph_map.insert(m.graph, id);
                    id
                }
                None => 0,
            },
        };
    }
    out
}

/// Place an object asset with its bottom centre at `at`; returns the new object ids (in order).
pub fn insert(scene: &mut Scene, a: &Asset, at: Vec3) -> Vec<u64> {
    let (mut tex_map, mut graph_map) = (HashMap::new(), HashMap::new());
    let ids: HashMap<u64, u64> = a.objects.iter().map(|o| (o.id, scene.fresh_id())).collect();
    let map = |id: u64| ids.get(&id).copied().unwrap_or(id);
    let d = at - a.pivot;
    let mut out = vec![];
    for src in &a.objects {
        let mut o = src.clone();
        o.id = map(o.id);
        o.name = scene.unique_name(&src.name);
        o.collection = 0;
        o.material = import_material(scene, a, &src.material, &mut tex_map, &mut graph_map);
        o.parent = o.parent.and_then(|p| ids.get(&p).copied());
        if let Some(s) = &mut o.skin {
            s.armature = map(s.armature);
        }
        if let Some(s) = &mut o.scatter {
            (s.source, s.target) = (map(s.source), map(s.target));
        }
        if let Some(h) = &mut o.hair {
            h.target = map(h.target);
        }
        if let Some(p) = &mut o.particles {
            (p.source, p.emitter) = (map(p.source), map(p.emitter));
        }
        if let Some(arm) = &mut o.armature {
            for b in &mut arm.bones {
                if let Some(k) = &mut b.ik {
                    k.target = map(k.target);
                }
            }
        }
        // transforms are world space (children too)
        o.transform.location += d;
        for k in &mut o.keyframes {
            k.transform.location += d;
        }
        if let Some(r) = &mut o.rigid {
            r.cache = Default::default();
            if let Some(t) = &mut r.origin {
                t.location += d;
            }
        }
        o.posed = None;
        o.posed_key = 0;
        out.push(o.id);
        scene.objects.push(o);
    }
    scene.update_derived();
    out
}

/// Put a material asset on objects.
pub fn apply_material(scene: &mut Scene, a: &Asset, ids: &[u64]) -> usize {
    let Some(m) = a.material else { return 0 };
    let (mut tex_map, mut graph_map) = (HashMap::new(), HashMap::new());
    let m = import_material(scene, a, &m, &mut tex_map, &mut graph_map);
    let mut n = 0;
    for id in ids {
        if let Some(o) = scene.get_mut(*id).filter(|o| o.is_mesh()) {
            o.material = m;
            n += 1;
        }
    }
    n
}

// ---------------------------------------------------------------------------------------------
// The library on disk
// ---------------------------------------------------------------------------------------------

pub fn library_dir() -> PathBuf {
    if let Ok(d) = std::env::var("RB_ASSET_DIR") {
        return PathBuf::from(d);
    }
    dirs::data_dir().or_else(dirs::config_dir).unwrap_or_else(std::env::temp_dir).join("rust_blender").join("assets")
}

pub fn thumb_cache_dir() -> PathBuf {
    dirs::cache_dir().unwrap_or_else(std::env::temp_dir).join("rust_blender").join("thumbs")
}

#[derive(Clone, Debug)]
pub struct Entry {
    pub name: String,
    pub kind: AssetKind,
    /// None = built in
    pub path: Option<PathBuf>,
    /// key for the thumbnail cache
    pub key: String,
}

fn file_stem_safe(name: &str) -> String {
    let s: String = name.chars().map(|c| if c.is_alphanumeric() || c == ' ' || c == '-' || c == '_' { c } else { '_' }).collect();
    let s = s.trim().to_string();
    if s.is_empty() { "Asset".into() } else { s }
}

/// Save to the library (a new file name when the name is taken). Returns the path.
pub fn save(dir: &Path, a: &Asset, thumb: Option<(usize, usize, &[u8])>) -> Result<PathBuf, String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let stem = file_stem_safe(&a.name);
    let mut p = dir.join(format!("{stem}.rbasset"));
    let mut k = 2;
    while p.exists() {
        p = dir.join(format!("{stem} {k}.rbasset"));
        k += 1;
    }
    let json = serde_json::to_string(a).map_err(|e| e.to_string())?;
    std::fs::write(&p, json).map_err(|e| format!("{}: {e}", p.display()))?;
    if let Some((w, h, rgba)) = thumb {
        if let Ok(png) = crate::textures::encode_png(w, h, rgba) {
            let _ = std::fs::write(p.with_extension("png"), png);
        }
    }
    Ok(p)
}

pub fn load(p: &Path) -> Result<Asset, String> {
    let s = std::fs::read_to_string(p).map_err(|e| format!("{}: {e}", p.display()))?;
    let a: Asset = serde_json::from_str(&s).map_err(|e| format!("{}: {e}", p.display()))?;
    if a.version > VERSION {
        return Err(format!("{} was saved by a newer Rust Blender", p.display()));
    }
    Ok(a)
}

/// The user's assets (sorted by name).
pub fn list(dir: &Path) -> Vec<Entry> {
    let mut out = vec![];
    let Ok(rd) = std::fs::read_dir(dir) else { return out };
    for e in rd.flatten() {
        let p = e.path();
        if p.extension().and_then(|x| x.to_str()) != Some("rbasset") {
            continue;
        }
        // only the header fields are needed
        #[derive(Deserialize)]
        struct Head {
            name: String,
            kind: AssetKind,
        }
        let Ok(s) = std::fs::read_to_string(&p) else { continue };
        let Ok(h) = serde_json::from_str::<Head>(&s) else { continue };
        let mtime = e.metadata().and_then(|m| m.modified()).ok().and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map(|d| d.as_secs()).unwrap_or(0);
        out.push(Entry { key: format!("user:{}:{mtime}", p.display()), name: h.name, kind: h.kind, path: Some(p) });
    }
    out.sort_by(|a, b| a.name.to_lowercase().cmp(&b.name.to_lowercase()));
    out
}

pub fn delete(p: &Path) -> Result<(), String> {
    std::fs::remove_file(p).map_err(|e| format!("{}: {e}", p.display()))?;
    let _ = std::fs::remove_file(p.with_extension("png"));
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// Built-in assets
// ---------------------------------------------------------------------------------------------

pub const BUILTIN_OBJECTS: &[&str] = &["Character", "Character with Hair", "Vase", "Rock", "Pine Tree", "Bush", "Flower", "Table", "Chair", "Furry Ball", "Domino", "Lamp"];

pub const BUILTIN_MATERIALS: &[&str] = &[
    "Gold", "Chrome", "Copper", "Glass", "Red Plastic", "Rubber", "Ceramic", "Neon Glow", "Rusty Metal", "Wood", "Wood Floor", "Marble", "Cobblestone", "Lava", "Bricks", "Tiles", "Granite",
];

pub fn builtin_entries() -> Vec<Entry> {
    let mut v: Vec<Entry> = BUILTIN_OBJECTS.iter().map(|n| Entry { name: n.to_string(), kind: AssetKind::Object, path: None, key: format!("builtin-obj:{n}:v1") }).collect();
    v.extend(BUILTIN_MATERIALS.iter().map(|n| Entry { name: n.to_string(), kind: AssetKind::Material, path: None, key: format!("builtin-mat:{n}:v2") }));
    v
}

fn cmd(json: &str) -> crate::editor::Command {
    serde_json::from_str(json).expect("builtin asset command")
}

/// Build a built-in asset by name.
pub fn builtin(name: &str) -> Option<Asset> {
    let low = name.to_lowercase();
    let mut ed = crate::editor::Editor::default();
    ed.execute(&[cmd(r#"{"action":"clear_scene"}"#)]);
    if let Some(m) = BUILTIN_MATERIALS.iter().any(|n| n.to_lowercase() == low).then(|| builtin_material(&low)).flatten() {
        let mut a = from_material(&Scene::default(), &m, name);
        if let Some(g) = crate::nodes::NodeGraph::preset(1, &low).filter(|_| m.graph != 0) {
            a.node_graphs = vec![g];
            if let Some(mm) = &mut a.material {
                mm.graph = 1;
            }
        }
        a.name = name.to_string();
        return Some(a);
    }
    let run = |ed: &mut crate::editor::Editor, cs: &[&str]| {
        let c: Vec<_> = cs.iter().map(|s| cmd(s)).collect();
        ed.execute(&c);
    };
    let roots: Vec<u64> = match low.as_str() {
        "character" | "character with hair" => {
            let (arm, body) = crate::character::add(&mut ed.scene, "Character", Vec3::ZERO, [0.78, 0.6, 0.48]);
            if low.contains("hair") {
                crate::character::add_hair(&mut ed.scene, body, [0.25, 0.14, 0.07]);
            }
            vec![arm]
        }
        "vase" => {
            let id = ed.scene.add(Primitive::Cylinder, Some("Vase"));
            let o = ed.scene.get_mut(id).unwrap();
            o.mesh = Some(crate::templates::vase());
            o.smooth_override = Some(true);
            o.material.color = [0.12, 0.32, 0.55];
            o.material.roughness = 0.18;
            vec![id]
        }
        "rock" | "pine tree" | "bush" | "flower" => {
            let spec = if low == "pine tree" { "tree" } else { low.as_str() };
            let (id, _) = ed.scatter_preset_pub(spec).ok()?;
            let o = ed.scene.get_mut(id).unwrap();
            o.visible = true;
            if low == "pine tree" {
                o.name = "Pine Tree".into();
            }
            vec![id]
        }
        "table" => {
            run(&mut ed, &[
                r#"{"action":"add","primitive":"cube","name":"Table","location":[0,0,0.76],"scale":[0.8,0.5,0.03]}"#,
                r#"{"action":"node_material","target":"Table","name":"wood"}"#,
                r#"{"action":"add","primitive":"cylinder","name":"Table Leg","location":[-0.7,-0.4,0.37],"scale":[0.035,0.035,0.37]}"#,
                r#"{"action":"set_color","target":"Table Leg","color":[0.3,0.18,0.1],"roughness":0.6}"#,
                r#"{"action":"add_modifier","target":"Table Leg","modifier":"array","number":2,"vector":[1.4,0,0]}"#,
                r#"{"action":"add_modifier","target":"Table Leg","modifier":"array","number":2,"vector":[0,0.8,0]}"#,
            ]);
            let (t, l) = (ed.scene.find_by_name("Table")?, ed.scene.find_by_name("Table Leg")?);
            ed.scene.get_mut(l).unwrap().parent = Some(t);
            vec![t]
        }
        "chair" => {
            run(&mut ed, &[
                r#"{"action":"add","primitive":"cube","name":"Chair","location":[0,0,0.45],"scale":[0.22,0.22,0.025]}"#,
                r#"{"action":"set_color","target":"Chair","color":[0.75,0.3,0.2],"roughness":0.5}"#,
                r#"{"action":"add","primitive":"cube","name":"Chair Back","location":[0,0.21,0.72],"scale":[0.22,0.02,0.25]}"#,
                r#"{"action":"set_color","target":"Chair Back","color":[0.75,0.3,0.2],"roughness":0.5}"#,
                r#"{"action":"add","primitive":"cylinder","name":"Chair Leg","location":[-0.19,-0.19,0.22],"scale":[0.02,0.02,0.22]}"#,
                r#"{"action":"set_color","target":"Chair Leg","color":[0.1,0.1,0.1],"metallic":1.0,"roughness":0.35}"#,
                r#"{"action":"add_modifier","target":"Chair Leg","modifier":"array","number":2,"vector":[0.38,0,0]}"#,
                r#"{"action":"add_modifier","target":"Chair Leg","modifier":"array","number":2,"vector":[0,0.38,0]}"#,
            ]);
            let c = ed.scene.find_by_name("Chair")?;
            for n in ["Chair Back", "Chair Leg"] {
                let id = ed.scene.find_by_name(n)?;
                ed.scene.get_mut(id).unwrap().parent = Some(c);
            }
            vec![c]
        }
        "furry ball" => {
            run(&mut ed, &[
                r#"{"action":"add","primitive":"uv_sphere","name":"Furry Ball","location":[0,0,0.5],"scale":0.5}"#,
                r#"{"action":"smooth","target":"Furry Ball"}"#,
                r#"{"action":"hair","target":"Furry Ball","name":"fur","color":[0.9,0.6,0.3]}"#,
            ]);
            vec![ed.scene.find_by_name("Furry Ball")?]
        }
        "domino" => {
            run(&mut ed, &[r#"{"action":"add","primitive":"cube","name":"Domino","location":[0,0,0.5],"scale":[0.06,0.25,0.5]}"#, r#"{"action":"set_color","target":"Domino","color":[0.95,0.95,0.92],"roughness":0.3}"#]);
            vec![ed.scene.find_by_name("Domino")?]
        }
        "lamp" => {
            run(&mut ed, &[
                r#"{"action":"add","primitive":"cylinder","name":"Lamp","location":[0,0,0.02],"scale":[0.15,0.15,0.02]}"#,
                r#"{"action":"set_color","target":"Lamp","color":[0.08,0.08,0.08],"metallic":1.0,"roughness":0.4}"#,
                r#"{"action":"add","primitive":"cylinder","name":"Lamp Pole","location":[0,0,0.8],"scale":[0.015,0.015,0.78]}"#,
                r#"{"action":"set_color","target":"Lamp Pole","color":[0.08,0.08,0.08],"metallic":1.0,"roughness":0.4}"#,
                r#"{"action":"add","primitive":"cone","name":"Lamp Shade","location":[0,0,1.6],"scale":[0.25,0.25,0.18]}"#,
                r#"{"action":"set_color","target":"Lamp Shade","color":[1.0,0.85,0.6],"emission":2.5}"#,
                r#"{"action":"add","primitive":"point_light","name":"Lamp Light","location":[0,0,1.5]}"#,
            ]);
            let l = ed.scene.find_by_name("Lamp")?;
            for n in ["Lamp Pole", "Lamp Shade", "Lamp Light"] {
                let id = ed.scene.find_by_name(n)?;
                let o = ed.scene.get_mut(id).unwrap();
                o.parent = Some(l);
                if let crate::scene::ObjectKind::Light(li) = &mut o.kind {
                    li.power = 150.0;
                    li.color = [1.0, 0.8, 0.55];
                }
            }
            vec![l]
        }
        _ => return None,
    };
    ed.scene.update_derived();
    let mut a = from_objects(&ed.scene, &roots, name)?;
    a.name = name.to_string();
    Some(a)
}

fn builtin_material(low: &str) -> Option<Material> {
    let m = |c: [f32; 3], r: f32, me: f32| Material { color: c, roughness: r, metallic: me, ..Default::default() };
    Some(match low {
        "gold" => m([1.0, 0.77, 0.34], 0.2, 1.0),
        "chrome" => m([0.95, 0.95, 0.96], 0.05, 1.0),
        "copper" => m([0.95, 0.55, 0.38], 0.3, 1.0),
        "glass" => Material { alpha: 0.1, ..m([0.92, 0.97, 1.0], 0.02, 0.0) },
        "red plastic" => m([0.8, 0.08, 0.06], 0.3, 0.0),
        "rubber" => m([0.06, 0.06, 0.06], 0.9, 0.0),
        "ceramic" => m([0.92, 0.92, 0.88], 0.12, 0.0),
        "neon glow" => Material { emission: 4.0, ..m([0.2, 0.8, 1.0], 0.5, 0.0) },
        _ => {
            let g = crate::nodes::NodeGraph::preset(1, low)?;
            let _ = g;
            Material { graph: 1, ..Material::default() }
        }
    })
}

// ---------------------------------------------------------------------------------------------
// Thumbnails
// ---------------------------------------------------------------------------------------------

/// What to draw a thumbnail of.
pub enum ThumbSource {
    Asset(Asset),
    /// A whole scene through its camera (templates).
    Scene(Scene),
}

/// RGBA8 thumbnail. Objects: framed from the front-left in a soft studio; materials: the lit
/// ball; scenes: through their camera.
pub fn render_thumb(src: &ThumbSource, w: usize, h: usize, samples: u32) -> Vec<u8> {
    use crate::viewport::Camera;
    match src {
        ThumbSource::Asset(a) if a.kind == AssetKind::Material => {
            let mut s = Scene::default();
            s.objects.clear();
            let mut tm = HashMap::new();
            let mut gm = HashMap::new();
            let m = import_material(&mut s, a, &a.material.unwrap_or_default(), &mut tm, &mut gm);
            let mut tc = crate::textures::TextureCache::default();
            let imgs: HashMap<u32, std::sync::Arc<crate::textures::TexImage>> = s.textures.iter().filter_map(|t| tc.get(t).map(|i| (t.id, i))).collect();
            let smp = move |id: u32, q: glam::Vec2| imgs.get(&id).map(|i| i.sample(q.x, q.y));
            let n = w.min(h);
            let ball = crate::nodes::preview_ball(s.graph(m.graph), m, n, &smp);
            // centre the square ball on a dark tile
            let mut out = vec![0u8; w * h * 4];
            for px in out.chunks_mut(4) {
                px.copy_from_slice(&[38, 38, 40, 255]);
            }
            let (ox, oy) = ((w - n) / 2, (h - n) / 2);
            for y in 0..n {
                for x in 0..n {
                    let s = &ball[(y * n + x) * 4..(y * n + x) * 4 + 4];
                    let a = s[3] as f32 / 255.0;
                    let d = ((oy + y) * w + ox + x) * 4;
                    for c in 0..3 {
                        out[d + c] = (s[c] as f32 * a + out[d + c] as f32 * (1.0 - a)).round() as u8;
                    }
                }
            }
            out
        }
        _ => {
            let (scene, cam) = match src {
                ThumbSource::Scene(s) => {
                    let mut s = s.clone();
                    s.update_derived();
                    let cam = s.objects.iter().find(|o| matches!(o.kind, crate::scene::ObjectKind::Camera(_))).map(|o| crate::viewport::Viewport::camera_for_render(o, w as f32 / h as f32));
                    (s, cam.unwrap_or_default())
                }
                ThumbSource::Asset(a) => {
                    let mut s = Scene::default();
                    s.objects.clear();
                    let ids = insert(&mut s, a, Vec3::ZERO);
                    // frame the visible meshes
                    let (mut lo, mut hi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
                    let mut cache = crate::scene::MeshCache::default();
                    for o in s.objects.iter().filter(|o| o.visible && o.is_mesh() && ids.contains(&o.id)) {
                        let m = cache.for_object(o);
                        for inst in o.instance_matrices() {
                            let mm = o.transform.matrix() * inst;
                            for p in &m.positions {
                                let q = mm.transform_point3(*p);
                                lo = lo.min(q);
                                hi = hi.max(q);
                            }
                        }
                    }
                    if lo.x > hi.x {
                        (lo, hi) = (Vec3::splat(-1.0), Vec3::ONE);
                    }
                    let c = (lo + hi) * 0.5;
                    let r = ((hi - lo).length() * 0.5).max(0.05);
                    let fov: f32 = 30.0;
                    let dist = r / (fov.to_radians() * 0.5).sin() * 1.05;
                    // a soft key light from the camera side
                    if !s.objects.iter().any(|o| o.light().is_some()) {
                        let l = s.add_light(crate::scene::LightType::Point, Some("Thumb Key"));
                        let o = s.get_mut(l).unwrap();
                        o.transform.location = c + Vec3::new(-1.2, -1.6, 1.8) * dist * 0.6;
                        if let crate::scene::ObjectKind::Light(li) = &mut o.kind {
                            li.power = 220.0 * (dist * 0.6).powi(2).max(0.1);
                        }
                    }
                    s.world.background = [0.17, 0.17, 0.18];
                    s.world.sun_strength = 1.2;
                    (s, Camera { target: c, yaw: 32.0, pitch: 18.0, dist, ortho: false, fov_deg: fov })
                }
            };
            let mut cache = crate::scene::MeshCache::default();
            let mut tex = crate::textures::TextureCache::default();
            let pt = crate::pathtrace::PtScene::gather(&scene, &mut cache, &mut tex, cam);
            crate::pathtrace::render_blocking(&pt, w, h, samples, true)
        }
    }
}

/// A cached thumbnail (PNG in the thumbnail cache), or None.
pub fn cached_thumb(key: &str) -> Option<(usize, usize, Vec<u8>)> {
    let p = thumb_cache_dir().join(format!("{}.png", key_file(key)));
    let bytes = std::fs::read(p).ok()?;
    let img = crate::textures::decode(&crate::textures::TexSource::Image { data: bytes }).ok()?;
    Some((img.w, img.h, img.rgba.into_iter().flatten().collect()))
}

pub fn store_thumb(key: &str, w: usize, h: usize, rgba: &[u8]) {
    let dir = thumb_cache_dir();
    let _ = std::fs::create_dir_all(&dir);
    if let Ok(png) = crate::textures::encode_png(w, h, rgba) {
        let _ = std::fs::write(dir.join(format!("{}.png", key_file(key))), png);
    }
}

fn key_file(key: &str) -> String {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    key.hash(&mut h);
    format!("{:016x}", h.finish())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn object_assets_round_trip_with_dependencies() {
        let mut ed = crate::editor::Editor::default();
        let c = |s: &str| serde_json::from_str::<crate::editor::Command>(s).unwrap();
        ed.execute(&[
            c(r#"{"action":"add","primitive":"uv_sphere","name":"Ball","location":[3,0,1]}"#),
            c(r#"{"action":"node_material","target":"Ball","name":"granite"}"#),
            c(r#"{"action":"hair","target":"Ball","name":"fur","number":30}"#),
            c(r#"{"action":"add","primitive":"cube","name":"Hat","location":[3,0,2.2],"scale":0.3}"#),
            c(r#"{"action":"set_texture","target":"Hat","name":"checker"}"#),
        ]);
        let ball = ed.scene.find_by_name("Ball").unwrap();
        let hat = ed.scene.find_by_name("Hat").unwrap();
        ed.scene.get_mut(hat).unwrap().parent = Some(ball);
        ed.scene.update_derived();
        let a = from_objects(&ed.scene, &[ball], "Fuzzy").unwrap();
        // ball + its hair + the hat (child)
        assert_eq!(a.objects.len(), 3, "{:?}", a.objects.iter().map(|o| &o.name).collect::<Vec<_>>());
        assert_eq!(a.node_graphs.len(), 2, "granite + hair strands");
        assert_eq!(a.textures.len(), 1);
        assert!((a.pivot - Vec3::new(3.0, 0.0, 0.0)).length() < 0.2, "{}", a.pivot);
        // through a file
        let dir = std::env::temp_dir().join(format!("rb_assets_{}", std::process::id()));
        let p = save(&dir, &a, None).unwrap();
        let p2 = save(&dir, &a, None).unwrap();
        assert_ne!(p, p2, "no overwrite");
        let l = list(&dir);
        assert_eq!(l.len(), 2);
        let b = load(&p).unwrap();
        // into another scene
        let mut s = Scene::default();
        let n0 = s.objects.len();
        let ids = insert(&mut s, &b, Vec3::new(-5.0, 2.0, 0.0));
        assert_eq!(ids.len(), 3);
        assert_eq!(s.objects.len(), n0 + 3);
        let nb = s.objects.iter().find(|o| o.name == "Ball").unwrap();
        let nh = s.objects.iter().find(|o| o.name == "Hat").unwrap();
        let hair = s.objects.iter().find(|o| o.hair.is_some()).unwrap();
        assert_eq!(nh.parent, Some(nb.id));
        assert_eq!(hair.hair.as_ref().unwrap().target, nb.id);
        assert!(s.graph(nb.material.graph).is_some() && s.texture(nh.material.texture).is_some());
        assert!((nb.transform.location - Vec3::new(-5.0, 2.0, 1.0)).length() < 0.2, "{}", nb.transform.location);
        assert!(hair.posed.as_ref().is_some_and(|p| !p.faces.is_empty()), "hair regrown");
        // inserting twice gives unique names and ids
        let ids2 = insert(&mut s, &b, Vec3::ZERO);
        assert!(ids2.iter().all(|i| !ids.contains(i)));
        assert!(s.objects.iter().any(|o| o.name == "Ball.001"));
        for e in l {
            delete(e.path.as_ref().unwrap()).unwrap();
        }
        assert!(list(&dir).is_empty());
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn builtins_build_and_material_assets_apply() {
        for e in builtin_entries() {
            let a = builtin(&e.name).unwrap_or_else(|| panic!("{}", e.name));
            assert_eq!(a.kind, e.kind, "{}", e.name);
            match a.kind {
                AssetKind::Object => assert!(!a.objects.is_empty(), "{}", e.name),
                AssetKind::Material => assert!(a.material.is_some(), "{}", e.name),
            }
        }
        let mut s = Scene::default();
        let cube = s.objects[0].id;
        let wood = builtin("Wood").unwrap();
        assert_eq!(apply_material(&mut s, &wood, &[cube]), 1);
        assert!(s.graph(s.get(cube).unwrap().material.graph).is_some_and(|g| g.name == "Wood"));
        let gold = builtin("Gold").unwrap();
        apply_material(&mut s, &gold, &[cube]);
        assert_eq!(s.get(cube).unwrap().material.metallic, 1.0);
        // thumbnails render
        let t = render_thumb(&ThumbSource::Asset(builtin("Vase").unwrap()), 32, 32, 2);
        assert_eq!(t.len(), 32 * 32 * 4);
        let centre = &t[(16 * 32 + 16) * 4..(16 * 32 + 16) * 4 + 3];
        let corner = &t[0..3];
        assert!(centre != corner, "the vase is in frame");
        let m = render_thumb(&ThumbSource::Asset(wood), 40, 30, 1);
        assert_eq!(m.len(), 40 * 30 * 4);
    }
}
