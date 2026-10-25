//! Ollama integration (local LLM) + an offline rule-based command parser fallback.

use crate::editor::Command;
use crossbeam_channel::{Receiver, Sender, unbounded};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

/// Models ranked by how well they follow JSON-schema structured output / tool-style
/// instructions at a size that runs on a typical workstation GPU.
/// `qwen3` is the default pick: excellent structured-output + tool-calling and spatial
/// reasoning for its size; 8B fits in ~6 GB VRAM. Smaller fallbacks follow.
pub const PREFERRED_MODELS: &[&str] = &[
    "qwen3:8b",
    "qwen3:14b",
    "qwen3",
    "qwen3:4b",
    "qwen2.5:7b",
    "qwen2.5",
    "qwen2.5-coder",
    "llama3.1:8b",
    "llama3.1",
    "mistral-nemo",
    "gemma3",
    "llama3.2",
    "phi4",
    "mistral",
];
pub const RECOMMENDED_MODEL: &str = "qwen3:8b";
pub const LIGHTWEIGHT_MODEL: &str = "qwen3:4b";

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ChatMsg {
    pub role: String,
    pub content: String,
}

/// Give up only if Ollama sends nothing at all for this long.
pub const STALL_SECS: u64 = 300;

pub enum AiRequest {
    ListModels,
    /// Ask Ollama how much of a loaded model sits in GPU memory (`/api/ps`).
    Ps(String),
    Chat {
        model: String,
        /// Optional fast decision model consulted first (see `router.rs`).
        router: Option<String>,
        history: Vec<ChatMsg>,
        scene: String,
        prompt: String,
        /// The prompt came from speech recognition (may contain mishearings; answers should be short).
        voice: bool,
    },
    Pull(String),
    /// Agent loop check stage: review flagged issues with `model` and propose fixes.
    Review { model: String, prompt: String, round: u32 },
    /// Ask a vision model to describe a picture as a buildable 3D scene.
    Describe { model: String, image_b64: String, request: String },
}

#[derive(Debug)]
pub enum AiEvent {
    Models(Result<Vec<String>, String>),
    /// Streaming progress: reply text so far + commands completed since the last delta.
    Delta {
        reply: String,
        commands: Vec<Command>,
    },
    /// End of a streamed answer. `commands` are any not already sent via `Delta`.
    Reply {
        reply: String,
        commands: Vec<Command>,
        raw: String,
        secs: f32,
    },
    Error(String),
    /// The decision model chose a route (shown in the chat before the builder answers).
    Routed { router: String, route: String, detail: String, secs: f32 },
    /// Result of a review round.
    Reviewed { model: String, reply: String, commands: Vec<Command>, round: u32, secs: f32, error: Option<String> },
    /// The decision model failed; the builder handles the request instead.
    RouterFailed(String),
    /// (model, fraction of the model in VRAM)
    Ps(String, f32),
    PullProgress(String, f32),
    PullDone(Result<String, String>),
    /// Vision model description of an attached picture.
    Described { model: String, text: Result<String, String>, secs: f32 },
}

pub struct AiClient {
    tx: Sender<AiRequest>,
    pub rx: Receiver<AiEvent>,
    pub cancel: Arc<AtomicBool>,
}

impl AiClient {
    pub fn new(host: String, repaint: impl Fn() + Send + Clone + 'static) -> Self {
        let (tx, req_rx) = unbounded::<AiRequest>();
        let (ev_tx, rx) = unbounded::<AiEvent>();
        let cancel = Arc::new(AtomicBool::new(false));
        let cancel2 = cancel.clone();
        std::thread::Builder::new()
            .name("ollama".into())
            .spawn(move || {
                // No overall deadline: local models can take minutes (cold load, CPU inference,
                // multi-GB pulls). Instead fail only if Ollama goes completely silent.
                let agent: ureq::Agent = ureq::Agent::config_builder()
                    .timeout_connect(Some(Duration::from_secs(3)))
                    .timeout_per_read(Some(Duration::from_secs(STALL_SECS)))
                    .http_status_as_error(false)
                    .build()
                    .into();
                let mut tool_caps: std::collections::HashMap<String, bool> = Default::default();
                let mut decide_modes: std::collections::HashMap<String, Option<String>> = Default::default();
                for req in req_rx {
                    match req {
                        AiRequest::ListModels => {
                            let _ = ev_tx.send(AiEvent::Models(list_models(&agent, &host)));
                        }
                        AiRequest::Ps(model) => {
                            if let Some(f) = vram_fraction(&agent, &host, &model) {
                                let _ = ev_tx.send(AiEvent::Ps(model, f));
                            }
                        }
                        AiRequest::Chat {
                            model,
                            router,
                            history,
                            scene,
                            prompt,
                            voice,
                        } => {
                            let t0 = std::time::Instant::now();
                            cancel2.store(false, Ordering::Relaxed);
                            let mut prompt = if voice {
                                format!("{prompt}\n\n(Spoken request, transcribed by speech recognition: silently fix obvious mishearings of 3D terms. The reply is read aloud, so keep it to one short sentence.)")
                            } else {
                                prompt
                            };
                            // ---- stage 1: decision model ----
                            if let Some(rm) = router.filter(|r| *r != model) {
                                let tool_mode = *tool_caps.entry(rm.clone()).or_insert_with(|| crate::router::capabilities(&agent, &host, &rm).iter().any(|c| c == "tools" || c == "decision"));
                                match crate::router::decide(&agent, &host, &rm, &history, &scene, &prompt, tool_mode, decide_modes.entry(rm.clone()).or_default()) {
                                    Ok((route, raw)) => {
                                        let secs = t0.elapsed().as_secs_f32();
                                        match route {
                                            crate::router::Route::Direct { reply, commands } => {
                                                let raw = json!({"reply": reply, "commands": commands}).to_string();
                                                let _ = ev_tx.send(AiEvent::Routed { router: rm.clone(), route: "direct".into(), detail: String::new(), secs });
                                                let _ = ev_tx.send(AiEvent::Reply { reply, commands, raw, secs });
                                                repaint();
                                                continue;
                                            }
                                            crate::router::Route::Answer(text) => {
                                                let _ = ev_tx.send(AiEvent::Routed { router: rm.clone(), route: "answer".into(), detail: String::new(), secs });
                                                let _ = ev_tx.send(AiEvent::Reply { reply: text.clone(), commands: vec![], raw: json!({"reply": text, "commands": []}).to_string(), secs });
                                                repaint();
                                                continue;
                                            }
                                            crate::router::Route::Delegate { plan, .. } => {
                                                let _ = ev_tx.send(AiEvent::Routed { router: rm.clone(), route: "delegate".into(), detail: plan.clone(), secs });
                                                repaint();
                                                if !plan.trim().is_empty() {
                                                    prompt = format!("{prompt}\n\n(Plan from the decision model — follow it if it helps: {plan})");
                                                }
                                                let _ = raw;
                                            }
                                        }
                                    }
                                    Err(e) => {
                                        let _ = ev_tx.send(AiEvent::RouterFailed(format!("{rm}: {e}")));
                                        repaint();
                                    }
                                }
                                if cancel2.load(Ordering::Relaxed) {
                                    let _ = ev_tx.send(AiEvent::Error("cancelled".into()));
                                    continue;
                                }
                            }
                            // ---- stage 2: builder model ----
                            let ev = match chat(
                                &agent, &host, &model, &history, &scene, &prompt, &ev_tx, &repaint,
                                &cancel2,
                            ) {
                                Ok((reply, commands, raw)) => AiEvent::Reply {
                                    reply,
                                    commands,
                                    raw,
                                    secs: t0.elapsed().as_secs_f32(),
                                },
                                Err(e) => AiEvent::Error(friendly_error(&e, &model)),
                            };
                            let _ = ev_tx.send(ev);
                        }
                        AiRequest::Review { model, prompt, round } => {
                            let t0 = std::time::Instant::now();
                            let (reply, commands, error) = match crate::router::review(&agent, &host, &model, &prompt) {
                                Ok((r, c)) => (r, c, None),
                                Err(e) => (String::new(), vec![], Some(e)),
                            };
                            let _ = ev_tx.send(AiEvent::Reviewed { model, reply, commands, round, secs: t0.elapsed().as_secs_f32(), error });
                        }
                        AiRequest::Describe { model, image_b64, request } => {
                            let t0 = std::time::Instant::now();
                            let text = describe_image(&agent, &host, &model, &image_b64, &request);
                            let _ = ev_tx.send(AiEvent::Described { model, text, secs: t0.elapsed().as_secs_f32() });
                        }
                        AiRequest::Pull(model) => {
                            let r = pull(&agent, &host, &model, &ev_tx, &repaint);
                            let _ = ev_tx.send(AiEvent::PullDone(r.map(|_| model)));
                        }
                    }
                    repaint();
                }
            })
            .expect("spawn ollama thread");
        Self { tx, rx, cancel }
    }

    pub fn send(&self, r: AiRequest) {
        let _ = self.tx.send(r);
    }
}

fn friendly_error(e: &str, model: &str) -> String {
    let l = e.to_lowercase();
    if l.contains("timeout") || l.contains("timed out") {
        format!(
            "Ollama stopped responding for {} min while running {model}. This usually means the model doesn't fit in your GPU's memory and is running (very slowly) on the CPU. Try a smaller model such as {LIGHTWEIGHT_MODEL} (Copilot ⚙ → Pull), and run `ollama ps` in a terminal to check the CPU/GPU split. (details: {e})",
            STALL_SECS / 60
        )
    } else if l.contains("connection refused") || l.contains("connect") && l.contains("failed") {
        format!("Can't reach Ollama — is it running? Start the Ollama app or run `ollama serve`. (details: {e})")
    } else {
        e.to_string()
    }
}

/// Fraction of a loaded model that lives in VRAM, from `/api/ps`.
fn vram_fraction(agent: &ureq::Agent, host: &str, model: &str) -> Option<f32> {
    let mut resp = agent.get(&format!("{host}/api/ps")).call().ok()?;
    let v: Value = resp.body_mut().read_json().ok()?;
    let m = v["models"].as_array()?.iter().find(|m| m["name"].as_str() == Some(model) || m["model"].as_str() == Some(model))?;
    let size = m["size"].as_f64()?;
    let vram = m["size_vram"].as_f64().unwrap_or(0.0);
    (size > 0.0).then(|| (vram / size) as f32)
}

fn list_models(agent: &ureq::Agent, host: &str) -> Result<Vec<String>, String> {
    let mut resp = agent
        .get(&format!("{host}/api/tags"))
        .call()
        .map_err(|e| format!("Ollama not reachable at {host} ({e}). Is `ollama serve` running?"))?;
    let v: Value = resp.body_mut().read_json().map_err(|e| e.to_string())?;
    Ok(v["models"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|m| m["name"].as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default())
}

/// Pick the best installed model according to `PREFERRED_MODELS`.
/// Vision-capable models, best first (all run in Ollama).
pub const VISION_MODELS: &[&str] = &["qwen2.5vl", "qwen3-vl", "gemma3", "llama3.2-vision", "minicpm-v", "llava-llama3", "llava", "moondream"];
pub const RECOMMENDED_VISION: &str = "qwen2.5vl:7b";

pub fn pick_vision_model(installed: &[String]) -> Option<String> {
    for pref in VISION_MODELS {
        // gemma3:1b has no vision
        if let Some(m) = installed.iter().find(|m| (m.as_str() == *pref || m.starts_with(&format!("{pref}:"))) && !m.starts_with("gemma3:1b") && !m.starts_with("gemma3:270m")) {
            return Some(m.clone());
        }
    }
    None
}

const DESCRIBE_PROMPT: &str = "You are helping rebuild this picture as a simple 3D scene made of primitives \
(cube, sphere, cylinder, cone, torus, plane) in a Z-up world measured in metres, standing on the ground (z = 0). \
List the main objects (at most 15). For each give: name, the primitive(s) that approximate it, approximate size \
in metres, position relative to the scene centre (x = right, y = away from the viewer, z = up), main colour and \
material (matte, glossy, metal, glass, glowing). Then one line about the lighting and the background. Be concrete \
and brief; no introduction.";

/// Ask a vision model to describe a picture (base64 PNG/JPEG) for rebuilding in 3D.
fn describe_image(agent: &ureq::Agent, host: &str, model: &str, image_b64: &str, request: &str) -> Result<String, String> {
    let mut prompt = DESCRIBE_PROMPT.to_string();
    if !request.trim().is_empty() {
        prompt += &format!("\nThe user wants: {request}");
    }
    let body = json!({
        "model": model,
        "messages": [{"role": "user", "content": prompt, "images": [image_b64]}],
        "stream": false,
        "keep_alive": "30m",
        "options": {"temperature": 0.2, "num_ctx": 8192}
    });
    let mut resp = agent.post(&format!("{host}/api/chat")).send_json(&body).map_err(|e| friendly_error(&e.to_string(), model))?;
    let v: Value = resp.body_mut().read_json().map_err(|e| format!("bad response from {model}: {e}"))?;
    if let Some(e) = v["error"].as_str() {
        return Err(format!("{model}: {e}"));
    }
    let text = v["message"]["content"].as_str().unwrap_or("").trim().to_string();
    // drop any <think> block
    let text = match text.find("</think>") {
        Some(i) => text[i + 8..].trim().to_string(),
        None => text,
    };
    if text.is_empty() {
        return Err(format!("{model} returned an empty description"));
    }
    Ok(text)
}

/// Downscale (max 1024 px) and re-encode a picture for a vision model; returns base64 PNG.
pub fn prepare_image(bytes: &[u8]) -> Result<String, String> {
    use base64::Engine;
    let img = image::load_from_memory(bytes).map_err(|e| format!("can't read the picture: {e}"))?.to_rgba8();
    let (w, h) = (img.width() as usize, img.height() as usize);
    let src = crate::textures::TexImage { w, h, rgba: img.pixels().map(|p| p.0).collect() };
    let s = (1024.0 / w.max(h) as f32).min(1.0);
    let small = if s < 1.0 { src.resized(((w as f32 * s) as usize).max(1), ((h as f32 * s) as usize).max(1)) } else { src };
    let raw: Vec<u8> = small.rgba.iter().flatten().copied().collect();
    let png = crate::textures::encode_png(small.w, small.h, &raw)?;
    Ok(base64::engine::general_purpose::STANDARD.encode(png))
}

pub fn pick_best_model(installed: &[String]) -> Option<String> {
    for pref in PREFERRED_MODELS {
        if let Some(m) = installed.iter().find(|m| {
            m.as_str() == *pref
                || m.starts_with(&format!("{pref}:"))
                || (pref.contains(':') && m.starts_with(pref))
        }) {
            return Some(m.clone());
        }
    }
    // Otherwise prefer anything that isn't an embedding / vision-only model.
    installed
        .iter()
        .find(|m| !m.contains("embed") && !m.contains("whisper") && !m.to_lowercase().starts_with("tev") && !VISION_MODELS.iter().any(|v| m.starts_with(v) && *v != "gemma3"))
        .cloned()
}

pub fn command_schema() -> Value {
    let vec3 = json!({"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3});
    json!({
        "type": "object",
        "properties": {
            "reply": {"type": "string"},
            "commands": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string", "enum": ACTIONS},
                        "target": {"type": "string"},
                        "primitive": {"type": "string", "enum": ["cube","plane","uv_sphere","ico_sphere","cylinder","cone","torus","monkey","point_light","sun","spot_light","camera"]},
                        "name": {"type": "string"},
                        "location": vec3,
                        "rotation": vec3,
                        "scale": vec3,
                        "vector": vec3,
                        "color": {"type": "string"},
                        "number": {"type": "number"},
                        "metallic": {"type": "number"},
                        "roughness": {"type": "number"},
                        "emission": {"type": "number"},
                        "alpha": {"type": "number"},
                        "modifier": {"type": "string", "enum": ["array","mirror","radial","subsurf"]},
                        "view": {"type": "string"},
                        "power": {"type": "number"},
                        "parent": {"type": "string"},
                        "object": {"type": "string"},
                        "collection": {"type": "string"},
                        "segments": {"type": "number"},
                        "look_at": {"type": "string"}
                    },
                    "required": ["action"]
                }
            }
        },
        "required": ["reply", "commands"]
    })
}

pub const ACTIONS: &[&str] = &[
    "add",
    "delete",
    "clear_scene",
    "select",
    "deselect_all",
    "move",
    "rotate",
    "scale",
    "set_transform",
    "set_color",
    "rename",
    "duplicate",
    "hide",
    "show",
    "add_modifier",
    "clear_modifiers",
    "keyframe",
    "clear_keyframes",
    "set_frame",
    "set_frame_range",
    "play",
    "stop",
    "view",
    "focus",
    "shading",
    "smooth",
    "set_light",
    "look_at",
    "camera_view",
    "render",
    "render_animation",
    "set_texture",
    "node_material",
    "bump",
    "roughness_map",
    "scatter",
    "hair",
    "groom",
    "hair_dynamics",
    "template",
    "add_asset",
    "save_asset",
    "realize",
    "rig",
    "pose_mode",
    "pose_bone",
    "reset_pose",
    "add_ik",
    "particles",
    "rigid_body",
    "bake_physics",
    "cloth",
    "remove_cloth",
    "add_shape_key",
    "set_shape_key",
    "remove_shape_key",
    "weight_paint",
    "play_action",
    "set_action",
    "new_action",
    "subsurf",
    "boolean",
    "unwrap",
    "parent",
    "clear_parent",
    "move_to_collection",
    "hide_collection",
    "show_collection",
    "auto_smooth",
    "edit_mode",
    "sculpt_mode",
    "object_mode",
    "select_faces",
    "extrude",
    "inset",
    "bevel",
    "subdivide",
    "loop_cut",
    "merge",
    "flip_normals",
    "flat",
    "set_detail",
    "set_background",
    "undo",
    "redo",
    "chat",
];

pub fn system_prompt(scene: &str) -> String {
    format!(
        r##"You are the AI co-pilot inside "Rust Blender", a 3D modeling app modeled on Blender.
Convert the user's request (typed or spoken, may contain speech-recognition typos) into editor commands.

COORDINATES: Z is up. X is right, -Y is toward the viewer (front), +Y is away. 1 unit = 1 meter.
Default primitives are 2 units wide (radius 1), centered on their location. To rest an object on the
ground (z=0) set location z = scale_z. "Above X" means larger z; "next to" means offset in x by ~2.5.

Respond ONLY with JSON: {{"reply": "<one short friendly sentence>", "commands": [ ... ]}}
Each command is an object with "action" plus only the fields it needs:
- add: primitive (cube|plane|uv_sphere|ico_sphere|cylinder|cone|torus|monkey), optional name, location [x,y,z],
  rotation [deg x,y,z], scale [x,y,z], color, metallic 0-1, roughness 0-1, emission 0-10, alpha 0-1,
  optional modifier + number + vector to add a modifier immediately.
- add a light or camera: primitive point_light|sun|spot_light|camera, location, optional look_at (object name
  or "x,y,z"), color (light color), power (watts for point/spot e.g. 1000, strength for sun e.g. 3).
  Lights/cameras point along their local -Z axis; prefer look_at over rotation for aiming them.
- set_light: target, color, power, number (spot cone angle in degrees).   look_at: target, look_at.
- camera_view: look through the scene camera (or view = camera).   render: render a final image through the camera (F12); name "path" = path traced (realistic, slower), number = samples.
- render_animation: render every frame of the timeline (PNG sequence + MP4 when ffmpeg is installed) into a
  folder next to the saved file; name "path" / "fast" picks the engine, number = samples.
- delete / select / hide / show / focus / smooth / flat: target
- move: target, vector [dx,dy,dz] (relative).   rotate: target, vector [deg x,y,z] (relative).
- scale: target, vector [fx,fy,fz] (multiplicative).   set_transform: target + absolute location/rotation/scale.
- set_color: target, color (name like "red", "light blue", "gold" or hex "#ff8800"), metallic, roughness, emission, alpha.
- rename: target, name.   duplicate: target, number (copies), vector (offset between copies).
- add_modifier: target, modifier (array: number=count, vector=offset | mirror: vector=axes e.g. [1,0,0] | radial: number=count, vector=[radius,0,0]).
- keyframe: target, number (frame), optional absolute location/rotation/scale to set before keying.
- set_frame: number.   set_frame_range: vector [start,end,0].   play / stop.
- view: view = front|back|left|right|top|bottom|iso|ortho|all|selected|zoom_in|zoom_out.
- shading: view = wireframe|solid|material.   set_detail: target, number 1-6 (mesh resolution).
- set_background: color; vector = sun direction; number = sun strength.
- bevel in object mode: target, number = width (e.g. 0.1), segments (rounds every edge of the object).
- HIERARCHY: add accepts parent = an existing object's name (the new object then moves with it) and
  collection = a group name (created if needed). parent: target, parent (name). clear_parent: target.
  move_to_collection: target, collection. hide_collection / show_collection: collection. For multi-part
  things (a car, a robot, a lamp) add the main body first and give the other parts parent = the body.
- boolean: target = object to change, object = the cutter, name = difference|union|intersect (difference cuts
  the cutter's shape out: doors, windows, holes). Add the cutter first (e.g. a cube overlapping the wall), then
  boolean; the cutter is hidden afterwards.
- subsurf: target, number = levels 1-3 (Catmull-Clark smooth subdivision; 0 removes). Use it for organic or
  rounded shapes (a cube with subsurf 2 becomes a rounded blob). auto_smooth: target, number = angle (30).
- unwrap: target, name = smart|cube|sphere|cylinder (UV unwrap so textures don't stretch; use after
  modelling, before set_texture).
- set_texture: target, name = checker|bricks|wood|marble|tiles|grid|noise|stripes|concrete|grass (or an
  image file path), number = repeats (e.g. 6 for a big floor), optional color = tint; name "none" removes it.
- node_material: target, name = rusty metal | wood | marble | cobblestone | lava | bricks | tiles (procedural
  shader-node materials with colour, roughness, metal and bumps; lava glows), number = repeats; "none" removes.
  Prefer it for realistic surfaces ("make it rusty", "a lava floor", "cobblestone street").
- bump: target, name = "auto" (bumps from the object's texture), a pattern / image / normal-map path, or
  "none"; number = strength (1 normal, 2-3 deep). Add "invert" to the name when light parts should be low.
  Use for "make the bricks bumpy", "rough stone surface", "add a normal map".
- scatter: target = surface mesh (ground, terrain, planet), name = rock|tree|grass|bush|flower or an
  existing object's name (it becomes the hidden template), number = count, scale = [min,max], seed, slope = max
  steepness in degrees (30 keeps things on flat ground), align = true to stand along the surface (planets),
  rotation = [tilt,tilt,spin]. Repeating it with the same target and name updates the scatter. realize: target
  = scatter object (bake to a real mesh). "a forest" = tree scatter; "a rocky field" = rock scatter.
- template: name = General | Empty | Product Shot | Turntable | Landscape | Interior | Physics Playground |
  Hair & Fur — REPLACES the whole scene with a ready-made one (studio, lights, camera, content). Only when the
  user asks for a new scene / a fresh start ("start a product shot", "new landscape scene").
- add_asset: name = an asset from the library — built-in objects Vase, Rock, Pine Tree, Bush, Flower, Table,
  Chair, Furry Ball, Domino, Lamp (or one the user saved), location = where (an object name puts it on top
  of it… use [x,y,z]); or a material asset (Gold, Chrome, Copper, Glass, Red Plastic, Rubber, Ceramic, Neon
  Glow, Wood Floor, Granite …) applied to target. "put a lamp next to the table", "make the cube chrome".
- save_asset: target = objects to save to the user's asset library, name = asset name; modifier = "material"
  saves only the target's material.
- hair: target = the mesh to grow it on, name = fur | hair | curly | shaggy | grass (strands) or "none" to
  remove; number = guide strands, length = metres, gravity = droop 0..1, curl = turns, color = hair colour,
  seed. Repeating it on the same target updates the hair (only numbers → tweak, a name → restyle). "a furry
  ball" = sphere + fur; "long hair on the head" = hair with length 0.6. groom: enter Hair Edit (brush
  grooming) on the selected hair. hair_dynamics: target = hair (or its surface) — simulate swinging strands
  over the animation; number = stiffness 0..1; name "off" removes it.
- particles: name = sparks | snow | rain | confetti | bubbles | smoke | fire (or object = an existing object to
  emit), target = the emitter mesh (sparks from a sphere, smoke from a chimney; snow / rain without a target get
  an emitter above the scene), number = particles per frame, power = speed, scale = size. Calling it with
  target = an existing "… Particles" object re-simulates it with the new values.
- rigid_body: target = an object that should fall / tumble / bounce (crates, balls, dominoes), name = active
  (default) | passive (an obstacle) | none, number = bounce 0..1. Every call re-simulates all rigid bodies over
  the timeline (other meshes are obstacles: put a ground plane under them); then play. bake_physics re-runs it
  after moving things. For a stack or a domino row, add the objects first, then rigid_body each.
- cloth: target = a plane / sheet above other objects (tablecloth, blanket, flag), name = pins: none | top |
  corners | left | right (a flag pinned on the left), number = stiffness 0.1-1. It is simulated over the
  timeline right away (the other meshes are colliders); then play. remove_cloth: target.
- SHAPE KEYS: add_shape_key: target, name, plus a generated shape: scale = [x,y,z] around the centre
  (squash [1.3,1.3,0.6], stretch [0.8,0.8,1.4]), vector = lean (offset at the top), power = inflate along
  normals; it starts at value 1. set_shape_key: target, name, number = 0..1. Animate with set_frame,
  set_shape_key, keyframe (target = the mesh). remove_shape_key: target, name.
- ARMATURES: rig: target = mesh, number = bones (2-8; adds an armature fitted along the mesh and binds it
  with automatic weights, e.g. a snake 6, a tail 3, a tree trunk 4). add primitive "armature" (number = bones)
  adds a bone chain; parent a mesh to an armature to bind it. pose_bone: target = armature (or the rigged
  mesh), name = bone ("Bone.001", or "bone 2"), rotation = [x,y,z] degrees in the bone's axes (x bends
  forward/back). reset_pose. add_ik: target = armature, name = bone (the hand/foot end), number = chain length
  (2 for arm or leg, 0 = to the root), object = existing target (otherwise an Empty "IK <bone>" is made at the
  tip); then move / keyframe that target object to pose the limb. add primitive "empty" adds a target point. weight_paint: target = rigged mesh, name = bone (opens Weight Paint so the user
  can paint which parts a bone moves). play_action / set_action: target = armature or character, name =
  action ("walk", "run": imported characters often have several), number = cross-fade frames (default 6
  while playing); new_action: name. Animate a pose: set_frame, pose_bone…, keyframe (target = armature), repeat.
- roughness_map: target, name = "auto" (from its texture) | pattern | image path | "none"; number = base
  roughness. Makes the grooves matte and the tops shinier in renders.
- MESH EDITING (Blender edit mode, works on the vertices/faces of one mesh object):
  edit_mode (target) -> select_faces (vector = face direction, e.g. [0,0,1] = top faces; name "extend" to add)
  -> extrude (number = distance along the face normal) / inset (number 0-1) / bevel (number = width,
  segments = 1 for a chamfer or 3-6 for rounded; bevels the edges of the selection) / subdivide / loop_cut
  (vector = axis the cut edges run along, e.g. [0,0,1] cuts a horizontal ring; number = cuts) / merge /
  flip_normals / delete (no target = delete selected faces) / move (vector, no target = move selected
  vertices) / scale (number) -> object_mode. Always finish with object_mode.
- clear_scene, deselect_all, undo, redo.   chat: when the user only asks a question (put the answer in reply).
"target" is an object name, "selected", "all", "last", or a kind like "spheres", "lights", "cameras".
Lights only affect the Material Preview shading mode; after adding lights you may switch with
{{"action":"shading","view":"material"}}.
For complex requests, break them into many simple commands and compute coordinates yourself
(e.g. a ring of 8 objects: compute x=r*cos(a), y=r*sin(a)). For animation: keyframe at a start frame,
then keyframe again at a later frame with the new location/rotation, then play.
Objects created earlier in the same batch can be targeted by the name you gave them.

CURRENT SCENE:
{scene}"##
    )
}

#[derive(Deserialize)]
struct AiOut {
    #[serde(default)]
    reply: String,
    #[serde(default)]
    commands: Vec<Value>,
}

/// Incremental parser for the streamed `{"reply": "...", "commands": [ {...}, ... ]}` object.
/// Yields the reply text as it grows and each command object as soon as its closing brace arrives.
#[derive(Default)]
pub struct StreamParser {
    pub buf: String,
    scanned: usize,
    in_string: bool,
    escape: bool,
    depth: i32,
    /// byte offset where the "commands" array starts (after '[')
    cmds_start: Option<usize>,
    obj_start: Option<usize>,
    pub emitted: usize,
}

impl StreamParser {
    pub fn push(&mut self, chunk: &str) -> Vec<Command> {
        self.buf.push_str(chunk);
        let mut out = vec![];
        if self.cmds_start.is_none() {
            if let Some(k) = find_key(&self.buf, "commands") {
                if let Some(b) = self.buf[k..].find('[') {
                    self.cmds_start = Some(k + b + 1);
                    self.scanned = k + b + 1;
                }
            }
        }
        let Some(_) = self.cmds_start else { return out };
        let bytes = self.buf.as_bytes();
        let mut i = self.scanned;
        while i < bytes.len() {
            let c = bytes[i];
            if self.in_string {
                if self.escape {
                    self.escape = false;
                } else if c == b'\\' {
                    self.escape = true;
                } else if c == b'"' {
                    self.in_string = false;
                }
            } else {
                match c {
                    b'"' => self.in_string = true,
                    b'{' => {
                        if self.depth == 0 {
                            self.obj_start = Some(i);
                        }
                        self.depth += 1;
                    }
                    b'}' => {
                        self.depth -= 1;
                        if self.depth == 0 {
                            if let Some(s) = self.obj_start.take() {
                                if let Ok(cmd) = serde_json::from_str::<Command>(&self.buf[s..=i]) {
                                    if !cmd.action.is_empty() {
                                        out.push(cmd);
                                    }
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
            i += 1;
        }
        self.scanned = i;
        self.emitted += out.len();
        out
    }

    /// Best-effort decode of the (possibly incomplete) "reply" string.
    pub fn reply_so_far(&self) -> String {
        let Some(k) = find_key(&self.buf, "reply") else {
            return String::new();
        };
        let rest = &self.buf[k..];
        let Some(q) = rest.find('"') else {
            return String::new();
        };
        let mut out = String::new();
        let mut chars = rest[q + 1..].chars();
        while let Some(ch) = chars.next() {
            match ch {
                '"' => break,
                '\\' => match chars.next() {
                    Some('n') => out.push('\n'),
                    Some('t') => out.push('\t'),
                    Some('u') => {
                        let hex: String = chars.by_ref().take(4).collect();
                        if let Some(c) = u32::from_str_radix(&hex, 16).ok().and_then(char::from_u32)
                        {
                            out.push(c);
                        }
                    }
                    Some(c) => out.push(c),
                    None => break,
                },
                c => out.push(c),
            }
        }
        out
    }
}

/// Position just after `"key"` and its colon.
fn find_key(s: &str, key: &str) -> Option<usize> {
    let pat = format!("\"{key}\"");
    let k = s.find(&pat)? + pat.len();
    let colon = s[k..].find(':')?;
    Some(k + colon + 1)
}

#[allow(clippy::too_many_arguments)]
fn chat(
    agent: &ureq::Agent,
    host: &str,
    model: &str,
    history: &[ChatMsg],
    scene: &str,
    prompt: &str,
    tx: &Sender<AiEvent>,
    repaint: &dyn Fn(),
    cancel: &AtomicBool,
) -> Result<(String, Vec<Command>, String), String> {
    use std::io::BufRead;
    let mut messages = vec![json!({"role": "system", "content": system_prompt(scene)})];
    // few-shot example keeps small models on-format
    messages.push(json!({"role": "user", "content": "put a red ball on top of the cube"}));
    messages.push(json!({"role": "assistant", "content": r#"{"reply":"Added a red ball resting on the cube.","commands":[{"action":"add","primitive":"uv_sphere","name":"Red Ball","location":[0,0,2],"scale":[1,1,1],"color":"red"}]}"#}));
    // Keep the context small (fits more GPUs, much faster prompt processing): newest turns
    // first, within a character budget.
    let mut budget = 6000usize;
    let mut recent = vec![];
    for m in history.iter().rev().take(8) {
        let c: String = m.content.chars().take(1500).collect();
        if c.len() > budget {
            break;
        }
        budget -= c.len();
        recent.push(json!({"role": m.role, "content": c}));
    }
    messages.extend(recent.into_iter().rev());
    messages.push(json!({"role": "user", "content": prompt}));
    let mut body = json!({
        "model": model,
        "messages": messages,
        "stream": true,
        "format": command_schema(),
        "think": false,
        "keep_alive": "30m",
        "options": {"temperature": 0.15, "num_ctx": 6144}
    });
    for attempt in 0..2 {
        let resp = agent
            .post(&format!("{host}/api/chat"))
            .send_json(&body)
            .map_err(|e| format!("Ollama request failed: {e}"))?;
        let status = resp.status();
        if !status.is_success() {
            let mut resp = resp;
            let v: Value = resp.body_mut().read_json().unwrap_or_default();
            let err = v["error"].as_str().unwrap_or("unknown error").to_string();
            // Older Ollama versions reject the "think" flag.
            if attempt == 0 && err.contains("think") {
                body.as_object_mut().unwrap().remove("think");
                continue;
            }
            return Err(format!("Ollama error ({status}): {err}"));
        }
        let reader = std::io::BufReader::new(
            resp.into_body()
                .into_with_config()
                .limit(64 * 1024 * 1024)
                .reader(),
        );
        let mut parser = StreamParser::default();
        let mut all_cmds = vec![];
        let mut last_reply_len = 0;
        for line in reader.lines() {
            if cancel.load(Ordering::Relaxed) {
                return Err("cancelled".into());
            }
            let line = line.map_err(|e| e.to_string())?;
            if line.trim().is_empty() {
                continue;
            }
            let v: Value =
                serde_json::from_str(&line).map_err(|e| format!("bad stream line: {e}"))?;
            if let Some(e) = v["error"].as_str() {
                return Err(format!("Ollama error: {e}"));
            }
            let chunk = v["message"]["content"].as_str().unwrap_or("");
            let new_cmds = parser.push(chunk);
            let reply = parser.reply_so_far();
            if !new_cmds.is_empty() || reply.len() != last_reply_len {
                last_reply_len = reply.len();
                all_cmds.extend(new_cmds.iter().cloned());
                let _ = tx.send(AiEvent::Delta {
                    reply,
                    commands: new_cmds,
                });
                repaint();
            }
            if v["done"].as_bool() == Some(true) {
                break;
            }
        }
        // Final, authoritative parse (also catches anything the incremental scanner missed).
        let raw = parser.buf.clone();
        return match parse_ai_content(&raw) {
            Ok((reply, cmds, raw)) => {
                let missing: Vec<Command> = cmds.iter().skip(all_cmds.len()).cloned().collect();
                Ok((reply, missing, raw))
            }
            Err(e) if !all_cmds.is_empty() => {
                let _ = e;
                Ok((parser.reply_so_far(), vec![], raw))
            }
            Err(e) => Err(e),
        };
    }
    Err("Ollama request failed".into())
}

/// Parse the model output; tolerant of code fences, <think> blocks and bare arrays.
pub fn parse_ai_content(content: &str) -> Result<(String, Vec<Command>, String), String> {
    let raw = content.to_string();
    let mut s = content.trim();
    if let Some(end) = s.find("</think>") {
        s = s[end + 8..].trim();
    }
    let s = s
        .trim_start_matches("```json")
        .trim_start_matches("```")
        .trim_end_matches("```")
        .trim();
    let start = s
        .find(['{', '['])
        .ok_or_else(|| format!("model did not return JSON: {raw}"))?;
    let s = &s[start..];
    let out: AiOut = if s.starts_with('[') {
        AiOut {
            reply: String::new(),
            commands: serde_json::from_str(s).map_err(|e| e.to_string())?,
        }
    } else {
        // trailing garbage tolerant
        let mut de = serde_json::Deserializer::from_str(s).into_iter::<Value>();
        let v = de
            .next()
            .ok_or("empty JSON")?
            .map_err(|e| format!("invalid JSON from model: {e}"))?;
        if v.get("action").is_some() {
            AiOut {
                reply: String::new(),
                commands: vec![v],
            }
        } else {
            serde_json::from_value(v).map_err(|e| e.to_string())?
        }
    };
    let cmds = out
        .commands
        .into_iter()
        .filter_map(|c| serde_json::from_value::<Command>(c).ok())
        .filter(|c| !c.action.is_empty())
        .collect();
    Ok((out.reply, cmds, raw))
}

fn pull(
    agent: &ureq::Agent,
    host: &str,
    model: &str,
    tx: &Sender<AiEvent>,
    repaint: &(impl Fn() + Send),
) -> Result<(), String> {
    use std::io::BufRead;
    let resp = agent
        .post(&format!("{host}/api/pull"))
        .send_json(json!({"model": model, "stream": true}))
        .map_err(|e| e.to_string())?;
    let reader = std::io::BufReader::new(resp.into_body().into_reader());
    for line in reader.lines() {
        let line = line.map_err(|e| e.to_string())?;
        let v: Value = serde_json::from_str(&line).unwrap_or_default();
        if let Some(e) = v["error"].as_str() {
            return Err(e.into());
        }
        let status = v["status"].as_str().unwrap_or("").to_string();
        let frac = match (v["completed"].as_f64(), v["total"].as_f64()) {
            (Some(c), Some(t)) if t > 0.0 => (c / t) as f32,
            _ => -1.0,
        };
        let _ = tx.send(AiEvent::PullProgress(status, frac));
        repaint();
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// Offline natural-language fallback (works without Ollama, and makes voice snappy for simple
// commands). Handles the most common Blender-style operations.
// ---------------------------------------------------------------------------------------------

fn words_to_num(w: &str) -> Option<f64> {
    let w = w.trim_matches(|c: char| !c.is_alphanumeric() && c != '.' && c != '-');
    if let Ok(n) = w.parse::<f64>() {
        return Some(n);
    }
    Some(match w {
        "zero" => 0.0,
        "one" | "a" | "an" => 1.0,
        "two" | "to" | "too" => 2.0,
        "three" => 3.0,
        "four" | "for" => 4.0,
        "five" => 5.0,
        "six" => 6.0,
        "seven" => 7.0,
        "eight" => 8.0,
        "nine" => 9.0,
        "ten" => 10.0,
        "eleven" => 11.0,
        "twelve" => 12.0,
        "fifteen" => 15.0,
        "twenty" => 20.0,
        "thirty" => 30.0,
        "forty" => 40.0,
        "forty-five" => 45.0,
        "fifty" => 50.0,
        "sixty" => 60.0,
        "ninety" => 90.0,
        "hundred" => 100.0,
        "half" => 0.5,
        "double" | "twice" => 2.0,
        _ => return None,
    })
}

fn first_number(words: &[&str], skip_articles: bool) -> Option<f64> {
    words
        .iter()
        .filter(|w| !skip_articles || !matches!(**w, "a" | "an" | "to" | "too" | "for"))
        .find_map(|w| words_to_num(w))
}

/// Returns `None` if the sentence isn't confidently understood (then the LLM should handle it).
pub fn parse_simple(text: &str) -> Option<Vec<Command>> {
    let t = text
        .to_lowercase()
        .replace([',', '!', '?'], " ")
        .replace('.', " . ");
    let t = t.replace(" . ", " ");
    let t = t.trim().trim_end_matches('.').trim();
    let words: Vec<&str> = t.split_whitespace().collect();
    if words.is_empty() {
        return None;
    }
    let has = |w: &str| words.contains(&w);
    let phrase = |p: &str| t.contains(p);

    // Too complex for rules -> LLM
    if words.len() > 9
        || phrase(" and ")
        || phrase(" then ")
        || phrase("ring of")
        || phrase("circle of")
        || phrase("row of")
        || phrase(" each ")
        || phrase("animate")
    {
        return None;
    }

    let single = |a: &str| Some(vec![Command::new(a)]);
    match t {
        "undo" | "undo that" | "go back" => return single("undo"),
        "redo" => return single("redo"),
        "play" | "play animation" | "start animation" => return single("play"),
        "stop" | "pause" | "stop animation" => return single("stop"),
        "delete" | "delete it" | "delete that" | "delete selected" | "remove it"
        | "remove that" => return single("delete"),
        "select all" => {
            return Some(vec![Command {
                action: "select".into(),
                target: Some("all".into()),
                ..Default::default()
            }]);
        }
        "deselect" | "deselect all" | "select none" => return single("deselect_all"),
        "clear scene" | "clear the scene" | "delete everything" | "new scene" => {
            return single("clear_scene");
        }
        "duplicate" | "duplicate it" | "copy it" => return single("duplicate"),
        "insert keyframe" | "keyframe" | "add keyframe" | "key it" => return single("keyframe"),
        "camera view" | "look through camera" | "look through the camera" | "view camera" => {
            return single("camera_view");
        }
        "render" | "render image" | "render it" | "render the scene" | "f12" | "take a picture" | "take a render" => {
            return single("render");
        }
        "render the animation" | "render animation" | "render a video" | "render the video" | "make a video" | "export the animation" => return single("render_animation"),
        "edit mode" | "enter edit mode" | "tab" => return single("edit_mode"),
        "sculpt mode" | "sculpt" | "enter sculpt mode" | "let me sculpt" => return single("sculpt_mode"),
        "object mode" | "exit edit mode" => return single("object_mode"),
        "subdivide" | "subdivide it" => return single("subdivide"),
        "merge" | "merge vertices" => return single("merge"),
        "flip normals" => return single("flip_normals"),
        // "make it wood", "add a brick texture", "give it a marble texture"
        _ if words.len() <= 6
            && crate::textures::match_pattern(t).is_some()
            && (t.starts_with("make it ")
                || (t.ends_with(" texture") && ["give it a ", "add a ", "apply a ", "use a "].iter().any(|p| t.starts_with(p)))) =>
        {
            let p = crate::textures::match_pattern(t).unwrap_or("checker");
            return Some(vec![Command { action: "set_texture".into(), name: Some(p.into()), ..Default::default() }]);
        }
        "subdivision surface" | "add subdivision" | "add subdivision surface" | "subsurf" | "subdivide smooth" | "smooth it out" | "make it smoother" => {
            return Some(vec![Command { action: "subsurf".into(), number: Some(2.0), ..Default::default() }]);
        }
        "remove subdivision" | "no subdivision" => {
            return Some(vec![Command { action: "subsurf".into(), number: Some(0.0), ..Default::default() }]);
        }
        "shade auto smooth" | "auto smooth" => {
            return Some(vec![Command { action: "auto_smooth".into(), number: Some(30.0), ..Default::default() }]);
        }
        "pose mode" | "enter pose mode" => return single("pose_mode"),
        "reset pose" | "reset the pose" | "rest pose" | "clear pose" => return single("reset_pose"),
        "rig it" | "rig this" | "add a rig" | "add an armature" | "add armature" | "auto rig" => {
            return Some(vec![Command { action: "rig".into(), number: Some(4.0), ..Default::default() }]);
        }
        "add trees" | "add a forest" | "scatter trees" | "plant trees" => return Some(vec![Command { action: "scatter".into(), name: Some("tree".into()), ..Default::default() }]),
        "add rocks" | "scatter rocks" | "add some rocks" => return Some(vec![Command { action: "scatter".into(), name: Some("rock".into()), ..Default::default() }]),
        "new product shot" | "start a product shot" | "product shot" | "studio scene" | "new studio scene" => return Some(vec![Command { action: "template".into(), name: Some("Product Shot".into()), ..Default::default() }]),
        "new landscape" | "new landscape scene" | "start a landscape" | "landscape scene" => return Some(vec![Command { action: "template".into(), name: Some("Landscape".into()), ..Default::default() }]),
        "new turntable" | "turntable" | "make a turntable" | "turntable scene" => return Some(vec![Command { action: "template".into(), name: Some("Turntable".into()), ..Default::default() }]),
        "new interior" | "interior scene" | "new room" | "start an interior" => return Some(vec![Command { action: "template".into(), name: Some("Interior".into()), ..Default::default() }]),
        "physics playground" | "new physics scene" => return Some(vec![Command { action: "template".into(), name: Some("Physics Playground".into()), ..Default::default() }]),
        "save as asset" | "save it as an asset" | "add to library" | "add it to the library" | "save to library" => return Some(vec![Command::new("save_asset")]),
        "add fur" | "make it furry" | "furry" | "give it fur" => return Some(vec![Command { action: "hair".into(), name: Some("fur".into()), ..Default::default() }]),
        "add hair" | "give it hair" | "grow hair" => return Some(vec![Command { action: "hair".into(), name: Some("hair".into()), ..Default::default() }]),
        "remove hair" | "remove the hair" | "remove fur" | "remove the fur" | "shave it" => return Some(vec![Command { action: "hair".into(), name: Some("none".into()), ..Default::default() }]),
        "add grass" | "scatter grass" | "grow grass" => return Some(vec![Command { action: "scatter".into(), name: Some("grass".into()), ..Default::default() }]),
        "make it bumpy" | "add bumps" | "bumpy" | "add a normal map" | "add normal map" | "add bump" | "add bump map" | "make the texture bumpy" => {
            return Some(vec![Command { action: "bump".into(), name: Some("auto".into()), number: Some(1.0), ..Default::default() }]);
        }
        "make it smooth again" | "remove the bumps" | "remove bumps" | "remove the normal map" | "no bumps" => {
            return Some(vec![Command { action: "bump".into(), name: Some("none".into()), ..Default::default() }]);
        }
        "remove the texture" | "remove texture" | "no texture" => {
            return Some(vec![Command { action: "set_texture".into(), name: Some("none".into()), ..Default::default() }]);
        }
        "bevel" | "bevel it" | "bevel the edges" | "bevel edges" | "chamfer" | "chamfer it" => {
            return Some(vec![Command { action: "bevel".into(), number: Some(0.1), segments: Some(1.0), ..Default::default() }]);
        }
        "round the edges" | "round it" | "round off the edges" | "smooth the edges" | "rounded bevel" => {
            return Some(vec![Command { action: "bevel".into(), number: Some(0.15), segments: Some(4.0), ..Default::default() }]);
        }
        "inset" | "inset faces" | "inset it" => {
            return Some(vec![Command {
                action: "inset".into(),
                number: Some(0.25),
                ..Default::default()
            }]);
        }
        "loop cut" | "add a loop cut" | "loopcut" => {
            return Some(vec![Command {
                action: "loop_cut".into(),
                vector: Some(json!([0, 0, 1])),
                ..Default::default()
            }]);
        }
        "select top face" | "select the top face" | "select top faces" => {
            return Some(vec![Command {
                action: "select_faces".into(),
                vector: Some(json!([0, 0, 1])),
                ..Default::default()
            }]);
        }
        _ => {}
    }
    for (v, names) in [
        ("front", &["front view", "view front"][..]),
        ("top", &["top view", "view top", "look from above"][..]),
        ("right", &["right view", "side view", "view right"][..]),
        ("left", &["left view", "view left"][..]),
        ("back", &["back view", "view back"][..]),
        ("bottom", &["bottom view"][..]),
        (
            "ortho",
            &["toggle ortho", "orthographic", "perspective"][..],
        ),
        (
            "all",
            &["frame all", "show everything", "zoom to fit", "view all"][..],
        ),
        (
            "selected",
            &["frame selected", "zoom to selected", "focus on it", "focus"][..],
        ),
        ("zoom_in", &["zoom in"][..]),
        ("zoom_out", &["zoom out"][..]),
    ] {
        if names
            .iter()
            .any(|n| t == *n || t == format!("{n} view").as_str())
        {
            return Some(vec![Command {
                action: "view".into(),
                view: Some(v.into()),
                ..Default::default()
            }]);
        }
    }
    if phrase("wireframe") {
        return Some(vec![Command {
            action: "shading".into(),
            view: Some("wireframe".into()),
            ..Default::default()
        }]);
    }
    if t == "solid" || t == "solid mode" || t == "solid shading" {
        return Some(vec![Command {
            action: "shading".into(),
            view: Some("solid".into()),
            ..Default::default()
        }]);
    }
    if phrase("material preview") || phrase("render mode") || t == "material mode" {
        return Some(vec![Command {
            action: "shading".into(),
            view: Some("material".into()),
            ..Default::default()
        }]);
    }
    if t == "shade smooth" || t == "smooth shading" || t == "make it smooth" {
        return single("smooth");
    }
    if t == "shade flat" || t == "flat shading" {
        return single("flat");
    }

    let verb = words[0];
    let color_in = |ws: &[&str]| -> Option<String> {
        for (i, w) in ws.iter().enumerate() {
            if crate::editor::parse_color_str(w).is_some() {
                if i > 0 && matches!(ws[i - 1], "light" | "dark") {
                    return Some(format!("{} {}", ws[i - 1], w));
                }
                return Some(w.to_string());
            }
        }
        None
    };

    // ADD
    if matches!(
        verb,
        "add" | "create" | "make" | "insert" | "spawn" | "new" | "give" | "place" | "put"
    ) {
        let prim = words.iter().find_map(|w| {
            let w = w.trim_end_matches('s');
            crate::mesh::Primitive::parse(w)
                .map(|_| w.to_string())
                .or_else(|| crate::mesh::Primitive::parse(w).map(|_| w.to_string()))
        });
        let prim = prim.or_else(|| {
            words
                .iter()
                .find(|w| crate::mesh::Primitive::parse(w).is_some())
                .map(|w| w.to_string())
        });
        if let Some(p) = prim {
            let count = words
                .iter()
                .take_while(|w| crate::mesh::Primitive::parse(w.trim_end_matches('s')).is_none())
                .find_map(|w| match *w {
                    "a" | "an" | "one" => None,
                    w => words_to_num(w).filter(|n| *n >= 2.0 && *n <= 50.0),
                });
            let color = color_in(&words);
            let mut loc = None;
            if phrase(" above") || phrase(" on top") {
                loc = Some(json!([0, 0, 2.5]));
            }
            let mut cmds = vec![Command {
                action: "add".into(),
                primitive: Some(p),
                color: color.map(Value::String),
                location: loc,
                ..Default::default()
            }];
            if let Some(n) = count {
                cmds.push(Command {
                    action: "duplicate".into(),
                    number: Some(n - 1.0),
                    vector: Some(json!([2.5, 0, 0])),
                    ..Default::default()
                });
            }
            return Some(cmds);
        }
        for (w, kind, loc) in [
            ("camera", "camera", json!([7.0, -7.0, 5.0])),
            ("spotlight", "spot_light", json!([3.0, -3.0, 4.0])),
            ("spot", "spot_light", json!([3.0, -3.0, 4.0])),
            ("sun", "sun", json!([0.0, 0.0, 8.0])),
            ("light", "point_light", json!([2.0, -2.0, 4.0])),
            ("lamp", "point_light", json!([2.0, -2.0, 4.0])),
        ] {
            if words.iter().any(|x| x.trim_end_matches('s') == w) {
                let look = if kind == "point_light" {
                    None
                } else {
                    Some(json!("0,0,0"))
                };
                return Some(vec![Command {
                    action: "add".into(),
                    primitive: Some(kind.into()),
                    location: Some(loc),
                    look_at: look,
                    color: color_in(&words).map(Value::String),
                    ..Default::default()
                }]);
            }
        }
        // "make it red" / "make it bigger"
        if verb == "make" {
            let rest = &words[1..];
            if let Some(c) = color_in(rest) {
                return Some(vec![Command {
                    action: "set_color".into(),
                    color: Some(Value::String(c)),
                    ..Default::default()
                }]);
            }
            if rest.iter().any(|w| matches!(*w, "bigger" | "larger")) {
                return Some(vec![Command {
                    action: "scale".into(),
                    number: Some(1.5),
                    ..Default::default()
                }]);
            }
            if rest.iter().any(|w| matches!(*w, "smaller" | "tiny")) {
                return Some(vec![Command {
                    action: "scale".into(),
                    number: Some(0.67),
                    ..Default::default()
                }]);
            }
            if rest
                .iter()
                .any(|w| matches!(*w, "metal" | "metallic" | "shiny" | "chrome"))
            {
                return Some(vec![Command {
                    action: "set_color".into(),
                    metallic: Some(1.0),
                    roughness: Some(0.2),
                    ..Default::default()
                }]);
            }
            if rest
                .iter()
                .any(|w| matches!(*w, "glow" | "glowing" | "emissive"))
            {
                return Some(vec![Command {
                    action: "set_color".into(),
                    emission: Some(2.0),
                    ..Default::default()
                }]);
            }
        }
        return None;
    }

    if verb == "extrude" {
        let d = first_number(&words[1..], true).unwrap_or(1.0);
        return Some(vec![Command {
            action: "extrude".into(),
            number: Some(d),
            ..Default::default()
        }]);
    }
    // COLOR
    if matches!(verb, "color" | "colour" | "paint") {
        let c = color_in(&words[1..])?;
        return Some(vec![Command {
            action: "set_color".into(),
            color: Some(Value::String(c)),
            ..Default::default()
        }]);
    }

    // MOVE
    if matches!(
        verb,
        "move" | "grab" | "shift" | "raise" | "lower" | "lift" | "push" | "slide"
    ) {
        let amount = first_number(&words[1..], true).unwrap_or(1.0) as f32;
        let dir =
            if verb == "raise" || verb == "lift" || has("up") || has("upward") || has("higher") {
                [0.0, 0.0, 1.0]
            } else if verb == "lower" || has("down") || has("downward") {
                [0.0, 0.0, -1.0]
            } else if has("left") {
                [-1.0, 0.0, 0.0]
            } else if has("right") {
                [1.0, 0.0, 0.0]
            } else if has("forward") || has("closer") || has("toward") {
                [0.0, -1.0, 0.0]
            } else if has("back") || has("backward") || has("away") {
                [0.0, 1.0, 0.0]
            } else if has("x") {
                [1.0, 0.0, 0.0]
            } else if has("y") {
                [0.0, 1.0, 0.0]
            } else if has("z") {
                [0.0, 0.0, 1.0]
            } else {
                return None;
            };
        return Some(vec![Command {
            action: "move".into(),
            vector: Some(json!([dir[0] * amount, dir[1] * amount, dir[2] * amount])),
            ..Default::default()
        }]);
    }

    // ROTATE
    if matches!(verb, "rotate" | "spin" | "turn" | "tilt") {
        let deg = first_number(&words[1..], true).unwrap_or(45.0) as f32;
        let v = if has("x") {
            [deg, 0.0, 0.0]
        } else if has("y") {
            [0.0, deg, 0.0]
        } else {
            [0.0, 0.0, deg]
        };
        return Some(vec![Command {
            action: "rotate".into(),
            vector: Some(json!(v)),
            ..Default::default()
        }]);
    }

    // SCALE
    if matches!(verb, "scale" | "resize" | "grow" | "shrink" | "enlarge") {
        let mut f =
            first_number(&words[1..], true).unwrap_or(if verb == "shrink" { 0.5 } else { 1.5 });
        if verb == "shrink" && f > 1.0 {
            f = 1.0 / f;
        }
        if has("percent") || has("%") {
            f /= 100.0;
        }
        let v = if has("x") && !has("y") && !has("z") {
            json!([f, 1, 1])
        } else if has("y") && !has("x") {
            json!([1, f, 1])
        } else if has("z") || has("taller") {
            json!([1, 1, f])
        } else {
            json!(f)
        };
        return Some(vec![Command {
            action: "scale".into(),
            vector: Some(v),
            ..Default::default()
        }]);
    }

    // SELECT / DELETE named
    if matches!(verb, "select" | "pick" | "choose") && words.len() >= 2 {
        let target = words[1..]
            .iter()
            .filter(|w| **w != "the")
            .copied()
            .collect::<Vec<_>>()
            .join(" ");
        return Some(vec![Command {
            action: "select".into(),
            target: Some(target),
            ..Default::default()
        }]);
    }
    if matches!(verb, "delete" | "remove" | "erase") && words.len() >= 2 {
        let target = words[1..]
            .iter()
            .filter(|w| **w != "the")
            .copied()
            .collect::<Vec<_>>()
            .join(" ");
        return Some(vec![Command {
            action: "delete".into(),
            target: Some(target),
            ..Default::default()
        }]);
    }
    if matches!(verb, "hide") {
        return single("hide");
    }
    if t == "show all" || t == "unhide" || t == "unhide all" {
        return single("show");
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn simple_parser() {
        let c = parse_simple("Add a red cube").unwrap();
        assert_eq!(c[0].action, "add");
        assert_eq!(c[0].primitive.as_deref(), Some("cube"));
        assert_eq!(c[0].color, Some(Value::String("red".into())));
        let c = parse_simple("move it up three").unwrap();
        assert_eq!(c[0].vector, Some(json!([0.0, 0.0, 3.0])));
        let c = parse_simple("rotate 90 degrees on x").unwrap();
        assert_eq!(c[0].vector, Some(json!([90.0, 0.0, 0.0])));
        assert_eq!(
            parse_simple("top view").unwrap()[0].view.as_deref(),
            Some("top")
        );
        assert_eq!(parse_simple("add five spheres").unwrap().len(), 2);
        assert!(parse_simple("build a castle with four towers and a moat").is_none());
        assert_eq!(
            parse_simple("make it light blue").unwrap()[0].color,
            Some(Value::String("light blue".into()))
        );
        assert_eq!(
            parse_simple("add a camera").unwrap()[0]
                .primitive
                .as_deref(),
            Some("camera")
        );
        assert_eq!(
            parse_simple("add a red spotlight").unwrap()[0]
                .primitive
                .as_deref(),
            Some("spot_light")
        );
        assert_eq!(
            parse_simple("camera view").unwrap()[0].action,
            "camera_view"
        );
        assert_eq!(parse_simple("render the scene").unwrap()[0].action, "render");
        assert_eq!(parse_simple("make it wood").unwrap()[0].name.as_deref(), Some("wood"));
        assert_eq!(parse_simple("add a brick texture").unwrap()[0].action, "set_texture");
        assert_eq!(parse_simple("make it bumpy").unwrap()[0].action, "bump");
        assert_eq!(parse_simple("rig it").unwrap()[0].action, "rig");
        assert_eq!(parse_simple("add grass").unwrap()[0].action, "scatter");
        assert_eq!(parse_simple("add a cube").unwrap()[0].action, "add");
    }
    #[test]
    fn parse_model_output() {
        let (r, c, _) = parse_ai_content("<think>hmm</think>```json\n{\"reply\":\"ok\",\"commands\":[{\"action\":\"add\",\"primitive\":\"torus\"}]}\n```").unwrap();
        assert_eq!(r, "ok");
        assert_eq!(c.len(), 1);
        let (_, c, _) = parse_ai_content("[{\"action\":\"undo\"}]").unwrap();
        assert_eq!(c[0].action, "undo");
    }
    #[test]
    fn stream_parser_incremental() {
        let full = r#"{"reply":"Building a \"snowman\" now","commands":[{"action":"add","primitive":"uv_sphere","name":"A {b}"},{"action":"move","vector":[0,0,1]}]}"#;
        let mut p = StreamParser::default();
        let mut got = vec![];
        let mut replies = vec![];
        for ch in full.as_bytes().chunks(7) {
            got.extend(p.push(std::str::from_utf8(ch).unwrap()));
            replies.push(p.reply_so_far());
        }
        assert_eq!(got.len(), 2);
        assert_eq!(got[0].name.as_deref(), Some("A {b}"));
        assert_eq!(got[1].action, "move");
        assert_eq!(p.reply_so_far(), "Building a \"snowman\" now");
        assert!(
            replies
                .iter()
                .any(|r| r.starts_with("Buil") && r.len() < 20)
        );
    }

    #[test]
    fn model_pick() {
        let installed = vec![
            "llama3.2:latest".to_string(),
            "qwen3:8b".to_string(),
            "nomic-embed-text:latest".to_string(),
        ];
        assert_eq!(pick_best_model(&installed).as_deref(), Some("qwen3:8b"));
        let installed = vec![
            "nomic-embed-text:latest".to_string(),
            "mistral:latest".to_string(),
        ];
        assert_eq!(
            pick_best_model(&installed).as_deref(),
            Some("mistral:latest")
        );
        assert!(command_schema()["properties"]["commands"].is_object());
    }
}
