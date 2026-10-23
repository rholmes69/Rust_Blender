//! Ollama integration (local LLM) + an offline rule-based command parser fallback.

use crate::editor::Command;
use crossbeam_channel::{Receiver, Sender, unbounded};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
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

pub enum AiRequest {
    ListModels,
    Chat { model: String, history: Vec<ChatMsg>, scene: String, prompt: String },
    Pull(String),
}

#[derive(Debug)]
pub enum AiEvent {
    Models(Result<Vec<String>, String>),
    Reply { reply: String, commands: Vec<Command>, raw: String, secs: f32 },
    Error(String),
    PullProgress(String, f32),
    PullDone(Result<String, String>),
}

pub struct AiClient {
    tx: Sender<AiRequest>,
    pub rx: Receiver<AiEvent>,
}

impl AiClient {
    pub fn new(host: String, repaint: impl Fn() + Send + Clone + 'static) -> Self {
        let (tx, req_rx) = unbounded::<AiRequest>();
        let (ev_tx, rx) = unbounded::<AiEvent>();
        std::thread::Builder::new()
            .name("ollama".into())
            .spawn(move || {
                let agent: ureq::Agent = ureq::Agent::config_builder()
                    .timeout_connect(Some(Duration::from_secs(3)))
                    .timeout_global(Some(Duration::from_secs(600)))
                    .http_status_as_error(false)
                    .build()
                    .into();
                for req in req_rx {
                    match req {
                        AiRequest::ListModels => {
                            let _ = ev_tx.send(AiEvent::Models(list_models(&agent, &host)));
                        }
                        AiRequest::Chat { model, history, scene, prompt } => {
                            let t0 = std::time::Instant::now();
                            let ev = match chat(&agent, &host, &model, &history, &scene, &prompt) {
                                Ok((reply, commands, raw)) => AiEvent::Reply { reply, commands, raw, secs: t0.elapsed().as_secs_f32() },
                                Err(e) => AiEvent::Error(e),
                            };
                            let _ = ev_tx.send(ev);
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
        Self { tx, rx }
    }

    pub fn send(&self, r: AiRequest) {
        let _ = self.tx.send(r);
    }
}

fn list_models(agent: &ureq::Agent, host: &str) -> Result<Vec<String>, String> {
    let mut resp = agent.get(&format!("{host}/api/tags")).call().map_err(|e| format!("Ollama not reachable at {host} ({e}). Is `ollama serve` running?"))?;
    let v: Value = resp.body_mut().read_json().map_err(|e| e.to_string())?;
    Ok(v["models"].as_array().map(|a| a.iter().filter_map(|m| m["name"].as_str().map(String::from)).collect()).unwrap_or_default())
}

/// Pick the best installed model according to `PREFERRED_MODELS`.
pub fn pick_best_model(installed: &[String]) -> Option<String> {
    for pref in PREFERRED_MODELS {
        if let Some(m) = installed.iter().find(|m| m.as_str() == *pref || m.starts_with(&format!("{pref}:")) || (pref.contains(':') && m.starts_with(pref))) {
            return Some(m.clone());
        }
    }
    // Otherwise prefer anything that isn't an embedding / vision-only model.
    installed.iter().find(|m| !m.contains("embed") && !m.contains("whisper")).cloned()
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
                        "primitive": {"type": "string", "enum": ["cube","plane","uv_sphere","ico_sphere","cylinder","cone","torus","monkey"]},
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
                        "modifier": {"type": "string", "enum": ["array","mirror","radial"]},
                        "view": {"type": "string"}
                    },
                    "required": ["action"]
                }
            }
        },
        "required": ["reply", "commands"]
    })
}

pub const ACTIONS: &[&str] = &[
    "add", "delete", "clear_scene", "select", "deselect_all", "move", "rotate", "scale", "set_transform",
    "set_color", "rename", "duplicate", "hide", "show", "add_modifier", "clear_modifiers", "keyframe",
    "clear_keyframes", "set_frame", "set_frame_range", "play", "stop", "view", "focus", "shading", "smooth",
    "flat", "set_detail", "set_background", "undo", "redo", "chat",
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
- clear_scene, deselect_all, undo, redo.   chat: when the user only asks a question (put the answer in reply).
"target" is an object name, "selected", "all", "last", or a kind like "spheres".
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

fn chat(agent: &ureq::Agent, host: &str, model: &str, history: &[ChatMsg], scene: &str, prompt: &str) -> Result<(String, Vec<Command>, String), String> {
    let mut messages = vec![json!({"role": "system", "content": system_prompt(scene)})];
    // few-shot example keeps small models on-format
    messages.push(json!({"role": "user", "content": "put a red ball on top of the cube"}));
    messages.push(json!({"role": "assistant", "content": r#"{"reply":"Added a red ball resting on the cube.","commands":[{"action":"add","primitive":"uv_sphere","name":"Red Ball","location":[0,0,2],"scale":[1,1,1],"color":"red"}]}"#}));
    for m in history.iter().rev().take(10).rev() {
        messages.push(json!({"role": m.role, "content": m.content}));
    }
    messages.push(json!({"role": "user", "content": prompt}));
    let body = json!({
        "model": model,
        "messages": messages,
        "stream": false,
        "format": command_schema(),
        "think": false,
        "keep_alive": "30m",
        "options": {"temperature": 0.15, "num_ctx": 8192}
    });
    let mut resp = agent.post(&format!("{host}/api/chat")).send_json(&body).map_err(|e| format!("Ollama request failed: {e}"))?;
    let status = resp.status();
    let v: Value = resp.body_mut().read_json().map_err(|e| format!("bad response from Ollama: {e}"))?;
    if !status.is_success() || v.get("error").is_some() {
        let err = v["error"].as_str().unwrap_or("unknown error").to_string();
        // Older Ollama versions reject the "think" flag for non-thinking models.
        if err.contains("think") {
            let mut body = body;
            body.as_object_mut().unwrap().remove("think");
            let mut resp = agent.post(&format!("{host}/api/chat")).send_json(&body).map_err(|e| e.to_string())?;
            let v: Value = resp.body_mut().read_json().map_err(|e| e.to_string())?;
            return parse_ai_content(v["message"]["content"].as_str().unwrap_or(""));
        }
        return Err(format!("Ollama error: {err}"));
    }
    parse_ai_content(v["message"]["content"].as_str().unwrap_or(""))
}

/// Parse the model output; tolerant of code fences, <think> blocks and bare arrays.
pub fn parse_ai_content(content: &str) -> Result<(String, Vec<Command>, String), String> {
    let raw = content.to_string();
    let mut s = content.trim();
    if let Some(end) = s.find("</think>") {
        s = s[end + 8..].trim();
    }
    let s = s.trim_start_matches("```json").trim_start_matches("```").trim_end_matches("```").trim();
    let start = s.find(['{', '[']).ok_or_else(|| format!("model did not return JSON: {raw}"))?;
    let s = &s[start..];
    let out: AiOut = if s.starts_with('[') {
        AiOut { reply: String::new(), commands: serde_json::from_str(s).map_err(|e| e.to_string())? }
    } else {
        // trailing garbage tolerant
        let mut de = serde_json::Deserializer::from_str(s).into_iter::<Value>();
        let v = de.next().ok_or("empty JSON")?.map_err(|e| format!("invalid JSON from model: {e}"))?;
        if v.get("action").is_some() {
            AiOut { reply: String::new(), commands: vec![v] }
        } else {
            serde_json::from_value(v).map_err(|e| e.to_string())?
        }
    };
    let cmds = out.commands.into_iter().filter_map(|c| serde_json::from_value::<Command>(c).ok()).filter(|c| !c.action.is_empty()).collect();
    Ok((out.reply, cmds, raw))
}

fn pull(agent: &ureq::Agent, host: &str, model: &str, tx: &Sender<AiEvent>, repaint: &(impl Fn() + Send)) -> Result<(), String> {
    use std::io::BufRead;
    let resp = agent.post(&format!("{host}/api/pull")).send_json(json!({"model": model, "stream": true})).map_err(|e| e.to_string())?;
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
        "zero" => 0.0, "one" | "a" | "an" => 1.0, "two" | "to" | "too" => 2.0, "three" => 3.0, "four" | "for" => 4.0,
        "five" => 5.0, "six" => 6.0, "seven" => 7.0, "eight" => 8.0, "nine" => 9.0, "ten" => 10.0,
        "eleven" => 11.0, "twelve" => 12.0, "fifteen" => 15.0, "twenty" => 20.0, "thirty" => 30.0,
        "forty" => 40.0, "forty-five" => 45.0, "fifty" => 50.0, "sixty" => 60.0, "ninety" => 90.0,
        "hundred" => 100.0, "half" => 0.5, "double" | "twice" => 2.0,
        _ => return None,
    })
}

fn first_number(words: &[&str], skip_articles: bool) -> Option<f64> {
    words.iter().filter(|w| !skip_articles || !matches!(**w, "a" | "an" | "to" | "too" | "for")).find_map(|w| words_to_num(w))
}

/// Returns `None` if the sentence isn't confidently understood (then the LLM should handle it).
pub fn parse_simple(text: &str) -> Option<Vec<Command>> {
    let t = text.to_lowercase().replace([',', '!', '?'], " ").replace('.', " . ");
    let t = t.replace(" . ", " ");
    let t = t.trim().trim_end_matches('.').trim();
    let words: Vec<&str> = t.split_whitespace().collect();
    if words.is_empty() {
        return None;
    }
    let has = |w: &str| words.contains(&w);
    let phrase = |p: &str| t.contains(p);

    // Too complex for rules -> LLM
    if words.len() > 9 || phrase(" and ") || phrase(" then ") || phrase("ring of") || phrase("circle of") || phrase("row of") || phrase(" each ") || phrase("animate") {
        return None;
    }

    let single = |a: &str| Some(vec![Command::new(a)]);
    match t {
        "undo" | "undo that" | "go back" => return single("undo"),
        "redo" => return single("redo"),
        "play" | "play animation" | "start animation" => return single("play"),
        "stop" | "pause" | "stop animation" => return single("stop"),
        "delete" | "delete it" | "delete that" | "delete selected" | "remove it" | "remove that" => return single("delete"),
        "select all" => return Some(vec![Command { action: "select".into(), target: Some("all".into()), ..Default::default() }]),
        "deselect" | "deselect all" | "select none" => return single("deselect_all"),
        "clear scene" | "clear the scene" | "delete everything" | "new scene" => return single("clear_scene"),
        "duplicate" | "duplicate it" | "copy it" => return single("duplicate"),
        "insert keyframe" | "keyframe" | "add keyframe" | "key it" => return single("keyframe"),
        _ => {}
    }
    for (v, names) in [
        ("front", &["front view", "view front"][..]),
        ("top", &["top view", "view top", "look from above"][..]),
        ("right", &["right view", "side view", "view right"][..]),
        ("left", &["left view", "view left"][..]),
        ("back", &["back view", "view back"][..]),
        ("bottom", &["bottom view"][..]),
        ("ortho", &["toggle ortho", "orthographic", "perspective"][..]),
        ("all", &["frame all", "show everything", "zoom to fit", "view all"][..]),
        ("selected", &["frame selected", "zoom to selected", "focus on it", "focus"][..]),
        ("zoom_in", &["zoom in"][..]),
        ("zoom_out", &["zoom out"][..]),
    ] {
        if names.iter().any(|n| t == *n || t == format!("{n} view").as_str()) {
            return Some(vec![Command { action: "view".into(), view: Some(v.into()), ..Default::default() }]);
        }
    }
    if phrase("wireframe") {
        return Some(vec![Command { action: "shading".into(), view: Some("wireframe".into()), ..Default::default() }]);
    }
    if t == "solid" || t == "solid mode" || t == "solid shading" {
        return Some(vec![Command { action: "shading".into(), view: Some("solid".into()), ..Default::default() }]);
    }
    if phrase("material preview") || phrase("render mode") || t == "material mode" {
        return Some(vec![Command { action: "shading".into(), view: Some("material".into()), ..Default::default() }]);
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
    if matches!(verb, "add" | "create" | "make" | "insert" | "spawn" | "new" | "give" | "place" | "put") {
        let prim = words.iter().find_map(|w| {
            let w = w.trim_end_matches('s');
            crate::mesh::Primitive::parse(w).map(|_| w.to_string()).or_else(|| crate::mesh::Primitive::parse(w).map(|_| w.to_string()))
        });
        let prim = prim.or_else(|| words.iter().find(|w| crate::mesh::Primitive::parse(w).is_some()).map(|w| w.to_string()));
        if let Some(p) = prim {
            let count = words.iter().take_while(|w| crate::mesh::Primitive::parse(w.trim_end_matches('s')).is_none()).find_map(|w| match *w {
                "a" | "an" | "one" => None,
                w => words_to_num(w).filter(|n| *n >= 2.0 && *n <= 50.0),
            });
            let color = color_in(&words);
            let mut loc = None;
            if phrase(" above") || phrase(" on top") {
                loc = Some(json!([0, 0, 2.5]));
            }
            let mut cmds = vec![Command { action: "add".into(), primitive: Some(p), color: color.map(Value::String), location: loc, ..Default::default() }];
            if let Some(n) = count {
                cmds.push(Command { action: "duplicate".into(), number: Some(n - 1.0), vector: Some(json!([2.5, 0, 0])), ..Default::default() });
            }
            return Some(cmds);
        }
        // "make it red" / "make it bigger"
        if verb == "make" {
            let rest = &words[1..];
            if let Some(c) = color_in(rest) {
                return Some(vec![Command { action: "set_color".into(), color: Some(Value::String(c)), ..Default::default() }]);
            }
            if rest.iter().any(|w| matches!(*w, "bigger" | "larger")) {
                return Some(vec![Command { action: "scale".into(), number: Some(1.5), ..Default::default() }]);
            }
            if rest.iter().any(|w| matches!(*w, "smaller" | "tiny")) {
                return Some(vec![Command { action: "scale".into(), number: Some(0.67), ..Default::default() }]);
            }
            if rest.iter().any(|w| matches!(*w, "metal" | "metallic" | "shiny" | "chrome")) {
                return Some(vec![Command { action: "set_color".into(), metallic: Some(1.0), roughness: Some(0.2), ..Default::default() }]);
            }
            if rest.iter().any(|w| matches!(*w, "glow" | "glowing" | "emissive")) {
                return Some(vec![Command { action: "set_color".into(), emission: Some(2.0), ..Default::default() }]);
            }
        }
        return None;
    }

    // COLOR
    if matches!(verb, "color" | "colour" | "paint") {
        let c = color_in(&words[1..])?;
        return Some(vec![Command { action: "set_color".into(), color: Some(Value::String(c)), ..Default::default() }]);
    }

    // MOVE
    if matches!(verb, "move" | "grab" | "shift" | "raise" | "lower" | "lift" | "push" | "slide") {
        let amount = first_number(&words[1..], true).unwrap_or(1.0) as f32;
        let dir = if verb == "raise" || verb == "lift" || has("up") || has("upward") || has("higher") {
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
        return Some(vec![Command { action: "move".into(), vector: Some(json!([dir[0] * amount, dir[1] * amount, dir[2] * amount])), ..Default::default() }]);
    }

    // ROTATE
    if matches!(verb, "rotate" | "spin" | "turn" | "tilt") {
        let deg = first_number(&words[1..], true).unwrap_or(45.0) as f32;
        let v = if has("x") { [deg, 0.0, 0.0] } else if has("y") { [0.0, deg, 0.0] } else { [0.0, 0.0, deg] };
        return Some(vec![Command { action: "rotate".into(), vector: Some(json!(v)), ..Default::default() }]);
    }

    // SCALE
    if matches!(verb, "scale" | "resize" | "grow" | "shrink" | "enlarge") {
        let mut f = first_number(&words[1..], true).unwrap_or(if verb == "shrink" { 0.5 } else { 1.5 });
        if verb == "shrink" && f > 1.0 {
            f = 1.0 / f;
        }
        if has("percent") || has("%") {
            f /= 100.0;
        }
        let v = if has("x") && !has("y") && !has("z") { json!([f, 1, 1]) } else if has("y") && !has("x") { json!([1, f, 1]) } else if has("z") || has("taller") { json!([1, 1, f]) } else { json!(f) };
        return Some(vec![Command { action: "scale".into(), vector: Some(v), ..Default::default() }]);
    }

    // SELECT / DELETE named
    if matches!(verb, "select" | "pick" | "choose") && words.len() >= 2 {
        let target = words[1..].iter().filter(|w| **w != "the").copied().collect::<Vec<_>>().join(" ");
        return Some(vec![Command { action: "select".into(), target: Some(target), ..Default::default() }]);
    }
    if matches!(verb, "delete" | "remove" | "erase") && words.len() >= 2 {
        let target = words[1..].iter().filter(|w| **w != "the").copied().collect::<Vec<_>>().join(" ");
        return Some(vec![Command { action: "delete".into(), target: Some(target), ..Default::default() }]);
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
        assert_eq!(parse_simple("top view").unwrap()[0].view.as_deref(), Some("top"));
        assert_eq!(parse_simple("add five spheres").unwrap().len(), 2);
        assert!(parse_simple("build a castle with four towers and a moat").is_none());
        assert_eq!(parse_simple("make it light blue").unwrap()[0].color, Some(Value::String("light blue".into())));
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
    fn model_pick() {
        let installed = vec!["llama3.2:latest".to_string(), "qwen3:8b".to_string(), "nomic-embed-text:latest".to_string()];
        assert_eq!(pick_best_model(&installed).as_deref(), Some("qwen3:8b"));
        let installed = vec!["nomic-embed-text:latest".to_string(), "mistral:latest".to_string()];
        assert_eq!(pick_best_model(&installed).as_deref(), Some("mistral:latest"));
        assert!(command_schema()["properties"]["commands"].is_object());
    }
}
