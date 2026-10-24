//! Two-model AI pipeline: a small, fast *decision model* (e.g. tev1) looks at every request
//! first and either
//!   * performs simple edits itself via tool calls (`scene_command`),
//!   * answers questions (`answer`), or
//!   * hands creative / multi-step builds to the larger *builder* model (qwen3) with a short plan
//!     (`delegate_to_builder`).
//!
//! Uses Ollama's native tool calling when the model reports the `tools` capability, otherwise a
//! JSON-schema constrained response with the same three routes.

use crate::ai::{ACTIONS, ChatMsg, command_schema};
use crate::editor::Command;
use serde_json::{Value, json};

#[derive(Clone, Debug, PartialEq)]
pub enum Route {
    /// The decision model handled it: apply these commands.
    Direct { reply: String, commands: Vec<Command> },
    /// Needs the builder model; `plan` is passed along as guidance.
    Delegate { plan: String, reason: String },
    /// Just an answer, no scene change.
    Answer(String),
}

/// Model capabilities from `/api/show` (e.g. ["completion", "tools"]).
pub fn capabilities(agent: &ureq::Agent, host: &str, model: &str) -> Vec<String> {
    let Ok(mut resp) = agent.post(&format!("{host}/api/show")).send_json(json!({"model": model})) else { return vec![] };
    let v: Value = resp.body_mut().read_json().unwrap_or_default();
    v["capabilities"].as_array().map(|a| a.iter().filter_map(|c| c.as_str().map(String::from)).collect()).unwrap_or_default()
}

/// Pick an installed decision model (anything named like tev1, or other small tool-calling models).
pub fn pick_decision_model(installed: &[String]) -> Option<String> {
    for pref in ["tev1", "tev"] {
        if let Some(m) = installed.iter().find(|m| m.to_lowercase().starts_with(pref) || m.to_lowercase().contains(&format!("/{pref}"))) {
            return Some(m.clone());
        }
    }
    None
}

fn command_params() -> Value {
    command_schema()["properties"]["commands"]["items"].clone()
}

pub fn tools() -> Value {
    json!([
        {"type": "function", "function": {
            "name": "scene_command",
            "description": "Perform ONE simple editor operation in the 3D scene (add, delete, select, move, rotate, scale, set_color, set_light, view, shading, keyframe, play, undo, edit_mode, extrude, ...). Call it several times for a few operations.",
            "parameters": command_params()
        }},
        {"type": "function", "function": {
            "name": "delegate_to_builder",
            "description": "Hand the request to the large builder model. Use for creative or multi-object builds (e.g. 'build a castle', 'make a solar system'), anything needing more than about 4 operations, layout math, or multi-step animation.",
            "parameters": {"type": "object", "properties": {
                "plan": {"type": "string", "description": "Short step-by-step plan for the builder."},
                "reason": {"type": "string"}
            }, "required": ["plan"]}
        }},
        {"type": "function", "function": {
            "name": "answer",
            "description": "Reply in text when the user only asks a question and nothing in the scene should change.",
            "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}
        }}
    ])
}

fn json_route_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "route": {"type": "string", "enum": ["direct", "delegate", "answer"]},
            "reply": {"type": "string"},
            "plan": {"type": "string"},
            "commands": {"type": "array", "items": command_params()}
        },
        "required": ["route", "reply"]
    })
}

pub fn decision_prompt(scene: &str, tool_mode: bool) -> String {
    let how = if tool_mode {
        "Respond ONLY with tool calls. Use scene_command (possibly several times) for simple edits, delegate_to_builder for complex builds, answer for questions."
    } else {
        r#"Respond ONLY with JSON: {"route":"direct|delegate|answer","reply":"short sentence","plan":"for delegate","commands":[...scene commands for direct...]}"#
    };
    format!(
        r#"You are the fast DECISION layer of "Rust Blender", a Blender-like 3D editor. Z is up, 1 unit = 1 m,
default objects are 2 units wide. Decide how to handle the user's request:
- SIMPLE and concrete (at most ~4 operations on existing objects or one new object): do it yourself.
- CREATIVE, multi-object, needs layout math or long animation: delegate to the builder with a short plan.
- A question with no scene change: answer.
Scene command actions: {actions}.
Targets: object name, "selected", "all", "last", or kinds like "spheres", "lights".
{how}

CURRENT SCENE:
{scene}"#,
        actions = ACTIONS.join(", ")
    )
}

/// Ask the decision model how to handle `prompt`.
/// `mode_cache` remembers which request style ("tools", "json", "plain") this model accepted,
/// so later requests don't retry the ones it rejects.
pub fn decide(agent: &ureq::Agent, host: &str, model: &str, history: &[ChatMsg], scene: &str, prompt: &str, tool_mode: bool, mode_cache: &mut Option<String>) -> Result<(Route, String), String> {
    let mut messages = vec![json!({"role": "system", "content": decision_prompt(scene, tool_mode)})];
    for m in history.iter().rev().take(4).rev() {
        let c: String = m.content.chars().take(600).collect();
        messages.push(json!({"role": m.role, "content": c}));
    }
    messages.push(json!({"role": "user", "content": prompt}));
    // Try the richest mode first and degrade gracefully: tool calling -> JSON schema ->
    // plain JSON instructions (some models, e.g. ones exposing a "decision" capability,
    // reject `tools`, `format` or `think`).
    let all: &[(&str, bool)] = if tool_mode { &[("tools", true), ("json", true), ("plain", false)] } else { &[("json", true), ("plain", false)] };
    let cached: Vec<(&str, bool)> = all.iter().copied().filter(|(m, _)| mode_cache.as_deref() == Some(*m)).collect();
    let modes: Vec<(&str, bool)> = if cached.is_empty() { all.to_vec() } else { cached.into_iter().chain(all.iter().copied().filter(|(m, _)| mode_cache.as_deref() != Some(*m))).collect() };
    let mut last_err = String::new();
    let mut v = Value::Null;
    for (mode, think_flag) in &modes {
        let mut msgs = messages.clone();
        if *mode != "tools" {
            msgs[0] = json!({"role": "system", "content": decision_prompt(scene, false)});
        }
        let mut body = json!({
            "model": model,
            "messages": msgs,
            "stream": false,
            "keep_alive": "30m",
            "options": {"temperature": 0.0, "num_ctx": 4096}
        });
        if *think_flag {
            body["think"] = json!(false);
        }
        match *mode {
            "tools" => body["tools"] = tools(),
            "json" => body["format"] = json_route_schema(),
            _ => {}
        }
        let mut resp = match agent.post(&format!("{host}/api/chat")).send_json(&body) {
            Ok(r) => r,
            Err(e) => {
                // e.g. a dropped keep-alive connection after an error response: try the next mode
                last_err = format!("request failed: {e}");
                continue;
            }
        };
        let status = resp.status();
        let val: Value = match resp.body_mut().read_json() {
            Ok(v) => v,
            Err(e) => {
                last_err = format!("bad response: {e}");
                continue;
            }
        };
        if status.is_success() && val.get("error").is_none() {
            v = val;
            *mode_cache = Some(mode.to_string());
            break;
        }
        last_err = val["error"].as_str().unwrap_or("unknown error").to_string();
    }
    if v.is_null() {
        return Err(format!("decision model error: {last_err}"));
    }
    let raw = v["message"].to_string();
    let route = parse_decision(&v["message"])?;
    Ok((route, raw))
}

/// Interpret a decision-model message (tool calls or JSON content).
pub fn parse_decision(msg: &Value) -> Result<Route, String> {
    if let Some(calls) = msg["tool_calls"].as_array().filter(|c| !c.is_empty()) {
        let mut commands = vec![];
        let mut answer = None;
        let mut delegate = None;
        for call in calls {
            let f = &call["function"];
            let name = f["name"].as_str().unwrap_or("");
            // arguments may arrive as an object or as a JSON string
            let args = match &f["arguments"] {
                Value::String(s) => serde_json::from_str(s).unwrap_or(Value::Null),
                v => v.clone(),
            };
            match name {
                "delegate_to_builder" => delegate = Some(Route::Delegate { plan: args["plan"].as_str().unwrap_or("").into(), reason: args["reason"].as_str().unwrap_or("").into() }),
                "answer" => answer = Some(args["text"].as_str().unwrap_or("").to_string()),
                _ => {
                    // "scene_command", or a model that calls the action name directly
                    let mut a = args.clone();
                    if a.get("action").is_none() && ACTIONS.contains(&name) {
                        a["action"] = json!(name);
                    }
                    if let Ok(c) = serde_json::from_value::<Command>(a) {
                        if !c.action.is_empty() {
                            commands.push(c);
                        }
                    }
                }
            }
        }
        if let Some(d) = delegate {
            return Ok(d);
        }
        if !commands.is_empty() {
            let reply = msg["content"].as_str().map(str::trim).filter(|s| !s.is_empty()).unwrap_or("Done.").to_string();
            return Ok(Route::Direct { reply, commands });
        }
        if let Some(a) = answer {
            return Ok(Route::Answer(a));
        }
        return Err("decision model made no usable tool call".into());
    }
    // JSON mode (or a tools model that answered in plain content)
    let content = msg["content"].as_str().unwrap_or("").trim();
    let content = match content.find("</think>") {
        Some(i) => content[i + 8..].trim(),
        None => content,
    };
    let content = content.trim_start_matches("```json").trim_start_matches("```").trim_end_matches("```").trim();
    let start = content.find('{').ok_or_else(|| format!("decision model gave no decision: {content}"))?;
    let v: Value = serde_json::Deserializer::from_str(&content[start..]).into_iter::<Value>().next().ok_or("empty decision")?.map_err(|e| e.to_string())?;
    let reply = v["reply"].as_str().unwrap_or("").to_string();
    match v["route"].as_str().unwrap_or("delegate") {
        "direct" => {
            let commands: Vec<Command> = v["commands"].as_array().map(|a| a.iter().filter_map(|c| serde_json::from_value(c.clone()).ok()).filter(|c: &Command| !c.action.is_empty()).collect()).unwrap_or_default();
            if commands.is_empty() {
                Ok(Route::Delegate { plan: v["plan"].as_str().unwrap_or("").into(), reason: "no commands".into() })
            } else {
                Ok(Route::Direct { reply: if reply.is_empty() { "Done.".into() } else { reply }, commands })
            }
        }
        "answer" => Ok(Route::Answer(if reply.is_empty() { v["plan"].as_str().unwrap_or("").into() } else { reply })),
        _ => Ok(Route::Delegate { plan: v["plan"].as_str().unwrap_or("").into(), reason: reply }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn tool_call_routes() {
        let m = json!({"content": "", "tool_calls": [
            {"function": {"name": "scene_command", "arguments": {"action": "add", "primitive": "cube", "color": "red"}}},
            {"function": {"name": "scene_command", "arguments": "{\"action\":\"move\",\"vector\":[0,0,2]}"}}
        ]});
        match parse_decision(&m).unwrap() {
            Route::Direct { commands, .. } => {
                assert_eq!(commands.len(), 2);
                assert_eq!(commands[1].action, "move");
            }
            r => panic!("{r:?}"),
        }
        let m = json!({"tool_calls": [{"function": {"name": "delegate_to_builder", "arguments": {"plan": "1. base 2. towers"}}}]});
        assert_eq!(parse_decision(&m).unwrap(), Route::Delegate { plan: "1. base 2. towers".into(), reason: String::new() });
        let m = json!({"tool_calls": [{"function": {"name": "answer", "arguments": {"text": "Press Tab."}}}]});
        assert_eq!(parse_decision(&m).unwrap(), Route::Answer("Press Tab.".into()));
        // model calling an action name directly as a tool
        let m = json!({"tool_calls": [{"function": {"name": "undo", "arguments": {}}}]});
        assert!(matches!(parse_decision(&m).unwrap(), Route::Direct { .. }));
    }
    #[test]
    fn json_routes() {
        let m = json!({"content": "{\"route\":\"direct\",\"reply\":\"ok\",\"commands\":[{\"action\":\"view\",\"view\":\"top\"}]}"});
        assert!(matches!(parse_decision(&m).unwrap(), Route::Direct { .. }));
        let m = json!({"content": "{\"route\":\"delegate\",\"reply\":\"complex\",\"plan\":\"build it\"}"});
        assert!(matches!(parse_decision(&m).unwrap(), Route::Delegate { .. }));
        let m = json!({"content": "{\"route\":\"direct\",\"reply\":\"x\",\"commands\":[]}"});
        assert!(matches!(parse_decision(&m).unwrap(), Route::Delegate { .. }), "empty direct falls back to the builder");
    }
    #[test]
    fn picks_decision_model() {
        let inst = vec!["qwen3:8b".to_string(), "tev1:latest".to_string()];
        assert_eq!(pick_decision_model(&inst).as_deref(), Some("tev1:latest"));
        assert!(tools().as_array().unwrap().len() == 3);
    }
}

/// Agent loop "check" stage: ask a model to review flagged issues and return fix commands.
pub fn review(agent: &ureq::Agent, host: &str, model: &str, prompt: &str) -> Result<(String, Vec<Command>), String> {
    let mut last_err = String::new();
    for (use_format, think_flag) in [(true, true), (false, false)] {
        let mut body = json!({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": false,
            "keep_alive": "30m",
            "options": {"temperature": 0.1, "num_ctx": 6144}
        });
        if use_format {
            body["format"] = command_schema();
        }
        if think_flag {
            body["think"] = json!(false);
        }
        let mut resp = agent.post(&format!("{host}/api/chat")).send_json(&body).map_err(|e| format!("review request failed: {e}"))?;
        let status = resp.status();
        let v: Value = resp.body_mut().read_json().map_err(|e| format!("bad review response: {e}"))?;
        if status.is_success() && v.get("error").is_none() {
            let content = v["message"]["content"].as_str().unwrap_or("");
            let (reply, cmds, _) = crate::ai::parse_ai_content(content)?;
            return Ok((reply, cmds));
        }
        last_err = v["error"].as_str().unwrap_or("unknown error").to_string();
    }
    Err(format!("review model error: {last_err}"))
}
