//! MCP (Model Context Protocol) server: lets external AI agents (Claude Desktop, Claude Code,
//! Cursor, VS Code, …) drive the running Rust Blender app.
//!
//! * In-app **Streamable HTTP** endpoint: `POST http://127.0.0.1:7878/mcp` (JSON responses).
//! * **stdio** shim for clients that launch servers as a process: `rust_blender --mcp-stdio`
//!   forwards JSON-RPC lines to the running app (and starts the app if needed).
//!
//! Protocol handling (initialize, tools/list, ping) happens on the server thread; tool calls are
//! forwarded to the UI thread, which owns the scene, and answered through a one-shot channel.

use crate::ai::command_schema;
use crossbeam_channel::{Receiver, Sender, bounded, unbounded};
use serde_json::{Value, json};
use std::io::{BufRead, Read, Write};
use std::time::Duration;

pub const DEFAULT_PORT: u16 = 7878;
const PROTOCOL_VERSION: &str = "2025-06-18";

/// A tool call waiting for the UI thread.
pub struct ToolCall {
    pub name: String,
    pub args: Value,
    pub client: String,
    pub reply: Sender<ToolResult>,
}

#[derive(Debug, Clone)]
pub struct ToolResult {
    pub text: String,
    pub is_error: bool,
    /// Optional PNG returned as MCP image content (render_image).
    pub image_png: Option<Vec<u8>>,
}

impl ToolResult {
    pub fn ok(text: impl Into<String>) -> Self {
        Self { text: text.into(), is_error: false, image_png: None }
    }
    pub fn err(text: impl Into<String>) -> Self {
        Self { text: text.into(), is_error: true, image_png: None }
    }
}

pub struct McpServer {
    pub rx: Receiver<ToolCall>,
    pub port: u16,
    server: std::sync::Arc<tiny_http::Server>,
}

impl Drop for McpServer {
    /// Stop listening so the port can be reused right away.
    fn drop(&mut self) {
        self.server.unblock();
    }
}

pub fn tools() -> Value {
    let cmd = command_schema()["properties"]["commands"]["items"].clone();
    json!([
        {
            "name": "run_commands",
            "description": "Run editor commands in the live Rust Blender scene (Blender-like, Z up, 1 unit = 1 m, default primitives are 2 units wide). Actions include add (primitives cube/plane/uv_sphere/ico_sphere/cylinder/cone/torus/monkey, point_light/sun/spot_light/camera), delete, select, move, rotate, scale, set_transform, set_color, rename, duplicate, add_modifier, keyframe, set_frame, play, view, shading, edit_mode, select_faces, extrude, inset, loop_cut, subdivide, object_mode, undo, redo. All commands of one call form a single undo step. Returns a per-command log.",
            "inputSchema": {"type": "object", "properties": {"commands": {"type": "array", "items": cmd}}, "required": ["commands"]}
        },
        {
            "name": "get_scene",
            "description": "Describe the current scene: every object with type, location, rotation, scale, colour and world-space bounding box, plus the selection and current frame.",
            "inputSchema": {"type": "object", "properties": {}}
        },
        {
            "name": "check_scene",
            "description": "Run Rust Blender's geometry checks on all objects: floating, sunk into the ground, hidden inside other objects, duplicates. Use after building to find mistakes.",
            "inputSchema": {"type": "object", "properties": {}}
        },
        {
            "name": "ask_copilot",
            "description": "Send a natural-language request to Rust Blender's own local AI Copilot (Ollama: decision model + builder). Returns immediately; the build appears live in the app.",
            "inputSchema": {"type": "object", "properties": {"prompt": {"type": "string"}}, "required": ["prompt"]}
        },
        {
            "name": "render_image",
            "description": "Render the scene through its camera (Material Preview shading with the scene lights, anti-aliased, no overlays) and return the image so you can see the result. Optional: width/height in pixels (default 960 wide at the scene's aspect), transparent background, and an absolute path to also save a PNG.",
            "inputSchema": {"type": "object", "properties": {"width": {"type": "integer"}, "height": {"type": "integer"}, "transparent": {"type": "boolean"}, "path": {"type": "string"}}}
        },
        {
            "name": "export_gltf",
            "description": "Export the scene as a binary glTF 2.0 (.glb) file at the given absolute path.",
            "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}
        }
    ])
}

fn rpc_result(id: &Value, result: Value) -> Value {
    json!({"jsonrpc": "2.0", "id": id, "result": result})
}
fn rpc_error(id: &Value, code: i64, msg: &str) -> Value {
    json!({"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": msg}})
}

/// Handle one JSON-RPC message. Returns None for notifications.
pub fn handle_message(msg: &Value, client: &str, calls: &Sender<ToolCall>, wake: &dyn Fn()) -> Option<Value> {
    let id = msg.get("id").cloned();
    let method = msg["method"].as_str().unwrap_or("");
    let id = id?; // notifications (no id) need no response
    Some(match method {
        "initialize" => {
            let requested = msg["params"]["protocolVersion"].as_str().unwrap_or(PROTOCOL_VERSION);
            rpc_result(
                &id,
                json!({
                    "protocolVersion": requested,
                    "capabilities": {"tools": {"listChanged": false}},
                    "serverInfo": {"name": "rust-blender", "title": "Rust Blender", "version": env!("CARGO_PKG_VERSION")},
                    "instructions": "Rust Blender is a Blender-like 3D editor. Call get_scene first, build with run_commands (batch several commands per call), then call check_scene and fix real problems."
                }),
            )
        }
        "ping" => rpc_result(&id, json!({})),
        "tools/list" => rpc_result(&id, json!({"tools": tools()})),
        "tools/call" => {
            let name = msg["params"]["name"].as_str().unwrap_or("").to_string();
            let args = msg["params"].get("arguments").cloned().unwrap_or(json!({}));
            let (tx, rx) = bounded(1);
            if calls.send(ToolCall { name, args, client: client.to_string(), reply: tx }).is_err() {
                return Some(rpc_error(&id, -32603, "Rust Blender is shutting down"));
            }
            wake();
            match rx.recv_timeout(Duration::from_secs(60)) {
                Ok(r) => {
                    let mut content = vec![json!({"type": "text", "text": r.text})];
                    if let Some(png) = &r.image_png {
                        content.push(json!({"type": "image", "data": base64(png), "mimeType": "image/png"}));
                    }
                    rpc_result(&id, json!({"content": content, "isError": r.is_error}))
                }
                Err(_) => rpc_error(&id, -32603, "Rust Blender did not answer in time (is a modal dialog open?)"),
            }
        }
        "resources/list" => rpc_result(&id, json!({"resources": []})),
        "prompts/list" => rpc_result(&id, json!({"prompts": []})),
        _ => rpc_error(&id, -32601, &format!("method not found: {method}")),
    })
}

/// Standard base64 (RFC 4648, padded).
pub fn base64(data: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for c in data.chunks(3) {
        let n = (c[0] as u32) << 16 | (*c.get(1).unwrap_or(&0) as u32) << 8 | *c.get(2).unwrap_or(&0) as u32;
        out.push(T[(n >> 18) as usize & 63] as char);
        out.push(T[(n >> 12) as usize & 63] as char);
        out.push(if c.len() > 1 { T[(n >> 6) as usize & 63] as char } else { '=' });
        out.push(if c.len() > 2 { T[n as usize & 63] as char } else { '=' });
    }
    out
}

/// Reject browser pages on other origins (DNS-rebinding protection, as the MCP spec requires).
fn origin_ok(origin: Option<&str>) -> bool {
    match origin {
        None => true,
        Some(o) => {
            let o = o.to_lowercase();
            o.starts_with("http://localhost") || o.starts_with("http://127.0.0.1") || o.starts_with("https://localhost") || o == "null"
        }
    }
}

impl McpServer {
    /// Start the HTTP endpoint on 127.0.0.1:`port`.
    pub fn start(port: u16, wake: impl Fn() + Send + 'static) -> Result<Self, String> {
        let server = std::sync::Arc::new(tiny_http::Server::http(("127.0.0.1", port)).map_err(|e| format!("can't listen on 127.0.0.1:{port}: {e}"))?);
        let srv = server.clone();
        let (tx, rx) = unbounded::<ToolCall>();
        std::thread::Builder::new()
            .name("mcp".into())
            .spawn(move || {
                for mut req in srv.incoming_requests() {
                    let origin = req.headers().iter().find(|h| h.field.equiv("Origin")).map(|h| h.value.as_str().to_string());
                    let client = req.headers().iter().find(|h| h.field.equiv("User-Agent")).map(|h| h.value.as_str().to_string()).unwrap_or_else(|| "MCP client".into());
                    let json_hdr = tiny_http::Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap();
                    if !origin_ok(origin.as_deref()) {
                        let _ = req.respond(tiny_http::Response::from_string("forbidden origin").with_status_code(403));
                        continue;
                    }
                    let path = req.url().split('?').next().unwrap_or("").to_string();
                    if path != "/mcp" {
                        let _ = req.respond(tiny_http::Response::from_string("Rust Blender MCP endpoint is POST /mcp").with_status_code(404));
                        continue;
                    }
                    if *req.method() != tiny_http::Method::Post {
                        // no server-initiated SSE stream
                        let _ = req.respond(tiny_http::Response::from_string("").with_status_code(405));
                        continue;
                    }
                    let mut body = String::new();
                    let _ = req.as_reader().take(8 * 1024 * 1024).read_to_string(&mut body);
                    let parsed: Result<Value, _> = serde_json::from_str(&body);
                    let reply = match parsed {
                        Err(e) => Some(rpc_error(&Value::Null, -32700, &format!("parse error: {e}"))),
                        Ok(Value::Array(batch)) => {
                            let out: Vec<Value> = batch.iter().filter_map(|m| handle_message(m, &client, &tx, &wake)).collect();
                            (!out.is_empty()).then_some(Value::Array(out))
                        }
                        Ok(m) => handle_message(&m, &client, &tx, &wake),
                    };
                    let _ = match reply {
                        Some(v) => req.respond(tiny_http::Response::from_string(v.to_string()).with_header(json_hdr)),
                        None => req.respond(tiny_http::Response::from_string("").with_status_code(202)),
                    };
                }
            })
            .map_err(|e| e.to_string())?;
        Ok(Self { rx, port, server })
    }
}

/// Claude Desktop / Cursor style config snippet for this executable.
pub fn client_config() -> String {
    let exe = std::env::current_exe().map(|p| p.display().to_string()).unwrap_or_else(|_| "rust_blender".into());
    serde_json::to_string_pretty(&json!({"mcpServers": {"rust-blender": {"command": exe, "args": ["--mcp-stdio"]}}})).unwrap()
}

/// `rust_blender --mcp-stdio`: bridge stdin/stdout JSON-RPC to the running app's HTTP endpoint.
/// Starts the app if it isn't running.
pub fn run_stdio_bridge() {
    let port: u16 = std::env::var("RUST_BLENDER_MCP_PORT").ok().and_then(|p| p.parse().ok()).unwrap_or(DEFAULT_PORT);
    let url = format!("http://127.0.0.1:{port}/mcp");
    let agent: ureq::Agent = ureq::Agent::config_builder().timeout_connect(Some(Duration::from_secs(2))).timeout_global(Some(Duration::from_secs(90))).http_status_as_error(false).build().into();
    let mut launched = false;
    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout();
    for line in stdin.lock().lines() {
        let Ok(line) = line else { break };
        if line.trim().is_empty() {
            continue;
        }
        let id = serde_json::from_str::<Value>(&line).ok().and_then(|v| v.get("id").cloned());
        let mut attempt = 0;
        let out = loop {
            match agent.post(&url).header("Content-Type", "application/json").header("User-Agent", "stdio-bridge").header("Accept", "application/json, text/event-stream").send(line.as_bytes()) {
                Ok(mut resp) => break resp.body_mut().read_to_string().ok().filter(|s| !s.trim().is_empty()),
                Err(_) if attempt == 0 && !launched => {
                    // App not running: start it and wait for the endpoint.
                    launched = true;
                    if let Ok(exe) = std::env::current_exe() {
                        let _ = std::process::Command::new(exe).stdin(std::process::Stdio::null()).stdout(std::process::Stdio::null()).stderr(std::process::Stdio::null()).spawn();
                    }
                    let t0 = std::time::Instant::now();
                    while t0.elapsed() < Duration::from_secs(30) && std::net::TcpStream::connect(("127.0.0.1", port)).is_err() {
                        std::thread::sleep(Duration::from_millis(300));
                    }
                    attempt += 1;
                }
                Err(e) => {
                    break id.as_ref().map(|id| rpc_error(id, -32000, &format!("Rust Blender is not running and could not be started ({e}). Open the app and make sure the MCP server is enabled in Copilot settings.")).to_string());
                }
            }
        };
        if let Some(o) = out {
            let _ = writeln!(stdout, "{}", o.replace('\n', " "));
            let _ = stdout.flush();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn protocol_basics() {
        let (tx, rx) = unbounded::<ToolCall>();
        let init = handle_message(&json!({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-03-26"}}), "t", &tx, &|| {}).unwrap();
        assert_eq!(init["result"]["protocolVersion"], "2025-03-26");
        assert!(init["result"]["capabilities"]["tools"].is_object());
        assert!(handle_message(&json!({"jsonrpc": "2.0", "method": "notifications/initialized"}), "t", &tx, &|| {}).is_none());
        let list = handle_message(&json!({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}), "t", &tx, &|| {}).unwrap();
        let names: Vec<&str> = list["result"]["tools"].as_array().unwrap().iter().map(|t| t["name"].as_str().unwrap()).collect();
        assert_eq!(names, vec!["run_commands", "get_scene", "check_scene", "ask_copilot", "render_image", "export_gltf"]);
        assert_eq!(base64(b"Man"), "TWFu");
        assert_eq!(base64(b"Ma"), "TWE=");
        assert_eq!(base64(b"M"), "TQ==");
        // tools/call is forwarded to the "UI thread"
        let h = std::thread::spawn(move || {
            let c = rx.recv().unwrap();
            assert_eq!(c.name, "get_scene");
            c.reply.send(ToolResult::ok("scene!")).unwrap();
        });
        let r = handle_message(&json!({"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "get_scene", "arguments": {}}}), "t", &tx, &|| {}).unwrap();
        h.join().unwrap();
        assert_eq!(r["result"]["content"][0]["text"], "scene!");
        assert_eq!(r["result"]["isError"], false);
        let bad = handle_message(&json!({"jsonrpc": "2.0", "id": 4, "method": "nope"}), "t", &tx, &|| {}).unwrap();
        assert_eq!(bad["error"]["code"], -32601);
        assert!(origin_ok(None) && origin_ok(Some("http://localhost:3000")) && !origin_ok(Some("https://evil.example")));
    }
}
