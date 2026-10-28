//! OpenRouter (https://openrouter.ai): cloud models — Claude, GPT, Gemini, Llama, Qwen,
//! DeepSeek … — through one OpenAI-compatible API and an API key. Any model name written as
//! `openrouter:<id>` (e.g. `openrouter:anthropic/claude-sonnet-4.5`) goes here instead of Ollama,
//! for the builder, the decision model, the reviewer and picture reading alike.

use serde_json::{Value, json};
use std::sync::RwLock;

pub const PREFIX: &str = "openrouter:";
pub const KEYS_URL: &str = "https://openrouter.ai/keys";

static KEY: RwLock<String> = RwLock::new(String::new());

/// API base; `RB_OPENROUTER_URL` overrides it (tests, proxies).
fn base() -> String {
    std::env::var("RB_OPENROUTER_URL").unwrap_or_else(|_| "https://openrouter.ai/api/v1".into()).trim_end_matches('/').to_string()
}

/// Set the API key (an empty key falls back to the `OPENROUTER_API_KEY` environment variable).
pub fn set_key(k: &str) {
    if let Ok(mut g) = KEY.write() {
        *g = k.trim().to_string();
    }
}

pub fn key() -> String {
    let k = KEY.read().map(|g| g.clone()).unwrap_or_default();
    if k.is_empty() { std::env::var("OPENROUTER_API_KEY").unwrap_or_default().trim().to_string() } else { k }
}

pub fn has_key() -> bool {
    !key().is_empty()
}

pub fn is_remote(model: &str) -> bool {
    model.starts_with(PREFIX)
}

/// `openrouter:vendor/model` → `vendor/model`.
pub fn model_id(model: &str) -> &str {
    model.strip_prefix(PREFIX).unwrap_or(model)
}

fn post(agent: &ureq::Agent, path: &str, body: &Value) -> Result<ureq::http::Response<ureq::Body>, String> {
    let k = key();
    if k.is_empty() {
        return Err(format!("no OpenRouter API key — add one in the Copilot settings (get one at {KEYS_URL})"));
    }
    agent
        .post(&format!("{}{path}", base()))
        .header("Authorization", &format!("Bearer {k}"))
        .header("HTTP-Referer", "https://github.com/rholmes69/Rust_Blender")
        .header("X-Title", "Rust Blender")
        .send_json(body)
        .map_err(|e| format!("can't reach OpenRouter: {e}"))
}

fn error_text(status: u16, v: &Value) -> String {
    let msg = v["error"]["message"].as_str().or_else(|| v["error"].as_str()).unwrap_or("unknown error");
    let hint = match status {
        401 => " — check the API key",
        402 => " — the account is out of credits",
        429 => " — rate limited, try again in a moment",
        _ => "",
    };
    format!("OpenRouter error {status}: {msg}{hint}")
}

/// Options for one request.
#[derive(Clone, Default)]
pub struct Opts {
    pub temperature: f32,
    /// Ask for a JSON object reply (`response_format`); dropped automatically if the model
    /// doesn't support it.
    pub json: bool,
    /// OpenAI-style tools.
    pub tools: Option<Value>,
    pub max_tokens: Option<u32>,
}

fn body(model: &str, messages: &[Value], o: &Opts, stream: bool) -> Value {
    let mut b = json!({"model": model_id(model), "messages": messages, "temperature": o.temperature, "stream": stream});
    if o.json {
        b["response_format"] = json!({"type": "json_object"});
    }
    if let Some(t) = &o.tools {
        b["tools"] = t.clone();
    }
    if let Some(m) = o.max_tokens {
        b["max_tokens"] = json!(m);
    }
    b
}

/// One complete answer: the assistant message (`content`, maybe `tool_calls`).
pub fn complete(agent: &ureq::Agent, model: &str, messages: &[Value], o: &Opts) -> Result<Value, String> {
    let mut o = o.clone();
    for _ in 0..2 {
        let mut resp = post(agent, "/chat/completions", &body(model, messages, &o, false))?;
        let status = resp.status().as_u16();
        let v: Value = resp.body_mut().read_json().map_err(|e| format!("bad response from OpenRouter: {e}"))?;
        if status >= 400 || v.get("error").is_some() {
            // some models don't do JSON mode: try once more without it
            if o.json && status == 400 {
                o.json = false;
                continue;
            }
            return Err(error_text(status, &v));
        }
        let msg = v["choices"][0]["message"].clone();
        if msg.is_null() {
            return Err(format!("OpenRouter returned no answer: {v}"));
        }
        return Ok(msg);
    }
    Err("OpenRouter request failed".into())
}

/// Stream an answer; `on_text` gets each piece of content. Returns the whole text.
pub fn stream(agent: &ureq::Agent, model: &str, messages: &[Value], o: &Opts, cancel: &std::sync::atomic::AtomicBool, on_text: &mut dyn FnMut(&str)) -> Result<String, String> {
    use std::io::BufRead;
    let mut o = o.clone();
    for _ in 0..2 {
        let resp = post(agent, "/chat/completions", &body(model, messages, &o, true))?;
        let status = resp.status().as_u16();
        if status >= 400 {
            let mut resp = resp;
            let v: Value = resp.body_mut().read_json().unwrap_or_default();
            if o.json && status == 400 {
                o.json = false;
                continue;
            }
            return Err(error_text(status, &v));
        }
        let reader = std::io::BufReader::new(resp.into_body().into_with_config().limit(64 * 1024 * 1024).reader());
        let mut all = String::new();
        for line in reader.lines() {
            if cancel.load(std::sync::atomic::Ordering::Relaxed) {
                return Err("cancelled".into());
            }
            let line = line.map_err(|e| e.to_string())?;
            let Some(data) = line.trim().strip_prefix("data:") else { continue }; // ": OPENROUTER PROCESSING" keep-alives
            let data = data.trim();
            if data == "[DONE]" {
                break;
            }
            let Ok(v) = serde_json::from_str::<Value>(data) else { continue };
            if v.get("error").is_some() {
                return Err(error_text(v["error"]["code"].as_u64().unwrap_or(500) as u16, &v));
            }
            if let Some(t) = v["choices"][0]["delta"]["content"].as_str() {
                if !t.is_empty() {
                    all.push_str(t);
                    on_text(t);
                }
            }
        }
        return Ok(all);
    }
    Err("OpenRouter request failed".into())
}

/// A model in OpenRouter's catalogue.
#[derive(Clone, Debug, PartialEq)]
pub struct ModelInfo {
    pub id: String,
    pub name: String,
    pub context: u64,
    /// USD per million tokens (prompt, completion); 0 = free.
    pub price_in: f64,
    pub price_out: f64,
    pub vision: bool,
    pub tools: bool,
}

pub fn parse_models(v: &Value) -> Vec<ModelInfo> {
    let per_m = |x: &Value| -> f64 { x.as_str().and_then(|s| s.parse::<f64>().ok()).or_else(|| x.as_f64()).unwrap_or(0.0) * 1e6 };
    let mut out: Vec<ModelInfo> = v["data"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|m| {
                    let id = m["id"].as_str()?.to_string();
                    let inputs: Vec<&str> = m["architecture"]["input_modalities"].as_array().map(|x| x.iter().filter_map(|s| s.as_str()).collect()).unwrap_or_default();
                    let params: Vec<&str> = m["supported_parameters"].as_array().map(|x| x.iter().filter_map(|s| s.as_str()).collect()).unwrap_or_default();
                    Some(ModelInfo {
                        name: m["name"].as_str().unwrap_or(&id).to_string(),
                        context: m["context_length"].as_u64().unwrap_or(0),
                        price_in: per_m(&m["pricing"]["prompt"]),
                        price_out: per_m(&m["pricing"]["completion"]),
                        vision: inputs.contains(&"image") || m["architecture"]["modality"].as_str().is_some_and(|s| s.contains("image")),
                        tools: params.contains(&"tools"),
                        id,
                    })
                })
                .collect()
        })
        .unwrap_or_default();
    out.sort_by(|a, b| a.id.cmp(&b.id));
    out
}

/// The catalogue (public; no key needed).
pub fn list_models(agent: &ureq::Agent) -> Result<Vec<ModelInfo>, String> {
    let mut resp = agent.get(&format!("{}/models", base())).call().map_err(|e| format!("can't reach OpenRouter: {e}"))?;
    let status = resp.status().as_u16();
    let v: Value = resp.body_mut().read_json().map_err(|e| format!("bad model list from OpenRouter: {e}"))?;
    if status >= 400 {
        return Err(error_text(status, &v));
    }
    Ok(parse_models(&v))
}

/// Check the key: a short description of the account (usage / limit).
pub fn check_key(agent: &ureq::Agent) -> Result<String, String> {
    let k = key();
    if k.is_empty() {
        return Err("no API key".into());
    }
    let mut resp = agent.get(&format!("{}/key", base())).header("Authorization", &format!("Bearer {k}")).call().map_err(|e| format!("can't reach OpenRouter: {e}"))?;
    let status = resp.status().as_u16();
    let v: Value = resp.body_mut().read_json().unwrap_or_default();
    if status >= 400 {
        return Err(error_text(status, &v));
    }
    let d = &v["data"];
    let used = d["usage"].as_f64().unwrap_or(0.0);
    let limit = d["limit"].as_f64();
    let free = d["is_free_tier"].as_bool().unwrap_or(false);
    Ok(match limit {
        Some(l) => format!("Key OK — ${used:.2} used of ${l:.2}{}", if free { " (free tier)" } else { "" }),
        None => format!("Key OK — ${used:.2} used{}", if free { " (free tier)" } else { "" }),
    })
}

/// A chat message with a picture (base64 PNG) for vision models.
pub fn image_message(text: &str, png_b64: &str) -> Value {
    json!({"role": "user", "content": [
        {"type": "text", "text": text},
        {"type": "image_url", "image_url": {"url": format!("data:image/png;base64,{png_b64}")}}
    ]})
}

/// Good starting points if they're in the catalogue (builder, cheap/fast, vision).
pub const SUGGESTED: &[&str] = &["anthropic/claude-sonnet-4.5", "anthropic/claude-haiku-4.5", "openai/gpt-4.1-mini", "openai/gpt-4o-mini", "google/gemini-2.5-flash", "google/gemini-2.5-pro", "qwen/qwen3-coder", "deepseek/deepseek-chat-v3.1", "meta-llama/llama-3.3-70b-instruct"];

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicBool;

    /// A local stand-in for the OpenRouter API.
    fn mock() -> (String, std::thread::JoinHandle<Vec<String>>) {
        let server = tiny_http::Server::http("127.0.0.1:0").unwrap();
        let addr = format!("http://{}", server.server_addr().to_ip().unwrap());
        let h = std::thread::spawn(move || {
            let mut seen = vec![];
            for _ in 0..3 {
                let Ok(mut rq) = server.recv() else { break };
                let auth = rq.headers().iter().find(|h| h.field.equiv("Authorization")).map(|h| h.value.to_string()).unwrap_or_default();
                let mut raw = vec![];
                let _ = rq.as_reader().read_to_end(&mut raw);
                // compact whatever JSON came in
                let b = serde_json::from_slice::<Value>(&raw).map(|v| v.to_string()).unwrap_or_default();
                seen.push(format!("{} {} {auth} {b}", rq.method(), rq.url()));
                let url = rq.url().to_string();
                let resp = if url.ends_with("/models") {
                    tiny_http::Response::from_string(r#"{"data":[{"id":"z/vision-1","name":"Z Vision","context_length":32000,"pricing":{"prompt":"0.000001","completion":"0.000002"},"architecture":{"input_modalities":["text","image"]},"supported_parameters":["tools","response_format"]},{"id":"a/text","name":"A","context_length":8000,"pricing":{"prompt":"0","completion":"0"},"architecture":{"input_modalities":["text"]}}]}"#)
                } else if b.contains("\"stream\":true") {
                    tiny_http::Response::from_string(": OPENROUTER PROCESSING\n\ndata: {\"choices\":[{\"delta\":{\"content\":\"{\\\"reply\\\":\\\"Hi\"}}]}\n\ndata: {\"choices\":[{\"delta\":{\"content\":\"\\\",\\\"commands\\\":[]}\"}}]}\n\ndata: [DONE]\n\n")
                } else {
                    tiny_http::Response::from_string(r#"{"choices":[{"message":{"role":"assistant","content":"{\"route\":\"answer\",\"reply\":\"Press Tab.\"}"}}]}"#)
                };
                let _ = rq.respond(resp);
            }
            seen
        });
        (addr, h)
    }

    #[test]
    fn talks_openai_style_with_the_key() {
        let (addr, h) = mock();
        // SAFETY: only this test uses RB_OPENROUTER_URL
        unsafe { std::env::set_var("RB_OPENROUTER_URL", &addr) };
        set_key("sk-or-test");
        assert!(is_remote("openrouter:a/b") && model_id("openrouter:a/b") == "a/b" && !is_remote("qwen3:8b"));
        let agent: ureq::Agent = ureq::Agent::config_builder().http_status_as_error(false).build().into();
        let models = list_models(&agent).unwrap();
        assert_eq!(models.len(), 2);
        let z = models.iter().find(|m| m.id == "z/vision-1").unwrap();
        assert!(z.vision && z.tools && (z.price_in - 1.0).abs() < 1e-9 && z.context == 32000);
        let msgs = vec![json!({"role": "user", "content": "hi"})];
        let mut pieces = vec![];
        let text = stream(&agent, "openrouter:a/text", &msgs, &Opts { json: true, ..Default::default() }, &AtomicBool::new(false), &mut |t| pieces.push(t.to_string())).unwrap();
        if text.is_empty() {
            unsafe { std::env::remove_var("RB_OPENROUTER_URL") };
            panic!("empty stream; requests: {:?}", h.join().unwrap());
        }
        assert_eq!(text, r#"{"reply":"Hi","commands":[]}"#);
        assert_eq!(pieces.len(), 2);
        let m = complete(&agent, "openrouter:a/text", &msgs, &Opts::default()).unwrap();
        assert!(m["content"].as_str().unwrap().contains("Press Tab"));
        unsafe { std::env::remove_var("RB_OPENROUTER_URL") };
        let seen = h.join().unwrap();
        assert!(seen.iter().filter(|s| s.contains("/chat/completions")).all(|s| s.contains("Bearer sk-or-test") && s.contains("\"model\":\"a/text\"")), "{seen:?}");
        assert!(seen.iter().any(|s| s.contains("response_format")));
    }
}
