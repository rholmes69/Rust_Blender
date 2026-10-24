//! Main application: layout, panels, AI copilot, voice, menus, timeline, file IO.

use crate::ai::{self, AiClient, AiEvent, AiRequest, ChatMsg};
use crate::editor::{Command, Editor, ViewPreset, ViewRequest};
use crate::mesh::Primitive;
use crate::scene::{Modifier, Scene};
use crate::theme::{self, color32};
use crate::viewport::{Tool, Viewport};
use crate::voice::{self, VoiceEngine, VoiceEvent};
use egui::{
    Align, Align2, Color32, CornerRadius, FontId, Key, Layout, Margin, RichText, Sense, Stroke, Ui,
    vec2,
};
use glam::Vec3;
use serde::{Deserialize, Serialize};
use std::time::Instant;

#[derive(Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    pub ollama_host: String,
    pub model: String,
    pub auto_select_model: bool,
    /// Fast decision/router model consulted before the builder model ("" = none).
    pub decision_model: String,
    /// Agent loop: inspect AI builds and let a reviewer fix mistakes.
    pub auto_check: bool,
    pub auto_decision_model: bool,
    pub auto_execute: bool,
    pub instant_commands: bool,
    pub voice_model: String,
    pub hands_free: bool,
    pub vad_threshold: f32,
    pub silence_secs: f32,
    pub speak_replies: bool,
    pub show_assistant: bool,
    pub ui_scale: f32,
    pub gpu_renderer: bool,
    /// MCP server for external agents (Claude Desktop, Claude Code, Cursor …).
    pub mcp_enabled: bool,
    pub mcp_port: u16,
    /// Speak a short confirmation after each voice request.
    pub voice_confirm: bool,
    /// Keep the mic open for a follow-up after each spoken reply.
    pub conversation: bool,
    /// Listen in the background for the wake phrase.
    pub wake_word: bool,
    pub wake_phrase: String,
    /// Vision model for pictures ("" = pick automatically).
    pub vision_model: String,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            ollama_host: std::env::var("OLLAMA_HOST")
                .map(|h| {
                    if h.starts_with("http") {
                        h
                    } else {
                        format!("http://{h}")
                    }
                })
                .unwrap_or_else(|_| "http://localhost:11434".into()),
            model: String::new(),
            auto_select_model: true,
            decision_model: String::new(),
            auto_check: true,
            auto_decision_model: true,
            auto_execute: true,
            instant_commands: true,
            voice_model: "base.en".into(),
            hands_free: true,
            vad_threshold: 0.02,
            silence_secs: 1.1,
            speak_replies: false,
            show_assistant: true,
            ui_scale: 1.0,
            gpu_renderer: true,
            mcp_enabled: true,
            mcp_port: crate::mcp::DEFAULT_PORT,
            voice_confirm: true,
            conversation: true,
            wake_word: false,
            wake_phrase: "blender".into(),
            vision_model: String::new(),
        }
    }
}

impl Settings {
    fn path() -> std::path::PathBuf {
        dirs::config_dir()
            .unwrap_or_else(std::env::temp_dir)
            .join("rust_blender")
            .join("settings.json")
    }
    fn load() -> Self {
        std::fs::read_to_string(Self::path())
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default()
    }
    fn save(&self) {
        let p = Self::path();
        let _ = std::fs::create_dir_all(p.parent().unwrap());
        let _ = std::fs::write(p, serde_json::to_string_pretty(self).unwrap_or_default());
    }
}

#[derive(Clone, Debug, PartialEq)]
enum OllamaStatus {
    Checking,
    Online,
    Offline(String),
}

#[derive(Clone, Debug, PartialEq)]
enum VoiceStatus {
    NotLoaded,
    Loading(String),
    Downloading(String, f32),
    Ready(String),
    Error(String),
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Role {
    User,
    Assistant,
    System,
}

#[derive(Clone, Debug)]
struct ChatEntry {
    role: Role,
    text: String,
    commands: Vec<Command>,
    log: Vec<String>,
    applied: bool,
    voice: bool,
    instant: bool,
    /// Which model(s) handled the request, e.g. "handled by tev1".
    via: Option<String>,
    streaming: bool,
    secs: f32,
}

impl ChatEntry {
    fn new(role: Role, text: impl Into<String>) -> Self {
        Self {
            role,
            text: text.into(),
            commands: vec![],
            log: vec![],
            applied: false,
            voice: false,
            instant: false,
            via: None,
            streaming: false,
            secs: 0.0,
        }
    }
}

struct ReviewState {
    request: String,
    subjects: Vec<u64>,
    round: u32,
    idx: usize,
}

struct Toast {
    text: String,
    color: Color32,
    born: Instant,
}

pub struct App {
    ed: Editor,
    vp: Viewport,
    ai: AiClient,
    voice: VoiceEngine,
    settings: Settings,
    models: Vec<String>,
    ollama: OllamaStatus,
    chat: Vec<ChatEntry>,
    history: Vec<ChatMsg>,
    input: String,
    waiting_ai: Option<Instant>,
    /// (chat index, undo snapshot taken) for the answer currently streaming
    streaming: Option<(usize, bool)>,
    /// models we already warned about running partly on the CPU
    warned_cpu: Vec<String>,
    ai_cancelled: bool,
    /// Current stage of a two-model request ("tev1 deciding", "qwen3:8b building").
    route_stage: Option<String>,
    pending_via: Option<String>,
    router_warned: bool,
    /// Agent loop state for the build being checked.
    review: Option<ReviewState>,
    pending_fixes: Vec<Command>,
    last_prompt: String,
    pull: Option<(String, f32)>,
    voice_status: VoiceStatus,
    transcribing: bool,
    pending_voice: bool,
    palette: Option<String>,
    palette_sel: usize,
    toasts: Vec<Toast>,
    play_accum: f64,
    last_time: f64,
    prop_snapshotted: bool,
    show_settings: bool,
    show_help: bool,
    focus_prompt: bool,
    props_tab: PropsTab,
    rename: Option<(u64, String)>,
    last_ollama_check: Instant,
    last_title: String,
    /// Picture attached to the next Copilot request: (name, base64 PNG, thumbnail).
    pending_image: Option<(String, String, egui::TextureHandle)>,
    /// Request text waiting for the vision model's description.
    vision_request: Option<String>,
    assistant_rect: egui::Rect,
    ctx_handle: egui::Context,
    /// Autosave every couple of minutes; offer recovery after a crash.
    last_autosave: Instant,
    recovery: Option<std::path::PathBuf>,
    /// UV editor window + cached background texture (texture id, signature, handle).
    show_uv: bool,
    uv_bg: Option<(u32, u64, egui::TextureHandle)>,
    uv_drag: Option<Vec<Vec<[f32; 2]>>>,
    outliner_collapsed: std::collections::HashSet<u64>,
    outliner_collapsed_cols: std::collections::HashSet<u32>,
    /// "Move to collection" popup (M)
    collection_menu: bool,
    /// Running MCP server (None = disabled or failed to start).
    mcp: Option<crate::mcp::McpServer>,
    mcp_status: String,
    mcp_calls: u32,
    /// Last F12 render and whether its window is open.
    render: Option<RenderView>,
    show_render: bool,
    /// Hands-free voice session.
    speaker: crate::voiceflow::Speaker,
    /// Chat index where the voice request currently being handled starts.
    voice_turn: Option<usize>,
    /// Re-open the mic for a follow-up once the spoken reply has finished.
    relisten: bool,
    /// What the current recording / transcription is for.
    listen_purpose: Listen,
    transcribe_purpose: Listen,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Listen {
    /// Mic button / shortcut.
    Command,
    /// Automatic follow-up after a spoken reply (gives up after a few silent seconds).
    FollowUp,
    /// Background listening for the wake phrase.
    Wake,
}

struct RenderView {
    result: crate::viewport::RenderResult,
    tex: egui::TextureHandle,
    saved: Option<std::path::PathBuf>,
    /// Running path trace that keeps refining `result`.
    job: Option<crate::pathtrace::PtJob>,
    progress: Option<(u32, u32, f32)>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum PropsTab {
    Object,
    Modifiers,
    Material,
    World,
    Scene,
}

impl App {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        theme::apply(&cc.egui_ctx);
        let settings = Settings::load();
        cc.egui_ctx
            .set_zoom_factor(settings.ui_scale.clamp(0.6, 2.5));
        let c1 = cc.egui_ctx.clone();
        let c2 = cc.egui_ctx.clone();
        let ai = AiClient::new(
            settings.ollama_host.trim_end_matches('/').to_string(),
            move || c1.request_repaint(),
        );
        ai.send(AiRequest::ListModels);
        let voice = VoiceEngine::new(move || c2.request_repaint());
        let mut vp = Viewport::default();
        if let Some(rs) = cc.wgpu_render_state.as_ref() {
            vp.gpu = crate::gpu::GpuHandle::install(rs);
        }
        vp.use_gpu = settings.gpu_renderer;
        let (mcp, mcp_status) = if settings.mcp_enabled {
            let c3 = cc.egui_ctx.clone();
            match crate::mcp::McpServer::start(settings.mcp_port, move || c3.request_repaint()) {
                Ok(m) => (Some(m), format!("listening on http://127.0.0.1:{}/mcp", settings.mcp_port)),
                Err(e) => (None, e),
            }
        } else {
            (None, "off".to_string())
        };
        let mut chat = vec![];
        chat.push(ChatEntry::new(
            Role::System,
            "Welcome to Rust Blender! Type or speak what you want to build — e.g. “make a snowman”, “add a ring of 8 glowing cubes”, or “rotate it 45 degrees”. Simple commands run instantly; complex ones go to your local Ollama model.",
        ));
        Self {
            ed: Editor::default(),
            vp,
            ai,
            voice,
            settings,
            models: vec![],
            ollama: OllamaStatus::Checking,
            chat,
            history: vec![],
            input: String::new(),
            waiting_ai: None,
            streaming: None,
            warned_cpu: vec![],
            ai_cancelled: false,
            route_stage: None,
            pending_via: None,
            router_warned: false,
            review: None,
            pending_fixes: vec![],
            last_prompt: String::new(),
            pull: None,
            voice_status: VoiceStatus::NotLoaded,
            transcribing: false,
            pending_voice: false,
            palette: None,
            palette_sel: 0,
            toasts: vec![],
            play_accum: 0.0,
            last_time: 0.0,
            prop_snapshotted: false,
            show_settings: false,
            show_help: false,
            focus_prompt: false,
            props_tab: PropsTab::Object,
            rename: None,
            last_ollama_check: Instant::now(),
            last_title: String::new(),
            pending_image: None,
            vision_request: None,
            assistant_rect: egui::Rect::NOTHING,
            ctx_handle: cc.egui_ctx.clone(),
            last_autosave: Instant::now(),
            show_uv: false,
            uv_bg: None,
            uv_drag: None,
            recovery: {
                // a lock file left behind means the last session didn't exit cleanly
                let (lock, auto) = (autosave_dir().join("session.lock"), autosave_dir().join("autosave.rblend"));
                let crashed = lock.exists() && auto.exists();
                let _ = std::fs::create_dir_all(autosave_dir());
                let _ = std::fs::write(&lock, std::process::id().to_string());
                crashed.then_some(auto)
            },
            outliner_collapsed: Default::default(),
            outliner_collapsed_cols: Default::default(),
            collection_menu: false,
            mcp,
            mcp_status,
            mcp_calls: 0,
            render: None,
            show_render: false,
            speaker: Default::default(),
            voice_turn: None,
            relisten: false,
            listen_purpose: Listen::Command,
            transcribe_purpose: Listen::Command,
        }
    }

    fn toast(&mut self, text: impl Into<String>, color: Color32) {
        self.toasts.push(Toast {
            text: text.into(),
            color,
            born: Instant::now(),
        });
    }

    // ------------------------------------------------------------------------------------------
    // AI / voice plumbing
    // ------------------------------------------------------------------------------------------
    fn poll_events(&mut self) {
        while let Ok(ev) = self.ai.rx.try_recv() {
            // After Stop, drop whatever the cancelled request still sends.
            if self.ai_cancelled {
                match ev {
                    AiEvent::Delta { .. } => continue,
                    AiEvent::Reply { .. } | AiEvent::Error(_) => {
                        self.ai_cancelled = false;
                        continue;
                    }
                    _ => {}
                }
            }
            match ev {
                AiEvent::Models(Ok(list)) => {
                    self.ollama = OllamaStatus::Online;
                    self.models = list;
                    if self.settings.auto_decision_model {
                        self.settings.decision_model = crate::router::pick_decision_model(&self.models).unwrap_or_default();
                    }
                    if self.settings.auto_select_model
                        || self.settings.model.is_empty()
                        || !self.models.contains(&self.settings.model)
                    {
                        if let Some(best) = ai::pick_best_model(&self.models) {
                            if self.settings.model != best
                                && (self.settings.auto_select_model
                                    || !self.models.contains(&self.settings.model))
                            {
                                self.settings.model = best;
                            }
                        }
                    }
                }
                AiEvent::Models(Err(e)) => {
                    self.ollama = OllamaStatus::Offline(e);
                }
                AiEvent::Reviewed { model, reply, commands, round, secs, error } => self.on_reviewed(model, reply, commands, round, secs, error),
                AiEvent::Routed { router, route, detail, secs } => {
                    match route.as_str() {
                        "delegate" => {
                            self.route_stage = Some(format!("{} building", self.settings.model));
                            let plan: String = detail.chars().take(240).collect();
                            let msg = if plan.trim().is_empty() {
                                format!("🧭 {router} → {} ({secs:.1}s): needs the builder model", self.settings.model)
                            } else {
                                format!("🧭 {router} → {} ({secs:.1}s) · plan: {plan}", self.settings.model)
                            };
                            self.chat.push(ChatEntry::new(Role::System, msg));
                        }
                        _ => {
                            self.route_stage = None;
                            self.pending_via = Some(format!("🧭 handled by {router} · {route} · {secs:.1}s"));
                        }
                    }
                }
                AiEvent::RouterFailed(e) => {
                    self.route_stage = Some(format!("{} thinking", self.settings.model));
                    if !self.router_warned {
                        self.router_warned = true;
                        self.chat.push(ChatEntry::new(Role::System, format!("⚠ Decision model failed, so {} handled this request instead. ({e})", self.settings.model)));
                    }
                }
                AiEvent::Delta { reply, commands } => {
                    let idx = self.stream_entry();
                    if !reply.is_empty() {
                        self.chat[idx].text = reply;
                    }
                    self.apply_streamed(idx, commands);
                }
                AiEvent::Reply {
                    reply,
                    commands,
                    raw,
                    secs,
                } => {
                    self.waiting_ai = None;
                    self.route_stage = None;
                    self.history.push(ChatMsg {
                        role: "assistant".into(),
                        content: raw,
                    });
                    let idx = self.stream_entry();
                    self.chat[idx].text = if reply.is_empty() {
                        "Done.".to_string()
                    } else {
                        reply.clone()
                    };
                    self.chat[idx].secs = secs;
                    self.apply_streamed(idx, commands);
                    let e = &self.chat[idx];
                    let adds = e.commands.iter().filter(|c| c.action == "add").count();
                    let has_view = e
                        .commands
                        .iter()
                        .any(|c| c.action == "view" || c.action == "focus");
                    if e.applied && adds >= 3 && !has_view {
                        self.ed.view_requests.push(ViewRequest::FrameAll);
                    }
                    self.chat[idx].streaming = false;
                    let snapped = self.streaming.is_some_and(|s| s.1);
                    self.streaming = None;
                    self.ai.send(AiRequest::Ps(self.settings.model.clone()));
                    if snapped {
                        self.start_check(idx);
                    }
                    if self.settings.speak_replies && self.voice_turn.is_none() {
                        self.speaker.speak(&reply);
                    }
                }
                AiEvent::Error(e) => {
                    self.waiting_ai = None;
                    self.route_stage = None;
                    self.pending_via = None;
                    let partial = self.streaming.take();
                    if let Some((i, _)) = partial {
                        self.chat[i].streaming = false;
                    }
                    if e == "cancelled" {
                        self.chat.push(ChatEntry::new(Role::System, if partial.is_some() { "Stopped. Changes made so far are kept (Ctrl+Z undoes them in one step)." } else { "Stopped." }));
                    } else {
                        self.chat
                            .push(ChatEntry::new(Role::System, format!("⚠ {e}")));
                        self.ai.send(AiRequest::ListModels);
                        self.ai.send(AiRequest::Ps(self.settings.model.clone()));
                    }
                }
                AiEvent::Ps(model, f) => {
                    if f < 0.9 && !self.warned_cpu.contains(&model) {
                        self.warned_cpu.push(model.clone());
                        let pct = ((1.0 - f) * 100.0).round();
                        let tip = if model.starts_with(ai::LIGHTWEIGHT_MODEL) { "Close other GPU-heavy apps, or try an even smaller model.".to_string() } else { format!("For faster replies try {} (Copilot ⚙ → Pull).", ai::LIGHTWEIGHT_MODEL) };
                        self.chat.push(ChatEntry::new(Role::System, format!("⚠ {pct}% of {model} is running on your CPU because it doesn't fully fit in GPU memory, so replies will be slow. {tip}")));
                    }
                }
                AiEvent::Described { model, text, secs } => self.on_described(model, text, secs),
                AiEvent::PullProgress(s, f) => self.pull = Some((s, f)),
                AiEvent::PullDone(r) => {
                    self.pull = None;
                    match r {
                        Ok(m) => {
                            self.toast(format!("Model {m} is ready"), theme::OK);
                            self.settings.model = m;
                        }
                        Err(e) => self.toast(format!("Pull failed: {e}"), theme::ERR),
                    }
                    self.ai.send(AiRequest::ListModels);
                }
            }
        }
        while let Ok(ev) = self.voice.rx.try_recv() {
            match ev {
                VoiceEvent::Status(s) => self.voice_status = VoiceStatus::Loading(s),
                VoiceEvent::Download(f, p) => self.voice_status = VoiceStatus::Downloading(f, p),
                VoiceEvent::Ready(n) => {
                    self.voice_status = VoiceStatus::Ready(n.clone());
                    self.toast(
                        format!("Whisper {n} loaded — voice control ready"),
                        theme::OK,
                    );
                    if self.pending_voice {
                        self.pending_voice = false;
                        self.start_recording();
                    }
                }
                VoiceEvent::Transcript { text, secs } => {
                    self.transcribing = false;
                    let clean = clean_transcript(&text);
                    match self.transcribe_purpose {
                        Listen::Wake => match crate::voiceflow::strip_wake(&clean, &self.settings.wake_phrase) {
                            None => {} // not for us; background listening resumes
                            Some(rest) if rest.trim().is_empty() => {
                                self.speaker.speak("Yes?");
                                self.relisten = true;
                            }
                            Some(rest) => self.handle_voice_text(rest, secs),
                        },
                        _ if clean.is_empty() => self.toast("Didn't catch that — try again", theme::WARN),
                        _ => self.handle_voice_text(clean, secs),
                    }
                }
                VoiceEvent::Error(e) => {
                    self.transcribing = false;
                    self.pending_voice = false;
                    if matches!(
                        self.voice_status,
                        VoiceStatus::Loading(_) | VoiceStatus::Downloading(..)
                    ) {
                        self.voice_status = VoiceStatus::Error(e.clone());
                    }
                    self.toast(e, theme::ERR);
                }
            }
        }
        if self.ollama != OllamaStatus::Online
            && self.last_ollama_check.elapsed().as_secs() > 8
            && self.waiting_ai.is_none()
        {
            self.last_ollama_check = Instant::now();
            self.ai.send(AiRequest::ListModels);
        }
    }

    /// Index of the chat entry receiving the current streamed answer (created on demand).
    fn stream_entry(&mut self) -> usize {
        if let Some(s) = &self.streaming {
            return s.0;
        }
        let mut e = ChatEntry::new(Role::Assistant, "…");
        e.via = self.pending_via.take();
        e.streaming = true;
        self.chat.push(e);
        let idx = self.chat.len() - 1;
        self.streaming = Some((idx, false));
        idx
    }

    /// Apply commands as they stream in, as a single undo step for the whole answer.
    fn apply_streamed(&mut self, idx: usize, cmds: Vec<Command>) {
        if cmds.is_empty() {
            if self.chat[idx].commands.is_empty() {
                self.chat[idx].applied = false;
            }
            return;
        }
        if self.settings.auto_execute {
            let snap = self.streaming.as_ref().is_some_and(|s| s.1);
            if !snap && cmds.iter().any(Editor::is_mutating) {
                self.ed.snapshot();
                if let Some(s) = &mut self.streaming {
                    s.1 = true;
                }
            }
            let log = self.ed.execute_no_snapshot(&cmds);
            self.chat[idx].log.extend(log);
            self.chat[idx].applied = true;
        }
        self.chat[idx].commands.extend(cmds);
    }

    // ---------------------------------------------------------------------------------------
    // Agent loop: check the result of an AI build and let a reviewer model fix mistakes
    // ---------------------------------------------------------------------------------------
    fn reviewer_model(&self) -> String {
        let d = &self.settings.decision_model;
        if !d.is_empty() && self.models.contains(d) { d.clone() } else { self.settings.model.clone() }
    }

    /// Called after an AI answer was applied (one undo snapshot was taken before it).
    fn start_check(&mut self, idx: usize) {
        if !self.settings.auto_check || !self.chat[idx].applied {
            return;
        }
        let e = &self.chat[idx];
        let failed: Vec<String> = e.log.iter().filter(|l| l.starts_with('✖')).cloned().collect();
        let builds = e.commands.iter().any(|c| matches!(c.action.as_str(), "add" | "move" | "set_transform" | "scale" | "duplicate" | "extrude"));
        if !builds && failed.is_empty() {
            return;
        }
        let Some((before, _)) = self.ed.undo_stack.last() else { return };
        let subjects = crate::agent::changed_objects(before, &self.ed.scene);
        self.review = Some(ReviewState { request: self.last_prompt.clone(), subjects, round: 0, idx });
        self.run_check(failed);
    }

    fn run_check(&mut self, failed: Vec<String>) {
        let Some(st) = self.review.as_mut() else { return };
        let issues = crate::agent::inspect(&self.ed.scene, &mut self.vp.mesh_cache, &st.subjects, &failed);
        st.round += 1;
        let round = st.round;
        let idx = st.idx;
        if issues.is_empty() {
            if round == 1 {
                if let Some(e) = self.chat.get_mut(idx) {
                    let v = e.via.take().map(|v| format!("{v} · ")).unwrap_or_default();
                    e.via = Some(format!("{v}🔍 checked: no problems found"));
                }
            } else {
                self.chat.push(ChatEntry::new(Role::System, "🔍 Re-checked: all issues resolved."));
            }
            self.review = None;
            return;
        }
        if round > 2 {
            let list: Vec<String> = issues.iter().map(|i| format!("• {}", i.text)).collect();
            self.chat.push(ChatEntry::new(Role::System, format!("🔍 Still flagged after 2 review rounds (probably intentional):\n{}", list.join("\n"))));
            self.review = None;
            return;
        }
        let list: Vec<String> = issues.iter().map(|i| format!("• {}", i.text)).collect();
        self.chat.push(ChatEntry::new(Role::System, format!("🔍 Check {round}: {} possible problem(s)\n{}", issues.len(), list.join("\n"))));
        let model = self.reviewer_model();
        let bounds = crate::agent::bounds_report(&self.ed.scene, &mut self.vp.mesh_cache);
        let req = self.review.as_ref().map(|r| r.request.clone()).unwrap_or_default();
        let prompt = crate::agent::review_prompt(&req, &issues, &bounds);
        self.pending_fixes = issues.into_iter().filter_map(|i| i.auto_fix).collect();
        self.route_stage = Some(format!("{model} reviewing"));
        self.waiting_ai = Some(Instant::now());
        self.ai.send(AiRequest::Review { model, prompt, round });
    }

    fn on_reviewed(&mut self, model: String, reply: String, commands: Vec<Command>, round: u32, secs: f32, error: Option<String>) {
        self.waiting_ai = None;
        self.route_stage = None;
        if self.review.as_ref().is_none_or(|r| r.round != round) {
            return; // stale
        }
        let (cmds, note) = match error {
            Some(e) => (std::mem::take(&mut self.pending_fixes), format!("reviewer unavailable ({e}); applied safe automatic fixes")),
            None => (commands, String::new()),
        };
        self.pending_fixes.clear();
        if cmds.is_empty() {
            let mut e = ChatEntry::new(Role::Assistant, if reply.is_empty() { "Those look intentional — no changes." } else { reply.as_str() });
            e.via = Some(format!("🔍 review by {} · round {round} · {secs:.1}s", short_model(&model)));
            self.chat.push(e);
            self.review = None;
            return;
        }
        // Fixes join the build's undo step (no new snapshot).
        let log = self.ed.execute_no_snapshot(&cmds);
        let failed: Vec<String> = log.iter().filter(|l| l.starts_with('✖')).cloned().collect();
        let mut e = ChatEntry::new(Role::Assistant, if reply.is_empty() { "Fixed.".to_string() } else { reply });
        e.commands = cmds;
        e.log = log;
        e.applied = true;
        e.secs = secs;
        e.via = Some(if note.is_empty() { format!("🔍 review by {} · round {round}", short_model(&model)) } else { format!("🔍 {note}") });
        self.chat.push(e);
        // include anything the fixes added in the next check
        if let (Some(st), Some((before, _))) = (self.review.as_mut(), self.ed.undo_stack.last()) {
            st.subjects = crate::agent::changed_objects(before, &self.ed.scene);
        }
        self.run_check(failed);
    }

    fn submit_prompt(&mut self, text: String, from_voice: bool, stt_secs: f32) {
        if let Some(img) = self.pending_image.take() {
            self.submit_with_image(text.trim().to_string(), img, from_voice);
            return;
        }
        let text = text.trim().to_string();
        self.last_prompt = text.clone();
        if text.is_empty() {
            return;
        }
        let mut u = ChatEntry::new(Role::User, text.clone());
        u.voice = from_voice;
        u.secs = stt_secs;
        self.chat.push(u);

        let online = self.ollama == OllamaStatus::Online && !self.settings.model.is_empty();
        if self.settings.instant_commands || !online {
            if let Some(cmds) = ai::parse_simple(&text) {
                let log = self.ed.execute(&cmds);
                let ok = log.iter().all(|l| l.starts_with('✔'));
                let mut e = ChatEntry::new(
                    Role::Assistant,
                    if ok {
                        "Done."
                    } else {
                        "Tried that, but something went wrong."
                    },
                );
                e.commands = cmds;
                e.log = log;
                e.applied = true;
                e.instant = true;
                self.history.push(ChatMsg {
                    role: "user".into(),
                    content: text.clone(),
                });
                self.history.push(ChatMsg {
                    role: "assistant".into(),
                    content: format!(
                        "{{\"reply\":\"Done.\",\"commands\":{}}}",
                        serde_json::to_string(&e.commands).unwrap_or_default()
                    ),
                });
                self.chat.push(e);
                return;
            }
        }
        if !online {
            let msg = match &self.ollama {
                OllamaStatus::Offline(_) => "Ollama isn't running, so I can only do simple commands right now. Start Ollama (ollama serve) and pull a model — see the Copilot settings.".to_string(),
                _ => "No Ollama model is installed yet. Pull one from the Copilot settings (recommended: qwen3:8b).".to_string(),
            };
            self.chat.push(ChatEntry::new(Role::System, msg));
            return;
        }
        let scene = self.ed.scene.summary(&self.ed.selection);
        let router = Some(self.settings.decision_model.clone()).filter(|d| !d.is_empty() && self.models.contains(d));
        self.route_stage = router.as_ref().map(|r| format!("{r} deciding"));
        self.ai.send(AiRequest::Chat {
            model: self.settings.model.clone(),
            router,
            history: self.history.clone(),
            scene,
            prompt: text.clone(),
            voice: from_voice,
        });
        self.history.push(ChatMsg {
            role: "user".into(),
            content: text,
        });
        self.waiting_ai = Some(Instant::now());
    }

    fn start_recording(&mut self) {
        self.start_listening(Listen::Command);
    }

    fn start_listening(&mut self, purpose: Listen) {
        if purpose != Listen::Wake {
            self.speaker.stop();
        }
        match &self.voice_status {
            VoiceStatus::Ready(_) => {}
            VoiceStatus::Loading(_) | VoiceStatus::Downloading(..) => {
                self.pending_voice = true;
                return;
            }
            _ => {
                self.pending_voice = true;
                self.voice_status = VoiceStatus::Loading("Preparing Whisper…".into());
                self.voice.load(&self.settings.voice_model);
                return;
            }
        }
        if let Err(e) = self.voice.recorder.start() {
            self.toast(format!("Microphone: {e}"), theme::ERR);
            if purpose == Listen::Wake {
                self.settings.wake_word = false;
            }
            return;
        }
        self.listen_purpose = purpose;
    }

    fn stop_recording(&mut self, transcribe: bool) {
        let pcm = self.voice.recorder.stop();
        if transcribe && pcm.len() > 16000 / 4 {
            self.transcribing = true;
            self.transcribe_purpose = self.listen_purpose;
            self.voice.transcribe(pcm);
        }
    }

    fn toggle_mic(&mut self) {
        self.relisten = false;
        if self.voice.recorder.recording && self.listen_purpose == Listen::Wake {
            // background wake listening → switch to a normal command recording
            self.voice.recorder.stop();
            self.start_listening(Listen::Command);
        } else if self.speaker.busy() {
            self.speaker.stop();
            self.start_listening(Listen::Command);
        } else if self.voice.recorder.recording {
            self.stop_recording(true);
        } else if self.pending_voice {
            self.pending_voice = false;
        } else {
            self.start_recording();
        }
    }

    /// Attach a picture (PNG/JPEG bytes) to the next Copilot request.
    fn attach_image(&mut self, ctx: &egui::Context, name: &str, bytes: &[u8]) {
        match ai::prepare_image(bytes) {
            Ok(b64) => {
                use base64::Engine;
                let png = base64::engine::general_purpose::STANDARD.decode(&b64).unwrap_or_default();
                let thumb = image::load_from_memory(&png).map(|i| i.thumbnail(96, 96).to_rgba8());
                let tex = match thumb {
                    Ok(t) => ctx.load_texture("attach", egui::ColorImage::from_rgba_unmultiplied([t.width() as usize, t.height() as usize], t.as_raw()), egui::TextureOptions::LINEAR),
                    Err(_) => ctx.load_texture("attach", egui::ColorImage::new([1, 1], vec![Color32::GRAY]), egui::TextureOptions::LINEAR),
                };
                self.pending_image = Some((name.to_string(), b64, tex));
                self.settings.show_assistant = true;
                self.focus_prompt = true;
                self.toast("Picture attached — describe what to build (or just press Enter to recreate it)", theme::OK);
            }
            Err(e) => self.toast(e, theme::ERR),
        }
    }

    fn vision_model(&self) -> Option<String> {
        if !self.settings.vision_model.is_empty() && self.models.contains(&self.settings.vision_model) {
            return Some(self.settings.vision_model.clone());
        }
        ai::pick_vision_model(&self.models)
    }

    /// Picture → vision model description → builder model.
    fn submit_with_image(&mut self, text: String, img: (String, String, egui::TextureHandle), from_voice: bool) {
        let (name, b64, _) = img;
        let request = if text.is_empty() { "Recreate this picture as a 3D scene".to_string() } else { text };
        let mut u = ChatEntry::new(Role::User, format!("🖼 {name}\n{request}"));
        u.voice = from_voice;
        self.chat.push(u);
        self.last_prompt = request.clone();
        if self.ollama != OllamaStatus::Online {
            self.chat.push(ChatEntry::new(Role::System, "Ollama isn't running, so pictures can't be read right now. Start Ollama and try again."));
            return;
        }
        let Some(vm) = self.vision_model() else {
            self.chat.push(ChatEntry::new(Role::System, format!("To build from pictures, install a vision model: `ollama pull {}` (Copilot ⚙ → Pull → {}). gemma3:4b and llama3.2-vision work too.", ai::RECOMMENDED_VISION, ai::RECOMMENDED_VISION)));
            return;
        };
        self.vision_request = Some(request.clone());
        self.route_stage = Some(format!("{} looking at the picture", short_model(&vm)));
        self.waiting_ai = Some(Instant::now());
        self.ai.send(AiRequest::Describe { model: vm, image_b64: b64, request });
    }

    fn on_described(&mut self, model: String, text: Result<String, String>, secs: f32) {
        self.waiting_ai = None;
        self.route_stage = None;
        let Some(request) = self.vision_request.take() else { return };
        match text {
            Err(e) => self.chat.push(ChatEntry::new(Role::System, format!("⚠ Couldn't read the picture: {e}"))),
            Ok(desc) => {
                let shown: String = desc.chars().take(1500).collect();
                let mut e = ChatEntry::new(Role::System, format!("🖼 {} sees ({secs:.1}s):\n{shown}", short_model(&model)));
                e.via = Some(format!("👁 vision: {}", short_model(&model)));
                self.chat.push(e);
                if self.settings.model.is_empty() {
                    self.chat.push(ChatEntry::new(Role::System, "No builder model installed — pull qwen3:8b to build the scene."));
                    return;
                }
                let prompt = format!("{request}\n\nBuild it from this description of the user's picture (written by a vision model). Use several objects, keep everything on the ground (z >= 0) unless it floats in the picture, and match the colours:\n{desc}");
                let scene = self.ed.scene.summary(&self.ed.selection);
                self.route_stage = Some(format!("{} building", self.settings.model));
                self.ai.send(AiRequest::Chat { model: self.settings.model.clone(), router: None, history: self.history.clone(), scene, prompt, voice: false });
                self.history.push(ChatMsg { role: "user".into(), content: format!("{request} (from a picture: {})", desc.chars().take(400).collect::<String>()) });
                self.waiting_ai = Some(Instant::now());
            }
        }
    }

    /// A finished transcript from the mic, follow-up or wake word.
    fn handle_voice_text(&mut self, text: String, secs: f32) {
        use crate::voiceflow::{VoiceIntent, fix_mishearings, voice_intent};
        let text = fix_mishearings(text.trim());
        if text.is_empty() {
            return;
        }
        let conv = self.settings.conversation;
        match voice_intent(&text) {
            Some(VoiceIntent::Stop) => {
                let busy = self.waiting_ai.is_some() || self.review.is_some();
                if busy {
                    self.cancel_ai();
                }
                self.voice_turn = None;
                self.speaker.speak(if busy { "Stopped." } else { "Okay." });
                self.relisten = conv;
            }
            Some(VoiceIntent::Undo) => {
                if self.waiting_ai.is_some() || self.review.is_some() {
                    self.cancel_ai();
                }
                let ok = self.ed.undo();
                self.chat.push(ChatEntry::new(Role::System, if ok { "🎤 Undone." } else { "🎤 Nothing to undo." }));
                self.speaker.speak(if ok { "Undone." } else { "Nothing to undo." });
                self.relisten = conv;
            }
            Some(VoiceIntent::Goodbye) => {
                self.speaker.speak("Okay, I'll stop listening.");
                self.relisten = false;
                self.toast(
                    if self.settings.wake_word {
                        format!("Say “{}” when you need me", self.settings.wake_phrase)
                    } else {
                        "Stopped listening — click the mic to talk again".to_string()
                    },
                    theme::TEXT_DIM,
                );
            }
            Some(VoiceIntent::Repeat) => {
                let last = if self.speaker.last.is_empty() { "I haven't done anything yet.".to_string() } else { self.speaker.last.clone() };
                self.speaker.speak(&last);
                self.relisten = conv;
            }
            None => {
                self.voice_turn = Some(self.chat.len());
                self.submit_prompt(text, true, secs);
            }
        }
    }

    /// Stop the running AI request (Stop button, "stop" by voice).
    fn cancel_ai(&mut self) {
        self.ai.cancel.store(true, std::sync::atomic::Ordering::Relaxed);
        // Stop immediately in the UI even if Ollama is still busy on its side.
        self.review = None;
        self.ai_cancelled = true;
        self.waiting_ai = None;
        self.route_stage = None;
        let partial = self.streaming.take();
        if let Some((i, _)) = partial {
            self.chat[i].streaming = false;
        }
        self.chat.push(ChatEntry::new(Role::System, if partial.is_some() { "Stopped. Changes made so far are kept (Ctrl+Z undoes them in one step)." } else { "Stopped." }));
    }

    /// Per-frame voice session: spoken confirmations, follow-up listening, wake word.
    fn voice_tick(&mut self, ctx: &egui::Context) {
        let ai_idle = self.waiting_ai.is_none() && self.streaming.is_none() && self.review.is_none();
        // 1. a voice request finished → say what happened
        if let Some(start) = self.voice_turn.filter(|_| ai_idle) {
            self.voice_turn = None;
            let entries = self.chat.get(start..).unwrap_or(&[]);
            let main = entries.iter().find(|e| e.role == Role::Assistant);
            let fixes: usize = entries
                .iter()
                .filter(|e| e.role == Role::Assistant && e.via.as_deref().is_some_and(|v| v.contains("review")))
                .map(|e| e.commands.len().min(1))
                .sum();
            let line = match main {
                Some(e) => {
                    let fixes = if main.is_some_and(|m| m.via.as_deref().is_some_and(|v| v.contains("review"))) { 0 } else { fixes };
                    crate::voiceflow::spoken_summary(&e.text, &e.commands, &e.log, fixes)
                }
                None if entries.iter().any(|e| e.text.starts_with('⚠')) => "Sorry, something went wrong. Check the chat.".to_string(),
                None if entries.iter().any(|e| e.text.starts_with("Stopped")) => String::new(),
                None => entries.iter().rev().find(|e| e.role == Role::System).map(|e| e.text.chars().take(160).collect()).unwrap_or_default(),
            };
            if !line.is_empty() && (self.settings.voice_confirm || self.settings.speak_replies) {
                self.speaker.speak(&line);
            }
            self.relisten = self.settings.conversation;
        }
        let ready = matches!(self.voice_status, VoiceStatus::Ready(_));
        let free = !self.voice.recorder.recording && !self.transcribing && !self.pending_voice;
        let speaking = self.speaker.busy();
        // 2. follow-up: re-open the mic once the reply has been spoken
        if self.relisten && free && !speaking && ready && ai_idle {
            self.relisten = false;
            self.start_listening(Listen::FollowUp);
        }
        // 3. a follow-up nobody answers ends the conversation
        if self.voice.recorder.recording
            && self.listen_purpose == Listen::FollowUp
            && !self.voice.recorder.heard_speech
            && self.voice.recorder.started.is_some_and(|t| t.elapsed().as_secs_f32() > 7.0)
        {
            self.voice.recorder.stop();
        }
        // 4. background wake-word listening
        if self.settings.wake_word {
            if self.voice_status == VoiceStatus::NotLoaded {
                self.voice_status = VoiceStatus::Loading("Preparing Whisper…".into());
                self.voice.load(&self.settings.voice_model);
            }
            if free && !speaking && ready && !self.relisten {
                self.start_listening(Listen::Wake);
            }
            if speaking && self.voice.recorder.recording && self.listen_purpose == Listen::Wake {
                self.voice.recorder.stop(); // don't listen to ourselves
            }
        } else if self.voice.recorder.recording && self.listen_purpose == Listen::Wake {
            self.voice.recorder.stop();
        }
        if speaking || self.relisten || self.settings.wake_word {
            ctx.request_repaint_after(std::time::Duration::from_millis(150));
        }
    }

    // ------------------------------------------------------------------------------------------
    // File IO
    // ------------------------------------------------------------------------------------------
    fn save(&mut self, save_as: bool) {
        let path = if save_as || self.ed.file_path.is_none() {
            rfd::FileDialog::new()
                .add_filter("Rust Blender scene", &["rblend", "json"])
                .set_file_name("scene.rblend")
                .save_file()
        } else {
            self.ed.file_path.clone()
        };
        if let Some(p) = path {
            match serde_json::to_string_pretty(&self.ed.scene)
                .map_err(|e| e.to_string())
                .and_then(|s| std::fs::write(&p, s).map_err(|e| e.to_string()))
            {
                Ok(_) => {
                    self.toast(format!("Saved {}", p.display()), theme::OK);
                    self.ed.file_path = Some(p);
                    self.ed.dirty = false;
                }
                Err(e) => self.toast(format!("Save failed: {e}"), theme::ERR),
            }
        }
    }

    fn open(&mut self) {
        if let Some(p) = rfd::FileDialog::new()
            .add_filter("Rust Blender scene", &["rblend", "json"])
            .pick_file()
        {
            self.open_path(&p);
        }
    }

    fn open_path(&mut self, p: &std::path::Path) {
        match std::fs::read_to_string(p)
            .map_err(|e| e.to_string())
            .and_then(|s| serde_json::from_str::<Scene>(&s).map_err(|e| e.to_string()))
        {
            Ok(s) => {
                self.ed.snapshot();
                self.ed.exit_edit();
                self.ed.scene = s;
                self.ed.reset_parent_cache();
                self.ed.selection.clear();
                self.ed.active = None;
                self.ed.file_path = Some(p.to_path_buf());
                self.ed.dirty = false;
                self.ed.view_requests.push(ViewRequest::FrameAll);
                self.toast(format!("Opened {}", p.display()), theme::OK);
            }
            Err(e) => self.toast(format!("Open failed: {e}"), theme::ERR),
        }
    }

    fn export_obj(&mut self) {
        if let Some(p) = rfd::FileDialog::new()
            .add_filter("Wavefront OBJ", &["obj"])
            .set_file_name("scene.obj")
            .save_file()
        {
            let s = self.ed.scene.to_obj(&mut self.vp.mesh_cache);
            match std::fs::write(&p, s) {
                Ok(_) => self.toast(format!("Exported {}", p.display()), theme::OK),
                Err(e) => self.toast(format!("Export failed: {e}"), theme::ERR),
            }
        }
    }

    // ------------------------------------------------------------------------------------------
    // MCP: tool calls from external agents, executed on the UI thread
    // ------------------------------------------------------------------------------------------
    fn restart_mcp(&mut self, ctx: &egui::Context) {
        self.mcp = None; // dropping the server frees the port
        if !self.settings.mcp_enabled {
            self.mcp_status = "off".into();
            return;
        }
        let c = ctx.clone();
        match crate::mcp::McpServer::start(self.settings.mcp_port, move || c.request_repaint()) {
            Ok(m) => {
                self.mcp_status = format!("listening on http://127.0.0.1:{}/mcp", m.port);
                self.mcp = Some(m);
            }
            Err(e) => self.mcp_status = e,
        }
    }

    fn poll_mcp(&mut self) {
        let calls: Vec<crate::mcp::ToolCall> = match &self.mcp {
            Some(m) => m.rx.try_iter().collect(),
            None => return,
        };
        for call in calls {
            self.mcp_calls += 1;
            let r = self.mcp_tool(&call.name, &call.args, &call.client);
            let _ = call.reply.send(r);
        }
    }

    fn mcp_tool(&mut self, name: &str, args: &serde_json::Value, client: &str) -> crate::mcp::ToolResult {
        use crate::mcp::ToolResult;
        let client_short: String = client.split(['/', ' ']).next().unwrap_or("agent").chars().take(24).collect();
        match name {
            "run_commands" => {
                let cmds: Vec<Command> = match serde_json::from_value(args.get("commands").cloned().unwrap_or_default()) {
                    Ok(c) => c,
                    Err(e) => return ToolResult::err(format!("invalid commands: {e}")),
                };
                if cmds.is_empty() {
                    return ToolResult::err("no commands given");
                }
                let log = self.ed.execute(&cmds);
                let failed = log.iter().filter(|l| l.starts_with('✖')).count();
                let mut e = ChatEntry::new(Role::System, format!("⚙ MCP ({client_short}): ran {} command(s)", cmds.len()));
                e.commands = cmds;
                e.log = log.clone();
                e.applied = true;
                self.chat.push(e);
                let text = format!("{}\n\n{} ok, {} failed. Objects now: {}", log.join("\n"), log.len() - failed, failed, self.ed.scene.objects.len());
                if failed == log.len() { ToolResult::err(text) } else { ToolResult::ok(text) }
            }
            "get_scene" => {
                let mut s = crate::agent::bounds_report(&self.ed.scene, &mut self.vp.mesh_cache);
                let hidden: Vec<&str> = self.ed.scene.objects.iter().filter(|o| !o.visible).map(|o| o.name.as_str()).collect();
                if !hidden.is_empty() {
                    s += &format!("hidden: {}\n", hidden.join(", "));
                }
                let names = |ids: &[u64]| -> Vec<String> {
                    ids.iter().filter_map(|id| self.ed.scene.objects.iter().find(|o| o.id == *id)).map(|o| format!("\"{}\"", o.name)).collect()
                };
                s += &format!(
                    "selection: [{}]  active: {}  frame: {}  mode: {}\n",
                    names(&self.ed.selection).join(", "),
                    self.ed.active.map(|a| names(&[a]).join("")).unwrap_or_else(|| "none".into()),
                    self.ed.frame,
                    if self.ed.edit.is_some() {
                        "edit"
                    } else if self.ed.sculpt.is_some() {
                        "sculpt"
                    } else if self.ed.pose.is_some() {
                        "pose"
                    } else if self.ed.weight_paint.is_some() {
                        "weight_paint"
                    } else {
                        "object"
                    }
                );
                if self.ed.scene.objects.is_empty() {
                    s = format!("The scene is empty.\n{s}");
                }
                ToolResult::ok(s)
            }
            "check_scene" => {
                let ids: Vec<u64> = self.ed.scene.objects.iter().map(|o| o.id).collect();
                let issues = crate::agent::inspect(&self.ed.scene, &mut self.vp.mesh_cache, &ids, &[]);
                if issues.is_empty() {
                    ToolResult::ok("No problems found (nothing floating, sunken, hidden inside another object or duplicated).")
                } else {
                    let lines: Vec<String> = issues.iter().map(|i| if i.object.is_empty() { format!("- {}", i.text) } else { format!("- \"{}\": {}", i.object, i.text) }).collect();
                    ToolResult::ok(format!("{} possible issue(s) — some may be intentional (e.g. something meant to hover):\n{}", issues.len(), lines.join("\n")))
                }
            }
            "ask_copilot" => {
                let Some(p) = args["prompt"].as_str().map(str::trim).filter(|p| !p.is_empty()) else {
                    return ToolResult::err("missing prompt");
                };
                if self.waiting_ai.is_some() || self.review.is_some() {
                    return ToolResult::err("the Copilot is busy with another request; try again shortly");
                }
                if let Some(ip) = args["image_path"].as_str().filter(|p| !p.is_empty()) {
                    match std::fs::read(ip) {
                        Ok(b) => {
                            let ctx = self.ctx_handle.clone();
                            let name = std::path::Path::new(ip).file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
                            self.attach_image(&ctx, &name, &b);
                            if self.pending_image.is_none() {
                                return ToolResult::err(format!("could not read the picture {ip}"));
                            }
                        }
                        Err(e) => return ToolResult::err(format!("can't read {ip}: {e}")),
                    }
                }
                self.submit_prompt(p.to_string(), false, 0.0);
                ToolResult::ok(format!("Sent to the Copilot ({}). The build streams into the app; call get_scene in a few seconds to see the result.", self.settings.model))
            }
            "export_gltf" => {
                let Some(path) = args["path"].as_str().filter(|p| !p.is_empty()) else {
                    return ToolResult::err("missing path");
                };
                let (bytes, st) = crate::gltf::export_glb(&self.ed.scene, &mut self.vp.mesh_cache);
                match std::fs::write(path, bytes) {
                    Ok(_) => ToolResult::ok(format!("wrote {path}: {} meshes, {} lights, {} cameras, {} animations, {} bytes", st.meshes, st.lights, st.cameras, st.animations, st.bytes)),
                    Err(e) => ToolResult::err(format!("could not write {path}: {e}")),
                }
            }
            "import_gltf" => {
                let Some(path) = args["path"].as_str().filter(|p| !p.is_empty()) else {
                    return ToolResult::err("missing path");
                };
                let log = self.ed.execute(&[Command { action: "import".into(), name: Some(path.to_string()), ..Default::default() }]);
                let line = log.first().cloned().unwrap_or_default();
                self.chat.push(ChatEntry::new(Role::System, format!("⚙ MCP ({client_short}): {}", line.trim_start_matches(['✔', '✖', ' ']))));
                if line.starts_with('✔') { ToolResult::ok(line) } else { ToolResult::err(line) }
            }
            "render_image" => {
                let out = self.ed.scene.output;
                let w = args["width"].as_u64().map(|v| v as usize).unwrap_or(if args["path"].is_string() { out.width as usize } else { 960 });
                let h = args["height"].as_u64().map(|v| v as usize).unwrap_or(w * out.height as usize / out.width.max(1) as usize);
                let transparent = args["transparent"].as_bool().unwrap_or(out.transparent);
                let path_traced = matches!(args["engine"].as_str(), Some("path" | "path_traced" | "cycles"));
                let r = if path_traced {
                    let t0 = Instant::now();
                    let samples = args["samples"].as_u64().unwrap_or(16).clamp(1, 256) as u32;
                    let (cam, cam_name) = self.vp.render_view(&self.ed, w as f32 / h as f32);
                    let sc = crate::pathtrace::PtScene::gather(&self.ed.scene, &mut self.vp.mesh_cache, &mut self.vp.tex_cache, cam);
                    let rgba = crate::pathtrace::render_blocking(&sc, w, h, samples, args["denoise"].as_bool().unwrap_or(true));
                    crate::viewport::RenderResult { width: w, height: h, rgba, camera: cam_name, ms: t0.elapsed().as_secs_f32() * 1000.0 }
                } else {
                    self.vp.render_image(&self.ed, w, h, transparent)
                };
                let png = match crate::textures::encode_png(r.width, r.height, &r.rgba) {
                    Ok(b) => b,
                    Err(e) => return ToolResult::err(e),
                };
                let mut text = format!(
                    "Rendered {}×{} {} in {:.0} ms.",
                    r.width,
                    r.height,
                    r.camera.as_ref().map(|c| format!("through camera \"{c}\"")).unwrap_or_else(|| "from the current viewport view (the scene has no camera)".into()),
                    r.ms
                );
                if let Some(path) = args["path"].as_str().filter(|p| !p.is_empty()) {
                    if let Err(e) = std::fs::write(path, &png) {
                        return ToolResult::err(format!("could not write {path}: {e}"));
                    }
                    text += &format!(" Saved to {path}.");
                }
                self.chat.push(ChatEntry::new(Role::System, format!("⚙ MCP ({client_short}): rendered {}×{}", r.width, r.height)));
                ToolResult { image_png: Some(png), ..ToolResult::ok(text) }
            }
            other => ToolResult::err(format!("unknown tool: {other}")),
        }
    }

    // ------------------------------------------------------------------------------------------
    // F12 render
    // ------------------------------------------------------------------------------------------
    fn render_now(&mut self, ctx: &egui::Context) {
        let out = self.ed.scene.output;
        if out.engine == crate::scene::Engine::PathTraced {
            self.start_path_trace(ctx);
            return;
        }
        let res = self.vp.render_image(&self.ed, out.width as usize, out.height as usize, out.transparent);
        let img = egui::ColorImage::from_rgba_unmultiplied([res.width, res.height], &res.rgba);
        let tex = ctx.load_texture("f12-render", img, egui::TextureOptions::LINEAR);
        self.toast(
            format!(
                "Rendered {}×{} {} in {:.0} ms",
                res.width,
                res.height,
                res.camera.as_ref().map(|c| format!("through \"{c}\"")).unwrap_or_else(|| "from the current view (no camera)".into()),
                res.ms
            ),
            theme::OK,
        );
        self.render = Some(RenderView { result: res, tex, saved: None, job: None, progress: None });
        self.show_render = true;
    }

    /// Write `autosave.rblend` every two minutes while there are unsaved changes, and show the
    /// recovery prompt after a crash.
    fn autosave_tick(&mut self, ctx: &egui::Context) {
        if self.ed.dirty && self.last_autosave.elapsed().as_secs() >= 120 && self.vp.modal.is_none() {
            self.last_autosave = Instant::now();
            let scene = self.ed.scene.clone();
            std::thread::spawn(move || {
                if let Ok(js) = serde_json::to_string(&scene) {
                    let p = autosave_dir().join("autosave.rblend");
                    let tmp = p.with_extension("tmp");
                    if std::fs::write(&tmp, js).is_ok() {
                        let _ = std::fs::rename(&tmp, &p);
                    }
                }
            });
        }
        let Some(p) = self.recovery.clone() else { return };
        let when = std::fs::metadata(&p).and_then(|m| m.modified()).ok().and_then(|t| t.elapsed().ok()).map(|d| {
            let m = d.as_secs() / 60;
            if m < 60 { format!("{m} min ago") } else { format!("{} h ago", m / 60) }
        });
        let mut choice = None;
        egui::Window::new("Recover unsaved work?")
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, vec2(0.0, 0.0))
            .show(ctx, |ui| {
                ui.label(format!("Rust Blender didn't close normally last time. An autosave{} is available.", when.map(|w| format!(" from {w}")).unwrap_or_default()));
                ui.horizontal(|ui| {
                    if ui.button("♻ Recover").clicked() {
                        choice = Some(true);
                    }
                    if ui.button("Discard").clicked() {
                        choice = Some(false);
                    }
                });
            });
        match choice {
            Some(true) => {
                self.open_path(&p);
                self.ed.file_path = None;
                self.ed.dirty = true;
                self.recovery = None;
            }
            Some(false) => {
                let _ = std::fs::remove_file(&p);
                self.recovery = None;
            }
            None => {}
        }
    }

    /// UV Editor: shows the active mesh's UV layout over its texture; drag moves the selected
    /// faces' UVs (all faces outside edit mode), the wheel scales them.
    fn uv_window(&mut self, ctx: &egui::Context) {
        if !self.show_uv {
            return;
        }
        let mut open = true;
        let id = self.ed.edit.as_ref().map(|s| s.obj).or(self.ed.active);
        let mut cmd: Option<Command> = None;
        let mut apply: Option<Vec<Vec<[f32; 2]>>> = None;
        let mut begin_drag = false;
        let mut end_drag = false;
        egui::Window::new("UV Editor").open(&mut open).default_size(vec2(460.0, 520.0)).resizable(true).show(ctx, |ui| {
            ui.horizontal_wrapped(|ui| {
                for (l, m) in [("Smart UV Project", "smart"), ("Cube", "cube"), ("Sphere", "sphere"), ("Cylinder", "cylinder")] {
                    if ui.button(l).on_hover_text("Unwrap the selected faces (edit mode) or the whole mesh").clicked() {
                        cmd = Some(Command { action: "unwrap".into(), name: Some(m.into()), ..Default::default() });
                    }
                }
                if ui.button("Checker texture").on_hover_text("Apply a checker texture to see stretching").clicked() {
                    cmd = Some(Command { action: "set_texture".into(), name: Some("checker".into()), ..Default::default() });
                }
            });
            let Some(o) = id.and_then(|i| self.ed.scene.get(i)).filter(|o| o.is_mesh()) else {
                ui.label(RichText::new("Select a mesh object").color(theme::TEXT_DIM));
                return;
            };
            let pm = o.mesh.clone().unwrap_or_else(|| o.poly());
            let uvs: Vec<Vec<[f32; 2]>> = if pm.has_uvs() { pm.uvs.clone() } else { (0..pm.faces.len()).map(|f| crate::mesh::box_uv(&pm.verts, &pm.faces[f])).collect() };
            let sel_faces: Vec<usize> = match &self.ed.edit {
                Some(st) if st.obj == o.id => crate::editmesh::selected_faces(&pm, &st.sel),
                _ => vec![],
            };
            ui.label(RichText::new(format!("{} · {} faces{}{}", o.name, pm.faces.len(), if pm.has_uvs() { format!(" · {} UV islands", crate::uv::islands(&pm)) } else { " · generated UVs (unwrap to edit)".into() }, if sel_faces.is_empty() { " · dragging moves all UVs" } else { "" })).small().color(theme::TEXT_DIM));
            // background texture
            let tex = self.ed.scene.texture(o.material.texture);
            if let Some(t) = tex {
                let sig = crate::textures::signature(&t.source);
                if self.uv_bg.as_ref().is_none_or(|(i, s, _)| *i != t.id || *s != sig) {
                    if let Some(img) = self.vp.tex_cache.get(t) {
                        let small = if img.w > 1024 || img.h > 1024 { img.resized(1024, 1024) } else { crate::textures::TexImage { w: img.w, h: img.h, rgba: img.rgba.clone() } };
                        let raw: Vec<u8> = small.rgba.iter().flatten().copied().collect();
                        let h = ctx.load_texture("uv-bg", egui::ColorImage::from_rgba_unmultiplied([small.w, small.h], &raw), egui::TextureOptions::LINEAR);
                        self.uv_bg = Some((t.id, sig, h));
                    }
                }
            }
            let side = ui.available_width().min(ui.available_height() - 4.0).max(120.0);
            let (rect, resp) = ui.allocate_exact_size(vec2(side, side), Sense::click_and_drag());
            let p = ui.painter_at(rect);
            p.rect_filled(rect, 0.0, Color32::from_gray(38));
            match (&self.uv_bg, tex) {
                (Some((tid, _, h)), Some(t)) if *tid == t.id => {
                    p.image(h.id(), rect, egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)), Color32::from_white_alpha(170));
                }
                _ => {
                    for k in 0..=8 {
                        let t = k as f32 / 8.0;
                        let c = Color32::from_gray(if k % 4 == 0 { 80 } else { 55 });
                        p.line_segment([rect.lerp_inside(vec2(t, 0.0)), rect.lerp_inside(vec2(t, 1.0))], Stroke::new(1.0, c));
                        p.line_segment([rect.lerp_inside(vec2(0.0, t)), rect.lerp_inside(vec2(1.0, t))], Stroke::new(1.0, c));
                    }
                }
            }
            let to_px = |q: [f32; 2]| rect.lerp_inside(vec2(q[0], q[1]));
            let sel: std::collections::HashSet<usize> = sel_faces.iter().copied().collect();
            for (fi, f) in uvs.iter().enumerate() {
                let pts: Vec<egui::Pos2> = f.iter().map(|q| to_px(*q)).collect();
                if sel.contains(&fi) {
                    p.add(egui::Shape::convex_polygon(pts.clone(), Color32::from_rgba_unmultiplied(255, 150, 40, 60), Stroke::NONE));
                }
                let col = if sel.contains(&fi) { Color32::from_rgb(255, 170, 70) } else { Color32::from_white_alpha(150) };
                let mut closed = pts.clone();
                closed.push(pts[0]);
                p.add(egui::Shape::line(closed, Stroke::new(1.0, col)));
            }
            p.rect_stroke(rect, 0.0, Stroke::new(1.0, Color32::from_gray(120)), egui::StrokeKind::Inside);
            // interaction: move / scale the selected (or all) UVs
            let movable: Vec<usize> = if sel_faces.is_empty() { (0..uvs.len()).collect() } else { sel_faces.clone() };
            if resp.drag_started() {
                begin_drag = true;
                self.uv_drag = Some(uvs.clone());
            }
            if let Some(base) = &self.uv_drag {
                if resp.dragged() {
                    let d = resp.drag_delta() / side;
                    let mut nu = uvs.clone();
                    let _ = base;
                    for &f in &movable {
                        for q in &mut nu[f] {
                            q[0] += d.x;
                            q[1] += d.y;
                        }
                    }
                    apply = Some(nu);
                }
            }
            if resp.drag_stopped() {
                end_drag = true;
            }
            let scroll = if resp.hovered() { ui.input(|i| i.smooth_scroll_delta.y) } else { 0.0 };
            if scroll != 0.0 {
                let f = (1.0 + scroll * 0.002).clamp(0.8, 1.25);
                let pts: Vec<[f32; 2]> = movable.iter().flat_map(|&f| uvs[f].iter().copied()).collect();
                let c = pts.iter().fold([0.0, 0.0], |a, b| [a[0] + b[0], a[1] + b[1]]);
                let c = [c[0] / pts.len().max(1) as f32, c[1] / pts.len().max(1) as f32];
                let mut nu = uvs.clone();
                for &fi in &movable {
                    for q in &mut nu[fi] {
                        q[0] = c[0] + (q[0] - c[0]) * f;
                        q[1] = c[1] + (q[1] - c[1]) * f;
                    }
                }
                apply = Some(nu);
                if !self.prop_snapshotted {
                    begin_drag = true;
                }
            }
            ui.label(RichText::new("Drag to move · wheel to scale · in edit mode only the selected faces change").small().color(theme::TEXT_DIM));
        });
        if let Some(c) = cmd {
            let log = self.ed.execute(&[c]);
            if let Some(l) = log.first() {
                self.toast(l.trim_start_matches(['✔', '✖', ' ']).to_string(), if l.starts_with('✔') { theme::OK } else { theme::WARN });
            }
        }
        if begin_drag && !self.prop_snapshotted {
            self.ed.snapshot();
            self.prop_snapshotted = true;
        }
        if let (Some(nu), Some(id)) = (apply, id) {
            if let Some(o) = self.ed.scene.get_mut(id) {
                let mut pm = o.mesh.clone().unwrap_or_else(|| o.poly());
                if nu.len() == pm.faces.len() {
                    pm.uvs = nu;
                    o.mesh = Some(pm);
                    self.ed.dirty = true;
                }
            }
        }
        if end_drag {
            self.uv_drag = None;
            self.prop_snapshotted = false;
        }
        if !open {
            self.show_uv = false;
        }
    }

    fn start_path_trace(&mut self, ctx: &egui::Context) {
        let out = self.ed.scene.output;
        let (w, h) = ((out.width as usize).clamp(16, 8192), (out.height as usize).clamp(16, 8192));
        let (cam, cam_name) = self.vp.render_view(&self.ed, w as f32 / h as f32);
        let t0 = Instant::now();
        let sc = crate::pathtrace::PtScene::gather(&self.ed.scene, &mut self.vp.mesh_cache, &mut self.vp.tex_cache, cam);
        let c = ctx.clone();
        let job = crate::pathtrace::start(sc, w, h, out.samples.clamp(1, 4096), out.denoise, move || c.request_repaint());
        let blank = vec![0u8; w * h * 4];
        let img = egui::ColorImage::from_rgba_unmultiplied([w, h], &blank);
        let tex = ctx.load_texture("f12-render", img, egui::TextureOptions::LINEAR);
        let result = crate::viewport::RenderResult { width: w, height: h, rgba: blank, camera: cam_name, ms: t0.elapsed().as_secs_f32() * 1000.0 };
        self.render = Some(RenderView { result, tex, saved: None, job: Some(job), progress: Some((0, out.samples, 0.0)) });
        self.show_render = true;
    }

    /// Pull refined images from a running path trace.
    fn poll_path_trace(&mut self) {
        let Some(rv) = self.render.as_mut() else { return };
        let Some(job) = &rv.job else { return };
        let mut last = None;
        while let Ok(u) = job.rx.try_recv() {
            last = Some(u);
        }
        if let Some(u) = last {
            let img = egui::ColorImage::from_rgba_unmultiplied([u.width, u.height], &u.rgba);
            rv.tex.set(img, egui::TextureOptions::LINEAR);
            rv.result.rgba = u.rgba;
            rv.result.ms = u.secs * 1000.0;
            rv.progress = Some((u.samples, u.total, u.secs));
            if u.samples >= u.total {
                rv.job = None;
                let msg = format!("Path traced {}×{}, {} samples in {:.1} s", u.width, u.height, u.total, u.secs);
                self.toast(msg, theme::OK);
            }
        }
    }

    fn save_render(&mut self) {
        let Some(rv) = &self.render else { return };
        let name = self
            .ed
            .file_path
            .as_ref()
            .and_then(|p| p.file_stem())
            .map(|s| format!("{}.png", s.to_string_lossy()))
            .unwrap_or_else(|| "render.png".into());
        if let Some(p) = rfd::FileDialog::new().add_filter("PNG image", &["png"]).set_file_name(name).save_file() {
            let r = &rv.result;
            match crate::textures::encode_png(r.width, r.height, &r.rgba).and_then(|b| std::fs::write(&p, b).map_err(|e| e.to_string())) {
                Ok(_) => {
                    self.toast(format!("Saved {}", p.display()), theme::OK);
                    if let Some(rv) = &mut self.render {
                        rv.saved = Some(p);
                    }
                }
                Err(e) => self.toast(format!("Save failed: {e}"), theme::ERR),
            }
        }
    }

    fn render_window(&mut self, ctx: &egui::Context) {
        if !self.show_render {
            return;
        }
        let mut open = true;
        let (mut rerender, mut save) = (false, false);
        let mut out = self.ed.scene.output;
        egui::Window::new("Render")
            .open(&mut open)
            .default_width(820.0)
            .resizable(true)
            .collapsible(false)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    if ui.button("📷 Render  F12").clicked() {
                        rerender = true;
                    }
                    if ui.add_enabled(self.render.is_some(), egui::Button::new("💾 Save PNG…")).clicked() {
                        save = true;
                    }
                    ui.separator();
                    for (l, w, h) in [("720p", 1280, 720), ("1080p", 1920, 1080), ("4K", 3840, 2160), ("Square", 1080, 1080)] {
                        if ui.selectable_label(out.width == w && out.height == h, l).clicked() {
                            out.width = w;
                            out.height = h;
                            rerender = true;
                        }
                    }
                    if out.engine == crate::scene::Engine::Fast && ui.checkbox(&mut out.transparent, "Transparent").changed() {
                        rerender = true;
                    }
                });
                ui.horizontal(|ui| {
                    ui.label("Engine");
                    let before = out.engine;
                    ui.selectable_value(&mut out.engine, crate::scene::Engine::Fast, "⚡ Fast").on_hover_text("Rasterized with ray-traced shadows and ambient occlusion — about a second");
                    ui.selectable_value(&mut out.engine, crate::scene::Engine::PathTraced, "✨ Path traced").on_hover_text("Bounced light, soft shadows, glossy reflections and glass; refines progressively");
                    if before != out.engine {
                        rerender = true;
                    }
                    if out.engine == crate::scene::Engine::PathTraced {
                        ui.label("Samples");
                        ui.add(egui::DragValue::new(&mut out.samples).range(1..=4096));
                        ui.checkbox(&mut out.denoise, "Denoise").on_hover_text("Edge-aware denoiser guided by surface colour and normals: clean images from few samples");
                        let running = self.render.as_ref().is_some_and(|r| r.job.is_some());
                        if running && ui.button("⏹ Stop").clicked() {
                            if let Some(rv) = self.render.as_mut() {
                                rv.job = None;
                            }
                        }
                    }
                });
                if let Some((done, total, secs)) = self.render.as_ref().and_then(|r| r.progress) {
                    let running = self.render.as_ref().is_some_and(|r| r.job.is_some());
                    ui.horizontal(|ui| {
                        ui.add(egui::ProgressBar::new(done as f32 / total.max(1) as f32).desired_width(260.0).text(format!("{done} / {total} samples · {secs:.1} s")));
                        if running {
                            let eta = if done > 0 { secs / done as f32 * (total - done) as f32 } else { 0.0 };
                            ui.label(RichText::new(format!("≈ {eta:.0} s left")).small().color(theme::TEXT_DIM));
                            ui.ctx().request_repaint_after(std::time::Duration::from_millis(100));
                        }
                    });
                }
                if let Some(rv) = &self.render {
                    let r = &rv.result;
                    ui.label(
                        RichText::new(format!(
                            "{}×{} · {} · {:.0} ms{}",
                            r.width,
                            r.height,
                            r.camera.as_deref().unwrap_or("current view (add a camera for framed renders)"),
                            r.ms,
                            rv.saved.as_ref().map(|p| format!(" · saved to {}", p.display())).unwrap_or_default()
                        ))
                        .small()
                        .color(theme::TEXT_DIM),
                    );
                    let avail = ui.available_width().max(200.0);
                    let size = vec2(avail, avail * r.height as f32 / r.width as f32);
                    let (rect, _) = ui.allocate_exact_size(size, Sense::hover());
                    if self.ed.scene.output.transparent {
                        // checkerboard behind transparent renders
                        let p = ui.painter_at(rect);
                        let cs = 12.0;
                        let (nx, ny) = ((rect.width() / cs).ceil() as i32, (rect.height() / cs).ceil() as i32);
                        for y in 0..ny {
                            for x in 0..nx {
                                let c = if (x + y) % 2 == 0 { Color32::from_gray(90) } else { Color32::from_gray(60) };
                                let r = egui::Rect::from_min_size(rect.min + vec2(x as f32 * cs, y as f32 * cs), vec2(cs, cs)).intersect(rect);
                                p.rect_filled(r, 0.0, c);
                            }
                        }
                    }
                    ui.painter_at(rect).image(rv.tex.id(), rect, egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)), Color32::WHITE);
                }
            });
        if out != self.ed.scene.output {
            self.ed.scene.output = out;
            self.ed.dirty = true;
        }
        if rerender {
            self.render_now(ctx);
        }
        if save {
            self.save_render();
        }
        if !open {
            self.show_render = false;
        }
    }

    fn import_gltf_dialog(&mut self) {
        if let Some(p) = rfd::FileDialog::new().add_filter("glTF 2.0", &["glb", "gltf"]).pick_file() {
            self.import_path(&p);
        }
    }

    /// Import a file by extension: glTF scenes, images as a texture on the selection, or open .rblend.
    fn import_path(&mut self, p: &std::path::Path) {
        let ext = p.extension().map(|e| e.to_string_lossy().to_lowercase()).unwrap_or_default();
        let cmd = match ext.as_str() {
            "glb" | "gltf" => Command { action: "import".into(), name: Some(p.display().to_string()), ..Default::default() },
            "png" | "jpg" | "jpeg" => Command { action: "set_texture".into(), name: Some(p.display().to_string()), ..Default::default() },
            "rblend" => {
                self.open_path(p);
                return;
            }
            _ => {
                self.toast(format!("Can't import .{ext} files (glb, gltf, png, jpg, rblend)"), theme::WARN);
                return;
            }
        };
        let log = self.ed.execute(&[cmd]);
        if let Some(l) = log.first() {
            let ok = l.starts_with('✔');
            self.toast(l.trim_start_matches(['✔', '✖', ' ']).to_string(), if ok { theme::OK } else { theme::ERR });
            self.chat.push(ChatEntry::new(Role::System, format!("📂 {}", l.trim_start_matches(['✔', ' ']))));
        }
    }

    fn export_gltf(&mut self) {
        if let Some(p) = rfd::FileDialog::new()
            .add_filter("glTF 2.0 binary", &["glb"])
            .set_file_name("scene.glb")
            .save_file()
        {
            let (bytes, st) = crate::gltf::export_glb(&self.ed.scene, &mut self.vp.mesh_cache);
            match std::fs::write(&p, bytes) {
                Ok(_) => self.toast(
                    format!(
                        "Exported {} ({} meshes, {} lights, {} cameras{}, {:.1} KB)",
                        p.display(),
                        st.meshes,
                        st.lights,
                        st.cameras,
                        if st.animations > 0 { ", animated" } else { "" },
                        st.bytes as f32 / 1024.0
                    ),
                    theme::OK,
                ),
                Err(e) => self.toast(format!("Export failed: {e}"), theme::ERR),
            }
        }
    }

    fn new_scene(&mut self) {
        self.ed.snapshot();
        self.ed.scene = Scene::default();
        self.ed.reset_parent_cache();
        self.ed.selection = self.ed.scene.objects.iter().map(|o| o.id).collect();
        self.ed.active = self.ed.selection.first().copied();
        self.ed.file_path = None;
        self.ed.frame = 1;
    }

    // ------------------------------------------------------------------------------------------
    // Global shortcuts
    // ------------------------------------------------------------------------------------------
    fn shortcuts(&mut self, ctx: &egui::Context) {
        use egui::Modifiers as M;
        let typing = ctx.egui_wants_keyboard_input();
        let c = |m: M, k: Key| ctx.input_mut(|i| i.consume_key(m, k));
        if c(M::COMMAND | M::SHIFT, Key::Z) || c(M::COMMAND, Key::Y) {
            if !self.ed.redo() {
                self.toast("Nothing to redo", theme::TEXT_DIM);
            }
        } else if !typing && c(M::COMMAND, Key::Z) {
            if !self.ed.undo() {
                self.toast("Nothing to undo", theme::TEXT_DIM);
            }
        }
        // Tab toggles edit mode while the pointer is over the viewport (handled before egui's
        // focus navigation can hand Tab to a text field).
        let over_vp = ctx
            .pointer_hover_pos()
            .is_some_and(|p| self.vp.rect().contains(p));
        if !typing && over_vp && c(M::COMMAND, Key::Tab) {
            ctx.memory_mut(|m| m.move_focus(egui::FocusDirection::None));
            let arm_active = self.ed.active.and_then(|id| self.ed.scene.get(id)).is_some_and(|o| o.armature.is_some());
            let r = if self.ed.sculpt.is_some() {
                self.ed.sculpt = None;
                Ok("Object mode".to_string())
            } else if self.ed.pose.is_some() || arm_active {
                self.ed.toggle_pose()
            } else if self.ed.weight_paint.is_some() || self.ed.active.and_then(|id| self.ed.scene.get(id)).is_some_and(|o| o.skin.is_some()) {
                self.ed.toggle_weight_paint()
            } else {
                let log = self.ed.execute(&[Command::new("sculpt_mode")]);
                let l = log.first().cloned().unwrap_or_default();
                if l.starts_with('✔') { Ok(l.trim_start_matches(['✔', ' ']).to_string()) } else { Err(l.trim_start_matches(['✖', ' ']).to_string()) }
            };
            match r {
                Ok(m) => self.toast(m, theme::OK),
                Err(e) => self.toast(e, theme::WARN),
            }
        }
        if !typing && over_vp && c(M::NONE, Key::Tab) {
            ctx.memory_mut(|m| m.move_focus(egui::FocusDirection::None));
            self.vp.loopcut = None;
            match self.ed.toggle_edit() {
                Ok(_) => {}
                Err(e) => self.toast(e, theme::WARN),
            }
        }
        if c(M::COMMAND | M::SHIFT, Key::S) {
            self.save(true);
        } else if c(M::COMMAND, Key::S) {
            self.save(false);
        }
        if c(M::COMMAND, Key::O) {
            self.open();
        }
        if c(M::COMMAND, Key::N) {
            self.new_scene();
        }
        // Ctrl+P parent to active, Alt+P clear parent, M move to collection (object mode, over the viewport)
        if !typing && !self.ed.editing() {
            let mut cmd = None;
            if c(M::COMMAND, Key::P) {
                if let Some(a) = self.ed.active {
                    let an = self.ed.scene.get(a).map(|o| o.name.clone()).unwrap_or_default();
                    cmd = Some(Command { action: "parent".into(), parent: Some(an), ..Default::default() });
                }
            } else if c(M::ALT, Key::P) {
                cmd = Some(Command::new("clear_parent"));
            } else if over_vp && !self.ed.selection.is_empty() && c(M::NONE, Key::M) {
                self.collection_menu = true;
            }
            if let Some(cmd) = cmd {
                let log = self.ed.execute(&[cmd]);
                if let Some(l) = log.first() {
                    self.toast(l.trim_start_matches(['✔', '✖', ' ']).to_string(), if l.starts_with('✔') { theme::OK } else { theme::WARN });
                }
            }
        }
        // Alt+0..4: Subdivision Surface levels on the selected objects (object mode)
        if !typing && !self.ed.editing() {
            for (k, n) in [(Key::Num0, 0.0), (Key::Num1, 1.0), (Key::Num2, 2.0), (Key::Num3, 3.0), (Key::Num4, 4.0)] {
                if c(M::ALT, k) {
                    let log = self.ed.execute(&[Command { action: "subsurf".into(), number: Some(n), ..Default::default() }]);
                    if let Some(l) = log.first() {
                        self.toast(l.trim_start_matches(['✔', '✖', ' ']).to_string(), if l.starts_with('✔') { theme::OK } else { theme::WARN });
                    }
                }
            }
        }
        if c(M::NONE, Key::F12) {
            self.render_now(ctx);
        }
        if c(M::NONE, Key::F11) {
            self.show_render = !self.show_render || self.render.is_none();
            if self.render.is_none() {
                self.render_now(ctx);
            }
        }
        if c(M::NONE, Key::F3) || c(M::COMMAND, Key::K) {
            self.palette = Some(String::new());
            self.palette_sel = 0;
        }
        if c(M::COMMAND, Key::Space) || (!typing && c(M::NONE, Key::Backtick)) {
            self.toggle_mic();
        }
        if c(M::COMMAND, Key::L) || (!typing && c(M::NONE, Key::Slash)) {
            self.settings.show_assistant = true;
            self.focus_prompt = true;
        }
        if !typing {
            if c(M::NONE, Key::Space) {
                self.ed.playing = !self.ed.playing;
            }
            if c(M::SHIFT, Key::ArrowLeft) {
                let f = self.ed.scene.frame_start;
                self.ed.set_frame(f);
            } else if c(M::SHIFT, Key::ArrowRight) {
                let f = self.ed.scene.frame_end;
                self.ed.set_frame(f);
            } else if c(M::NONE, Key::ArrowLeft) {
                let f = self.ed.frame - 1;
                self.ed.set_frame(f);
            } else if c(M::NONE, Key::ArrowRight) {
                let f = self.ed.frame + 1;
                self.ed.set_frame(f);
            }
            if c(M::NONE, Key::F1) {
                self.show_help = !self.show_help;
            }
            if c(M::NONE, Key::N) {
                self.settings.show_assistant = !self.settings.show_assistant;
            }
        }
        if self.voice.recorder.recording && ctx.input(|i| i.key_pressed(Key::Escape)) {
            self.stop_recording(false);
        }
    }

    fn tick_playback(&mut self, ctx: &egui::Context) {
        let now = ctx.input(|i| i.time);
        let dt = (now - self.last_time).clamp(0.0, 0.25);
        self.last_time = now;
        if self.ed.playing {
            self.play_accum += dt;
            let step = 1.0 / self.ed.scene.fps.max(1) as f64;
            while self.play_accum >= step {
                self.play_accum -= step;
                let mut f = self.ed.frame + 1;
                if f > self.ed.scene.frame_end {
                    f = self.ed.scene.frame_start;
                }
                self.ed.set_frame(f);
            }
            ctx.request_repaint();
        } else {
            self.play_accum = 0.0;
        }
        if self.voice.recorder.recording {
            let auto = self.settings.hands_free || self.listen_purpose != Listen::Command;
            if auto
                && self
                    .voice
                    .recorder
                    .update_vad(self.settings.vad_threshold, self.settings.silence_secs)
            {
                // only transcribe if someone actually spoke
                let heard = self.voice.recorder.heard_speech || self.listen_purpose == Listen::Command;
                self.stop_recording(heard);
            }
            ctx.request_repaint();
        }
    }

    // ------------------------------------------------------------------------------------------
    // Top bar
    // ------------------------------------------------------------------------------------------
    fn top_bar(&mut self, ui: &mut Ui) {
        egui::Panel::top("topbar")
            .frame(
                egui::Frame::new()
                    .fill(theme::BG_HEADER)
                    .inner_margin(Margin::symmetric(8, 4)),
            )
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label(
                        RichText::new("◆")
                            .color(Color32::from_rgb(255, 136, 0))
                            .size(18.0),
                    )
                    .on_hover_text("Rust Blender");
                    egui::MenuBar::new().ui(ui, |ui| {
                        ui.menu_button("File", |ui| {
                            if ui.button("New            Ctrl+N").clicked() {
                                self.new_scene();
                            }
                            if ui.button("Open…          Ctrl+O").clicked() {
                                self.open();
                            }
                            if ui.button("Save           Ctrl+S").clicked() {
                                self.save(false);
                            }
                            if ui.button("Save As…  Ctrl+Shift+S").clicked() {
                                self.save(true);
                            }
                            ui.separator();
                            if ui.button("Import glTF 2.0 (.glb/.gltf)…").clicked() {
                                self.import_gltf_dialog();
                            }
                            if ui.button("Export glTF 2.0 (.glb)…").clicked() {
                                self.export_gltf();
                            }
                            if ui.button("Export Wavefront (.obj)…").clicked() {
                                self.export_obj();
                            }
                            ui.separator();
                            if ui.button("Quit").clicked() {
                                ui.ctx().send_viewport_cmd(egui::ViewportCommand::Close);
                            }
                        });
                        ui.menu_button("Edit", |ui| {
                            if ui
                                .add_enabled(
                                    !self.ed.undo_stack.is_empty(),
                                    egui::Button::new("Undo   Ctrl+Z"),
                                )
                                .clicked()
                            {
                                self.ed.undo();
                            }
                            if ui
                                .add_enabled(
                                    !self.ed.redo_stack.is_empty(),
                                    egui::Button::new("Redo   Ctrl+Shift+Z"),
                                )
                                .clicked()
                            {
                                self.ed.redo();
                            }
                            ui.separator();
                            if ui.button("Duplicate   Shift+D").clicked() {
                                self.ed.duplicate_selected(Vec3::new(2.5, 0.0, 0.0));
                            }
                            if ui.button("Delete   X").clicked() {
                                self.ed.delete_selected();
                            }
                            if ui.button("Select All   A").clicked() {
                                self.ed.select_all_toggle();
                            }
                            ui.separator();
                            if ui.button("Command Search…   F3").clicked() {
                                self.palette = Some(String::new());
                            }
                            if ui.button("Preferences…").clicked() {
                                self.show_settings = true;
                                self.settings.show_assistant = true;
                            }
                        });
                        ui.menu_button("Add", |ui| {
                            self.add_menu_items(ui);
                        });
                        if self.ed.editing() {
                            ui.menu_button("Mesh", |ui| {
                                use crate::editor::EditOp;
                                let mut op: Option<EditOp> = None;
                                let mut tool: Option<&str> = None;
                                if ui.button("Extrude   E").clicked() {
                                    tool = Some("extrude");
                                }
                                if ui.button("Inset Faces   I").clicked() {
                                    tool = Some("inset");
                                }
                                if ui.button("Bevel Edges   Ctrl+B").clicked() {
                                    tool = Some("bevel");
                                }
                                ui.menu_button("UV Unwrap   U", |ui| {
                                    for (l, m) in [("Smart UV Project", "smart"), ("Cube Projection", "cube"), ("Sphere Projection", "sphere"), ("Cylinder Projection", "cylinder")] {
                                        if ui.button(l).clicked() {
                                            self.ed.execute(&[Command { action: "unwrap".into(), name: Some(m.into()), ..Default::default() }]);
                                            ui.close();
                                        }
                                    }
                                });
                                if ui.button("UV Editor…").clicked() {
                                    self.show_uv = true;
                                }
                                if ui.button("Loop Cut   Ctrl+R").clicked() {
                                    tool = Some("loopcut");
                                }
                                ui.separator();
                                let mut p = self.vp.proportional;
                                if ui.checkbox(&mut p, "Proportional Editing   O").changed() {
                                    self.vp.proportional = p;
                                }
                                ui.separator();
                                if ui.button("Subdivide").clicked() {
                                    op = Some(EditOp::Subdivide);
                                }
                                if ui.button("Merge at Center   M").clicked() {
                                    op = Some(EditOp::Merge);
                                }
                                if ui.button("Flip Normals   Alt+N").clicked() {
                                    op = Some(EditOp::FlipNormals);
                                }
                                if ui.button("Delete   X").clicked() {
                                    op = Some(EditOp::Delete);
                                }
                                ui.separator();
                                if ui.button("Select All   A").clicked() {
                                    op = Some(EditOp::SelectAll);
                                }
                                if ui.button("Select None   Alt+A").clicked() {
                                    op = Some(EditOp::SelectNone);
                                }
                                if ui.button("Invert Selection   Ctrl+I").clicked() {
                                    op = Some(EditOp::SelectInvert);
                                }
                                if ui.button("Select Top Faces").clicked() {
                                    op = Some(EditOp::SelectNormal(Vec3::Z, false));
                                }
                                if ui.button("Select Bottom Faces").clicked() {
                                    op = Some(EditOp::SelectNormal(-Vec3::Z, false));
                                }
                                ui.separator();
                                if ui.button("Exit Edit Mode   Tab").clicked() {
                                    self.ed.exit_edit();
                                }
                                if let Some(op) = op {
                                    if let Err(e) = self.ed.edit_op(op) {
                                        self.toast(e, theme::WARN);
                                    }
                                }
                                match tool {
                                    Some("extrude") => {
                                        let c = self.vp.rect().center();
                                        self.vp.begin_extrude(&mut self.ed, c);
                                    }
                                    Some("inset") => {
                                        let c = self.vp.rect().center() + vec2(80.0, 0.0);
                                        self.vp.begin_inset(&mut self.ed, c);
                                    }
                                    Some("loopcut") => self.vp.loopcut = Some(1),
                                    Some("bevel") => {
                                        let c = self.vp.rect().center() + vec2(60.0, 0.0);
                                        self.vp.begin_bevel(&mut self.ed, c);
                                    }
                                    _ => {}
                                }
                            });
                        }
                        ui.menu_button("Object", |ui| {
                            if ui
                                .button(if self.ed.editing() {
                                    "Object Mode   Tab"
                                } else {
                                    "Edit Mode   Tab"
                                })
                                .clicked()
                            {
                                if let Err(e) = self.ed.toggle_edit() {
                                    self.toast(e, theme::WARN);
                                }
                            }
                            ui.separator();
                            if ui.button("Shade Smooth").clicked() {
                                self.ed.execute(&[Command::new("smooth")]);
                            }
                            if ui.button("Shade Flat").clicked() {
                                self.ed.execute(&[Command::new("flat")]);
                            }
                            if ui.button("Shade Auto Smooth").clicked() {
                                self.ed.execute(&[Command::new("auto_smooth")]);
                            }
                            if ui.button("UV Editor…").clicked() {
                                self.show_uv = true;
                            }
                            if ui.button("Smart UV Project").clicked() {
                                self.ed.execute(&[Command { action: "unwrap".into(), name: Some("smart".into()), ..Default::default() }]);
                            }
                            ui.separator();
                            ui.menu_button("Scatter", |ui| {
                                for (label, what) in [("Rocks", "rock"), ("Trees", "tree"), ("Grass", "grass"), ("Bushes", "bush"), ("Flowers", "flower")] {
                                    if ui.button(format!("{label} on the active object")).clicked() {
                                        let log = self.ed.execute(&[Command { action: "scatter".into(), name: Some(what.into()), ..Default::default() }]);
                                        if let Some(l) = log.first() {
                                            self.toast(l.trim_start_matches(['✔', '✖', ' ']).to_string(), if l.starts_with('✔') { theme::OK } else { theme::WARN });
                                        }
                                        ui.close();
                                    }
                                }
                                if ui.button("Selected on Active").on_hover_text("Copies of the other selected object spread over the active one").clicked() {
                                    let other = self.ed.selection.iter().copied().find(|id| Some(*id) != self.ed.active).and_then(|id| self.ed.scene.get(id)).map(|o| o.name.clone());
                                    let log = match other {
                                        Some(n) => self.ed.execute(&[Command { action: "scatter".into(), object: Some(n), ..Default::default() }]),
                                        None => vec!["✖ select the object to scatter, then (Shift) the surface last".into()],
                                    };
                                    if let Some(l) = log.first() {
                                        self.toast(l.trim_start_matches(['✔', '✖', ' ']).to_string(), if l.starts_with('✔') { theme::OK } else { theme::WARN });
                                    }
                                    ui.close();
                                }
                                if ui.button("Make Real").on_hover_text("Turn the active scatter into an ordinary mesh").clicked() {
                                    self.ed.execute(&[Command::new("realize")]);
                                    ui.close();
                                }
                            });
                            ui.menu_button("Rig", |ui| {
                                for n in [2usize, 3, 4, 6, 8] {
                                    if ui.button(format!("Auto Rig · {n} bones")).on_hover_text("Add an armature fitted to the active mesh and bind it with automatic weights").clicked() {
                                        let log = self.ed.execute(&[Command { action: "rig".into(), number: Some(n as f64), ..Default::default() }]);
                                        if let Some(l) = log.first() {
                                            self.toast(l.trim_start_matches(['✔', '✖', ' ']).to_string(), if l.starts_with('✔') { theme::OK } else { theme::WARN });
                                        }
                                        ui.close();
                                    }
                                }
                                ui.label(RichText::new("Or select meshes, then the armature, and Ctrl+P (automatic weights). Ctrl+Tab on the armature for Pose Mode.").small().color(theme::TEXT_DIM));
                            });
                            ui.menu_button("Boolean (active ◂ selected)", |ui| {
                                for (label, op) in [("Difference (cut out)", "difference"), ("Union (join)", "union"), ("Intersect", "intersect")] {
                                    if ui.button(label).on_hover_text("Changes the active object using the other selected objects, which are then hidden in a \"Cutters\" collection").clicked() {
                                        let log = self.ed.execute(&[Command { action: "boolean".into(), name: Some(op.into()), ..Default::default() }]);
                                        if let Some(l) = log.first() {
                                            self.toast(l.trim_start_matches(['✔', '✖', ' ']).to_string(), if l.starts_with('✔') { theme::OK } else { theme::WARN });
                                        }
                                        ui.close();
                                    }
                                }
                            });
                            ui.separator();
                            if ui.button("Insert Keyframe   I").clicked() {
                                self.ed.insert_keyframes();
                            }
                            if ui.button("Clear Keyframes").clicked() {
                                self.ed.execute(&[Command::new("clear_keyframes")]);
                            }
                            ui.separator();
                            if ui.button("Hide Selected   H").clicked() {
                                self.ed.execute(&[Command::new("hide")]);
                            }
                            if ui.button("Reveal Hidden   Alt+H").clicked() {
                                self.ed.execute(&[Command::new("show")]);
                            }
                            ui.separator();
                            if ui.button("Snap Cursor to Selected").clicked() {
                                if let Some(o) = self.ed.active.and_then(|a| self.ed.scene.get(a)) {
                                    self.vp.cursor = o.transform.location;
                                }
                            }
                            if ui.button("Reset Cursor").clicked() {
                                self.vp.cursor = Vec3::ZERO;
                            }
                        });
                        ui.menu_button("View", |ui| {
                            for (n, p) in [
                                ("Front   1", ViewPreset::Front),
                                ("Right   3", ViewPreset::Right),
                                ("Top   7", ViewPreset::Top),
                                ("Back   Ctrl+1", ViewPreset::Back),
                                ("Left   Ctrl+3", ViewPreset::Left),
                                ("Bottom   Ctrl+7", ViewPreset::Bottom),
                                ("User (Iso)", ViewPreset::Iso),
                            ] {
                                if ui.button(n).clicked() {
                                    self.ed.view_requests.push(ViewRequest::Preset(p));
                                }
                            }
                            ui.separator();
                            if ui.button("Camera   Numpad 0").clicked() {
                                self.ed.view_requests.push(ViewRequest::Camera);
                            }
                            if ui
                                .button("Align Active Camera to View   Ctrl+Alt+0")
                                .clicked()
                            {
                                self.ed.view_requests.push(ViewRequest::AlignCameraToView);
                            }
                            if ui.button("Toggle Perspective/Ortho   5").clicked() {
                                self.ed.view_requests.push(ViewRequest::ToggleOrtho);
                            }
                            if ui.button("Frame All   Home").clicked() {
                                self.ed.view_requests.push(ViewRequest::FrameAll);
                            }
                            if ui.button("Frame Selected   F").clicked() {
                                self.ed.view_requests.push(ViewRequest::FrameSelected);
                            }
                            ui.separator();
                            ui.checkbox(&mut self.settings.show_assistant, "AI Copilot panel   N");
                            ui.add(
                                egui::Slider::new(&mut self.settings.ui_scale, 0.7..=2.0)
                                    .text("UI scale"),
                            );
                            if ui.button("Apply UI scale").clicked() {
                                ui.ctx().set_zoom_factor(self.settings.ui_scale);
                                self.settings.save();
                            }
                        });
                        ui.menu_button("Render", |ui| {
                            if ui.button("Render Image   F12").clicked() {
                                self.ed.render_request = true;
                            }
                            if ui.add_enabled(self.render.is_some(), egui::Button::new("View Render   F11")).clicked() {
                                self.show_render = true;
                            }
                            if ui.add_enabled(self.render.is_some(), egui::Button::new("Save Render as PNG…")).clicked() {
                                self.save_render();
                            }
                        });
                        ui.menu_button("Help", |ui| {
                            if ui.button("Keyboard & Voice Reference   F1").clicked() {
                                self.show_help = true;
                            }
                            ui.hyperlink_to("Get Ollama", "https://ollama.com/download");
                        });

                        ui.add_space(16.0);
                        for (tab, active) in
                            [("Layout", true), ("Modeling", false), ("Animation", false)]
                        {
                            let txt = RichText::new(tab).size(13.0);
                            let r = ui.add(egui::Button::selectable(active, txt));
                            if r.clicked() {
                                match tab {
                                    "Modeling" => {
                                        self.vp.tool = Tool::Move;
                                        self.vp.overlays.wire_overlay = true;
                                    }
                                    "Animation" => self.ed.auto_key = true,
                                    _ => self.vp.overlays.wire_overlay = false,
                                }
                            }
                        }

                        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                            self.mic_button(ui, 15.0);
                            ui.add_space(6.0);
                            let (dot, label) = match &self.ollama {
                                OllamaStatus::Online if !self.settings.model.is_empty() => {
                                    if !self.settings.decision_model.is_empty() && self.models.contains(&self.settings.decision_model) {
                                        (theme::OK, format!("✦ {} → {}", short_model(&self.settings.decision_model), short_model(&self.settings.model)))
                                    } else {
                                        (theme::OK, format!("✦ {}", self.settings.model))
                                    }
                                }
                                OllamaStatus::Online => (theme::WARN, "✦ no model".into()),
                                OllamaStatus::Checking => (theme::TEXT_DIM, "✦ connecting…".into()),
                                OllamaStatus::Offline(_) => (theme::ERR, "✦ Ollama offline".into()),
                            };
                            let r = egui::Frame::new()
                                .fill(Color32::from_rgb(50, 44, 66))
                                .corner_radius(10.0)
                                .inner_margin(Margin::symmetric(10, 3))
                                .show(ui, |ui| {
                                    ui.horizontal(|ui| {
                                        let (rr, _) =
                                            ui.allocate_exact_size(vec2(8.0, 8.0), Sense::hover());
                                        ui.painter().circle_filled(rr.center(), 4.0, dot);
                                        ui.label(
                                            RichText::new(label)
                                                .color(Color32::from_rgb(214, 196, 255)),
                                        );
                                    });
                                });
                            let r = r
                                .response
                                .interact(Sense::click())
                                .on_hover_text("AI Copilot — click to configure");
                            if r.clicked() {
                                self.settings.show_assistant = true;
                                self.show_settings = !self.show_settings;
                            }
                            ui.add_space(10.0);
                            let name = self
                                .ed
                                .file_path
                                .as_ref()
                                .and_then(|p| p.file_name())
                                .map(|s| s.to_string_lossy().into_owned())
                                .unwrap_or_else(|| "untitled.rblend".into());
                            ui.label(
                                RichText::new(format!(
                                    "{}{}",
                                    name,
                                    if self.ed.dirty { " •" } else { "" }
                                ))
                                .color(theme::TEXT_DIM),
                            );
                        });
                    });
                });
            });
    }

    fn add_menu_items(&mut self, ui: &mut Ui) {
        ui.label(RichText::new("Mesh").color(theme::TEXT_DIM).small());
        for p in Primitive::ALL {
            if ui.button(format!("{}  {}", p.icon(), p.label())).clicked() {
                let at = self.vp.cursor;
                self.ed.add_primitive(p, at);
                ui.close();
            }
        }
        ui.separator();
        ui.label(
            RichText::new("Light / Camera")
                .color(theme::TEXT_DIM)
                .small(),
        );
        for (label, kind) in ADD_SPECIAL {
            if ui.button(*label).clicked() {
                self.add_special(kind);
                ui.close();
            }
        }
        ui.separator();
        ui.label(RichText::new("Armature").color(theme::TEXT_DIM).small());
        if ui.button("🏃  Single Bone").clicked() {
            let at = self.vp.cursor;
            self.ed.execute(&[Command { action: "add".into(), primitive: Some("armature".into()), location: Some(serde_json::json!([at.x, at.y, at.z])), ..Default::default() }]);
            ui.close();
        }
        if ui.button("🏃  Bone Chain (4)").clicked() {
            let at = self.vp.cursor;
            self.ed.execute(&[Command { action: "add".into(), primitive: Some("armature".into()), number: Some(4.0), location: Some(serde_json::json!([at.x, at.y, at.z])), ..Default::default() }]);
            ui.close();
        }
    }

    fn add_special(&mut self, kind: &str) {
        let at = self.vp.cursor
            + if kind == "camera" {
                Vec3::new(0.0, -8.0, 2.0)
            } else {
                Vec3::new(0.0, 0.0, 4.0)
            };
        let mut c = Command {
            action: "add".into(),
            primitive: Some(kind.into()),
            location: Some(serde_json::json!([at.x, at.y, at.z])),
            ..Default::default()
        };
        if kind != "light" {
            c.look_at = Some(serde_json::json!([
                self.vp.cursor.x,
                self.vp.cursor.y,
                self.vp.cursor.z
            ]));
        }
        self.ed.execute(&[c]);
    }

    // ------------------------------------------------------------------------------------------
    // Mic button
    // ------------------------------------------------------------------------------------------
    fn mic_button(&mut self, ui: &mut Ui, radius: f32) -> egui::Response {
        let size = vec2(radius * 2.0 + 6.0, radius * 2.0 + 6.0);
        let (rect, resp) = ui.allocate_exact_size(size, Sense::click());
        let c = rect.center();
        let p = ui.painter();
        let waking = self.voice.recorder.recording && self.listen_purpose == Listen::Wake;
        let rec = self.voice.recorder.recording && !waking;
        let t = ui.input(|i| i.time) as f32;
        if waking {
            let lvl = (self.voice.recorder.level() * 18.0).clamp(0.0, 1.0);
            p.circle_stroke(c, radius + 2.5 + lvl * 3.0, Stroke::new(1.5, theme::OK.gamma_multiply(0.8)));
        }
        if self.speaker.busy() {
            let a = (t * 6.0).sin() * 0.5 + 0.5;
            p.circle_stroke(c, radius + 3.0 + a * 2.0, Stroke::new(1.5, theme::AI_ACCENT2));
            ui.ctx().request_repaint();
        }
        let busy = self.transcribing
            || self.pending_voice
            || matches!(
                self.voice_status,
                VoiceStatus::Loading(_) | VoiceStatus::Downloading(..)
            );
        if rec {
            let lvl = (self.voice.recorder.level() * 18.0).clamp(0.0, 1.0);
            p.circle_filled(c, radius + 3.0 + lvl * 6.0, theme::REC.gamma_multiply(0.25));
            p.circle_filled(c, radius, theme::REC);
            p.rect_filled(
                egui::Rect::from_center_size(c, vec2(radius * 0.7, radius * 0.7)),
                2.0,
                Color32::WHITE,
            );
        } else {
            let fill = if resp.hovered() {
                theme::AI_ACCENT
            } else {
                Color32::from_rgb(110, 80, 190)
            };
            p.circle_filled(c, radius, fill);
            if busy {
                let a = t * 5.0;
                let pts: Vec<egui::Pos2> = (0..20)
                    .map(|i| {
                        let aa = a + i as f32 * 0.22;
                        c + vec2(aa.cos(), aa.sin()) * (radius + 2.5)
                    })
                    .collect();
                p.add(egui::Shape::line(pts, Stroke::new(2.0, theme::AI_ACCENT2)));
                ui.ctx().request_repaint();
            }
            // microphone glyph
            let w = radius * 0.42;
            let body =
                egui::Rect::from_center_size(c - vec2(0.0, radius * 0.15), vec2(w, radius * 0.9));
            p.rect_filled(body, CornerRadius::same((w * 0.5) as u8), Color32::WHITE);
            let arc: Vec<egui::Pos2> = (0..=12)
                .map(|i| {
                    let a = std::f32::consts::PI * i as f32 / 12.0;
                    c + vec2(
                        -a.cos() * radius * 0.45,
                        a.sin() * radius * 0.42 + radius * 0.02,
                    )
                })
                .collect();
            p.add(egui::Shape::line(arc, Stroke::new(1.6, Color32::WHITE)));
            p.line_segment(
                [c + vec2(0.0, radius * 0.45), c + vec2(0.0, radius * 0.65)],
                Stroke::new(1.6, Color32::WHITE),
            );
        }
        let tip = if waking {
            format!("Waiting for “{}…” — or click to talk now", self.settings.wake_phrase)
        } else if self.speaker.busy() {
            "Speaking — click to interrupt and talk".to_string()
        } else if rec && self.listen_purpose == Listen::FollowUp {
            "Listening for a follow-up… (stay silent to stop)".to_string()
        } else if rec {
            "Listening… click (or Ctrl+Space) to finish".to_string()
        } else {
            match &self.voice_status {
                VoiceStatus::Ready(m) => {
                    format!("Voice control (Whisper {m}) — click or Ctrl+Space / ` to talk")
                }
                VoiceStatus::Downloading(f, pr) => {
                    format!("Downloading Whisper {f}: {:.0}%", pr.max(0.0) * 100.0)
                }
                VoiceStatus::Loading(s) => s.clone(),
                VoiceStatus::Error(e) => format!("Voice error: {e}"),
                VoiceStatus::NotLoaded => {
                    "Voice control — click to enable (downloads a local Whisper model once)".into()
                }
            }
        };
        let resp = resp.on_hover_text(tip);
        if resp.clicked() {
            self.toggle_mic();
        }
        resp
    }

    // ------------------------------------------------------------------------------------------
    // AI Copilot panel
    // ------------------------------------------------------------------------------------------
    fn assistant_panel(&mut self, ui: &mut Ui) {
        if !self.settings.show_assistant {
            return;
        }
        egui::Panel::left("assistant")
            .default_size(340.0)
            .min_size(260.0)
            .max_size(620.0)
            .frame(
                egui::Frame::new()
                    .fill(Color32::from_rgb(34, 33, 38))
                    .inner_margin(Margin::same(10)),
            )
            .show(ui, |ui| {
                self.assistant_rect = ui.max_rect();
                // Header
                ui.horizontal(|ui| {
                    ui.label(RichText::new("✦").size(20.0).color(theme::AI_ACCENT));
                    ui.vertical(|ui| {
                        ui.label(RichText::new("AI Copilot").strong().size(16.0));
                        ui.label(
                            RichText::new("voice + natural language, 100% local")
                                .small()
                                .color(theme::TEXT_DIM),
                        );
                    });
                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        if ui
                            .add(egui::Button::new("⚙").frame(false))
                            .on_hover_text("Copilot settings")
                            .clicked()
                        {
                            self.show_settings = !self.show_settings;
                        }
                        if ui
                            .add(egui::Button::new("🗑").frame(false))
                            .on_hover_text("Clear conversation")
                            .clicked()
                        {
                            self.chat.truncate(1);
                            self.history.clear();
                        }
                    });
                });
                ui.add_space(4.0);
                // Gradient rule
                let (r, _) =
                    ui.allocate_exact_size(vec2(ui.available_width(), 2.0), Sense::hover());
                let mut m = egui::Mesh::default();
                m.colored_vertex(r.left_top(), theme::AI_ACCENT);
                m.colored_vertex(r.right_top(), theme::AI_ACCENT2);
                m.colored_vertex(r.right_bottom(), theme::AI_ACCENT2);
                m.colored_vertex(r.left_bottom(), theme::AI_ACCENT);
                m.add_triangle(0, 1, 2);
                m.add_triangle(0, 2, 3);
                ui.painter().add(egui::Shape::mesh(m));
                ui.add_space(6.0);

                if self.show_settings {
                    self.copilot_settings(ui);
                    ui.separator();
                }
                if let OllamaStatus::Offline(e) = self.ollama.clone() {
                    egui::Frame::new()
                        .fill(Color32::from_rgb(64, 40, 40))
                        .corner_radius(6.0)
                        .inner_margin(8.0)
                        .show(ui, |ui| {
                            ui.label(
                                RichText::new("Ollama is not running")
                                    .strong()
                                    .color(theme::WARN),
                            );
                            ui.add(
                                egui::Label::new(
                                    RichText::new(e.as_str()).small().color(theme::TEXT_DIM),
                                )
                                .wrap(),
                            );
                            ui.label(
                                RichText::new("Install Ollama, then in a terminal run:").small(),
                            );
                            ui.code(format!("ollama pull {}", ai::RECOMMENDED_MODEL));
                            ui.label(
                                RichText::new("Simple commands still work offline.")
                                    .small()
                                    .color(theme::TEXT_DIM),
                            );
                            if ui.button("Retry connection").clicked() {
                                self.ollama = OllamaStatus::Checking;
                                self.ai.send(AiRequest::ListModels);
                            }
                        });
                } else if self.ollama == OllamaStatus::Online
                    && self.models.is_empty()
                    && self.pull.is_none()
                {
                    egui::Frame::new()
                        .fill(Color32::from_rgb(52, 46, 30))
                        .corner_radius(6.0)
                        .inner_margin(8.0)
                        .show(ui, |ui| {
                            ui.label(
                                RichText::new("No models installed")
                                    .strong()
                                    .color(theme::WARN),
                            );
                            ui.horizontal(|ui| {
                                if ui
                                    .button(format!(
                                        "⬇ Pull {} (recommended)",
                                        ai::RECOMMENDED_MODEL
                                    ))
                                    .clicked()
                                {
                                    self.ai.send(AiRequest::Pull(ai::RECOMMENDED_MODEL.into()));
                                    self.pull = Some(("starting".into(), -1.0));
                                }
                            });
                            if ui
                                .button(format!("⬇ Pull {} (low-VRAM)", ai::LIGHTWEIGHT_MODEL))
                                .clicked()
                            {
                                self.ai.send(AiRequest::Pull(ai::LIGHTWEIGHT_MODEL.into()));
                                self.pull = Some(("starting".into(), -1.0));
                            }
                        });
                }
                if let Some((s, f)) = &self.pull {
                    ui.label(RichText::new(format!("Pulling model: {s}")).small());
                    ui.add(
                        egui::ProgressBar::new(f.max(0.0))
                            .animate(*f < 0.0)
                            .show_percentage(),
                    );
                }
                if let VoiceStatus::Downloading(f, p) = &self.voice_status {
                    ui.label(RichText::new(format!("Downloading Whisper ({f})…")).small());
                    ui.add(
                        egui::ProgressBar::new(p.max(0.0))
                            .show_percentage()
                            .animate(*p < 0.0),
                    );
                }

                // Composer at the bottom
                let composer_h = 112.0;
                let chat_h = (ui.available_height() - composer_h).max(80.0);
                egui::ScrollArea::vertical()
                    .max_height(chat_h)
                    .auto_shrink([false, false])
                    .stick_to_bottom(true)
                    .show(ui, |ui| {
                        self.chat_messages(ui);
                    });
                ui.add_space(4.0);
                self.composer(ui);
            });
    }

    fn chat_messages(&mut self, ui: &mut Ui) {
        let width = ui.available_width();
        let mut apply_idx = None;
        let show_suggestions = self.chat.len() <= 1;
        for (i, e) in self.chat.iter().enumerate() {
            match e.role {
                Role::User => {
                    ui.with_layout(Layout::right_to_left(Align::Min), |ui| {
                        egui::Frame::new()
                            .fill(theme::ACCENT)
                            .corner_radius(CornerRadius {
                                nw: 12,
                                ne: 3,
                                sw: 12,
                                se: 12,
                            })
                            .inner_margin(Margin::symmetric(10, 6))
                            .show(ui, |ui| {
                                ui.set_max_width(width * 0.8);
                                ui.label(
                                    RichText::new(format!(
                                        "{}{}",
                                        if e.voice { "🎤 " } else { "" },
                                        e.text
                                    ))
                                    .color(Color32::WHITE),
                                );
                            });
                    });
                }
                Role::Assistant => {
                    ui.horizontal_top(|ui| {
                        let (r, _) = ui.allocate_exact_size(vec2(22.0, 22.0), Sense::hover());
                        ui.painter().circle_filled(
                            r.center(),
                            11.0,
                            if e.instant {
                                Color32::from_rgb(60, 120, 90)
                            } else {
                                theme::AI_ACCENT
                            },
                        );
                        ui.painter().text(
                            r.center(),
                            Align2::CENTER_CENTER,
                            if e.instant { "⚡" } else { "✦" },
                            FontId::proportional(12.0),
                            Color32::WHITE,
                        );
                        egui::Frame::new()
                            .fill(Color32::from_rgb(50, 48, 58))
                            .corner_radius(CornerRadius {
                                nw: 3,
                                ne: 12,
                                sw: 12,
                                se: 12,
                            })
                            .inner_margin(Margin::symmetric(10, 6))
                            .show(ui, |ui| {
                                ui.set_max_width(width * 0.82);
                                ui.vertical(|ui| {
                                    if e.streaming {
                                        let blink = (ui.input(|i| i.time) * 2.0) as i64 % 2 == 0;
                                        ui.label(format!(
                                            "{}{}",
                                            e.text,
                                            if blink { " ▍" } else { "  " }
                                        ));
                                    } else {
                                        ui.label(&e.text);
                                    }
                                    if let Some(v) = &e.via {
                                        ui.label(RichText::new(v).small().color(theme::TEXT_DIM));
                                    }
                                    if !e.commands.is_empty() {
                                        let errs =
                                            e.log.iter().filter(|l| l.starts_with('✖')).count();
                                        let head = format!(
                                            "{} action{}{}{}",
                                            e.commands.len(),
                                            if e.commands.len() == 1 { "" } else { "s" },
                                            if errs > 0 {
                                                format!(" · {errs} failed")
                                            } else {
                                                String::new()
                                            },
                                            if e.secs > 0.0 {
                                                format!(" · {:.1}s", e.secs)
                                            } else {
                                                String::new()
                                            }
                                        );
                                        egui::CollapsingHeader::new(
                                            RichText::new(head).small().color(if errs > 0 {
                                                theme::WARN
                                            } else {
                                                theme::TEXT_DIM
                                            }),
                                        )
                                        .id_salt(("cmds", i))
                                        .show(ui, |ui| {
                                            if e.log.is_empty() {
                                                for c in &e.commands {
                                                    ui.label(
                                                        RichText::new(
                                                            serde_json::to_string(c)
                                                                .unwrap_or_default(),
                                                        )
                                                        .monospace()
                                                        .small(),
                                                    );
                                                }
                                            } else {
                                                for l in &e.log {
                                                    let col = if l.starts_with('✔') {
                                                        theme::OK
                                                    } else {
                                                        theme::ERR
                                                    };
                                                    ui.label(RichText::new(l).small().color(col));
                                                }
                                            }
                                        });
                                        if !e.applied && ui.button("▶ Apply").clicked() {
                                            apply_idx = Some(i);
                                        }
                                    }
                                });
                            });
                    });
                }
                Role::System => {
                    egui::Frame::new()
                        .fill(Color32::from_rgb(40, 40, 46))
                        .corner_radius(8.0)
                        .inner_margin(Margin::symmetric(10, 6))
                        .show(ui, |ui| {
                            ui.set_width(width - 20.0);
                            ui.label(RichText::new(&e.text).color(theme::TEXT_DIM).small());
                        });
                }
            }
            ui.add_space(6.0);
        }
        if let Some(i) = apply_idx {
            let cmds = self.chat[i].commands.clone();
            self.chat[i].log = self.ed.execute(&cmds);
            self.chat[i].applied = true;
        }
        if let Some(t0) = self.waiting_ai {
            let mut stop = false;
            ui.horizontal(|ui| {
                ui.spinner();
                let n = self
                    .streaming
                    .map(|(i, _)| self.chat[i].commands.len())
                    .unwrap_or(0);
                let what = if self.streaming.is_some() {
                    format!("streaming · {n} action{}", if n == 1 { "" } else { "s" })
                } else {
                    self.route_stage.clone().map(|s| s.trim_start_matches(&format!("{} ", self.settings.model)).to_string()).unwrap_or_else(|| "thinking".into())
                };
                ui.label(
                    RichText::new(format!(
                        "{} {what}… {:.1}s",
                        self.settings.model,
                        t0.elapsed().as_secs_f32()
                    ))
                    .color(theme::TEXT_DIM),
                );
                stop = ui
                    .small_button("■ Stop")
                    .on_hover_text("Stop generating (keeps changes made so far)")
                    .clicked();
            });
            if self.streaming.is_none() && t0.elapsed().as_secs() >= 20 {
                ui.label(RichText::new("Still waiting for the first word — the first request loads the model into memory, and models that don't fit in GPU memory run slowly on the CPU.").small().color(theme::TEXT_DIM));
            }
            if stop {
                self.cancel_ai();
                self.voice_turn = None;
            }
            ui.ctx().request_repaint();
        }
        if self.transcribing {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label(RichText::new("Transcribing speech…").color(theme::TEXT_DIM));
            });
        }
        if show_suggestions {
            ui.add_space(4.0);
            ui.label(RichText::new("Try:").small().color(theme::TEXT_DIM));
            let sugg = [
                "Build a snowman",
                "Make a ring of 12 glowing cubes",
                "Create a table with four legs",
                "Animate the cube spinning over 60 frames",
                "Make everything shiny gold",
                "Add a red torus above the cube",
                "top view",
                "Make a solar system",
            ];
            let mut clicked = None;
            ui.horizontal_wrapped(|ui| {
                for s in sugg {
                    let b = egui::Button::new(RichText::new(s).small())
                        .corner_radius(10.0)
                        .fill(Color32::from_rgb(52, 46, 70));
                    if ui.add(b).clicked() {
                        clicked = Some(s.to_string());
                    }
                }
            });
            if let Some(s) = clicked {
                self.submit_prompt(s, false, 0.0);
            }
        }
    }

    fn composer(&mut self, ui: &mut Ui) {
        egui::Frame::new()
            .fill(Color32::from_rgb(26, 25, 30))
            .stroke(Stroke::new(
                1.0,
                if self.voice.recorder.recording {
                    theme::REC
                } else {
                    Color32::from_rgb(70, 60, 100)
                },
            ))
            .corner_radius(10.0)
            .inner_margin(8.0)
            .show(ui, |ui| {
                if self.voice.recorder.recording {
                    ui.horizontal(|ui| {
                        ui.label(RichText::new("● Listening").color(theme::REC).strong());
                        let lvl = (self.voice.recorder.level() * 18.0).clamp(0.0, 1.0);
                        let (r, _) = ui.allocate_exact_size(
                            vec2(ui.available_width() - 4.0, 18.0),
                            Sense::hover(),
                        );
                        let t = ui.input(|i| i.time) as f32;
                        let n = 32;
                        for k in 0..n {
                            let x = r.left() + r.width() * (k as f32 + 0.5) / n as f32;
                            let h = (2.0
                                + lvl * 16.0 * (0.5 + 0.5 * ((t * 9.0 + k as f32 * 0.7).sin())))
                            .min(r.height());
                            ui.painter().line_segment(
                                [
                                    egui::pos2(x, r.center().y - h * 0.5),
                                    egui::pos2(x, r.center().y + h * 0.5),
                                ],
                                Stroke::new(2.5, theme::REC.gamma_multiply(0.6 + 0.4 * lvl)),
                            );
                        }
                    });
                }
                let mut drop_image = false;
                if let Some((name, _, tex)) = &self.pending_image {
                    ui.horizontal(|ui| {
                        ui.add(egui::Image::new(tex).max_size(vec2(48.0, 48.0)).corner_radius(4.0));
                        ui.label(RichText::new(format!("🖼 {name}")).small());
                        if ui.small_button("✖").on_hover_text("Remove the picture").clicked() {
                            drop_image = true;
                        }
                    });
                }
                if drop_image {
                    self.pending_image = None;
                }
                let hint = if self.voice.recorder.recording {
                    "Speak now…"
                } else if self.pending_image.is_some() {
                    "What should I build from this picture?  (Enter = recreate it)"
                } else {
                    "Describe what to build or change…  (Enter to send)"
                };
                let te = egui::TextEdit::multiline(&mut self.input)
                    .hint_text(hint)
                    .desired_rows(2)
                    .desired_width(f32::INFINITY)
                    .frame(egui::Frame::NONE)
                    .return_key(None);
                let r = ui.add(te);
                if self.focus_prompt {
                    r.request_focus();
                    self.focus_prompt = false;
                }
                let enter =
                    r.has_focus() && ui.input(|i| i.key_pressed(Key::Enter) && !i.modifiers.shift);
                if r.has_focus() && ui.input(|i| i.key_pressed(Key::Escape)) {
                    r.surrender_focus();
                }
                ui.horizontal(|ui| {
                    self.mic_button(ui, 13.0);
                    if ui.add(egui::Button::new(RichText::new("📎").size(14.0)).frame(false)).on_hover_text("Attach a picture or sketch: a vision model describes it and the Copilot builds it (you can also drop an image on this panel)").clicked() {
                        if let Some(p) = rfd::FileDialog::new().add_filter("Image", &["png", "jpg", "jpeg"]).pick_file() {
                            if let Ok(b) = std::fs::read(&p) {
                                let name = p.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
                                let ctx = ui.ctx().clone();
                                self.attach_image(&ctx, &name, &b);
                            }
                        }
                    }
                    ui.label(
                        RichText::new(if self.voice.recorder.recording && self.listen_purpose == Listen::Wake {
                            format!("say “{}…”", self.settings.wake_phrase)
                        } else if self.voice.recorder.recording && self.listen_purpose == Listen::FollowUp {
                            "listening for a follow-up…".to_string()
                        } else if self.speaker.busy() {
                            "speaking…".to_string()
                        } else if self.settings.hands_free {
                            "hands-free".to_string()
                        } else {
                            "push-to-talk".to_string()
                        })
                        .small()
                        .color(theme::TEXT_DIM),
                    );
                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        let can = (!self.input.trim().is_empty() || self.pending_image.is_some()) && self.waiting_ai.is_none();
                        let send = ui.add_enabled(
                            can,
                            egui::Button::new(RichText::new("Send ➤").strong())
                                .fill(theme::AI_ACCENT)
                                .corner_radius(8.0),
                        );
                        if (send.clicked() || enter) && can {
                            let t = std::mem::take(&mut self.input);
                            let t = t.trim_end_matches('\n').to_string();
                            self.submit_prompt(t, false, 0.0);
                            r.surrender_focus();
                        } else if enter {
                            // swallow the newline produced by Enter
                            while self.input.ends_with('\n') {
                                self.input.pop();
                            }
                        }
                    });
                });
            });
    }

    fn copilot_settings(&mut self, ui: &mut Ui) {
        egui::Frame::new()
            .fill(Color32::from_rgb(42, 40, 48))
            .corner_radius(8.0)
            .inner_margin(8.0)
            .show(ui, |ui| {
                ui.set_max_width(ui.available_width());
                ui.label(RichText::new("Language model (Ollama)").strong());
                egui::Grid::new("ai_settings")
                    .num_columns(2)
                    .spacing([8.0, 6.0])
                    .show(ui, |ui| {
                        ui.label("Host");
                        if ui
                            .text_edit_singleline(&mut self.settings.ollama_host)
                            .lost_focus()
                        {
                            self.ai = AiClient::new(
                                self.settings.ollama_host.trim_end_matches('/').to_string(),
                                {
                                    let c = ui.ctx().clone();
                                    move || c.request_repaint()
                                },
                            );
                            self.ollama = OllamaStatus::Checking;
                            self.ai.send(AiRequest::ListModels);
                        }
                        ui.end_row();
                        ui.label("Model");
                        ui.horizontal(|ui| {
                            egui::ComboBox::from_id_salt("model")
                                .selected_text(if self.settings.model.is_empty() {
                                    "—"
                                } else {
                                    &self.settings.model
                                })
                                .width(160.0)
                                .show_ui(ui, |ui| {
                                    for m in self.models.clone() {
                                        let best = ai::pick_best_model(&self.models).as_deref()
                                            == Some(m.as_str());
                                        if ui
                                            .selectable_label(
                                                self.settings.model == m,
                                                format!(
                                                    "{m}{}",
                                                    if best { "  ★ best fit" } else { "" }
                                                ),
                                            )
                                            .clicked()
                                        {
                                            self.settings.model = m;
                                            self.settings.auto_select_model = false;
                                        }
                                    }
                                });
                            if ui
                                .small_button("⟲")
                                .on_hover_text("Refresh model list")
                                .clicked()
                            {
                                self.ai.send(AiRequest::ListModels);
                            }
                        });
                        ui.end_row();
                        ui.label("");
                        ui.checkbox(
                            &mut self.settings.auto_select_model,
                            "Auto-pick best installed model",
                        );
                        ui.end_row();
                        ui.label("Decision").on_hover_text("A small, fast tool-calling model (e.g. tev1) that reads every request first: it does simple edits itself and hands complex builds to the builder model above.");
                        ui.horizontal(|ui| {
                            let sel = if self.settings.decision_model.is_empty() { "None (builder only)".to_string() } else { self.settings.decision_model.clone() };
                            egui::ComboBox::from_id_salt("decision_model").selected_text(sel).width(160.0).show_ui(ui, |ui| {
                                if ui.selectable_label(self.settings.decision_model.is_empty(), "None (builder only)").clicked() {
                                    self.settings.decision_model.clear();
                                    self.settings.auto_decision_model = false;
                                }
                                for m in self.models.clone() {
                                    if m == self.settings.model {
                                        continue;
                                    }
                                    if ui.selectable_label(self.settings.decision_model == m, &m).clicked() {
                                        self.settings.decision_model = m;
                                        self.settings.auto_decision_model = false;
                                    }
                                }
                            });
                        });
                        ui.end_row();
                        ui.label("");
                        ui.checkbox(&mut self.settings.auto_decision_model, "Auto-use tev1 when installed");
                        ui.end_row();
                        ui.label("");
                        ui.checkbox(&mut self.settings.auto_check, "🔍 Check & fix AI builds (agent loop)").on_hover_text("After a build, Rust checks for floating, sunken, hidden or duplicate objects and failed commands; the decision model (or builder) reviews and fixes real mistakes — up to 2 rounds, all in one undo step.");
                        ui.end_row();
                        ui.label("MCP");
                        ui.horizontal(|ui| {
                            let on = self.mcp.is_some();
                            ui.label(RichText::new(if on { "●" } else { "○" }).color(if on { theme::OK } else { Color32::GRAY }));
                            ui.label(RichText::new(&self.mcp_status).small());
                        })
                        .response
                        .on_hover_text(format!("External agents (Claude Desktop, Claude Code, Cursor …) can drive this app. {} tool call(s) so far.", self.mcp_calls));
                        ui.end_row();
                        ui.label("");
                        ui.horizontal(|ui| {
                            let mut changed = ui.checkbox(&mut self.settings.mcp_enabled, "MCP server").changed();
                            ui.label("port");
                            changed |= ui.add(egui::DragValue::new(&mut self.settings.mcp_port).range(1024..=65535)).lost_focus();
                            if changed {
                                let ctx = ui.ctx().clone();
                                self.restart_mcp(&ctx);
                                self.settings.save();
                            }
                            if ui.small_button("📋 Copy client config").on_hover_text("Claude Desktop / Cursor JSON (stdio). For Claude Code: claude mcp add --transport http rust-blender http://127.0.0.1:7878/mcp").clicked() {
                                ui.ctx().copy_text(crate::mcp::client_config());
                                self.toast("MCP client config copied", theme::OK);
                            }
                        });
                        ui.end_row();
                        ui.label("Vision");
                        ui.horizontal(|ui| {
                            let auto = ai::pick_vision_model(&self.models);
                            let cur = if self.settings.vision_model.is_empty() { format!("auto ({})", auto.as_deref().map(short_model).unwrap_or("none installed")) } else { self.settings.vision_model.clone() };
                            egui::ComboBox::from_id_salt("vision_model").selected_text(cur).show_ui(ui, |ui| {
                                ui.selectable_value(&mut self.settings.vision_model, String::new(), "auto");
                                for m in self.models.iter().filter(|m| ai::VISION_MODELS.iter().any(|v| m.starts_with(v))) {
                                    ui.selectable_value(&mut self.settings.vision_model, m.clone(), m.as_str());
                                }
                            }).response.on_hover_text("Reads pictures you attach (📎 or drop on this panel) so the Copilot can build them");
                            let have = self.models.iter().any(|m| m.starts_with("qwen2.5vl"));
                            if !have && ui.add_enabled(self.pull.is_none() && self.ollama == OllamaStatus::Online, egui::Button::new(RichText::new(format!("Pull {}", ai::RECOMMENDED_VISION)).small())).clicked() {
                                self.ai.send(AiRequest::Pull(ai::RECOMMENDED_VISION.into()));
                                self.pull = Some(("starting".into(), -1.0));
                            }
                        });
                        ui.end_row();
                        ui.label("Pull");
                        ui.horizontal(|ui| {
                            for m in [ai::RECOMMENDED_MODEL, ai::LIGHTWEIGHT_MODEL, "qwen3:14b"] {
                                let have = self.models.iter().any(|x| x == m);
                                if ui
                                    .add_enabled(
                                        !have
                                            && self.pull.is_none()
                                            && self.ollama == OllamaStatus::Online,
                                        egui::Button::new(RichText::new(m).small()),
                                    )
                                    .on_hover_text(if have {
                                        "installed"
                                    } else {
                                        "download via Ollama"
                                    })
                                    .clicked()
                                {
                                    self.ai.send(AiRequest::Pull(m.into()));
                                    self.pull = Some(("starting".into(), -1.0));
                                }
                            }
                        });
                        ui.end_row();
                        ui.label("");
                        ui.checkbox(
                            &mut self.settings.auto_execute,
                            "Apply AI actions automatically",
                        );
                        ui.end_row();
                        ui.label("");
                        ui.checkbox(
                            &mut self.settings.instant_commands,
                            "⚡ Run simple commands instantly (no LLM)",
                        );
                        ui.end_row();
                    });
                ui.add_space(4.0);
                ui.label(RichText::new("Voice (Whisper, on-device)").strong());
                egui::Grid::new("voice_settings")
                    .num_columns(2)
                    .spacing([8.0, 6.0])
                    .show(ui, |ui| {
                        ui.label("Model");
                        let before = self.settings.voice_model.clone();
                        egui::ComboBox::from_id_salt("vmodel")
                            .selected_text(&self.settings.voice_model)
                            .show_ui(ui, |ui| {
                                for (n, _) in voice::WHISPER_MODELS {
                                    let desc = match *n {
                                        "tiny.en" => "fastest · 75 MB",
                                        "base.en" => "balanced · 145 MB",
                                        "small.en" => "accurate · 480 MB",
                                        _ => "fast + accurate · 330 MB",
                                    };
                                    ui.selectable_value(
                                        &mut self.settings.voice_model,
                                        n.to_string(),
                                        format!("{n}  ({desc})"),
                                    );
                                }
                            });
                        if before != self.settings.voice_model {
                            self.voice_status = VoiceStatus::NotLoaded;
                        }
                        ui.end_row();
                        ui.label("");
                        ui.checkbox(
                            &mut self.settings.hands_free,
                            "Hands-free (auto-stop when you pause)",
                        );
                        ui.end_row();
                        ui.label("Sensitivity");
                        ui.add(
                            egui::Slider::new(&mut self.settings.vad_threshold, 0.003..=0.08)
                                .logarithmic(true)
                                .text("threshold"),
                        );
                        ui.end_row();
                        ui.label("Pause");
                        ui.add(
                            egui::Slider::new(&mut self.settings.silence_secs, 0.5..=3.0)
                                .suffix(" s"),
                        );
                        ui.end_row();
                        ui.label("");
                        ui.checkbox(&mut self.settings.voice_confirm, "Say what I did after voice requests")
                            .on_hover_text("A short spoken confirmation, e.g. “Added UV Sphere Ball.” Say “repeat that” to hear it again.");
                        ui.end_row();
                        ui.label("");
                        ui.checkbox(&mut self.settings.conversation, "Conversation: listen again after each reply")
                            .on_hover_text("The mic re-opens after the spoken reply; stay silent for 7 s or say “that's all” to stop. “No, undo that” or “stop” work at any time.");
                        ui.end_row();
                        ui.label("Wake word");
                        ui.horizontal(|ui| {
                            ui.checkbox(&mut self.settings.wake_word, "");
                            ui.add_enabled(self.settings.wake_word, egui::TextEdit::singleline(&mut self.settings.wake_phrase).desired_width(90.0))
                                .on_hover_text("Say e.g. “Blender, add a red cube”. Listening runs locally with Whisper; nothing leaves your computer.");
                        });
                        ui.end_row();
                        ui.label("");
                        ui.checkbox(&mut self.settings.speak_replies, "🔊 Also speak typed replies");
                        ui.end_row();
                    });
                ui.vertical(|ui| {
                    let state = match &self.voice_status {
                        VoiceStatus::Ready(m) => format!("Whisper {m} ready"),
                        VoiceStatus::NotLoaded => "not loaded".into(),
                        VoiceStatus::Loading(s) => s.clone(),
                        VoiceStatus::Downloading(f, p) => {
                            format!("downloading {f} {:.0}%", p * 100.0)
                        }
                        VoiceStatus::Error(e) => format!("error: {e}"),
                    };
                    ui.add(
                        egui::Label::new(RichText::new(state).small().color(theme::TEXT_DIM))
                            .wrap(),
                    );
                    if ui.small_button("Load now").clicked() {
                        self.voice_status = VoiceStatus::Loading("Preparing Whisper…".into());
                        self.voice.load(&self.settings.voice_model);
                    }
                });
                if ui.button("Save settings").clicked() {
                    self.settings.save();
                    self.toast("Settings saved", theme::OK);
                }
            });
    }

    // ------------------------------------------------------------------------------------------
    // Right side: outliner + properties
    // ------------------------------------------------------------------------------------------
    fn right_panel(&mut self, ui: &mut Ui) {
        egui::Panel::right("props")
            .default_size(300.0)
            .min_size(240.0)
            .frame(egui::Frame::new().fill(theme::BG_PANEL).inner_margin(0))
            .show(ui, |ui| {
                let total = ui.available_height();
                let out_h = (total * 0.36).clamp(140.0, 360.0);
                egui::Frame::new()
                    .fill(theme::BG_EDITOR)
                    .inner_margin(Margin::same(6))
                    .show(ui, |ui| {
                        ui.set_height(out_h);
                        self.outliner(ui);
                    });
                let (r, _) =
                    ui.allocate_exact_size(vec2(ui.available_width(), 3.0), Sense::hover());
                ui.painter().rect_filled(r, 0.0, theme::BG_DEEP);
                egui::Frame::new()
                    .fill(theme::BG_EDITOR)
                    .inner_margin(Margin::same(6))
                    .show(ui, |ui| {
                        ui.set_min_height(ui.available_height());
                        self.properties(ui);
                    });
            });
    }

    fn outliner(&mut self, ui: &mut Ui) {
        ui.horizontal(|ui| {
            ui.label(RichText::new("☰ Outliner").strong());
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                ui.menu_button("＋", |ui| self.add_menu_items(ui))
                    .response
                    .on_hover_text("Add object");
            });
        });
        ui.separator();
        egui::ScrollArea::vertical()
            .id_salt("outliner")
            .auto_shrink([false, false])
            .show(ui, |ui| {
                // rows: collections as folders, objects as a parent/child tree inside them
                enum Row {
                    Col(u32),
                    Obj(u64, usize, bool),
                }
                let mut rows = vec![];
                {
                    let sc = &self.ed.scene;
                    let mut cols: Vec<u32> = vec![0];
                    cols.extend(sc.collections.iter().map(|c| c.id));
                    fn push_tree(sc: &Scene, id: u64, depth: usize, rows: &mut Vec<Row>, collapsed: &std::collections::HashSet<u64>) {
                        let col = sc.get(id).map(|o| o.collection).unwrap_or(0);
                        let kids: Vec<u64> = sc.objects.iter().filter(|c| c.parent == Some(id) && c.collection == col).map(|c| c.id).collect();
                        rows.push(Row::Obj(id, depth, !kids.is_empty()));
                        if depth < 32 && !collapsed.contains(&id) {
                            for k in kids {
                                push_tree(sc, k, depth + 1, rows, collapsed);
                            }
                        }
                    }
                    for c in cols {
                        rows.push(Row::Col(c));
                        if self.outliner_collapsed_cols.contains(&c) {
                            continue;
                        }
                        for o in sc.objects.iter().filter(|o| o.collection == c) {
                            let root = match o.parent.and_then(|p| sc.get(p)) {
                                Some(p) => p.collection != c,
                                None => true,
                            };
                            if root {
                                push_tree(sc, o.id, 0, &mut rows, &self.outliner_collapsed);
                            }
                        }
                    }
                }
                let mut select = None;
                let mut toggle_vis = None;
                let mut delete = None;
                let mut start_rename = None;
                let mut cmd: Option<Command> = None;
                for (row, r) in rows.iter().enumerate() {
                    let (id, depth, has_kids) = match r {
                        Row::Col(cid) => {
                            let cid = *cid;
                            let (name, vis) = if cid == 0 {
                                ("Scene Collection".to_string(), true)
                            } else {
                                self.ed.scene.collections.iter().find(|c| c.id == cid).map(|c| (c.name.clone(), c.visible)).unwrap_or_default()
                            };
                            ui.horizontal(|ui| {
                                let open = !self.outliner_collapsed_cols.contains(&cid);
                                if ui.add(egui::Button::new(RichText::new(if open { "⏷" } else { "⏵" }).small()).frame(false)).clicked() {
                                    if open {
                                        self.outliner_collapsed_cols.insert(cid);
                                    } else {
                                        self.outliner_collapsed_cols.remove(&cid);
                                    }
                                }
                                let lr = ui.add(egui::Label::new(RichText::new(format!("🗀 {name}")).color(if vis { theme::TEXT_DIM } else { Color32::from_gray(90) })).sense(Sense::click()));
                                lr.context_menu(|ui| {
                                    if ui.button("Move selection here   M").clicked() {
                                        cmd = Some(Command { action: "move_to_collection".into(), collection: Some(name.clone()), ..Default::default() });
                                        ui.close();
                                    }
                                    if cid != 0 && ui.button("Delete collection (keep objects)").clicked() {
                                        cmd = Some(Command { action: "delete_collection".into(), collection: Some(name.clone()), ..Default::default() });
                                        ui.close();
                                    }
                                });
                                if cid != 0 {
                                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                                        ui.add_space(6.0);
                                        if ui.add(egui::Button::new(RichText::new(if vis { "👁" } else { "–" }).size(13.0)).frame(false)).on_hover_text("Show / hide everything in this collection").clicked() {
                                            cmd = Some(Command { action: if vis { "hide_collection" } else { "show_collection" }.into(), collection: Some(name.clone()), ..Default::default() });
                                        }
                                    });
                                }
                            });
                            continue;
                        }
                        Row::Obj(id, d, k) => (*id, *d, *k),
                    };
                    let id = &id;
                    let o = self.ed.scene.get(*id).unwrap().clone();
                    let selected = self.ed.selection.contains(id);
                    let active = self.ed.active == Some(*id);
                    let (rect_full, resp) =
                        ui.allocate_exact_size(vec2(ui.available_width(), 22.0), Sense::click());
                    let indent = 10.0 + depth as f32 * 14.0;
                    let rect = egui::Rect::from_min_max(rect_full.min + vec2(indent, 0.0), rect_full.max);
                    if has_kids {
                        let tri = egui::Rect::from_center_size(rect.left_center() + vec2(2.0, 0.0), vec2(14.0, 18.0));
                        let tr = ui.interact(tri, ui.scope_id().with(("fold", id)), Sense::click());
                        let open = !self.outliner_collapsed.contains(id);
                        if tr.clicked() {
                            if open {
                                self.outliner_collapsed.insert(*id);
                            } else {
                                self.outliner_collapsed.remove(id);
                            }
                        }
                    }
                    let bg = if active {
                        Color32::from_rgb(52, 78, 120)
                    } else if selected {
                        Color32::from_rgb(45, 60, 85)
                    } else if resp.hovered() {
                        Color32::from_rgb(60, 60, 60)
                    } else if row % 2 == 0 {
                        Color32::from_rgb(43, 43, 43)
                    } else {
                        Color32::from_rgb(40, 40, 40)
                    };
                    ui.painter().rect_filled(rect_full, 3.0, bg);
                    let p = ui.painter();
                    if has_kids {
                        p.text(rect.left_center() + vec2(2.0, 0.0), Align2::CENTER_CENTER, if self.outliner_collapsed.contains(id) { "⏵" } else { "⏷" }, FontId::proportional(9.0), theme::TEXT_DIM);
                    }
                    let (icon, icol) = match &o.kind {
                        crate::scene::ObjectKind::Mesh => ("▽", Color32::from_rgb(255, 160, 80)),
                        crate::scene::ObjectKind::Light(_) => {
                            ("💡", Color32::from_rgb(255, 220, 120))
                        }
                        crate::scene::ObjectKind::Camera(_) => {
                            ("🎥", Color32::from_rgb(140, 200, 255))
                        }
                        crate::scene::ObjectKind::Armature => ("🏃", Color32::from_rgb(170, 220, 170)),
                        crate::scene::ObjectKind::Empty => ("✛", Color32::from_rgb(200, 200, 200)),
                    };
                    p.text(
                        rect.left_center() + vec2(14.0, 0.0),
                        Align2::LEFT_CENTER,
                        icon,
                        FontId::proportional(11.0),
                        icol,
                    );
                    let dot = o.light().map(|l| l.color).unwrap_or(o.material.color);
                    if !matches!(o.kind, crate::scene::ObjectKind::Camera(_)) {
                        p.circle_filled(rect.left_center() + vec2(34.0, 0.0), 5.0, color32(dot));
                    }
                    let name_col = if o.visible {
                        if active {
                            Color32::from_rgb(255, 200, 120)
                        } else {
                            theme::TEXT
                        }
                    } else {
                        theme::TEXT_DIM
                    };
                    if self.rename.as_ref().is_some_and(|(rid, _)| rid == id) {
                        let mut child = ui.new_child(egui::UiBuilder::new().max_rect(
                            egui::Rect::from_min_max(
                                rect.left_top() + vec2(44.0, 1.0),
                                rect.right_bottom() - vec2(50.0, 1.0),
                            ),
                        ));
                        let (_, buf) = self.rename.as_mut().unwrap();
                        let te = child.text_edit_singleline(buf);
                        te.request_focus();
                        if te.lost_focus() {
                            let (rid, name) = self.rename.take().unwrap();
                            if !name.trim().is_empty() {
                                self.ed.execute(&[Command {
                                    action: "rename".into(),
                                    target: Some(
                                        self.ed
                                            .scene
                                            .get(rid)
                                            .map(|o| o.name.clone())
                                            .unwrap_or_default(),
                                    ),
                                    name: Some(name),
                                    ..Default::default()
                                }]);
                            }
                        }
                    } else {
                        let label = if o.modifiers.is_empty() {
                            o.name.clone()
                        } else {
                            format!("{}  🔧", o.name)
                        };
                        p.text(
                            rect.left_center() + vec2(46.0, 0.0),
                            Align2::LEFT_CENTER,
                            label,
                            FontId::proportional(13.0),
                            name_col,
                        );
                        if !o.keyframes.is_empty() {
                            p.text(
                                rect.right_center() - vec2(44.0, 0.0),
                                Align2::RIGHT_CENTER,
                                "◆",
                                FontId::proportional(11.0),
                                theme::KEYFRAME,
                            );
                        }
                    }
                    let eye = egui::Rect::from_center_size(
                        rect_full.right_center() - vec2(16.0, 0.0),
                        vec2(20.0, 20.0),
                    );
                    let eye_resp =
                        ui.interact(eye, ui.scope_id().with(("eye", id)), Sense::click());
                    ui.painter().text(
                        eye.center(),
                        Align2::CENTER_CENTER,
                        if o.visible { "👁" } else { "–" },
                        FontId::proportional(13.0),
                        if eye_resp.hovered() {
                            Color32::WHITE
                        } else {
                            theme::TEXT_DIM
                        },
                    );
                    if eye_resp.clicked() {
                        toggle_vis = Some(*id);
                    } else if resp.double_clicked() {
                        start_rename = Some(*id);
                    } else if resp.clicked() {
                        select =
                            Some((*id, ui.input(|i| i.modifiers.shift || i.modifiers.command)));
                    }
                    resp.context_menu(|ui| {
                        if ui.button("Rename").clicked() {
                            start_rename = Some(*id);
                            ui.close();
                        }
                        if ui.button("Duplicate").clicked() {
                            self.ed.select_only(*id);
                            self.ed.duplicate_selected(Vec3::new(2.5, 0.0, 0.0));
                            ui.close();
                        }
                        if ui.button("Delete").clicked() {
                            delete = Some(*id);
                            ui.close();
                        }
                        ui.separator();
                        if let Some(a) = self.ed.active.filter(|a| a != id) {
                            let an = self.ed.scene.get(a).map(|o| o.name.clone()).unwrap_or_default();
                            if ui.button(format!("Parent to \"{an}\"   Ctrl+P")).clicked() {
                                cmd = Some(Command { action: "parent".into(), target: Some(o.name.clone()), parent: Some(an), ..Default::default() });
                                ui.close();
                            }
                        }
                        if o.parent.is_some() && ui.button("Clear Parent   Alt+P").clicked() {
                            cmd = Some(Command { action: "clear_parent".into(), target: Some(o.name.clone()), ..Default::default() });
                            ui.close();
                        }
                        ui.menu_button("Move to Collection", |ui| {
                            let mut names = vec!["Scene Collection".to_string()];
                            names.extend(self.ed.scene.collections.iter().map(|c| c.name.clone()));
                            for n in names {
                                if ui.button(&n).clicked() {
                                    cmd = Some(Command { action: "move_to_collection".into(), target: Some(o.name.clone()), collection: Some(n), ..Default::default() });
                                    ui.close();
                                }
                            }
                            if ui.button("＋ New Collection").clicked() {
                                let n = (1..).map(|i| format!("Collection {i}")).find(|n| !self.ed.scene.collections.iter().any(|c| &c.name == n)).unwrap();
                                cmd = Some(Command { action: "move_to_collection".into(), target: Some(o.name.clone()), collection: Some(n), ..Default::default() });
                                ui.close();
                            }
                        });
                    });
                }
                if let Some(c) = cmd {
                    let log = self.ed.execute(&[c]);
                    if let Some(l) = log.first() {
                        self.toast(l.trim_start_matches(['✔', '✖', ' ']).to_string(), if l.starts_with('✔') { theme::OK } else { theme::WARN });
                    }
                }
                if let Some((id, add)) = select {
                    if self.ed.edit.as_ref().is_some_and(|s| s.obj != id) {
                        self.ed.exit_edit();
                    }
                    if add {
                        self.ed.toggle_select(id)
                    } else {
                        self.ed.select_only(id)
                    }
                }
                if let Some(id) = toggle_vis {
                    self.ed.snapshot();
                    if let Some(o) = self.ed.scene.get_mut(id) {
                        o.visible = !o.visible;
                    }
                }
                if let Some(id) = delete {
                    self.ed.select_only(id);
                    self.ed.delete_selected();
                }
                if let Some(id) = start_rename {
                    self.rename = self.ed.scene.get(id).map(|o| (id, o.name.clone()));
                }
                if self.ed.scene.objects.is_empty() {
                    ui.label(
                        RichText::new("Empty scene — press Shift+A or ask the Copilot")
                            .small()
                            .color(theme::TEXT_DIM),
                    );
                }
            });
    }

    fn properties(&mut self, ui: &mut Ui) {
        ui.horizontal(|ui| {
            for (t, icon, tip) in [
                (PropsTab::Object, "▣", "Object"),
                (PropsTab::Modifiers, "🔧", "Modifiers"),
                (PropsTab::Material, "◕", "Material"),
                (PropsTab::World, "🌐", "World"),
                (PropsTab::Scene, "🎬", "Scene / Output"),
            ] {
                if ui
                    .add(egui::Button::selectable(
                        self.props_tab == t,
                        RichText::new(icon).size(15.0),
                    ))
                    .on_hover_text(tip)
                    .clicked()
                {
                    self.props_tab = t;
                }
            }
        });
        ui.separator();
        // Track property edits as single undo steps.
        if !ui.input(|i| i.pointer.any_down()) && !ui.ctx().egui_wants_keyboard_input() {
            self.prop_snapshotted = false;
        }
        egui::ScrollArea::vertical()
            .id_salt("props")
            .auto_shrink([false, false])
            .show(ui, |ui| match self.props_tab {
                PropsTab::World => self.world_props(ui),
                PropsTab::Scene => self.scene_props(ui),
                _ => {
                    let Some(id) = self.ed.active else {
                        ui.label(RichText::new("No active object").color(theme::TEXT_DIM));
                        return;
                    };
                    let Some(orig) = self.ed.scene.get(id).cloned() else {
                        return;
                    };
                    let mut o = orig.clone();
                    let mut tex_action = None;
                    match self.props_tab {
                        PropsTab::Object => object_props(ui, &mut o),
                        PropsTab::Modifiers => modifier_props(ui, &mut o),
                        PropsTab::Material => tex_action = material_props(ui, &mut o, &self.ed.scene.textures),
                        _ => {}
                    }
                    if let Some(a) = ui.ctx().data_mut(|d| d.remove_temp::<String>(egui::Id::unique("cloth_action"))) {
                        // keep slider edits, then run the bake
                        if o != orig {
                            self.ed.snapshot();
                            *self.ed.scene.get_mut(id).unwrap() = o.clone();
                        }
                        let cmd = match a.as_str() {
                            "<remove>" => Command { action: "remove_cloth".into(), target: Some(o.name.clone()), ..Default::default() },
                            "<bake>" => Command { action: "bake_cloth".into(), target: Some(o.name.clone()), ..Default::default() },
                            pins => Command { action: "cloth".into(), target: Some(o.name.clone()), name: Some(pins.into()), ..Default::default() },
                        };
                        let log = self.ed.execute(&[cmd]);
                        if let Some(l) = log.first() {
                            self.toast(l.trim_start_matches(['✔', '✖', ' ']).to_string(), if l.starts_with('✔') { theme::OK } else { theme::WARN });
                        }
                        return;
                    }
                    if let Some((a, i)) = ui.ctx().data_mut(|d| d.remove_temp::<(String, usize)>(egui::Id::unique("shape_action"))) {
                        let r = match a.as_str() {
                            "add" => {
                                self.ed.snapshot();
                                self.ed.add_shape_key(id, "Key").and_then(|k| {
                                    self.ed.begin_shape_edit(id, k)?;
                                    self.ed.enter_edit(id)
                                }).map(|_| "New shape key: edit or sculpt it, Tab to store".to_string())
                            }
                            "edit" => {
                                self.ed.snapshot();
                                self.ed.begin_shape_edit(id, i).and_then(|m| self.ed.enter_edit(id).map(|_| m))
                            }
                            "finish" => {
                                self.ed.exit_edit();
                                self.ed.sculpt = None;
                                Ok("Stored the shape".into())
                            }
                            "key" => {
                                self.ed.snapshot();
                                let f = self.ed.frame;
                                if let Some(k) = self.ed.scene.get_mut(id).and_then(|o| o.shape_keys.get_mut(i)) {
                                    k.insert_key(f);
                                }
                                Ok(format!("Keyed the shape at frame {f}"))
                            }
                            _ => {
                                self.ed.snapshot();
                                self.ed.finish_shape_edit();
                                if let Some(o) = self.ed.scene.get_mut(id) {
                                    if i < o.shape_keys.len() {
                                        o.shape_keys.remove(i);
                                    }
                                    o.posed_key = 0;
                                }
                                Ok("Removed the shape key".into())
                            }
                        };
                        match r {
                            Ok(m) => self.toast(m, theme::OK),
                            Err(e) => self.toast(e, theme::WARN),
                        }
                        return;
                    }
                    if let Some(spec) = tex_action {
                        let pick = || match rfd::FileDialog::new().add_filter("Image", &["png", "jpg", "jpeg"]).pick_file() {
                            Some(p) => p.display().to_string(),
                            None => String::new(),
                        };
                        let (cmd, spec) = match spec.as_str() {
                            "<load>" => ("set_texture", pick()),
                            "<load-normal>" => ("bump", pick()),
                            "<load-rough>" => ("roughness_map", pick()),
                            "<bump>" => ("bump", "auto".into()),
                            "<bump-invert>" => ("bump", "invert".into()),
                            "<rough>" => ("roughness_map", "auto".into()),
                            _ => ("set_texture", spec),
                        };
                        if !spec.is_empty() {
                            let log = self.ed.execute(&[Command { action: cmd.into(), target: Some(o.name.clone()), name: Some(spec), ..Default::default() }]);
                            if let Some(l) = log.first() {
                                self.toast(l.trim_start_matches(['✔', '✖', ' ']).to_string(), if l.starts_with('✔') { theme::OK } else { theme::ERR });
                            }
                        }
                        return;
                    }
                    if o != orig {
                        if !self.prop_snapshotted {
                            self.ed.snapshot();
                            self.prop_snapshotted = true;
                        }
                        // apply transform delta to the rest of the selection for multi-edit
                        if o.name != orig.name {
                            o.name = self.ed.scene.unique_name(&o.name);
                        }
                        let switched = match (&orig.armature, &o.armature) {
                            (Some(a0), Some(a1)) if a0.active_action != a1.active_action || a0.actions.len() != a1.actions.len() => a1.actions.get(a1.active_action).and_then(|a| a.range()),
                            _ => None,
                        };
                        *self.ed.scene.get_mut(id).unwrap() = o;
                        if let Some((lo, hi)) = switched {
                            self.ed.scene.frame_start = lo;
                            self.ed.scene.frame_end = hi.max(lo + 1);
                            self.ed.set_frame(lo);
                        }
                        self.ed.dirty = true;
                    }
                }
            });
    }

    fn world_props(&mut self, ui: &mut Ui) {
        let mut w = self.ed.scene.world.clone();
        section(ui, "Viewport background", |ui| {
            ui.horizontal(|ui| {
                ui.label("Color");
                ui.color_edit_button_rgb(&mut w.background);
            });
        });
        section(ui, "Sun light (Material Preview)", |ui| {
            vec3_edit(ui, "Direction", &mut w.sun_dir, 0.01);
            ui.add(egui::Slider::new(&mut w.sun_strength, 0.0..=4.0).text("Strength"));
            ui.add(egui::Slider::new(&mut w.ambient, 0.0..=1.0).text("Ambient"));
        });
        if w != self.ed.scene.world {
            if !self.prop_snapshotted {
                self.ed.snapshot();
                self.prop_snapshotted = true;
            }
            self.ed.scene.world = w;
        }
    }

    fn scene_props(&mut self, ui: &mut Ui) {
        section(ui, "Frame range", |ui| {
            ui.horizontal(|ui| {
                ui.label("Start");
                ui.add(egui::DragValue::new(&mut self.ed.scene.frame_start).range(0..=100000));
                ui.label("End");
                ui.add(egui::DragValue::new(&mut self.ed.scene.frame_end).range(1..=100000));
            });
            ui.horizontal(|ui| {
                ui.label("FPS");
                ui.add(egui::DragValue::new(&mut self.ed.scene.fps).range(1..=240));
            });
        });
        section(ui, "Output", |ui| {
            let o = &mut self.ed.scene.output;
            ui.horizontal(|ui| {
                ui.label("Resolution");
                ui.add(egui::DragValue::new(&mut o.width).range(16..=8192).suffix(" px"));
                ui.label("×");
                ui.add(egui::DragValue::new(&mut o.height).range(16..=8192).suffix(" px"));
            });
            ui.checkbox(&mut o.transparent, "Transparent background");
            if ui.button("📷 Render Image  F12").clicked() {
                self.ed.render_request = true;
            }
        });
        section(ui, "Statistics", |ui| {
            let (o, v, t) = self.vp.stats;
            ui.label(format!("{o} objects · {v} vertices · {t} triangles"));
            ui.label(format!("Undo steps: {}", self.ed.undo_stack.len()));
        });
        section(ui, "Export", |ui| {
            if ui
                .button("Export glTF 2.0 (.glb)…")
                .on_hover_text("Meshes, materials, lights, cameras and animation")
                .clicked()
            {
                self.export_gltf();
            }
            if ui.button("Export Wavefront OBJ…").clicked() {
                self.export_obj();
            }
        });
    }

    // ------------------------------------------------------------------------------------------
    // Timeline + status bar
    // ------------------------------------------------------------------------------------------
    fn bottom(&mut self, ui: &mut Ui) {
        egui::Panel::bottom("status").frame(egui::Frame::new().fill(theme::BG_HEADER).inner_margin(Margin::symmetric(10, 3))).show(ui, |ui| {
            ui.horizontal(|ui| {
                let hint = if !self.vp.status_hint.is_empty() {
                    self.vp.status_hint.clone()
                } else {
                    "🖱 LMB select · MMB orbit · Shift+MMB pan · Wheel zoom · G/R/S transform · Shift+A add · F3 search · Ctrl+Space voice · / ask AI".into()
                };
                ui.label(RichText::new(hint).small().color(theme::TEXT_DIM));
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    let (o, v, t) = self.vp.stats;
                    ui.label(RichText::new(format!("Rust Blender 0.1 · {o} obj · {v} verts · {t} tris")).small().color(theme::TEXT_DIM));
                });
            });
        });
        egui::Panel::bottom("timeline")
            .exact_size(66.0)
            .frame(
                egui::Frame::new()
                    .fill(theme::BG_EDITOR)
                    .inner_margin(Margin::symmetric(8, 4)),
            )
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    let s = &mut self.ed;
                    if ui
                        .button("⏮")
                        .on_hover_text("Jump to start (Shift+←)")
                        .clicked()
                    {
                        let f = s.scene.frame_start;
                        s.set_frame(f);
                    }
                    if ui.button("◀").on_hover_text("Previous frame (←)").clicked() {
                        let f = s.frame - 1;
                        s.set_frame(f);
                    }
                    let play = if s.playing { "⏸" } else { "▶" };
                    if ui
                        .add(
                            egui::Button::new(RichText::new(play).size(14.0)).fill(if s.playing {
                                theme::ACCENT
                            } else {
                                theme::WIDGET
                            }),
                        )
                        .on_hover_text("Play / pause (Space)")
                        .clicked()
                    {
                        s.playing = !s.playing;
                    }
                    if ui.button("▶|").on_hover_text("Next frame (→)").clicked() {
                        let f = s.frame + 1;
                        s.set_frame(f);
                    }
                    if ui
                        .button("⏭")
                        .on_hover_text("Jump to end (Shift+→)")
                        .clicked()
                    {
                        let f = s.scene.frame_end;
                        s.set_frame(f);
                    }
                    ui.separator();
                    let rec_col = if s.auto_key {
                        theme::REC
                    } else {
                        theme::TEXT_DIM
                    };
                    if ui
                        .add(egui::Button::new(RichText::new("⏺").color(rec_col)))
                        .on_hover_text("Auto keying: record transforms as keyframes")
                        .clicked()
                    {
                        s.auto_key = !s.auto_key;
                    }
                    if ui
                        .button(RichText::new("◆ Key").color(theme::KEYFRAME))
                        .on_hover_text("Insert keyframe on selection (I)")
                        .clicked()
                    {
                        s.insert_keyframes();
                    }
                    ui.separator();
                    ui.label("Frame");
                    let mut f = s.frame;
                    if ui.add(egui::DragValue::new(&mut f)).changed() {
                        s.set_frame(f);
                    }
                    ui.label("Start");
                    ui.add(egui::DragValue::new(&mut s.scene.frame_start));
                    ui.label("End");
                    ui.add(egui::DragValue::new(&mut s.scene.frame_end));
                });
                // scrubber
                let (rect, resp) = ui
                    .allocate_exact_size(vec2(ui.available_width(), 24.0), Sense::click_and_drag());
                let p = ui.painter();
                p.rect_filled(rect, 3.0, theme::BG_DEEP);
                let (fs, fe) = (
                    self.ed.scene.frame_start,
                    self.ed.scene.frame_end.max(self.ed.scene.frame_start + 1),
                );
                let x_of = |f: i32| rect.left() + (f - fs) as f32 / (fe - fs) as f32 * rect.width();
                let step = ((fe - fs) / 12).max(1);
                let step = [1, 2, 5, 10, 20, 25, 50, 100, 250, 500]
                    .into_iter()
                    .find(|s| *s >= step)
                    .unwrap_or(step);
                let mut f = (fs / step) * step;
                while f <= fe {
                    if f >= fs {
                        let x = x_of(f);
                        p.line_segment(
                            [egui::pos2(x, rect.top()), egui::pos2(x, rect.bottom())],
                            Stroke::new(1.0, Color32::from_gray(55)),
                        );
                        p.text(
                            egui::pos2(x + 2.0, rect.top() + 1.0),
                            Align2::LEFT_TOP,
                            f.to_string(),
                            FontId::proportional(9.5),
                            theme::TEXT_DIM,
                        );
                    }
                    f += step;
                }
                // keyframes of selection (all objects dimmer)
                for o in &self.ed.scene.objects {
                    let sel = self.ed.selection.contains(&o.id);
                    for k in &o.keyframes {
                        if k.frame < fs || k.frame > fe {
                            continue;
                        }
                        let c = egui::pos2(x_of(k.frame), rect.center().y + 4.0);
                        let r = if sel { 5.0 } else { 3.5 };
                        let col = if sel {
                            theme::KEYFRAME
                        } else {
                            Color32::from_gray(130)
                        };
                        p.add(egui::Shape::convex_polygon(
                            vec![
                                c + vec2(0.0, -r),
                                c + vec2(r, 0.0),
                                c + vec2(0.0, r),
                                c + vec2(-r, 0.0),
                            ],
                            col,
                            Stroke::new(1.0, Color32::BLACK),
                        ));
                    }
                }
                let x = x_of(self.ed.frame.clamp(fs, fe));
                p.line_segment(
                    [egui::pos2(x, rect.top()), egui::pos2(x, rect.bottom())],
                    Stroke::new(2.0, theme::ACCENT_BRIGHT),
                );
                let tag =
                    egui::Rect::from_center_size(egui::pos2(x, rect.top() + 6.0), vec2(28.0, 13.0));
                p.rect_filled(tag, 3.0, theme::ACCENT);
                p.text(
                    tag.center(),
                    Align2::CENTER_CENTER,
                    self.ed.frame.to_string(),
                    FontId::proportional(10.0),
                    Color32::WHITE,
                );
                if resp.dragged() || resp.clicked() {
                    if let Some(pp) = resp.interact_pointer_pos() {
                        let f = fs
                            + (((pp.x - rect.left()) / rect.width()) * (fe - fs) as f32).round()
                                as i32;
                        self.ed.set_frame(f.clamp(fs, fe));
                    }
                }
            });
    }

    // ------------------------------------------------------------------------------------------
    // Popups: add menu, command palette, help, toasts
    // ------------------------------------------------------------------------------------------
    fn popups(&mut self, ctx: &egui::Context) {
        if self.collection_menu {
            let mut close = ctx.input(|i| i.key_pressed(Key::Escape));
            let at = ctx.pointer_latest_pos().unwrap_or(self.vp.rect().center());
            let mut pick: Option<String> = None;
            let area = egui::Area::new(egui::Id::unique("collection_menu"))
                .fixed_pos(at - vec2(20.0, 12.0))
                .constrain(true)
                .order(egui::Order::Foreground)
                .show(ctx, |ui| {
                    egui::Frame::popup(ui.style()).show(ui, |ui| {
                        ui.set_min_width(180.0);
                        ui.label(RichText::new("Move to Collection").strong());
                        ui.separator();
                        let mut names = vec!["Scene Collection".to_string()];
                        names.extend(self.ed.scene.collections.iter().map(|c| c.name.clone()));
                        for n in names {
                            if ui.button(&n).clicked() {
                                pick = Some(n);
                            }
                        }
                        ui.separator();
                        if ui.button("＋ New Collection").clicked() {
                            pick = Some((1..).map(|i| format!("Collection {i}")).find(|n| !self.ed.scene.collections.iter().any(|c| &c.name == n)).unwrap());
                        }
                    });
                });
            if area.response.clicked_elsewhere() && pick.is_none() {
                close = true;
            }
            if let Some(n) = pick {
                let log = self.ed.execute(&[Command { action: "move_to_collection".into(), collection: Some(n), ..Default::default() }]);
                if let Some(l) = log.first() {
                    self.toast(l.trim_start_matches(['✔', '✖', ' ']).to_string(), theme::OK);
                }
                close = true;
            }
            if close {
                self.collection_menu = false;
            }
        }
        if let Some(at) = self.vp.add_menu_at {
            let mut close = ctx.input(|i| i.key_pressed(Key::Escape));
            let area = egui::Area::new(egui::Id::unique("add_menu"))
                .fixed_pos(at - vec2(20.0, 12.0))
                .constrain(true)
                .order(egui::Order::Foreground)
                .show(ctx, |ui| {
                    egui::Frame::popup(ui.style()).show(ui, |ui| {
                        ui.set_min_width(170.0);
                        ui.label(RichText::new("Add  ▸  Mesh").strong());
                        ui.separator();
                        for p in Primitive::ALL {
                            if ui
                                .add(
                                    egui::Button::new(format!("{}  {}", p.icon(), p.label()))
                                        .frame(false)
                                        .min_size(vec2(170.0, 22.0)),
                                )
                                .clicked()
                            {
                                let c = self.vp.cursor;
                                self.ed.add_primitive(p, c);
                                close = true;
                            }
                        }
                        ui.separator();
                        for (label, kind) in ADD_SPECIAL {
                            if ui
                                .add(
                                    egui::Button::new(*label)
                                        .frame(false)
                                        .min_size(vec2(170.0, 22.0)),
                                )
                                .clicked()
                            {
                                self.add_special(kind);
                                close = true;
                            }
                        }
                    });
                });
            if area.response.clicked_elsewhere() {
                close = true;
            }
            if close {
                self.vp.add_menu_at = None;
            }
        }

        if self.palette.is_some() {
            let mut run: Option<PaletteAction> = None;
            let mut close = false;
            egui::Window::new("palette")
                .title_bar(false)
                .anchor(Align2::CENTER_TOP, vec2(0.0, 80.0))
                .fixed_size(vec2(520.0, 0.0))
                .show(ctx, |ui| {
                    let q = self.palette.as_mut().unwrap();
                    ui.horizontal(|ui| {
                        ui.label(RichText::new("🔍").size(16.0));
                        let r = ui.add(
                            egui::TextEdit::singleline(q)
                                .hint_text("Search commands — or type anything to ask the AI")
                                .desired_width(f32::INFINITY)
                                .font(egui::TextStyle::Heading),
                        );
                        r.request_focus();
                    });
                    ui.separator();
                    let ql = q.to_lowercase();
                    let items: Vec<(String, PaletteAction)> = palette_items()
                        .into_iter()
                        .filter(|(n, _)| {
                            ql.is_empty()
                                || ql.split_whitespace().all(|w| n.to_lowercase().contains(w))
                        })
                        .take(10)
                        .collect();
                    let n_items = items.len();
                    if ui.input(|i| i.key_pressed(Key::ArrowDown)) {
                        self.palette_sel = (self.palette_sel + 1).min(n_items);
                    }
                    if ui.input(|i| i.key_pressed(Key::ArrowUp)) {
                        self.palette_sel = self.palette_sel.saturating_sub(1);
                    }
                    for (i, (name, act)) in items.iter().enumerate() {
                        let r = ui.add(
                            egui::Button::selectable(i == self.palette_sel, name.as_str())
                                .min_size(vec2(ui.available_width(), 24.0)),
                        );
                        if r.clicked() {
                            run = Some(act.clone());
                        }
                    }
                    let ask = format!("✦ Ask AI: “{}”", q);
                    if !q.trim().is_empty() {
                        let r = ui.add(
                            egui::Button::selectable(
                                self.palette_sel == n_items,
                                RichText::new(&ask).color(Color32::from_rgb(200, 180, 255)),
                            )
                            .min_size(vec2(ui.available_width(), 24.0)),
                        );
                        if r.clicked() {
                            run = Some(PaletteAction::Ask(q.clone()));
                        }
                    }
                    if ui.input(|i| i.key_pressed(Key::Enter)) {
                        run = if self.palette_sel < n_items {
                            Some(items[self.palette_sel].1.clone())
                        } else if !q.trim().is_empty() {
                            Some(PaletteAction::Ask(q.clone()))
                        } else {
                            None
                        };
                    }
                    if ui.input(|i| i.key_pressed(Key::Escape)) {
                        close = true;
                    }
                });
            if let Some(a) = run {
                self.palette = None;
                self.run_palette(a);
            } else if close {
                self.palette = None;
            }
        }

        if self.show_help {
            let mut open = true;
            egui::Window::new("Keyboard & Voice Reference")
                .open(&mut open)
                .default_width(560.0)
                .collapsible(false)
                .show(ctx, |ui| {
                    egui::Grid::new("help")
                        .striped(true)
                        .num_columns(2)
                        .show(ui, |ui| {
                            for (k, d) in HELP {
                                ui.label(
                                    RichText::new(*k)
                                        .monospace()
                                        .color(Color32::from_rgb(255, 200, 120)),
                                );
                                ui.label(*d);
                                ui.end_row();
                            }
                        });
                    ui.separator();
                    ui.label(RichText::new("Voice examples").strong());
                    for s in [
                        "“add a blue sphere”",
                        "“move it up two”",
                        "“rotate ninety degrees on x”",
                        "“make it bigger”",
                        "“build a little house with a red roof”",
                        "“animate the cube bouncing”",
                        "“top view” / “zoom out” / “undo”",
                    ] {
                        ui.label(s);
                    }
                });
            self.show_help = open;
        }

        // toasts
        self.toasts.retain(|t| t.born.elapsed().as_secs_f32() < 4.0);
        if !self.toasts.is_empty() {
            egui::Area::new(egui::Id::unique("toasts"))
                .anchor(Align2::CENTER_BOTTOM, vec2(0.0, -110.0))
                .order(egui::Order::Tooltip)
                .interactable(false)
                .show(ctx, |ui| {
                    for t in &self.toasts {
                        let a = (4.0 - t.born.elapsed().as_secs_f32()).clamp(0.0, 1.0);
                        egui::Frame::new()
                            .fill(Color32::from_black_alpha((220.0 * a) as u8))
                            .stroke(Stroke::new(1.0, t.color.gamma_multiply(a)))
                            .corner_radius(8.0)
                            .inner_margin(Margin::symmetric(12, 6))
                            .show(ui, |ui| {
                                ui.set_max_width(520.0);
                                ui.add(
                                    egui::Label::new(
                                        RichText::new(&t.text)
                                            .color(Color32::WHITE.gamma_multiply(a)),
                                    )
                                    .wrap(),
                                );
                            });
                        ui.add_space(4.0);
                    }
                });
            ctx.request_repaint_after(std::time::Duration::from_millis(100));
        }
    }

    fn run_palette(&mut self, a: PaletteAction) {
        match a {
            PaletteAction::Cmd(c) => {
                let log = self.ed.execute(&[c]);
                if let Some(l) = log.first() {
                    self.toast(
                        l.clone(),
                        if l.starts_with('✔') {
                            theme::OK
                        } else {
                            theme::ERR
                        },
                    );
                }
            }
            PaletteAction::Add(p) => {
                let c = self.vp.cursor;
                self.ed.add_primitive(p, c);
            }
            PaletteAction::Ask(q) => {
                self.settings.show_assistant = true;
                self.submit_prompt(q, false, 0.0);
            }
            PaletteAction::Voice => self.toggle_mic(),
            PaletteAction::Save => self.save(false),
            PaletteAction::Open => self.open(),
            PaletteAction::Export => self.export_obj(),
            PaletteAction::ExportGltf => self.export_gltf(),
            PaletteAction::ImportGltf => self.import_gltf_dialog(),
            PaletteAction::Tool(t) => self.vp.tool = t,
        }
    }
}

#[derive(Clone)]
enum PaletteAction {
    Cmd(Command),
    Add(Primitive),
    Ask(String),
    Voice,
    Save,
    Open,
    Export,
    ExportGltf,
    ImportGltf,
    Tool(Tool),
}

const ADD_SPECIAL: &[(&str, &str)] = &[
    ("💡  Point Light", "light"),
    ("☀  Sun", "sun"),
    ("🔦  Spot Light", "spot"),
    ("🎥  Camera", "camera"),
];

fn palette_items() -> Vec<(String, PaletteAction)> {
    let mut v: Vec<(String, PaletteAction)> = Primitive::ALL
        .iter()
        .map(|p| {
            (
                format!("Add ▸ Mesh ▸ {}", p.label()),
                PaletteAction::Add(*p),
            )
        })
        .collect();
    let view = |n: &str| Command {
        action: "view".into(),
        view: Some(n.into()),
        ..Default::default()
    };
    let shade = |n: &str| Command {
        action: "shading".into(),
        view: Some(n.into()),
        ..Default::default()
    };
    v.extend([
        (
            "Voice ▸ Start / stop listening".to_string(),
            PaletteAction::Voice,
        ),
        (
            "Add ▸ Light ▸ Point".into(),
            PaletteAction::Cmd(Command {
                action: "add".into(),
                primitive: Some("light".into()),
                location: Some(serde_json::json!([0, 0, 4])),
                ..Default::default()
            }),
        ),
        (
            "Add ▸ Light ▸ Sun".into(),
            PaletteAction::Cmd(Command {
                action: "add".into(),
                primitive: Some("sun".into()),
                location: Some(serde_json::json!([0, 0, 6])),
                look_at: Some(serde_json::json!([1, 1, 0])),
                ..Default::default()
            }),
        ),
        (
            "Add ▸ Light ▸ Spot".into(),
            PaletteAction::Cmd(Command {
                action: "add".into(),
                primitive: Some("spot".into()),
                location: Some(serde_json::json!([3, -3, 4])),
                look_at: Some(serde_json::json!([0, 0, 0])),
                ..Default::default()
            }),
        ),
        (
            "Add ▸ Camera".into(),
            PaletteAction::Cmd(Command {
                action: "add".into(),
                primitive: Some("camera".into()),
                location: Some(serde_json::json!([7, -7, 5])),
                look_at: Some(serde_json::json!([0, 0, 0])),
                ..Default::default()
            }),
        ),
        (
            "View ▸ Camera (look through)".into(),
            PaletteAction::Cmd(view("camera")),
        ),
        (
            "View ▸ Align Active Camera to View".into(),
            PaletteAction::Cmd(view("align_camera")),
        ),
        (
            "Edit ▸ Undo".into(),
            PaletteAction::Cmd(Command::new("undo")),
        ),
        (
            "Edit ▸ Redo".into(),
            PaletteAction::Cmd(Command::new("redo")),
        ),
        (
            "Object ▸ Delete".into(),
            PaletteAction::Cmd(Command::new("delete")),
        ),
        (
            "Object ▸ Duplicate".into(),
            PaletteAction::Cmd(Command::new("duplicate")),
        ),
        (
            "Object ▸ Shade Smooth".into(),
            PaletteAction::Cmd(Command::new("smooth")),
        ),
        (
            "Object ▸ Shade Flat".into(),
            PaletteAction::Cmd(Command::new("flat")),
        ),
        (
            "Object ▸ Insert Keyframe".into(),
            PaletteAction::Cmd(Command::new("keyframe")),
        ),
        (
            "Object ▸ Clear Keyframes".into(),
            PaletteAction::Cmd(Command::new("clear_keyframes")),
        ),
        (
            "Object ▸ Add Array Modifier".into(),
            PaletteAction::Cmd(Command {
                action: "add_modifier".into(),
                modifier: Some("array".into()),
                ..Default::default()
            }),
        ),
        (
            "Object ▸ Add Mirror Modifier".into(),
            PaletteAction::Cmd(Command {
                action: "add_modifier".into(),
                modifier: Some("mirror".into()),
                ..Default::default()
            }),
        ),
        (
            "Object ▸ Add Radial Array Modifier".into(),
            PaletteAction::Cmd(Command {
                action: "add_modifier".into(),
                modifier: Some("radial".into()),
                ..Default::default()
            }),
        ),
        (
            "Object ▸ Hide".into(),
            PaletteAction::Cmd(Command::new("hide")),
        ),
        (
            "Object ▸ Reveal All".into(),
            PaletteAction::Cmd(Command::new("show")),
        ),
        (
            "Select ▸ All".into(),
            PaletteAction::Cmd(Command {
                action: "select".into(),
                target: Some("all".into()),
                ..Default::default()
            }),
        ),
        (
            "Select ▸ None".into(),
            PaletteAction::Cmd(Command::new("deselect_all")),
        ),
        ("View ▸ Front".into(), PaletteAction::Cmd(view("front"))),
        ("View ▸ Top".into(), PaletteAction::Cmd(view("top"))),
        ("View ▸ Right".into(), PaletteAction::Cmd(view("right"))),
        ("View ▸ Frame All".into(), PaletteAction::Cmd(view("all"))),
        (
            "View ▸ Frame Selected".into(),
            PaletteAction::Cmd(view("selected")),
        ),
        (
            "View ▸ Toggle Perspective/Ortho".into(),
            PaletteAction::Cmd(view("ortho")),
        ),
        (
            "Shading ▸ Wireframe".into(),
            PaletteAction::Cmd(shade("wireframe")),
        ),
        ("Shading ▸ Solid".into(), PaletteAction::Cmd(shade("solid"))),
        (
            "Shading ▸ Material Preview".into(),
            PaletteAction::Cmd(shade("material")),
        ),
        (
            "Playback ▸ Play".into(),
            PaletteAction::Cmd(Command::new("play")),
        ),
        (
            "Playback ▸ Stop".into(),
            PaletteAction::Cmd(Command::new("stop")),
        ),
        (
            "Scene ▸ Clear All Objects".into(),
            PaletteAction::Cmd(Command::new("clear_scene")),
        ),
        ("File ▸ Save".into(), PaletteAction::Save),
        ("File ▸ Open".into(), PaletteAction::Open),
        ("File ▸ Export OBJ".into(), PaletteAction::Export),
        (
            "File ▸ Export glTF (.glb)".into(),
            PaletteAction::ExportGltf,
        ),
        ("File ▸ Import glTF (.glb/.gltf)".into(), PaletteAction::ImportGltf),
        (
            "Render ▸ Render Image (F12)".into(),
            PaletteAction::Cmd(Command::new("render")),
        ),
        (
            "Tool ▸ Select Box".into(),
            PaletteAction::Tool(Tool::Select),
        ),
        ("Tool ▸ Move".into(), PaletteAction::Tool(Tool::Move)),
        ("Tool ▸ Rotate".into(), PaletteAction::Tool(Tool::Rotate)),
        ("Tool ▸ Scale".into(), PaletteAction::Tool(Tool::Scale)),
        (
            "Mode ▸ Toggle Edit Mode".into(),
            PaletteAction::Cmd(Command::new("edit_mode")),
        ),
        (
            "Mode ▸ Object Mode".into(),
            PaletteAction::Cmd(Command::new("object_mode")),
        ),
        (
            "Mesh ▸ Extrude 1 unit".into(),
            PaletteAction::Cmd(Command {
                action: "extrude".into(),
                number: Some(1.0),
                ..Default::default()
            }),
        ),
        (
            "Mesh ▸ Inset Faces".into(),
            PaletteAction::Cmd(Command {
                action: "inset".into(),
                number: Some(0.25),
                ..Default::default()
            }),
        ),
        (
            "Mesh ▸ Bevel Edges".into(),
            PaletteAction::Cmd(Command {
                action: "bevel".into(),
                number: Some(0.1),
                segments: Some(1.0),
                ..Default::default()
            }),
        ),
        (
            "Mesh ▸ Bevel (rounded)".into(),
            PaletteAction::Cmd(Command {
                action: "bevel".into(),
                number: Some(0.15),
                segments: Some(4.0),
                ..Default::default()
            }),
        ),
        (
            "Mesh ▸ Subdivide".into(),
            PaletteAction::Cmd(Command::new("subdivide")),
        ),
        (
            "Mesh ▸ Loop Cut (horizontal)".into(),
            PaletteAction::Cmd(Command {
                action: "loop_cut".into(),
                vector: Some(serde_json::json!([0, 0, 1])),
                ..Default::default()
            }),
        ),
        (
            "Mesh ▸ Merge at Center".into(),
            PaletteAction::Cmd(Command::new("merge")),
        ),
        (
            "Mesh ▸ Flip Normals".into(),
            PaletteAction::Cmd(Command::new("flip_normals")),
        ),
        (
            "Mesh ▸ Select Top Faces".into(),
            PaletteAction::Cmd(Command {
                action: "select_faces".into(),
                vector: Some(serde_json::json!([0, 0, 1])),
                ..Default::default()
            }),
        ),
    ]);
    v
}

const HELP: &[(&str, &str)] = &[
    (
        "LMB / Shift+LMB",
        "Select / extend selection (drag for box select)",
    ),
    ("MMB drag", "Orbit (Alt+LMB on laptops)"),
    ("Shift+MMB", "Pan"),
    ("Wheel / Ctrl+MMB", "Zoom"),
    (
        "G / R / S",
        "Grab, Rotate, Scale  (then X/Y/Z to constrain, type numbers, Ctrl snaps)",
    ),
    ("Shift+A", "Add menu (objects are placed at the 3D cursor)"),
    ("Shift+RMB", "Place 3D cursor on the ground"),
    ("Shift+D", "Duplicate and move"),
    ("X / Delete", "Delete selection"),
    ("A / Alt+A", "Select all / none"),
    ("H / Alt+H", "Hide / reveal"),
    (
        "1 3 7 (Ctrl)",
        "Front, Right, Top (Back, Left, Bottom) views",
    ),
    ("5", "Toggle orthographic / perspective"),
    (
        "0 / Ctrl+Alt+0",
        "Look through camera / align camera to view",
    ),
    ("F / Home", "Frame selected / all"),
    ("Z / Alt+Z", "Cycle shading / X-ray"),
    ("W", "Cycle tool (select, move, rotate, scale)"),
    ("I", "Insert keyframe"),
    ("Space", "Play / pause"),
    ("← → / Shift+← →", "Step frame / jump to start-end"),
    ("F3 / Ctrl+K", "Command search (falls back to AI)"),
    ("F12 / F11", "Render image through the camera / show the last render"),
    ("Alt+1/2/3 (Alt+0)", "Subdivision Surface level on the selection (remove)"),
    ("Ctrl+P / Alt+P", "Parent selection to the active object / clear parent"),
    ("Ctrl+Tab", "Sculpt mode on the active mesh (X draw, I inflate, S smooth, G grab, C crease, T flatten; [ ] radius)"),
    ("M", "Move selection to a collection"),
    ("Ctrl+Space or `", "Voice: start / stop listening"),
    ("/ or Ctrl+L", "Focus the AI prompt"),
    ("N", "Toggle the AI Copilot panel"),
    (
        "Ctrl+Z / Ctrl+Shift+Z",
        "Undo / redo (AI batches are one step)",
    ),
    ("Ctrl+S / Ctrl+O / Ctrl+N", "Save / open / new"),
    ("Tab", "Toggle Edit Mode on the active mesh"),
    ("1 / 2 / 3 (edit)", "Vertex / edge / face select"),
    ("E / I (edit)", "Extrude along normal / inset faces"),
    ("Ctrl+R (edit)", "Loop cut (wheel or +/- = number of cuts)"),
    ("Ctrl+B (edit)", "Bevel selected edges (mouse = width, wheel = segments)"),
    ("O (edit)", "Proportional editing on/off (wheel = radius during G/R/S)"),
    (
        "M / X / Alt+N (edit)",
        "Merge at centre / delete / flip normals",
    ),
];

fn section(ui: &mut Ui, title: &str, add: impl FnOnce(&mut Ui)) {
    egui::CollapsingHeader::new(RichText::new(title).strong())
        .default_open(true)
        .show(ui, |ui| {
            egui::Frame::new()
                .fill(Color32::from_rgb(58, 58, 58))
                .corner_radius(5.0)
                .inner_margin(8.0)
                .show(ui, |ui| {
                    ui.set_width(ui.available_width());
                    add(ui);
                });
        });
}

fn vec3_edit(ui: &mut Ui, label: &str, v: &mut Vec3, speed: f64) {
    ui.label(RichText::new(label).color(theme::TEXT_DIM));
    for (i, (ax, col)) in [
        ("X", Color32::from_rgb(200, 70, 80)),
        ("Y", Color32::from_rgb(120, 180, 40)),
        ("Z", Color32::from_rgb(60, 130, 220)),
    ]
    .into_iter()
    .enumerate()
    {
        ui.horizontal(|ui| {
            ui.label(RichText::new(ax).color(col).strong());
            ui.add_sized(
                vec2(ui.available_width(), 20.0),
                egui::DragValue::new(&mut v[i]).speed(speed).max_decimals(3),
            );
        });
    }
}

fn object_props(ui: &mut Ui, o: &mut crate::scene::Object) {
    ui.horizontal(|ui| {
        ui.label(RichText::new("▽").color(Color32::from_rgb(255, 160, 80)));
        ui.text_edit_singleline(&mut o.name);
    });
    section(ui, "Transform", |ui| {
        vec3_edit(ui, "Location", &mut o.transform.location, 0.02);
        ui.add_space(4.0);
        vec3_edit(ui, "Rotation (°)", &mut o.transform.rotation, 0.5);
        ui.add_space(4.0);
        vec3_edit(ui, "Scale", &mut o.transform.scale, 0.01);
    });
    match &mut o.kind {
        crate::scene::ObjectKind::Light(l) => {
            section(ui, "Light", |ui| {
                ui.horizontal(|ui| {
                    for t in [
                        crate::scene::LightType::Point,
                        crate::scene::LightType::Sun,
                        crate::scene::LightType::Spot,
                    ] {
                        if ui
                            .add(egui::Button::selectable(l.light_type == t, t.label()))
                            .clicked()
                            && l.light_type != t
                        {
                            l.light_type = t;
                            l.power = if t == crate::scene::LightType::Sun {
                                3.0
                            } else {
                                1000.0
                            };
                        }
                    }
                });
                ui.horizontal(|ui| {
                    ui.label("Color");
                    ui.color_edit_button_rgb(&mut l.color);
                });
                let (range, unit) = if l.light_type == crate::scene::LightType::Sun {
                    (0.0..=20.0, "Strength")
                } else {
                    (0.0..=10000.0, "Power (W)")
                };
                ui.add(
                    egui::Slider::new(&mut l.power, range)
                        .text(unit)
                        .logarithmic(l.light_type != crate::scene::LightType::Sun),
                );
                if l.light_type == crate::scene::LightType::Spot {
                    ui.add(egui::Slider::new(&mut l.spot_angle, 1.0..=180.0).text("Spot size °"));
                    ui.add(egui::Slider::new(&mut l.spot_blend, 0.0..=1.0).text("Blend"));
                }
                ui.label(
                    RichText::new("Lights shade the scene in Material Preview (Z).")
                        .small()
                        .color(theme::TEXT_DIM),
                );
            });
        }
        crate::scene::ObjectKind::Camera(c) => {
            section(ui, "Camera", |ui| {
                let mut lens = 18.0 / (c.fov_deg.to_radians() * 0.5).tan(); // 36x24 sensor, vertical
                if ui
                    .add(
                        egui::Slider::new(&mut lens, 10.0..=200.0)
                            .text("Focal length (mm)")
                            .logarithmic(true),
                    )
                    .changed()
                {
                    c.fov_deg = (2.0 * (18.0 / lens).atan()).to_degrees();
                }
                ui.add(egui::Slider::new(&mut c.fov_deg, 5.0..=120.0).text("Field of view °"));
                ui.label(
                    RichText::new(
                        "Numpad 0 looks through the camera; Ctrl+Alt+0 aligns it to the view.",
                    )
                    .small()
                    .color(theme::TEXT_DIM),
                );
            });
        }
        crate::scene::ObjectKind::Mesh => {
            scatter_props(ui, o);
            shape_key_props(ui, o);
            cloth_props(ui, o);
        }
        crate::scene::ObjectKind::Armature => armature_props(ui, o),
        crate::scene::ObjectKind::Empty => {}
    }
    if o.is_mesh() {
        section(ui, "Mesh", |ui| {
            if let Some(pm) = &o.mesh {
                ui.label(format!(
                    "Edited mesh · {} vertices · {} faces",
                    pm.verts.len(),
                    pm.faces.len()
                ));
                if ui
                    .button("⟲ Revert to procedural primitive")
                    .on_hover_text("Discard mesh edits")
                    .clicked()
                {
                    o.mesh = None;
                }
            } else {
                egui::ComboBox::from_label("Primitive")
                    .selected_text(o.primitive.label())
                    .show_ui(ui, |ui| {
                        for p in Primitive::ALL {
                            ui.selectable_value(&mut o.primitive, p, p.label());
                        }
                    });
                ui.add(egui::Slider::new(&mut o.detail, 1..=6).text("Resolution"));
            }
            let mut smooth = o.smooth();
            ui.horizontal(|ui| {
                if ui.checkbox(&mut smooth, "Shade smooth").changed() {
                    o.smooth_override = Some(smooth);
                }
                let mut auto = o.auto_smooth > 0.0;
                if ui.add_enabled(smooth, egui::Checkbox::new(&mut auto, "Auto")).on_hover_text("Shade Auto Smooth: keep edges sharper than the angle hard").changed() {
                    o.auto_smooth = if auto { 30.0 } else { 0.0 };
                }
                if smooth && o.auto_smooth > 0.0 {
                    ui.add(egui::DragValue::new(&mut o.auto_smooth).range(1.0..=179.0).suffix("°").speed(0.5));
                }
            });
            ui.checkbox(&mut o.visible, "Visible in viewport");
        });
    }
    if !o.keyframes.is_empty() {
        section(ui, "Animation", |ui| {
            ui.label(format!(
                "{} keyframes: {}",
                o.keyframes.len(),
                o.keyframes
                    .iter()
                    .map(|k| k.frame.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
            if ui.button("Clear keyframes").clicked() {
                o.keyframes.clear();
            }
        });
    }
}

fn modifier_props(ui: &mut Ui, o: &mut crate::scene::Object) {
    ui.menu_button("＋ Add Modifier", |ui| {
        if ui.button("Array").clicked() {
            o.modifiers.push(Modifier::Array {
                count: 3,
                offset: Vec3::new(2.2, 0.0, 0.0),
            });
            ui.close();
        }
        if ui.button("Mirror").clicked() {
            o.modifiers.push(Modifier::Mirror {
                x: true,
                y: false,
                z: false,
            });
            ui.close();
        }
        if ui.button("Radial Array").clicked() {
            o.modifiers.push(Modifier::Radial {
                count: 8,
                radius: 3.0,
            });
            ui.close();
        }
        if ui.button("Subdivision Surface").clicked() {
            o.modifiers.push(Modifier::Subsurf { levels: 2 });
            o.smooth_override = Some(true);
            ui.close();
        }
    });
    let mut remove = None;
    let mut up = None;
    for (i, m) in o.modifiers.iter_mut().enumerate() {
        let title = format!("🔧 {}", m.label());
        egui::CollapsingHeader::new(RichText::new(title).strong())
            .id_salt(("mod", i))
            .default_open(true)
            .show(ui, |ui| {
                egui::Frame::new()
                    .fill(Color32::from_rgb(58, 58, 58))
                    .corner_radius(5.0)
                    .inner_margin(8.0)
                    .show(ui, |ui| {
                        ui.set_width(ui.available_width());
                        match m {
                            Modifier::Array { count, offset } => {
                                ui.add(egui::Slider::new(count, 1..=100).text("Count"));
                                vec3_edit(ui, "Offset", offset, 0.02);
                            }
                            Modifier::Mirror { x, y, z } => {
                                ui.horizontal(|ui| {
                                    ui.toggle_value(x, "X");
                                    ui.toggle_value(y, "Y");
                                    ui.toggle_value(z, "Z");
                                });
                            }
                            Modifier::Radial { count, radius } => {
                                ui.add(egui::Slider::new(count, 1..=100).text("Count"));
                                ui.add(egui::Slider::new(radius, 0.0..=30.0).text("Radius"));
                            }
                            Modifier::Subsurf { levels } => {
                                ui.add(egui::Slider::new(levels, 0..=4).text("Levels"));
                                ui.label(RichText::new("Catmull–Clark. Edit mode shows the cage; Alt+1/2/3 sets levels.").small().color(theme::TEXT_DIM));
                            }
                        }
                        ui.horizontal(|ui| {
                            if i > 0 && ui.small_button("▲").clicked() {
                                up = Some(i);
                            }
                            if ui.small_button("✖ Remove").clicked() {
                                remove = Some(i);
                            }
                        });
                    });
            });
    }
    if let Some(i) = up {
        o.modifiers.swap(i, i - 1);
    }
    if let Some(i) = remove {
        o.modifiers.remove(i);
    }
    if o.modifiers.is_empty() {
        ui.label(
            RichText::new("No modifiers. Try asking the Copilot: “add a radial array of 6”.")
                .small()
                .color(theme::TEXT_DIM),
        );
    }
}

/// Returns a texture request: a pattern name, an image path, "none", or "<load>" (file dialog).
fn material_props(ui: &mut Ui, o: &mut crate::scene::Object, textures: &[crate::textures::Texture]) -> Option<String> {
    if !o.is_mesh() {
        ui.label(
            RichText::new("Lights and cameras have no material — see the Object tab.")
                .color(theme::TEXT_DIM),
        );
        return None;
    }
    let mut action = None;
    // preview swatch
    let (r, _) = ui.allocate_exact_size(vec2(ui.available_width(), 56.0), Sense::hover());
    let p = ui.painter();
    p.rect_filled(r, 6.0, Color32::from_rgb(30, 30, 30));
    let c = r.center();
    let base = o.material.color;
    for i in (0..24).rev() {
        let t = i as f32 / 24.0;
        let lit = 0.35 + 0.75 * (1.0 - t);
        let spec = (1.0 - t).powf(1.0 / (o.material.roughness.max(0.05) * 0.5))
            * (0.3 + o.material.metallic * 0.5);
        let col = [
            base[0] * lit + spec,
            base[1] * lit + spec,
            base[2] * lit + spec,
        ];
        p.circle_filled(
            c + vec2(-t * 4.0, -t * 4.0) * -1.0 + vec2(-4.0 * (1.0 - t), -4.0 * (1.0 - t)),
            24.0 * (1.0 - t * 0.85),
            color32(col),
        );
    }
    section(ui, "Surface", |ui| {
        ui.horizontal(|ui| {
            ui.label("Base Color");
            ui.color_edit_button_rgb(&mut o.material.color);
        });
        ui.horizontal_wrapped(|ui| {
            for (n, c) in crate::editor::NAMED_COLORS.iter().take(16) {
                let (r, resp) = ui.allocate_exact_size(vec2(16.0, 16.0), Sense::click());
                ui.painter().rect_filled(r, 3.0, color32(*c));
                if resp.on_hover_text(*n).clicked() {
                    o.material.color = *c;
                }
            }
        });
        ui.add(egui::Slider::new(&mut o.material.metallic, 0.0..=1.0).text("Metallic"));
        ui.add(egui::Slider::new(&mut o.material.roughness, 0.0..=1.0).text("Roughness"));
        ui.add(egui::Slider::new(&mut o.material.emission, 0.0..=5.0).text("Emission"));
        ui.add(egui::Slider::new(&mut o.material.alpha, 0.05..=1.0).text("Alpha"));
        ui.label(
            RichText::new("Tip: switch to Material Preview (Z) to see metal/roughness.")
                .small()
                .color(theme::TEXT_DIM),
        );
    });
    section(ui, "Texture", |ui| {
        let current = textures.iter().find(|t| t.id == o.material.texture).map(|t| t.name.clone()).unwrap_or_else(|| "None".into());
        ui.horizontal(|ui| {
            ui.label("Image");
            egui::ComboBox::from_id_salt("tex_pick").selected_text(current).show_ui(ui, |ui| {
                ui.selectable_value(&mut o.material.texture, 0, "None");
                for t in textures {
                    ui.selectable_value(&mut o.material.texture, t.id, &t.name);
                }
            });
            if ui.button("Load image…").clicked() {
                action = Some("<load>".to_string());
            }
        });
        ui.label(RichText::new("Procedural").small().color(theme::TEXT_DIM));
        ui.horizontal_wrapped(|ui| {
            for (name, _, _) in crate::textures::PROCEDURAL {
                if ui.small_button(*name).clicked() {
                    action = Some(name.to_string());
                }
            }
        });
        if o.material.texture != 0 {
            ui.add(egui::Slider::new(&mut o.material.tex_scale, 0.1..=20.0).logarithmic(true).text("Repeats"));
            ui.label(RichText::new("The base colour tints the texture (white = original colours). Seen in Material Preview and F12 renders.").small().color(theme::TEXT_DIM));
        }
    });
    section(ui, "Surface Detail", |ui| {
        let name_of = |id: u32| textures.iter().find(|t| t.id == id).map(|t| t.name.clone()).unwrap_or_else(|| "None".into());
        ui.horizontal(|ui| {
            ui.label("Normal");
            egui::ComboBox::from_id_salt("normal_pick").selected_text(name_of(o.material.normal_tex)).show_ui(ui, |ui| {
                ui.selectable_value(&mut o.material.normal_tex, 0, "None");
                for t in textures {
                    ui.selectable_value(&mut o.material.normal_tex, t.id, &t.name);
                }
            });
            if ui.button("Load…").on_hover_text("Load a normal map image (or any picture: a bump map is made from it)").clicked() {
                action = Some("<load-normal>".to_string());
            }
        });
        ui.horizontal(|ui| {
            if ui.add_enabled(o.material.texture != 0, egui::Button::new("Bumps from texture")).on_hover_text("Generate a normal map from the colour texture (dark = low)").clicked() {
                action = Some("<bump>".to_string());
            }
            if ui.add_enabled(o.material.texture != 0, egui::Button::new("Inverted")).on_hover_text("Generate it with light = low").clicked() {
                action = Some("<bump-invert>".to_string());
            }
        });
        if o.material.normal_tex != 0 {
            ui.add(egui::Slider::new(&mut o.material.normal_strength, 0.0..=4.0).text("Strength"));
        }
        ui.horizontal(|ui| {
            ui.label("Roughness");
            egui::ComboBox::from_id_salt("rough_pick").selected_text(name_of(o.material.rough_tex)).show_ui(ui, |ui| {
                ui.selectable_value(&mut o.material.rough_tex, 0, "None");
                for t in textures {
                    ui.selectable_value(&mut o.material.rough_tex, t.id, &t.name);
                }
            });
            if ui.button("Load…").on_hover_text("Load a roughness map (grey) or a glTF metallic-roughness map").clicked() {
                action = Some("<load-rough>".to_string());
            }
        });
        if o.material.texture != 0 && ui.button("Roughness from texture").on_hover_text("Rough in the grooves, shinier on top").clicked() {
            action = Some("<rough>".to_string());
        }
        ui.label(RichText::new("Normal maps show in Material Preview; roughness maps in path-traced renders.").small().color(theme::TEXT_DIM));
    });
    action
}

fn shape_key_props(ui: &mut Ui, o: &mut crate::scene::Object) {
    if !o.is_mesh() || o.scatter.is_some() {
        return;
    }
    let editing = o.shape_editing;
    section(ui, "Shape Keys", |ui| {
        let mut act: Option<(&str, usize)> = None;
        for (i, k) in o.shape_keys.iter_mut().enumerate() {
            ui.horizontal(|ui| {
                ui.label(RichText::new(&k.name).small());
                ui.add_enabled(!editing, egui::Slider::new(&mut k.value, 0.0..=1.0).show_value(true));
                if ui.small_button("✎").on_hover_text("Edit or sculpt this shape").clicked() {
                    act = Some(("edit", i));
                }
                if ui.small_button("◆").on_hover_text("Keyframe the value at the current frame").clicked() {
                    act = Some(("key", i));
                }
                if ui.small_button("✖").on_hover_text("Remove").clicked() {
                    act = Some(("remove", i));
                }
            });
        }
        ui.horizontal(|ui| {
            if ui.button("+ Add Shape Key").clicked() {
                act = Some(("add", 0));
            }
            if editing && ui.button("Finish editing").clicked() {
                act = Some(("finish", 0));
            }
        });
        if let Some(a) = act {
            ui.ctx().data_mut(|d| d.insert_temp(egui::Id::unique("shape_action"), (a.0.to_string(), a.1)));
        }
        if o.shape_keys.is_empty() {
            ui.label(RichText::new("Shapes to blend (expressions, squash & stretch); animate their values with ◆.").small().color(theme::TEXT_DIM));
        }
    });
}

fn cloth_props(ui: &mut Ui, o: &mut crate::scene::Object) {
    if o.scatter.is_some() || o.skin.is_some() {
        return;
    }
    section(ui, "Physics", |ui| {
        let mut act: Option<String> = None;
        match o.cloth.as_mut() {
            None => {
                ui.horizontal_wrapped(|ui| {
                    for (l, pins) in [("Cloth", "none"), ("Cloth · pin top", "top"), ("Cloth · pin corners", "corners"), ("Flag (pin left)", "left")] {
                        if ui.button(l).clicked() {
                            act = Some(pins.into());
                        }
                    }
                });
                ui.label(RichText::new("Drapes over the other meshes (spheres stay round, everything else is a box) and is baked for the timeline's frames.").small().color(theme::TEXT_DIM));
            }
            Some(c) => {
                ui.add(egui::Slider::new(&mut c.stiffness, 0.05..=1.0).text("Stiffness"));
                ui.add(egui::Slider::new(&mut c.bending, 0.0..=1.0).text("Bending"));
                ui.add(egui::Slider::new(&mut c.damping, 0.9..=1.0).text("Air (keep)"));
                ui.add(egui::Slider::new(&mut c.gravity, 0.0..=30.0).text("Gravity"));
                ui.label(RichText::new(format!("{} pinned · {} baked frames", c.pins.len(), c.cache.len())).small().color(theme::TEXT_DIM));
                ui.horizontal_wrapped(|ui| {
                    if ui.button("Bake").clicked() {
                        act = Some("<bake>".into());
                    }
                    egui::ComboBox::from_id_salt("pins").selected_text("Pin…").show_ui(ui, |ui| {
                        for p in ["none", "top", "corners", "left", "right", "front", "back", "selected"] {
                            if ui.selectable_label(false, p).clicked() {
                                act = Some(p.into());
                            }
                        }
                    });
                    if ui.button("Remove").clicked() {
                        act = Some("<remove>".into());
                    }
                });
            }
        }
        if let Some(a) = act {
            ui.ctx().data_mut(|d| d.insert_temp(egui::Id::unique("cloth_action"), a));
        }
    });
}

fn scatter_props(ui: &mut Ui, o: &mut crate::scene::Object) {
    let Some(sc) = o.scatter.as_mut() else { return };
    section(ui, "Scatter", |ui| {
        ui.add(egui::Slider::new(&mut sc.count, 1..=5000).logarithmic(true).text("Count"));
        ui.add(egui::DragValue::new(&mut sc.seed).prefix("Seed "));
        ui.horizontal(|ui| {
            ui.label("Scale");
            ui.add(egui::DragValue::new(&mut sc.scale_min).range(0.01..=100.0).speed(0.01).prefix("min "));
            ui.add(egui::DragValue::new(&mut sc.scale_max).range(0.01..=100.0).speed(0.01).prefix("max "));
        });
        if sc.scale_max < sc.scale_min {
            sc.scale_max = sc.scale_min;
        }
        ui.add(egui::Slider::new(&mut sc.spin, 0.0..=360.0).text("Random spin °"));
        ui.add(egui::Slider::new(&mut sc.tilt, 0.0..=90.0).text("Random tilt °"));
        ui.add(egui::Slider::new(&mut sc.max_slope, 0.0..=180.0).text("Max slope °"));
        ui.checkbox(&mut sc.align, "Align to surface normal");
        ui.label(RichText::new("Edit the hidden source object (outliner) to change every copy. Object ▸ Scatter ▸ Make Real bakes it.").small().color(theme::TEXT_DIM));
    });
}

fn armature_props(ui: &mut Ui, o: &mut crate::scene::Object) {
    let Some(arm) = o.armature.as_mut() else { return };
    section(ui, "Armature", |ui| {
        ui.checkbox(&mut arm.rest_position, "Rest Position");
        if ui.button("Reset Pose").clicked() {
            arm.reset_pose();
        }
        let keyed = arm.bones.iter().filter(|b| !b.keys.is_empty()).count();
        ui.label(RichText::new(format!("{} bones{}", arm.bones.len(), if keyed > 0 { format!(", {keyed} animated") } else { String::new() })).small().color(theme::TEXT_DIM));
    });
    if !arm.actions.is_empty() {
        section(ui, "Actions", |ui| {
            let mut pick = None;
            for (i, a) in arm.actions.iter().enumerate() {
                let r = a.range().map(|(l, h)| format!("  {l}–{h}")).unwrap_or_default();
                if ui.selectable_label(i == arm.active_action, format!("{}{r}", a.name)).clicked() && i != arm.active_action {
                    pick = Some(i);
                }
            }
            if let Some(i) = pick {
                // cross-fade into the new action
                let start = arm.actions.get(i).and_then(|a| a.range()).map(|r| r.0 as f32).unwrap_or(1.0);
                arm.set_action_blended(i, start, 6.0);
            }
            if ui.small_button("New Action").clicked() {
                arm.new_action("Action");
            }
            ui.label(RichText::new("Switching sets the timeline to the action's frames.").small().color(theme::TEXT_DIM));
        });
    } else if arm.has_keys() && ui.small_button("Store as Action").on_hover_text("Keep this animation as a named action and start another").clicked() {
        arm.new_action("Action");
    }
    section(ui, "Bones (pose)", |ui| {
        egui::ScrollArea::vertical().max_height(260.0).id_salt("bones").show(ui, |ui| {
            for b in arm.bones.iter_mut() {
                let mut e = b.pose.euler_deg();
                ui.horizontal(|ui| {
                    ui.label(RichText::new(&b.name).small()).on_hover_text("Rotation in the bone's own axes (°)");
                });
                ui.horizontal(|ui| {
                    let mut ch = false;
                    for (k, l) in ["X", "Y", "Z"].iter().enumerate() {
                        ch |= ui.add(egui::DragValue::new(&mut e[k]).speed(0.5).prefix(format!("{l} ")).suffix("°").max_decimals(1)).changed();
                    }
                    if ch {
                        b.pose.set_euler_deg(e);
                    }
                    if ui.small_button("⟲").on_hover_text("Clear").clicked() {
                        b.pose = Default::default();
                    }
                });
            }
        });
        ui.label(RichText::new("Pose Mode (Ctrl+Tab): click a bone, R rotate, G move, I keyframe.").small().color(theme::TEXT_DIM));
    });
}

fn short_model(m: &str) -> &str {
    m.strip_suffix(":latest").unwrap_or(m)
}

fn clean_transcript(t: &str) -> String {
    let t = t.trim();
    let lower = t.to_lowercase();
    // Whisper hallucinations on silence
    for junk in [
        "[blank_audio]",
        "(silence)",
        "[silence]",
        "thank you.",
        "thanks for watching",
        "you",
        "[music]",
        "(music)",
    ] {
        if lower == junk {
            return String::new();
        }
    }
    t.trim_matches(|c: char| c == '"' || c.is_whitespace())
        .to_string()
}


impl eframe::App for App {
    fn ui(&mut self, ui: &mut Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        self.poll_events();
        self.ed.propagate_parents();
        self.poll_mcp();
        self.voice_tick(&ctx);
        self.shortcuts(&ctx);
        self.tick_playback(&ctx);

        self.top_bar(ui);
        self.bottom(ui);
        self.assistant_panel(ui);
        self.right_panel(ui);
        egui::CentralPanel::no_frame().show(ui, |ui| {
            self.vp.ui(ui, &mut self.ed);
        });
        // children follow whatever the viewport just moved
        self.ed.propagate_parents();
        self.popups(&ctx);
        self.poll_path_trace();
        self.autosave_tick(&ctx);
        let drop_at = ctx.input(|i| i.pointer.hover_pos()).or(ctx.pointer_latest_pos());
        for f in ctx.input(|i| i.raw.dropped_files.clone()) {
            let p = f.path().to_path_buf();
            let ext = p.extension().map(|e| e.to_string_lossy().to_lowercase()).unwrap_or_default();
            let over_chat = drop_at.is_some_and(|q| self.assistant_rect.contains(q));
            if over_chat && matches!(ext.as_str(), "png" | "jpg" | "jpeg") {
                if let Ok(b) = std::fs::read(&p) {
                    let name = p.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
                    self.attach_image(&ctx, &name, &b);
                }
                continue;
            }
            self.import_path(&p);
        }
        if std::mem::take(&mut self.ed.render_request) {
            self.render_now(&ctx);
        }
        self.render_window(&ctx);
        self.uv_window(&ctx);

        let title = format!(
            "Rust Blender{}{}",
            self.ed
                .file_path
                .as_ref()
                .map(|p| format!(" — {}", p.display()))
                .unwrap_or_default(),
            if self.ed.dirty { " *" } else { "" }
        );
        if title != self.last_title {
            ctx.send_viewport_cmd(egui::ViewportCommand::Title(title.clone()));
            self.last_title = title;
        }
    }

    fn on_exit(&mut self) {
        self.settings.gpu_renderer = self.vp.use_gpu;
        self.settings.save();
        // clean exit: no recovery needed next time
        let _ = std::fs::remove_file(autosave_dir().join("session.lock"));
        let _ = std::fs::remove_file(autosave_dir().join("autosave.rblend"));
    }
}


fn autosave_dir() -> std::path::PathBuf {
    dirs::config_dir().unwrap_or_else(std::env::temp_dir).join("rust_blender")
}
