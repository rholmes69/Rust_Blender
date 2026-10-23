//! Voice control: microphone capture (cpal) + fully local speech-to-text with
//! OpenAI Whisper running on the pure-Rust `candle` ML framework.
//! Model weights are downloaded once from Hugging Face and cached.

use crossbeam_channel::{Receiver, Sender, unbounded};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

pub const WHISPER_MODELS: &[(&str, &str)] = &[
    ("tiny.en", "openai/whisper-tiny.en"),
    ("base.en", "openai/whisper-base.en"),
    ("small.en", "openai/whisper-small.en"),
    ("distil-small.en", "distil-whisper/distil-small.en"),
];

#[derive(Debug)]
pub enum VoiceEvent {
    Status(String),
    Download(String, f32),
    Ready(String),
    Transcript { text: String, secs: f32 },
    Error(String),
}

pub enum VoiceRequest {
    Load(String),
    Transcribe(Vec<f32>), // 16 kHz mono
}

pub struct VoiceEngine {
    tx: Sender<VoiceRequest>,
    pub rx: Receiver<VoiceEvent>,
    pub recorder: Recorder,
}

impl VoiceEngine {
    pub fn new(repaint: impl Fn() + Send + Clone + 'static) -> Self {
        let (tx, req_rx) = unbounded::<VoiceRequest>();
        let (ev_tx, rx) = unbounded::<VoiceEvent>();
        let rp = repaint.clone();
        std::thread::Builder::new()
            .name("whisper".into())
            .spawn(move || worker(req_rx, ev_tx, rp))
            .expect("spawn whisper thread");
        Self {
            tx,
            rx,
            recorder: Recorder::new(repaint),
        }
    }
    pub fn load(&self, model: &str) {
        let _ = self.tx.send(VoiceRequest::Load(model.to_string()));
    }
    pub fn transcribe(&self, pcm16k: Vec<f32>) {
        let _ = self.tx.send(VoiceRequest::Transcribe(pcm16k));
    }
}

// ------------------------------------------------------------------------------------------------
// Recorder
// ------------------------------------------------------------------------------------------------

#[derive(Default)]
struct RecShared {
    samples: Vec<f32>,
    level: f32,
}

pub struct Recorder {
    shared: Arc<Mutex<RecShared>>,
    #[cfg(feature = "voice")]
    stream: Option<cpal::Stream>,
    pub sample_rate: u32,
    pub recording: bool,
    pub started: Option<std::time::Instant>,
    pub heard_speech: bool,
    pub silence_since: Option<std::time::Instant>,
    pub device_name: String,
    repaint: Arc<dyn Fn() + Send + Sync>,
}

impl Recorder {
    fn new(repaint: impl Fn() + Send + Clone + 'static) -> Self {
        let r = Mutex::new(repaint);
        Self {
            shared: Default::default(),
            #[cfg(feature = "voice")]
            stream: None,
            sample_rate: 16000,
            recording: false,
            started: None,
            heard_speech: false,
            silence_since: None,
            device_name: String::new(),
            repaint: Arc::new(move || (r.lock().unwrap())()),
        }
    }

    pub fn level(&self) -> f32 {
        self.shared.lock().map(|s| s.level).unwrap_or(0.0)
    }

    #[cfg(feature = "voice")]
    pub fn start(&mut self) -> Result<(), String> {
        use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
        let host = cpal::default_host();
        let dev = host.default_input_device().ok_or("No microphone found")?;
        self.device_name = dev
            .description()
            .map(|d| d.to_string())
            .unwrap_or_else(|_| "Microphone".into());
        let cfg = dev.default_input_config().map_err(|e| e.to_string())?;
        let channels = cfg.channels() as usize;
        self.sample_rate = cfg.sample_rate();
        {
            let mut s = self.shared.lock().unwrap();
            s.samples.clear();
            s.level = 0.0;
        }
        let shared = self.shared.clone();
        let repaint = self.repaint.clone();
        let fmt = cfg.sample_format();
        let config: cpal::StreamConfig = cfg.config();
        let push = move |mono: &mut dyn Iterator<Item = f32>| {
            let mut s = shared.lock().unwrap();
            let start = s.samples.len();
            s.samples.extend(mono);
            let chunk = &s.samples[start..];
            if !chunk.is_empty() {
                let rms = (chunk.iter().map(|x| x * x).sum::<f32>() / chunk.len() as f32).sqrt();
                s.level = s.level * 0.6 + rms * 0.4;
            }
            drop(s);
            repaint();
        };
        let err = |e| eprintln!("audio stream error: {e}");
        let stream = match fmt {
            cpal::SampleFormat::F32 => {
                let push = push;
                dev.build_input_stream::<f32, _, _>(
                    config,
                    move |d, _| {
                        push(
                            &mut d
                                .chunks(channels)
                                .map(|c| c.iter().sum::<f32>() / channels as f32),
                        )
                    },
                    err,
                    None,
                )
            }
            cpal::SampleFormat::I16 => {
                let push = push;
                dev.build_input_stream::<i16, _, _>(
                    config,
                    move |d, _| {
                        push(&mut d.chunks(channels).map(|c| {
                            c.iter().map(|&x| x as f32 / 32768.0).sum::<f32>() / channels as f32
                        }))
                    },
                    err,
                    None,
                )
            }
            cpal::SampleFormat::U16 => {
                let push = push;
                dev.build_input_stream::<u16, _, _>(
                    config,
                    move |d, _| {
                        push(&mut d.chunks(channels).map(|c| {
                            c.iter()
                                .map(|&x| (x as f32 - 32768.0) / 32768.0)
                                .sum::<f32>()
                                / channels as f32
                        }))
                    },
                    err,
                    None,
                )
            }
            cpal::SampleFormat::I32 => {
                let push = push;
                dev.build_input_stream::<i32, _, _>(
                    config,
                    move |d, _| {
                        push(&mut d.chunks(channels).map(|c| {
                            c.iter().map(|&x| x as f32 / 2147483648.0).sum::<f32>()
                                / channels as f32
                        }))
                    },
                    err,
                    None,
                )
            }
            other => return Err(format!("unsupported microphone sample format {other:?}")),
        }
        .map_err(|e| e.to_string())?;
        stream.play().map_err(|e| e.to_string())?;
        self.stream = Some(stream);
        self.recording = true;
        self.started = Some(std::time::Instant::now());
        self.heard_speech = false;
        self.silence_since = None;
        Ok(())
    }

    #[cfg(not(feature = "voice"))]
    pub fn start(&mut self) -> Result<(), String> {
        Err("Built without the `voice` feature".into())
    }

    /// Stop and return 16 kHz mono PCM.
    pub fn stop(&mut self) -> Vec<f32> {
        #[cfg(feature = "voice")]
        {
            self.stream = None;
        }
        self.recording = false;
        let s = std::mem::take(&mut self.shared.lock().unwrap().samples);
        resample(&s, self.sample_rate, 16000)
    }

    /// Simple energy voice-activity detection for hands-free auto-stop.
    /// Returns true when the user has finished speaking.
    pub fn update_vad(&mut self, threshold: f32, silence_secs: f32) -> bool {
        if !self.recording {
            return false;
        }
        let lvl = self.level();
        let now = std::time::Instant::now();
        if lvl > threshold {
            self.heard_speech = true;
            self.silence_since = None;
        } else if self.heard_speech {
            let since = *self.silence_since.get_or_insert(now);
            if now.duration_since(since).as_secs_f32() > silence_secs {
                return true;
            }
        }
        // hard limit
        self.started
            .is_some_and(|s| now.duration_since(s).as_secs_f32() > 28.0)
    }
}

pub fn resample(x: &[f32], from: u32, to: u32) -> Vec<f32> {
    if from == to || x.is_empty() {
        return x.to_vec();
    }
    let ratio = from as f64 / to as f64;
    let n = (x.len() as f64 / ratio) as usize;
    // windowed average (cheap low-pass) + linear interpolation
    let k = ratio.ceil() as usize;
    (0..n)
        .map(|i| {
            let pos = i as f64 * ratio;
            let i0 = pos as usize;
            if k > 1 {
                let a = i0.saturating_sub(k / 2);
                let b = (i0 + k / 2 + 1).min(x.len());
                x[a..b].iter().sum::<f32>() / (b - a) as f32
            } else {
                let f = (pos - i0 as f64) as f32;
                let a = x[i0.min(x.len() - 1)];
                let b = x[(i0 + 1).min(x.len() - 1)];
                a + (b - a) * f
            }
        })
        .collect()
}

// ------------------------------------------------------------------------------------------------
// Whisper worker
// ------------------------------------------------------------------------------------------------

pub fn model_dir(name: &str) -> PathBuf {
    dirs::cache_dir()
        .unwrap_or_else(std::env::temp_dir)
        .join("rust_blender")
        .join("whisper")
        .join(name)
}

fn worker(rx: Receiver<VoiceRequest>, tx: Sender<VoiceEvent>, repaint: impl Fn()) {
    #[cfg(feature = "voice")]
    let mut model: Option<whisper::Model> = None;
    for req in rx {
        match req {
            VoiceRequest::Load(name) => {
                #[cfg(feature = "voice")]
                {
                    let _ = tx.send(VoiceEvent::Status(format!("Loading Whisper {name}…")));
                    repaint();
                    match whisper::Model::load(&name, &tx, &repaint) {
                        Ok(m) => {
                            model = Some(m);
                            let _ = tx.send(VoiceEvent::Ready(name));
                        }
                        Err(e) => {
                            let _ = tx.send(VoiceEvent::Error(format!("Whisper load failed: {e}")));
                        }
                    }
                }
                #[cfg(not(feature = "voice"))]
                {
                    let _ = (name, &tx);
                }
            }
            VoiceRequest::Transcribe(pcm) => {
                #[cfg(feature = "voice")]
                {
                    let t0 = std::time::Instant::now();
                    match model.as_mut() {
                        None => {
                            let _ =
                                tx.send(VoiceEvent::Error("Whisper model not loaded yet".into()));
                        }
                        Some(m) => match m.transcribe(&pcm) {
                            Ok(text) => {
                                let _ = tx.send(VoiceEvent::Transcript {
                                    text,
                                    secs: t0.elapsed().as_secs_f32(),
                                });
                            }
                            Err(e) => {
                                let _ = tx
                                    .send(VoiceEvent::Error(format!("Transcription failed: {e}")));
                            }
                        },
                    }
                }
                #[cfg(not(feature = "voice"))]
                {
                    let _ = pcm;
                }
            }
        }
        repaint();
    }
}

#[cfg(feature = "voice")]
mod whisper {
    use super::*;
    use candle_core::{D, DType, Device, IndexOp, Tensor};
    use candle_nn::VarBuilder;
    use candle_transformers::models::whisper::{self as w, Config, model::Whisper};
    use std::collections::HashMap;
    use std::io::{Read, Write};

    pub struct Model {
        model: Whisper,
        config: Config,
        mel_filters: Vec<f32>,
        tok: Tokenizer,
        device: Device,
    }

    fn download(
        repo: &str,
        file: &str,
        dest: &std::path::Path,
        tx: &Sender<VoiceEvent>,
        repaint: &dyn Fn(),
    ) -> Result<(), String> {
        if dest.exists()
            && std::fs::metadata(dest)
                .map(|m| m.len() > 0)
                .unwrap_or(false)
        {
            return Ok(());
        }
        std::fs::create_dir_all(dest.parent().unwrap()).map_err(|e| e.to_string())?;
        let agent: ureq::Agent = ureq::Agent::config_builder()
            .http_status_as_error(false)
            .build()
            .into();
        let mut last_err = String::new();
        for rev in ["main", "refs%2Fpr%2F15", "refs%2Fpr%2F13", "refs%2Fpr%2F10"] {
            let url = format!("https://huggingface.co/{repo}/resolve/{rev}/{file}");
            let resp = match agent.get(&url).call() {
                Ok(r) => r,
                Err(e) => {
                    last_err = e.to_string();
                    continue;
                }
            };
            if !resp.status().is_success() {
                last_err = format!("HTTP {} for {url}", resp.status());
                continue;
            }
            let total: f32 = resp
                .headers()
                .get("content-length")
                .and_then(|v| v.to_str().ok())
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.0);
            let mut reader = resp
                .into_body()
                .into_with_config()
                .limit(4_000_000_000)
                .reader();
            let tmp = dest.with_extension("part");
            let mut f = std::fs::File::create(&tmp).map_err(|e| e.to_string())?;
            let mut buf = vec![0u8; 1 << 16];
            let mut done = 0f32;
            let mut last = std::time::Instant::now();
            loop {
                let n = reader.read(&mut buf).map_err(|e| e.to_string())?;
                if n == 0 {
                    break;
                }
                f.write_all(&buf[..n]).map_err(|e| e.to_string())?;
                done += n as f32;
                if last.elapsed().as_millis() > 150 {
                    last = std::time::Instant::now();
                    let _ = tx.send(VoiceEvent::Download(
                        file.to_string(),
                        if total > 0.0 { done / total } else { -1.0 },
                    ));
                    repaint();
                }
            }
            drop(f);
            std::fs::rename(&tmp, dest).map_err(|e| e.to_string())?;
            return Ok(());
        }
        Err(format!(
            "could not download {file} from {repo}: {last_err}. You can also download it manually into {}",
            dest.parent()
                .map(|p| p.display().to_string())
                .unwrap_or_default()
        ))
    }

    impl Model {
        pub fn load(
            name: &str,
            tx: &Sender<VoiceEvent>,
            repaint: &dyn Fn(),
        ) -> Result<Self, String> {
            let repo = WHISPER_MODELS
                .iter()
                .find(|(n, _)| *n == name)
                .map(|(_, r)| *r)
                .unwrap_or("openai/whisper-base.en");
            let dir = model_dir(name);
            for f in ["config.json", "tokenizer.json", "model.safetensors"] {
                download(repo, f, &dir.join(f), tx, repaint)?;
            }
            let config: Config = serde_json::from_str(
                &std::fs::read_to_string(dir.join("config.json")).map_err(|e| e.to_string())?,
            )
            .map_err(|e| e.to_string())?;
            let tok = Tokenizer::load(&dir.join("tokenizer.json"))?;
            let device = Device::Cpu;
            let vb = unsafe {
                VarBuilder::from_mmaped_safetensors(
                    &[dir.join("model.safetensors")],
                    w::DTYPE,
                    &device,
                )
                .map_err(|e| e.to_string())?
            };
            let model = Whisper::load(&vb, config.clone()).map_err(|e| e.to_string())?;
            let mel_filters = mel_filterbank(config.num_mel_bins, w::N_FFT, w::SAMPLE_RATE as f32);
            Ok(Self {
                model,
                config,
                mel_filters,
                tok,
                device,
            })
        }

        pub fn transcribe(&mut self, pcm: &[f32]) -> Result<String, String> {
            self.run(pcm).map_err(|e| e.to_string())
        }

        fn run(&mut self, pcm: &[f32]) -> candle_core::Result<String> {
            let mut pcm = pcm.to_vec();
            // normalise level
            let peak = pcm.iter().fold(0f32, |m, x| m.max(x.abs()));
            if peak > 1e-4 {
                for x in &mut pcm {
                    *x = *x / peak * 0.9;
                }
            }
            pcm.truncate(w::N_SAMPLES);
            if pcm.len() < w::SAMPLE_RATE {
                pcm.resize(w::SAMPLE_RATE, 0.0);
            }
            let mel = w::audio::pcm_to_mel(&self.config, &pcm, &self.mel_filters);
            let n_mels = self.config.num_mel_bins;
            let frames = mel.len() / n_mels;
            let mel = Tensor::from_vec(mel, (1, n_mels, frames), &self.device)?;
            let mel = mel.narrow(2, 0, frames.min(w::N_FRAMES))?;
            self.model.reset_kv_cache();
            let audio = self.model.encoder.forward(&mel, true)?;

            let sot = self.tok.id(w::SOT_TOKEN)?;
            let transcribe = self.tok.id(w::TRANSCRIBE_TOKEN)?;
            let no_ts = self.tok.id(w::NO_TIMESTAMPS_TOKEN)?;
            let eot = self.tok.id(w::EOT_TOKEN)?;
            let mut tokens = vec![sot];
            if self.tok.multilingual {
                if let Ok(en) = self.tok.id("<|en|>") {
                    tokens.push(en);
                }
            }
            tokens.push(transcribe);
            tokens.push(no_ts);
            let prompt_len = tokens.len();

            let vocab = self.config.vocab_size;
            let mut sup = vec![0f32; vocab];
            for &t in &self.config.suppress_tokens {
                if (t as usize) < vocab {
                    sup[t as usize] = f32::NEG_INFINITY;
                }
            }
            sup[no_ts as usize] = f32::NEG_INFINITY;
            // never emit timestamp tokens
            for (i, s) in sup.iter_mut().enumerate() {
                if i as u32 > no_ts {
                    *s = f32::NEG_INFINITY;
                }
            }
            let sup = Tensor::new(sup.as_slice(), &self.device)?;

            let max_len = self.config.max_target_positions / 2;
            for i in 0..max_len {
                let t = Tensor::new(tokens.as_slice(), &self.device)?.unsqueeze(0)?;
                let ys = self.model.decoder.forward(&t, &audio, i == 0)?;
                let (_, seq, _) = ys.dims3()?;
                let logits = self
                    .model
                    .decoder
                    .final_linear(&ys.i((..1, seq - 1..))?)?
                    .i(0)?
                    .i(0)?;
                let logits = logits.broadcast_add(&sup)?;
                let next = logits.argmax(D::Minus1)?.to_scalar::<u32>()?;
                if next == eot {
                    break;
                }
                tokens.push(next);
                // repetition guard
                if tokens.len() > prompt_len + 8 {
                    let n = tokens.len();
                    if tokens[n - 4..] == tokens[n - 8..n - 4] {
                        break;
                    }
                }
            }
            let _ = DType::F32;
            Ok(self.tok.decode(&tokens[prompt_len..]).trim().to_string())
        }
    }

    /// Minimal byte-level BPE decoder for the Whisper tokenizer (avoids the heavy `tokenizers` crate).
    struct Tokenizer {
        id_to_tok: HashMap<u32, String>,
        tok_to_id: HashMap<String, u32>,
        special_start: u32,
        byte_dec: HashMap<char, u8>,
        multilingual: bool,
    }

    impl Tokenizer {
        fn load(path: &std::path::Path) -> Result<Self, String> {
            let v: serde_json::Value =
                serde_json::from_str(&std::fs::read_to_string(path).map_err(|e| e.to_string())?)
                    .map_err(|e| e.to_string())?;
            let mut tok_to_id: HashMap<String, u32> = HashMap::new();
            if let Some(vocab) = v["model"]["vocab"].as_object() {
                for (k, id) in vocab {
                    tok_to_id.insert(k.clone(), id.as_u64().unwrap_or(0) as u32);
                }
            }
            let mut special_start = u32::MAX;
            if let Some(added) = v["added_tokens"].as_array() {
                for t in added {
                    let id = t["id"].as_u64().unwrap_or(0) as u32;
                    let c = t["content"].as_str().unwrap_or("").to_string();
                    if t["special"].as_bool().unwrap_or(false) || c.starts_with("<|") {
                        special_start = special_start.min(id);
                    }
                    tok_to_id.insert(c, id);
                }
            }
            let id_to_tok = tok_to_id.iter().map(|(k, v)| (*v, k.clone())).collect();
            let multilingual = tok_to_id.contains_key("<|fr|>");
            Ok(Self {
                id_to_tok,
                tok_to_id,
                special_start,
                byte_dec: byte_decoder(),
                multilingual,
            })
        }
        fn id(&self, t: &str) -> candle_core::Result<u32> {
            self.tok_to_id
                .get(t)
                .copied()
                .ok_or_else(|| candle_core::Error::Msg(format!("token {t} missing")))
        }
        fn decode(&self, ids: &[u32]) -> String {
            let mut bytes = vec![];
            for id in ids {
                if *id >= self.special_start {
                    continue;
                }
                if let Some(s) = self.id_to_tok.get(id) {
                    for ch in s.chars() {
                        if let Some(b) = self.byte_dec.get(&ch) {
                            bytes.push(*b);
                        }
                    }
                }
            }
            String::from_utf8_lossy(&bytes).into_owned()
        }
    }

    fn byte_decoder() -> HashMap<char, u8> {
        let mut bs: Vec<u32> = (b'!' as u32..=b'~' as u32)
            .chain(0xA1..=0xAC)
            .chain(0xAE..=0xFF)
            .collect();
        let mut cs = bs.clone();
        let mut n = 0;
        for b in 0..256u32 {
            if !bs.contains(&b) {
                bs.push(b);
                cs.push(256 + n);
                n += 1;
            }
        }
        bs.iter()
            .zip(cs.iter())
            .map(|(b, c)| (char::from_u32(*c).unwrap(), *b as u8))
            .collect()
    }

    /// Slaney-style mel filterbank, identical to librosa.filters.mel(sr, n_fft, n_mels) used by Whisper.
    pub fn mel_filterbank(n_mels: usize, n_fft: usize, sr: f32) -> Vec<f32> {
        let n_bins = n_fft / 2 + 1;
        let f_sp = 200.0 / 3.0;
        let min_log_hz = 1000.0f32;
        let min_log_mel = min_log_hz / f_sp;
        let logstep = (6.4f32).ln() / 27.0;
        let hz_to_mel = |f: f32| {
            if f >= min_log_hz {
                min_log_mel + (f / min_log_hz).ln() / logstep
            } else {
                f / f_sp
            }
        };
        let mel_to_hz = |m: f32| {
            if m >= min_log_mel {
                min_log_hz * (logstep * (m - min_log_mel)).exp()
            } else {
                f_sp * m
            }
        };
        let fmax = sr / 2.0;
        let (mmin, mmax) = (hz_to_mel(0.0), hz_to_mel(fmax));
        let mel_f: Vec<f32> = (0..n_mels + 2)
            .map(|i| mel_to_hz(mmin + (mmax - mmin) * i as f32 / (n_mels + 1) as f32))
            .collect();
        let fft_f: Vec<f32> = (0..n_bins)
            .map(|i| fmax * i as f32 / (n_bins - 1) as f32)
            .collect();
        let mut out = vec![0f32; n_mels * n_bins];
        for m in 0..n_mels {
            let enorm = 2.0 / (mel_f[m + 2] - mel_f[m]);
            for (k, &f) in fft_f.iter().enumerate() {
                let lower = (f - mel_f[m]) / (mel_f[m + 1] - mel_f[m]);
                let upper = (mel_f[m + 2] - f) / (mel_f[m + 2] - mel_f[m + 1]);
                out[m * n_bins + k] = lower.min(upper).max(0.0) * enorm;
            }
        }
        out
    }

    #[cfg(test)]
    mod tests {
        #[test]
        fn mel_bank_shape() {
            let f = super::mel_filterbank(80, 400, 16000.0);
            assert_eq!(f.len(), 80 * 201);
            // first filter peaks at bin 1 with librosa value ~0.02486
            assert!((f[1] - 0.024_86).abs() < 1e-3, "{}", f[1]);
            assert!(f.iter().all(|x| *x >= 0.0));
        }
        #[test]
        fn bytes() {
            let d = super::byte_decoder();
            assert_eq!(d.len(), 256);
            assert_eq!(d[&'Ġ'], b' ');
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn resample_len() {
        let x = vec![0.5f32; 48000];
        let y = super::resample(&x, 48000, 16000);
        assert_eq!(y.len(), 16000);
        assert!((y[100] - 0.5).abs() < 1e-5);
    }
}
