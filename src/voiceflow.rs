//! Hands-free voice workflow: wake word, voice intents (stop / undo / repeat / goodbye),
//! fixes for common speech-recognition mishearings, short spoken confirmations and a
//! text-to-speech player that can be interrupted.

use crate::editor::Command;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VoiceIntent {
    /// Cancel the running AI request and stop talking.
    Stop,
    /// Undo the last change.
    Undo,
    /// Leave conversation / wake mode.
    Goodbye,
    /// Say the last confirmation again.
    Repeat,
}

fn norm(text: &str) -> String {
    let t: String = text
        .to_lowercase()
        .chars()
        .map(|c| if c.is_alphanumeric() || c == '\'' || c.is_whitespace() { c } else { ' ' })
        .collect();
    t.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Short utterances that control the voice session rather than the scene.
pub fn voice_intent(text: &str) -> Option<VoiceIntent> {
    let t = norm(text);
    let t = t.trim_start_matches("okay ").trim_start_matches("ok ");
    let t = t.trim_start_matches("no ").trim_start_matches("nope ");
    match t {
        "stop" | "cancel" | "never mind" | "nevermind" | "stop that" | "cancel that" | "shut up" | "be quiet" | "quiet" | "hold on" | "wait" => {
            Some(VoiceIntent::Stop)
        }
        "undo" | "undo that" | "undo it" | "scratch that" | "take that back" | "go back" | "revert that" | "that's wrong" | "wrong" => Some(VoiceIntent::Undo),
        "goodbye" | "bye" | "stop listening" | "that's all" | "that is all" | "i'm done" | "we're done" | "thanks that's all" | "thank you that's all" | "go to sleep" => {
            Some(VoiceIntent::Goodbye)
        }
        "repeat" | "repeat that" | "say that again" | "what did you do" | "what did you just do" | "what happened" => Some(VoiceIntent::Repeat),
        _ if t == "no" || t == "nope" => Some(VoiceIntent::Undo),
        _ => None,
    }
}

/// If `text` starts with the wake phrase ("blender", "hey blender", …), return what follows it.
/// Also accepts a few ways Whisper commonly mishears "Blender".
pub fn strip_wake(text: &str, wake: &str) -> Option<String> {
    let wake = norm(wake);
    if wake.is_empty() {
        return None;
    }
    let mut variants = vec![wake.clone()];
    if wake == "blender" {
        variants.extend(["blend her", "blenders", "lender", "the blender", "blunder"].map(String::from));
    }
    let original_words: Vec<&str> = text.split_whitespace().collect();
    let t = norm(text);
    for prefix in ["hey ", "ok ", "okay ", "hi ", "yo ", ""] {
        for v in &variants {
            let p = format!("{prefix}{v}");
            if t == p || t.starts_with(&format!("{p} ")) {
                // keep the user's original casing/punctuation for the rest
                let skip = p.split_whitespace().count();
                let rest = original_words.get(skip..).unwrap_or(&[]).join(" ");
                return Some(rest.trim_start_matches([',', '.', '!', '?', ':', ' ']).trim().to_string());
            }
        }
    }
    None
}

/// Fix frequent speech-recognition mistakes for 3D vocabulary.
pub fn fix_mishearings(text: &str) -> String {
    const FIXES: &[(&str, &str)] = &[
        ("queue", "cube"),
        ("cue", "cube"),
        ("cubed", "cube"),
        ("taurus", "torus"),
        ("torres", "torus"),
        ("tourist", "torus"),
        ("spear", "sphere"),
        ("sphear", "sphere"),
        ("cylindar", "cylinder"),
        ("in set", "inset"),
        ("key frame", "keyframe"),
        ("key-frame", "keyframe"),
        ("loopcut", "loop cut"),
        ("wire frame", "wireframe"),
        ("you v sphere", "uv sphere"),
        ("uv-sphere", "uv sphere"),
        ("ico-sphere", "ico sphere"),
        ("icosphere", "ico sphere"),
        ("extrude it", "extrude"),
        ("monkey head", "monkey"),
        ("susan", "monkey"),
        ("suzanne", "monkey"),
    ];
    let toks: Vec<&str> = text.split_whitespace().collect();
    let core = |w: &str| w.chars().filter(|c| c.is_alphanumeric() || *c == '\'' || *c == '-').collect::<String>().to_lowercase();
    let mut out: Vec<String> = vec![];
    let mut i = 0;
    'outer: while i < toks.len() {
        // longest phrase first
        let mut fixes: Vec<&(&str, &str)> = FIXES.iter().collect();
        fixes.sort_by_key(|(f, _)| std::cmp::Reverse(f.split(' ').count()));
        for (from, to) in fixes {
            let fw: Vec<&str> = from.split(' ').collect();
            if i + fw.len() <= toks.len() && fw.iter().enumerate().all(|(k, w)| core(toks[i + k]) == *w) {
                let last = toks[i + fw.len() - 1];
                let trail: String = last.chars().rev().take_while(|c| !c.is_alphanumeric()).collect::<String>().chars().rev().collect();
                out.push(format!("{to}{trail}"));
                i += fw.len();
                continue 'outer;
            }
        }
        out.push(toks[i].to_string());
        i += 1;
    }
    out.join(" ")
}

fn plural(n: usize, one: &str, many: &str) -> String {
    if n == 1 { format!("1 {one}") } else { format!("{n} {many}") }
}

/// Turn one log line ("✔ Added UV Sphere \"Moon\"") into speakable text ("added UV Sphere Moon").
fn speakable(line: &str) -> String {
    let l = line.trim_start_matches(['✔', '✖', ' ']);
    // drop numeric details: "Moved "Cube" by [0.00, 0.00, 2.00]" → "Moved Cube"
    let l = l.split(" by [").next().unwrap_or(l);
    let l = l.split(" to [").next().unwrap_or(l);
    let l = l.split(" at [").next().unwrap_or(l);
    strip_suffixes(&l.replace('"', "")).trim().to_string()
}

/// "Cube.001" → "Cube" (Blender-style duplicate suffixes read badly aloud).
fn strip_suffixes(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = String::with_capacity(s.len());
    let mut i = 0;
    while i < s.len() {
        if b[i] == b'.' && i + 4 <= s.len() && b[i + 1..i + 4].iter().all(u8::is_ascii_digit) && (i + 4 == s.len() || !b[i + 4].is_ascii_digit()) {
            i += 4;
            continue;
        }
        let ch = s[i..].chars().next().unwrap();
        out.push(ch);
        i += ch.len_utf8();
    }
    out
}

/// "Added UV Sphere \"Ball\"" + colour "red" → "Added a red UV sphere called Ball".
fn added_phrase(line: &str, color: Option<&str>) -> String {
    let l = line.trim_start_matches(['✔', ' ']);
    let Some(rest) = l.strip_prefix("Added ") else { return speakable(line) };
    let (kind, name) = match rest.find(" \"") {
        Some(q) => (&rest[..q], rest[q + 2..].trim_end_matches('"')),
        None => (rest, ""),
    };
    let base = strip_suffixes(name);
    let kind_l = if kind.starts_with("UV") { format!("UV {}", kind[3..].to_lowercase()) } else { kind.to_lowercase() };
    let desc = match color.filter(|c| !c.starts_with('#') && c.len() < 20) {
        Some(c) => format!("{c} {kind_l}"),
        None => kind_l.clone(),
    };
    let article = if desc.starts_with(['a', 'e', 'i', 'o', 'u']) { "an" } else { "a" };
    let named = !base.is_empty() && !base.to_lowercase().starts_with(&kind.to_lowercase()) && !kind.to_lowercase().starts_with(&base.to_lowercase());
    if named { format!("Added {article} {desc} called {base}") } else { format!("Added {article} {desc}") }
}

/// One short sentence describing what a voice request did, for text-to-speech.
pub fn spoken_summary(reply: &str, commands: &[Command], log: &[String], fixes: usize) -> String {
    let ok: Vec<&String> = log.iter().filter(|l| l.starts_with('✔')).collect();
    let failed = log.iter().filter(|l| l.starts_with('✖')).count();
    let mut parts: Vec<String> = vec![];
    if commands.is_empty() && log.is_empty() {
        // an answer: speak its first one or two sentences
        let mut s = String::new();
        for sent in reply.split_inclusive(['.', '!', '?']) {
            if s.len() + sent.len() > 220 && !s.is_empty() {
                break;
            }
            s.push_str(sent);
            if s.matches(['.', '!', '?']).count() >= 2 {
                break;
            }
        }
        let s = s.trim();
        return if s.is_empty() { "Done.".into() } else { s.chars().take(260).collect() };
    }
    let adds = ok.iter().filter(|l| l.contains("Added ")).count();
    let others: Vec<&&String> = ok.iter().filter(|l| !l.contains("Added ") && !l.contains("Selected ")).collect();
    if adds == 1 {
        let color = commands.iter().find(|c| c.action == "add").and_then(|c| c.color.as_ref()).and_then(|v| v.as_str());
        parts.push(added_phrase(ok.iter().find(|l| l.contains("Added ")).unwrap(), color));
    } else if adds > 1 {
        parts.push(format!("Added {}", plural(adds, "object", "objects")));
    }
    if others.len() <= 2 {
        parts.extend(others.iter().map(|l| speakable(l)));
    } else {
        parts.push(format!("made {}", plural(others.len(), "change", "changes")));
    }
    if parts.is_empty() && !ok.is_empty() {
        parts.push(speakable(ok[0]));
    }
    let mut s = match parts.len() {
        0 => String::new(),
        1 => parts[0].clone(),
        _ => {
            let last = parts.pop().unwrap();
            format!("{} and {}", parts.join(", "), lowercase_first(&last))
        }
    };
    if fixes > 0 {
        s += &format!(", then fixed {}", plural(fixes, "problem", "problems"));
    }
    if failed > 0 {
        s = if s.is_empty() {
            format!("Sorry, {} failed", plural(failed, "command", "commands"))
        } else {
            format!("{s}, but {} failed", plural(failed, "command", "commands"))
        };
    }
    if s.is_empty() {
        s = "Done".into();
    }
    let mut s = upper_first(&s);
    if !s.ends_with(['.', '!', '?']) {
        s.push('.');
    }
    s
}

fn upper_first(s: &str) -> String {
    let mut c = s.chars();
    c.next().map(|f| f.to_uppercase().collect::<String>() + c.as_str()).unwrap_or_default()
}
fn lowercase_first(s: &str) -> String {
    // keep "UV Sphere" etc. intact; only lowercase a leading verb
    let mut it = s.splitn(2, ' ');
    let first = it.next().unwrap_or("");
    let rest = it.next().map(|r| format!(" {r}")).unwrap_or_default();
    if first.chars().skip(1).all(|c| c.is_lowercase()) { format!("{}{rest}", first.to_lowercase()) } else { s.to_string() }
}

// ------------------------------------------------------------------------------------------------
// Text to speech
// ------------------------------------------------------------------------------------------------

/// Speaks one utterance at a time with the OS voice; can be interrupted.
#[derive(Default)]
pub struct Speaker {
    child: Option<std::process::Child>,
    pub last: String,
}

impl Speaker {
    pub fn speak(&mut self, text: &str) {
        self.stop();
        let text: String = text.replace(['"', '\'', '`', '$', '\n'], " ").chars().take(400).collect();
        if text.trim().is_empty() {
            return;
        }
        self.last = text.clone();
        self.child = spawn_tts(&text);
    }

    pub fn stop(&mut self) {
        if let Some(mut c) = self.child.take() {
            let _ = c.kill();
            let _ = c.wait();
        }
    }

    pub fn busy(&mut self) -> bool {
        match &mut self.child {
            Some(c) => match c.try_wait() {
                Ok(None) => true,
                _ => {
                    self.child = None;
                    false
                }
            },
            None => false,
        }
    }
}

impl Drop for Speaker {
    fn drop(&mut self) {
        self.stop();
    }
}

fn spawn_tts(text: &str) -> Option<std::process::Child> {
    use std::process::{Command as P, Stdio};
    #[cfg(target_os = "windows")]
    {
        use std::os::windows::process::CommandExt;
        let script = format!(
            "Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; $s.Rate = 1; $s.Speak(\"{text}\")"
        );
        P::new("powershell").args(["-NoProfile", "-NonInteractive", "-Command", &script]).creation_flags(0x08000000).stdout(Stdio::null()).stderr(Stdio::null()).spawn().ok()
    }
    #[cfg(target_os = "macos")]
    {
        P::new("say").arg(text).stdout(Stdio::null()).stderr(Stdio::null()).spawn().ok()
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        P::new("espeak").arg(text).stdout(Stdio::null()).stderr(Stdio::null()).spawn().or_else(|_| P::new("spd-say").args(["-w", text]).stdout(Stdio::null()).stderr(Stdio::null()).spawn()).ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intents() {
        assert_eq!(voice_intent("Stop."), Some(VoiceIntent::Stop));
        assert_eq!(voice_intent("No, undo that!"), Some(VoiceIntent::Undo));
        assert_eq!(voice_intent("no"), Some(VoiceIntent::Undo));
        assert_eq!(voice_intent("Okay, that's all."), Some(VoiceIntent::Goodbye));
        assert_eq!(voice_intent("What did you do?"), Some(VoiceIntent::Repeat));
        assert_eq!(voice_intent("stop the animation at frame 20"), None);
        assert_eq!(voice_intent("add a cube"), None);
    }

    #[test]
    fn wake_word() {
        assert_eq!(strip_wake("Blender, add a red cube.", "blender").as_deref(), Some("add a red cube."));
        assert_eq!(strip_wake("Hey Blender add a cube", "blender").as_deref(), Some("add a cube"));
        assert_eq!(strip_wake("blend her, make it bigger", "blender").as_deref(), Some("make it bigger"));
        assert_eq!(strip_wake("Blender.", "blender").as_deref(), Some(""));
        assert_eq!(strip_wake("I like my blender", "blender"), None);
        assert_eq!(strip_wake("add a cube", "blender"), None);
        assert_eq!(strip_wake("Computer, add a cube", "computer").as_deref(), Some("add a cube"));
    }

    #[test]
    fn mishearings() {
        assert_eq!(fix_mishearings("add a red queue"), "add a red cube");
        assert_eq!(fix_mishearings("Add a Taurus."), "Add a torus.");
        assert_eq!(fix_mishearings("in set the top face"), "inset the top face");
        assert_eq!(fix_mishearings("insert a key frame"), "insert a keyframe");
        assert_eq!(fix_mishearings("rescue the cat"), "rescue the cat");
    }

    #[test]
    fn summaries() {
        let log = vec!["✔ Added UV Sphere \"Ball\"".to_string()];
        let c = vec![Command::new("add")];
        assert_eq!(spoken_summary("Done.", &c, &log, 0), "Added a UV sphere called Ball.");
        let red = vec![Command { action: "add".into(), color: Some(serde_json::json!("red")), ..Default::default() }];
        assert_eq!(spoken_summary("", &red, &["✔ Added Cube \"Cube.001\"".to_string()], 0), "Added a red cube.");
        assert_eq!(spoken_summary("", &c, &["✔ Moved \"Cube.001\" by [0.00, 0.00, 2.00]".to_string()], 0), "Moved Cube.");
        let log = vec!["✔ Added Cube \"A\"".into(), "✔ Added Cube \"B\"".into(), "✔ Added Cone \"C\"".into(), "✔ Moved \"A\" by [0.00, 0.00, 1.00]".into(), "✖ no object matches \"x\"".into()];
        assert_eq!(spoken_summary("", &c, &log, 1), "Added 3 objects and moved A, then fixed 1 problem, but 1 command failed.");
        assert_eq!(spoken_summary("Press E to extrude. Then drag. Then click.", &[], &[], 0), "Press E to extrude. Then drag.");
    }
}
