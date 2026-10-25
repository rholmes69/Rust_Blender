//! Rendering the timeline: one PNG per frame (either engine), optionally joined into an MP4 with
//! ffmpeg when it is installed. Frames render in the background so the app stays usable.

use crossbeam_channel::{Receiver, unbounded};
use std::path::{Path, PathBuf};
use std::time::Instant;

pub struct FrameDone {
    pub frame: i32,
    pub width: usize,
    pub height: usize,
    pub rgba: Vec<u8>,
    pub result: Result<PathBuf, String>,
}

pub struct AnimJob {
    pub dir: PathBuf,
    pub start: i32,
    pub end: i32,
    pub next: i32,
    pub path_traced: bool,
    pub samples: u32,
    pub denoise: bool,
    pub width: usize,
    pub height: usize,
    pub transparent: bool,
    pub video: bool,
    pub fps: u32,
    pub started: Instant,
    pub done: usize,
    /// frame that was showing before the render (restored at the end)
    pub restore_frame: i32,
    pub worker: Option<Receiver<FrameDone>>,
    pub encoder: Option<Receiver<Result<PathBuf, String>>>,
    pub error: Option<String>,
    pub cancelled: bool,
}

impl AnimJob {
    pub fn total(&self) -> usize {
        (self.end - self.start + 1).max(0) as usize
    }
    pub fn eta_secs(&self) -> f32 {
        if self.done == 0 {
            return 0.0;
        }
        let per = self.started.elapsed().as_secs_f32() / self.done as f32;
        per * (self.total() - self.done) as f32
    }
}

pub fn frame_path(dir: &Path, frame: i32) -> PathBuf {
    dir.join(format!("frame_{frame:04}.png"))
}

/// Encode + write one frame on a worker thread (PNG compression is the slow part).
pub fn write_frame_async(dir: PathBuf, frame: i32, width: usize, height: usize, rgba: Vec<u8>) -> Receiver<FrameDone> {
    let (tx, rx) = unbounded();
    std::thread::spawn(move || {
        let p = frame_path(&dir, frame);
        let result = crate::textures::encode_png(width, height, &rgba).and_then(|b| std::fs::write(&p, b).map_err(|e| format!("{}: {e}", p.display()))).map(|_| p);
        let _ = tx.send(FrameDone { frame, width, height, rgba, result });
    });
    rx
}

/// Path trace one frame on a worker thread and write it.
pub fn path_trace_async(sc: crate::pathtrace::PtScene, dir: PathBuf, frame: i32, width: usize, height: usize, samples: u32, denoise: bool) -> Receiver<FrameDone> {
    let (tx, rx) = unbounded();
    std::thread::spawn(move || {
        let rgba = crate::pathtrace::render_blocking(&sc, width, height, samples, denoise);
        let p = frame_path(&dir, frame);
        let result = crate::textures::encode_png(width, height, &rgba).and_then(|b| std::fs::write(&p, b).map_err(|e| format!("{}: {e}", p.display()))).map(|_| p);
        let _ = tx.send(FrameDone { frame, width, height, rgba, result });
    });
    rx
}

/// An `ffmpeg` command that doesn't flash a console window on Windows.
fn ffmpeg() -> std::process::Command {
    #[allow(unused_mut)]
    let mut c = std::process::Command::new("ffmpeg");
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        c.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    }
    c
}

/// Is `ffmpeg` on the PATH?
pub fn ffmpeg_available() -> bool {
    ffmpeg().arg("-version").stdout(std::process::Stdio::null()).stderr(std::process::Stdio::null()).status().is_ok_and(|s| s.success())
}

/// Join `frame_%04d.png` (starting at `start`) into an H.264 MP4 next to the frames.
pub fn encode_video(dir: &Path, start: i32, fps: u32) -> Result<PathBuf, String> {
    let name = dir.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_else(|| "render".into());
    let out = dir.join(format!("{name}.mp4"));
    let status = ffmpeg()
        .args(["-y", "-loglevel", "error", "-framerate", &fps.max(1).to_string(), "-start_number", &start.to_string(), "-i"])
        .arg(dir.join("frame_%04d.png"))
        // even dimensions for yuv420p
        .args(["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18"])
        .arg(&out)
        .output()
        .map_err(|e| format!("ffmpeg didn't start: {e}"))?;
    if status.status.success() {
        Ok(out)
    } else {
        Err(format!("ffmpeg failed: {}", String::from_utf8_lossy(&status.stderr).trim()))
    }
}

pub fn encode_video_async(dir: PathBuf, start: i32, fps: u32) -> Receiver<Result<PathBuf, String>> {
    let (tx, rx) = unbounded();
    std::thread::spawn(move || {
        let _ = tx.send(encode_video(&dir, start, fps));
    });
    rx
}

/// Default output folder: next to the saved file (`<name>_render`), else in Videos / home.
pub fn default_dir(file: Option<&Path>) -> PathBuf {
    match file {
        Some(f) => {
            let stem = f.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_else(|| "scene".into());
            f.parent().map(|p| p.to_path_buf()).unwrap_or_default().join(format!("{stem}_render"))
        }
        None => dirs::video_dir().or_else(dirs::home_dir).unwrap_or_else(std::env::temp_dir).join("rust_blender_render"),
    }
}
