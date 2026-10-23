@echo off
rem Build (first run takes a few minutes) and launch Rust Blender
where ollama >/dev/null 2>/dev/null || echo [warn] Ollama not found - install from https://ollama.com/download then run: ollama pull qwen3:8b
cargo run --release
pause
