# Rust Blender

A Blender-style 3D editor written in Rust. You can drive it with the mouse and keyboard like Blender, by typing plain English, or by speaking. **Everything runs on your machine.** An **Ollama** language model plans the edits and **OpenAI Whisper**, running in pure Rust through `candle`, turns your speech into text.

![layout](docs/layout.png)

## Features

| Area | What you get |
|---|---|
| **Viewport** | A multithreaded software rasterizer with a real depth buffer. It outlines selections the way Blender does (taken from the object-ID buffer), picks objects pixel-exactly, and has Wireframe, Solid (studio light) and Material Preview shading (sun, sky, Fresnel, metallic and roughness). It also has X-ray, a fading floor grid, a 3D cursor and a navigation gizmo you can click or drag. |
| **Editing** | Cube, plane, UV sphere, ico sphere, cylinder, cone, torus and a "blob" head. You can transform with modal **G / R / S** like Blender: constrain to an axis with X, Y or Z, type an exact number, or hold Ctrl to snap. There are also Move, Rotate and Scale gizmos, box select, duplicate, hide/reveal, unlimited undo/redo, and Array, Mirror and Radial-Array modifiers. |
| **Animation** | Keyframes with eased interpolation, a timeline with keyframe diamonds, auto-keying and playback. |
| **Panels** | Outliner (rename with a double-click, visibility toggles, context menu), Properties (Object, Modifiers, Material, World and Scene tabs), F3 command search and a Shift+A add menu. |
| **AI Copilot** | Chat panel with structured JSON-schema output from Ollama. Each AI batch is a single undo step. Simple commands run **instantly without the LLM**. The app auto-picks the best installed model and can download models from inside the app. |
| **Voice** | Hands-free mode stops recording automatically when you pause. It shows a live level meter, uses local Whisper (tiny, base, small or distil) and can optionally read replies aloud. |
| **Files** | Save and open `.rblend` (JSON) scenes and export Wavefront `.obj`. |

## Recommended Ollama model

The default is **`qwen3:8b`**. Qwen3 follows JSON-schema "structured outputs" and tool-style instructions reliably and is good at the spatial arithmetic that scene building needs (for example, "a ring of 12 cubes"). The 8B model runs comfortably on a GPU with about 6–8 GB of VRAM.

| Your hardware | Model |
|---|---|
| 8 GB+ GPU (recommended) | `qwen3:8b` |
| Low VRAM / CPU only | `qwen3:4b` |
| 16 GB+ GPU, best quality | `qwen3:14b` |

The app ranks whatever you have installed and picks the best fit: qwen3 first, then qwen2.5, llama3.1, mistral-nemo, gemma3 and so on. You can always override the choice in **Copilot ⚙ → Model**.

## Setup (Windows)

1. Install **Rust**: <https://rustup.rs>. You also need the *MSVC C++ Build Tools*; rustup offers to install them.
2. Install **Ollama**: <https://ollama.com/download>, then run:
   ```powershell
   ollama pull qwen3:8b
   ```
3. Build and run (use `--release`, because the software renderer and Whisper are much faster that way):
   ```powershell
   cd Rust_Blender
   cargo run --release
   ```
   Or double-click `run.bat`.

The first time you click the 🎤 mic, the Whisper model is downloaded once from Hugging Face (about 145 MB for `base.en`). It is cached in `%LOCALAPPDATA%\rust_blender\whisper\<model>`. If that download is blocked on your network, put `config.json`, `tokenizer.json` and `model.safetensors` from `huggingface.co/openai/whisper-base.en` into that folder yourself.

To build without voice support (smaller and faster to compile), run `cargo run --release --no-default-features`.

## Talking to it

Click the mic (or press **Ctrl+Space** or the **`** key), speak, and pause. Examples:

* "add a blue sphere", "move it up two", "rotate ninety degrees on x", "make it bigger", "top view", "undo". These go through the instant command parser.
* "build a snowman", "make a solar system with orbiting planets", "create a table with four legs", "put a ring of 8 glowing cubes around the torus", "animate the cube spinning 360 degrees over 60 frames". These go to the LLM.

Follow-ups work because the AI sees the current scene and the conversation history: "make the second one red", "now hide the hat".

## Keyboard (Blender-compatible)

| Keys | Action |
|---|---|
| LMB / Shift+LMB / drag | select / extend / box select |
| MMB (Alt+LMB), Shift+MMB, wheel | orbit, pan, zoom |
| G R S then X/Y/Z, digits, Ctrl | grab / rotate / scale, constrain, exact value, snap |
| Shift+A, Shift+D, X, H / Alt+H | add, duplicate, delete, hide / reveal |
| 1 3 7 (Ctrl), 5, 0, F, Home | views, ortho toggle, user view, frame selected / all |
| Z, Alt+Z, W | cycle shading, X-ray, cycle tool |
| I, Space, ← → | keyframe, play, step |
| F3 / Ctrl+K | command search (falls back to the AI) |
| / or Ctrl+L, N | focus the AI prompt, toggle the Copilot panel |
| Ctrl+Z / Ctrl+Shift+Z, Ctrl+S / O / N | undo / redo, save / open / new |

## Architecture

```
src/
  main.rs      eframe bootstrap
  app.rs       layout, menus, Copilot chat, voice UX, outliner, properties, timeline, palette
  viewport.rs  camera, scene → primitives, navigation, modal transforms, gizmos, HUD
  render.rs    multithreaded z-buffer rasterizer, ID buffer, selection outline
  editor.rs    editor state, undo, and the command executor shared by UI / AI / voice
  ai.rs        Ollama client, model ranking, JSON schema + system prompt, offline parser
  voice.rs     cpal mic capture, VAD, resampling, candle Whisper + tiny BPE decoder
  scene.rs     objects, transforms, materials, modifiers, keyframes, OBJ export
  mesh.rs      procedural primitives with welded vertices, normals, edge adjacency
  theme.rs     Blender-like dark theme
```

The AI never touches the scene directly. It produces a list of flat `Command` records, for example `{"action":"add","primitive":"cone","location":[0,0,2],"color":"orange"}`, which `Editor::execute` validates and applies. That keeps the AI's actions safe, undoable and visible in the chat log.

Run the tests with `cargo test`.
