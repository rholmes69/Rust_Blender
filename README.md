# Rust Blender

A Blender-style 3D editor written in Rust. You can drive it with the mouse and keyboard like Blender, by typing plain English, or by speaking. **Everything runs on your machine.** An **Ollama** language model plans the edits and **OpenAI Whisper**, running in pure Rust through `candle`, turns your speech into text.

![layout](docs/layout.png)

## Features

| Area | What you get |
|---|---|
| **Viewport** | A **GPU renderer** (wgpu) with 4× MSAA, a real depth buffer and selection outlines drawn in a shader. Picking is pixel-exact from an ID buffer that is read back asynchronously. A multithreaded **software renderer** takes over automatically if the GPU path isn't available, and you can toggle it under *Overlays*. Shading modes are Wireframe, Solid (studio light) and Material Preview (scene lights, Fresnel, metallic and roughness). Also X-ray, a fading grid, a 3D cursor and a clickable navigation gizmo. |
| **Object mode** | Cube, plane, UV/ico sphere, cylinder, cone, torus and a "blob" head. Modal **G / R / S** transforms like Blender, with X/Y/Z axis locks, typed numbers and Ctrl to snap. Move, Rotate and Scale gizmos, box select, duplicate, hide/reveal, unlimited undo, and Array, Mirror and Radial-Array modifiers. |
| **Edit mode** (Tab) | Polygon (n-gon) meshes with **vertex / edge / face** selection (1 / 2 / 3), box select and G/R/S on components. **Extrude** (E, constrained to the face normal), **Inset** (I), **Loop Cut** (Ctrl+R, wheel or +/- sets the cut count) with a live preview, **Subdivide**, **Merge** (M), **Delete** (X) and **Flip Normals** (Alt+N). |
| **Lights & cameras** | Point, Sun and Spot lights that light the Material Preview, and camera objects. **Numpad 0** looks through the camera, with a passepartout around the frame; **Ctrl+Alt+0** aligns the camera to the view. A "look at" command aims lights and cameras. The default scene is Blender's: a cube, a light and a camera. |
| **Animation** | Keyframes with eased interpolation, a timeline, auto-keying and playback. |
| **AI Copilot** | **Streams** replies from Ollama: the text appears live, and each command is applied the moment it finishes generating. There's a Stop button, and the whole answer is a single undo step. Simple commands run instantly without the LLM. The app auto-picks the best installed model and can download models from inside the app. The AI can also edit meshes (select faces, extrude, inset, loop cut and so on). |
| **Voice** | Local Whisper (tiny, base, small or distil) with hands-free auto-stop and a level meter. Replies can optionally be read aloud. |
| **Files** | Save and open `.rblend`. Export **glTF 2.0 (.glb)** with meshes (modifiers applied), PBR materials, emissive strength, lights (`KHR_lights_punctual`), cameras and baked animation; the output passes the Khronos glTF-Validator with 0 errors and 0 warnings. Also exports Wavefront `.obj`. |

## Recommended Ollama model

The default is **`qwen3:8b`**. Qwen3 follows JSON-schema "structured outputs" and tool-style instructions reliably and is good at the spatial arithmetic that scene building needs (for example, "a ring of 12 cubes"). The 8B model runs comfortably on a GPU with about 6–8 GB of VRAM.

| Your hardware | Model |
|---|---|
| 8 GB+ GPU (recommended) | `qwen3:8b` |
| Low VRAM / CPU only | `qwen3:4b` |
| 16 GB+ GPU, best quality | `qwen3:14b` |

The app ranks whatever you have installed and picks the best fit: qwen3 first, then qwen2.5, llama3.1, mistral-nemo, gemma3 and so on. You can always override the choice in **Copilot ⚙ → Model**.

## Two-model AI: decision model + builder

Install a small tool-calling **decision model** (for example `tev1`) alongside qwen3. The app picks up anything named `tev1` automatically; to choose one yourself, use Copilot ⚙ → **Decision**. Every request then goes through two stages:

1. The **decision model** reads the request plus the scene and picks one of Ollama's tools:
   * `scene_command` handles simple edits ("put a small blue ball on the cube") **itself**, usually in well under a second;
   * `answer` replies to questions ("how do I extrude?") without touching the scene;
   * `delegate_to_builder` hands creative, multi-object builds to the builder model, together with a short plan.
2. When the request is delegated, the **builder model** (qwen3) streams the full build and uses the plan as guidance.

The chat shows who handled each request ("handled by tev1 · direct" or "tev1 → qwen3:8b · plan: …"). If the decision model doesn't support Ollama tools, the app falls back to JSON output with the same routes. If the decision model fails, qwen handles the request. Simple commands like "undo" or "top view" still skip both models.

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
* "extrude the top of the cube by two", "add a loop cut around the middle", "inset the top face" use edit mode.
* "add a spotlight pointing at the cube", "look through the camera".
* "build a snowman", "make a solar system with orbiting planets", "create a table with four legs", "put a ring of 8 glowing cubes around the torus", "animate the cube spinning 360 degrees over 60 frames". These go to the LLM.

Follow-ups work because the AI sees the current scene and the conversation history: "make the second one red", "now hide the hat".

## Keyboard (Blender-compatible)

| Keys | Action |
|---|---|
| LMB / Shift+LMB / drag | select / extend / box select |
| MMB (Alt+LMB), Shift+MMB, wheel | orbit, pan, zoom |
| G R S then X/Y/Z, digits, Ctrl | grab / rotate / scale, constrain, exact value, snap |
| Shift+A, Shift+D, X, H / Alt+H | add, duplicate, delete, hide / reveal |
| 1 3 7 (Ctrl), 5, F, Home | views, ortho toggle, frame selected / all |
| Z, Alt+Z, W | cycle shading, X-ray, cycle tool |
| I, Space, ← → | keyframe, play, step |
| F3 / Ctrl+K | command search (falls back to the AI) |
| / or Ctrl+L, N | focus the AI prompt, toggle the Copilot panel |
| Ctrl+Z / Ctrl+Shift+Z, Ctrl+S / O / N | undo / redo, save / open / new |
| Tab | edit mode on the active mesh |
| 1 / 2 / 3 (edit) | vertex / edge / face select |
| E, I, Ctrl+R (edit) | extrude, inset, loop cut |
| M, X, Alt+N (edit) | merge, delete, flip normals |
| Numpad 0, Ctrl+Alt+0 | look through the camera, align the camera to the view |

## Architecture

```
src/
  main.rs      eframe bootstrap
  app.rs       layout, menus, Copilot chat, voice UX, outliner, properties, timeline, palette
  viewport.rs  camera, scene → primitives, navigation, modal transforms, gizmos, HUD
  viewport/edit.rs  edit-mode drawing, component picking, extrude / inset / loop-cut tools
  gpu.rs       wgpu renderer: MSAA colour pass, ID/depth pass + async readback, outline composite
  render.rs    multithreaded software z-buffer rasterizer (fallback), ID buffer, selection outline
  editmesh.rs  mesh operators (extrude, inset, loop cut, subdivide, merge, delete …)
  gltf.rs      glTF 2.0 / .glb exporter
  editor.rs    editor state, undo, and the command executor shared by UI / AI / voice
  ai.rs        Ollama client, model ranking, JSON schema + system prompt, offline parser
  voice.rs     cpal mic capture, VAD, resampling, candle Whisper + tiny BPE decoder
  scene.rs     objects, transforms, materials, modifiers, keyframes, OBJ export
  mesh.rs      PolyMesh (n-gons) + render mesh, ear-clipping triangulation, procedural primitives
  theme.rs     Blender-like dark theme
```

The AI never touches the scene directly. It produces a list of flat `Command` records, for example `{"action":"add","primitive":"cone","location":[0,0,2],"color":"orange"}`, which `Editor::execute` validates and applies. That keeps the AI's actions safe, undoable and visible in the chat log.

Run the tests with `cargo test`. `rust_blender --export-demo out.glb` builds a demo scene from `docs/demo_scene.json` without opening a window and exports it, which is handy for checking the glTF exporter.
