# Tasks: SHAMAN.OS Voice Application

## Task List

### Phase 1: Project Scaffold

- [x] 1.1 Create shamanos_app/ directory structure with all required folders and placeholder files
  - `package.json`, `main.js`, `preload.js`, `renderer/index.html`, `backend/stt.js`, `backend/tts.js`, `backend/ollama.js`, `backend/audio_capture.js`, `scripts/setup_whisper.sh`, `scripts/setup_kokoro.sh`, `scripts/kokoro_tts.py`, `models/.gitkeep`, `assets/.gitkeep`, `README.md`

- [x] 1.2 Write package.json with Electron dependency, main entry point, and npm scripts
  - `"main": "main.js"`, `"start": "electron ."`, electron ^28, no bundler

- [x] 1.3 Write README.md with setup instructions (setup_whisper.sh, setup_kokoro.sh, ollama import, npm start)

---

### Phase 2: Electron Main Process

- [x] 2.1 Implement main.js — window creation
  - Fullscreen frameless BrowserWindow, backgroundColor `#020209`, contextIsolation true, nodeIntegration false, preload path

- [x] 2.2 Implement main.js — microphone permission request
  - Call `systemPreferences.askForMediaAccess('microphone')` at app ready

- [x] 2.3 Implement main.js — IPC handler: check-deps
  - Calls `stt.checkInstalled()`, `tts.checkInstalled()`, `ollama.ping()`, returns DepsStatus object

- [x] 2.4 Implement main.js — IPC handler: start-listening
  - Calls `audio_capture.recordUntilSilence()`, returns WAV path

- [x] 2.5 Implement main.js — IPC handler: transcribe
  - Calls `stt.transcribe(wavPath)`, returns transcript string

- [x] 2.6 Implement main.js — IPC handler: query-guide
  - Calls `ollama.chat(text, history)`, returns response string

- [x] 2.7 Implement main.js — IPC handler: speak
  - Calls `tts.synthesize(text)`, plays WAV via shell (afplay on macOS), deletes temp file after playback

---

### Phase 3: Preload Bridge

- [x] 3.1 Implement preload.js
  - `contextBridge.exposeInMainWorld('shaman', { checkDeps, startListening, transcribe, queryGuide, speak })`
  - All methods invoke corresponding `ipcRenderer.invoke` calls

---

### Phase 4: Backend Modules

- [x] 4.1 Implement backend/ollama.js
  - `ping()` — GET `http://localhost:11434/api/tags`, return boolean
  - `chat(userMessage, history)` — POST to `/api/chat`, model `shamanos`, inject system prompt, trim history to 6 messages, return response string

- [x] 4.2 Implement backend/audio_capture.js
  - `recordUntilSilence()` — spawn sox with 16kHz mono WAV output and silence detection args
  - Enforce 0.8s minimum duration, 120s safety timeout
  - Return WAV path on success, throw on sox error

- [x] 4.3 Implement backend/stt.js
  - `checkInstalled()` — verify WHISPER_BIN and WHISPER_MODEL paths exist
  - `transcribe(audioPath)` — execFile whisper-cli, strip noise tokens, return trimmed transcript

- [x] 4.4 Implement backend/tts.js
  - `checkInstalled()` — run `python -c "import kokoro"`, return boolean
  - `synthesize(text)` — spawn `python scripts/kokoro_tts.py` with args, return output WAV path

---

### Phase 5: Setup Scripts

- [x] 5.1 Write scripts/setup_whisper.sh
  - Clone `ggerganov/whisper.cpp` into `models/whisper.cpp`
  - Build with cmake (`cmake -B build && cmake --build build --config Release`)
  - Download `ggml-tiny.en.bin` to `models/`
  - Idempotent (skip steps if already done)

- [x] 5.2 Write scripts/setup_kokoro.sh
  - `pip install kokoro soundfile numpy`
  - Print success message

- [x] 5.3 Write scripts/kokoro_tts.py
  - Accept `--text`, `--voice`, `--speed`, `--out` via argparse
  - Use `KPipeline` from `kokoro` to synthesize
  - Write WAV to `--out` path using `soundfile`
  - Exit 0 on success, 1 on error

---

### Phase 6: Renderer — State Machine and Voice Loop

- [x] 6.1 Implement renderer/index.html — HTML structure and CSS
  - Full-viewport layout, two canvas elements (background, orb), guide text overlay, status label
  - Import Rajdhani 300 and Share Tech Mono from Google Fonts
  - All styles inline in `<style>` block

- [x] 6.2 Implement renderer — AppState machine
  - Define 6 states, transition function, state change handler that updates visuals and audio

- [x] 6.3 Implement renderer — voice loop
  - Tap handler → LISTENING → transcribe → THINKING → query → SPEAKING → IDLE
  - Empty transcript handling (return to IDLE silently)
  - Error handling (any pipeline error → IDLE)

- [x] 6.4 Implement renderer — HOLD mode
  - Pointer hold >600ms during SPEAKING or THINKING → HOLD state
  - Display breath prompt, 16s timer, return to IDLE

- [x] 6.5 Implement renderer — dependency check and SETUP state
  - Call `checkDeps` on load, poll every 5s while in SETUP
  - Show per-dependency status and setup instructions

---

### Phase 7: Ambient Audio

- [x] 7.1 Implement Web Audio ambient synthesis
  - Bass drone: 40Hz sine, gain 0.18
  - Binaural beats: 200Hz left / 204Hz right
  - Slow pad: 80Hz triangle with 0.125Hz LFO gain (0.05–0.12)
  - Shimmer: 2400Hz sine, gain 0.02
  - All nodes created once at startup

- [x] 7.2 Implement per-state volume ramping
  - VOLUME_MAP: `{IDLE:0.30, LISTENING:0.12, THINKING:0.22, SPEAKING:0.08, HOLD:0.18, SETUP:0.00}`
  - Use `linearRampToValueAtTime` with 1.5s ramp on every state change

---

### Phase 8: Visual Animation

- [x] 8.1 Implement background canvas animation
  - Rotating 3D polyhedra (projected to 2D)
  - Lissajous knot (3:2 ratio, time-evolving)
  - 5 spirograph petals
  - 4 corner bezier tendrils
  - Trail fade: `rgba(2, 2, 9, 0.15)` per frame

- [x] 8.2 Implement orb canvas animation
  - 80-vertex outer blob with 5 harmonic noise functions
  - Inner blob at 0.6× radius
  - Star-tetrahedron (two counter-rotating triangles)
  - 12 radial spokes with pulsing length
  - State-driven color and speed changes per ORB_SPEED_MAP and color map

---

### Phase 9: Integration and Polish

- [x] 9.1 Wire full voice loop end-to-end
  - Verify IDLE → LISTENING → THINKING → SPEAKING → IDLE cycle works with real Ollama + whisper + kokoro

- [x] 9.2 Verify HOLD mode interrupt at each pipeline stage

- [x] 9.3 Verify SETUP state auto-transitions to IDLE when all deps become ready

- [x] 9.4 Verify temp file cleanup after each voice cycle

- [x] 9.5 Verify no zombie sox/whisper/python processes remain after app close
  - Add `app.on('before-quit')` handler to kill any running child processes
