# Requirements: SHAMAN.OS Voice Application

## Introduction

SHAMAN.OS is a fully offline macOS desktop application that provides a psychedelic trip guide experience through a continuous voice loop. The application runs entirely on-device using Electron, local LLM inference via Ollama, whisper.cpp for speech-to-text, and Kokoro TTS for speech synthesis. There is no internet dependency, no accounts, and no persistent user data.

## Requirements

### 1. Application Shell

#### 1.1 Window Configuration
The application MUST create a fullscreen frameless Electron window with backgroundColor `#020209`, `contextIsolation: true`, and `nodeIntegration: false`.

**Acceptance Criteria**:
- AC1: BrowserWindow is created with `frame: false`, `backgroundColor: '#020209'`, `contextIsolation: true`, `nodeIntegration: false`
- AC2: Window enters fullscreen on macOS at launch
- AC3: Window loads `renderer/index.html` as its content

#### 1.2 Microphone Permission
The application MUST request macOS microphone permission before the voice loop can begin.

**Acceptance Criteria**:
- AC1: `systemPreferences.askForMediaAccess('microphone')` is called at startup
- AC2: If permission is denied, the app displays a message in SETUP state and does not attempt recording

#### 1.3 IPC Channel Registration
The main process MUST register all five IPC handlers before the renderer window loads.

**Acceptance Criteria**:
- AC1: `ipcMain.handle` is registered for: `check-deps`, `start-listening`, `transcribe`, `query-guide`, `speak`
- AC2: Each handler returns a structured response or throws a catchable error
- AC3: Errors are returned as `{ error: string }` objects, never as unhandled rejections

#### 1.4 Context Bridge API
The preload script MUST expose `window.shaman` with exactly five methods via `contextBridge.exposeInMainWorld`.

**Acceptance Criteria**:
- AC1: `window.shaman` exposes: `checkDeps`, `startListening`, `transcribe`, `queryGuide`, `speak`
- AC2: No raw Node.js APIs are exposed to the renderer
- AC3: All methods return Promises

---

### 2. Voice Pipeline

#### 2.1 Microphone Recording
The application MUST record microphone input until silence is detected, producing a 16kHz mono WAV file.

**Acceptance Criteria**:
- AC1: Sox is invoked with `-r 16000 -c 1` and silence detection parameters (1.5s silence threshold)
- AC2: Minimum recording duration is 0.8 seconds (recordings shorter than this are discarded)
- AC3: A safety timeout of 120 seconds terminates recording if silence is never detected
- AC4: Output is written to `/tmp/shaman_rec.wav` (or timestamped variant)
- AC5: Function throws if sox exits with non-zero code

#### 2.2 Speech-to-Text Transcription
The application MUST transcribe recorded audio using the compiled whisper-cli binary with the ggml-tiny.en model.

**Acceptance Criteria**:
- AC1: `whisper-cli` is invoked with `-m ggml-tiny.en.bin -f <wavPath> --output-txt --no-timestamps`
- AC2: Returned transcript is trimmed of leading/trailing whitespace
- AC3: Whisper noise tokens (`[BLANK_AUDIO]`, `[MUSIC]`, `[NOISE]`) are treated as empty transcripts
- AC4: `checkInstalled()` returns false if binary or model file is missing
- AC5: Function throws if binary exits with non-zero code

#### 2.3 LLM Inference
The application MUST query the local Ollama service with the user's transcript and maintain a rolling conversation history.

**Acceptance Criteria**:
- AC1: Requests are sent to `http://localhost:11434/api/chat` with model `shamanos`
- AC2: The system prompt from the Modelfile is prepended to every request
- AC3: Conversation history is capped at 6 messages (3 user/assistant turns); oldest pairs are trimmed first
- AC4: `num_predict` is set to 200 in every request
- AC5: `ping()` returns `true` if Ollama responds to a HEAD/GET request, `false` otherwise
- AC6: Function throws on network error or non-2xx HTTP response

#### 2.4 Text-to-Speech Synthesis
The application MUST synthesize the Guide's response using Kokoro TTS with voice `af_sky` at speed 0.9.

**Acceptance Criteria**:
- AC1: `scripts/kokoro_tts.py` is invoked with `--voice af_sky --speed 0.9 --out /tmp/shaman_tts_<timestamp>.wav`
- AC2: Function returns the path to the synthesized WAV file
- AC3: `checkInstalled()` returns false if the `kokoro` Python package is not importable
- AC4: If synthesis fails, the voice loop continues silently (guide text is still displayed)
- AC5: Temp WAV files are deleted after playback completes

---

### 3. State Machine

#### 3.1 State Transitions
The renderer MUST implement a six-state machine with defined valid transitions.

**Acceptance Criteria**:
- AC1: Valid states are: `IDLE`, `SETUP`, `LISTENING`, `THINKING`, `SPEAKING`, `HOLD`
- AC2: Tap on IDLE → LISTENING
- AC3: Silence detected in LISTENING → THINKING
- AC4: Inference complete in THINKING → SPEAKING
- AC5: TTS playback complete in SPEAKING → IDLE
- AC6: Any error in the pipeline → IDLE (with optional brief error display)
- AC7: No undefined state transitions exist — every (state, event) pair maps to a defined next state

#### 3.2 HOLD Mode
The application MUST support a long-press interrupt that pauses the pipeline and initiates a 16-second breath cycle.

**Acceptance Criteria**:
- AC1: A pointer hold of >600ms during SPEAKING or THINKING triggers HOLD state
- AC2: HOLD state displays a breath prompt and slows the orb to 0.15× speed
- AC3: HOLD state automatically returns to IDLE after exactly 16 seconds
- AC4: Long press during IDLE or LISTENING has no effect

---

### 4. Visual Interface

#### 4.1 Background Canvas
The background canvas MUST render a continuous generative fractal animation with four visual elements.

**Acceptance Criteria**:
- AC1: Canvas renders rotating 3D polyhedra projected to 2D
- AC2: Canvas renders a Lissajous knot (3:2 ratio) that evolves over time
- AC3: Canvas renders 5 spirograph petals rotating slowly
- AC4: Canvas renders bezier tendrils from each corner
- AC5: Each frame applies a trail fade using `rgba(2, 2, 9, 0.15)` clear

#### 4.2 Orb Canvas
The orb canvas MUST render a morphing blob that responds to application state.

**Acceptance Criteria**:
- AC1: Outer blob uses 80 vertices with 5 harmonic noise functions
- AC2: Inner blob renders at 0.6× the outer blob radius
- AC3: A star-tetrahedron (two overlapping triangles) rotates inside the orb
- AC4: 12 radial spokes pulse outward from the orb center
- AC5: Orb color and animation speed change per state according to the state visual mapping table
- AC6: All 80 computed vertex coordinates are finite numbers for any elapsed time value

#### 4.3 State Visual Mapping
Each application state MUST produce a distinct visual appearance.

**Acceptance Criteria**:
- AC1: IDLE — deep violet orb, 0.4× speed
- AC2: LISTENING — cyan-teal orb, 1.2× speed
- AC3: THINKING — amber-gold orb, 0.8× speed
- AC4: SPEAKING — warm white-rose orb, 1.0× speed
- AC5: HOLD — deep indigo orb, 0.15× speed
- AC6: SETUP — grey orb, 0.2× speed

#### 4.4 Guide Text Display
The Guide's response text MUST fade in during SPEAKING state and fade out when the state changes.

**Acceptance Criteria**:
- AC1: Text uses Rajdhani 300 font, italic style
- AC2: Text fades in with a CSS opacity transition (≥0.5s)
- AC3: Text is cleared/faded out when state transitions away from SPEAKING
- AC4: Text is centered on screen, overlaid above the orb canvas

#### 4.5 Status Labels
Application state MUST be indicated by a status label in Share Tech Mono font.

**Acceptance Criteria**:
- AC1: Status label uses Share Tech Mono font
- AC2: Label text reflects current state (e.g., "listening...", "thinking...", "speaking...")
- AC3: Label is not shown during IDLE state (or shows a minimal ambient indicator)

---

### 5. Ambient Audio

#### 5.1 Audio Synthesis
The application MUST synthesize ambient audio entirely via Web Audio API with no audio files.

**Acceptance Criteria**:
- AC1: Bass drone: 40Hz sine oscillator, gain 0.18
- AC2: Binaural beats: 200Hz left channel, 204Hz right channel (4Hz theta difference)
- AC3: Slow pad: 80Hz triangle oscillator with LFO gain modulation (0.05–0.12, 8s period)
- AC4: Shimmer: 2400Hz sine oscillator, gain 0.02
- AC5: All oscillators are created once at startup and never recreated

#### 5.2 Volume Per State
Master volume MUST ramp smoothly to a target level when the application state changes.

**Acceptance Criteria**:
- AC1: IDLE → 0.30 master gain
- AC2: LISTENING → 0.12 master gain
- AC3: THINKING → 0.22 master gain
- AC4: SPEAKING → 0.08 master gain
- AC5: HOLD → 0.18 master gain
- AC6: Volume transitions use `linearRampToValueAtTime` with a 1.5s ramp duration
- AC7: Audio does not play during SETUP state (gain = 0)

---

### 6. Dependency Management

#### 6.1 Dependency Check at Startup
The application MUST check all required local dependencies before entering the voice loop.

**Acceptance Criteria**:
- AC1: `check-deps` IPC handler verifies: whisper binary, whisper model, kokoro Python package, Ollama reachability
- AC2: Returns `{ whisper: bool, kokoro: bool, ollama: bool, allReady: bool }`
- AC3: If `allReady` is false, renderer enters SETUP state
- AC4: Renderer polls `check-deps` every 5 seconds while in SETUP state
- AC5: Renderer transitions to IDLE automatically when `allReady` becomes true

#### 6.2 Setup Scripts
The repository MUST include shell scripts that install all native dependencies.

**Acceptance Criteria**:
- AC1: `scripts/setup_whisper.sh` clones whisper.cpp, builds with cmake, downloads `ggml-tiny.en.bin` to `models/`
- AC2: `scripts/setup_kokoro.sh` installs `kokoro`, `soundfile`, and `numpy` via pip
- AC3: Both scripts are idempotent (safe to run multiple times)
- AC4: Both scripts print clear progress messages

#### 6.3 Kokoro TTS Python Script
The repository MUST include `scripts/kokoro_tts.py` as a standalone CLI script.

**Acceptance Criteria**:
- AC1: Script accepts `--text`, `--voice`, `--speed`, `--out` arguments
- AC2: Script writes a valid WAV file to the path specified by `--out`
- AC3: Script exits with code 0 on success, non-zero on failure
- AC4: Script uses `KPipeline` from the `kokoro` package

---

### 7. Security and Privacy

#### 7.1 No External Network Requests
The application MUST NOT make any network requests to external hosts.

**Acceptance Criteria**:
- AC1: The only HTTP request target is `http://localhost:11434`
- AC2: Google Fonts are the only exception (loaded once at startup for UI rendering)
- AC3: No telemetry, analytics, or crash reporting is included

#### 7.2 No Persistent User Data
The application MUST NOT write any user interaction data to disk.

**Acceptance Criteria**:
- AC1: Conversation history exists only in renderer memory and is lost on app close
- AC2: No session logs, transcripts, or response files are written to disk
- AC3: Temp audio files (`/tmp/shaman_*.wav`) are deleted after each use

#### 7.3 Renderer Isolation
The renderer process MUST NOT have direct access to Node.js APIs.

**Acceptance Criteria**:
- AC1: `nodeIntegration: false` in BrowserWindow webPreferences
- AC2: `contextIsolation: true` in BrowserWindow webPreferences
- AC3: All Node.js operations are performed in the main process and exposed only via the context bridge
