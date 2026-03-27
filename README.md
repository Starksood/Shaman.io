# SHAMAN.OS

A fully offline macOS desktop app that puts a psychedelic trip guide in your pocket. Speak freely — the Guide listens, responds through synthesized speech, and holds space with ambient generative audio and a live fractal visual. No internet, no accounts, no cloud.

The project has two parts:

- **`shamanos_app/`** — Electron desktop app (voice loop, visuals, audio)
- **`shamanos_training/`** — Fine-tune pipeline that produces the `shamanos` Ollama model

---
<img src ="https://github.com/Starksood/Shaman.io/blob/main/Screenshot%202026-03-26%20at%2022.01.56.png">
<img src ="https://github.com/Starksood/Shaman.io/blob/main/Screenshot%202026-03-26%20at%2022.02.26.png">


## How it works

```
You speak
  → whisper.cpp transcribes locally
    → Ollama (shamanos model) generates a response
      → Kokoro TTS synthesizes speech
        → fractal visuals + ambient audio react to state
```

Everything runs on your machine. The only outbound request is to `localhost:11434` (Ollama).

---

## Prerequisites

- macOS (Apple Silicon recommended)
- Node.js 18+
- Python 3.9+
- [Ollama](https://ollama.com) installed
- `sox` — `brew install sox`
- `cmake` — `brew install cmake`

---

## Quick Start

### 1. Train the model (first time only)

```bash
cd shamanos_training
bash setup.sh
source venv/bin/activate
huggingface-cli login   # requires Llama 3.2 access on HuggingFace

python3 0_audit.py /path/to/your/dataset.jsonl
python3 1_train.py /path/to/your/dataset.jsonl
python3 2_convert.py
```

This produces `shamanos_1b_q4km.gguf` (~700MB). See [Training Pipeline](#training-pipeline) for details.

### 2. Load the model into Ollama

```bash
ollama create shamanos -f shamanos_training/Modelfile
```

### 3. Set up and run the app

```bash
cd shamanos_app
npm install
npm run setup     # builds whisper.cpp + installs Kokoro TTS
npm start
```

The app opens fullscreen. Tap anywhere to start speaking.

---

## App Controls

| Gesture | Action |
|---|---|
| Tap anywhere | Start speaking — the Guide listens until you go silent |
| Long press (>600ms) during SPEAKING or THINKING | Enter HOLD mode — 16-second breath work pause |

The voice loop returns to IDLE automatically after each response.

---

## App Architecture

```
Electron main process
  ├── backend/audio_capture.js  — sox mic recorder (16kHz mono WAV)
  ├── backend/stt.js            — whisper.cpp wrapper
  ├── backend/ollama.js         — Ollama HTTP client (localhost:11434)
  └── backend/tts.js            — Kokoro TTS wrapper (af_sky, speed 0.9)

Renderer (single HTML file)
  ├── State machine: IDLE → LISTENING → THINKING → SPEAKING → IDLE
  ├── Background canvas — rotating polyhedra, Lissajous knot, spirograph
  ├── Orb canvas — morphing blob that reacts to state
  └── Web Audio — bass drone (40Hz), binaural beats (4Hz theta), slow pad, shimmer
```

### App States

| State | Orb | Ambient Volume |
|---|---|---|
| IDLE | deep violet, 0.4× speed | 0.30 |
| LISTENING | cyan-teal, 1.2× speed | 0.12 |
| THINKING | amber-gold, 0.8× speed | 0.22 |
| SPEAKING | warm white-rose, 1.0× speed | 0.08 |
| HOLD | deep indigo, 0.15× speed | 0.18 |
| SETUP | grey, 0.2× speed | 0.00 |

On first launch the app checks all dependencies. If anything is missing it enters **SETUP** state and shows what needs to be installed. It transitions to IDLE automatically once everything is ready.

---

## Training Pipeline

Located in `shamanos_training/`. Four sequential scripts, all local — no cloud, no Docker.

### Prerequisites

- Apple Silicon Mac with 8GB unified RAM
- HuggingFace account with [Llama 3.2 1B Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) access approved

### Setup

```bash
cd shamanos_training
bash setup.sh
source venv/bin/activate
huggingface-cli login
```

### Pipeline Steps

**Step 0 — Audit dataset**
```bash
python3 0_audit.py /path/to/dataset.jsonl
```
Validates structure, checks for duplicates, flags over-length records. Writes `audit_report.json`. Fix any `error`-level flags before continuing.

**Step 1 — Fine-tune (4–8 hours)**
```bash
python3 1_train.py /path/to/dataset.jsonl
```
LoRA fine-tune on MPS (r=8, alpha=16). Targets attention projection layers. Output: `shamanos_adapter/`

**Step 2 — Convert to GGUF**
```bash
python3 2_convert.py
```
Merges adapter into base weights, converts to Q4_K_M GGUF via llama.cpp (cloned and built on first run). Output: `shamanos_1b_q4km.gguf`

**Step 3 — Validate**
```bash
ollama serve   # separate terminal if not already running
python3 3_test.py
```
Runs 5 standardized prompts against fine-tuned vs base model. Output: `test_results.json`

### Time Estimates (8GB Apple Silicon)

| Step | Time |
|---|---|
| Audit | < 1 min |
| Training | 4–8 hours |
| Conversion | 30–60 min |
| Testing | 5–10 min |

### Output Artifacts

| Path | Description |
|---|---|
| `audit_report.json` | Dataset validation report |
| `shamanos_adapter/` | LoRA weights — keep this |
| `shamanos_merged/` | Full merged model — large, deletable |
| `shamanos_1b.gguf` | GGUF f16 — large, deletable |
| `shamanos_1b_q4km.gguf` | Final quantized model — use this |
| `test_results.json` | Validation comparison output |

### Memory Troubleshooting

If `1_train.py` crashes with OOM, edit the constants at the top of the file. Apply in order:

```python
MAX_SEQ_LENGTH = 256      # A: halves memory (was 512)
LORA_RANK = 4             # B: fewer trainable params (was 8)
GRAD_ACCUMULATION = 16    # C: smaller memory per step (was 8)
```

---

## Project Structure

```
shamanos_app/
  main.js                 Electron main process + IPC handlers
  preload.js              Context bridge (window.shaman API)
  renderer/index.html     Full UI — state machine, canvas, Web Audio
  backend/
    audio_capture.js      Sox mic recorder
    stt.js                whisper.cpp wrapper
    ollama.js             Ollama HTTP client
    tts.js                Kokoro TTS wrapper
  scripts/
    setup_whisper.sh      Builds whisper.cpp, downloads model
    setup_kokoro.sh       Installs Kokoro Python package
    kokoro_tts.py         TTS CLI script
  models/
    ggml-tiny.en.bin      Whisper tiny English model
    whisper.cpp/          whisper.cpp source (built by setup)

shamanos_training/
  0_audit.py              Dataset validation
  1_train.py              LoRA fine-tuning
  2_convert.py            Adapter merge + GGUF export
  3_test.py               Ollama validation
  Modelfile               Ollama persona config
  setup.sh                Python venv + dependency setup
  requirements.txt        Python dependencies

JSONL Files/              Training dataset files
```

---

## Security & Privacy

- No external network requests (only `localhost:11434`)
- No user data written to disk — conversation history lives in memory only
- Temp audio files (`/tmp/shaman_*.wav`) deleted after each use
- Renderer runs with `contextIsolation: true`, `nodeIntegration: false`
