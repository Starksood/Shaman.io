# SHAMAN.OS

A fully offline macOS desktop app that puts a psychedelic trip guide in your pocket. Speak freely — the Guide listens, responds through synthesized speech, and holds space with ambient generative audio and a live fractal visual. No internet, no accounts, no cloud. Everything runs on your machine.

Built with Electron, Ollama (local LLM), whisper.cpp (speech-to-text), and Kokoro TTS (text-to-speech).

---

## Prerequisites

- **Node.js 18+**
- **Python 3.9+**
- **Ollama** — [ollama.com](https://ollama.com) — with the `shamanos` model loaded
- **sox** — `brew install sox`
- **cmake** — `brew install cmake`

---

## Setup

**1. Install Node dependencies**

```bash
npm install
```

**2. Build whisper.cpp and install Kokoro TTS**

```bash
npm run setup
```

This runs `scripts/setup_whisper.sh` (clones and builds whisper.cpp, downloads the tiny English model) and `scripts/setup_kokoro.sh` (installs the Kokoro Python package).

**3. Load the shamanos model into Ollama**

```bash
ollama create shamanos -f ../shamanos_training/Modelfile
```

Adjust the path to `Modelfile` if your training directory is elsewhere.

---

## Run

```bash
npm start
```

The app opens fullscreen. Ambient audio and visuals start immediately.

---

## Controls

| Gesture | Action |
|---|---|
| **Tap anywhere** | Start speaking — the Guide listens until you go silent |
| **Long press (>600ms)** during SPEAKING or THINKING | Enter HOLD mode — a 16-second breath work pause |

After each response the loop returns to IDLE automatically. Tap again to continue the conversation.

---

## First Launch

On first launch the app checks that all dependencies are ready. If anything is missing it enters **SETUP** state and shows what still needs to be installed. Once all deps are detected it transitions to IDLE on its own — no restart needed.

---

## Architecture

```
Electron main process
  ├── backend/audio_capture.js  — sox mic recorder
  ├── backend/stt.js            — whisper.cpp wrapper
  ├── backend/ollama.js         — Ollama HTTP client
  └── backend/tts.js            — Kokoro TTS wrapper

Renderer (single HTML file)
  ├── State machine (IDLE → LISTENING → THINKING → SPEAKING → IDLE)
  ├── Background canvas — rotating polyhedra, Lissajous knot, spirograph
  ├── Orb canvas — morphing blob that reacts to state
  └── Web Audio — bass drone, binaural beats, slow pad, shimmer
```

All inference and audio processing is local. The only outbound request is to `localhost:11434` (Ollama).
