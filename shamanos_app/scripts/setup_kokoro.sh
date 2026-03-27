#!/usr/bin/env bash
# setup_kokoro.sh — Install Kokoro TTS Python dependencies
set -e

echo "[setup_kokoro] Installing Python dependencies..."
pip3 install kokoro soundfile numpy --quiet

echo "[setup_kokoro] Verifying Kokoro installation..."
python3 -c "import kokoro; print('Kokoro ready')"

echo "[setup_kokoro] Setup complete."
