#!/usr/bin/env bash
# setup_whisper.sh — Build whisper.cpp and download ggml-tiny.en.bin model
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="$SCRIPT_DIR/../models"

mkdir -p "$MODELS_DIR"

# Clone whisper.cpp if not already present
if [ ! -d "$MODELS_DIR/whisper.cpp" ]; then
  echo "[setup_whisper] Cloning whisper.cpp..."
  git clone https://github.com/ggerganov/whisper.cpp "$MODELS_DIR/whisper.cpp"
else
  echo "[setup_whisper] whisper.cpp already cloned, skipping."
fi

# Build whisper-cli if not already built
WHISPER_BIN="$MODELS_DIR/whisper.cpp/build/bin/whisper-cli"
if [ ! -f "$WHISPER_BIN" ]; then
  echo "[setup_whisper] Building whisper-cli..."
  cd "$MODELS_DIR/whisper.cpp"
  mkdir -p build
  cd build
  cmake .. -DCMAKE_BUILD_TYPE=Release
  cmake --build . --target whisper-cli -j4
  echo "[setup_whisper] Build complete."
else
  echo "[setup_whisper] whisper-cli already built, skipping."
fi

# Download model if not already present
MODEL_FILE="$MODELS_DIR/ggml-tiny.en.bin"
if [ ! -f "$MODEL_FILE" ]; then
  echo "[setup_whisper] Downloading ggml-tiny.en.bin..."
  curl -L https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin \
    -o "$MODEL_FILE"
  echo "[setup_whisper] Model downloaded."
else
  echo "[setup_whisper] ggml-tiny.en.bin already present, skipping."
fi

echo "[setup_whisper] Setup complete."
