#!/bin/bash
set -e

# Resolve the directory containing this script so it works regardless of
# where the user invokes it from (e.g. bash shamanos_training/setup.sh).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==> Creating Python virtual environment at venv/"
python3 -m venv "$SCRIPT_DIR/venv"

echo "==> Activating virtual environment"
source "$SCRIPT_DIR/venv/bin/activate"

echo "==> Upgrading pip"
pip install --upgrade pip

echo "==> Installing dependencies from requirements.txt"
pip install -r "$SCRIPT_DIR/requirements.txt"

echo "==> Checking MPS (Apple Silicon GPU) availability"
python3 -c "
import torch
avail = torch.backends.mps.is_available()
print('MPS (Apple Silicon GPU) — AVAILABLE' if avail else 'WARNING: MPS not available — training will fall back to CPU')
"

echo ""
echo "==> Setup complete."
echo "    To authenticate with HuggingFace (required for Llama 3.2 model access), run:"
echo "        huggingface-cli login"
echo "    Then follow the prompts to enter your HuggingFace access token."
