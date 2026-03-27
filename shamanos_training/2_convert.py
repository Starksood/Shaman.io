"""
2_convert.py — Merge LoRA adapter into base model and export to GGUF Q4_K_M.

Steps:
  1. Load base model on CPU (float16) + LoRA adapter, merge and save safetensors.
  2. Clone and build llama.cpp if not already present.
  3. Convert merged model to GGUF f16.
  4. Quantize to Q4_K_M.
"""

import os
import sys
import shutil
import subprocess

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
ADAPTER_DIR = "./shamanos_adapter"
MERGED_DIR = "./shamanos_merged"
GGUF_F16 = "./shamanos_1b.gguf"
GGUF_QUANTIZED = "./shamanos_1b_q4km.gguf"
LLAMACPP_DIR = "./llama.cpp"


# ---------------------------------------------------------------------------
# Step 1: Merge LoRA adapter into base model
# ---------------------------------------------------------------------------

def merge_adapter() -> None:
    """Load base model + LoRA adapter, merge weights, save safetensors."""
    print(f"[1/4] Loading base model '{MODEL_ID}' on CPU (float16)…")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )

    print(f"[1/4] Loading tokenizer from '{ADAPTER_DIR}'…")
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)

    print(f"[1/4] Loading LoRA adapter from '{ADAPTER_DIR}'…")
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR, torch_dtype=torch.float16)

    print("[1/4] Merging adapter weights into base model…")
    model = model.merge_and_unload()

    print(f"[1/4] Saving merged model to '{MERGED_DIR}' (safetensors, max shard 2 GB)…")
    model.save_pretrained(MERGED_DIR, safe_serialization=True, max_shard_size="2GB")

    print(f"[1/4] Saving tokenizer to '{MERGED_DIR}'…")
    tokenizer.save_pretrained(MERGED_DIR)

    print(f"[1/4] Merge complete → {MERGED_DIR}")


# ---------------------------------------------------------------------------
# Step 2: Clone llama.cpp
# ---------------------------------------------------------------------------

def setup_llamacpp() -> None:
    """Clone llama.cpp repository if not already present."""
    if os.path.exists(LLAMACPP_DIR):
        print(f"[2/4] llama.cpp already present at '{LLAMACPP_DIR}', skipping clone.")
        return

    if shutil.which("cmake") is None:
        print("Install cmake via `brew install cmake`")
        sys.exit(1)

    print(f"[2/4] Cloning llama.cpp into '{LLAMACPP_DIR}'…")
    subprocess.run(
        ["git", "clone", "https://github.com/ggerganov/llama.cpp", LLAMACPP_DIR],
        check=True,
    )

    print("[2/4] Installing llama.cpp Python requirements…")
    subprocess.run(
        ["pip", "install", "-r", f"{LLAMACPP_DIR}/requirements.txt"],
        check=True,
    )

    print("[2/4] llama.cpp setup complete.")


# ---------------------------------------------------------------------------
# Step 3: Build llama.cpp
# ---------------------------------------------------------------------------

def build_llamacpp() -> None:
    """Build llama-quantize binary via cmake if not already built."""
    quantize_bin = f"{LLAMACPP_DIR}/build/bin/llama-quantize"
    if os.path.exists(quantize_bin):
        print(f"[3/4] llama.cpp already built ({quantize_bin}), skipping build.")
        return

    if shutil.which("cmake") is None:
        print("Install cmake via `brew install cmake`")
        sys.exit(1)

    build_dir = f"{LLAMACPP_DIR}/build"
    os.makedirs(build_dir, exist_ok=True)

    print(f"[3/4] Configuring cmake in '{build_dir}'…")
    subprocess.run(
        ["cmake", "..", "-DCMAKE_BUILD_TYPE=Release"],
        cwd=build_dir,
        check=True,
    )

    print("[3/4] Building llama-quantize target…")
    subprocess.run(
        ["cmake", "--build", ".", "--config", "Release", "--target", "llama-quantize", "-j4"],
        cwd=build_dir,
        check=True,
    )

    print("[3/4] llama.cpp build complete.")


# ---------------------------------------------------------------------------
# Step 4: Convert merged model to GGUF f16
# ---------------------------------------------------------------------------

def convert_to_gguf() -> None:
    """Convert merged HuggingFace model to GGUF float16 format."""
    print(f"[4a/4] Converting '{MERGED_DIR}' → GGUF f16 '{GGUF_F16}'…")
    subprocess.run(
        [
            "python3",
            f"{LLAMACPP_DIR}/convert_hf_to_gguf.py",
            MERGED_DIR,
            "--outfile", GGUF_F16,
            "--outtype", "f16",
        ],
        check=True,
    )
    print(f"GGUF f16 saved: {GGUF_F16}")


# ---------------------------------------------------------------------------
# Step 5: Quantize GGUF to Q4_K_M
# ---------------------------------------------------------------------------

def quantize_gguf() -> None:
    """Quantize the f16 GGUF to Q4_K_M using llama-quantize."""
    # Prefer the cmake-built binary; fall back to repo root location.
    primary = f"{LLAMACPP_DIR}/build/bin/llama-quantize"
    fallback = f"{LLAMACPP_DIR}/llama-quantize"
    quantize_bin = primary if os.path.exists(primary) else fallback

    print(f"[4b/4] Quantizing '{GGUF_F16}' → Q4_K_M '{GGUF_QUANTIZED}'…")
    subprocess.run(
        [quantize_bin, GGUF_F16, GGUF_QUANTIZED, "Q4_K_M"],
        check=True,
    )

    size_bytes = os.path.getsize(GGUF_QUANTIZED)
    size_mb = size_bytes / (1024 * 1024)
    print(f"Quantized GGUF: {GGUF_QUANTIZED} ({size_mb:.0f} MB)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    merge_adapter()
    setup_llamacpp()
    build_llamacpp()
    convert_to_gguf()
    quantize_gguf()
    print("Conversion complete. Next step: run 3_test.py")
