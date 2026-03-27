"""
1_train.py — SHAMAN.OS LoRA Fine-Tuning Script

Fine-tunes Llama 3.2 1B Instruct with LoRA on MPS / CUDA / CPU.
Constants at the top are the primary OOM tuning knobs.
"""

import json
import sys
import os
from datetime import datetime, timezone

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig

# ---------------------------------------------------------------------------
# OOM tuning knobs — adjust these first when hitting memory limits
# ---------------------------------------------------------------------------
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
OUTPUT_DIR = "./shamanos_checkpoints"
FINAL_ADAPTER_DIR = "./shamanos_adapter"
MAX_SEQ_LENGTH = 512
LORA_RANK = 8
LORA_ALPHA = 16
BATCH_SIZE = 1
GRAD_ACCUMULATION = 8
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
WARMUP_STEPS = 10


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------

def get_device() -> str:
    """Return the best available device and print which one is being used."""
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"[device] Using: {device}")
    return device


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset_from_jsonl(path: str) -> Dataset:
    """Load a JSONL file and return a HuggingFace Dataset."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"[dataset] Loaded {len(records)} triples from {path}")
    return Dataset.from_list(records)


# ---------------------------------------------------------------------------
# Chat-template formatting
# ---------------------------------------------------------------------------

def format_for_training(example, tokenizer) -> dict:
    """Apply the model's chat template to a single example."""
    formatted_text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": formatted_text}


# ---------------------------------------------------------------------------
# Audit warning check
# ---------------------------------------------------------------------------

def check_audit_warnings():
    """Warn if audit_report.json contains error-level flags."""
    audit_path = "audit_report.json"
    if not os.path.exists(audit_path):
        return
    with open(audit_path, "r", encoding="utf-8") as f:
        report = json.load(f)
    flags = report.get("flags", [])
    error_count = sum(1 for flag in flags if flag.get("severity") == "error")
    if error_count > 0:
        print(
            f"WARNING: audit_report.json contains {error_count} error-level flags. "
            "Fix dataset issues before training."
        )


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(jsonl_path: str):
    """Full training pipeline: load → configure → train → save."""

    check_audit_warnings()
    device = get_device()

    # ------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------
    print(f"[model] Loading tokenizer for {MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    print(f"[model] Loading model for {MODEL_ID} on {device} ...")
    if device == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(model)

    # ------------------------------------------------------------------
    # LoRA
    # ------------------------------------------------------------------
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    raw_dataset = load_dataset_from_jsonl(jsonl_path)
    formatted_dataset = raw_dataset.map(
        lambda ex: format_for_training(ex, tokenizer)
    )
    split = formatted_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    print(f"[dataset] Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    # ------------------------------------------------------------------
    # Training arguments
    # ------------------------------------------------------------------
    bf16_supported = device == "cuda" and torch.cuda.is_bf16_supported()
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION,
        warmup_steps=WARMUP_STEPS,
        learning_rate=LEARNING_RATE,
        fp16=(device == "cuda"),
        bf16=bf16_supported,
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=25,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        dataloader_pin_memory=False,
        gradient_checkpointing=True,
        optim="adamw_torch",
        dataloader_num_workers=0,
        remove_unused_columns=False,
        dataset_text_field="text",
        max_length=MAX_SEQ_LENGTH,
        packing=False,
    )

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
    )

    # ------------------------------------------------------------------
    # Pre-training info
    # ------------------------------------------------------------------
    if device == "mps":
        allocated_mb = torch.mps.current_allocated_memory() / (1024 ** 2)
        print(f"[memory] MPS allocated before training: {allocated_mb:.1f} MB")

    effective_batch = BATCH_SIZE * GRAD_ACCUMULATION
    steps_per_epoch = max(1, len(train_dataset) // effective_batch)
    total_steps = steps_per_epoch * NUM_EPOCHS
    print(f"[training] Effective batch size: {effective_batch}")
    print(f"[training] Estimated steps/epoch: {steps_per_epoch}, total: {total_steps}")

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    print("[training] Starting training ...")
    trainer.train()

    # ------------------------------------------------------------------
    # Save adapter
    # ------------------------------------------------------------------
    print(f"[save] Saving LoRA adapter to {FINAL_ADAPTER_DIR} ...")
    model.save_pretrained(FINAL_ADAPTER_DIR)
    tokenizer.save_pretrained(FINAL_ADAPTER_DIR)

    # ------------------------------------------------------------------
    # Training metadata
    # ------------------------------------------------------------------
    metadata = {
        "model_id": MODEL_ID,
        "dataset_path": jsonl_path,
        "train_size": len(train_dataset),
        "eval_size": len(eval_dataset),
        "num_epochs": NUM_EPOCHS,
        "lora_rank": LORA_RANK,
        "max_seq_length": MAX_SEQ_LENGTH,
        "device": device,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    metadata_path = os.path.join(FINAL_ADAPTER_DIR, "training_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"[save] Training metadata written to {metadata_path}")
    print("[done] Training complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python 1_train.py <path_to_dataset.jsonl>")
        sys.exit(1)
    train(sys.argv[1])
