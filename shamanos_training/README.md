# SHAMAN.OS Fine-Tune Pipeline

Local fine-tuning for Llama 3.2 1B Instruct on Apple Silicon 8GB RAM.
No cloud. No Docker. No unsloth.

---

## Prerequisites

- Apple Silicon Mac (M1/M2/M3) with 8GB unified RAM
- Python 3.9+
- [Ollama](https://ollama.ai) installed and running
- HuggingFace account with Llama 3.2 access approved
  → Request at: https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
- cmake: `brew install cmake`
- git

---

## Setup

```bash
bash setup.sh
source venv/bin/activate
huggingface-cli login
```

---

## Run in order

### Step 0 — Audit your dataset

```bash
python3 0_audit.py /path/to/dataset_shamanos.jsonl
```

Writes `audit_report.json`. Fix any error-level flags before continuing.

### Step 1 — Fine-tune (4–8 hours on 8GB Mac)

```bash
python3 1_train.py /path/to/dataset_shamanos.jsonl
```

Watch the loss. It should drop from ~2.0 toward ~0.8 over 3 epochs.
Output: `shamanos_adapter/`

### Step 2 — Convert to GGUF

```bash
python3 2_convert.py
```

Clones and builds llama.cpp on first run (~10 min).
Output: `shamanos_1b_q4km.gguf` (~700 MB)

### Step 3 — Test the model

```bash
ollama serve   # in a separate terminal if not already running
python3 3_test.py
```

Compares fine-tuned vs base `llama3.2:1b` on 5 guidance scenarios.
Output: `test_results.json`

---

## What to look for during training

**Good signs:**
- Loss drops from ~2.0 to below 1.0 over 3 epochs
- Eval loss tracks training loss (no large gap)
- Loss curve is smooth, not spiky

**Warning signs:**
- Loss does not drop at all → dataset format issue, check `audit_report.json`
- Eval loss rises while training loss drops → overfitting, reduce `NUM_EPOCHS` to 2
- OOM crash → see Memory Troubleshooting below

---

## Memory Troubleshooting

If `1_train.py` crashes with out-of-memory, edit the constants at the top of the file.
Apply in this order — try A first, then B, then C.

**Option A — Reduce sequence length (try first)**
```python
MAX_SEQ_LENGTH = 256   # was 512 — halves memory during training
```

**Option B — Reduce LoRA rank**
```python
LORA_RANK = 4   # was 8 — fewer trainable parameters
```

**Option C — Increase gradient accumulation**
```python
GRAD_ACCUMULATION = 16   # was 8 — smaller effective memory per step
```

---

## Time estimates (8GB Apple Silicon)

| Step       | Time            |
|------------|-----------------|
| Audit      | < 1 minute      |
| Training   | 4–8 hours       |
| Conversion | 30–60 minutes   |
| Testing    | 5–10 minutes    |

---

## Output artifacts

| Path                      | Description                              |
|---------------------------|------------------------------------------|
| `audit_report.json`       | Dataset validation report                |
| `shamanos_adapter/`       | LoRA weights — keep this                 |
| `shamanos_merged/`        | Full merged model — large, deletable     |
| `shamanos_1b.gguf`        | GGUF f16 — large, deletable              |
| `shamanos_1b_q4km.gguf`   | Final quantized model — use this         |
| `test_results.json`       | Validation comparison output             |

---

## Verification

```bash
# Confirm all files are present
ls shamanos_training/

# Verify MPS is available
source venv/bin/activate
python3 -c "import torch; print(torch.backends.mps.is_available())"
# Should print: True

# Test the audit script
python3 0_audit.py /path/to/your/dataset.jsonl
```
