# Implementation Plan: SHAMAN.OS Fine-Tune Pipeline

## Overview

Implement eight files in `shamanos_training/` as a sequential Python pipeline: environment setup, dataset auditing, LoRA fine-tuning on MPS, adapter merging + GGUF conversion, Ollama validation, a Modelfile, and a README. Each step builds on the previous and is independently runnable.

## Tasks

- [x] 1. Create project skeleton and `requirements.txt`
  - Create `shamanos_training/` directory
  - Write `requirements.txt` with pinned minimum versions for `torch>=2.1.0`, `transformers>=4.40.0`, `datasets>=2.18.0`, `trl>=0.8.0`, `peft>=0.10.0`, `accelerate>=0.27.0`, `bitsandbytes>=0.43.0`, `sentencepiece`, `protobuf`, `huggingface_hub`, `scipy`, `numpy`, `httpx`, `hypothesis`
  - No `unsloth` entry
  - _Requirements: 1.3_

- [x] 2. Write `setup.sh`
  - Create `shamanos_training/setup.sh`
  - Create a Python venv, activate it, and run `pip install -r requirements.txt`
  - After install, run a one-liner Python check: `python -c "import torch; print('MPS:', torch.backends.mps.is_available())"` and print a warning if MPS is False
  - Prompt user to run `huggingface-cli login` for model access
  - No hardcoded credentials or API keys
  - _Requirements: 1.1, 1.2, 1.4_

- [x] 3. Implement `0_audit.py` — data models and core check functions
  - Create `shamanos_training/0_audit.py`
  - Define `AuditFlag` and `AuditReport` dataclasses matching the design
  - Implement `check_structure(record, idx)`: returns error flags for missing `messages` key, wrong length, or wrong role order `[system, user, assistant]`
  - Implement `check_empty_fields(record, idx)`: returns warning flags for any empty `content` field
  - Implement `check_literary_analysis(record, idx)`: returns info flag if assistant content lacks literary-analysis keywords (narrative, theme, character, etc.)
  - Implement `check_first_person(record, idx)`: returns info flag if assistant content uses first-person pronouns
  - Implement `check_sequence_length(record, idx, max_tokens=512)`: tokenizes with `transformers` AutoTokenizer and returns warning flag if total tokens > 512
  - _Requirements: 2.2, 2.3, 2.5, 2.6, 2.7_

  - [ ]* 3.1 Write property test for `check_structure` (Property 1)
    - Use `hypothesis` to generate well-formed records and assert empty flag list
    - Use `hypothesis` to generate structurally invalid records and assert at least one error flag
    - **Property 1: Well-formed records produce no structural flags**
    - **Validates: Requirements 2.2, 8.4**

  - [ ]* 3.2 Write property test for `check_sequence_length` (Property 5)
    - Generate records whose tokenized length is above and below 512 and assert correct flag presence
    - **Property 5: Over-length records are always flagged**
    - **Validates: Requirements 2.5**

- [x] 4. Implement `0_audit.py` — aggregate checks and `run_audit`
  - Implement `check_field_lengths(records)`: computes `FieldLengthStats` (min, max, mean, median word counts) across all assistant fields
  - Implement `check_duplicates(records, threshold=0.70)`: O(n²) Jaccard overlap on user-field tokens; each pair `(i, j)` with `i < j` reported at most once
  - Implement `run_audit(dataset_path, output_path)`: loads JSONL line-by-line (catching JSON parse errors as error flags), runs all 7 checks, assembles `AuditReport`, writes `audit_report.json`, prints human-readable summary
  - _Requirements: 2.1, 2.4, 2.8, 2.9, 2.10, 2.11_

  - [ ]* 4.1 Write property test for `check_duplicates` (Property 2)
    - Assert every reported pair `(i, j)` satisfies `i < j`
    - Assert every pair with Jaccard overlap > threshold appears exactly once
    - **Property 2: Duplicate detection is asymmetric and complete**
    - **Validates: Requirements 2.4, 8.3**

  - [ ]* 4.2 Write property test for `check_field_lengths` (Property 4)
    - Generate arbitrary non-empty record lists and assert `min_words <= mean_words <= max_words` and `min_words <= median_words <= max_words`
    - **Property 4: Field-length statistics ordering invariant**
    - **Validates: Requirements 2.8, 8.2**

  - [ ]* 4.3 Write property test for `run_audit` flag count consistency (Property 3)
    - Generate arbitrary JSONL datasets, run `run_audit`, and assert total flag count equals sum of per-check counts in `summary`
    - **Property 3: Audit report flag count is internally consistent**
    - **Validates: Requirements 8.1, 2.9**

- [x] 5. Checkpoint — audit script complete
  - Ensure all tests pass, ask the user if questions arise.

- [x] 6. Implement `1_train.py`
  - Create `shamanos_training/1_train.py`
  - Define named constants at top: `LORA_RANK = 8`, `MAX_SEQ_LENGTH = 512`, `GRADIENT_ACCUMULATION_STEPS = 8`
  - Implement `load_model_and_tokenizer(model_id)`: detects device (`mps`/`cuda`/`cpu`), loads model in `torch.float16` on MPS or with 4-bit quantization on CUDA/CPU, sets `pad_token = eos_token` if missing
  - Implement `apply_lora(model, config)`: applies `LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj","v_proj","k_proj","o_proj"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")` and prints trainable parameter count
  - Implement `format_dataset(dataset, tokenizer)`: applies the model's built-in chat template to each record
  - Implement `build_trainer(model, tokenizer, dataset, args)`: constructs `SFTTrainer` with `per_device_train_batch_size=1`, `gradient_accumulation_steps=8`, `max_seq_length=512`, `fp16=True`, `gradient_checkpointing=True`, `dataloader_pin_memory=False`, `dataloader_num_workers=0`
  - Before training, read `audit_report.json` if it exists and print a warning listing error-level flag count
  - Save only the LoRA adapter and tokenizer config to `lora_adapter/`
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 3.10_

- [x] 7. Implement `2_convert.py`
  - Create `shamanos_training/2_convert.py`
  - Implement `merge_adapter(base_model_id, adapter_path, output_path)`: loads base model on CPU in `torch.float16`, loads LoRA adapter via `PeftModel.from_pretrained`, calls `merge_and_unload()`, saves merged model as safetensors to `merged_model/`
  - Implement `clone_llama_cpp(target_dir)` and `build_llama_cpp(llama_cpp_dir)`: clone official repo and build with `cmake`; check for `cmake` on PATH first and print `"Install cmake via \`brew install cmake\`"` then exit non-zero if missing
  - Implement `convert_to_gguf_f16` and `quantize_to_q4km` using `subprocess.run` with `check=True`
  - Print path and file size of `model_q4km.gguf` on completion
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7_

- [x] 8. Implement `3_test.py`
  - Create `shamanos_training/3_test.py`
  - Define `TestPrompt` and `TestResult` dataclasses
  - Define 5 standardized `TestPrompt` instances covering literary-analysis scenarios
  - Implement `import_model_to_ollama(modelfile_path, model_name)`: calls `ollama create` via subprocess
  - Implement `run_prompt(model_name, prompt, system)`: POSTs to `http://localhost:11434/api/generate` via `httpx` with `timeout=120`; on `ConnectError` prints `"Start Ollama with \`ollama serve\` before running this script"` and exits non-zero; on timeout raises `httpx.TimeoutException` and records failure
  - Implement `compare_responses(prompts, ft_model, base_model)`: runs all 5 prompts against both models, prints side-by-side comparison
  - Implement `save_results(results, output_path)`: writes `test_results.json`
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6_

- [x] 9. Write `Modelfile`
  - Create `shamanos_training/Modelfile`
  - Reference `model_q4km.gguf` as the model source
  - Define a SHAMAN.OS system prompt consistent with literary-analysis training data
  - Set `temperature`, `top_p`, and `top_k` parameters appropriate for literary analysis
  - _Requirements: 6.1, 6.2, 6.3_

- [x] 10. Write `README.md`
  - Create `shamanos_training/README.md`
  - Document prerequisites: Apple Silicon Mac, Ollama installed, HuggingFace account with Llama 3.2 access
  - Provide numbered step-by-step instructions for `setup.sh` → `0_audit.py` → `1_train.py` → `2_convert.py` → `3_test.py`
  - Document `LORA_RANK`, `MAX_SEQ_LENGTH`, `GRADIENT_ACCUMULATION_STEPS` constants and when to reduce them for OOM recovery
  - List all output artifacts and expected locations (`audit_report.json`, `lora_adapter/`, `merged_model/`, `model_q4km.gguf`, `test_results.json`)
  - Include time estimates (~2–4 hours training) and memory troubleshooting tips
  - _Requirements: 7.1, 7.2, 7.3, 7.4_

- [x] 11. Final checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for a faster MVP
- Property tests use `hypothesis` and live alongside the audit module
- Constants at the top of `1_train.py` are the primary knobs for OOM recovery
- `2_convert.py` performs the merge on CPU to avoid MPS memory pressure
- All scripts are independently runnable; no shared state beyond files on disk
