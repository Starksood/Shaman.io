# Requirements Document

## Introduction

The SHAMAN.OS Fine-Tune Pipeline is a fully local, end-to-end ML pipeline for Apple Silicon Macs with 8GB unified RAM. It audits a JSONL dataset of ~400 conversation triples, fine-tunes Llama 3.2 1B Instruct using LoRA on the MPS backend, merges and exports the adapter to GGUF Q4_K_M format, and validates the result via Ollama. All deliverables live in `shamanos_training/` as eight files. No cloud dependencies, no Docker, no unsloth.

---

## Glossary

- **Auditor**: The `0_audit.py` script responsible for dataset validation.
- **Trainer**: The `1_train.py` script responsible for LoRA fine-tuning.
- **Converter**: The `2_convert.py` script responsible for merging the adapter and producing a GGUF file.
- **Tester**: The `3_test.py` script responsible for Ollama-based validation.
- **Pipeline**: The four sequential scripts collectively (`0_audit.py` → `1_train.py` → `2_convert.py` → `3_test.py`).
- **AuditFlag**: A structured record of a validation issue found by the Auditor, containing index, check name, severity, message, and detail.
- **AuditReport**: The JSON file written by the Auditor summarising all flags and statistics.
- **LoRA_Adapter**: The PEFT checkpoint produced by the Trainer, containing only the low-rank weight deltas.
- **Merged_Model**: The full float16 safetensors model produced by the Converter after baking LoRA weights into the base model.
- **GGUF_Model**: The quantized Q4_K_M GGUF file produced by the Converter, ready for Ollama inference.
- **MPS**: Apple Metal Performance Shaders GPU backend used for training acceleration.
- **Ollama**: The local inference server used by the Tester to run and compare models.
- **llama.cpp**: The locally-built C++ toolkit used by the Converter for GGUF conversion and quantization.
- **SFTTrainer**: The supervised fine-tuning trainer from the TRL library.
- **TestResult**: A structured record pairing a prompt name with fine-tuned and base model responses.

---

## Requirements

### Requirement 1: Environment Setup

**User Story:** As a developer, I want a single setup script that prepares the Python environment and validates Apple Silicon prerequisites, so that I can start the pipeline without manual dependency management.

#### Acceptance Criteria

1. WHEN `setup.sh` is executed, THE Pipeline SHALL create a Python virtual environment, install all packages listed in `requirements.txt`, and verify that the MPS backend is available.
2. IF the MPS backend is not detected during setup, THEN THE Pipeline SHALL print a warning message indicating that training will fall back to CPU.
3. THE `requirements.txt` SHALL specify minimum version constraints for `torch`, `transformers`, `datasets`, `trl`, `peft`, `accelerate`, `bitsandbytes`, and `httpx`.
4. THE `setup.sh` SHALL NOT require any cloud credentials, API keys, or network access beyond the initial package installation.

---

### Requirement 2: Dataset Auditing

**User Story:** As a developer, I want to validate my JSONL dataset before spending compute on training, so that I can catch structural and quality issues early.

#### Acceptance Criteria

1. WHEN `0_audit.py` is executed with a JSONL file path, THE Auditor SHALL load every line, run all seven validation checks against each record, and write an `audit_report.json` file.
2. WHEN a record is missing the `messages` key, or the `messages` array does not contain exactly three items with roles `[system, user, assistant]` in that order, THE Auditor SHALL emit an AuditFlag with `severity="error"` for that record.
3. WHEN a record contains any empty `content` field in any message, THE Auditor SHALL emit an AuditFlag with `severity="warning"` for that record.
4. WHEN the user-field token overlap between any two records exceeds 70%, THE Auditor SHALL emit an AuditFlag with `severity="warning"` identifying both record indices.
5. WHEN a record's tokenized length exceeds 512 tokens, THE Auditor SHALL emit an AuditFlag with `severity="warning"` for that record.
6. WHEN a record's assistant response lacks literary-analysis language (e.g., references to narrative, theme, or character), THE Auditor SHALL emit an AuditFlag with `severity="info"` for that record.
7. WHEN a record's assistant response is written in first person, THE Auditor SHALL emit an AuditFlag with `severity="info"` for that record.
8. THE Auditor SHALL compute field-length statistics (min, max, mean, median word counts) across all records and include them in `audit_report.json`.
9. THE `audit_report.json` SHALL contain the total record count, all AuditFlags, field-length statistics, duplicate pairs, and a per-check flag count summary.
10. THE Auditor SHALL print a human-readable summary to stdout after writing `audit_report.json`.
11. IF a JSONL line cannot be parsed as valid JSON, THEN THE Auditor SHALL emit an AuditFlag with `severity="error"` for that line index and continue processing remaining lines.

---

### Requirement 3: LoRA Fine-Tuning

**User Story:** As a developer, I want to fine-tune Llama 3.2 1B Instruct with LoRA on my Apple Silicon Mac, so that I can adapt the model to my dataset without exceeding 8GB RAM.

#### Acceptance Criteria

1. WHEN `1_train.py` is executed, THE Trainer SHALL detect the available device (`mps`, `cuda`, or `cpu`) and configure the model dtype and loading strategy accordingly.
2. WHILE training on MPS, THE Trainer SHALL load the model in `torch.float16` without 4-bit quantization.
3. WHILE training on CUDA or CPU, THE Trainer SHALL load the model with 4-bit quantization via `bitsandbytes`.
4. THE Trainer SHALL apply LoRA with `r=8`, `lora_alpha=16`, targeting `["q_proj", "v_proj", "k_proj", "o_proj"]`, `lora_dropout=0.05`, `bias="none"`, and `task_type="CAUSAL_LM"`.
5. THE Trainer SHALL format the dataset using the model's built-in chat template before training.
6. THE Trainer SHALL run SFTTrainer with `per_device_train_batch_size=1`, `gradient_accumulation_steps=8`, `max_seq_length=512`, `fp16=True` on MPS, `gradient_checkpointing=True`, `dataloader_pin_memory=False`, and `dataloader_num_workers=0`.
7. WHEN training completes, THE Trainer SHALL save only the LoRA adapter weights and tokenizer config to `lora_adapter/`, without duplicating the full base model weights.
8. WHEN `1_train.py` is executed and `audit_report.json` exists with error-level flags, THE Trainer SHALL print a warning listing the error count before proceeding.
9. THE Trainer SHALL print the count of trainable parameters after applying LoRA.
10. THE key training hyperparameters (`LORA_RANK`, `MAX_SEQ_LENGTH`, `GRADIENT_ACCUMULATION_STEPS`) SHALL be defined as named constants at the top of `1_train.py` to facilitate memory tuning.

---

### Requirement 4: Adapter Merging and GGUF Conversion

**User Story:** As a developer, I want to merge the LoRA adapter into the base model and export a quantized GGUF file, so that I can run the fine-tuned model locally via Ollama.

#### Acceptance Criteria

1. WHEN `2_convert.py` is executed, THE Converter SHALL load the base model on CPU in `torch.float16`, load the LoRA adapter via PEFT, call `merge_and_unload()`, and save the merged model as safetensors to `merged_model/`.
2. WHEN the merged model has been saved, THE Converter SHALL convert it to a float16 GGUF file using `llama.cpp/convert_hf_to_gguf.py`.
3. WHEN the float16 GGUF file exists, THE Converter SHALL quantize it to Q4_K_M using the `llama-quantize` binary, producing `model_q4km.gguf`.
4. IF the `llama.cpp/` directory does not exist, THEN THE Converter SHALL clone the llama.cpp repository and build it via `cmake` before proceeding with conversion.
5. IF `cmake` is not found on PATH, THEN THE Converter SHALL print the message "Install cmake via `brew install cmake`" and exit with a non-zero code.
6. THE Converter SHALL perform the merge step on CPU to avoid MPS memory pressure during the merge operation.
7. WHEN conversion completes, THE Converter SHALL print the path and file size of the resulting `model_q4km.gguf`.

---

### Requirement 5: Ollama Model Validation

**User Story:** As a developer, I want to compare the fine-tuned model against the base model using standardized prompts via Ollama, so that I can verify the fine-tuning had the intended effect.

#### Acceptance Criteria

1. WHEN `3_test.py` is executed, THE Tester SHALL import the GGUF model into Ollama using the provided `Modelfile`.
2. THE Tester SHALL run exactly 5 standardized test prompts against both the fine-tuned model and the base Llama 3.2 1B model via the Ollama HTTP API.
3. WHEN all prompts have been evaluated, THE Tester SHALL save structured results to `test_results.json` containing the prompt name, fine-tuned response, base response, and timestamp for each prompt.
4. THE Tester SHALL print a side-by-side comparison of fine-tuned and base responses to stdout for each prompt.
5. IF the Ollama server is not reachable at `http://localhost:11434`, THEN THE Tester SHALL print "Start Ollama with `ollama serve` before running this script" and exit with a non-zero code.
6. IF an Ollama API call does not return a response within 120 seconds, THEN THE Tester SHALL raise a timeout error and record the failure in `test_results.json`.

---

### Requirement 6: Ollama Modelfile

**User Story:** As a developer, I want a Modelfile that configures the fine-tuned model's persona and system prompt for Ollama, so that inference behavior matches the training intent.

#### Acceptance Criteria

1. THE `Modelfile` SHALL reference the `model_q4km.gguf` file produced by the Converter.
2. THE `Modelfile` SHALL define a system prompt that reflects the SHAMAN.OS persona consistent with the training data.
3. THE `Modelfile` SHALL set inference parameters (`temperature`, `top_p`, `top_k`) appropriate for literary analysis responses.

---

### Requirement 7: Documentation

**User Story:** As a developer, I want a README that explains how to run the full pipeline end-to-end, so that I can reproduce the process without referring to external documentation.

#### Acceptance Criteria

1. THE `README.md` SHALL document the prerequisites (Apple Silicon Mac, Ollama installed, HuggingFace account with model access).
2. THE `README.md` SHALL provide step-by-step instructions for running each of the four pipeline scripts in order.
3. THE `README.md` SHALL document the memory tuning constants in `1_train.py` and when to adjust them.
4. THE `README.md` SHALL list all output artifacts and their expected locations.

---

### Requirement 8: Audit Data Integrity

**User Story:** As a developer, I want the audit report to be internally consistent, so that I can trust the statistics it reports.

#### Acceptance Criteria

1. THE Auditor SHALL ensure the total flag count in `audit_report.json` equals the sum of all per-check flag counts in the summary.
2. THE Auditor SHALL ensure the field-length statistics satisfy `min_words <= mean_words <= max_words`.
3. WHEN the Auditor reports a duplicate pair `(i, j)`, THE Auditor SHALL ensure `i < j` so that no pair is reported more than once.
4. THE Auditor SHALL ensure that `check_structure` returns an empty flag list for any record that has a `messages` key containing exactly three items with roles `[system, user, assistant]` in that order.
