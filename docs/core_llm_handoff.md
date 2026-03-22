# core_llm handoff / runbook (2026-03-22)

This document captures the current state and exact commands to continue the
Japanese small LLM experiments (pretrain + SFT), including the new LLaMA
variant and Ollama/GGUF pipeline.

## Goal (current)

1) Keep GPT-style pipeline working (baseline).  
2) Build **LLaMA-style** variant for GGUF → Ollama distribution.  
3) Evaluate **before** export and only export if eval is stable.

## Environment

- Workdir: `~/LLM/core_llm`
- Activate venv: `source ../venv/bin/activate`
- Always set `PYTHONPATH=src`

## Important mechanics

- `train` and `train_sft` resume if `latest.pt` exists.
- Use a new work dir for clean comparisons.
- Do not change model config or tokenizer when resuming.

## Evaluation settings (use these)

Short, low-temp, stop-at-`。`:

```bash
--max-new-tokens 64 --temperature 0.2 --repetition-penalty 1.2 --stop "。"
```

`evaluate_prompt_set` comparison scoring was relaxed to accept “Aは…Bは…” without explicit keywords.

## Current GPT baseline (for reference)

Baseline GPT is still OK, but **not Ollama-compatible**. It is here for
comparison only; the current priority is LLaMA.

## LLaMA implementation (added in repo)

Implemented LLaMA-style model in core:

- `core_llm/model/llama.py` (RMSNorm + RoPE + SwiGLU)
- `core_llm/model/factory.py` (model_type routing)
- `configs/model_llama_small_ja_sample.yaml` **in repo root** (`~/LLM/configs/`)
- `model_type: llama` in config

Model selection now uses `build_model()`.

## LLaMA low-LR base run (latest, good)

Base run:

```
data/runs/wiki_tiny_sample_20260320_042159_model_llama_small_ja_sample_train_small_sample_50k_llama_lowlr_docs200000__step19500of50000__early
```

- docs: 200k
- best_val_perplexity: ~27.93
- early_stop: step 19500
- config: `../configs/model_llama_small_ja_sample.yaml`
- train config: `../configs/train_small_sample_50k_llama_lowlr.yaml`

Train config (root `../configs/`):

```yaml
batch_size: 2
seq_len: 256
learning_rate: 1e-4
weight_decay: 0.1
grad_accum_steps: 4
warmup_steps: 1000
total_steps: 50000
eval_every: 250
save_every: 250
seed: 42
device: auto
amp: false
min_lr_ratio: 0.1
grad_clip: 1.0
early_stopping_patience: 30
```

## LLaMA SFT run (latest)

```
data/runs/sft_20260322_072421_wiki_tiny_sample_20260320_042159_model_llama_sma_train_sft_small_sample_sft_ja__step1500of2000__early
```

Eval (short+stop):

- qa_ok_rate: **0.90**
- summary: 0.5
- comparison: 1.0
- reasoning: 1.0
- procedure: 0.5

Evaluate command:

```bash
PYTHONPATH=src python3 -m core_llm.scripts.evaluate_prompt_set \
  --checkpoint <SFT_RUN>/checkpoints/best.pt \
  --tokenizer <LLAMA_TOKENIZER>/tokenizer/tokenizer.model \
  --questions data/raw/sft/eval_questions_ja.jsonl \
  --output data/eval/sft_eval_llama_lowlr_stop.jsonl \
  --max-new-tokens 64 \
  --temperature 0.2 \
  --repetition-penalty 1.2 \
  --stop "。"
```

## HF export (LLaMA)

Export script added:

```
core_llm/scripts/export_hf_llama.py
```

Usage:

```bash
PYTHONPATH=src python3 -m core_llm.scripts.export_hf_llama \
  --checkpoint <SFT_RUN>/checkpoints/best.pt \
  --tokenizer <BASE_TOKENIZER>/tokenizer/tokenizer.model \
  --output-dir data/export/llama_sft_lowlr_YYYYMMDD
```

This writes:
`config.json`, `tokenizer.model`, `tokenizer_config.json`, `special_tokens_map.json`,
and `pytorch_model.bin` (HF-style tensor names).

## GGUF conversion (llama.cpp)

```bash
cd ~/llama.cpp
python3 convert_hf_to_gguf.py ~/LLM/core_llm/data/export/llama_sft_lowlr_YYYYMMDD \
  --outfile ~/LLM/core_llm/data/export/llama_sft_lowlr_YYYYMMDD.gguf

./build/bin/llama-quantize \
  ~/LLM/core_llm/data/export/llama_sft_lowlr_YYYYMMDD.gguf \
  ~/LLM/core_llm/data/export/llama_sft_lowlr_YYYYMMDD.Q4_K_M.gguf q4_K_M
```

## Ollama (local)

**Ollama server was crashing with EOF. Root cause: runner crash and context mismatch.**
`num_ctx` must be 256 (model context length), and GPU on MX450 is unstable.

Recommended Modelfile:

```text
FROM /home/aki/LLM/core_llm/data/export/llama_sft_lowlr_YYYYMMDD.Q4_K_M.gguf
TEMPLATE """{{ .Prompt }}"""
PARAMETER temperature 0.2
PARAMETER top_p 0.95
PARAMETER num_ctx 256
PARAMETER num_gpu 0
```

If Ollama still returns `EOF`, run the server manually with CPU only:

```bash
sudo snap stop ollama
OLLAMA_NUM_GPU=0 OLLAMA_LLM_LIBRARY=cpu OLLAMA_CONTEXT_LENGTH=256 ollama serve
```

Then:

```bash
ollama run <modelname> < prompt.txt
```

## Prompt format (important)

The model was trained on **SFT prompt format**:

```
### Instruction
...

### Response
```

If you prompt with raw “こんにちは” it will often degrade.

## Known issues / next steps

1) LLaMA outputs still somewhat garbled when prompted casually.  
   Use SFT prompt format for evaluation.
2) Ollama runner crashes on GPU.  
   Use CPU mode (`num_gpu 0`, manual serve) to avoid EOF.
3) If quality is still weak, increase dataset to 400k and/or model size.

## Repo changes (recent)

- Added LLaMA model + factory routing.
- Added HF export script for LLaMA.
- Added stop support in eval/generate.
- Added lots of short-summary SFT seeds.
- Ctrl+C now sends Discord failure notices.
