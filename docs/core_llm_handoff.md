# core_llm handoff / runbook

This document captures the current state and the exact commands to continue work
on the Japanese small LLM experiments (pretrain + SFT). It is written for a new
agent to pick up without ambiguity.

## Goal

Improve QA behavior. SFT helps format, but base model strength is the current
bottleneck. The priority is to strengthen base pretraining, then re-run SFT.

## Environment

- Workdir: `~/LLM/core_llm`
- Activate venv: `source ../venv/bin/activate`
- Use `python3` (no `python` on the machine)
- Always set `PYTHONPATH=src`

## Important mechanics

- `train` and `train_sft` resume automatically if `latest.pt` exists.
- For **fresh comparisons**, always use a **new** `--work-dir`.
- For **resume**, keep the same `--checkpoint-dir` and only increase
  `total_steps` in the train config.
- Do not change model config or tokenizer when resuming; checkpoints are
  incompatible otherwise.

## Current best checkpoint (base)

- Base run: `data/runs/wiki_small_200k_30k`
- Evaluation:
  - `val_perplexity`: `64.60`
- Previous best for reference:
  - `data/runs/wiki_small_100k_30k_bs4` -> `val_perplexity: 65.05`
- Use:
  - Checkpoint: `data/runs/wiki_small_200k_30k/checkpoints/best.pt`
  - Tokenizer: `data/runs/wiki_small_200k_30k/tokenizer/tokenizer.model`

Re-evaluate base:

```bash
PYTHONPATH=src python3 -m core_llm.scripts.eval_perplexity \
  --checkpoint data/runs/wiki_small_200k_30k/checkpoints/best.pt \
  --data-dir data/runs/wiki_small_200k_30k/prepared
```

## Current best SFT result

- Run: `data/runs/sft_small_100k_300_tight_after_46k_fresh`
- `val_perplexity`: `26.69`
- Output is improved in form but still drifts semantically.

## SFT seed data

Seed files:

- `data/raw/sft/qa_seed.jsonl` (300 items, tightened answers)
- `data/raw/sft/qa_seed_core_ja.jsonl` (120 items, focused core set; not good in practice)

Notes:

- The 120-item core set performed worse than the 300-item set.
- Keep using the 300-item set unless a new curated set is built.

## Fresh SFT run template (recommended)

Use the improved base and run SFT in a new workdir:

```bash
PYTHONPATH=src python3 -m core_llm.scripts.prepare_sft_manifest \
  --input data/raw/sft/qa_seed.jsonl \
  --output data/manifests/sft_ja.jsonl

PYTHONPATH=src python3 -m core_llm.scripts.train_sft \
  --base-checkpoint data/runs/wiki_small_200k_30k/checkpoints/best.pt \
  --tokenizer data/runs/wiki_small_200k_30k/tokenizer/tokenizer.model \
  --manifest data/manifests/sft_ja.jsonl \
  --train-config configs/train_sft_small_sample.yaml \
  --work-dir data/runs/sft_small_200k_300_after_base64p6_fresh
```

Check:

```bash
tail -n 20 data/runs/sft_small_200k_300_after_base64p6_fresh/checkpoints/train_metrics.jsonl

PYTHONPATH=src python3 -m core_llm.scripts.generate \
  --checkpoint data/runs/sft_small_200k_300_after_base64p6_fresh/checkpoints/best.pt \
  --tokenizer data/runs/wiki_small_200k_30k/tokenizer/tokenizer.model \
  --prompt "### Instruction\n人工知能とは何ですか？\n\n### Response\n" \
  --max-new-tokens 120
```

Fixed prompt evaluation:

```bash
PYTHONPATH=src python3 -m core_llm.scripts.evaluate_prompt_set \
  --checkpoint data/runs/sft_small_200k_300_after_base64p6_fresh/checkpoints/best.pt \
  --tokenizer data/runs/wiki_small_200k_30k/tokenizer/tokenizer.model \
  --questions data/raw/sft/eval_questions_ja.jsonl \
  --output data/eval/sft_eval_200k_300_after_base64p6_fresh.jsonl
```

## Next recommended step (SFT refresh)

Adopt `wiki_small_200k_30k` as the base and run a fresh SFT comparison with the
300-item tightened seed set. Compare it against the previous SFT best
(`data/runs/sft_small_100k_300_tight_after_46k_fresh`, `val_perplexity: 26.69`)
using fixed prompt evaluation.

```bash
PYTHONPATH=src python3 -m core_llm.scripts.prepare_sft_manifest \
  --input data/raw/sft/qa_seed.jsonl \
  --output data/manifests/sft_ja.jsonl

PYTHONPATH=src python3 -m core_llm.scripts.train_sft \
  --base-checkpoint data/runs/wiki_small_200k_30k/checkpoints/best.pt \
  --tokenizer data/runs/wiki_small_200k_30k/tokenizer/tokenizer.model \
  --manifest data/manifests/sft_ja.jsonl \
  --train-config configs/train_sft_small_sample.yaml \
  --work-dir data/runs/sft_small_200k_300_after_base64p6_fresh

PYTHONPATH=src python3 -m core_llm.scripts.evaluate_prompt_set \
  --checkpoint data/runs/sft_small_200k_300_after_base64p6_fresh/checkpoints/best.pt \
  --tokenizer data/runs/wiki_small_200k_30k/tokenizer/tokenizer.model \
  --questions data/raw/sft/eval_questions_ja.jsonl \
  --output data/eval/sft_eval_200k_300_after_base64p6_fresh.jsonl
```

## Known pitfalls

- If `train_sft` finishes with `step` greater than expected, you reused a
  workdir and resumed by accident. Use a new `--work-dir`.
- `data/eval/*.jsonl` is not auto-updated; always run `evaluate_prompt_set`.
- `eval/perplexity.json` is stale unless `eval_perplexity` is run.

## Git notes

Recent commits:

- `ef386bf` Add focused core SFT seed set (`qa_seed_core_ja.jsonl`)
- `73051f1` Tighten SFT seed answers for QA quality
- `998a94f` Improve SFT seed QA quality and add lint

