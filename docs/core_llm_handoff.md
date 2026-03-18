# core_llm handoff / runbook

This document captures the current state and exact commands to continue the
Japanese small LLM experiments (pretrain + SFT). It is meant for handoff to a
new session without ambiguity.

## Goal

Build a more general QA-capable Japanese LLM. The current bottlenecks are:

1) Short-form answer control (especially comparison/summary tasks).
2) Evaluation stability (small eval set and strict heuristics).

Base pretraining is now at 100k docs; SFT is being tuned to hit the eval set.

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

## Current best base run (pretrain)

- Base run:
  `data/runs/wiki_tiny_sample_20260317_115406_model_small_ja_sample_train_small_sample_30k_docs100000__step18000of30000__early`
- Best val perplexity: `52.47` (early stopped at 18k / 30k)
- Tokenizer:
  `.../tokenizer/tokenizer.model`
- Checkpoint:
  `.../checkpoints/best.pt`

Re-evaluate base:

```bash
PYTHONPATH=src python3 -m core_llm.scripts.eval_perplexity \
  --checkpoint data/runs/wiki_tiny_sample_20260317_115406_model_small_ja_sample_train_small_sample_30k_docs100000__step18000of30000__early/checkpoints/best.pt \
  --data-dir data/runs/wiki_tiny_sample_20260317_115406_model_small_ja_sample_train_small_sample_30k_docs100000__step18000of30000__early/prepared
```

## Current best SFT baseline

We evaluate SFT using **short-answer settings** to avoid runaway generations.
This improved summary scores from 0.0 to 0.5, but comparison stayed at 0.0
because the evaluation requires explicit words like `違い`.

Recommended eval settings:

```bash
--max-new-tokens 64 --temperature 0.2 --repetition-penalty 1.2 --stop "。"
```

## SFT seed data (current)

- `core_llm/data/raw/sft/qa_seed.jsonl` expanded to **~870 items**.
- Many variants were added for summary/comparison and short-answer control.
- Comparison scoring uses string checks; SFT answers now include explicit
  `違いは...` phrasing to satisfy evaluation.

Lint seed data:

```bash
PYTHONPATH=src python3 -m core_llm.scripts.lint_sft_seed \
  --input data/raw/sft/qa_seed.jsonl
```

## Standard SFT run template (current)

```bash
PYTHONPATH=src python3 -m core_llm.scripts.prepare_sft_manifest \
  --input data/raw/sft/qa_seed.jsonl \
  --output data/manifests/sft_ja.jsonl

PYTHONPATH=src python3 -m core_llm.scripts.train_sft \
  --base-checkpoint data/runs/wiki_tiny_sample_20260317_115406_model_small_ja_sample_train_small_sample_30k_docs100000__step18000of30000__early/checkpoints/best.pt \
  --tokenizer data/runs/wiki_tiny_sample_20260317_115406_model_small_ja_sample_train_small_sample_30k_docs100000__step18000of30000__early/tokenizer/tokenizer.model \
  --manifest data/manifests/sft_ja.jsonl \
  --train-config configs/train_sft_small_sample.yaml
```

Evaluate (short + stop):

```bash
PYTHONPATH=src python3 -m core_llm.scripts.evaluate_prompt_set \
  --checkpoint <SFT_RUN>/checkpoints/best.pt \
  --tokenizer data/runs/wiki_tiny_sample_20260317_115406_model_small_ja_sample_train_small_sample_30k_docs100000__step18000of30000__early/tokenizer/tokenizer.model \
  --questions data/raw/sft/eval_questions_ja.jsonl \
  --output data/eval/sft_eval_clean_100k_<TAG>_stop.jsonl \
  --max-new-tokens 64 \
  --temperature 0.2 \
  --repetition-penalty 1.2 \
  --stop "。"
```

## Why comparison was stuck at 0.0

`evaluate_prompt_set.py` marks comparison as correct only if response contains
one of: `違い`, `一方`, `それぞれ`, `対して`, `比較`. Short answers without these
words are graded as 0 even if correct. Therefore SFT data now uses
`違いは...` phrasing.

## New CLI support

`evaluate_prompt_set` and `generate` now accept:

```
--stop "。"
```

The stop term is trimmed from the output; this is essential to enforce
single-sentence comparisons.

## Next recommended steps

1) Re-run SFT with the latest seed (explicit `違い` answers).
2) Evaluate with short settings + stop.
3) If comparison improves, lock this as the eval standard.
4) If general capability is still weak, scale base to 200k docs / 50k steps.

## Known pitfalls

- SFT can resume unintentionally if a workdir is reused.
- `evaluate_prompt_set` defaults (long) cause runaway generations.
- Comparison scoring is string-based; ensure `違い` appears.

## Git notes (recent)

- `ae07c83` Add stop sequence trimming to eval and generate
- `58ec6b9` Add one-sentence comparison SFT variants
- `427d80c` Add eval-matched summary/comparison SFT variants
