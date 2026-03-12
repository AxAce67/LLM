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

- Base run: `data/runs/wiki_small_100k_30k_bs4`
- Resume improved base (46.5k steps) evaluation:
  - `val_perplexity` improved from `80.06` to `65.05` (re-evaluated)
- Use:
  - Checkpoint: `data/runs/wiki_small_100k_30k_bs4/checkpoints/best.pt`
  - Tokenizer: `data/runs/wiki_small_100k_30k_bs4/tokenizer/tokenizer.model`

Re-evaluate base:

```bash
PYTHONPATH=src python3 -m core_llm.scripts.eval_perplexity \
  --checkpoint data/runs/wiki_small_100k_30k_bs4/checkpoints/best.pt \
  --data-dir data/runs/wiki_small_100k_30k_bs4/prepared
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
  --base-checkpoint data/runs/wiki_small_100k_30k_bs4/checkpoints/best.pt \
  --tokenizer data/runs/wiki_small_100k_30k_bs4/tokenizer/tokenizer.model \
  --manifest data/manifests/sft_ja.jsonl \
  --train-config configs/train_sft_small_sample.yaml \
  --work-dir data/runs/sft_small_100k_300_tight_after_46k_fresh
```

Check:

```bash
tail -n 20 data/runs/sft_small_100k_300_tight_after_46k_fresh/checkpoints/train_metrics.jsonl

PYTHONPATH=src python3 -m core_llm.scripts.generate \
  --checkpoint data/runs/sft_small_100k_300_tight_after_46k_fresh/checkpoints/best.pt \
  --tokenizer data/runs/wiki_small_100k_30k_bs4/tokenizer/tokenizer.model \
  --prompt "### Instruction\n人工知能とは何ですか？\n\n### Response\n" \
  --max-new-tokens 120
```

Fixed prompt evaluation:

```bash
PYTHONPATH=src python3 -m core_llm.scripts.evaluate_prompt_set \
  --checkpoint data/runs/sft_small_100k_300_tight_after_46k_fresh/checkpoints/best.pt \
  --tokenizer data/runs/wiki_small_100k_30k_bs4/tokenizer/tokenizer.model \
  --questions data/raw/sft/eval_questions_ja.jsonl \
  --output data/eval/sft_eval_100k_300_tight_after_46k_fresh.jsonl
```

## Next recommended step (base expansion)

Resume `wiki_small_200k_30k` to 50k steps and re-evaluate. If
`val_perplexity < 65.05`, adopt it as the new base.

```bash
cp configs/train_small_sample_30k.yaml configs/train_small_sample_50k_200k_resume.yaml
sed -i 's/total_steps: 30000/total_steps: 50000/' configs/train_small_sample_50k_200k_resume.yaml

PYTHONPATH=src python3 -m core_llm.scripts.train \
  --config configs/model_small_ja_sample.yaml \
  --train-config configs/train_small_sample_50k_200k_resume.yaml \
  --data-dir data/runs/wiki_small_200k_30k/prepared \
  --checkpoint-dir data/runs/wiki_small_200k_30k/checkpoints

PYTHONPATH=src python3 -m core_llm.scripts.eval_perplexity \
  --checkpoint data/runs/wiki_small_200k_30k/checkpoints/best.pt \
  --data-dir data/runs/wiki_small_200k_30k/prepared
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

