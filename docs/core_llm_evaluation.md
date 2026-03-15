# `core_llm` Evaluation Policy

## Primary metric

- validation perplexity

## Secondary check

- smoke generation prompts

## Non-goals

- no keyword fuzzy score
- no DB-backed evaluation runs
- no product benchmark claims

## Current output

- `core_llm/data/eval/perplexity.json`
- sample run output: `<work-dir>/eval/perplexity.json`
- sample run summary: `<work-dir>/run_summary.json`
- `evaluate_prompt_set` outputs `<path>.summary.json` with heuristic QA metrics

## Interpretation

- lower perplexity is better
- smoke generation is qualitative only
- the initial target is correctness of the pipeline, not strong general-purpose quality
- inspect `run_summary.json` first after a sample run
- compare multiple `run_summary.json` files when you change data mixes or configs

## Prompt-set evaluation

```bash
python3 -m core_llm.scripts.evaluate_prompt_set \
  --checkpoint <work-dir>/checkpoints/best.pt \
  --tokenizer <work-dir>/tokenizer/tokenizer.model \
  --questions data/raw/sft/eval_questions_ja.jsonl \
  --output data/eval/sft_eval.jsonl
```

Key fields in `*.summary.json`:

- `qa_ok_rate`: heuristic correctness proxy
- `avg_repeat_trigram_ratio`: repetition indicator
- `avg_symbol_ratio` / `avg_latin_ratio`: noise indicators
- `category_stats`: per-category QA OK rate
