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

## Interpretation

- lower perplexity is better
- smoke generation is qualitative only
- the initial target is correctness of the pipeline, not strong general-purpose quality
- inspect `run_summary.json` first after a sample run
