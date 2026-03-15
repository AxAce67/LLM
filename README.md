# Self-Built LLM Research Repo

[日本語版はこちら](README.ja.md)

This repository focuses on building a small Japanese-centric base model end-to-end: tokenizer, pretraining, and SFT.  
Active development lives in [`core_llm/`](core_llm). Legacy infrastructure is frozen under [`legacy/`](legacy).

## Structure

- [`core_llm/`](core_llm): main research codebase
- [`docs/`](docs): architecture, data format, training, evaluation
- [`legacy/`](legacy): frozen legacy system

## Quickstart

```bash
cd core_llm
source ../venv/bin/activate

PYTHONPATH=src python3 -m core_llm.scripts.train_tokenizer \
  --config configs/tokenizer_ja_base.yaml \
  --manifest data/manifests/train_manifest.jsonl

PYTHONPATH=src python3 -m core_llm.scripts.prepare_dataset \
  --config configs/model_tiny_ja.yaml \
  --manifest data/manifests/train_manifest.jsonl

PYTHONPATH=src python3 -m core_llm.scripts.train \
  --config configs/model_tiny_ja.yaml \
  --train-config configs/train_local_cpu.yaml
```

## Run Tracking

- Each run produces a `run_summary.json` and a `run_log.jsonl`.
- A global log is written to `data/runs/run_log.jsonl` (daily rotated as `run_log_YYYYMMDD.jsonl`).
- Run directories are auto-renamed to include run labels (steps/resume/early stop).

## Evaluation

- `evaluate_prompt_set` writes `*.summary.json` with heuristic QA metrics.
- Use `qa_ok_rate` and category stats to compare runs quickly.

## Testing

```bash
./venv/bin/python -m pytest -q
python3 -m compileall core_llm/src/core_llm
```

## Legacy

Legacy code is kept for reference only:

- [`legacy/app.py`](legacy/app.py)
- [`legacy/main_controller.py`](legacy/main_controller.py)
- [`legacy/docker-compose.yml`](legacy/docker-compose.yml)

## Docs

- [`docs/core_llm_architecture.md`](docs/core_llm_architecture.md)
- [`docs/core_llm_data.md`](docs/core_llm_data.md)
- [`docs/core_llm_training.md`](docs/core_llm_training.md)
- [`docs/core_llm_evaluation.md`](docs/core_llm_evaluation.md)
