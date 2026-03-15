# 自作 LLM 研究リポジトリ

English version: [README.md](README.md)

このリポジトリは、日本語中心の小型 base model を end-to-end（tokenizer / pretraining / SFT）で構築するための研究コードです。  
開発の中心は [`core_llm/`](core_llm) に集約し、旧系統は [`legacy/`](legacy) に凍結しています。

## 構成

- [`core_llm/`](core_llm): 研究・実装の本体
- [`docs/`](docs): 構成・データ仕様・学習・評価の文書
- [`legacy/`](legacy): 旧運用系（参照専用）

## クイックスタート

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

## 実行ログと整理

- 各 run に `run_summary.json` と `run_log.jsonl` を出力します。
- グローバルログは `data/runs/run_log.jsonl`（日次で `run_log_YYYYMMDD.jsonl` も作成）。
- run ディレクトリは `run_label`（step/resume/early stop）に合わせて自動リネームされます。

## 評価

- `evaluate_prompt_set` は `*.summary.json` に簡易 QA 指標を出力します。
- `qa_ok_rate` とカテゴリ別統計で粗い比較が可能です。

## テスト

```bash
./venv/bin/python -m pytest -q
python3 -m compileall core_llm/src/core_llm
```

## 旧系統

旧運用系は参照のみで凍結済みです。

- [`legacy/app.py`](legacy/app.py)
- [`legacy/main_controller.py`](legacy/main_controller.py)
- [`legacy/docker-compose.yml`](legacy/docker-compose.yml)

## 関連ドキュメント

- [`docs/core_llm_architecture.md`](docs/core_llm_architecture.md)
- [`docs/core_llm_data.md`](docs/core_llm_data.md)
- [`docs/core_llm_training.md`](docs/core_llm_training.md)
- [`docs/core_llm_evaluation.md`](docs/core_llm_evaluation.md)
