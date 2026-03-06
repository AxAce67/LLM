# Self-Built LLM Research Repo

このリポジトリの主系統は、[`core_llm`](\/Users/Akihiro/llm/build-llm-from-scratch/core_llm) です。  
目的は、**tokenizer から pretraining まで自前で構築する日本語中心の小型 base model** を研究・実装することです。

## Current status

- 主系統: `core_llm/`
- 旧系統: crawler / dashboard / DB / HA ベースの運用コード
- 方針: 旧系統は `legacy` 扱いで凍結し、新規開発は `core_llm` に集約する

## Primary entrypoint

作業を始めるなら、まず [`core_llm/README.md`](\/Users/Akihiro/llm/build-llm-from-scratch/core_llm/README.md) を見てください。

主な実行コマンド:

```bash
cd core_llm
source ../venv/bin/activate

PYTHONPATH=src python -m core_llm.scripts.train_tokenizer \
  --config configs/tokenizer_ja_base.yaml \
  --manifest data/manifests/train_manifest.jsonl

PYTHONPATH=src python -m core_llm.scripts.prepare_dataset \
  --config configs/model_tiny_ja.yaml \
  --manifest data/manifests/train_manifest.jsonl

PYTHONPATH=src python -m core_llm.scripts.train \
  --config configs/model_tiny_ja.yaml \
  --train-config configs/train_local_cpu.yaml
```

## Repository map

- [`core_llm`](\/Users/Akihiro/llm/build-llm-from-scratch/core_llm): 新研究系。今後の本体
- [`docs/migration_to_core_llm.md`](\/Users/Akihiro/llm/build-llm-from-scratch/docs/migration_to_core_llm.md): 移行方針
- [`docs/legacy_system.md`](\/Users/Akihiro/llm/build-llm-from-scratch/docs/legacy_system.md): 旧系統の扱い
- [`docs/core_llm_architecture.md`](\/Users/Akihiro/llm/build-llm-from-scratch/docs/core_llm_architecture.md): 新研究系の構成
- [`docs/core_llm_data.md`](\/Users/Akihiro/llm/build-llm-from-scratch/docs/core_llm_data.md): データ仕様
- [`docs/core_llm_training.md`](\/Users/Akihiro/llm/build-llm-from-scratch/docs/core_llm_training.md): 学習フロー
- [`docs/core_llm_evaluation.md`](\/Users/Akihiro/llm/build-llm-from-scratch/docs/core_llm_evaluation.md): 評価方針
- [`legacy`](\/Users/Akihiro/llm/build-llm-from-scratch/legacy): 旧運用系の退避先

ルート直下には原則として次だけを置く方針です。

- `core_llm/`
- `legacy/`
- `docs/`
- `.github/`
- 最小限の repo 設定ファイル

## Legacy code

旧運用系は現在 [`legacy`](\/Users/Akihiro/llm/build-llm-from-scratch/legacy) に移動済みです。

主な対象:

- [`legacy/app.py`](\/Users/Akihiro/llm/build-llm-from-scratch/legacy/app.py)
- [`legacy/main_controller.py`](\/Users/Akihiro/llm/build-llm-from-scratch/legacy/main_controller.py)
- [`legacy/data_collector`](\/Users/Akihiro/llm/build-llm-from-scratch/legacy/data_collector)
- [`legacy/data_preprocessor`](\/Users/Akihiro/llm/build-llm-from-scratch/legacy/data_preprocessor)
- [`legacy/model`](\/Users/Akihiro/llm/build-llm-from-scratch/legacy/model)
- [`legacy/docker-compose.yml`](\/Users/Akihiro/llm/build-llm-from-scratch/legacy/docker-compose.yml)
- [`legacy/setup.sh`](\/Users/Akihiro/llm/build-llm-from-scratch/legacy/setup.sh)

これらは参照用・退避用です。新しい機能追加は行わない前提です。

## Verification

新研究系の現在の確認コマンド:

```bash
./venv/bin/python -m pytest core_llm/tests -q
python3 -m compileall core_llm/src/core_llm
```

## Next migration steps

1. 旧系統への README 導線を縮小する
2. CI を `core_llm` 中心に寄せる
3. 旧系統の残存資産を `legacy` 基準でさらに縮小する
4. ルートの依存と開発手順を研究系基準に統一する
