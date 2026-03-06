# Legacy System

以下は旧運用系です。

- FastAPI dashboard
- crawler engine
- PostgreSQL ベースの収集管理
- HA / remote control
- Docker / setup ベースの常駐運用

## Why it is legacy

- 目的が「自作 base model の研究」ではなく「収集運用基盤」に寄っている
- `core_llm` と設計思想が異なる
- 今後は新研究系に機能を集約するため

## Legacy scope

- `legacy/app.py`
- `legacy/main_controller.py`
- `legacy/data_collector/`
- `legacy/data_preprocessor/`
- `legacy/model/`
- `legacy/templates/`
- `legacy/docker-compose.yml`
- `legacy/setup.sh`
- `legacy/ops/`
- `legacy/dataset/`
- `legacy/models/`
- `legacy/checkpoints/`
- `legacy/system_status.json`
- `legacy/.env.example`
- `legacy/.dockerignore`

## Rules

- 原則として新機能を追加しない
- セキュリティ修正や退避のための修正だけ許可する
- 新しい学習・評価・推論の開発は `core_llm/` に集約する
- root の標準 `pytest` 対象には含めない
- テストが必要な場合だけ個別に `PYTHONPATH=legacy pytest legacy/tests` のように明示実行する

## Legacy docs

旧文書は `docs/legacy/` に移動済み。
