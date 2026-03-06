# Legacy Operational System

このディレクトリは旧運用系の退避先です。

含まれるもの:

- crawler
- dashboard
- PostgreSQL 依存の収集管理
- HA / remote control
- Docker / setup ベースの運用コード

## Status

- 新規開発対象ではありません
- 保守・参照・退避のために残しています
- 新しい研究開発は `../core_llm/` で行います

## Running the old system

旧コードを使う場合は、このディレクトリを基準に実行してください。

例:

```bash
cd legacy
source ../venv/bin/activate
python app.py
```

Compose 検証や旧 Docker 実行もこのディレクトリ基準で行います。
