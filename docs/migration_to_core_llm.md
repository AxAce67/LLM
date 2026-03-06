# Migration To `core_llm`

## Goal

このリポジトリを、旧運用系から `core_llm` 中心の研究リポジトリへ移行する。

## Decision

- 新規開発は `core_llm` に限定する
- 旧 crawler / dashboard / DB / HA 系は legacy 扱いにする
- 旧系統は即削除せず、段階的に凍結して退避する

## Phase 1

- ルート README を `core_llm` 中心に更新
- `core_llm` の tests / docs / configs を正規入口にする
- CI に `core_llm` のテストを含める

## Phase 2

- 旧系統を `legacy/` 配下へ移動する
- 旧 import / path 前提を壊さないよう参照だけ調整する
- `docker-compose.yml`, `setup.sh`, `app.py` を legacy 明記に変える
- 旧 `dataset/`, `models/`, `checkpoints/`, `.env.example`, `.dockerignore` も `legacy/` へ移す

## Phase 3

- 旧系統の GitHub Actions 対象を縮小する
- 旧 README 由来の運用説明を削除する
- 研究系で使わない依存をルートから整理する

## Do not do during Phase 1

- 旧系統の即時削除
- 旧系統への新機能追加
- `core_llm` と legacy のコード共有

## Exit criteria

- 初見の開発者が README を見て `core_llm` を本体だと理解できる
- `core_llm/tests` が CI で実行される
- 旧系統が legacy として文書化されている
