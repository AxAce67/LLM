# `core_llm` 構成

English: [core_llm_architecture.md](core_llm_architecture.md)

## 目的

`core_llm` は、小型 base model を自前で構築するための最小構成の研究スタックです。

対象:

- tokenizer 学習
- manifest ベースの前処理
- binary dataset 作成
- decoder-only transformer の pretraining
- checkpoint / resume
- perplexity 評価
- CLI 推論
- run summary / run log

対象外:

- dashboard
- crawler
- DB
- HA
- RAG
- API server

## 全体フロー

```text
raw text -> manifest -> tokenizer -> tokenized binaries -> train -> checkpoints -> eval / generate
```

## ディレクトリ

- `core_llm/configs/`: tokenizer / model / train config
- `core_llm/data/`: local artifacts
- `core_llm/src/core_llm/`: implementation
- `core_llm/tests/`: unit + small integration tests
- `core_llm/data/runs/`: run summaries + logs

## ランタイム方針

- single-machine first
- CPU / CUDA / MPS support
- CUDA only for AMP
- no distributed training in initial scope

## 旧系統との分離

`core_llm` は `legacy/` を参照しません。
