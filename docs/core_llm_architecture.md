# `core_llm` Architecture

## Goal

`core_llm` は、自作 base model の研究を目的とした最小構成です。

対象:

- tokenizer 学習
- manifest ベースの前処理
- binary dataset 作成
- decoder-only transformer pretraining
- checkpoint / resume
- perplexity 評価
- CLI 推論

対象外:

- dashboard
- crawler
- DB
- HA
- RAG
- API server

## Top-level flow

```text
raw text -> manifest -> tokenizer -> tokenized binaries -> train -> checkpoints -> eval / generate
```

## Directories

- `core_llm/configs/`: tokenizer / model / train config
- `core_llm/data/`: local artifacts
- `core_llm/src/core_llm/`: implementation
- `core_llm/tests/`: unit + small integration tests

## Runtime model

- single-machine first
- CPU / CUDA / MPS support
- CUDA only for AMP
- no distributed training in initial scope

## Separation from legacy

`core_llm` does not import anything from `legacy/`.
