# `core_llm` 学習フロー

English: [core_llm_training.md](core_llm_training.md)

## 手順

1. manifest 準備
2. tokenizer 学習
3. dataset 準備
4. 学習
5. perplexity 評価
6. 生成のスモーク確認

## 推奨の最初の流れ

1. `prepare_wikipedia_manifest`
2. `train_tokenizer`
3. `prepare_dataset`
4. `train`

## サンプル実行

```bash
python3 -m core_llm.scripts.run_wiki_tiny
```

## Discord 通知（任意）

`core_llm/.env.local` に `DISCORD_WEBHOOK_URL` を入れると、
`run_wiki_tiny` と `run_pretrain_mix` が完了通知を送ります。
`DISCORD_MENTION` を入れるとメンションを固定できます。

## 複数ソースのサンプル

```bash
python3 -m core_llm.scripts.discover_government_seed_urls \
  --output data/seed_urls/government_ja.txt \
  --limit 150

python3 -m core_llm.scripts.fetch_government_corpus \
  --seed-file data/seed_urls/government_ja.txt \
  --output-dir data/raw/curated/government_ja

python3 -m core_llm.scripts.prepare_curated_manifests \
  --raw-root data/raw/curated \
  --manifest-dir data/manifests

python3 -m core_llm.scripts.run_pretrain_mix \
  --manifest data/manifests/wikipedia_ja.jsonl \
  --manifest data/manifests/government_ja.jsonl
```

## 生成物のレイアウト

- `manifests/`
- `tokenizer/`
- `prepared/`
- `checkpoints/`
- `eval/`
- `run_summary.json`
- `run_log.jsonl`

`run_pretrain_mix` も同じ構成で、manifest だけ `pretrain_mix_ja.jsonl` になります。

## マルチソースの流れ

1. ソースごとに manifest を作る
2. `merge_manifests` で統合
3. 統合 manifest で tokenizer を学習
4. dataset 準備
5. 学習と比較

```bash
python3 -m core_llm.scripts.merge_manifests \
  --input data/manifests/wikipedia_ja.jsonl \
  --input data/manifests/local_notes_ja.jsonl \
  --output data/manifests/pretrain_mix_ja.jsonl
```

## 注意

サンプル configs は小さく設定しています。まずは動作確認用途で使用し、スケールは後から。

## チェックポイント

- `<work-dir>/checkpoints/latest.pt`
- `<work-dir>/checkpoints/best.pt`
- `<work-dir>/checkpoints/train_metrics.jsonl`

## チェックポイント内容

- step
- model_config
- train_config
- model_state_dict
- optimizer_state_dict
- scheduler_state_dict
- best_val_perplexity
- latest_train_loss
- rng_state

## 検証

- vocab 不一致はエラー
- dataset が小さすぎる場合はエラー
- tokenizer 不在はエラー
- primary metric は val perplexity

## Run 管理

- `run_summary.json` に configs と主要指標を保存
- `python3 -m core_llm.scripts.index_runs` で一覧
- `python3 -m core_llm.scripts.compare_runs --run <run-dir> --run <run-dir>` で比較
- `python3 -m core_llm.scripts.show_run_log --limit 50` で直近ログ確認

## SFT

```bash
python3 -m core_llm.scripts.prepare_sft_manifest \
  --input data/raw/sft/qa_seed.jsonl \
  --output data/manifests/sft_ja.jsonl

python3 -m core_llm.scripts.train_sft \
  --base-checkpoint <work-dir>/checkpoints/best.pt \
  --tokenizer <work-dir>/tokenizer/tokenizer.model \
  --manifest data/manifests/sft_ja.jsonl \
  --train-config configs/train_sft_small_sample.yaml
```
