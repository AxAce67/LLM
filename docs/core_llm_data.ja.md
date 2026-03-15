# `core_llm` データ仕様

English: [core_llm_data.md](core_llm_data.md)

## Manifest スキーマ

各行が JSON です。

```json
{
  "id": "doc-000001",
  "text": "本文...",
  "lang": "ja",
  "source": "wikipedia_ja",
  "license": "cc-by-sa-4.0",
  "split_hint": "train"
}
```

## ルール

- `lang` は現フェーズで `ja` 固定
- `license` は必須
- 短文は除外
- 重複は除外
- URL 多めの文は除外

## ローカルソースの作成

- `prepare_manifest` は `.txt` と `.md` を対象
- `--include-ext` で拡張子を制限
- `--id-prefix` でソース別 ID を固定化
- `--split-hint train|val|auto` を指定可能
- `*.report.json` が同時出力される
- `prepare_curated_manifests` は curated ソースをまとめて作成
- 現在の curated は `local_notes_ja`, `tech_docs_ja`, `government_ja`
- `fetch_government_corpus` は `data/raw/curated/government_ja/` を構築可能
- `discover_government_seed_urls` で Digital Agency の sitemap から seed を作成

## Dataset 生成物

- `core_llm/data/prepared/train.bin`
- `core_llm/data/prepared/val.bin`
- `core_llm/data/prepared/metadata.json`

## Tokenizer 生成物

- `core_llm/data/tokenizer/tokenizer.model`
- `core_llm/data/tokenizer/tokenizer.vocab`
- `core_llm/data/tokenizer/tokenizer_meta.json`

## Tokenizer 安全設定

- `input_sentence_size=0` は禁止（OOM 回避）
- 例: `1000000` のような cap を使用

## Wikipedia 生データ

- `core_llm/data/raw/wikipedia/jawiki-latest-pages-articles.xml.bz2`

## Wikipedia manifest

- `source`: `wikipedia_ja`
- `license`: `cc-by-sa-4.0`
- `id`: `jawiki:<page_id>`

## Wikipedia report

- manifest と同階層の `*.report.json`
- kept / filtered の集計

## マルチソース workflow

- ソースごとに manifest を作成
- `merge_manifests` で統合
- 統合 report には `source_counts` と `license_counts`
- 重複は統合前に除去
- `run_pretrain_mix` で一括実行
