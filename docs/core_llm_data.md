# `core_llm` Data Format

## Manifest schema

Each line is a JSON object.

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

## Rules

- `lang` must be `ja` in the current phase
- `license` must be non-empty
- short text is filtered
- duplicate text is filtered
- URL-heavy text is filtered

## Local source manifest preparation

- `prepare_manifest` scans `.txt` and `.md` by default
- use `--include-ext` to restrict file types
- use `--id-prefix` to keep source-specific ids stable
- use `--split-hint train|val|auto` when preparing fixed validation sets
- a `*.report.json` file is written alongside the manifest by default
- `prepare_curated_manifests` builds all known curated source manifests under one raw root
- current curated presets are `local_notes_ja`, `tech_docs_ja`, and `government_ja`
- `fetch_government_corpus` can populate `data/raw/curated/government_ja/` from an allowlisted seed URL file
- `discover_government_seed_urls` can rebuild `data/seed_urls/government_ja.txt` from the Digital Agency sitemap

## Dataset artifacts

- `core_llm/data/prepared/train.bin`
- `core_llm/data/prepared/val.bin`
- `core_llm/data/prepared/metadata.json`

## Tokenizer artifacts

- `core_llm/data/tokenizer/tokenizer.model`
- `core_llm/data/tokenizer/tokenizer.vocab`
- `core_llm/data/tokenizer/tokenizer_meta.json`

## Wikipedia raw cache

- `core_llm/data/raw/wikipedia/jawiki-latest-pages-articles.xml.bz2`

## Wikipedia manifest semantics

- `source`: `wikipedia_ja`
- `license`: `cc-by-sa-4.0`
- `id`: `jawiki:<page_id>`

## Wikipedia report file

- output manifest alongside `*.report.json`
- includes kept and filtered counters

## Multi-source manifest workflow

- create one manifest per source first
- merge them with `python -m core_llm.scripts.merge_manifests`
- merged report includes `source_counts` and `license_counts`
- duplicate text across sources is removed before dataset preparation
- `python -m core_llm.scripts.run_pretrain_mix` runs merge, tokenizer, dataset, train, and eval in one flow
