# core_llm

`core_llm` is a fully decoupled research core for building a small Japanese-first base model from scratch.

## What it is

- Tokenizer training from manifest files
- Dataset preparation into binary token files
- Decoder-only Transformer pretraining
- Local checkpointing and resume
- Validation perplexity evaluation
- CLI-only generation

## What it is not

- No dependency on the legacy crawler, dashboard, DB, or HA control code
- No large-scale web crawling in the initial scope
- No RAG, API server, or chat tuning in the initial scope

## Layout

- `configs/`: default tokenizer/model/train configs
- `data/`: local manifests, tokenizer artifacts, prepared binaries, checkpoints, eval outputs
- `scripts/`: thin wrappers that call the package modules
- `src/core_llm/`: implementation
- `tests/`: unit and small integration tests

## Quickstart

```bash
cd core_llm
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-core.txt
pip install -e .
```

Prepare a manifest from raw `.txt` files:

```bash
python -m core_llm.scripts.prepare_manifest \
  --input-dir data/raw \
  --output data/manifests/train_manifest.jsonl \
  --source local_text \
  --license permissive-user-provided
```

`prepare_manifest` also supports `.md`, source-specific id prefixes, fixed split hints, and a companion report JSON.

Prepare all curated local sources in one step:

```bash
python -m core_llm.scripts.prepare_curated_manifests \
  --raw-root data/raw/curated \
  --manifest-dir data/manifests
```

Expected directories:

- `data/raw/curated/local_notes_ja/`
- `data/raw/curated/tech_docs_ja/`
- `data/raw/curated/government_ja/`

Fetch `government_ja` texts from seed URLs:

```bash
python -m core_llm.scripts.fetch_government_corpus \
  --seed-file data/seed_urls/government_ja.txt \
  --output-dir data/raw/curated/government_ja
```

Refresh the seed URL file automatically from the Digital Agency sitemap:

```bash
python -m core_llm.scripts.discover_government_seed_urls \
  --output data/seed_urls/government_ja.txt \
  --limit 150
```

Prepare a manifest directly from Japanese Wikipedia:

```bash
python -m core_llm.scripts.prepare_wikipedia_manifest \
  --lang ja \
  --output data/manifests/wikipedia_ja.jsonl
```

This downloads and caches `jawiki-latest-pages-articles.xml.bz2` under `data/raw/wikipedia/`.
The dump is large, so make sure you have enough disk space before running it.

Train the tokenizer:

```bash
python -m core_llm.scripts.train_tokenizer \
  --config configs/tokenizer_ja_base.yaml \
  --manifest data/manifests/train_manifest.jsonl
```

For large manifests such as `wikipedia_ja_200k.jsonl`, prefer the sampled tokenizer config to avoid exhausting memory during SentencePiece training:

```bash
python -m core_llm.scripts.train_tokenizer \
  --config configs/tokenizer_ja_large_sample_safe.yaml \
  --manifest data/manifests/wikipedia_ja_200k.jsonl
```

Prepare the dataset:

```bash
python -m core_llm.scripts.prepare_dataset \
  --config configs/model_tiny_ja.yaml \
  --manifest data/manifests/train_manifest.jsonl
```

Train:

```bash
python -m core_llm.scripts.train \
  --config configs/model_tiny_ja.yaml \
  --train-config configs/train_local_cpu.yaml
```

Evaluate:

```bash
python -m core_llm.scripts.eval_perplexity \
  --checkpoint data/checkpoints/latest.pt
```

Generate:

```bash
python -m core_llm.scripts.generate \
  --checkpoint data/checkpoints/latest.pt \
  --prompt "人工知能とは"
```

Prepare an SFT manifest from a QA JSONL:

```bash
python -m core_llm.scripts.prepare_sft_manifest \
  --input data/raw/sft/qa_seed.jsonl \
  --output data/manifests/sft_ja.jsonl
```

Train SFT from a pretrained base checkpoint:

```bash
python -m core_llm.scripts.train_sft \
  --base-checkpoint data/runs/wiki_small_100k_30k_bs4/checkpoints/best.pt \
  --tokenizer data/runs/wiki_small_100k_30k_bs4/tokenizer/tokenizer.model \
  --manifest data/manifests/sft_ja.jsonl \
  --train-config configs/train_sft_small_sample.yaml \
  --work-dir data/runs/sft_small_sample
```

Run a fixed prompt set against a base or SFT checkpoint:

```bash
python -m core_llm.scripts.evaluate_prompt_set \
  --checkpoint data/runs/sft_small_sample/checkpoints/best.pt \
  --tokenizer data/runs/wiki_small_100k_30k_bs4/tokenizer/tokenizer.model \
  --questions data/raw/sft/eval_questions_ja.jsonl \
  --output data/eval/sft_small_sample_eval.jsonl
```

Run a tiny Wikipedia sample training:

```bash
python -m core_llm.scripts.run_wiki_tiny \
  --work-dir data/runs/wiki_tiny_sample
```

This is a sample run, not a full Wikipedia training run.
All generated artifacts are stored under the given `work-dir`.

Send a Discord notification when a run finishes:

```bash
cat > .env.local <<'EOF'
DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/..."
DISCORD_MENTION="<@123456789012345678>"
EOF

python -m core_llm.scripts.run_wiki_tiny \
  --work-dir data/runs/wiki_tiny_sample
```

`run_wiki_tiny` and `run_pretrain_mix` load `core_llm/.env.local` automatically if it exists.

Run a mixed-source tiny sample training:

```bash
python -m core_llm.scripts.run_pretrain_mix \
  --work-dir data/runs/pretrain_mix_sample \
  --manifest data/manifests/wikipedia_ja.jsonl \
  --manifest data/manifests/government_ja.jsonl
```

This keeps the same artifact layout as `run_wiki_tiny`, but trains on a merged curated manifest.

Merge multiple curated manifest sources:

```bash
python -m core_llm.scripts.merge_manifests \
  --input data/manifests/wikipedia_ja.jsonl \
  --input data/manifests/tech_docs_ja.jsonl \
  --output data/manifests/pretrain_mix_ja.jsonl
```

Index or compare runs:

```bash
python -m core_llm.scripts.index_runs --runs-dir data/runs
python -m core_llm.scripts.compare_runs \
  --run data/runs/wiki_tiny_sample \
  --run data/runs/wiki_tiny_sample_2
```

## Data policy

The initial implementation is intentionally strict:

- `lang` must be `ja`
- `license` must be non-empty
- short, duplicate, or URL-heavy samples are filtered out
- the initial intended sources are permissive and curated
- Japanese Wikipedia dump can be ingested directly into a manifest
- multiple manifest sources can be merged into one pretraining mix

## Legacy relation

This directory is separate from the legacy operational system in the repo. The old crawler/dashboard/DB stack remains unchanged and is not imported by `core_llm`.
