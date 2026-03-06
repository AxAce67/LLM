# `core_llm` Training Flow

## Sequence

1. Prepare manifest
2. Train tokenizer
3. Prepare dataset
4. Train model
5. Evaluate perplexity
6. Generate smoke outputs

## Recommended first workflow

1. `prepare_wikipedia_manifest`
2. `train_tokenizer`
3. `prepare_dataset`
4. `train`

## Sample run command

```bash
python -m core_llm.scripts.run_wiki_tiny \
  --work-dir data/runs/wiki_tiny_sample
```

## Mixed-source sample run

```bash
python -m core_llm.scripts.discover_government_seed_urls \
  --output data/seed_urls/government_ja.txt \
  --limit 150

python -m core_llm.scripts.fetch_government_corpus \
  --seed-file data/seed_urls/government_ja.txt \
  --output-dir data/raw/curated/government_ja

python -m core_llm.scripts.prepare_curated_manifests \
  --raw-root data/raw/curated \
  --manifest-dir data/manifests

python -m core_llm.scripts.run_pretrain_mix \
  --work-dir data/runs/pretrain_mix_sample \
  --manifest data/manifests/wikipedia_ja.jsonl \
  --manifest data/manifests/government_ja.jsonl
```

## Sample artifact layout

- `manifests/`
- `tokenizer/`
- `prepared/`
- `checkpoints/`
- `eval/`
- `run_summary.json`

The same layout is used for `run_pretrain_mix`, except the manifest is `pretrain_mix_ja.jsonl`.

## Multi-source workflow

1. prepare source manifests independently
2. merge them into one curated training manifest
3. train tokenizer on the merged manifest
4. prepare dataset
5. train and compare runs

```bash
python -m core_llm.scripts.merge_manifests \
  --input data/manifests/wikipedia_ja.jsonl \
  --input data/manifests/local_notes_ja.jsonl \
  --output data/manifests/pretrain_mix_ja.jsonl
```

## Note

The sample configs are intentionally smaller than the default longer-run configs.
Use them to validate the workflow before scaling up.

## Checkpoints

- `core_llm/data/checkpoints/latest.pt`
- `core_llm/data/checkpoints/best.pt`
- `core_llm/data/checkpoints/train_metrics.jsonl`

## Checkpoint contents

- step
- model_config
- train_config
- model_state_dict
- optimizer_state_dict
- scheduler_state_dict
- best_val_perplexity
- latest_train_loss
- rng_state

## Validation

- vocab mismatch is an error
- too-small dataset is an error
- missing tokenizer is an error
- val perplexity is the primary metric

## Run management

- `run_summary.json` stores config snapshots and key metrics for one run
- `python -m core_llm.scripts.index_runs --runs-dir data/runs` builds a comparable run list
- `python -m core_llm.scripts.compare_runs --run <run-dir> --run <run-dir>` compares selected runs
