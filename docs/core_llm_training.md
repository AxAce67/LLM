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

## Sample artifact layout

- `manifests/`
- `tokenizer/`
- `prepared/`
- `checkpoints/`
- `eval/`
- `run_summary.json`

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
