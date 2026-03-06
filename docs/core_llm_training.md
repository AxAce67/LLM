# `core_llm` Training Flow

## Sequence

1. Prepare manifest
2. Train tokenizer
3. Prepare dataset
4. Train model
5. Evaluate perplexity
6. Generate smoke outputs

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
