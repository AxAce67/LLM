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

## Dataset artifacts

- `core_llm/data/prepared/train.bin`
- `core_llm/data/prepared/val.bin`
- `core_llm/data/prepared/metadata.json`

## Tokenizer artifacts

- `core_llm/data/tokenizer/tokenizer.model`
- `core_llm/data/tokenizer/tokenizer.vocab`
- `core_llm/data/tokenizer/tokenizer_meta.json`
