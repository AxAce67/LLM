from __future__ import annotations

import argparse
from pathlib import Path

from core_llm.config import load_model_config
from core_llm.data.binary_writer import BinarySplitWriter, write_metadata
from core_llm.data.cleaning import is_usable_text, looks_japanese, normalize_text
from core_llm.data.dedup import is_duplicate
from core_llm.data.manifest_schema import iter_manifest
from core_llm.data.split import assign_split
from core_llm.tokenizer.encode import load_tokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--tokenizer", default="data/tokenizer/tokenizer.model")
    ap.add_argument("--output-dir", default="data/prepared")
    ap.add_argument("--min-chars", type=int, default=80)
    args = ap.parse_args()

    model_config = load_model_config(args.config)
    tokenizer = load_tokenizer(args.tokenizer)
    if tokenizer.get_piece_size() != model_config.vocab_size:
        raise ValueError("Tokenizer vocab size does not match model config vocab_size")

    output_dir = Path(args.output_dir)
    train_writer = BinarySplitWriter(output_dir / "train.bin")
    val_writer = BinarySplitWriter(output_dir / "val.bin")
    seen: set[str] = set()
    kept = filtered = duplicate = 0
    for row in iter_manifest(args.manifest):
        text = normalize_text(row.text)
        if row.lang != "ja" or not row.license or not looks_japanese(text) or not is_usable_text(text, min_chars=args.min_chars):
            filtered += 1
            continue
        if is_duplicate(text, seen):
            duplicate += 1
            continue
        tokens = tokenizer.encode_as_ids(text)
        if not tokens:
            filtered += 1
            continue
        tokens.append(tokenizer.eos_id())
        split = row.split_hint if row.split_hint in {"train", "val"} else assign_split(row.id)
        if split == "val":
            val_writer.write_tokens(tokens)
        else:
            train_writer.write_tokens(tokens)
        kept += 1
    train_writer.close()
    val_writer.close()
    write_metadata(
        output_dir / "metadata.json",
        {
            "kept_docs": kept,
            "filtered_docs": filtered,
            "duplicate_docs": duplicate,
            "train_tokens": train_writer.token_count,
            "val_tokens": val_writer.token_count,
            "vocab_size": tokenizer.get_piece_size(),
        },
    )
    print(f"Prepared dataset in {output_dir}")


if __name__ == "__main__":
    main()
