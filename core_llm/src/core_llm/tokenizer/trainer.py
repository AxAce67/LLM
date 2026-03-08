from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import sentencepiece as spm

from core_llm.config import TokenizerConfig, dump_dataclass_jsonable
from core_llm.data.manifest_schema import iter_manifest


def resolve_tokenizer_threads(config: TokenizerConfig) -> int:
    if config.num_threads > 0:
        return config.num_threads
    return max(1, os.cpu_count() or 1)


def train_tokenizer(
    manifest_path: str | Path,
    output_dir: str | Path,
    config: TokenizerConfig,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_prefix = output_dir / "tokenizer"
    fd, temp_path = tempfile.mkstemp(suffix=".txt")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for row in iter_manifest(manifest_path):
                f.write(row.text)
                f.write("\n\n")
        spm.SentencePieceTrainer.train(
            input=temp_path,
            model_prefix=str(model_prefix),
            vocab_size=config.vocab_size,
            model_type=config.model_type,
            character_coverage=config.character_coverage,
            input_sentence_size=config.input_sentence_size,
            shuffle_input_sentence=config.shuffle_input_sentence,
            pad_id=config.special_tokens["pad_id"],
            unk_id=config.special_tokens["unk_id"],
            bos_id=config.special_tokens["bos_id"],
            eos_id=config.special_tokens["eos_id"],
            num_threads=resolve_tokenizer_threads(config),
            hard_vocab_limit=False,
        )
        meta_path = output_dir / "tokenizer_meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(dump_dataclass_jsonable(config), f, ensure_ascii=False, indent=2)
    finally:
        Path(temp_path).unlink(missing_ok=True)
    return model_prefix.with_suffix(".model")
