from __future__ import annotations

from pathlib import Path

import sentencepiece as spm


def load_tokenizer(path: str | Path) -> spm.SentencePieceProcessor:
    proc = spm.SentencePieceProcessor()
    ok = proc.load(str(path))
    if not ok:
        raise FileNotFoundError(f"Failed to load tokenizer: {path}")
    return proc


def encode_text(tokenizer: spm.SentencePieceProcessor, text: str) -> list[int]:
    return tokenizer.encode_as_ids(text)


def decode_ids(tokenizer: spm.SentencePieceProcessor, ids: list[int]) -> str:
    return tokenizer.decode_ids(ids)
