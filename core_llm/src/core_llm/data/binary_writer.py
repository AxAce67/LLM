from __future__ import annotations

import json
from array import array
from pathlib import Path


class BinarySplitWriter:
    def __init__(self, path: str | Path, flush_threshold: int = 200_000):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_bytes(b"")
        self.flush_threshold = flush_threshold
        self.buffer = array("H")
        self.token_count = 0
        self.doc_count = 0

    def write_tokens(self, tokens: list[int]) -> None:
        if not tokens:
            return
        self.buffer.extend(tokens)
        self.token_count += len(tokens)
        self.doc_count += 1
        if len(self.buffer) >= self.flush_threshold:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        with open(self.path, "ab") as f:
            self.buffer.tofile(f)
        self.buffer = array("H")

    def close(self) -> None:
        self.flush()


def write_metadata(path: str | Path, payload: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
