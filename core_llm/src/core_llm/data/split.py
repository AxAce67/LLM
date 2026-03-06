from __future__ import annotations

import hashlib


def assign_split(doc_id: str, val_ratio: float = 0.05) -> str:
    digest = hashlib.sha1(doc_id.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    return "val" if bucket < val_ratio else "train"
