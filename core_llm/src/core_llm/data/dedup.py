from __future__ import annotations

import hashlib


def fingerprint(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def is_duplicate(text: str, seen: set[str]) -> bool:
    fp = fingerprint(text)
    if fp in seen:
        return True
    seen.add(fp)
    return False
