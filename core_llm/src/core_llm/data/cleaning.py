from __future__ import annotations

import re


URL_RE = re.compile(r"https?://\S+")
JA_CHAR_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]")


def normalize_text(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def url_density(text: str) -> int:
    return len(URL_RE.findall(text or ""))


def looks_japanese(text: str) -> bool:
    cleaned = normalize_text(text)
    if not cleaned:
        return False
    ja_chars = len(JA_CHAR_RE.findall(cleaned))
    return (ja_chars / max(1, len(cleaned))) >= 0.05


def is_usable_text(text: str, min_chars: int = 80, max_urls: int = 8) -> bool:
    cleaned = normalize_text(text)
    if len(cleaned) < min_chars:
        return False
    if url_density(cleaned) > max_urls:
        return False
    return True
