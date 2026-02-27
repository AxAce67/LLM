import hashlib
import re
from typing import Set


def normalize_text(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_low_quality(text: str, min_chars: int = 120) -> bool:
    if not text or len(text) < min_chars:
        return True
    unique_ratio = len(set(text)) / max(1, len(text))
    if unique_ratio < 0.05:
        return True
    url_like = text.count("http://") + text.count("https://")
    if url_like > 10:
        return True
    return False


def text_fingerprint(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def is_duplicate(text: str, seen_hashes: Set[str]) -> bool:
    fp = text_fingerprint(text)
    if fp in seen_hashes:
        return True
    seen_hashes.add(fp)
    return False


def quality_score(text: str) -> float:
    """
    0.0 - 1.0 の簡易品質スコア。
    情報密度・文字多様性・ノイズ率（URL過多/極端な短文）を組み合わせる。
    """
    t = normalize_text(text or "")
    if not t:
        return 0.0

    length = len(t)
    unique_ratio = len(set(t)) / max(1, length)
    line_count = max(1, t.count("\n") + 1)
    avg_line_len = length / line_count
    url_count = t.count("http://") + t.count("https://")

    length_component = min(1.0, length / 1400.0)
    diversity_component = min(1.0, unique_ratio / 0.22)
    structure_component = min(1.0, avg_line_len / 120.0)
    noise_penalty = min(0.5, url_count * 0.03)

    score = (0.45 * length_component) + (0.35 * diversity_component) + (0.20 * structure_component) - noise_penalty
    return max(0.0, min(1.0, score))
