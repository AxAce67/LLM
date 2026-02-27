import hashlib
import re
from typing import Set


_JA_CHAR_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]")
_EMAIL_RE = re.compile(r"\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}\b")
_PHONE_RE = re.compile(r"\b(?:\+?\d{1,3}[-\s]?)?(?:\d{2,4}[-\s]?){2,4}\d{2,4}\b")
_JAPAN_POSTAL_RE = re.compile(r"\b\d{3}-\d{4}\b")
_IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")


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


def detect_language(text: str) -> str:
    """
    シンプルな言語判定（日本語/英語）。
    収集時のlang属性が不正確なケースを補助する。
    """
    t = normalize_text(text or "")
    if not t:
        return "unknown"
    ja_chars = len(_JA_CHAR_RE.findall(t))
    ratio = ja_chars / max(1, len(t))
    return "ja" if ratio >= 0.12 else "en"


def contains_pii(text: str) -> bool:
    """
    メール・電話・郵便番号・IPv4のような個人/機微情報を簡易検知する。
    """
    t = text or ""
    return any(
        p.search(t)
        for p in (_EMAIL_RE, _PHONE_RE, _JAPAN_POSTAL_RE, _IPV4_RE)
    )


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
