"""Livedoor News Corpus downloader and manifest builder.

The livedoor news corpus (CC BY-ND 2.1 JP) is a standard Japanese NLP dataset
with ~7,376 articles across 9 news categories.

Download URL: https://www.rondhuit.com/download/ldcc-20140209.tar.gz

File format per article:
  Line 1: article URL
  Line 2: publication date (ISO format)
  Line 3+: article body text
"""
from __future__ import annotations

import json
import tarfile
import urllib.request
from pathlib import Path

from core_llm.data.cleaning import is_usable_text, normalize_text, strip_noisy_lines
from core_llm.data.manifest_schema import ManifestRecord


LIVEDOOR_URL = "https://www.rondhuit.com/download/ldcc-20140209.tar.gz"
LIVEDOOR_FILENAME = "ldcc-20140209.tar.gz"
LIVEDOOR_LICENSE = "cc-by-nd-2.1-jp"
LIVEDOOR_SOURCE = "livedoor_news"

CATEGORIES = [
    "dokujo-tsushin",
    "it-life-hack",
    "kaden-channel",
    "livedoor-homme",
    "movie-enter",
    "peachy",
    "smax",
    "sports-watch",
    "topic-news",
]


def download_livedoor(raw_dir: Path, *, refresh: bool = False) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_dir / LIVEDOOR_FILENAME
    if dest.exists() and not refresh:
        return dest
    print(f"Downloading livedoor news corpus → {dest}")
    urllib.request.urlretrieve(LIVEDOOR_URL, dest)
    return dest


def _parse_article(text: str) -> tuple[str, str]:
    """Return (url, body) from raw article text."""
    lines = text.splitlines()
    if len(lines) < 3:
        return "", ""
    url = lines[0].strip()
    # line 1 = url, line 2 = date, line 3+ = body
    body = "\n".join(lines[2:]).strip()
    return url, body


def build_livedoor_manifest(
    output_path: Path,
    raw_dir: Path,
    *,
    min_chars: int = 120,
    max_docs: int | None = None,
    refresh: bool = False,
) -> dict:
    archive = download_livedoor(raw_dir, refresh=refresh)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records: list[ManifestRecord] = []
    seen: set[str] = set()

    with tarfile.open(archive, "r:gz") as tar:
        members = sorted(tar.getnames())
        for name in members:
            if max_docs is not None and len(records) >= max_docs:
                break
            # skip non-article files (README, LICENSE, etc.)
            parts = name.split("/")
            if len(parts) < 3:
                continue
            category = parts[1]
            filename = parts[2]
            if category not in CATEGORIES:
                continue
            if not filename.endswith(".txt"):
                continue
            # skip LICENSE.txt etc.
            if not filename[0].isdigit():
                continue

            member = tar.getmember(name)
            f = tar.extractfile(member)
            if f is None:
                continue
            raw = f.read().decode("utf-8", errors="replace")
            url, body = _parse_article(raw)

            body = normalize_text(body)
            body = strip_noisy_lines(body)

            if not is_usable_text(body, min_chars=min_chars):
                continue
            if body in seen:
                continue
            seen.add(body)

            doc_id = f"livedoor:{category}:{filename.replace('.txt', '')}"
            records.append(ManifestRecord(
                id=doc_id,
                text=body,
                lang="ja",
                source=LIVEDOOR_SOURCE,
                license=LIVEDOOR_LICENSE,
                split_hint="auto",
            ))

    with open(output_path, "w", encoding="utf-8") as out:
        for rec in records:
            out.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")

    report = {
        "source": LIVEDOOR_SOURCE,
        "output": str(output_path),
        "total_records": len(records),
        "min_chars": min_chars,
    }
    print(json.dumps(report, ensure_ascii=False))
    return report
