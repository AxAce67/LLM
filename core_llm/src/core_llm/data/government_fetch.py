from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

from core_llm.data.cleaning import is_usable_text, looks_japanese, normalize_text


ALLOWED_GOVERNMENT_DOMAINS = (
    "go.jp",
    "digital.go.jp",
    "e-gov.go.jp",
    "data.go.jp",
)

NOISE_LINE_PATTERNS = (
    re.compile(r"^最終更新日:?$"),
    re.compile(r"^\d{4}年\d{1,2}月\d{1,2}日$"),
    re.compile(r"^公表日[:：]\s*\d{4}年\d{1,2}月\d{1,2}日$"),
    re.compile(r"^資料\d+([:-].*)?$"),
    re.compile(r"^議事次第$"),
    re.compile(r"^関連政策$"),
    re.compile(r"^問合せ先$"),
    re.compile(r"^電話[:：].*$"),
    re.compile(r"^メール[:：].*$"),
    re.compile(r"^_atmark_$"),
)

NOISE_SUBSTRINGS = (
    "（PDF／",
    "（PDF/",
    "PDF／",
    "PDF/",
    "KB）",
    "KB)",
)


def is_allowed_government_url(url: str) -> bool:
    hostname = (urlparse(url).hostname or "").lower()
    return any(hostname == domain or hostname.endswith(f".{domain}") for domain in ALLOWED_GOVERNMENT_DOMAINS)


def load_seed_urls(seed_file: str | Path) -> list[str]:
    urls: list[str] = []
    for raw_line in Path(seed_file).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        urls.append(line)
    return urls


def extract_text_from_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag_name in ("script", "style", "nav", "header", "footer", "aside", "noscript"):
        for tag in soup.find_all(tag_name):
            tag.decompose()
    root = soup.find("article") or soup.find("main") or soup.body or soup
    text = root.get_text("\n", strip=True)
    text = normalize_text(text)
    return text


def clean_government_text(text: str) -> str:
    cleaned_lines: list[str] = []
    skip_next_date = False
    for raw_line in normalize_text(text).splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if skip_next_date and re.match(r"^\d{4}年\d{1,2}月\d{1,2}日$", line):
            skip_next_date = False
            continue
        skip_next_date = False
        if line == "最終更新日:":
            skip_next_date = True
            continue
        if any(pattern.match(line) for pattern in NOISE_LINE_PATTERNS):
            continue
        if any(token in line for token in NOISE_SUBSTRINGS):
            continue
        cleaned_lines.append(line)
    return normalize_text("\n".join(cleaned_lines))


def fetch_government_corpus(
    *,
    seed_file: str | Path,
    output_dir: str | Path,
    min_chars: int = 200,
    timeout: int = 20,
    max_pages: int | None = None,
    refresh: bool = False,
    report_path: str | Path | None = None,
) -> dict:
    seed_urls = load_seed_urls(seed_file)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    report = {
        "seed_file": str(seed_file),
        "output_dir": str(output_dir),
        "requested_urls": 0,
        "fetched_pages": 0,
        "saved_docs": 0,
        "filtered_short": 0,
        "filtered_non_japanese": 0,
        "filtered_disallowed_domain": 0,
        "fetch_errors": 0,
    }

    session = requests.Session()
    session.headers.update({"User-Agent": "core_llm/0.1 government_ja fetcher"})
    urls = seed_urls[:max_pages] if max_pages is not None else seed_urls
    for url in urls:
        report["requested_urls"] += 1
        if not is_allowed_government_url(url):
            report["filtered_disallowed_domain"] += 1
            continue
        file_hash = hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]
        target_path = output_root / f"gov_{file_hash}.txt"
        if target_path.exists() and not refresh:
            report["saved_docs"] += 1
            continue
        try:
            response = session.get(url, timeout=timeout)
            response.raise_for_status()
        except requests.RequestException:
            report["fetch_errors"] += 1
            continue
        report["fetched_pages"] += 1
        text = clean_government_text(extract_text_from_html(response.text))
        if not is_usable_text(text, min_chars=min_chars):
            report["filtered_short"] += 1
            continue
        if not looks_japanese(text):
            report["filtered_non_japanese"] += 1
            continue
        target_path.write_text(text, encoding="utf-8")
        report["saved_docs"] += 1

    target_report = Path(report_path) if report_path is not None else output_root / "fetch.report.json"
    target_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
