from __future__ import annotations

import bz2
import json
import re
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from pathlib import Path

from core_llm.data.cleaning import is_usable_text, looks_japanese, normalize_text, strip_noisy_lines
from core_llm.data.dedup import is_duplicate
from core_llm.data.manifest_schema import ManifestRecord


SUPPORTED_LANGS = {"ja"}
WIKI_XML_NS = "{http://www.mediawiki.org/xml/export-0.10/}"


def latest_dump_url(lang: str) -> str:
    if lang not in SUPPORTED_LANGS:
        raise ValueError(f"Unsupported language: {lang}")
    return f"https://dumps.wikimedia.org/{lang}wiki/latest/{lang}wiki-latest-pages-articles.xml.bz2"


def resolve_dump_path(raw_dir: Path, lang: str) -> Path:
    if lang not in SUPPORTED_LANGS:
        raise ValueError(f"Unsupported language: {lang}")
    return raw_dir / f"{lang}wiki-latest-pages-articles.xml.bz2"


def _progress_hook(block_count: int, block_size: int, total_size: int) -> None:
    if total_size <= 0 or block_count % 512 != 0:
        return
    done_mb = (block_count * block_size) / (1024 * 1024)
    total_mb = total_size / (1024 * 1024)
    print(f"[WikipediaDump] download {done_mb:.1f}/{total_mb:.1f} MB")


def download_dump(lang: str, raw_dir: Path, refresh: bool = False) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    target = resolve_dump_path(raw_dir, lang)
    if target.exists() and not refresh:
        return target
    part_path = target.with_suffix(target.suffix + ".part")
    part_path.unlink(missing_ok=True)
    url = latest_dump_url(lang)
    try:
        urllib.request.urlretrieve(url, part_path, reporthook=_progress_hook)
        part_path.replace(target)
    except Exception as exc:
        part_path.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to download dump from {url}: {exc}") from exc
    return target


def _strip_tag(tag: str) -> str:
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def iter_wiki_pages_from_bz2(dump_path: Path) -> Iterator[dict]:
    try:
        with bz2.open(dump_path, "rb") as fh:
            context = ET.iterparse(fh, events=("end",))
            for _, elem in context:
                if _strip_tag(elem.tag) != "page":
                    continue
                title = ""
                page_id = ""
                ns = ""
                redirect = False
                text = ""
                for child in list(elem):
                    tag = _strip_tag(child.tag)
                    if tag == "title":
                        title = (child.text or "").strip()
                    elif tag == "id" and not page_id:
                        page_id = (child.text or "").strip()
                    elif tag == "ns":
                        ns = (child.text or "").strip()
                    elif tag == "redirect":
                        redirect = True
                    elif tag == "revision":
                        for rev_child in list(child):
                            if _strip_tag(rev_child.tag) == "text":
                                text = rev_child.text or ""
                                break
                yield {
                    "title": title,
                    "page_id": page_id,
                    "ns": ns,
                    "redirect": redirect,
                    "text": text,
                }
                elem.clear()
    except ET.ParseError as exc:
        raise RuntimeError(f"Failed to parse dump {dump_path}: {exc}") from exc


def _remove_templates(text: str) -> str:
    previous = None
    current = text
    # Conservative repeated pass to handle simple nested template cases.
    for _ in range(6):
        if current == previous:
            break
        previous = current
        current = re.sub(r"\{\{[^{}]*\}\}", "", current, flags=re.DOTALL)
    return current


def extract_plaintext_from_revision_text(wikitext: str) -> str:
    text = wikitext or ""
    text = _remove_templates(text)
    text = re.sub(r"<ref[^>]*>.*?</ref>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\[\[(?:Category|カテゴリ):[^\]]+\]\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\[\[(?:File|Image|ファイル):[^\]]+\]\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"={2,}\s*(.*?)\s*={2,}", r"\1", text)
    text = re.sub(r"^\s*[*#:;].*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\|.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\{\|.*?\|\}", "", text, flags=re.DOTALL)
    text = re.sub(r"''+", "", text)
    text = normalize_text(text)
    text = strip_noisy_lines(text)
    return normalize_text(text)


def page_to_manifest_record(page: dict, *, min_chars: int) -> ManifestRecord | None:
    if str(page.get("ns", "")) != "0":
        return None
    if bool(page.get("redirect")):
        return None
    text = extract_plaintext_from_revision_text(str(page.get("text", "")))
    if not text:
        return None
    if not looks_japanese(text):
        return None
    if not is_usable_text(text, min_chars=min_chars):
        return None
    page_id = str(page.get("page_id", "")).strip()
    if not page_id:
        return None
    return ManifestRecord(
        id=f"jawiki:{page_id}",
        text=text,
        lang="ja",
        source="wikipedia_ja",
        license="cc-by-sa-4.0",
        split_hint="auto",
    )


def build_wikipedia_manifest(
    *,
    lang: str,
    output_path: Path,
    raw_dir: Path,
    dump_path: Path | None = None,
    min_chars: int = 120,
    max_docs: int | None = None,
    refresh: bool = False,
    report_path: Path | None = None,
) -> dict:
    if lang not in SUPPORTED_LANGS:
        raise ValueError(f"Unsupported language: {lang}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    resolved_dump = dump_path or download_dump(lang, raw_dir, refresh=refresh)
    report = {
        "lang": lang,
        "dump_path": str(resolved_dump),
        "total_pages": 0,
        "kept_docs": 0,
        "filtered_namespace": 0,
        "filtered_redirect": 0,
        "filtered_short": 0,
        "filtered_non_japanese": 0,
        "filtered_duplicate": 0,
        "filtered_empty": 0,
    }
    seen_hashes: set[str] = set()
    kept_rows = 0
    with open(output_path, "w", encoding="utf-8") as out_f:
        for page in iter_wiki_pages_from_bz2(resolved_dump):
            report["total_pages"] += 1
            if str(page.get("ns", "")) != "0":
                report["filtered_namespace"] += 1
                continue
            if bool(page.get("redirect")):
                report["filtered_redirect"] += 1
                continue
            text = extract_plaintext_from_revision_text(str(page.get("text", "")))
            if not text:
                report["filtered_empty"] += 1
                continue
            if not looks_japanese(text):
                report["filtered_non_japanese"] += 1
                continue
            if not is_usable_text(text, min_chars=min_chars):
                report["filtered_short"] += 1
                continue
            if is_duplicate(text, seen_hashes):
                report["filtered_duplicate"] += 1
                continue
            row = ManifestRecord(
                id=f"jawiki:{str(page.get('page_id', '')).strip()}",
                text=text,
                lang="ja",
                source="wikipedia_ja",
                license="cc-by-sa-4.0",
                split_hint="auto",
            )
            out_f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")
            kept_rows += 1
            report["kept_docs"] = kept_rows
            if max_docs is not None and kept_rows >= max_docs:
                break
    if report_path is None:
        report_path = output_path.with_suffix(".report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report
