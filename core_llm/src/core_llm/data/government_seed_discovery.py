from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import requests


SITEMAP_INDEX_URL = "https://www.digital.go.jp/sitemap.xml"
SITEMAP_NS = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
ALLOWED_PREFIXES = (
    "/policies/",
    "/news/",
    "/resources/",
    "/services/",
    "/about/",
    "/councils/",
    "/laws/",
    "/applications/",
)
EXCLUDED_PARTS = (
    "/en/",
    "/assets/",
    "/search",
    "/sitemap",
    "/rss",
    "/contact",
    "/site-policy",
    "/social-media-policy",
    "/privacy-policy",
    "/accessibility-statement",
    "/timelines/",
    "/404/",
)
EXCLUDED_EXACT = {
    "https://www.digital.go.jp/",
    "https://www.digital.go.jp/policies",
    "https://www.digital.go.jp/news",
    "https://www.digital.go.jp/resources",
    "https://www.digital.go.jp/services",
    "https://www.digital.go.jp/about",
    "https://www.digital.go.jp/councils",
    "https://www.digital.go.jp/laws",
    "https://www.digital.go.jp/applications",
}


def _parse_xml_locs(xml_text: str) -> list[str]:
    root = ET.fromstring(xml_text)
    return [loc.text for loc in root.findall(".//sm:loc", SITEMAP_NS) if loc.text]


def should_include_seed_url(url: str) -> bool:
    if not url.startswith("https://www.digital.go.jp/"):
        return False
    normalized = url.rstrip("/")
    if normalized in EXCLUDED_EXACT:
        return False
    if any(part in url for part in EXCLUDED_PARTS):
        return False
    path = url.removeprefix("https://www.digital.go.jp")
    return any(path.startswith(prefix) for prefix in ALLOWED_PREFIXES)


def discover_government_seed_urls(*, limit: int = 150, timeout: int = 20) -> list[str]:
    session = requests.Session()
    session.headers.update({"User-Agent": "core_llm/0.1 government_ja seed discovery"})
    sitemap_pages = _parse_xml_locs(session.get(SITEMAP_INDEX_URL, timeout=timeout).text)

    candidates: list[str] = []
    seen: set[str] = set()
    for sitemap_url in sitemap_pages:
        for url in _parse_xml_locs(session.get(sitemap_url, timeout=timeout).text):
            if not should_include_seed_url(url):
                continue
            if url in seen:
                continue
            seen.add(url)
            candidates.append(url)
            if len(candidates) >= limit:
                return candidates
    return candidates


def write_seed_urls(path: str | Path, urls: list[str]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(urls) + "\n", encoding="utf-8")
