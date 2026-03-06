import os
import time
from urllib.parse import urlparse, parse_qs, unquote

import requests
from bs4 import BeautifulSoup


DEFAULT_DISCOVERY_QUERIES = [
    "latest ai research",
    "machine learning engineering blog",
    "python backend architecture",
    "llm training optimization",
    "data engineering tutorial",
]


def _normalize_result_url(raw_url: str) -> str:
    url = (raw_url or "").strip()
    if not url:
        return ""
    # DuckDuckGo redirect format: //duckduckgo.com/l/?uddg=<encoded_url>
    if "duckduckgo.com/l/" in url and "uddg=" in url:
        parsed = urlparse(url)
        qs = parse_qs(parsed.query)
        uddg = qs.get("uddg", [])
        if uddg:
            return unquote(uddg[0]).strip()
    if url.startswith("//"):
        url = "https:" + url
    return url


def _is_acceptable_url(url: str) -> bool:
    if not url.startswith(("http://", "https://")):
        return False
    low = url.lower()
    blocked = [
        "google.com",
        "duckduckgo.com",
        "youtube.com",
        "facebook.com",
        "instagram.com",
        "x.com",
        "twitter.com",
    ]
    return not any(b in low for b in blocked)


def _search_duckduckgo(query: str, max_results: int = 8) -> list[str]:
    headers = {"User-Agent": "DIY-LLM-Discovery/1.0"}
    params = {"q": query}
    response = requests.get("https://duckduckgo.com/html/", params=params, headers=headers, timeout=12)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    links = []
    for a in soup.select("a.result__a, a.result__url"):
        href = _normalize_result_url(a.get("href", ""))
        if not href or not _is_acceptable_url(href):
            continue
        links.append(href.split("#")[0])
        if len(links) >= max_results:
            break
    return links


def discover_seed_urls(max_urls: int = 20, max_results_per_query: int = 8) -> list[str]:
    query_env = os.environ.get("AUTO_DISCOVERY_QUERIES", "").strip()
    queries = [q.strip() for q in query_env.split(",") if q.strip()] if query_env else DEFAULT_DISCOVERY_QUERIES

    discovered = []
    seen = set()
    for q in queries:
        try:
            urls = _search_duckduckgo(q, max_results=max_results_per_query)
            for u in urls:
                if u in seen:
                    continue
                seen.add(u)
                discovered.append(u)
                if len(discovered) >= max_urls:
                    return discovered
            time.sleep(0.4)
        except Exception as e:
            print(f"[AutoDiscovery] query failed '{q}': {e}")
    return discovered


if __name__ == "__main__":
    urls = discover_seed_urls()
    print(f"discovered={len(urls)}")
    for u in urls:
        print(u)
