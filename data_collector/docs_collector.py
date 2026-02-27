import os
import sys
import time
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_collector.db_manager import DBManager


DEFAULT_DOC_URLS = [
    "https://docs.python.org/3/tutorial/index.html",
    "https://pytorch.org/docs/stable/index.html",
    "https://www.postgresql.org/docs/current/index.html",
    "https://kubernetes.io/docs/concepts/overview/",
]


def _extract_text(url: str) -> str:
    headers = {"User-Agent": "DIY-LLM-Docs/1.0"}
    response = requests.get(url, timeout=15, headers=headers)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, "html.parser")
    paragraphs = soup.find_all("p")
    text = "\n".join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])
    return text


def collect_docs() -> int:
    db = DBManager()
    env_urls = os.environ.get("DOC_SEED_URLS", "").strip()
    urls = [u.strip() for u in env_urls.split(",") if u.strip()] if env_urls else DEFAULT_DOC_URLS
    saved = 0
    for url in urls:
        if db.is_url_crawled(url):
            continue
        try:
            content = _extract_text(url)
            if len(content) < 400:
                continue
            db.insert_crawled_data(
                url=url,
                domain=urlparse(url).netloc,
                title=url,
                content=content,
                source_type="docs",
                language="en",
            )
            saved += 1
            time.sleep(0.2)
        except Exception as e:
            print(f"[Docs] Failed {url}: {e}")
    print(f"[Docs] Saved {saved} docs pages.")
    return saved


if __name__ == "__main__":
    collect_docs()
