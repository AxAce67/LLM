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
    "https://docs.docker.com/",
    "https://fastapi.tiangolo.com/",
    "https://numpy.org/doc/stable/",
    "https://pandas.pydata.org/docs/",
    "https://huggingface.co/docs/transformers/index",
    "https://huggingface.co/docs/peft/index",
    "https://docs.ray.io/en/latest/",
    "https://docs.llamaindex.ai/en/stable/",
]


def _extract_text(url: str) -> str:
    headers = {"User-Agent": os.environ.get("COLLECTOR_USER_AGENT", "DIY-LLM-Docs/1.0")}
    response = requests.get(url, timeout=15, headers=headers)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, "html.parser")
    paragraphs = soup.find_all("p")
    text = "\n".join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])
    return text


def collect_docs() -> dict:
    db = DBManager()
    env_urls = os.environ.get("DOC_SEED_URLS", "").strip()
    urls = [u.strip() for u in env_urls.split(",") if u.strip()] if env_urls else DEFAULT_DOC_URLS
    stats = {"saved": 0, "failed": 0, "skipped_duplicate": 0, "skipped_short": 0}
    for url in urls:
        if db.is_url_crawled(url):
            stats["skipped_duplicate"] += 1
            continue
        try:
            content = _extract_text(url)
            if len(content) < 400:
                stats["skipped_short"] += 1
                continue
            db.insert_crawled_data(
                url=url,
                domain=urlparse(url).netloc,
                title=url,
                content=content,
                source_type="docs",
                language="en",
            )
            stats["saved"] += 1
            time.sleep(0.2)
        except Exception as e:
            stats["failed"] += 1
            print(f"[Docs] Failed {url}: {e}")
    print(
        "[Docs] "
        f"saved={stats['saved']} "
        f"failed={stats['failed']} "
        f"skipped_dup={stats['skipped_duplicate']} "
        f"skipped_short={stats['skipped_short']}"
    )
    return stats


if __name__ == "__main__":
    collect_docs()
