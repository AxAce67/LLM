import os
import sys
import time
import requests
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
from urllib.parse import urlparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_collector.db_manager import DBManager


DEFAULT_RSS_FEEDS = [
    "https://openai.com/news/rss.xml",
    "https://aws.amazon.com/blogs/machine-learning/feed/",
    "https://aws.amazon.com/blogs/architecture/feed/",
    "https://developers.googleblog.com/en/rss/",
    "https://research.google/blog/rss/",
    "https://engineering.fb.com/feed/",
    "https://netflixtechblog.com/feed",
    "https://engineering.atspotify.com/feed/",
    "https://www.anthropic.com/news/rss.xml",
    "https://huggingface.co/blog/feed.xml",
    "https://www.databricks.com/blog/category/engineering/feed",
    "https://www.cncf.io/feed/",
    "https://www.postgresql.org/about/news.rss",
]


def _fetch_text(url: str) -> str:
    headers = {"User-Agent": os.environ.get("COLLECTOR_USER_AGENT", "DIY-LLM-RSS/1.0")}
    response = requests.get(url, timeout=12, headers=headers)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, "html.parser")
    paragraphs = soup.find_all("p")
    return "\n".join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])


def collect_from_rss(max_items_per_feed: int = 5):
    db = DBManager()
    feed_list = os.environ.get("RSS_FEEDS", "").strip()
    feeds = [u.strip() for u in feed_list.split(",") if u.strip()] if feed_list else DEFAULT_RSS_FEEDS
    stats = {
        "saved": 0,
        "fetch_failed": 0,
        "feed_failed": 0,
        "skipped_duplicate": 0,
        "skipped_short": 0,
    }

    for feed_url in feeds:
        try:
            feed_xml = requests.get(
                feed_url,
                timeout=12,
                headers={"User-Agent": os.environ.get("COLLECTOR_USER_AGENT", "DIY-LLM-RSS/1.0")},
            ).text
            root = ET.fromstring(feed_xml)
            items = root.findall(".//item")[:max_items_per_feed]
            for item in items:
                link_el = item.find("link")
                title_el = item.find("title")
                if link_el is None or not link_el.text:
                    continue
                url = link_el.text.strip()
                if db.is_url_crawled(url):
                    stats["skipped_duplicate"] += 1
                    continue
                try:
                    content = _fetch_text(url)
                    if len(content) < 200:
                        stats["skipped_short"] += 1
                        continue
                    db.insert_crawled_data(
                        url=url,
                        domain=urlparse(url).netloc,
                        title=(title_el.text.strip() if title_el is not None and title_el.text else ""),
                        content=content,
                        source_type="rss",
                        language="en",
                    )
                    stats["saved"] += 1
                    time.sleep(0.2)
                except Exception as inner:
                    stats["fetch_failed"] += 1
                    print(f"[RSS] Failed article fetch {url}: {inner}")
        except Exception as e:
            stats["feed_failed"] += 1
            print(f"[RSS] Failed feed {feed_url}: {e}")

    print(
        "[RSS] "
        f"saved={stats['saved']} "
        f"failed_fetch={stats['fetch_failed']} "
        f"failed_feed={stats['feed_failed']} "
        f"skipped_dup={stats['skipped_duplicate']} "
        f"skipped_short={stats['skipped_short']}"
    )
    return stats
