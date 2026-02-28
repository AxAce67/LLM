import os
import sys
import time
import requests
import xml.etree.ElementTree as ET
from urllib.parse import urlparse

from bs4 import BeautifulSoup

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_collector.db_manager import DBManager


DEFAULT_NEWS_FEEDS = [
    "https://feeds.arstechnica.com/arstechnica/index",
    "https://www.theverge.com/rss/index.xml",
    "https://www.wired.com/feed/rss",
    "https://feeds.bbci.co.uk/news/technology/rss.xml",
    "https://www.nature.com/subjects/machine-learning.rss",
    "https://www.infoq.com/feed/",
    "https://zenn.dev/topics/ai/feed",
    "https://qiita.com/tags/llm/feed",
]


def _detect_language(text: str) -> str:
    return "ja" if any("\u3040" <= c <= "\u309F" for c in text) else "en"


def _fetch_article_text(url: str) -> str:
    headers = {"User-Agent": os.environ.get("COLLECTOR_USER_AGENT", "DIY-LLM-News/1.0")}
    response = requests.get(url, timeout=12, headers=headers)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, "html.parser")
    paragraphs = soup.find_all("p")
    return "\n".join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])


def _collect_from_rss_feeds(db: DBManager, max_items_per_feed: int, stats: dict) -> int:
    feed_env = os.environ.get("NEWS_FEEDS", "").strip()
    feeds = [u.strip() for u in feed_env.split(",") if u.strip()] if feed_env else DEFAULT_NEWS_FEEDS
    saved = 0
    for feed_url in feeds:
        try:
            xml_text = requests.get(
                feed_url,
                timeout=12,
                headers={"User-Agent": os.environ.get("COLLECTOR_USER_AGENT", "DIY-LLM-News/1.0")},
            ).text
            root = ET.fromstring(xml_text)
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
                    content = _fetch_article_text(url)
                    if len(content) < 300:
                        stats["skipped_short"] += 1
                        continue
                    db.insert_crawled_data(
                        url=url,
                        domain=urlparse(url).netloc,
                        title=(title_el.text.strip() if title_el is not None and title_el.text else ""),
                        content=content,
                        source_type="news",
                        language=_detect_language(content),
                    )
                    saved += 1
                    time.sleep(0.2)
                except Exception as inner:
                    stats["fetch_failed"] += 1
                    print(f"[News RSS] Failed article fetch {url}: {inner}")
        except Exception as e:
            stats["feed_failed"] += 1
            print(f"[News RSS] Failed feed {feed_url}: {e}")
    return saved


def _collect_from_hackernews(db: DBManager, max_items: int, stats: dict) -> int:
    saved = 0
    try:
        ids = requests.get(
            "https://hacker-news.firebaseio.com/v0/topstories.json",
            timeout=10,
            headers={"User-Agent": os.environ.get("COLLECTOR_USER_AGENT", "DIY-LLM-News/1.0")},
        ).json()[:max_items]
    except Exception as e:
        stats["feed_failed"] += 1
        print(f"[HN] Failed topstories fetch: {e}")
        return 0

    for story_id in ids:
        try:
            item = requests.get(
                f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json",
                timeout=10,
                headers={"User-Agent": os.environ.get("COLLECTOR_USER_AGENT", "DIY-LLM-News/1.0")},
            ).json()
            if not item:
                continue
            url = (item.get("url") or "").strip()
            title = (item.get("title") or "").strip()
            if not url or db.is_url_crawled(url):
                stats["skipped_duplicate"] += 1
                continue
            content = _fetch_article_text(url)
            if len(content) < 300:
                stats["skipped_short"] += 1
                continue
            db.insert_crawled_data(
                url=url,
                domain=urlparse(url).netloc,
                title=title,
                content=content,
                source_type="news",
                language=_detect_language(content),
            )
            saved += 1
            time.sleep(0.2)
        except Exception as e:
            stats["hn_failed"] += 1
            print(f"[HN] Failed item {story_id}: {e}")
    return saved


def collect_news(max_items_per_feed: int = 5, max_hn_items: int = 15) -> dict:
    db = DBManager()
    stats = {
        "saved": 0,
        "fetch_failed": 0,
        "feed_failed": 0,
        "hn_failed": 0,
        "skipped_duplicate": 0,
        "skipped_short": 0,
    }
    stats["saved"] += _collect_from_rss_feeds(db, max_items_per_feed=max_items_per_feed, stats=stats)
    stats["saved"] += _collect_from_hackernews(db, max_items=max_hn_items, stats=stats)
    print(
        "[News] "
        f"saved={stats['saved']} "
        f"failed_fetch={stats['fetch_failed']} "
        f"failed_feed={stats['feed_failed']} "
        f"failed_hn={stats['hn_failed']} "
        f"skipped_dup={stats['skipped_duplicate']} "
        f"skipped_short={stats['skipped_short']}"
    )
    return stats


if __name__ == "__main__":
    collect_news()
