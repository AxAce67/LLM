import os
import sys
import time
import urllib.parse
import xml.etree.ElementTree as ET

import requests

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_collector.db_manager import DBManager


DEFAULT_ARXIV_QUERY = 'cat:cs.LG OR cat:cs.CL OR cat:cs.AI'


def collect_arxiv(max_results: int = 30) -> dict:
    db = DBManager()
    query = os.environ.get("ARXIV_QUERY", DEFAULT_ARXIV_QUERY).strip() or DEFAULT_ARXIV_QUERY
    encoded = urllib.parse.quote(query)
    url = (
        "http://export.arxiv.org/api/query"
        f"?search_query={encoded}&sortBy=submittedDate&sortOrder=descending&start=0&max_results={max(1, max_results)}"
    )
    headers = {"User-Agent": "DIY-LLM-arXiv/1.0"}
    stats = {"saved": 0, "failed": 0, "skipped_duplicate": 0, "skipped_short": 0}
    try:
        xml_text = requests.get(url, timeout=20, headers=headers).text
        root = ET.fromstring(xml_text)
        ns = {"a": "http://www.w3.org/2005/Atom"}
        entries = root.findall("a:entry", ns)
        for entry in entries:
            id_el = entry.find("a:id", ns)
            title_el = entry.find("a:title", ns)
            summary_el = entry.find("a:summary", ns)
            if id_el is None or summary_el is None:
                continue
            paper_url = (id_el.text or "").strip()
            if not paper_url or db.is_url_crawled(paper_url):
                stats["skipped_duplicate"] += 1
                continue
            title = (title_el.text or "").strip() if title_el is not None else ""
            summary = (summary_el.text or "").strip()
            if len(summary) < 120:
                stats["skipped_short"] += 1
                continue
            content = f"Title: {title}\n\nAbstract:\n{summary}"
            db.insert_crawled_data(
                url=paper_url,
                domain="arxiv.org",
                title=title[:255],
                content=content,
                source_type="arxiv",
                language="en",
            )
            stats["saved"] += 1
            time.sleep(0.05)
    except Exception as e:
        stats["failed"] += 1
        print(f"[arXiv] collection failed: {e}")
    print(
        "[arXiv] "
        f"saved={stats['saved']} "
        f"failed={stats['failed']} "
        f"skipped_dup={stats['skipped_duplicate']} "
        f"skipped_short={stats['skipped_short']}"
    )
    return stats


if __name__ == "__main__":
    collect_arxiv()
