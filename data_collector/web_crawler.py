import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.robotparser import RobotFileParser
import sys
import os
import threading

# ルートディレクトリのモジュールをインポートするためのパス設定
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_collector.db_manager import DBManager

_robots_cache = {}
_robots_lock = threading.Lock()
_domain_last_request = {}
_domain_time_lock = threading.Lock()
_domain_locks = {}
_domain_locks_lock = threading.Lock()


def _get_domain_lock(domain: str):
    with _domain_locks_lock:
        if domain not in _domain_locks:
            _domain_locks[domain] = threading.Semaphore(max(1, int(os.environ.get("CRAWLER_DOMAIN_CONCURRENCY", "2"))))
        return _domain_locks[domain]


def _allowed_by_robots(url: str, user_agent: str = "DIY-LLM-Crawler") -> bool:
    if os.environ.get("CRAWLER_RESPECT_ROBOTS", "1") != "1":
        return True
    parsed = urlparse(url)
    key = f"{parsed.scheme}://{parsed.netloc}"
    with _robots_lock:
        rp = _robots_cache.get(key)
        if rp is None:
            rp = RobotFileParser()
            rp.set_url(f"{key}/robots.txt")
            try:
                rp.read()
            except Exception:
                # robotsが取れない場合は保守的に許可（収集停止を避ける）
                pass
            _robots_cache[key] = rp
    try:
        return rp.can_fetch(user_agent, url)
    except Exception:
        return True


def _respect_domain_rate_limit(domain: str):
    min_interval = float(os.environ.get("CRAWLER_DOMAIN_MIN_INTERVAL_SEC", "0.5"))
    if min_interval <= 0:
        return
    with _domain_time_lock:
        now = time.time()
        last = _domain_last_request.get(domain, 0.0)
        wait_sec = (last + min_interval) - now
        if wait_sec > 0:
            time.sleep(wait_sec)
        _domain_last_request[domain] = time.time()


def crawl_url(url, db_manager):
    """単一のURLをクロールしてPostgreSQLに保存し、ページ内のリンクを返す"""
    if db_manager.is_url_crawled(url):
        print(f"[Skip] Already crawled: {url}")
        return []

    parsed = urlparse(url)
    domain = parsed.netloc
    if not _allowed_by_robots(url):
        print(f"[Skip] robots.txt disallow: {url}")
        return []

    domain_lock = _get_domain_lock(domain)
    acquired = domain_lock.acquire(timeout=2.0)
    if not acquired:
        print(f"[Skip] domain semaphore busy: {url}")
        return []

    print(f"[Crawl] Fetching: {url}")
    try:
        _respect_domain_rate_limit(domain)
        # User-Agentを設定してブロックを回避しやすくする
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; DIY-LLM-Crawler/1.0)",
            "Accept-Language": "ja,en-US;q=0.9,en;q=0.8"
        }
        response = requests.get(url, headers=headers, timeout=10, allow_redirects=True)
        response.raise_for_status()
    except Exception as e:
        print(f"[Error] Failed to fetch {url}: {e}")
        return []
    finally:
        # response 成功/失敗に関係なくロック解放
        if acquired:
            domain_lock.release()

    # HTMLのパース
    soup = BeautifulSoup(response.content, "html.parser")
    
    # 簡易的な本文抽出: <p>タグのテキストを結合
    paragraphs = soup.find_all('p')
    text_content = "\n".join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])

    title = soup.title.string.strip() if soup.title and soup.title.string else ""
    
    # 日本語か英語か簡易判定 (平仮名が含まれていれば日本語とする)
    language = "ja" if any('\u3040' <= c <= '\u309F' for c in text_content) else "en"
    
    # WikipediaかWebかを自動判定
    source_type = "wikipedia" if "wikipedia.org" in url else "web"
    
    # 意味のある長さ（例えば100文字以上）のテキストのみDBに保存
    if len(text_content) > 100:
        success = db_manager.insert_crawled_data(
            url=url,
            domain=domain,
            title=title,
            content=text_content,
            source_type=source_type,
            language=language
        )
        if success:
            print(f"[Saved] Inserted to DB: {url} ({len(text_content)} chars)")
            
    # 深掘りするためのリンクをページ内から抽出
    new_links = []
    for a_tag in soup.find_all('a', href=True):
        href = a_tag['href']
        full_url = urljoin(url, href)
        # 外部ツール等への迷惑を防ぐため、今回は「同一ドメイン内」を優先的に辿る安全な設計
        if full_url.startswith("http") and domain in full_url:
            # フラグメント（#）を取り除く
            clean_url = full_url.split('#')[0]
            new_links.append(clean_url)
            
    return list(set(new_links)) # 重複リンクを除去して返す

def start_crawler(seed_urls, max_workers=5, max_pages=50):
    """
    マルチスレッドで並行してクローラーを動かすメイン関数
    max_workers: 同時に動かすクローラーの数（並列アクセス数）
    max_pages: 今回の実行で最大何ページまでクロールするか
    """
    db = DBManager()
    urls_to_crawl = set(seed_urls)
    crawled_count = 0
    
    print(f"Starting Multi-threaded Crawler with {max_workers} workers...")
    
    # スレッドプールで並列化（I/Oバウンドな通信処理を高速化）
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        while urls_to_crawl and crawled_count < max_pages:
            # キューからワーカー数分だけURLを取り出す
            batch = list(urls_to_crawl)[:max_workers]
            for u in batch:
                urls_to_crawl.remove(u)
                
            # 並列実行のタスクを登録
            futures = {executor.submit(crawl_url, url, db): url for url in batch}
            
            # 完了したタスクから順次処理
            for future in as_completed(futures):
                try:
                    crawled_count += 1
                    links = future.result()
                    
                    # 見つかった新しいリンクをキューに追加（メモリ爆発を防ぐため上限を設ける）
                    for link in links:
                        if len(urls_to_crawl) < 5000:
                            urls_to_crawl.add(link)
                except Exception as e:
                    print(f"[Worker Error] {e}")
                    
            # サーバーに優しくするためバッチごとに少し待機
            time.sleep(float(os.environ.get("CRAWLER_BATCH_SLEEP_SEC", "0.6")))
            
    print(f"Crawling finished. Processed {crawled_count} pages.")

if __name__ == "__main__":
    # テスト単体起動用
    seeds = [
        "https://ja.wikipedia.org/wiki/Python", # テスト用にPythonのWikiページ
    ]
    start_crawler(seeds, max_workers=3, max_pages=10)
