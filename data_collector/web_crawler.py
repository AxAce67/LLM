import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import os

# ルートディレクトリのモジュールをインポートするためのパス設定
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_collector.db_manager import DBManager

def crawl_url(url, db_manager):
    """単一のURLをクロールしてSupabaseに保存し、ページ内のリンクを返す"""
    if db_manager.is_url_crawled(url):
        print(f"[Skip] Already crawled: {url}")
        return []

    print(f"[Crawl] Fetching: {url}")
    try:
        # User-Agentを設定してブロックを回避しやすくする
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; DIY-LLM-Crawler/1.0)",
            "Accept-Language": "ja,en-US;q=0.9,en;q=0.8"
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
    except Exception as e:
        print(f"[Error] Failed to fetch {url}: {e}")
        return []

    # HTMLのパース
    soup = BeautifulSoup(response.content, "html.parser")
    
    # 簡易的な本文抽出: <p>タグのテキストを結合
    paragraphs = soup.find_all('p')
    text_content = "\n".join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])

    domain = urlparse(url).netloc
    title = soup.title.string.strip() if soup.title and soup.title.string else ""
    
    # 日本語か英語か簡易判定 (平仮名が含まれていれば日本語とする)
    language = "ja" if any('\u3040' <= c <= '\u309F' for c in text_content) else "en"
    
    # 意味のある長さ（例えば100文字以上）のテキストのみSupabaseに保存
    if len(text_content) > 100:
        success = db_manager.insert_crawled_data(
            url=url,
            domain=domain,
            title=title,
            content=text_content,
            source_type="web",
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
            time.sleep(1)
            
    print(f"Crawling finished. Processed {crawled_count} pages.")

if __name__ == "__main__":
    # テスト単体起動用
    seeds = [
        "https://ja.wikipedia.org/wiki/Python", # テスト用にPythonのWikiページ
    ]
    start_crawler(seeds, max_workers=3, max_pages=10)
