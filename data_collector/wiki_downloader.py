import os

DATASET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset", "wikipedia")

def download_wikipedia_dump(language="ja"):
    """
    Wikipediaの公式ダンプ（記事の塊）を直接ローカルストレージにダウンロードし、展開する。
    注意：数GB〜数十GBになるため、クラウドDB（Supabase等）の無料枠には保存せず、
    各PC（またはコンテナのマウントボリューム）に直接保管するハイブリッド戦略を採用する。
    """
    os.makedirs(DATASET_DIR, exist_ok=True)
    
    print(f"Starting Wikipedia ({language}) dataset preparation...")
    print("-------------------------------------------------------------------------")
    print(f"[IMPORTANT] Wikipedia data for '{language}' can be extremely large (>15GB uncompressed).")
    print(f"To save database costs and prevent Supabase free-tier limits from exceeding,")
    print(f"we will download and store this data LOCALLY in: \n{DATASET_DIR}")
    print("-------------------------------------------------------------------------")
    
    # ※ 実運用時は以下のようなフローになる
    # 1. urllib等で https://dumps.wikimedia.org/jawiki/latest/jawiki-latest-pages-articles.xml.bz2 をダウンロード
    # 2. wikiextractor パッケージ等を用いて .bz2 をローカルでクリーンなテキストファイル群にパースする
    
    print("[TODO] Please implement actual download logic (urllib) and extraction (wikiextractor).")
    
if __name__ == "__main__":
    download_wikipedia_dump("ja")
