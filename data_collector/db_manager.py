import os
from dotenv import load_dotenv
from supabase import create_client, Client

# 環境変数の読み込み (.env ファイルがあればそこから、なければOSの環境変数)
load_dotenv()

class DBManager:
    def __init__(self):
        url: str = os.environ.get("SUPABASE_URL")
        key: str = os.environ.get("SUPABASE_KEY")
        
        if not url or not key:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in .env or environment variables.")
        
        self.supabase: Client = create_client(url, key)

    def is_url_crawled(self, url: str) -> bool:
        """
        指定されたURLが既にDBに存在するか（他のワーカーがクロール済みか）を確認する
        """
        try:
            response = self.supabase.table("crawled_data").select("id").eq("url", url).limit(1).execute()
            # データが1件でも返ってくれば、すでに収集済み
            return len(response.data) > 0
        except Exception as e:
            print(f"[DB Error] Failed to check URL existence '{url}': {e}")
            # エラー時は重複しているとみなし、安全側に倒す（無駄なクロールとDBエラーを防ぐ）
            return True

    def insert_crawled_data(self, url: str, domain: str, title: str, content: str, source_type: str, language: str = "ja") -> bool:
        """
        クロールしたデータをSupabaseに保存する
        """
        try:
            data = {
                "url": url,
                "domain": domain,
                "title": title[:255] if title else "", # 長すぎるタイトルをクリップ
                "content": content,
                "char_count": len(content),
                "source_type": source_type,
                "language": language
            }
            response = self.supabase.table("crawled_data").insert(data).execute()
            return True
        except Exception as e:
            # URLのUNIQUE制約違反（同時アクセスで競合した場合など）の場合もここでキャッチされる
            print(f"[DB Error] Failed to insert data for '{url}': {e}")
            return False

    def get_stats(self) -> dict:
        """
        現在のデータ収集の統計情報を取得する（ダッシュボード表示用）
        """
        try:
            # Supabase Python Client (PostgREST) では count付きのselectを発行
            res_wiki = self.supabase.table("crawled_data").select("*", count="exact").eq("source_type", "wikipedia").limit(0).execute()
            res_web = self.supabase.table("crawled_data").select("*", count="exact").eq("source_type", "web").limit(0).execute()
            
            wiki_count = res_wiki.count if res_wiki.count else 0
            web_count = res_web.count if res_web.count else 0
            
            # 本格的な運用ではPostgreSQLのRPC(Stored Procedure)で集計するのがベターだが、今回は簡易実装
            return {
                "collected_docs_wiki": wiki_count,
                "collected_docs_web": web_count,
                # 簡単なサイズ概算 (1文字 = 3バイト計算)
                "dataset_size_mb": round((wiki_count + web_count) * 1000 * 3 / (1024 * 1024), 2)
            }
        except Exception as e:
            print(f"[DB Error] Failed to get stats: {e}")
            return {
                "collected_docs_wiki": 0,
                "collected_docs_web": 0,
                "dataset_size_mb": 0.0
            }

    # --- Node Status Management ---

    def upsert_node_heartbeat(self, node_id: str, role: str, status: str, cpu_usage: float, ram_usage: float):
        """
        自身の稼働状態（Heartbeat）とリソース状況を書き込む・更新する
        """
        try:
            import datetime
            data = {
                "node_id": node_id,
                "role": role,
                "status": status,
                "cpu_usage": cpu_usage,
                "ram_usage": ram_usage,
                "last_heartbeat": datetime.datetime.now(datetime.timezone.utc).isoformat()
            }
            # upsert により既存レコードがあれば上書きする
            self.supabase.table("system_nodes").upsert(data).execute()
        except Exception as e:
            print(f"[DB Error] Failed to update heartbeat for node {node_id}: {e}")

    def get_my_target_status(self, node_id: str) -> str:
        """
        自身に対するダッシュボードからの命令(start/stop)を取得する
        """
        try:
            res = self.supabase.table("system_nodes").select("target_status").eq("node_id", node_id).execute()
            if res.data and len(res.data) > 0:
                return res.data[0].get("target_status", "unspecified")
            return "unspecified"
        except Exception as e:
            print(f"[DB Error] Failed to get target status for node {node_id}: {e}")
            return "unspecified"

    def get_all_nodes(self) -> list:
        """
        ダッシュボード用: すべてのノードの情報を取得する
        """
        try:
            res = self.supabase.table("system_nodes").select("*").execute()
            return res.data if res.data else []
        except Exception as e:
            print(f"[DB Error] Failed to get all nodes: {e}")
            return []

    def set_node_target_status(self, node_id: str, target_status: str):
        """
        ダッシュボード用: 特定ノードへの命令をDBに書き込む
        """
        try:
            self.supabase.table("system_nodes").update({"target_status": target_status}).eq("node_id", node_id).execute()
        except Exception as e:
            print(f"[DB Error] Failed to set target status for node {node_id}: {e}")

if __name__ == "__main__":
    # 接続テスト
    print("Testing DB Connection...")
    try:
        db = DBManager()
        stats = db.get_stats()
        print(f"Connection Successful! Current DB Stats: {stats}")
    except Exception as e:
        print(f"Connection Failed: {e}")
