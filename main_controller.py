import time
import json
import os
import threading
import psutil

# ルートディレクトリのモジュールをインポートするためのパス設定
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_collector import web_crawler
from data_collector import wiki_downloader
from data_collector.db_manager import DBManager

# 状態保存用のファイルパス
STATUS_FILE = "system_status.json"

class SystemState:
    def __init__(self, is_dashboard=False):
        import uuid
        self.is_dashboard = is_dashboard
        # マスター(親機)かワーカー(子機)かの判定。環境変数で切り替える
        self.role = os.environ.get("SYSTEM_ROLE", "master").lower()
        self.node_id = str(uuid.uuid4())
        
        self.state = {
            "node_id": self.node_id,
            "is_running": False,
            "current_phase": "Idle",
            "role": self.role,
            "logs": [],
            "stats": {
                "collected_docs_wiki": 0,
                "collected_docs_web": 0,
                "dataset_size_mb": 0.0,
                "current_epoch": 0,
                "current_loss": 0.0,
                "active_workers": 0
            },
            "system": {
                "cpu_percent": 0.0,
                "ram_percent": 0.0
            }
        }
        self.db_manager = DBManager()
        self.save()

    def update_phase(self, phase):
        self.state["current_phase"] = phase
        self.log(f"Phase changed to: {phase}")
        self.save()

    def log(self, message):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        self.state["logs"].append(f"[{timestamp}] {message}")
        # 最新の50件のみ保持
        self.state["logs"] = self.state["logs"][-50:]
        print(f"[{timestamp}] {message}")
        self.save()

    def save(self):
        try:
            with open(STATUS_FILE, "w", encoding="utf-8") as f:
                json.dump(self.state, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving status: {e}")

    def load(self):
        if os.path.exists(STATUS_FILE):
            try:
                with open(STATUS_FILE, "r", encoding="utf-8") as f:
                    self.state = json.load(f)
            except Exception as e:
                print(f"Error loading status: {e}")
                
        # データベースから最新の統計情報を取得して結合する
        try:
            db_stats = self.db_manager.get_stats()
            self.state["stats"]["collected_docs_wiki"] = db_stats["collected_docs_wiki"]
            self.state["stats"]["collected_docs_web"]  = db_stats["collected_docs_web"]
            self.state["stats"]["dataset_size_mb"]     = db_stats["dataset_size_mb"]
        except Exception as e:
            print(f"Stats check error: {e}")
            
        # 現在のCPU/RAM状態を常に最新に保つ
        cpu_usage, ram_usage = 0.0, 0.0
        try:
            import psutil
            cpu_usage = psutil.cpu_percent()
            ram_usage = psutil.virtual_memory().percent
            self.state["system"]["cpu_percent"] = cpu_usage
            self.state["system"]["ram_percent"] = ram_usage
        except:
            pass

        # --- Node Heartbeat & Remote Control ---
        if not self.is_dashboard:
            try:
                current_status = "running" if self.state.get("is_running") else "paused"
                self.db_manager.upsert_node_heartbeat(
                    node_id=self.node_id,
                    role=self.role,
                    status=current_status,
                    cpu_usage=cpu_usage,
                    ram_usage=ram_usage
                )
                
                target = self.db_manager.get_my_target_status(self.node_id)
                if target == "start" and not self.state["is_running"]:
                    self.log(f"[Network] Received remote command: START for node {self.node_id[-4:]}")
                    self.set_running(True)
                    self.db_manager.set_node_target_status(self.node_id, "unspecified")
                elif target == "stop" and self.state["is_running"]:
                    self.log(f"[Network] Received remote command: STOP for node {self.node_id[-4:]}")
                    self.set_running(False)
                    self.db_manager.set_node_target_status(self.node_id, "unspecified")
            except Exception as e:
                print(f"Heartbeat error: {e}")

    def set_running(self, running: bool):
        self.state["is_running"] = running
        self.save()


def run_pipeline(state: SystemState):
    """
    メインの処理ループ。バックグラウンドスレッドで実行される。
    """
    state.log("Pipeline started.")
    wiki_downloaded = False  # 初回のみWikiダンプをダウンロードするフラグ
    
    while True:
        state.load() # 外部からの操作（停止命令など）を読み込む
        
        if not state.state["is_running"]:
            state.log("Pipeline paused. Waiting...")
            time.sleep(5)
            continue
            
        try:
            # ---------------------------------------------------------
            # フェーズ0: Wikipedia ダンプの一括ダウンロード（初回起動時のみ）
            # ---------------------------------------------------------
            if not wiki_downloaded:
                state.update_phase("Downloading Wikipedia")
                state.log("[Phase 0] Starting Wikipedia dump download (first-time only)...")
                try:
                    article_count = wiki_downloader.download_wikipedia_dump("ja")
                    state.log(f"[Phase 0] Wikipedia download complete. Total articles available: {article_count}")
                except Exception as e:
                    state.log(f"[Phase 0] Wikipedia download error (skipping): {e}")
                wiki_downloaded = True  # 成功・失敗に関わらず次サイクルはスキップ
            # ---------------------------------------------------------
            # フェーズ1: データ収集 (Web Crawl)
            # ---------------------------------------------------------
            state.update_phase("Data Collection")
            
            # --- 動的リソース監視（オートスケール） ---
            # CPUのアイドル割合（100 - 使用率）と空きメモリから適正な並列数を算出
            cpu_usage = psutil.cpu_percent(interval=1.0)
            mem_info = psutil.virtual_memory()
            available_mem_gb = mem_info.available / (1024 ** 3)
            
            # 基準: CPU空き5%につき1ワーカー、メモリ空き0.2GBにつき1ワーカーとし、厳しい方を採用
            # (最小1スレッド、最大40スレッドに安全制限)
            allowed_by_cpu = max(1, int((100.0 - cpu_usage) / 5.0))
            allowed_by_mem = max(1, int(available_mem_gb / 0.2))
            
            target_workers = min(allowed_by_cpu, allowed_by_mem)
            target_workers = min(40, max(1, target_workers))
            
            # ダッシュボード表示用に現在決定したワーカー数を記録
            state.state["stats"]["active_workers"] = target_workers
            state.save()
            
            state.log(f"[Auto-Scale] CPU Usage:{cpu_usage}%, Free Mem:{available_mem_gb:.1f}GB -> Launching {target_workers} workers.")
            
            # 分散ワーカーの起動：複数のシードURLからクローリング開始
            seeds = [
                # テクノロジー系
                "https://ja.wikipedia.org/wiki/Python",
                "https://ja.wikipedia.org/wiki/Linux",
                "https://ja.wikipedia.org/wiki/Docker",
                "https://ja.wikipedia.org/wiki/%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92",
                "https://ja.wikipedia.org/wiki/%E4%BA%BA%E5%B7%A5%E7%9F%A5%E8%83%BD",
                "https://ja.wikipedia.org/wiki/%E6%B7%B1%E5%B1%A4%E5%AD%A6%E7%BF%92",
                "https://ja.wikipedia.org/wiki/%E3%83%8B%E3%83%A5%E3%83%BC%E3%83%A9%E3%83%AB%E3%83%8D%E3%83%83%E3%83%88%E3%83%AF%E3%83%BC%E3%82%AF",
                "https://ja.wikipedia.org/wiki/%E8%87%AA%E7%84%B6%E8%A8%80%E8%AA%9E%E5%87%A6%E7%90%86",
                "https://ja.wikipedia.org/wiki/%E3%83%88%E3%83%A9%E3%83%B3%E3%82%B9%E3%83%95%E3%82%A9%E3%83%BC%E3%83%9E%E3%83%BC_(%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92%E3%83%A2%E3%83%87%E3%83%AB)",
                "https://ja.wikipedia.org/wiki/%E3%82%AF%E3%83%A9%E3%82%A6%E3%83%89%E3%82%B3%E3%83%B3%E3%83%94%E3%83%A5%E3%83%BC%E3%83%86%E3%82%A3%E3%83%B3%E3%82%B0",
                "https://ja.wikipedia.org/wiki/Kubernetes",
                "https://ja.wikipedia.org/wiki/Rust_(%E3%83%97%E3%83%AD%E3%82%B0%E3%83%A9%E3%83%9F%E3%83%B3%E3%82%B0%E8%A8%80%E8%AA%9E)",
                "https://ja.wikipedia.org/wiki/Go_(%E3%83%97%E3%83%AD%E3%82%B0%E3%83%A9%E3%83%9F%E3%83%B3%E3%82%B0%E8%A8%80%E8%AA%9E)",
                "https://ja.wikipedia.org/wiki/JavaScript",
                "https://ja.wikipedia.org/wiki/TypeScript",
                "https://ja.wikipedia.org/wiki/TensorFlow",
                "https://ja.wikipedia.org/wiki/PyTorch",
                "https://ja.wikipedia.org/wiki/OpenAI",
                "https://ja.wikipedia.org/wiki/%E3%82%B3%E3%83%B3%E3%83%94%E3%83%A5%E3%83%BC%E3%82%BF%E3%83%93%E3%82%B8%E3%83%A7%E3%83%B3",
                "https://ja.wikipedia.org/wiki/GPU",
                # 科学・数学
                "https://ja.wikipedia.org/wiki/%E7%B5%B1%E8%A8%88%E5%AD%A6",
                "https://ja.wikipedia.org/wiki/%E7%B7%9A%E5%BD%A2%E4%BB%A3%E6%95%B0",
                "https://ja.wikipedia.org/wiki/%E5%BE%AE%E7%A9%8D%E5%88%86%E5%AD%A6",
                "https://ja.wikipedia.org/wiki/%E7%A2%BA%E7%8E%87%E8%AB%96",
                "https://ja.wikipedia.org/wiki/%E9%87%8F%E5%AD%90%E5%8A%9B%E5%AD%A6",
                "https://ja.wikipedia.org/wiki/%E7%9B%B8%E5%AF%BE%E6%80%A7%E7%90%86%E8%AB%96",
                "https://ja.wikipedia.org/wiki/%E5%8C%96%E5%AD%A6",
                "https://ja.wikipedia.org/wiki/%E7%94%9F%E7%89%A9%E5%AD%A6",
                # 歴史・文化
                "https://ja.wikipedia.org/wiki/%E6%97%A5%E6%9C%AC%E3%81%AE%E6%AD%B4%E5%8F%B2",
                "https://ja.wikipedia.org/wiki/%E7%AC%AC%E4%BA%8C%E6%AC%A1%E4%B8%96%E7%95%8C%E5%A4%A7%E6%88%A6",
                "https://ja.wikipedia.org/wiki/%E5%93%B2%E5%AD%A6",
                "https://ja.wikipedia.org/wiki/%E7%B5%8C%E6%B8%88%E5%AD%A6",
            ]
            try:
                # 動的に計算された target_workers スレッドで並列実行
                web_crawler.start_crawler(seeds, max_workers=target_workers, max_pages=200)
            except Exception as e:
                state.log(f"Crawler error: {e}")
            
            # ---------------------------------------------------------
            # フェーズ2以降（マスターノードのみ実行）
            # ---------------------------------------------------------
            if state.role == "worker":
                state.log("[Worker Mode] Skipping preprocessing and training. Waiting for next crawl cycle...")
                time.sleep(10)
                continue
            
            # ---------------------------------------------------------
            # フェーズ2: 前処理・トークナイズ (Master Only)
            # ---------------------------------------------------------
            if not state.state["is_running"]: continue
            state.update_phase("Preprocessing")
            state.log("Preprocessing and tokenizing text data...")
            try:
                from data_preprocessor import prepare_dataset
                # 取得済みのデータを集めて学習用の高速バイナリ(train.bin)に変換する
                prepare_dataset.prepare_dataset(vocab_size=8000)
            except Exception as e:
                state.log(f"Preprocess error: {e}")

            # ---------------------------------------------------------
            # フェーズ3: モデル学習
            # ---------------------------------------------------------
            if not state.state["is_running"]: continue
            state.update_phase("Training")
            state.log("Running training loop for a few steps...")
            
            try:
                from model import trainer
                # 設定されたステップ数だけ学習し、重み情報（チェックポイント）を保存して帰ってくる
                epoch, loss = trainer.train_step(max_steps=50)
                
                # 統計情報の更新
                state.state["stats"]["current_epoch"] = epoch
                state.state["stats"]["current_loss"] = round(loss, 4) if loss else 0.0
                state.save()
            except Exception as e:
                import traceback
                error_info = traceback.format_exc()
                state.log(f"Training error: {e}\n{error_info}")

            state.log("One pipeline cycle completed. Sleeping before next cycle.")
            time.sleep(5)
            
        except Exception as e:
            state.log(f"Error in pipeline: {e}")
            state.set_running(False)
            time.sleep(5)

if __name__ == "__main__":
    # このスクリプト自体が直接実行された場合のテスト用エントリポイント
    state_manager = SystemState()
    state_manager.set_running(True)
    run_pipeline(state_manager)
