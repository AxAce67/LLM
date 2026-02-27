import time
import json
import os
import threading
import psutil

# ルートディレクトリのモジュールをインポートするためのパス設定
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_collector import web_crawler
from data_collector.db_manager import DBManager

# 状態保存用のファイルパス
STATUS_FILE = "system_status.json"

class SystemState:
    def __init__(self):
        import uuid
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
    
    while True:
        state.load() # 外部からの操作（停止命令など）を読み込む
        
        if not state.state["is_running"]:
            state.log("Pipeline paused. Waiting...")
            time.sleep(5)
            continue
            
        try:
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
            seeds = ["https://ja.wikipedia.org/wiki/Python", "https://ja.wikipedia.org/wiki/Linux", "https://ja.wikipedia.org/wiki/Docker"]
            try:
                # 動的に計算された target_workers スレッドで並列実行
                web_crawler.start_crawler(seeds, max_workers=target_workers, max_pages=15)
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
