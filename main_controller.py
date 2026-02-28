import time
import json
import os
import threading
import psutil
import hashlib
import shutil
from contextlib import contextmanager

try:
    import fcntl
except ImportError:
    fcntl = None

# ルートディレクトリのモジュールをインポートするためのパス設定
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_collector import web_crawler
from data_collector import wiki_downloader
from data_collector import rss_collector
from data_collector import news_collector
from data_collector import auto_discovery
from data_collector import arxiv_collector
from data_collector import docs_collector
from data_collector.db_manager import DBManager
from runtime.auto_tuner import detect_runtime_profile

# 状態保存用のファイルパス
STATUS_FILE = os.environ.get("STATUS_FILE", "system_status.json")

class SystemState:
    def __init__(self, is_dashboard=False):
        import uuid
        self.is_dashboard = is_dashboard
        # マスター(親機)かワーカー(子機)かの判定。環境変数で切り替える
        self.role = os.environ.get("SYSTEM_ROLE", "master").lower()
        self.heartbeat_interval_sec = max(5, int(os.environ.get("NODE_HEARTBEAT_INTERVAL_SEC", "10")))
        self.node_id = self._load_or_create_node_id()
        
        self.state = {
            "node_id": self.node_id,
            "is_running": False,
            "current_phase": "Idle",
            "role": self.role,
            "logs": [],
            "stats": {
                "collected_docs_wiki": 0,
                "collected_docs_web": 0,
                "collected_docs_rss": 0,
                "collected_docs_news": 0,
                "collected_docs_arxiv": 0,
                "collected_docs_docs": 0,
                "blocked_docs": 0,
                "dataset_size_mb": 0.0,
                "current_epoch": 0,
                "current_loss": 0.0,
                "current_val_loss": 0.0,
                "best_val_loss": 0.0,
                "active_workers": 0
            },
            "system": {
                "cpu_percent": 0.0,
                "ram_percent": 0.0,
                "cpu_cores": 0,
                "total_ram_gb": 0.0,
                "available_ram_gb": 0.0,
                "suggested_model_size": "small",
                "suggested_max_tokens": 256,
            }
        }
        self.db_manager = DBManager()
        self.status_lock_path = f"{STATUS_FILE}.lock"
        self._ckpt_stats_cache = None
        self._ckpt_stats_mtime = None
        self._heartbeat_thread = None
        self._heartbeat_stop = threading.Event()
        if not self.is_dashboard:
            self._start_heartbeat_thread()
            self.save()
        else:
            # Dashboard側は初期状態を上書き保存しない（学習状態リセットを避ける）
            self.load()

    def _load_or_create_node_id(self) -> str:
        import uuid
        env_node_id = (os.environ.get("NODE_ID") or "").strip()
        if env_node_id:
            return env_node_id
        # checkpoints は docker-compose でホストへ永続化されるため、
        # コンテナ再作成後も node_id を維持できる。
        node_id_file = os.environ.get("NODE_ID_FILE", os.path.join("checkpoints", "node_id.txt"))
        try:
            if os.path.exists(node_id_file):
                with open(node_id_file, "r", encoding="utf-8") as f:
                    existing = f.read().strip()
                    if existing:
                        return existing
            generated = str(uuid.uuid4())
            with open(node_id_file, "w", encoding="utf-8") as f:
                f.write(generated)
            return generated
        except Exception:
            return str(uuid.uuid4())

    def _start_heartbeat_thread(self):
        if self._heartbeat_thread and self._heartbeat_thread.is_alive():
            return

        def _loop():
            while not self._heartbeat_stop.is_set():
                try:
                    cpu_usage = psutil.cpu_percent()
                    ram_usage = psutil.virtual_memory().percent
                    current_status = "running" if self.state.get("is_running") else "paused"
                    self.db_manager.upsert_node_heartbeat(
                        node_id=self.node_id,
                        role=self.role,
                        status=current_status,
                        cpu_usage=cpu_usage,
                        ram_usage=ram_usage,
                        active_workers=int(self.state.get("stats", {}).get("active_workers", 0)),
                    )
                except Exception as e:
                    print(f"Heartbeat thread error: {e}")
                self._heartbeat_stop.wait(self.heartbeat_interval_sec)

        self._heartbeat_thread = threading.Thread(target=_loop, daemon=True)
        self._heartbeat_thread.start()

    @contextmanager
    def _status_file_lock(self):
        lock_handle = None
        try:
            if fcntl is not None:
                lock_handle = open(self.status_lock_path, "w", encoding="utf-8")
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            yield
        finally:
            if fcntl is not None and lock_handle is not None:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
                lock_handle.close()

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
            with self._status_file_lock():
                with open(STATUS_FILE, "w", encoding="utf-8") as f:
                    json.dump(self.state, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving status: {e}")

    def load(self):
        if os.path.exists(STATUS_FILE):
            try:
                with self._status_file_lock():
                    with open(STATUS_FILE, "r", encoding="utf-8") as f:
                        self.state = json.load(f)
            except Exception as e:
                print(f"Error loading status: {e}")
                
        # データベースから最新の統計情報を取得して結合する
        try:
            db_stats = self.db_manager.get_stats()
            self.state["stats"]["collected_docs_wiki"] = db_stats["collected_docs_wiki"]
            self.state["stats"]["collected_docs_web"]  = db_stats["collected_docs_web"]
            self.state["stats"]["collected_docs_rss"]  = db_stats.get("collected_docs_rss", 0)
            self.state["stats"]["collected_docs_news"] = db_stats.get("collected_docs_news", 0)
            self.state["stats"]["collected_docs_arxiv"] = db_stats.get("collected_docs_arxiv", 0)
            self.state["stats"]["collected_docs_docs"] = db_stats.get("collected_docs_docs", 0)
            self.state["stats"]["blocked_docs"] = db_stats.get("blocked_docs", 0)
            self.state["stats"]["dataset_size_mb"]     = db_stats["dataset_size_mb"]
        except Exception as e:
            print(f"Stats check error: {e}")

        self._restore_training_stats_from_checkpoint_if_needed()
            
        # 現在のCPU/RAM状態を常に最新に保つ
        cpu_usage, ram_usage = 0.0, 0.0
        try:
            import psutil
            cpu_usage = psutil.cpu_percent()
            ram_usage = psutil.virtual_memory().percent
            profile = detect_runtime_profile()
            self.state["system"]["cpu_percent"] = cpu_usage
            self.state["system"]["ram_percent"] = ram_usage
            self.state["system"]["cpu_cores"] = profile["cpu_cores"]
            self.state["system"]["total_ram_gb"] = profile["total_ram_gb"]
            self.state["system"]["available_ram_gb"] = profile["available_ram_gb"]
            self.state["system"]["suggested_model_size"] = profile["model_size"]
            self.state["system"]["suggested_max_tokens"] = profile["max_generate_tokens"]
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
                    ram_usage=ram_usage,
                    active_workers=int(self.state.get("stats", {}).get("active_workers", 0)),
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

    def _restore_training_stats_from_checkpoint_if_needed(self):
        stats = self.state.get("stats", {})
        already_has_values = any([
            float(stats.get("current_epoch", 0) or 0) > 0,
            float(stats.get("current_loss", 0) or 0) > 0,
            float(stats.get("current_val_loss", 0) or 0) > 0,
            float(stats.get("best_val_loss", 0) or 0) > 0,
        ])
        if already_has_values:
            return

        base_dir = os.path.dirname(os.path.abspath(__file__))
        ckpt_path = os.path.join(base_dir, "checkpoints", "ckpt_latest.pt")
        if not os.path.exists(ckpt_path):
            return
        try:
            mtime = os.path.getmtime(ckpt_path)
            if self._ckpt_stats_cache is not None and self._ckpt_stats_mtime == mtime:
                cached = self._ckpt_stats_cache
            else:
                import torch
                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                cached = {
                    "step": int(ckpt.get("step", 0) or 0),
                    "train_loss": float(ckpt.get("loss", 0.0) or 0.0),
                    "val_loss": ckpt.get("val_loss"),
                    "best_val_loss": ckpt.get("best_val_loss"),
                }
                self._ckpt_stats_cache = cached
                self._ckpt_stats_mtime = mtime

            if int(cached.get("step", 0)) > 0:
                stats["current_epoch"] = int(cached.get("step", 0))
                stats["current_loss"] = round(float(cached.get("train_loss", 0.0) or 0.0), 4)
                val_loss = cached.get("val_loss")
                best_val = cached.get("best_val_loss")
                stats["current_val_loss"] = round(float(val_loss), 4) if val_loss is not None else 0.0
                stats["best_val_loss"] = round(float(best_val), 4) if best_val is not None else 0.0
        except Exception as e:
            print(f"Checkpoint stats restore error: {e}")

    def set_running(self, running: bool):
        self.state["is_running"] = running
        self.save()


def _stable_hash(value: str) -> int:
    return int(hashlib.sha1(value.encode("utf-8")).hexdigest(), 16)


def _assign_seed_urls_to_node(seed_urls: list[str], node_id: str, active_node_ids: list[str]) -> list[str]:
    if not seed_urls:
        return []
    node_ids = sorted(set([n for n in active_node_ids if n]))
    if node_id not in node_ids:
        node_ids.append(node_id)
        node_ids.sort()
    total = max(1, len(node_ids))
    my_index = node_ids.index(node_id)
    assigned = [u for u in seed_urls if (_stable_hash(u) % total) == my_index]
    if assigned:
        return assigned
    # シードが少ない場合に空割り当てを避ける
    fallback_idx = my_index % len(seed_urls)
    return [seed_urls[fallback_idx]]


def _checkpoint_paths() -> tuple[str, str]:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_dir = os.path.join(base_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    latest = os.path.join(checkpoint_dir, "ckpt_latest.pt")
    backup = os.path.join(checkpoint_dir, "ckpt_latest.pretrain.bak.pt")
    return latest, backup


def _run_training_with_retry(state: SystemState, max_steps: int):
    from model import trainer

    retries = max(0, int(os.environ.get("TRAIN_RETRY_MAX", "2")))
    latest_ckpt, backup_ckpt = _checkpoint_paths()
    if os.path.exists(latest_ckpt):
        try:
            shutil.copy2(latest_ckpt, backup_ckpt)
            state.log("[TrainJob] Backed up latest checkpoint before training.")
        except Exception as e:
            state.log(f"[TrainJob] Backup warning: {e}")

    attempt = 0
    while True:
        attempt += 1
        try:
            state.log(f"[TrainJob] Attempt {attempt}/{retries + 1}")
            return trainer.train_step(max_steps=max_steps, log_fn=state.log)
        except Exception as e:
            state.log(f"[TrainJob] Attempt {attempt} failed: {e}")
            if os.path.exists(backup_ckpt):
                try:
                    shutil.copy2(backup_ckpt, latest_ckpt)
                    state.log("[TrainJob] Rolled back ckpt_latest.pt from backup.")
                except Exception as restore_err:
                    state.log(f"[TrainJob] Rollback error: {restore_err}")
            if attempt > retries:
                raise
            time.sleep(float(os.environ.get("TRAIN_RETRY_BACKOFF_SEC", "2.0")))


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
            # (最小1スレッド、最大は環境変数 MAX_CRAWLER_WORKERS で制限)
            allowed_by_cpu = max(1, int((100.0 - cpu_usage) / 5.0))
            allowed_by_mem = max(1, int(available_mem_gb / 0.2))
            
            target_workers = min(allowed_by_cpu, allowed_by_mem)
            realtime_tune = os.environ.get("AUTO_TUNE_REALTIME", "1") == "1"
            if realtime_tune:
                profile = detect_runtime_profile()
                max_workers_limit = int(os.environ.get("MAX_CRAWLER_WORKERS", str(profile["max_crawler_workers"])))
            else:
                max_workers_limit = int(os.environ.get("MAX_CRAWLER_WORKERS", "8"))
            target_workers = min(max_workers_limit, max(1, target_workers))
            
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
            fixed_source_mode = os.environ.get("ENABLE_FIXED_SOURCE_MODE", "1") == "1"
            enable_web_crawler = os.environ.get("ENABLE_WEB_CRAWLER", "0") == "1"
            if fixed_source_mode:
                custom_web_seeds = os.environ.get("WEB_SEED_URLS", "").strip()
                if custom_web_seeds:
                    seeds = [u.strip() for u in custom_web_seeds.split(",") if u.strip()]
                else:
                    seeds = []
                state.log("[Collection] Fixed source mode is enabled (RSS/News/arXiv/Docs priority).")

            if os.environ.get("ENABLE_AUTO_DISCOVERY", "1") == "1" and not fixed_source_mode:
                try:
                    discovered = auto_discovery.discover_seed_urls(
                        max_urls=int(os.environ.get("AUTO_DISCOVERY_SEEDS_PER_CYCLE", "24")),
                        max_results_per_query=int(os.environ.get("AUTO_DISCOVERY_RESULTS_PER_QUERY", "8")),
                    )
                    if discovered:
                        seeds.extend(discovered)
                        state.log(f"[AutoDiscovery] Added {len(discovered)} web seeds from live search.")
                except Exception as e:
                    state.log(f"Auto discovery error: {e}")
            elif os.environ.get("ENABLE_AUTO_DISCOVERY", "1") == "1" and fixed_source_mode:
                state.log("[Collection] AutoDiscovery is ignored because fixed source mode is enabled.")

            if enable_web_crawler and seeds:
                try:
                    include_master = os.environ.get("COLLECTION_INCLUDE_MASTER", "1") == "1"
                    active_nodes = state.db_manager.get_active_collector_nodes(
                        online_window_sec=int(os.environ.get("COLLECTION_NODE_WINDOW_SEC", "60")),
                        include_master=include_master,
                    )
                    active_node_ids = [n.get("node_id") for n in active_nodes if n.get("node_id")]
                    assigned_seeds = _assign_seed_urls_to_node(seeds, state.node_id, active_node_ids)
                    max_pages_total = int(os.environ.get("MAX_CRAWL_PAGES_PER_CYCLE", "200"))
                    node_count = max(1, len(set(active_node_ids + [state.node_id])))
                    max_pages_for_this_node = max(20, max_pages_total // node_count)
                    state.log(
                        f"[Shard] node={state.node_id[-8:]} assigned_seeds={len(assigned_seeds)}/{len(seeds)} "
                        f"active_nodes={node_count} max_pages={max_pages_for_this_node}"
                    )
                    web_crawler.start_crawler(
                        assigned_seeds,
                        max_workers=target_workers,
                        max_pages=max_pages_for_this_node,
                    )
                except Exception as e:
                    state.log(f"Crawler error: {e}")
            elif enable_web_crawler and not seeds:
                state.log("[Collection] Web crawler is enabled but no seed URLs are configured.")
            else:
                state.log("[Collection] Web crawler is disabled (ENABLE_WEB_CRAWLER=0).")

            # RSSソースの追加収集（Wikipedia以外）
            if os.environ.get("ENABLE_RSS_COLLECTOR", "1") == "1":
                try:
                    rss_result = rss_collector.collect_from_rss(
                        max_items_per_feed=int(os.environ.get("RSS_ITEMS_PER_FEED", "5"))
                    )
                    if isinstance(rss_result, dict):
                        state.log(
                            "[Data Source] RSS "
                            f"saved={rss_result.get('saved', 0)} "
                            f"failed_fetch={rss_result.get('fetch_failed', 0)} "
                            f"failed_feed={rss_result.get('feed_failed', 0)} "
                            f"skipped_dup={rss_result.get('skipped_duplicate', 0)} "
                            f"skipped_short={rss_result.get('skipped_short', 0)}"
                        )
                    else:
                        state.log(f"[Data Source] RSS collector saved {rss_result} articles.")
                except Exception as e:
                    state.log(f"RSS collector error: {e}")

            if os.environ.get("ENABLE_NEWS_COLLECTOR", "1") == "1":
                try:
                    news_result = news_collector.collect_news(
                        max_items_per_feed=int(os.environ.get("NEWS_ITEMS_PER_FEED", "5")),
                        max_hn_items=int(os.environ.get("HN_ITEMS", "15")),
                    )
                    if isinstance(news_result, dict):
                        state.log(
                            "[Data Source] News "
                            f"saved={news_result.get('saved', 0)} "
                            f"failed_fetch={news_result.get('fetch_failed', 0)} "
                            f"failed_feed={news_result.get('feed_failed', 0)} "
                            f"failed_hn={news_result.get('hn_failed', 0)} "
                            f"skipped_dup={news_result.get('skipped_duplicate', 0)} "
                            f"skipped_short={news_result.get('skipped_short', 0)}"
                        )
                    else:
                        state.log(f"[Data Source] News collector saved {news_result} articles.")
                except Exception as e:
                    state.log(f"News collector error: {e}")

            if os.environ.get("ENABLE_ARXIV_COLLECTOR", "1") == "1":
                try:
                    arxiv_result = arxiv_collector.collect_arxiv(
                        max_results=int(os.environ.get("ARXIV_MAX_RESULTS", "30"))
                    )
                    if isinstance(arxiv_result, dict):
                        state.log(
                            "[Data Source] arXiv "
                            f"saved={arxiv_result.get('saved', 0)} "
                            f"failed={arxiv_result.get('failed', 0)} "
                            f"skipped_dup={arxiv_result.get('skipped_duplicate', 0)}"
                        )
                    else:
                        state.log(f"[Data Source] arXiv collector saved {arxiv_result} abstracts.")
                except Exception as e:
                    state.log(f"arXiv collector error: {e}")

            if os.environ.get("ENABLE_DOCS_COLLECTOR", "1") == "1":
                try:
                    docs_result = docs_collector.collect_docs()
                    if isinstance(docs_result, dict):
                        state.log(
                            "[Data Source] Docs "
                            f"saved={docs_result.get('saved', 0)} "
                            f"failed={docs_result.get('failed', 0)} "
                            f"skipped_dup={docs_result.get('skipped_duplicate', 0)} "
                            f"skipped_short={docs_result.get('skipped_short', 0)}"
                        )
                    else:
                        state.log(f"[Data Source] Docs collector saved {docs_result} pages.")
                except Exception as e:
                    state.log(f"Docs collector error: {e}")

            try:
                db_insert_metrics = state.db_manager.get_and_reset_insert_metrics()
                if db_insert_metrics:
                    parts = []
                    for source_type in sorted(db_insert_metrics.keys()):
                        item = db_insert_metrics[source_type]
                        parts.append(
                            f"{source_type}:ok={item.get('ok', 0)}/ng={item.get('ng', 0)}"
                        )
                    state.log(f"[DB Write Summary] {' | '.join(parts)}")
            except Exception as e:
                state.log(f"[DB Write Summary] error: {e}")
            
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
            skip_training_this_cycle = False
            try:
                from data_preprocessor import prepare_dataset
                # 取得済みのデータを集めて学習用の高速バイナリ(train.bin)に変換する
                prep_result = prepare_dataset.prepare_dataset(
                    vocab_size=int(os.environ.get("TRAIN_VOCAB_SIZE", "8000")),
                    progress_cb=state.log,
                )
                if isinstance(prep_result, dict):
                    state.log(
                        "[Dataset] "
                        f"docs={prep_result.get('total_docs', 0)} "
                        f"blocked={prep_result.get('blocked_docs', 0)} "
                        f"filtered={prep_result.get('filtered_docs', 0)} "
                        f"dedup={prep_result.get('duplicate_docs', 0)}"
                    )
                    if (prep_result.get("train_tokens", 0) + prep_result.get("val_tokens", 0)) <= 0:
                        skip_training_this_cycle = True
                elif prep_result is False:
                    skip_training_this_cycle = True
            except Exception as e:
                state.log(f"Preprocess error: {e}")
                skip_training_this_cycle = True

            # ---------------------------------------------------------
            # フェーズ3: モデル学習
            # ---------------------------------------------------------
            if not state.state["is_running"]: continue
            if skip_training_this_cycle:
                state.log("[Training] Skipped: dataset has no trainable tokens in this cycle.")
                state.log("One pipeline cycle completed. Sleeping before next cycle.")
                time.sleep(5)
                continue
            state.update_phase("Training")
            state.log("Running training loop for a few steps...")
            
            try:
                # 設定されたステップ数だけ学習し、重み情報（チェックポイント）を保存して帰ってくる
                train_result = _run_training_with_retry(
                    state,
                    max_steps=int(os.environ.get("TRAIN_STEPS_PER_CYCLE", "50")),
                )

                # 旧戻り値(tuple)との互換を維持
                if isinstance(train_result, dict):
                    state.state["stats"]["current_epoch"] = train_result.get("epoch", 0)
                    train_loss = train_result.get("train_loss")
                    val_loss = train_result.get("val_loss")
                    best_val = train_result.get("best_val_loss")
                    state.state["stats"]["current_loss"] = round(train_loss, 4) if train_loss is not None else 0.0
                    state.state["stats"]["current_val_loss"] = round(val_loss, 4) if val_loss is not None else 0.0
                    state.state["stats"]["best_val_loss"] = round(best_val, 4) if best_val is not None else 0.0
                else:
                    epoch, loss = train_result
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
