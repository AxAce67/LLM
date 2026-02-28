import datetime
import os
import threading
import time
from typing import Generator

import psycopg2
from dotenv import load_dotenv
from psycopg2.extras import DictCursor
from data_preprocessor.quality_filter import quality_score

# 環境変数の読み込み (.env ファイルがあればそこから、なければOSの環境変数)
load_dotenv()


class DBManager:
    _insert_metrics_lock = threading.Lock()
    _insert_metrics = {}
    _fs_stats_lock = threading.Lock()
    _fs_stats_cache = None
    _fs_stats_cached_at = 0.0

    def __init__(self):
        self.database_url = os.environ.get("DATABASE_URL")
        if not self.database_url:
            raise ValueError("DATABASE_URL must be set.")
        self._init_postgres_schema()

    def _connect(self):
        return psycopg2.connect(self.database_url)

    def _init_postgres_schema(self):
        query = """
        CREATE TABLE IF NOT EXISTS crawled_data (
            id BIGSERIAL PRIMARY KEY,
            url TEXT UNIQUE NOT NULL,
            domain TEXT NOT NULL,
            title VARCHAR(255) NOT NULL DEFAULT '',
            content TEXT NOT NULL,
            char_count INTEGER NOT NULL DEFAULT 0,
            source_type VARCHAR(32) NOT NULL,
            language VARCHAR(16) NOT NULL DEFAULT 'ja',
            quality_score DOUBLE PRECISION NOT NULL DEFAULT 0.0,
            license_tag TEXT NOT NULL DEFAULT 'unknown',
            allowed_for_training BOOLEAN NOT NULL DEFAULT TRUE,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        CREATE INDEX IF NOT EXISTS idx_crawled_data_source_type ON crawled_data(source_type);
        CREATE INDEX IF NOT EXISTS idx_crawled_data_created_at ON crawled_data(created_at);
        CREATE INDEX IF NOT EXISTS idx_crawled_data_allowed_for_training ON crawled_data(allowed_for_training);

        CREATE TABLE IF NOT EXISTS source_policies (
            id BIGSERIAL PRIMARY KEY,
            domain_pattern TEXT UNIQUE NOT NULL,
            source_type VARCHAR(32) NOT NULL DEFAULT 'web',
            license_tag TEXT NOT NULL DEFAULT 'unknown',
            allow_training BOOLEAN NOT NULL DEFAULT TRUE,
            base_weight DOUBLE PRECISION NOT NULL DEFAULT 1.0,
            notes TEXT NOT NULL DEFAULT '',
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        CREATE INDEX IF NOT EXISTS idx_source_policies_source_type ON source_policies(source_type);

        CREATE TABLE IF NOT EXISTS system_nodes (
            node_id TEXT PRIMARY KEY,
            role VARCHAR(32) NOT NULL,
            status VARCHAR(32) NOT NULL DEFAULT 'paused',
            cpu_usage DOUBLE PRECISION NOT NULL DEFAULT 0,
            ram_usage DOUBLE PRECISION NOT NULL DEFAULT 0,
            active_workers INTEGER NOT NULL DEFAULT 0,
            target_status VARCHAR(32) NOT NULL DEFAULT 'unspecified',
            last_heartbeat TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );

        CREATE TABLE IF NOT EXISTS evaluation_runs (
            id BIGSERIAL PRIMARY KEY,
            model_tag TEXT NOT NULL DEFAULT 'default',
            avg_score DOUBLE PRECISION NOT NULL DEFAULT 0,
            result_json JSONB NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        CREATE INDEX IF NOT EXISTS idx_evaluation_runs_created_at ON evaluation_runs(created_at DESC);

        CREATE TABLE IF NOT EXISTS model_versions (
            id BIGSERIAL PRIMARY KEY,
            model_tag TEXT NOT NULL,
            checkpoint_path TEXT NOT NULL,
            source_checkpoint TEXT NOT NULL DEFAULT '',
            avg_score DOUBLE PRECISION NOT NULL DEFAULT 0,
            promoted BOOLEAN NOT NULL DEFAULT FALSE,
            notes TEXT NOT NULL DEFAULT '',
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        CREATE INDEX IF NOT EXISTS idx_model_versions_created_at ON model_versions(created_at DESC);

        CREATE TABLE IF NOT EXISTS dataset_versions (
            id BIGSERIAL PRIMARY KEY,
            dataset_tag TEXT NOT NULL,
            train_tokens BIGINT NOT NULL DEFAULT 0,
            val_tokens BIGINT NOT NULL DEFAULT 0,
            total_docs BIGINT NOT NULL DEFAULT 0,
            blocked_docs BIGINT NOT NULL DEFAULT 0,
            filtered_docs BIGINT NOT NULL DEFAULT 0,
            duplicate_docs BIGINT NOT NULL DEFAULT 0,
            source_breakdown JSONB NOT NULL DEFAULT '{}'::jsonb,
            metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        CREATE INDEX IF NOT EXISTS idx_dataset_versions_created_at ON dataset_versions(created_at DESC);

        CREATE INDEX IF NOT EXISTS idx_crawled_data_fts
            ON crawled_data USING GIN (to_tsvector('simple', coalesce(title, '') || ' ' || coalesce(content, '')));
        """
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(query)
                # Backward-compatible schema fixes for existing deployments
                cur.execute("ALTER TABLE crawled_data ADD COLUMN IF NOT EXISTS quality_score DOUBLE PRECISION NOT NULL DEFAULT 0.0")
                cur.execute("ALTER TABLE crawled_data ADD COLUMN IF NOT EXISTS license_tag TEXT NOT NULL DEFAULT 'unknown'")
                cur.execute("ALTER TABLE crawled_data ADD COLUMN IF NOT EXISTS allowed_for_training BOOLEAN NOT NULL DEFAULT TRUE")
                cur.execute("ALTER TABLE source_policies ADD COLUMN IF NOT EXISTS source_type VARCHAR(32) NOT NULL DEFAULT 'web'")
                cur.execute("ALTER TABLE source_policies ADD COLUMN IF NOT EXISTS license_tag TEXT NOT NULL DEFAULT 'unknown'")
                cur.execute("ALTER TABLE source_policies ADD COLUMN IF NOT EXISTS allow_training BOOLEAN NOT NULL DEFAULT TRUE")
                cur.execute("ALTER TABLE source_policies ADD COLUMN IF NOT EXISTS base_weight DOUBLE PRECISION NOT NULL DEFAULT 1.0")
                cur.execute("ALTER TABLE source_policies ADD COLUMN IF NOT EXISTS notes TEXT NOT NULL DEFAULT ''")
                cur.execute("ALTER TABLE source_policies ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()")
                cur.execute("ALTER TABLE source_policies ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()")
                cur.execute("ALTER TABLE system_nodes ADD COLUMN IF NOT EXISTS role VARCHAR(32) NOT NULL DEFAULT 'worker'")
                cur.execute("ALTER TABLE system_nodes ADD COLUMN IF NOT EXISTS status VARCHAR(32) NOT NULL DEFAULT 'paused'")
                cur.execute("ALTER TABLE system_nodes ADD COLUMN IF NOT EXISTS cpu_usage DOUBLE PRECISION NOT NULL DEFAULT 0")
                cur.execute("ALTER TABLE system_nodes ADD COLUMN IF NOT EXISTS ram_usage DOUBLE PRECISION NOT NULL DEFAULT 0")
                cur.execute("ALTER TABLE system_nodes ADD COLUMN IF NOT EXISTS active_workers INTEGER NOT NULL DEFAULT 0")
                cur.execute("ALTER TABLE system_nodes ADD COLUMN IF NOT EXISTS target_status VARCHAR(32) NOT NULL DEFAULT 'unspecified'")
                cur.execute("ALTER TABLE system_nodes ADD COLUMN IF NOT EXISTS last_heartbeat TIMESTAMPTZ NOT NULL DEFAULT NOW()")
            conn.commit()
        self._bootstrap_default_source_policies()

    def _bootstrap_default_source_policies(self):
        defaults = [
            ("wikipedia.org", "wikipedia", "cc-by-sa", True, 1.4, "Wikipedia content"),
            ("arxiv.org", "arxiv", "arxiv", True, 1.3, "arXiv abstracts"),
            ("github.com", "code", "repo-license-unknown", False, 0.8, "Requires explicit repo license checks"),
            ("docs.python.org", "docs", "python-docs-license", True, 1.3, "Python docs"),
            ("pytorch.org", "docs", "site-specific", True, 1.25, "PyTorch docs"),
            ("postgresql.org", "docs", "postgresql-docs-license", True, 1.2, "PostgreSQL docs"),
            ("kubernetes.io", "docs", "cc-by-4.0", True, 1.2, "Kubernetes docs"),
            ("openai.com", "rss", "site-specific", True, 1.1, "OpenAI news/blog"),
            ("aws.amazon.com", "rss", "site-specific", True, 1.0, "AWS blog"),
            ("googleblog.com", "rss", "site-specific", True, 1.0, "Google developer blog"),
            ("research.google", "rss", "site-specific", True, 1.0, "Google research blog"),
            ("huggingface.co", "rss", "site-specific", True, 1.1, "Hugging Face blog/docs"),
            ("databricks.com", "rss", "site-specific", True, 1.0, "Databricks engineering blog"),
            ("cncf.io", "rss", "site-specific", True, 1.0, "CNCF blog"),
            ("theverge.com", "news", "site-specific", True, 0.9, "News"),
            ("wired.com", "news", "site-specific", True, 0.9, "News"),
            ("arstechnica.com", "news", "site-specific", True, 0.9, "News"),
            ("bbc.co.uk", "news", "site-specific", True, 0.9, "News"),
            ("nature.com", "news", "site-specific", True, 1.0, "Research news"),
            ("infoq.com", "news", "site-specific", True, 1.0, "Engineering news"),
            ("zenn.dev", "news", "site-specific", True, 1.0, "Japanese technical articles"),
            ("qiita.com", "news", "site-specific", True, 0.95, "Japanese technical articles"),
            ("numpy.org", "docs", "site-specific", True, 1.2, "NumPy docs"),
            ("pandas.pydata.org", "docs", "site-specific", True, 1.2, "Pandas docs"),
            ("docs.docker.com", "docs", "site-specific", True, 1.2, "Docker docs"),
            ("fastapi.tiangolo.com", "docs", "site-specific", True, 1.25, "FastAPI docs"),
            ("docs.ray.io", "docs", "site-specific", True, 1.2, "Ray docs"),
            ("llamaindex.ai", "docs", "site-specific", True, 1.2, "LlamaIndex docs"),
            ("facebook.com", "web", "restricted", False, 0.0, "Blocked"),
            ("instagram.com", "web", "restricted", False, 0.0, "Blocked"),
            ("x.com", "web", "restricted", False, 0.0, "Blocked"),
            ("twitter.com", "web", "restricted", False, 0.0, "Blocked"),
        ]
        query = """
        INSERT INTO source_policies (domain_pattern, source_type, license_tag, allow_training, base_weight, notes)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (domain_pattern) DO NOTHING
        """
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    for row in defaults:
                        cur.execute(query, row)
                conn.commit()
        except Exception as e:
            print(f"[DB Error] Failed to bootstrap source policies: {e}")

    def _resolve_source_policy(self, domain: str, source_type: str) -> dict:
        domain = (domain or "").lower().strip()
        if not domain:
            return {"license_tag": "unknown", "allow_training": True, "base_weight": 1.0}
        try:
            with self._connect() as conn:
                with conn.cursor(cursor_factory=DictCursor) as cur:
                    cur.execute(
                        """
                        SELECT domain_pattern, source_type, license_tag, allow_training, base_weight
                        FROM source_policies
                        WHERE %s LIKE ('%%' || domain_pattern)
                          AND (source_type = %s OR source_type = 'web')
                        ORDER BY LENGTH(domain_pattern) DESC
                        LIMIT 1
                        """,
                        (domain, source_type),
                    )
                    row = cur.fetchone()
                    if not row:
                        return {"license_tag": "unknown", "allow_training": True, "base_weight": 1.0}
                    return {
                        "license_tag": row["license_tag"],
                        "allow_training": bool(row["allow_training"]),
                        "base_weight": float(row["base_weight"]),
                    }
        except Exception as e:
            print(f"[DB Error] Failed to resolve source policy for domain={domain}: {e}")
            return {"license_tag": "unknown", "allow_training": True, "base_weight": 1.0}

    def is_url_crawled(self, url: str) -> bool:
        """指定されたURLが既にDBに存在するかを確認する"""
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1 FROM crawled_data WHERE url = %s LIMIT 1", (url,))
                    return cur.fetchone() is not None
        except Exception as e:
            print(f"[DB Error] Failed to check URL existence '{url}': {e}")
            return True

    def insert_crawled_data(self, url: str, domain: str, title: str, content: str, source_type: str, language: str = "ja") -> bool:
        """クロールしたデータをPostgreSQLに保存する"""
        try:
            policy = self._resolve_source_policy(domain=domain, source_type=source_type)
            q_score = quality_score(content)
            query = """
            INSERT INTO crawled_data (url, domain, title, content, char_count, source_type, language, quality_score, license_tag, allowed_for_training)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (url) DO NOTHING
            """
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        query,
                        (
                            url,
                            domain,
                            (title or "")[:255],
                            content,
                            len(content),
                            source_type,
                            language,
                            q_score,
                            policy["license_tag"],
                            policy["allow_training"],
                        ),
                    )
                conn.commit()
            with self._insert_metrics_lock:
                metric = self._insert_metrics.setdefault(source_type or "unknown", {"ok": 0, "ng": 0})
                metric["ok"] += 1
            return True
        except Exception as e:
            print(f"[DB Error] Failed to insert data for '{url}': {e}")
            with self._insert_metrics_lock:
                metric = self._insert_metrics.setdefault(source_type or "unknown", {"ok": 0, "ng": 0})
                metric["ng"] += 1
            return False

    def get_and_reset_insert_metrics(self) -> dict:
        with self._insert_metrics_lock:
            snapshot = {k: dict(v) for k, v in self._insert_metrics.items()}
            self._insert_metrics.clear()
            return snapshot

    def get_stats(self) -> dict:
        """現在のデータ収集の統計情報を取得する（ダッシュボード表示用）"""
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT COUNT(*) FROM crawled_data WHERE source_type = 'wikipedia'")
                    wiki_count = cur.fetchone()[0] or 0
                    cur.execute("SELECT COUNT(*) FROM crawled_data WHERE source_type = 'web'")
                    web_count = cur.fetchone()[0] or 0
                    cur.execute("SELECT COUNT(*) FROM crawled_data WHERE source_type = 'rss'")
                    rss_count = cur.fetchone()[0] or 0
                    cur.execute("SELECT COUNT(*) FROM crawled_data WHERE source_type = 'news'")
                    news_count = cur.fetchone()[0] or 0
                    cur.execute("SELECT COUNT(*) FROM crawled_data WHERE source_type = 'arxiv'")
                    arxiv_count = cur.fetchone()[0] or 0
                    cur.execute("SELECT COUNT(*) FROM crawled_data WHERE source_type = 'docs'")
                    docs_count = cur.fetchone()[0] or 0
                    cur.execute("SELECT COUNT(*) FROM crawled_data WHERE allowed_for_training = FALSE")
                    blocked_count = cur.fetchone()[0] or 0
                    cur.execute("SELECT COUNT(*) FROM crawled_data")
                    total_docs = cur.fetchone()[0] or 0

            fs_stats = self._get_local_corpus_fs_stats()
            wiki_local_count = int(fs_stats.get("wiki_local_count", 0))
            wiki_local_mb = float(fs_stats.get("wiki_local_size_mb", 0.0))
            train_bin_mb = float(fs_stats.get("train_bin_mb", 0.0))
            val_bin_mb = float(fs_stats.get("val_bin_mb", 0.0))
            wiki_total = max(int(wiki_count), wiki_local_count)
            db_estimate_mb = float(total_docs * 1000 * 3 / (1024 * 1024))
            total_volume_mb = db_estimate_mb + wiki_local_mb + train_bin_mb + val_bin_mb

            return {
                "collected_docs_wiki": wiki_total,
                "collected_docs_web": web_count,
                "collected_docs_rss": rss_count,
                "collected_docs_news": news_count,
                "collected_docs_arxiv": arxiv_count,
                "collected_docs_docs": docs_count,
                "blocked_docs": blocked_count,
                "total_docs": total_docs,
                "dataset_size_mb": round(total_volume_mb, 2),
                "wiki_local_docs": wiki_local_count,
            }
        except Exception as e:
            print(f"[DB Error] Failed to get stats: {e}")
            return {
                "collected_docs_wiki": 0,
                "collected_docs_web": 0,
                "collected_docs_rss": 0,
                "collected_docs_news": 0,
                "collected_docs_arxiv": 0,
                "collected_docs_docs": 0,
                "blocked_docs": 0,
                "total_docs": 0,
                "dataset_size_mb": 0.0,
                "wiki_local_docs": 0,
            }

    def _get_local_corpus_fs_stats(self) -> dict:
        """
        ローカルコーパス容量の簡易集計（重いためキャッシュ）。
        """
        cache_sec = max(30, int(os.environ.get("CORPUS_FS_CACHE_SEC", "300")))
        now = time.time()
        with self._fs_stats_lock:
            if self._fs_stats_cache is not None and (now - self._fs_stats_cached_at) < cache_sec:
                return dict(self._fs_stats_cache)

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        wiki_dir = os.path.join(base_dir, "dataset", "wikipedia")
        train_bin = os.path.join(base_dir, "dataset", "train.bin")
        val_bin = os.path.join(base_dir, "dataset", "val.bin")

        wiki_count = 0
        wiki_bytes = 0
        try:
            if os.path.isdir(wiki_dir):
                for root, _, files in os.walk(wiki_dir):
                    for name in files:
                        if name.startswith("wiki_") and name.endswith(".txt"):
                            wiki_count += 1
                            try:
                                wiki_bytes += os.path.getsize(os.path.join(root, name))
                            except Exception:
                                pass
        except Exception:
            pass

        train_bytes = os.path.getsize(train_bin) if os.path.exists(train_bin) else 0
        val_bytes = os.path.getsize(val_bin) if os.path.exists(val_bin) else 0
        stats = {
            "wiki_local_count": wiki_count,
            "wiki_local_size_mb": wiki_bytes / (1024 * 1024),
            "train_bin_mb": train_bytes / (1024 * 1024),
            "val_bin_mb": val_bytes / (1024 * 1024),
        }
        with self._fs_stats_lock:
            self._fs_stats_cache = dict(stats)
            self._fs_stats_cached_at = now
        return stats

    def stream_crawled_contents(self, batch_size: int = 1000) -> Generator[str, None, None]:
        with self._connect() as conn:
            with conn.cursor(name="crawled_data_stream", cursor_factory=DictCursor) as cur:
                cur.itersize = batch_size
                cur.execute("SELECT content FROM crawled_data ORDER BY id ASC")
                while True:
                    rows = cur.fetchmany(batch_size)
                    if not rows:
                        break
                    for row in rows:
                        text = (row["content"] or "").strip()
                        if text:
                            yield text

    def stream_crawled_documents(self, batch_size: int = 1000) -> Generator[dict, None, None]:
        with self._connect() as conn:
            with conn.cursor(name="crawled_data_docs_stream", cursor_factory=DictCursor) as cur:
                cur.itersize = batch_size
                cur.execute(
                    """
                    SELECT content, source_type, language, quality_score, license_tag, allowed_for_training
                    FROM crawled_data
                    ORDER BY id ASC
                    """
                )
                while True:
                    rows = cur.fetchmany(batch_size)
                    if not rows:
                        break
                    for row in rows:
                        text = (row["content"] or "").strip()
                        if not text:
                            continue
                        yield {
                            "content": text,
                            "source_type": (row["source_type"] or "web"),
                            "language": (row["language"] or "ja"),
                            "quality_score": float(row["quality_score"] or 0.0),
                            "license_tag": (row["license_tag"] or "unknown"),
                            "allowed_for_training": bool(row["allowed_for_training"]),
                        }

    def upsert_source_policy(
        self,
        domain_pattern: str,
        source_type: str,
        license_tag: str,
        allow_training: bool,
        base_weight: float,
        notes: str = "",
    ):
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO source_policies
                        (domain_pattern, source_type, license_tag, allow_training, base_weight, notes, updated_at)
                        VALUES (%s, %s, %s, %s, %s, %s, NOW())
                        ON CONFLICT (domain_pattern) DO UPDATE
                        SET source_type = EXCLUDED.source_type,
                            license_tag = EXCLUDED.license_tag,
                            allow_training = EXCLUDED.allow_training,
                            base_weight = EXCLUDED.base_weight,
                            notes = EXCLUDED.notes,
                            updated_at = NOW()
                        """,
                        (
                            (domain_pattern or "").lower().strip(),
                            (source_type or "web").lower().strip(),
                            (license_tag or "unknown").strip(),
                            bool(allow_training),
                            float(base_weight),
                            notes or "",
                        ),
                    )
                conn.commit()
        except Exception as e:
            print(f"[DB Error] Failed to upsert source policy: {e}")

    def list_source_policies(self, limit: int = 200) -> list[dict]:
        try:
            with self._connect() as conn:
                with conn.cursor(cursor_factory=DictCursor) as cur:
                    cur.execute(
                        """
                        SELECT id, domain_pattern, source_type, license_tag, allow_training, base_weight, notes, created_at, updated_at
                        FROM source_policies
                        ORDER BY updated_at DESC
                        LIMIT %s
                        """,
                        (max(1, min(limit, 1000)),),
                    )
                    return [dict(row) for row in cur.fetchall()]
        except Exception as e:
            print(f"[DB Error] Failed to list source policies: {e}")
            return []

    def upsert_node_heartbeat(
        self,
        node_id: str,
        role: str,
        status: str,
        cpu_usage: float,
        ram_usage: float,
        active_workers: int = 0,
    ):
        """自身の稼働状態（Heartbeat）とリソース状況を書き込む・更新する"""
        try:
            query = """
            INSERT INTO system_nodes (node_id, role, status, cpu_usage, ram_usage, active_workers, last_heartbeat)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (node_id) DO UPDATE
            SET role = EXCLUDED.role,
                status = EXCLUDED.status,
                cpu_usage = EXCLUDED.cpu_usage,
                ram_usage = EXCLUDED.ram_usage,
                active_workers = EXCLUDED.active_workers,
                last_heartbeat = EXCLUDED.last_heartbeat
            """
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        query,
                        (
                            node_id,
                            role,
                            status,
                            cpu_usage,
                            ram_usage,
                            int(max(0, active_workers)),
                            datetime.datetime.now(datetime.timezone.utc),
                        ),
                    )
                conn.commit()
        except Exception as e:
            print(f"[DB Error] Failed to update heartbeat for node {node_id}: {e}")

    def get_my_target_status(self, node_id: str) -> str:
        """自身に対するダッシュボードからの命令(start/stop)を取得する"""
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT target_status FROM system_nodes WHERE node_id = %s", (node_id,))
                    row = cur.fetchone()
                    return row[0] if row and row[0] else "unspecified"
        except Exception as e:
            print(f"[DB Error] Failed to get target status for node {node_id}: {e}")
            return "unspecified"

    def get_all_nodes(self) -> list:
        """ダッシュボード用: すべてのノードの情報を取得する"""
        try:
            # Node table is meant for "currently connected" visibility.
            # Keep stale/offline historical rows out by default.
            window_hours = int(os.environ.get("NODE_LIST_WINDOW_HOURS", "2"))
            with self._connect() as conn:
                with conn.cursor(cursor_factory=DictCursor) as cur:
                    cur.execute(
                        """
                        SELECT node_id, role, status, cpu_usage, ram_usage, active_workers, target_status, last_heartbeat
                        FROM system_nodes
                        WHERE last_heartbeat >= (NOW() - (%s || ' hours')::interval)
                        ORDER BY last_heartbeat DESC
                        """,
                        (max(1, window_hours),),
                    )
                    return [dict(row) for row in cur.fetchall()]
        except Exception as e:
            print(f"[DB Error] Failed to get all nodes: {e}")
            return []

    def set_node_target_status(self, node_id: str, target_status: str):
        """ダッシュボード用: 特定ノードへの命令をDBに書き込む"""
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute("UPDATE system_nodes SET target_status = %s WHERE node_id = %s", (target_status, node_id))
                conn.commit()
        except Exception as e:
            print(f"[DB Error] Failed to set target status for node {node_id}: {e}")

    def set_all_nodes_target_status(self, target_status: str, role: str = "worker"):
        """ダッシュボード用: 指定ロールの全ノードに命令を送る"""
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    if role == "all":
                        cur.execute("UPDATE system_nodes SET target_status = %s", (target_status,))
                    else:
                        cur.execute(
                            "UPDATE system_nodes SET target_status = %s WHERE role = %s",
                            (target_status, role),
                        )
                conn.commit()
        except Exception as e:
            print(f"[DB Error] Failed to set target status for all nodes (role={role}): {e}")

    def get_active_collector_nodes(self, online_window_sec: int = 60, include_master: bool = True) -> list[dict]:
        """
        直近 heartbeat がある収集ノードを返す。
        include_master=False の場合は worker のみ返す。
        """
        try:
            roles = ("worker", "master") if include_master else ("worker",)
            with self._connect() as conn:
                with conn.cursor(cursor_factory=DictCursor) as cur:
                    cur.execute(
                        """
                        SELECT node_id, role, status, last_heartbeat
                        FROM system_nodes
                        WHERE role = ANY(%s)
                          AND status = 'running'
                          AND last_heartbeat >= (NOW() - (%s || ' seconds')::interval)
                        ORDER BY node_id ASC
                        """,
                        (list(roles), int(max(10, online_window_sec))),
                    )
                    return [dict(row) for row in cur.fetchall()]
        except Exception as e:
            print(f"[DB Error] Failed to fetch active collector nodes: {e}")
            return []

    def search_relevant_contents(self, query: str, limit: int = 3) -> list[str]:
        """
        RAG向け: 簡易全文検索で関連コンテンツを返す。
        PostgreSQLの標準機能のみで動く軽量実装。
        """
        query = (query or "").strip()
        if not query:
            return []
        terms = [t for t in query.split() if len(t) >= 2][:5]
        if not terms:
            terms = [query[:32]]
        like_patterns = [f"%{term}%" for term in terms]

        sql_fts = """
        SELECT content
        FROM crawled_data
        WHERE to_tsvector('simple', coalesce(title, '') || ' ' || coalesce(content, ''))
              @@ websearch_to_tsquery('simple', %s)
        ORDER BY ts_rank_cd(
            to_tsvector('simple', coalesce(title, '') || ' ' || coalesce(content, '')),
            websearch_to_tsquery('simple', %s)
        ) DESC,
        CASE source_type
            WHEN 'rss' THEN 3
            WHEN 'wikipedia' THEN 2
            ELSE 1
        END DESC,
        CASE language
            WHEN 'ja' THEN 2
            WHEN 'en' THEN 1
            ELSE 0
        END DESC,
        created_at DESC
        LIMIT %s
        """
        sql_like = """
        SELECT content
        FROM crawled_data
        WHERE content ILIKE ANY(%s)
        ORDER BY char_count DESC, created_at DESC
        LIMIT %s
        """
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    capped = max(1, min(limit, 8))
                    cur.execute(sql_fts, (query, query, capped))
                    rows = cur.fetchall()
                    if not rows:
                        cur.execute(sql_like, (like_patterns, capped))
                        rows = cur.fetchall()
                    return [row[0] for row in rows if row and row[0]]
        except Exception as e:
            print(f"[DB Error] Failed RAG search: {e}")
            return []

    def insert_evaluation_run(self, model_tag: str, avg_score: float, result_json: dict):
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO evaluation_runs (model_tag, avg_score, result_json)
                        VALUES (%s, %s, %s::jsonb)
                        """,
                        (model_tag, avg_score, json_dumps(result_json)),
                    )
                conn.commit()
        except Exception as e:
            print(f"[DB Error] Failed to insert evaluation run: {e}")

    def get_latest_evaluation_runs(self, limit: int = 20) -> list[dict]:
        try:
            with self._connect() as conn:
                with conn.cursor(cursor_factory=DictCursor) as cur:
                    cur.execute(
                        """
                        SELECT id, model_tag, avg_score, result_json, created_at
                        FROM evaluation_runs
                        ORDER BY created_at DESC
                        LIMIT %s
                        """,
                        (max(1, min(limit, 100)),),
                    )
                    return [dict(row) for row in cur.fetchall()]
        except Exception as e:
            print(f"[DB Error] Failed to fetch evaluation runs: {e}")
            return []

    def get_latest_evaluation_run(self):
        runs = self.get_latest_evaluation_runs(limit=1)
        return runs[0] if runs else None

    def insert_model_version(
        self,
        model_tag: str,
        checkpoint_path: str,
        source_checkpoint: str,
        avg_score: float,
        promoted: bool = False,
        notes: str = "",
    ):
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO model_versions
                        (model_tag, checkpoint_path, source_checkpoint, avg_score, promoted, notes)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        """,
                        (model_tag, checkpoint_path, source_checkpoint, avg_score, promoted, notes),
                    )
                conn.commit()
        except Exception as e:
            print(f"[DB Error] Failed to insert model version: {e}")

    def list_model_versions(self, limit: int = 20) -> list[dict]:
        try:
            with self._connect() as conn:
                with conn.cursor(cursor_factory=DictCursor) as cur:
                    cur.execute(
                        """
                        SELECT id, model_tag, checkpoint_path, source_checkpoint, avg_score, promoted, notes, created_at
                        FROM model_versions
                        ORDER BY created_at DESC
                        LIMIT %s
                        """,
                        (max(1, min(limit, 100)),),
                    )
                    return [dict(row) for row in cur.fetchall()]
        except Exception as e:
            print(f"[DB Error] Failed to list model versions: {e}")
            return []

    def insert_dataset_version(
        self,
        dataset_tag: str,
        train_tokens: int,
        val_tokens: int,
        total_docs: int,
        blocked_docs: int,
        filtered_docs: int,
        duplicate_docs: int,
        source_breakdown: dict,
        metadata_json: dict,
    ):
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO dataset_versions
                        (dataset_tag, train_tokens, val_tokens, total_docs, blocked_docs, filtered_docs, duplicate_docs, source_breakdown, metadata_json)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb)
                        """,
                        (
                            dataset_tag,
                            int(train_tokens),
                            int(val_tokens),
                            int(total_docs),
                            int(blocked_docs),
                            int(filtered_docs),
                            int(duplicate_docs),
                            json_dumps(source_breakdown or {}),
                            json_dumps(metadata_json or {}),
                        ),
                    )
                conn.commit()
        except Exception as e:
            print(f"[DB Error] Failed to insert dataset version: {e}")

    def get_latest_dataset_versions(self, limit: int = 20) -> list[dict]:
        try:
            with self._connect() as conn:
                with conn.cursor(cursor_factory=DictCursor) as cur:
                    cur.execute(
                        """
                        SELECT id, dataset_tag, train_tokens, val_tokens, total_docs, blocked_docs, filtered_docs, duplicate_docs, source_breakdown, metadata_json, created_at
                        FROM dataset_versions
                        ORDER BY created_at DESC
                        LIMIT %s
                        """,
                        (max(1, min(limit, 200)),),
                    )
                    return [dict(row) for row in cur.fetchall()]
        except Exception as e:
            print(f"[DB Error] Failed to list dataset versions: {e}")
            return []


def json_dumps(value) -> str:
    import json
    return json.dumps(value, ensure_ascii=False)


if __name__ == "__main__":
    print("Testing DB Connection...")
    try:
        db = DBManager()
        stats = db.get_stats()
        print(f"Connection Successful! Current DB Stats: {stats}")
    except Exception as e:
        print(f"Connection Failed: {e}")
