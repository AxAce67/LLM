# Architecture

## 役割分担

- 親機 (master):
  - ダッシュボード/API (`app.py`)
  - 学習/評価実行
  - ローカル PostgreSQL と DB Web UI (Adminer)
- 子機 (worker):
  - Web収集専用
  - 親機の PostgreSQL へ書き込み

## 構成図

```text
                  +----------------------+
                  |   Dashboard / API    |
                  |   (master: app.py)   |
                  +----------+-----------+
                             |
                             v
                  +----------------------+
                  | PostgreSQL (master)  |
                  | crawled_data, evals  |
                  +----------+-----------+
                             ^
                 write data  |
 +---------------------------+---------------------------+
 |                           |                           |
 v                           v                           v
+----------------+  +----------------+         +----------------+
| worker node #1 |  | worker node #2 |   ...   | worker node #N |
| web crawler    |  | web crawler    |         | web crawler    |
+----------------+  +----------------+         +----------------+
```

## 起動パターン

- 親機:
  - `docker compose --profile localdb --profile master up -d --build`
- 子機:
  - `SYSTEM_ROLE=worker` と `DATABASE_URL=...@<master_ip>:5432/...`
  - `docker compose up -d --build`

## 監視対象（最低限）

- ノード生存: `system_nodes.last_heartbeat`
- 収集量: `crawled_data` 件数推移
- 学習品質: `evaluation_runs.avg_score` と `ckpt_best.pt`
- API保護: `ADMIN_API_TOKEN_REQUIRED=1`
