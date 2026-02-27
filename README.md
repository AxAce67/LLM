# AI Factory Engine (LLM Builder)

ゼロから構築する分散型の大規模言語モデル（LLM）データ収集・学習・推論プラットフォームです。
Webクローラーが自動でデータを収集し、親機がトークナイズとトランスフォーマーモデルの継続学習を行います。すべてはモダンなWebダッシュボードから監視・制御可能です。

## 1. これからの使い方（運用フロー）

本システムは、**「起動して放置する」** タイプのバックグラウンドアプリケーション（サーバーアプリ）です。

1. **システムの起動**: クローラーと学習エンジンを「Ignite(起動)」します。
2. **データの蓄積**: 時間が経つにつれて「Corpus Storage」の数値が増え、独自の知識データがDBに蓄積されていきます。
3. **継続的な学習**: 親機（Master）が定期的に集めたデータを学習し、「Neural Net Training」のEpoch（世代）が進み、Loss（誤差）が下がっていきます。数日〜数週間稼働させ続けることで、AIが賢くなっていきます。
4. **テスト推論**: ダッシュボードの「AI Chat Inference」タブで、プロンプトを入力してAIの成長度合い（知能）をテストします。

## 2. 起動方法とアプリ化について

システムを簡単に起動・デプロイできるように、すでに**Bash化（シェルスクリプト化）およびDocker化**が完了しています。
デスクトップアプリ（.exeや.app）というよりも、サーバーやPCのバックグラウンドで24時間動き続ける「Webサービス・デーモン」として設計されています。

**◆ ローカル開発・確認用（現在手元で行っている方法）**
```bash
source venv/bin/activate
python3 app.py
```
-> `http://localhost:8000` にアクセス

**◆ 完全自動デプロイ・永続化用（インストーラー兼アプリ化）**
付属の `setup.sh` を実行するだけで、必要な環境の構築からDockerによるバックグラウンド起動（PCを閉じても裏で動き続ける状態）まで全自動で行われます。インターネット経由でのワンライナー起動もサポートしています。

```bash
# 【推奨】GitHub経由での一発起動コマンド（他のPCや友人に教える用）
curl -sL https://raw.githubusercontent.com/AxAce67/LLM/main/setup.sh | bash

# すでにファイル群をダウンロード済みのディレクトリで起動する場合
chmod +x setup.sh
./setup.sh
```

※ インストール実行時、ターミナル上で対話的に以下の入力が求められます。
1. システムの役割選択（親機／子機）
2. PostgreSQL接続情報

## 2.1 ローカルPostgreSQL運用（推奨）

親機と子機で役割を分けます。

- 親機: `postgres` コンテナを起動し、学習/ダッシュボードを担当
- 子機: クローラー専用で動き、親機のPostgreSQLへ直接書き込み

- 親機 `DATABASE_URL=postgresql://<user>:<pass>@postgres:5432/<db>`
- 子機 `DATABASE_URL=postgresql://<user>:<pass>@<master_ip>:5432/<db>`

この構成では、クロールデータ・ノード状態はすべてローカルに保存されます（150GBクラスのローカル容量を活用可能）。

親機ではDB Web UI (Adminer) も同時起動します。

- `http://localhost:8080`（既定）
- Server: `postgres`
- Username / Password / Database: `setup.sh` で入力した値
- ダッシュボード右上にも `DB Console` リンクが表示されます

## 3. 今後のアップデート・開発ロードマップ

ここから先、さらにシステムを強化するための「次なる開発ステップ」のアイディアです。

### 段階 1: モデルの知能と実用性の向上
- **モデルの大型化**: `transformer.py` のレイヤー数やEmbeddingの次元数を増やし、より賢いモデル構成に変更する。
- **RAG（検索拡張生成）の導入**: 実際のAIチャット画面で、AIが回答を作る前にPostgreSQLのDBから関連するクロール済み文章を検索してきて、それをカンペとして回答させる仕組み（より正確な知識を答えられるようになります）。

### 段階 2: クラウドでの本格運用と分散処理
- **クラウドサーバーへの親機設置**: AWSやGoogle Cloudなどに親機（Master）をデプロイし、24時間365日高速に学習し続ける本格的なサーバーを構築。
- **複数マシンの接続**: 余っている古いPCやノートパソコンで `./setup.sh` を実行し「Worker（子機）」としてセットアップ。ネットワーク越しに親機へデータを送り続ける強力な「Swarm（群れ）」を作ります。

### 段階 3: アプリケーションへの組み込み
- 今アクセスしているAPI（`/api/generate` など）を利用して、独自のiOS/Androidアプリや、LINEボットなどを作成し、裏側の脳みそとしてこの「AI Factory Engine」を連携させる。

---
*Created by the Advanced Agentic Coding Assistant.*

## 安定運用のための現在の既定値

- 推論APIの `max_tokens` は `1..256` に制限
- ダッシュボードの Chat で `Max Tokens` を変更可能（上限は `MAX_GENERATE_TOKENS`）
- クローラ並列数は `MAX_CRAWLER_WORKERS`（既定: `8`）で上限化
- 学習前処理はストリーミング書き込み（全トークンをメモリ保持しない）
- `ENABLE_TORCH_COMPILE=0` が既定（メモリ急増を避ける）

## LLM品質を上げるための運用手順

1. データ源を増やす:
- Wikipedia + Web Crawl に加えて RSS収集を有効化（`ENABLE_RSS_COLLECTOR=1`）
- News収集を有効化（`ENABLE_NEWS_COLLECTOR=1`）
- arXiv収集を有効化（`ENABLE_ARXIV_COLLECTOR=1`）
- 技術ドキュメント収集を有効化（`ENABLE_DOCS_COLLECTOR=1`）
- News RSSは `NEWS_FEEDS`（カンマ区切り）で追加可能
- Hacker News収集量は `HN_ITEMS` で調整
- Web上の新規URLを自動発見（手動登録なし）:
- `ENABLE_AUTO_DISCOVERY=1`
- `AUTO_DISCOVERY_QUERIES`（カンマ区切り）で探索テーマを指定
- `AUTO_DISCOVERY_SEEDS_PER_CYCLE` / `AUTO_DISCOVERY_RESULTS_PER_QUERY` で収集量を調整
- `ARXIV_QUERY` / `ARXIV_MAX_RESULTS` で論文収集条件を調整
- `DOC_SEED_URLS`（カンマ区切り）でドキュメント収集先を追加
2. 学習設定を段階的に拡張:
- `MODEL_SIZE=tiny/small/base/medium`
- `TRAIN_STEPS_PER_CYCLE`, `TRAIN_GRAD_ACCUM_STEPS`, `TRAIN_WARMUP_STEPS` を調整
- `VAL_EVAL_EVERY`, `VAL_EVAL_BATCHES`, `EARLY_STOPPING_PATIENCE` で検証と早期停止を調整
3. 推論時チューニング:
- ダッシュボードの Chat で `temperature/top_p/repetition_penalty` を調整
4. RAGを使う:
- `ENABLE_RAG=1` でDB検索コンテキストを回答に注入
5. 定期評価:
- `python3 eval/evaluate_model.py` でベンチマークスコアを比較
- `EVAL_SAVE_TO_DB=1` で評価結果を `evaluation_runs` テーブルに保存
- API: `GET /api/evals` で最新評価履歴を取得
- API: `POST /api/evals/run` で評価実行、`GET /api/evals/status` で進捗確認
- ダッシュボードの `Evaluation History` から `Run Eval` 実行可能

6. 分散収集の重複削減:
- 親機/子機で同じシードを重複クロールしないよう、URLをノードごとに自動シャーディング
- `COLLECTION_INCLUDE_MASTER=1` で親機も収集参加
- `MAX_CRAWL_PAGES_PER_CYCLE` を稼働ノード数で自動分割

## モデル昇格（Production運用）

評価済みの最良チェックポイントを本番推論用に固定できます。

- `ckpt_best.pt` -> `ckpt_production.pt` へ昇格
- 昇格履歴は `model_versions` テーブルに保存
- API:
- `GET /api/models` でモデル履歴取得
- `POST /api/models/promote` で昇格実行
- ダッシュボードの `Model Registry` から `Promote Best -> Production` 実行可能
- 昇格は `PROMOTION_MIN_SCORE`（既定 `0.72`）以上の評価スコア時のみ許可
- 緊急時のみ `FORCE_PROMOTE=1` で閾値を無視可能

推論時の読み込み優先順位:

- `USE_PRODUCTION_MODEL=1`（既定）:
- `checkpoints/ckpt_production.pt` があれば優先して使用、なければ `ckpt_latest.pt`
- `USE_PRODUCTION_MODEL=0`:
- 常に `checkpoints/ckpt_latest.pt` を使用

## CPU 1コア100%になる場合

小規模バッチやデフォルトスレッド設定だと、CPU環境では1コアに偏ることがあります。異常ではありませんが改善可能です。

- `PYTORCH_CPU_THREADS` をコア数に合わせて設定（例: 4coreなら `3` or `4`）
- `TRAIN_BATCH_SIZE` / `TRAIN_GRAD_ACCUM_STEPS` を調整
- `SPM_NUM_THREADS` でTokenizer学習の並列度を上げる
- `AUTO_TUNE=1` + `AUTO_TUNE_REALTIME=1` で環境に合わせて自動調整

## 自動判定（どんな環境でも動かしやすくする）

以下を有効化すると、CPU/RAM/GPUを見て推奨値を自動反映します。

- `AUTO_TUNE=1`:
  - 起動時に `MODEL_SIZE`, `TRAIN_BATCH_SIZE`, `TRAIN_SEQ_LEN`, `MAX_GENERATE_TOKENS` の既定値を環境に合わせる
- `AUTO_TUNE_REALTIME=1`:
  - 稼働中も空きメモリを見てクローラ並列上限などを保守的に調整

ダッシュボードの `Auto Profile` に現在の判定結果が表示されます。

## 収集ガード（安全運用）

- `CRAWLER_RESPECT_ROBOTS=1`: `robots.txt` を尊重
- `CRAWLER_DOMAIN_MIN_INTERVAL_SEC=0.5`: 同一ドメインへの最小アクセス間隔
- `CRAWLER_DOMAIN_CONCURRENCY=2`: 同一ドメイン同時接続上限
- `CRAWLER_BATCH_SLEEP_SEC=0.6`: バッチ間スリープ

## 管理API認証

- `ADMIN_API_TOKEN` を設定すると管理系APIでトークン必須
- ヘッダー: `x-admin-token: <token>`
- 対象:
- `POST /api/control`
- `POST /api/nodes/control-all`
- `POST /api/nodes/{node_id}/control`
- `POST /api/models/promote`
- `POST /api/policies`
- `POST /api/evals/run`

## データ品質フィルタ

前処理時に以下を実施します。

- 正規化（空白・改行整理）
- 低品質文の除外（短すぎる/文字多様性が低すぎる）
- ハッシュベース重複除外
- ソース品質重み付け（Wikipedia/News/RSS/Web）
- DB保存時に `quality_score` を自動算出
- `allowed_for_training=false` のデータは前処理段階で自動除外

調整パラメータ:

- `ENABLE_TEXT_DEDUP=1`
- `MIN_TRAIN_CHARS=120`
- `SOURCE_WEIGHT_WIKIPEDIA=1.4`
- `SOURCE_WEIGHT_ARXIV=1.35`
- `SOURCE_WEIGHT_DOCS=1.3`
- `SOURCE_WEIGHT_NEWS=1.25`
- `SOURCE_WEIGHT_RSS=1.1`
- `SOURCE_WEIGHT_WEB=1.0`
- `QUALITY_WEIGHT_FLOOR=0.35`
- `QUALITY_WEIGHT_BOOST=0.9`

## ライセンス/利用可否ポリシー

- `source_policies` テーブルで、ドメイン単位に学習可否を管理
- `allow_training=false` を設定したドメインは収集しても学習対象から除外
- API:
- `GET /api/policies` でポリシー一覧
- `POST /api/policies` でポリシー追加・更新
- ダッシュボードの `Source Policies` から編集可能

## データセット版管理

- 前処理ごとに `dataset_versions` テーブルへ記録
- API: `GET /api/datasets`
- ダッシュボードの `Dataset Versions` で履歴確認可能

## 学習ジョブの再試行/ロールバック

- 学習前に `ckpt_latest.pt` を `ckpt_latest.pretrain.bak.pt` へバックアップ
- 失敗時は自動でロールバックして再試行
- `TRAIN_RETRY_MAX=2`
- `TRAIN_RETRY_BACKOFF_SEC=2.0`

## 推論軽量化（CPU配布向け）

- `ENABLE_CPU_QUANTIZATION=1` で int8 動的量子化を有効化

## バックアップ/復元

```bash
# バックアップ作成
./ops/backup_local.sh

# 復元
./ops/restore_local.sh /path/to/backups/backup_YYYYMMDD_HHMMSS
```

## アップデート運用

`version.json` を配布サーバー側で更新し、クライアントは以下を実行:

```bash
# モデル更新
VERSION_URL=https://example.com/version.json ./ops/update-model.sh

# アプリ更新
VERSION_URL=https://example.com/version.json ./ops/update-app.sh
```

## E2E確認

```bash
# サービス起動後の疎通確認
BASE_URL=http://localhost:8000 ./ops/e2e_pipeline.sh

# バックアップ/復元テスト
./ops/test_backup_restore.sh
```

## 移行ブランチ運用

- `main`: 安定運用ブランチ
- `migration-ollama`: Ollama/LM Studio互換化の移行ブランチ
- 詳細: `docs/migration_branch_strategy.md`

## Ollama / LM Studio 移行（第1段）

最小フロー:

```bash
# 1) 依存導入
pip install -r requirements-migration.txt

# 2) 既存DBからHF学習テキスト生成
python3 migration_hf/prepare_hf_text.py

# 3) LoRA学習（例: Qwen2.5 1.5B）
python3 migration_hf/train_lora.py \
  --base_model Qwen/Qwen2.5-1.5B-Instruct \
  --train_text dataset/hf/train.txt \
  --output_dir models/hf_lora

# 3-b) 実行場所は自分で選び、自動判定で学習設定を調整（推奨）
BASE_MODEL=Qwen/Qwen2.5-1.5B-Instruct \
TRAIN_TEXT=dataset/hf/train.txt \
OUTPUT_DIR=models/hf_lora \
./migration_hf/run_train_auto.sh

# 4) GGUF変換（llama.cppが必要）
./migration_hf/export_gguf.sh ~/llama.cpp Qwen/Qwen2.5-1.5B-Instruct ./models/qwen2.5-1.5b.gguf
```

注記:
- これは移行の第一段（学習・変換の土台）です。
- 本番配布前に推論品質評価とモデルカード整備を行ってください。
- 「学習場所（ローカル/VPS）」は手動選択、`run_train_auto.sh` がそのマシン上で自動チューニングを適用します。

### 実行場所の選び方（ローカル / VPS）

1. ローカルで学習したい場合:
- そのPCで上記コマンドを実行

2. VPSで学習したい場合:
- VPSへこのリポジトリを配置して同じコマンドを実行
- `BASE_MODEL` などはVPS側で指定

共通:
- 場所は手動選択
- 設定は自動判定（CPU/GPU/RAM）

### ダッシュボード統合（Migration Ops）

- `Migration Ops (HF / GGUF)` パネルから以下を実行可能:
- HF LoRA学習開始
- GGUF変換開始
- 進捗ステータス確認

## 学習再現性と検証

- `TRAIN_SEED=42` で乱数を固定
- `ckpt_latest.pt` に加えて、検証ロス改善時に `ckpt_best.pt` を保存
- ダッシュボードで `Train Loss` と `Val / Best` を確認可能
