#!/bin/bash
# ==============================================================================
# 自作LLM分散収集システム（Docker）完全自動インストーラー
# 使い方: curl -sL https://raw.githubusercontent.com/.../setup.sh | bash
#         または bash setup.sh [--master]
# ==============================================================================

set -e

DRY_RUN=0
for arg in "$@"; do
    case "$arg" in
        --dry-run)
            DRY_RUN=1
            ;;
    esac
done

# ------------------------------------------------------------------------------

# --- 1. 既存親機の自動検知 ---
echo -e "\n============================================="
echo -e " LLM Data Collector 自動セットアップウィザード"
echo -e "=============================================\n"

SYSTEM_ROLE=""
DOCKER_PROFILE=""
DETECTED_MASTER=0
POSTGRES_HOST="postgres"
POSTGRES_PORT="5432"

echo -e "既存の親機がある場合はホスト/IPを入力してください。"
read -p "既存親機ホスト/IP (未入力で新規クラスタ): " EXISTING_MASTER_HOST < /dev/tty
if [ -n "$EXISTING_MASTER_HOST" ]; then
    read -p "親機ダッシュボードポート [8000]: " EXISTING_MASTER_DASHBOARD_PORT < /dev/tty
    DASH_PORT="${EXISTING_MASTER_DASHBOARD_PORT:-8000}"
    if command -v curl >/dev/null 2>&1; then
        MASTER_ROLE=$(curl -fsS --max-time 3 "http://${EXISTING_MASTER_HOST}:${DASH_PORT}/api/runtime-config" | python3 -c "import json,sys; print(json.load(sys.stdin).get('system_role',''))" 2>/dev/null || true)
        if [ "$MASTER_ROLE" = "master" ]; then
            DETECTED_MASTER=1
            SYSTEM_ROLE="worker"
            DOCKER_PROFILE=""
            POSTGRES_HOST="$EXISTING_MASTER_HOST"
            echo -e "\n[Auto Detect] 既存の親機を検知しました。役割は Worker 固定になります。"
        fi
    fi
fi

# --- 2. システムの役割（Role）選択 ---
if [ "$DETECTED_MASTER" -eq 0 ]; then
    echo -e "このマシンの役割（Role）を選択してください。"
    echo -e "  1) 子機 / Worker (データ収集専用・推奨)"
    echo -e "  2) 親機 / Master (ダッシュボード稼働・データ収集兼任)"

    while true; do
        read -p "番号を入力 (1 または 2): " ROLE_CHOICE < /dev/tty
        case $ROLE_CHOICE in
            1)
                SYSTEM_ROLE="worker"
                DOCKER_PROFILE=""
                echo -e "\n[🤖 Worker Mode] 子機としてインストールを開始します。"
                break
                ;;
            2)
                SYSTEM_ROLE="master"
                DOCKER_PROFILE="--profile master"
                echo -e "\n[🌟 Master Mode] 親機としてインストールを開始します。"
                break
                ;;
            *)
                echo "[!] 無効な入力です。1 または 2 を入力してください。"
                ;;
        esac
    done
fi

# --- 3. PostgreSQL接続設定 ---
echo -e "\n============================================="
echo -e " LLM Data Collector 自動セットアップウィザード"
echo -e "=============================================\n"
echo -e "PostgreSQLの接続設定を行います。"
POSTGRES_DB="llm"
POSTGRES_USER="llm_user"
POSTGRES_PASSWORD="llm_pass"
ADMINER_PORT="8080"
BOOTSTRAP_TOKEN=""
ALLOW_BOOTSTRAP_PASSWORD="0"

read -p "Postgres DB名 [llm]: " INPUT_DB < /dev/tty
read -p "Postgres ユーザー [llm_user]: " INPUT_USER < /dev/tty
read -p "Postgres パスワード [llm_pass]: " INPUT_PASS < /dev/tty
POSTGRES_DB="${INPUT_DB:-llm}"
POSTGRES_USER="${INPUT_USER:-llm_user}"
POSTGRES_PASSWORD="${INPUT_PASS:-llm_pass}"

if [[ "$SYSTEM_ROLE" == "master" ]]; then
    POSTGRES_HOST="postgres"
    POSTGRES_PORT="5432"
    DOCKER_PROFILE="$DOCKER_PROFILE --profile localdb"
    read -p "DB Web UI(Adminer)ポート [8080]: " INPUT_ADMINER_PORT < /dev/tty
    ADMINER_PORT="${INPUT_ADMINER_PORT:-8080}"
    read -p "Worker自動設定用Bootstrap token (空欄で無効): " INPUT_BOOTSTRAP_TOKEN < /dev/tty
    BOOTSTRAP_TOKEN="${INPUT_BOOTSTRAP_TOKEN}"
    if [ -n "$BOOTSTRAP_TOKEN" ]; then
        read -p "BootstrapでDBパスワードも配布する? [y/N]: " INPUT_ALLOW_BOOTSTRAP_PASSWORD < /dev/tty
        if [[ "${INPUT_ALLOW_BOOTSTRAP_PASSWORD}" =~ ^[Yy]$ ]]; then
            ALLOW_BOOTSTRAP_PASSWORD="1"
        fi
    fi
    echo -e "親機はローカルPostgreSQLコンテナとAdminerを起動します。"
else
    echo -e "子機は親機のPostgreSQLへ接続します。"
    if [ "$DETECTED_MASTER" -eq 1 ]; then
        echo "親機Postgresホスト/IP: ${POSTGRES_HOST} (auto)"
        read -p "親機Postgresポート [5432]: " INPUT_PORT < /dev/tty
        POSTGRES_PORT="${INPUT_PORT:-5432}"
    else
        read -p "親機Postgresホスト/IP: " INPUT_HOST < /dev/tty
        read -p "親機Postgresポート [5432]: " INPUT_PORT < /dev/tty
        if [ -z "$INPUT_HOST" ]; then
            echo "[!] 親機Postgresホストは必須です。"
            exit 1
        fi
        POSTGRES_HOST="$INPUT_HOST"
        POSTGRES_PORT="${INPUT_PORT:-5432}"
    fi

    # 親機からDB設定を自動取得（任意）
    if command -v curl >/dev/null 2>&1; then
        read -p "親機からDB設定を自動取得しますか？ [Y/n]: " AUTO_BOOTSTRAP < /dev/tty
        AUTO_BOOTSTRAP="${AUTO_BOOTSTRAP:-Y}"
        if [[ "$AUTO_BOOTSTRAP" =~ ^[Yy]$ ]]; then
            read -p "Bootstrap token (未設定なら空欄): " BOOTSTRAP_TOKEN_INPUT < /dev/tty
            BOOTSTRAP_RESP=$(curl -fsS --max-time 5 "http://${POSTGRES_HOST}:8000/api/bootstrap?token=${BOOTSTRAP_TOKEN_INPUT}" 2>/dev/null || true)
            if [ -n "$BOOTSTRAP_RESP" ]; then
                BS_ROLE=$(printf "%s" "$BOOTSTRAP_RESP" | python3 -c "import json,sys; print(json.load(sys.stdin).get('system_role',''))" 2>/dev/null || true)
                BS_DB=$(printf "%s" "$BOOTSTRAP_RESP" | python3 -c "import json,sys; print(json.load(sys.stdin).get('postgres_db',''))" 2>/dev/null || true)
                BS_USER=$(printf "%s" "$BOOTSTRAP_RESP" | python3 -c "import json,sys; print(json.load(sys.stdin).get('postgres_user',''))" 2>/dev/null || true)
                BS_PORT=$(printf "%s" "$BOOTSTRAP_RESP" | python3 -c "import json,sys; print(json.load(sys.stdin).get('postgres_port',''))" 2>/dev/null || true)
                BS_PASS=$(printf "%s" "$BOOTSTRAP_RESP" | python3 -c "import json,sys; print(json.load(sys.stdin).get('postgres_password',''))" 2>/dev/null || true)
                if [ "$BS_ROLE" = "master" ]; then
                    if [ -n "$BS_DB" ]; then POSTGRES_DB="$BS_DB"; fi
                    if [ -n "$BS_USER" ]; then POSTGRES_USER="$BS_USER"; fi
                    if [ -n "$BS_PORT" ]; then POSTGRES_PORT="$BS_PORT"; fi
                    if [ -n "$BS_PASS" ]; then POSTGRES_PASSWORD="$BS_PASS"; fi
                    echo "[Auto Bootstrap] 親機からDB設定を取得しました。"
                else
                    echo "[Auto Bootstrap] 取得できませんでした（親機応答なし or token不一致）。"
                fi
            else
                echo "[Auto Bootstrap] 取得できませんでした。手動設定を継続します。"
            fi
        fi
    fi
fi

echo -e "\n[✓] PostgreSQL設定を一時記憶しました。"

if [ "$DRY_RUN" -eq 1 ]; then
    echo -e "\n[Dry Run] 実際のインストールは行いません。予定内容のみ表示します。"
    echo "  - role: ${SYSTEM_ROLE}"
    echo "  - postgres host: ${POSTGRES_HOST}:${POSTGRES_PORT}"
    echo "  - postgres db/user: ${POSTGRES_DB} / ${POSTGRES_USER}"
    if [[ "$SYSTEM_ROLE" == "master" ]]; then
        echo "  - profiles: ${DOCKER_PROFILE}"
        echo "  - adminer port: ${ADMINER_PORT}"
    fi
    echo "  - next actions:"
    echo "    1) Docker存在確認（不足時は自動インストール）"
    echo "    2) リポジトリ最新化（git fetch/pull）"
    echo "    3) .env生成"
    echo "    4) docker compose up -d --build"
    exit 0
fi

# --- 4. Docker のインストール確認と自動導入 (Ubuntu/Debian系想定) ---
SUDO=""
if [ "$(id -u)" -ne 0 ]; then
    SUDO="sudo "
fi

echo -e "\n[1/3] 🐳 Docker環境のチェックとインストール..."
if ! command -v docker &> /dev/null; then
    echo "Dockerがインストールされていません。自動インストールを開始します（数分かかります）..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    ${SUDO}sh get-docker.sh
    rm get-docker.sh
    echo "Dockerのインストールが完了しました。"
else
    echo "Dockerは既にインストールされています。スキップします。"
fi

# --- 4. ソースコードの同期と最新化 (Auto-Update) ---
echo -e "\n[2/4] 📥 ソースコードの取得・最新化..."
REPO_URL="https://github.com/AxAce67/LLM.git"
TARGET_DIR="llm-factory-engine"

# setup.sh が単独で（ディレクトリ外で）実行された場合や初回セットアップ時
if [ ! -d ".git" ]; then
    if [ ! -d "$TARGET_DIR" ]; then
        echo "リポジトリをクローンします..."
        git clone "$REPO_URL" "$TARGET_DIR"
    fi
    # クローン先のディレクトリに移動
    cd "$TARGET_DIR" || exit
fi

# 既にGit管理下（またはクローン直後）の場合は最新コードを引っ張ってくる
echo "最新のアップデートを確認しています..."
git fetch
LOCAL=$(git rev-parse HEAD)
REMOTE=$(git rev-parse @{u} 2>/dev/null || git rev-parse origin/main)

if [ "$LOCAL" != "$REMOTE" ]; then
    echo "🚀 更新が見つかりました！最新のコードを適応します。"
    git pull origin main
else
    echo "✓ コードはすでに最新です。"
fi

# 最新版の互換性のため docker compose V2 コマンドを確認
DOCKER_COMPOSE_CMD="${SUDO}docker compose"
if ! ${SUDO}docker compose version &> /dev/null; then
    if command -v docker-compose &> /dev/null; then
         DOCKER_COMPOSE_CMD="${SUDO}docker-compose"
    else
         echo "警告: docker-composeが見つかりません。セットアップが失敗する可能性があります。"
    fi
fi

# --- 4. プロジェクトの確保 (カレントディレクトリ) ---
# ※ このスクリプト自体が git clone されたディレクトリに存在するか、
# URL実行された場合は自動でリポジトリを落とすといった処理を書く。
# 今回は既にコード群があるディレクトリ（または展開済み）で実行される前提とする。
echo -e "\n[2/3] ⚙️ 環境変数ファイル(.env)の生成..."
cat << EOF > .env
SYSTEM_ROLE=${SYSTEM_ROLE}
POSTGRES_DB=${POSTGRES_DB}
POSTGRES_USER=${POSTGRES_USER}
POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
POSTGRES_HOST=${POSTGRES_HOST}
POSTGRES_PORT=${POSTGRES_PORT}
ADMINER_PORT=${ADMINER_PORT}
DATABASE_URL=postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@${POSTGRES_HOST}:${POSTGRES_PORT}/${POSTGRES_DB}
MAX_CRAWLER_WORKERS=8
ENABLE_TORCH_COMPILE=0
MAX_GENERATE_TOKENS=256
MODEL_SIZE=small
PYTORCH_CPU_THREADS=3
TRAIN_GRAD_ACCUM_STEPS=1
TRAIN_WARMUP_STEPS=20
TRAIN_MIN_LR_RATIO=0.1
AUTO_TUNE=1
AUTO_TUNE_REALTIME=1
ENABLE_RAG=1
ENABLE_RSS_COLLECTOR=1
RSS_ITEMS_PER_FEED=5
SPM_NUM_THREADS=3
ENABLE_TEXT_DEDUP=1
MIN_TRAIN_CHARS=120
TRAIN_STEPS_PER_CYCLE=50
TRAIN_SEED=42
VAL_EVAL_EVERY=25
VAL_EVAL_BATCHES=20
EARLY_STOPPING_PATIENCE=6
BOOTSTRAP_TOKEN=${BOOTSTRAP_TOKEN}
ALLOW_BOOTSTRAP_PASSWORD=${ALLOW_BOOTSTRAP_PASSWORD}
EOF
echo ".env を作成・更新しました。"

# システムが未稼働の場合にエラーになるのを防ぐためダミーのjsonを作っておく
if [ ! -f system_status.json ]; then
    echo '{"is_running": false}' > system_status.json
fi

# --- 5. バックグラウンド起動 ---
echo -e "\n[3/3] 🚀 LLMエンジンのビルドおよびバックグラウンド起動..."
# Masterならダッシュボードも立ち上がる / Workerならクローラーだけ
eval "$DOCKER_COMPOSE_CMD $DOCKER_PROFILE up -d --build"

echo -e "\n============================================="
echo -e "🎉 インストールと起動がすべて完了しました！"
if [[ "$SYSTEM_ROLE" == "master" ]]; then
    echo -e "📡 ブラウザから http://<このPCのIPアドレス>:8000 にアクセスして"
    echo -e "   稼働状況ダッシュボードを確認できます。"
    echo -e "🗄️ DB Web UI(Adminer): http://localhost:${ADMINER_PORT}"
else
    echo -e "🤖 この子機はバックグラウンドで黙々と自動クロールを続けます。"
    echo -e "   親機のダッシュボードから、世界中の子機の働きを一括確認できます。"
fi
echo -e "💡 停止する場合はコマンド: $DOCKER_COMPOSE_CMD $DOCKER_PROFILE down"
echo -e "=============================================\n"
