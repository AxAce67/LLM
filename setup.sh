#!/bin/bash
# ==============================================================================
# 自作LLM分散収集システム（Docker）完全自動インストーラー
# 使い方: curl -sL https://raw.githubusercontent.com/.../setup.sh | bash
#         または bash setup.sh [--master]
# ==============================================================================

set -e

# ------------------------------------------------------------------------------

# --- 1. システムの役割（Role）選択 ---
echo -e "\n============================================="
echo -e " LLM Data Collector 自動セットアップウィザード"
echo -e "=============================================\n"

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

# --- 2. クラウドDB（Supabase）接続情報の設定 ---
echo -e "\n============================================="
echo -e " LLM Data Collector 自動セットアップウィザード"
echo -e "=============================================\n"
echo -e "[設定] データを保存・共有するためのSupabase接続情報を入力してください。"
echo -e "※この情報はあなたのPC内にのみ保存され（.env）、外部には送信されません。"

while true; do
    read -p "Supabase URL: " SUPABASE_URL < /dev/tty
    if [ -n "$SUPABASE_URL" ]; then
        break
    else
        echo "[!] URLは必須です。もう一度入力してください。"
    fi
done

while true; do
    read -p "Supabase ANON KEY: " SUPABASE_KEY < /dev/tty
    if [ -n "$SUPABASE_KEY" ]; then
        break
    else
        echo "[!] KEYは必須です。もう一度入力してください。"
    fi
done

echo -e "\n[✓] 接続情報を一時記憶しました。"

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
SUPABASE_URL=${SUPABASE_URL}
SUPABASE_KEY=${SUPABASE_KEY}
SYSTEM_ROLE=${SYSTEM_ROLE}
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
else
    echo -e "🤖 この子機はバックグラウンドで黙々と自動クロールを続けます。"
    echo -e "   親機のダッシュボードから、世界中の子機の働きを一括確認できます。"
fi
echo -e "💡 停止する場合はコマンド: $DOCKER_COMPOSE_CMD $DOCKER_PROFILE down"
echo -e "=============================================\n"
