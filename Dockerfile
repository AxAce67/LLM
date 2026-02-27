# 自作LLM自動収集システム用 Dockerfile
# より軽量で安定した環境を構築するためslimイメージを使用
FROM python:3.10-slim

# コンテナ内の作業ディレクトリ設定
WORKDIR /app

# 必要なシステムパッケージのインストール (コンパイル不要なものは省く)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 依存パッケージ情報のコピーとインストール
# (キャッシュを利用してビルドを高速化するため、コードより先にCOPYする)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションのコード全体をコピー
COPY . .

# 起動コマンド（docker-compose側で上書き可能だが、無指定時のデフォルト）
CMD ["python3", "main_controller.py"]
