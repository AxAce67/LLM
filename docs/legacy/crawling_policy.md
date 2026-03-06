# Crawling Policy

このプロジェクトは自動収集を行うため、運用時に以下を必ず守ってください。

## 1. robots.txt とアクセス頻度

- `CRAWLER_RESPECT_ROBOTS=1` を維持する
- `CRAWLER_DOMAIN_MIN_INTERVAL_SEC` を 0.5 以上にする
- `CRAWLER_DOMAIN_CONCURRENCY` は 1-2 を推奨
- 高頻度収集が必要な場合でも、対象サイトの利用規約を優先する

## 2. User-Agent と識別性

- クローラーは `DIY-LLM-Crawler/1.0` User-Agent を送信
- 本番では連絡先を含む専用 User-Agent への変更を推奨

## 3. 学習利用可否とライセンス

- `source_policies` でドメインごとに `allow_training` を管理する
- ライセンス不明ソースは `allow_training=false` を推奨
- 公開/配布モデルは、学習ソースの利用許諾を必ず確認する

## 4. 個人情報（PII）対策

- `ENABLE_PII_FILTER=1` を維持する
- メール/電話/郵便番号/IPを含む文書は学習前に除外される
- より厳密な用途では追加の匿名化処理を実装する

## 5. ドメイン制御

- `CRAWLER_ALLOW_SUBDOMAINS=0` を既定とする
- 収集対象を増やす場合は `source_policies` で明示的に許可する

## 6. 禁止事項

- ログイン必須・課金必須コンテンツの無断収集
- スクレイピング禁止サイトへの収集実行
- 収集データの再配布時に出典/ライセンスを無視する運用
