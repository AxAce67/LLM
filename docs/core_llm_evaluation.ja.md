# `core_llm` 評価方針

English: [core_llm_evaluation.md](core_llm_evaluation.md)

## 主指標

- validation perplexity

## 副チェック

- 生成のスモーク確認

## 非対象

- キーワード系の自動スコア
- DB 連動の評価実行
- プロダクト向けベンチマーク主張

## 現在の出力

- `core_llm/data/eval/perplexity.json`
- run の評価: `<work-dir>/eval/perplexity.json`
- run の summary: `<work-dir>/run_summary.json`
- `evaluate_prompt_set` の summary: `<path>.summary.json`

## 解釈

- perplexity が低いほど良い
- 生成のスモーク確認は定性評価
- 初期目標はパイプラインの正しさ
- 変更後は `run_summary.json` をまず確認

## プロンプトセット評価

```bash
python3 -m core_llm.scripts.evaluate_prompt_set \
  --checkpoint <work-dir>/checkpoints/best.pt \
  --tokenizer <work-dir>/tokenizer/tokenizer.model \
  --questions data/raw/sft/eval_questions_ja.jsonl \
  --output data/eval/sft_eval.jsonl
```

`*.summary.json` の主な指標:

- `qa_ok_rate`: 簡易正答率
- `avg_repeat_trigram_ratio`: 反復の強さ
- `avg_symbol_ratio` / `avg_latin_ratio`: ノイズ傾向
- `category_stats`: カテゴリ別 QA OK 率
