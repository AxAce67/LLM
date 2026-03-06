from __future__ import annotations

import argparse
from pathlib import Path

from core_llm.env import load_env_file
from core_llm.notify.discord import (
    build_failure_message,
    build_run_message,
    resolve_discord_settings,
    send_discord_message,
)
from core_llm.pipeline.wiki_tiny import run_wiki_tiny_pipeline


def main() -> None:
    load_env_file(".env.local")
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--lang", default="ja")
    ap.add_argument("--raw-dir", default="data/raw/wikipedia")
    ap.add_argument("--dump-path")
    ap.add_argument("--max-docs", type=int, default=5000)
    ap.add_argument("--min-chars", type=int, default=120)
    ap.add_argument("--tokenizer-config", default="configs/tokenizer_ja_tiny_sample.yaml")
    ap.add_argument("--model-config", default="configs/model_tiny_ja_sample.yaml")
    ap.add_argument("--train-config", default="configs/train_tiny_sample_cpu.yaml")
    ap.add_argument("--skip-manifest", action="store_true")
    ap.add_argument("--skip-tokenizer", action="store_true")
    ap.add_argument("--skip-dataset", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--refresh-dump", action="store_true")
    ap.add_argument("--discord-webhook-url")
    ap.add_argument("--discord-mention")
    args = ap.parse_args()

    webhook_url, mention = resolve_discord_settings(args.discord_webhook_url, args.discord_mention)

    try:
        summary = run_wiki_tiny_pipeline(
            work_dir=Path(args.work_dir),
            lang=args.lang,
            raw_dir=Path(args.raw_dir),
            dump_path=Path(args.dump_path) if args.dump_path else None,
            max_docs=args.max_docs,
            min_chars=args.min_chars,
            tokenizer_config=Path(args.tokenizer_config),
            model_config=Path(args.model_config),
            train_config=Path(args.train_config),
            skip_manifest=args.skip_manifest,
            skip_tokenizer=args.skip_tokenizer,
            skip_dataset=args.skip_dataset,
            skip_train=args.skip_train,
            skip_eval=args.skip_eval,
            refresh_dump=args.refresh_dump,
        )
        if webhook_url:
            send_discord_message(webhook_url, build_run_message(summary, mention=mention, success=True))
        print(summary)
    except Exception as exc:
        if webhook_url:
            send_discord_message(
                webhook_url,
                build_failure_message(
                    work_dir=args.work_dir,
                    run_type="wiki_tiny_sample",
                    error=str(exc),
                    mention=mention,
                ),
            )
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
