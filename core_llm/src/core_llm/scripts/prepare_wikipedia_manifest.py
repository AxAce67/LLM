from __future__ import annotations

import argparse
from pathlib import Path

from core_llm.data.wiki_dump import build_wikipedia_manifest
from core_llm.env import load_env_file
from core_llm.notify.discord import (
    build_command_failure_message,
    build_command_success_message,
    resolve_discord_settings,
    send_discord_message,
)


def main() -> None:
    load_env_file(".env.local")
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="ja")
    ap.add_argument("--output", required=True)
    ap.add_argument("--raw-dir", default="data/raw/wikipedia")
    ap.add_argument("--dump-path")
    ap.add_argument("--min-chars", type=int, default=120)
    ap.add_argument("--max-docs", type=int)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--keep-dump", action="store_true")
    ap.add_argument("--report-path")
    ap.add_argument("--discord-webhook-url")
    ap.add_argument("--discord-mention")
    args = ap.parse_args()
    webhook_url, mention = resolve_discord_settings(args.discord_webhook_url, args.discord_mention)

    try:
        report = build_wikipedia_manifest(
            lang=args.lang,
            output_path=Path(args.output),
            raw_dir=Path(args.raw_dir),
            dump_path=Path(args.dump_path) if args.dump_path else None,
            min_chars=args.min_chars,
            max_docs=args.max_docs,
            refresh=args.refresh,
            report_path=Path(args.report_path) if args.report_path else None,
        )
        if webhook_url:
            send_discord_message(
                webhook_url,
                build_command_success_message(
                    command_name="prepare_wikipedia_manifest",
                    payload={
                        "output": args.output,
                        "kept_docs": report.get("kept_docs", 0),
                        "total_pages": report.get("total_pages", 0),
                    },
                    mention=mention,
                ),
            )
        print(report)
        if report["kept_docs"] <= 0:
            raise SystemExit(1)
    except Exception as exc:
        if webhook_url:
            send_discord_message(
                webhook_url,
                build_command_failure_message(
                    command_name="prepare_wikipedia_manifest",
                    error=str(exc),
                    mention=mention,
                ),
            )
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
