from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from core_llm.env import load_env_file
from core_llm.notify.discord import (
    build_failure_message,
    build_run_message,
    build_run_renamed_message,
    build_run_started_message,
    resolve_discord_settings,
    send_discord_message,
)
from core_llm.pipeline.run_utils import apply_run_label_dir, build_default_work_dir, log_run_event, rewrite_summary_paths
from core_llm.pipeline.wiki_tiny import run_wiki_tiny_pipeline


def main() -> None:
    load_env_file(".env.local")
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir")
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
    work_dir = Path(args.work_dir) if args.work_dir else build_default_work_dir(
        "wiki_tiny_sample",
        tags=[
            Path(args.model_config).stem,
            Path(args.train_config).stem,
            f"docs{args.max_docs}",
        ],
    )

    webhook_url, mention = resolve_discord_settings(args.discord_webhook_url, args.discord_mention)
    global_log = Path("data/runs/run_log.jsonl")
    run_log = work_dir / "run_log.jsonl"
    log_run_event(
        global_log,
        {
            "event": "run_start",
            "command": "run_wiki_tiny",
            "work_dir": str(work_dir),
            "argv": sys.argv,
        },
        rotate_daily=True,
    )
    log_run_event(
        run_log,
        {
            "event": "run_start",
            "command": "run_wiki_tiny",
            "work_dir": str(work_dir),
            "args": vars(args),
        },
    )

    try:
        if webhook_url:
            send_discord_message(
                webhook_url,
                build_run_started_message(
                    work_dir=str(work_dir),
                    run_type="wiki_tiny_sample",
                    mention=mention,
                    payload={
                        "model_config": args.model_config,
                        "train_config": args.train_config,
                        "max_docs": args.max_docs,
                    },
                ),
            )
        summary = run_wiki_tiny_pipeline(
            work_dir=work_dir,
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
        final_dir = apply_run_label_dir(work_dir, summary.get("run_label", ""))
        if final_dir != work_dir:
            summary = rewrite_summary_paths(summary, work_dir, final_dir)
            summary["work_dir"] = str(final_dir)
            (final_dir / "run_summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            log_run_event(
                global_log,
                {
                    "event": "run_rename",
                    "command": "run_wiki_tiny",
                    "work_dir": str(work_dir),
                    "new_work_dir": str(final_dir),
                },
                rotate_daily=True,
            )
            log_run_event(
                final_dir / "run_log.jsonl",
                {
                    "event": "run_rename",
                    "work_dir": str(work_dir),
                    "new_work_dir": str(final_dir),
                },
            )
            run_log = final_dir / "run_log.jsonl"
            if webhook_url:
                send_discord_message(
                    webhook_url,
                    build_run_renamed_message(
                        work_dir=str(work_dir),
                        new_work_dir=str(final_dir),
                        run_type="wiki_tiny_sample",
                        mention=mention,
                    ),
                )
        log_run_event(
            global_log,
            {
                "event": "run_success",
                "command": "run_wiki_tiny",
                "work_dir": str(final_dir),
                "summary_path": str(final_dir / "run_summary.json"),
            },
            rotate_daily=True,
        )
        log_run_event(
            run_log,
            {
                "event": "run_success",
                "summary": summary,
            },
        )
        print(summary)
    except KeyboardInterrupt as exc:
        log_run_event(
            global_log,
            {
                "event": "run_error",
                "command": "run_wiki_tiny",
                "work_dir": str(work_dir),
                "error": "Interrupted (KeyboardInterrupt)",
            },
            rotate_daily=True,
        )
        log_run_event(
            run_log,
            {
                "event": "run_error",
                "error": "Interrupted (KeyboardInterrupt)",
            },
        )
        if webhook_url:
            send_discord_message(
                webhook_url,
                build_failure_message(
                    work_dir=str(work_dir),
                    run_type="wiki_tiny_sample",
                    error="Interrupted (KeyboardInterrupt)",
                    mention=mention,
                ),
            )
        raise SystemExit("Interrupted") from exc
    except Exception as exc:
        log_run_event(
            global_log,
            {
                "event": "run_error",
                "command": "run_wiki_tiny",
                "work_dir": str(work_dir),
                "error": str(exc),
            },
            rotate_daily=True,
        )
        log_run_event(
            run_log,
            {
                "event": "run_error",
                "error": str(exc),
            },
        )
        if webhook_url:
            send_discord_message(
                webhook_url,
                build_failure_message(
                    work_dir=str(work_dir),
                    run_type="wiki_tiny_sample",
                    error=str(exc),
                    mention=mention,
                ),
            )
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
