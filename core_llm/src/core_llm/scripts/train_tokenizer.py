from __future__ import annotations

import argparse
from pathlib import Path

from core_llm.config import load_tokenizer_config
from core_llm.env import load_env_file
from core_llm.notify.discord import (
    build_command_failure_message,
    build_command_started_message,
    build_command_success_message,
    resolve_discord_settings,
    send_discord_message,
)
from core_llm.tokenizer.trainer import train_tokenizer


def main() -> None:
    load_env_file(".env.local")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output-dir", default="data/tokenizer")
    ap.add_argument("--discord-webhook-url")
    ap.add_argument("--discord-mention")
    args = ap.parse_args()
    webhook_url, mention = resolve_discord_settings(args.discord_webhook_url, args.discord_mention)

    try:
        if webhook_url:
            send_discord_message(
                webhook_url,
                build_command_started_message(
                    command_name="train_tokenizer",
                    payload={
                        "manifest": args.manifest,
                        "output_dir": args.output_dir,
                    },
                    mention=mention,
                ),
            )
        config = load_tokenizer_config(args.config)
        model_path = train_tokenizer(args.manifest, Path(args.output_dir), config)
        if webhook_url:
            send_discord_message(
                webhook_url,
                build_command_success_message(
                    command_name="train_tokenizer",
                    payload={
                        "manifest": args.manifest,
                        "output_dir": args.output_dir,
                        "tokenizer_model": str(model_path),
                    },
                    mention=mention,
                ),
            )
        print(f"Tokenizer saved to {model_path}")
    except Exception as exc:
        if webhook_url:
            send_discord_message(
                webhook_url,
                build_command_failure_message(
                    command_name="train_tokenizer",
                    error=str(exc),
                    mention=mention,
                ),
            )
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
