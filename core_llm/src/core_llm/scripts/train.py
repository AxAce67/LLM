from __future__ import annotations

import argparse
from pathlib import Path

from core_llm.config import load_model_config, load_train_config
from core_llm.env import load_env_file
from core_llm.notify.discord import (
    build_command_failure_message,
    build_command_started_message,
    build_command_success_message,
    resolve_discord_settings,
    send_discord_message,
)
from core_llm.seed import set_seed
from core_llm.train.loop import train_model


def main() -> None:
    load_env_file(".env.local")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--train-config", required=True)
    ap.add_argument("--data-dir", default="data/prepared")
    ap.add_argument("--checkpoint-dir", default="data/checkpoints")
    ap.add_argument("--discord-webhook-url")
    ap.add_argument("--discord-mention")
    args = ap.parse_args()
    webhook_url, mention = resolve_discord_settings(args.discord_webhook_url, args.discord_mention)

    try:
        if webhook_url:
            send_discord_message(
                webhook_url,
                build_command_started_message(
                    command_name="train",
                    payload={
                        "config": args.config,
                        "train_config": args.train_config,
                        "data_dir": args.data_dir,
                        "checkpoint_dir": args.checkpoint_dir,
                    },
                    mention=mention,
                ),
            )
        model_config = load_model_config(args.config)
        train_config = load_train_config(args.train_config)
        set_seed(train_config.seed)
        result = train_model(
            data_dir=Path(args.data_dir),
            checkpoint_dir=Path(args.checkpoint_dir),
            model_config=model_config,
            train_config=train_config,
        )
        if webhook_url:
            send_discord_message(
                webhook_url,
                build_command_success_message(
                    command_name="train",
                    payload={
                        "checkpoint_dir": args.checkpoint_dir,
                        "step": result.get("step", "-"),
                        "latest_train_loss": result.get("latest_train_loss", "-"),
                        "best_val_perplexity": result.get("best_val_perplexity", "-"),
                    },
                    mention=mention,
                ),
            )
        print(result)
    except Exception as exc:
        if webhook_url:
            send_discord_message(
                webhook_url,
                build_command_failure_message(
                    command_name="train",
                    error=str(exc),
                    mention=mention,
                ),
            )
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
