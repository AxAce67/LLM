from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from core_llm.config import load_model_config, load_train_config
from core_llm.env import load_env_file
from core_llm.notify.discord import (
    build_command_failure_message,
    build_command_progress_message,
    build_command_started_message,
    build_command_success_message,
    collect_machine_status,
    format_duration,
    format_timestamp,
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
    ap.add_argument("--fresh", action="store_true", help="fail if latest.pt already exists")
    ap.add_argument("--discord-webhook-url")
    ap.add_argument("--discord-mention")
    args = ap.parse_args()
    webhook_url, mention = resolve_discord_settings(args.discord_webhook_url, args.discord_mention)

    try:
        if args.fresh and (Path(args.checkpoint_dir) / "latest.pt").exists():
            raise SystemExit(f"Refusing to resume: latest.pt exists in {args.checkpoint_dir}")
        if webhook_url:
            started_payload = {
                "config": args.config,
                "train_config": args.train_config,
                "data_dir": args.data_dir,
                "checkpoint_dir": args.checkpoint_dir,
            }
            started_payload.update(collect_machine_status())
            send_discord_message(
                webhook_url,
                build_command_started_message(command_name="train", payload=started_payload, mention=mention),
            )
        model_config = load_model_config(args.config)
        train_config = load_train_config(args.train_config)
        set_seed(train_config.seed)

        def handle_progress(payload: dict[str, Any]) -> None:
            if not webhook_url:
                return
            progress_payload = {
                "checkpoint_dir": args.checkpoint_dir,
                "step": f"{payload.get('step', '-')}/{payload.get('total_steps', '-')}",
                "latest_train_loss": payload.get("latest_train_loss", "-"),
                "best_val_perplexity": payload.get("best_val_perplexity", "-"),
                "val_perplexity": payload.get("val_perplexity", "-"),
                "elapsed": format_duration(payload.get("elapsed_seconds")),
                "sec_per_step": (
                    f"{float(payload['seconds_per_step']):.2f}"
                    if payload.get("seconds_per_step") is not None
                    else "-"
                ),
                "estimated_finish": format_timestamp(payload.get("estimated_finish")),
            }
            progress_payload.update(collect_machine_status())
            send_discord_message(
                webhook_url,
                build_command_progress_message(
                    command_name="train",
                    payload=progress_payload,
                    mention=mention,
                ),
            )

        result = train_model(
            data_dir=Path(args.data_dir),
            checkpoint_dir=Path(args.checkpoint_dir),
            model_config=model_config,
            train_config=train_config,
            progress_callback=handle_progress,
        )
        if webhook_url:
            success_payload = {
                "checkpoint_dir": args.checkpoint_dir,
                "step": result.get("step", "-"),
                "latest_train_loss": result.get("latest_train_loss", "-"),
                "best_val_perplexity": result.get("best_val_perplexity", "-"),
                "duration": format_duration(result.get("duration_seconds")),
            }
            success_payload.update(collect_machine_status())
            send_discord_message(
                webhook_url,
                build_command_success_message(
                    command_name="train",
                    payload=success_payload,
                    mention=mention,
                ),
            )
        print(result)
    except Exception as exc:
        if webhook_url:
            failure_message = build_command_failure_message(
                command_name="train",
                error=str(exc),
                mention=mention,
            )
            machine_lines = "\n".join(f"{key}: {value}" for key, value in collect_machine_status().items())
            send_discord_message(
                webhook_url,
                f"{failure_message}\n{machine_lines}",
            )
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
