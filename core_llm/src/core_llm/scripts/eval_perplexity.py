from __future__ import annotations

import argparse
import json
from pathlib import Path

from core_llm.env import load_env_file
from core_llm.eval.perplexity import evaluate_checkpoint_perplexity
from core_llm.notify.discord import (
    build_command_failure_message,
    build_command_started_message,
    build_command_success_message,
    resolve_discord_settings,
    send_discord_message,
)


def main() -> None:
    load_env_file(".env.local")
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-dir", default="data/prepared")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--output", default="data/eval/perplexity.json")
    ap.add_argument("--discord-webhook-url")
    ap.add_argument("--discord-mention")
    args = ap.parse_args()
    webhook_url, mention = resolve_discord_settings(args.discord_webhook_url, args.discord_mention)

    try:
        if webhook_url:
            send_discord_message(
                webhook_url,
                build_command_started_message(
                    command_name="eval_perplexity",
                    payload={
                        "checkpoint": args.checkpoint,
                        "data_dir": args.data_dir,
                        "output": args.output,
                    },
                    mention=mention,
                ),
            )
        result = evaluate_checkpoint_perplexity(args.checkpoint, args.data_dir, args.batch_size, args.device)
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        if webhook_url:
            send_discord_message(
                webhook_url,
                build_command_success_message(
                    command_name="eval_perplexity",
                    payload={
                        "checkpoint": args.checkpoint,
                        "val_loss": result.get("val_loss", "-"),
                        "val_perplexity": result.get("val_perplexity", "-"),
                        "output": args.output,
                    },
                    mention=mention,
                ),
            )
        print(json.dumps(result, ensure_ascii=False))
    except KeyboardInterrupt as exc:
        if webhook_url:
            send_discord_message(
                webhook_url,
                build_command_failure_message(
                    command_name="eval_perplexity",
                    error="Interrupted (KeyboardInterrupt)",
                    mention=mention,
                ),
            )
        raise SystemExit("Interrupted") from exc
    except Exception as exc:
        if webhook_url:
            send_discord_message(
                webhook_url,
                build_command_failure_message(
                    command_name="eval_perplexity",
                    error=str(exc),
                    mention=mention,
                ),
            )
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
