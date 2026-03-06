from __future__ import annotations

import os
from typing import Any

import requests


def resolve_discord_settings(
    webhook_url: str | None = None,
    mention: str | None = None,
) -> tuple[str | None, str | None]:
    return webhook_url or os.getenv("DISCORD_WEBHOOK_URL"), mention or os.getenv("DISCORD_MENTION")


def build_run_message(summary: dict[str, Any], *, mention: str | None = None, success: bool = True) -> str:
    prefix = "Training completed" if success else "Training failed"
    lines = []
    if mention:
        lines.append(mention)
    lines.extend(
        [
            prefix,
            f"run: {summary.get('work_dir', '-')}",
            f"type: {summary.get('run_type', '-')}",
            f"best_val_perplexity: {summary.get('best_val_perplexity', '-')}",
            f"latest_eval_perplexity: {summary.get('latest_eval_perplexity', '-')}",
            f"train_tokens: {summary.get('train_tokens', '-')}",
            f"steps: {', '.join(summary.get('steps', [])) if summary.get('steps') else '-'}",
        ]
    )
    return "\n".join(lines)


def build_failure_message(
    *,
    work_dir: str,
    run_type: str,
    error: str,
    mention: str | None = None,
) -> str:
    lines = []
    if mention:
        lines.append(mention)
    lines.extend(
        [
            "Training failed",
            f"run: {work_dir}",
            f"type: {run_type}",
            f"error: {error}",
        ]
    )
    return "\n".join(lines)


def send_discord_message(webhook_url: str, content: str, timeout: int = 10) -> None:
    response = requests.post(webhook_url, json={"content": content}, timeout=timeout)
    response.raise_for_status()

