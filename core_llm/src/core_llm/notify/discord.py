from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
from datetime import datetime
from typing import Any

import requests


def resolve_discord_settings(
    webhook_url: str | None = None,
    mention: str | None = None,
) -> tuple[str | None, str | None]:
    return webhook_url or os.getenv("DISCORD_WEBHOOK_URL"), mention or os.getenv("DISCORD_MENTION")


def build_run_message(summary: dict[str, Any], *, mention: str | None = None, success: bool = True) -> str:
    prefix = "✅ Training completed" if success else "❌ Training failed"
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


def build_run_started_message(
    *,
    work_dir: str,
    run_type: str,
    mention: str | None = None,
    payload: dict[str, Any] | None = None,
) -> str:
    lines = []
    if mention:
        lines.append(mention)
    lines.extend(
        [
            "▶ Training started",
            f"run: {work_dir}",
            f"type: {run_type}",
        ]
    )
    if payload:
        for key, value in payload.items():
            lines.append(f"{key}: {value}")
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
            "❌ Training failed",
            f"run: {work_dir}",
            f"type: {run_type}",
            f"error: {error}",
        ]
    )
    return "\n".join(lines)


def build_run_renamed_message(
    *,
    work_dir: str,
    new_work_dir: str,
    run_type: str,
    mention: str | None = None,
) -> str:
    lines = []
    if mention:
        lines.append(mention)
    lines.extend(
        [
            "🔁 Run renamed",
            f"run: {work_dir}",
            f"new_run: {new_work_dir}",
            f"type: {run_type}",
        ]
    )
    return "\n".join(lines)


def build_command_success_message(
    *,
    command_name: str,
    payload: dict[str, Any],
    mention: str | None = None,
) -> str:
    lines = []
    if mention:
        lines.append(mention)
    lines.extend(
        [
            "✅ Command completed",
            f"command: {command_name}",
        ]
    )
    for key, value in payload.items():
        lines.append(f"{key}: {value}")
    return "\n".join(lines)


def build_command_progress_message(
    *,
    command_name: str,
    payload: dict[str, Any],
    mention: str | None = None,
) -> str:
    lines = []
    if mention:
        lines.append(mention)
    lines.extend(
        [
            "⏳ Command progress",
            f"command: {command_name}",
        ]
    )
    for key, value in payload.items():
        lines.append(f"{key}: {value}")
    return "\n".join(lines)


def build_command_started_message(
    *,
    command_name: str,
    payload: dict[str, Any],
    mention: str | None = None,
) -> str:
    lines = []
    if mention:
        lines.append(mention)
    lines.extend(
        [
            "▶ Command started",
            f"command: {command_name}",
        ]
    )
    for key, value in payload.items():
        lines.append(f"{key}: {value}")
    return "\n".join(lines)


def build_command_failure_message(
    *,
    command_name: str,
    error: str,
    mention: str | None = None,
) -> str:
    lines = []
    if mention:
        lines.append(mention)
    lines.extend(
        [
            "❌ Command failed",
            f"command: {command_name}",
            f"error: {error}",
        ]
    )
    return "\n".join(lines)


def send_discord_message(webhook_url: str, content: str, timeout: int = 10) -> bool:
    try:
        response = requests.post(webhook_url, json={"content": content}, timeout=timeout)
        response.raise_for_status()
        return True
    except requests.RequestException as exc:
        print(f"[discord] failed to send notification: {exc}", file=sys.stderr)
        return False


def format_duration(seconds: float | int | None) -> str:
    if seconds is None:
        return "-"
    total = max(0, int(round(float(seconds))))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def format_timestamp(seconds: float | int | None) -> str:
    if seconds is None:
        return "-"
    return datetime.fromtimestamp(float(seconds)).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def _linux_memory_status() -> tuple[str, str]:
    meminfo_path = "/proc/meminfo"
    if not os.path.exists(meminfo_path):
        return "-", "-"
    values: dict[str, int] = {}
    with open(meminfo_path, "r", encoding="utf-8") as f:
        for line in f:
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            parts = value.strip().split()
            if not parts:
                continue
            try:
                values[key] = int(parts[0])
            except ValueError:
                continue
    total_kib = values.get("MemTotal")
    available_kib = values.get("MemAvailable")
    if not total_kib or available_kib is None:
        return "-", "-"
    used_kib = max(0, total_kib - available_kib)
    gib = 1024 * 1024
    return f"{used_kib / gib:.1f}GiB", f"{total_kib / gib:.1f}GiB"


def collect_machine_status() -> dict[str, str]:
    status: dict[str, str] = {
        "host": socket.gethostname(),
        "cpu_threads": str(max(1, os.cpu_count() or 1)),
    }
    ram_used, ram_total = _linux_memory_status()
    if ram_used != "-" and ram_total != "-":
        status["ram"] = f"{ram_used} / {ram_total}"
    total, used, free = shutil.disk_usage(".")
    gib = 1024**3
    status["disk_free"] = f"{free / gib:.1f}GiB"
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        ).strip()
        if output:
            name, gpu_util, mem_used, mem_total, temp = [part.strip() for part in output.split(",", 4)]
            status["gpu"] = name
            status["gpu_util"] = f"{gpu_util}%"
            status["vram"] = f"{mem_used}MiB / {mem_total}MiB"
            status["gpu_temp"] = f"{temp}C"
    except (subprocess.SubprocessError, FileNotFoundError, ValueError):
        pass
    return status
