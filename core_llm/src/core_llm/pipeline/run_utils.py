from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


def _sanitize_tag(tag: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_]+", "", tag)
    return cleaned[:48]


def build_default_work_dir(
    run_type: str,
    tags: Iterable[str],
    *,
    base_dir: str | Path = "data/runs",
) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    safe_tags = [_sanitize_tag(tag) for tag in tags if tag]
    parts = [run_type, ts, *[tag for tag in safe_tags if tag]]
    name = "_".join(parts)
    return Path(base_dir) / name


def log_run_event(log_path: str | Path, payload: dict) -> None:
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"ts": datetime.now(timezone.utc).isoformat(), **payload}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
