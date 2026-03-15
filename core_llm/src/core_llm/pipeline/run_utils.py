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


def log_run_event(log_path: str | Path, payload: dict, *, rotate_daily: bool = False) -> None:
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"ts": datetime.now(timezone.utc).isoformat(), **payload}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    if rotate_daily:
        date_tag = datetime.now(timezone.utc).strftime("%Y%m%d")
        rotated = path.with_name(f"{path.stem}_{date_tag}{path.suffix}")
        with rotated.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def apply_run_label_dir(work_dir: Path, run_label: str) -> Path:
    if not run_label:
        return work_dir
    if work_dir.name == run_label:
        return work_dir
    target = work_dir.parent / run_label
    if target.exists():
        for idx in range(1, 1000):
            candidate = work_dir.parent / f"{run_label}__dup{idx}"
            if not candidate.exists():
                target = candidate
                break
        else:
            raise FileExistsError(f"Cannot rename run dir; too many collisions for {run_label}")
    work_dir.rename(target)
    return target


def rewrite_summary_paths(summary: dict, old_dir: Path, new_dir: Path) -> dict:
    old = str(old_dir)
    new = str(new_dir)
    for key, value in list(summary.items()):
        if isinstance(value, str) and value.startswith(old):
            summary[key] = new + value[len(old) :]
    return summary
