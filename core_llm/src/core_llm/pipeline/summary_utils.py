from __future__ import annotations

import json
from pathlib import Path


from core_llm.train.checkpoint import load_checkpoint


def resolve_best_val_perplexity(checkpoint_dir: str | Path, fallback: float | None = None) -> float | None:
    best_path = Path(checkpoint_dir) / "best.pt"
    latest_path = Path(checkpoint_dir) / "latest.pt"
    if best_path.exists():
        payload = load_checkpoint(best_path, "cpu")
        return payload.get("best_val_perplexity", fallback)
    if latest_path.exists():
        payload = load_checkpoint(latest_path, "cpu")
        return payload.get("best_val_perplexity", fallback)
    return fallback


def read_training_status(metrics_path: str | Path) -> dict[str, int | bool | None]:
    path = Path(metrics_path)
    if not path.exists():
        return {"last_step": None, "early_stopped": False, "early_stop_step": None}
    last_step = None
    early_stopped = False
    early_stop_step = None
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("event") == "train_step":
                last_step = row.get("step", last_step)
            if row.get("event") == "early_stop":
                early_stopped = True
                early_stop_step = row.get("step")
    return {"last_step": last_step, "early_stopped": early_stopped, "early_stop_step": early_stop_step}


def build_run_label(
    work_dir: str | Path | None,
    *,
    last_step: int | None,
    total_steps: int | None,
    early_stopped: bool,
    resumed_from_step: int | None = None,
) -> str:
    name = Path(work_dir).name if work_dir else "run"
    parts = [name]
    if last_step is not None:
        if total_steps is not None:
            parts.append(f"step{last_step}of{total_steps}")
        else:
            parts.append(f"step{last_step}")
    if early_stopped:
        parts.append("early")
    if resumed_from_step is not None:
        parts.append(f"resume{resumed_from_step}")
    return "__".join(parts)
