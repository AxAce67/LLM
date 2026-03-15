from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _summary_path(run_ref: str | Path) -> Path:
    path = Path(run_ref)
    if path.is_dir():
        return path / "run_summary.json"
    return path


def load_run_summary(run_ref: str | Path) -> dict[str, Any]:
    path = _summary_path(run_ref)
    if not path.exists():
        raise FileNotFoundError(f"Run summary does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def build_run_index(runs_dir: str | Path) -> list[dict[str, Any]]:
    root = Path(runs_dir)
    entries: list[dict[str, Any]] = []
    for path in sorted(root.rglob("run_summary.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        entries.append(
            {
                "run_name": summary.get("run_label") or summary.get("run_name") or path.parent.name,
                "run_label": summary.get("run_label"),
                "work_dir": summary.get("work_dir"),
                "run_type": summary.get("run_type"),
                "created_at": summary.get("created_at"),
                "kept_docs": summary.get("kept_docs"),
                "train_tokens": summary.get("train_tokens"),
                "best_val_perplexity": summary.get("best_val_perplexity"),
                "last_step": summary.get("last_step"),
                "early_stopped": summary.get("early_stopped"),
                "completed_steps": summary.get("steps", []),
            }
        )
    entries.sort(key=lambda item: (item.get("created_at") or "", item.get("run_name") or ""))
    return entries


def compare_runs(run_refs: list[str | Path]) -> list[dict[str, Any]]:
    compared: list[dict[str, Any]] = []
    for run_ref in run_refs:
        summary = load_run_summary(run_ref)
        run_name = Path(summary["work_dir"]).name if summary.get("work_dir") else Path(run_ref).stem
        compared.append(
            {
                "run_name": summary.get("run_label") or summary.get("run_name") or run_name,
                "run_label": summary.get("run_label"),
                "work_dir": summary.get("work_dir"),
                "run_type": summary.get("run_type"),
                "created_at": summary.get("created_at"),
                "kept_docs": summary.get("kept_docs"),
                "train_tokens": summary.get("train_tokens"),
                "best_val_perplexity": summary.get("best_val_perplexity"),
                "last_step": summary.get("last_step"),
                "early_stopped": summary.get("early_stopped"),
                "tokenizer_vocab_size": summary.get("tokenizer_config", {}).get("vocab_size"),
                "model_vocab_size": summary.get("model_config", {}).get("vocab_size"),
                "block_size": summary.get("model_config", {}).get("block_size"),
                "steps": summary.get("steps", []),
            }
        )
    return compared
