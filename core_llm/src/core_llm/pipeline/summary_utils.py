from __future__ import annotations

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
