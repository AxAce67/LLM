from __future__ import annotations

from pathlib import Path

from core_llm.pipeline.summary_utils import resolve_best_val_perplexity
from core_llm.train.checkpoint import save_checkpoint


def test_resolve_best_val_perplexity_prefers_best_checkpoint(tmp_path: Path):
    checkpoint_dir = tmp_path / "checkpoints"
    save_checkpoint(checkpoint_dir / "latest.pt", {"best_val_perplexity": 900.0})
    save_checkpoint(checkpoint_dir / "best.pt", {"best_val_perplexity": 700.0})

    assert resolve_best_val_perplexity(checkpoint_dir, fallback=1200.0) == 700.0


def test_resolve_best_val_perplexity_falls_back_to_latest_or_fallback(tmp_path: Path):
    checkpoint_dir = tmp_path / "checkpoints"
    save_checkpoint(checkpoint_dir / "latest.pt", {"best_val_perplexity": 800.0})
    assert resolve_best_val_perplexity(checkpoint_dir, fallback=1200.0) == 800.0

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    assert resolve_best_val_perplexity(empty_dir, fallback=1200.0) == 1200.0
