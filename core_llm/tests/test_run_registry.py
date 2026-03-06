from __future__ import annotations

import json
from pathlib import Path

from core_llm.pipeline.run_registry import build_run_index, compare_runs


def test_build_run_index_and_compare_runs(tmp_path: Path):
    run_a = tmp_path / "run-a"
    run_b = tmp_path / "nested" / "run-b"
    run_a.mkdir(parents=True)
    run_b.mkdir(parents=True)
    (run_a / "run_summary.json").write_text(
        json.dumps(
            {
                "run_type": "wiki_tiny_sample",
                "created_at": "2026-03-06T01:00:00+00:00",
                "work_dir": str(run_a),
                "kept_docs": 10,
                "train_tokens": 1000,
                "best_val_perplexity": 12.3,
                "steps": ["manifest", "tokenizer", "dataset", "train", "eval"],
                "tokenizer_config": {"vocab_size": 8000},
                "model_config": {"vocab_size": 8000, "block_size": 256},
            }
        ),
        encoding="utf-8",
    )
    (run_b / "run_summary.json").write_text(
        json.dumps(
            {
                "run_type": "wiki_tiny_sample",
                "created_at": "2026-03-06T02:00:00+00:00",
                "work_dir": str(run_b),
                "kept_docs": 12,
                "train_tokens": 1200,
                "best_val_perplexity": 10.1,
                "steps": ["manifest", "tokenizer"],
                "tokenizer_config": {"vocab_size": 16000},
                "model_config": {"vocab_size": 16000, "block_size": 512},
            }
        ),
        encoding="utf-8",
    )

    index = build_run_index(tmp_path)
    compared = compare_runs([run_a, run_b / "run_summary.json"])

    assert [entry["run_name"] for entry in index] == ["run-a", "run-b"]
    assert index[0]["best_val_perplexity"] == 12.3
    assert compared[1]["tokenizer_vocab_size"] == 16000
    assert compared[1]["block_size"] == 512
