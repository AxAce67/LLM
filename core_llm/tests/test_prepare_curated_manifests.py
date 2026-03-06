from __future__ import annotations

import json
import sys
from pathlib import Path


def test_prepare_curated_manifests_builds_expected_outputs(tmp_path: Path):
    raw_root = tmp_path / "raw" / "curated"
    notes_dir = raw_root / "local_notes_ja"
    tech_dir = raw_root / "tech_docs_ja"
    notes_dir.mkdir(parents=True)
    tech_dir.mkdir(parents=True)
    (notes_dir / "note1.md").write_text(
        "これは手元ノート由来の日本語テキストです。十分に長い本文として扱います。",
        encoding="utf-8",
    )
    (tech_dir / "guide1.txt").write_text(
        "これは技術文書由来の日本語テキストです。こちらも十分に長い本文として扱います。",
        encoding="utf-8",
    )

    manifest_dir = tmp_path / "manifests"
    summary_path = tmp_path / "curated_sources.report.json"

    from core_llm.scripts.prepare_curated_manifests import main as prepare_curated_main

    old_argv = sys.argv
    sys.argv = [
        "prepare_curated_manifests",
        "--raw-root",
        str(raw_root),
        "--manifest-dir",
        str(manifest_dir),
        "--summary-report",
        str(summary_path),
        "--min-chars",
        "20",
    ]
    try:
        prepare_curated_main()
    finally:
        sys.argv = old_argv

    notes_manifest = manifest_dir / "local_notes_ja.jsonl"
    tech_manifest = manifest_dir / "tech_docs_ja.jsonl"
    assert notes_manifest.exists()
    assert tech_manifest.exists()

    notes_rows = [json.loads(line) for line in notes_manifest.read_text(encoding="utf-8").splitlines()]
    tech_rows = [json.loads(line) for line in tech_manifest.read_text(encoding="utf-8").splitlines()]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert notes_rows[0]["id"].startswith("notes:")
    assert tech_rows[0]["id"].startswith("tech:")
    assert summary["sources"]["local_notes_ja"]["kept_docs"] == 1
    assert summary["sources"]["tech_docs_ja"]["kept_docs"] == 1
