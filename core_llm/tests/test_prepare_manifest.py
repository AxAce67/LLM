from __future__ import annotations

import json
import sys
from pathlib import Path


def test_prepare_manifest_supports_md_and_report(tmp_path: Path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "doc1.txt").write_text(
        "人工知能は計算機科学の分野です。十分に長い日本語テキストです。これはサンプル本文です。",
        encoding="utf-8",
    )
    nested = source_dir / "notes"
    nested.mkdir()
    (nested / "doc2.md").write_text(
        "# 見出し\n\nこれは技術メモ由来の日本語本文です。こちらも十分に長い説明文として扱います。",
        encoding="utf-8",
    )
    (source_dir / "short.txt").write_text("短い", encoding="utf-8")
    output_path = tmp_path / "manifest.jsonl"
    report_path = tmp_path / "manifest.report.json"

    from core_llm.scripts.prepare_manifest import main as prepare_manifest_main

    old_argv = sys.argv
    sys.argv = [
        "prepare_manifest",
        "--input-dir",
        str(source_dir),
        "--output",
        str(output_path),
        "--source",
        "local_notes_ja",
        "--license",
        "permissive-user-provided",
        "--min-chars",
        "20",
        "--id-prefix",
        "notes:",
        "--split-hint",
        "train",
        "--report-path",
        str(report_path),
    ]
    try:
        prepare_manifest_main()
    finally:
        sys.argv = old_argv

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert len(rows) == 2
    assert rows[0]["id"].startswith("notes:")
    assert rows[0]["split_hint"] == "train"
    assert report["kept_docs"] == 2
    assert report["filtered_short"] == 1
    assert report["extensions"] == [".md", ".txt"]


def test_prepare_manifest_include_ext_filters_files(tmp_path: Path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "doc1.txt").write_text(
        "人工知能は計算機科学の分野です。十分に長い日本語テキストです。これはサンプル本文です。",
        encoding="utf-8",
    )
    (source_dir / "doc2.md").write_text(
        "これは Markdown 側の日本語本文です。こちらも十分に長い説明として扱います。",
        encoding="utf-8",
    )
    output_path = tmp_path / "manifest.jsonl"

    from core_llm.scripts.prepare_manifest import main as prepare_manifest_main

    old_argv = sys.argv
    sys.argv = [
        "prepare_manifest",
        "--input-dir",
        str(source_dir),
        "--output",
        str(output_path),
        "--source",
        "tech_docs_ja",
        "--license",
        "cc-by-4.0",
        "--min-chars",
        "20",
        "--include-ext",
        "md",
    ]
    try:
        prepare_manifest_main()
    finally:
        sys.argv = old_argv

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 1
    assert rows[0]["id"] == "doc2"
