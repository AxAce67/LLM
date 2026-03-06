from __future__ import annotations

import json
from pathlib import Path

from core_llm.data.manifest_ops import merge_manifests
from core_llm.data.manifest_schema import ManifestRecord, write_manifest


def test_merge_manifests_merges_sources_and_deduplicates(tmp_path: Path):
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    write_manifest(
        first,
        [
            ManifestRecord(
                id="doc-1",
                text="人工知能は計算機科学の分野です。十分に長い日本語の本文です。",
                lang="ja",
                source="wikipedia_ja",
                license="cc-by-sa-4.0",
            ),
            ManifestRecord(
                id="doc-2",
                text="これは短い",
                lang="ja",
                source="wikipedia_ja",
                license="cc-by-sa-4.0",
            ),
        ],
    )
    write_manifest(
        second,
        [
            ManifestRecord(
                id="doc-1",
                text="人工知能は計算機科学の分野です。十分に長い日本語の本文です。",
                lang="ja",
                source="tech_docs_ja",
                license="cc-by-4.0",
            ),
            ManifestRecord(
                id="doc-1",
                text="技術文書の別本文です。こちらも十分に長い日本語テキストとして扱います。",
                lang="ja",
                source="tech_docs_ja",
                license="cc-by-4.0",
            ),
        ],
    )

    output = tmp_path / "merged.jsonl"
    report = merge_manifests(inputs=[first, second], output=output, min_chars=20)
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

    assert report["kept_docs"] == 2
    assert report["filtered_short"] == 1
    assert report["filtered_duplicate_text"] == 1
    assert report["deduplicated_id_collisions"] == 1
    assert report["source_counts"] == {"tech_docs_ja": 1, "wikipedia_ja": 1}
    assert len(rows) == 2
    assert rows[0]["source"] == "wikipedia_ja"
    assert rows[1]["id"].startswith("doc-1:")
