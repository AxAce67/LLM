import json

import pytest

from core_llm.data.manifest_schema import ManifestRecord, iter_manifest


def test_manifest_record_requires_fields():
    with pytest.raises(ValueError):
        ManifestRecord.from_dict({"id": "x", "text": "abc"})


def test_iter_manifest_reads_rows(tmp_path):
    path = tmp_path / "manifest.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "doc-1",
                "text": "人工知能の説明文です。" * 10,
                "lang": "ja",
                "source": "fixture",
                "license": "cc-by-sa-4.0",
                "split_hint": "train",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    rows = list(iter_manifest(path))
    assert len(rows) == 1
    assert rows[0].lang == "ja"
