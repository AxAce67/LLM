from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture()
def sample_manifest(tmp_path: Path) -> Path:
    rows = [
        {
            "id": f"doc-{idx:03d}",
            "text": (
                "人工知能は計算機が知的な処理を行うための技術です。"
                "機械学習や自然言語処理の基盤になります。"
                f" これはサンプル文書 {idx} です。"
            ),
            "lang": "ja",
            "source": "fixture",
            "license": "cc-by-sa-4.0",
            "split_hint": "train" if idx < 6 else "val",
        }
        for idx in range(8)
    ]
    path = tmp_path / "manifest.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path
