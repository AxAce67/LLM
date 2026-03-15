from __future__ import annotations

import bz2
import json
import sys
from pathlib import Path

from core_llm.config import TokenizerConfig
from core_llm.tokenizer.trainer import train_tokenizer


def _write_dump(path: Path) -> None:
    pages = []
    for idx in range(6):
        pages.append(
            f"""
            <page>
              <title>記事{idx}</title>
              <ns>0</ns>
              <id>{idx + 1}</id>
              <revision><id>{idx + 10}</id><text xml:space="preserve">人工知能は計算機科学の分野であり、日本語の学習用テキストです。これは検証用の本文 {idx} です。</text></revision>
            </page>
            """
        )
    xml = '<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.10/">' + "".join(pages) + "</mediawiki>"
    with bz2.open(path, "wt", encoding="utf-8") as f:
        f.write(xml)


def test_prepare_wikipedia_manifest_and_dataset_pipeline(tmp_path: Path):
    dump_path = tmp_path / "jawiki-latest-pages-articles.xml.bz2"
    _write_dump(dump_path)
    manifest_path = tmp_path / "wikipedia_ja.jsonl"
    report_path = tmp_path / "wikipedia_ja.report.json"

    from core_llm.scripts.prepare_wikipedia_manifest import main as prepare_wiki_main

    old_argv = sys.argv
    sys.argv = [
        "prepare_wikipedia_manifest",
        "--lang", "ja",
        "--output", str(manifest_path),
        "--dump-path", str(dump_path),
        "--min-chars", "40",
        "--report-path", str(report_path),
    ]
    try:
        prepare_wiki_main()
    finally:
        sys.argv = old_argv

    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert rows
    assert rows[0]["source"] == "wikipedia_ja"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["kept_docs"] > 0

    model_path = train_tokenizer(
        manifest_path,
        tmp_path / "tokenizer",
        TokenizerConfig(vocab_size=64, input_sentence_size=1000),
    )
    cfg_path = tmp_path / "model.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "vocab_size: 64",
                "block_size: 16",
                "n_layer: 2",
                "n_head: 2",
                "n_embd: 32",
                "dropout: 0.1",
                "bias: false",
            ]
        ),
        encoding="utf-8",
    )

    from core_llm.scripts.prepare_dataset import main as prepare_dataset_main

    old_argv = sys.argv
    sys.argv = [
        "prepare_dataset",
        "--config", str(cfg_path),
        "--manifest", str(manifest_path),
        "--tokenizer", str(model_path),
        "--output-dir", str(tmp_path / "prepared"),
        "--min-chars", "20",
    ]
    try:
        prepare_dataset_main()
    finally:
        sys.argv = old_argv

    metadata = json.loads((tmp_path / "prepared" / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["train_tokens"] > 0


def test_prepare_wikipedia_manifest_respects_max_docs(tmp_path: Path):
    dump_path = tmp_path / "jawiki-latest-pages-articles.xml.bz2"
    _write_dump(dump_path)
    manifest_path = tmp_path / "limited.jsonl"

    from core_llm.scripts.prepare_wikipedia_manifest import main as prepare_wiki_main

    old_argv = sys.argv
    sys.argv = [
        "prepare_wikipedia_manifest",
        "--lang", "ja",
        "--output", str(manifest_path),
        "--dump-path", str(dump_path),
        "--min-chars", "40",
        "--max-docs", "2",
    ]
    try:
        prepare_wiki_main()
    finally:
        sys.argv = old_argv

    rows = [line for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 2
