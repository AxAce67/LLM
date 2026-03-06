from __future__ import annotations

import bz2
from pathlib import Path

from core_llm.data.wiki_dump import (
    extract_plaintext_from_revision_text,
    iter_wiki_pages_from_bz2,
    latest_dump_url,
    page_to_manifest_record,
)


def _write_fixture(path: Path) -> None:
    xml = """<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.10/">
    <page>
      <title>人工知能</title>
      <ns>0</ns>
      <id>1</id>
      <revision><id>11</id><text xml:space="preserve">'''人工知能'''は[[計算機]]の知的処理である。==概要==<ref>note</ref></text></revision>
    </page>
    <page>
      <title>Category page</title>
      <ns>14</ns>
      <id>2</id>
      <revision><id>12</id><text xml:space="preserve">カテゴリです。</text></revision>
    </page>
    <page>
      <title>Redirect page</title>
      <ns>0</ns>
      <id>3</id>
      <redirect title="人工知能" />
      <revision><id>13</id><text xml:space="preserve">#REDIRECT [[人工知能]]</text></revision>
    </page>
    </mediawiki>"""
    with bz2.open(path, "wt", encoding="utf-8") as f:
        f.write(xml)


def test_latest_dump_url_ja():
    assert latest_dump_url("ja").endswith("/jawiki/latest/jawiki-latest-pages-articles.xml.bz2")


def test_iter_wiki_pages_from_bz2(tmp_path: Path):
    dump_path = tmp_path / "jawiki.xml.bz2"
    _write_fixture(dump_path)
    pages = list(iter_wiki_pages_from_bz2(dump_path))
    assert len(pages) == 3
    assert pages[0]["title"] == "人工知能"
    assert pages[1]["ns"] == "14"
    assert pages[2]["redirect"] is True


def test_extract_plaintext_from_revision_text_removes_markup():
    text = extract_plaintext_from_revision_text("'''人工知能'''は[[計算機|コンピュータ]]の技術である。<ref>x</ref>")
    assert "コンピュータ" in text
    assert "<ref>" not in text
    assert "[[" not in text


def test_page_to_manifest_record_filters_namespace_and_redirect():
    assert page_to_manifest_record({"page_id": "1", "ns": "14", "redirect": False, "text": "日本語" * 100}, min_chars=10) is None
    assert page_to_manifest_record({"page_id": "2", "ns": "0", "redirect": True, "text": "日本語" * 100}, min_chars=10) is None


def test_page_to_manifest_record_accepts_japanese_main_page():
    record = page_to_manifest_record(
        {
            "page_id": "10",
            "ns": "0",
            "redirect": False,
            "text": "人工知能は計算機科学の重要な分野です。" * 8,
        },
        min_chars=40,
    )
    assert record is not None
    assert record.id == "jawiki:10"
    assert record.source == "wikipedia_ja"
