from __future__ import annotations

import json
from pathlib import Path

from core_llm.data.government_fetch import (
    clean_government_text,
    extract_text_from_html,
    fetch_government_corpus,
    is_allowed_government_url,
)


class _DummyResponse:
    def __init__(self, text: str, status_code: int = 200):
        self.text = text
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError("bad status")


def test_is_allowed_government_url():
    assert is_allowed_government_url("https://www.digital.go.jp/resources/open_data")
    assert is_allowed_government_url("https://example.go.jp/policy")
    assert not is_allowed_government_url("https://example.com/page")


def test_extract_text_from_html_prefers_main_content():
    html = """
    <html>
      <body>
        <header>menu</header>
        <main>
          <h1>公的データ</h1>
          <p>これは日本語の本文です。十分に長い説明文です。</p>
        </main>
        <footer>footer</footer>
      </body>
    </html>
    """
    text = extract_text_from_html(html)
    assert "これは日本語の本文です" in text
    assert "menu" not in text


def test_clean_government_text_removes_metadata_lines():
    raw = "\n".join(
        [
            "最終更新日:",
            "2026年1月30日",
            "議事次第",
            "資料1：説明資料（PDF／112KB）",
            "関連政策",
            "問合せ先",
            "電話：03-0000-0000",
            "メール：sample_atmark_digital.go.jp",
            "これは日本語の本文です。",
            "行政サービスの改善を進めます。",
        ]
    )
    cleaned = clean_government_text(raw)
    assert "最終更新日" not in cleaned
    assert "2026年1月30日" not in cleaned
    assert "議事次第" not in cleaned
    assert "資料1" not in cleaned
    assert "問合せ先" not in cleaned
    assert "電話：" not in cleaned
    assert "メール：" not in cleaned
    assert "これは日本語の本文です。" in cleaned
    assert "行政サービスの改善を進めます。" in cleaned


def test_fetch_government_corpus_saves_text_files(tmp_path: Path, monkeypatch):
    seed_file = tmp_path / "government_ja.txt"
    seed_file.write_text(
        "\n".join(
            [
                "https://www.digital.go.jp/resources/open_data",
                "https://example.com/disallowed",
            ]
        ),
        encoding="utf-8",
    )

    def fake_get(self, url: str, timeout: int):
        return _DummyResponse(
            "<html><body><main>これは公的オープンデータの説明文です。十分に長い日本語の本文として扱います。</main></body></html>"
        )

    monkeypatch.setattr("requests.Session.get", fake_get)
    output_dir = tmp_path / "government_ja"
    report = fetch_government_corpus(
        seed_file=seed_file,
        output_dir=output_dir,
        min_chars=20,
        timeout=5,
    )

    text_files = list(output_dir.glob("gov_*.txt"))
    assert len(text_files) == 1
    assert report["requested_urls"] == 2
    assert report["saved_docs"] == 1
    assert report["filtered_disallowed_domain"] == 1
    report_payload = json.loads((output_dir / "fetch.report.json").read_text(encoding="utf-8"))
    assert report_payload["saved_docs"] == 1


def test_fetch_government_corpus_cleans_saved_text(tmp_path: Path, monkeypatch):
    seed_file = tmp_path / "government_ja.txt"
    seed_file.write_text("https://www.digital.go.jp/resources/open_data", encoding="utf-8")

    def fake_get(self, url: str, timeout: int):
        return _DummyResponse(
            """
            <html><body><main>
            <h1>政策ページ</h1>
            <p>最終更新日:</p>
            <p>2026年1月30日</p>
            <p>資料1：説明資料（PDF／112KB）</p>
            <p>これは公的オープンデータの説明文です。十分に長い日本語の本文として扱います。</p>
            </main></body></html>
            """
        )

    monkeypatch.setattr("requests.Session.get", fake_get)
    output_dir = tmp_path / "government_ja"
    fetch_government_corpus(seed_file=seed_file, output_dir=output_dir, min_chars=20, timeout=5)
    saved = next(output_dir.glob("gov_*.txt")).read_text(encoding="utf-8")
    assert "最終更新日" not in saved
    assert "PDF／112KB" not in saved
    assert "これは公的オープンデータの説明文です" in saved
