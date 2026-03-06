from __future__ import annotations

from pathlib import Path

from core_llm.data.government_seed_discovery import should_include_seed_url, write_seed_urls


def test_should_include_seed_url_filters_expected_paths():
    assert should_include_seed_url("https://www.digital.go.jp/policies/base_registry")
    assert should_include_seed_url("https://www.digital.go.jp/news/382c3937-f43c-4452-ae27-2ea7bb66ec75")
    assert should_include_seed_url("https://www.digital.go.jp/about/organization")
    assert not should_include_seed_url("https://www.digital.go.jp/en/policies/base_registry")
    assert not should_include_seed_url("https://www.digital.go.jp/assets/example.pdf")
    assert not should_include_seed_url("https://example.com/policies/base_registry")


def test_write_seed_urls_writes_plain_list(tmp_path: Path):
    target = tmp_path / "government_ja.txt"
    write_seed_urls(target, ["https://www.digital.go.jp/policies/base_registry"])
    assert target.read_text(encoding="utf-8").strip() == "https://www.digital.go.jp/policies/base_registry"
