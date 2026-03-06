from __future__ import annotations

import os
from pathlib import Path

from core_llm.env import load_env_file


def test_load_env_file_sets_missing_variables(tmp_path: Path, monkeypatch):
    env_file = tmp_path / ".env.local"
    env_file.write_text(
        "\n".join(
            [
                "# comment",
                "DISCORD_WEBHOOK_URL='https://example.invalid/webhook'",
                'DISCORD_MENTION="<@123>"',
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
    monkeypatch.delenv("DISCORD_MENTION", raising=False)

    load_env_file(env_file)

    assert os.environ["DISCORD_WEBHOOK_URL"] == "https://example.invalid/webhook"
    assert os.environ["DISCORD_MENTION"] == "<@123>"


def test_load_env_file_does_not_override_existing(monkeypatch, tmp_path: Path):
    env_file = tmp_path / ".env.local"
    env_file.write_text("DISCORD_MENTION='<@123>'", encoding="utf-8")
    monkeypatch.setenv("DISCORD_MENTION", "<@999>")

    load_env_file(env_file)

    assert os.environ["DISCORD_MENTION"] == "<@999>"

