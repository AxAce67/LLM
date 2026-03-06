from __future__ import annotations

from core_llm.notify.discord import (
    build_failure_message,
    build_run_message,
    resolve_discord_settings,
)


def test_resolve_discord_settings_prefers_explicit(monkeypatch):
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://example.invalid/env")
    monkeypatch.setenv("DISCORD_MENTION", "<@123>")
    webhook_url, mention = resolve_discord_settings("https://example.invalid/explicit", "@here")
    assert webhook_url == "https://example.invalid/explicit"
    assert mention == "@here"


def test_resolve_discord_settings_falls_back_to_env(monkeypatch):
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://example.invalid/env")
    monkeypatch.setenv("DISCORD_MENTION", "<@123>")
    webhook_url, mention = resolve_discord_settings()
    assert webhook_url == "https://example.invalid/env"
    assert mention == "<@123>"


def test_build_run_message_includes_key_fields():
    summary = {
        "work_dir": "data/runs/example",
        "run_type": "wiki_tiny_sample",
        "best_val_perplexity": 12.34,
        "latest_eval_perplexity": 13.37,
        "train_tokens": 12345,
        "steps": ["train", "eval"],
    }
    message = build_run_message(summary, mention="@here")
    assert "@here" in message
    assert "Training completed" in message
    assert "data/runs/example" in message
    assert "12.34" in message
    assert "train, eval" in message


def test_build_failure_message_includes_error():
    message = build_failure_message(
        work_dir="data/runs/example",
        run_type="pretrain_mix_sample",
        error="boom",
        mention="@here",
    )
    assert "@here" in message
    assert "Training failed" in message
    assert "pretrain_mix_sample" in message
    assert "boom" in message
