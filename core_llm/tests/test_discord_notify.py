from __future__ import annotations

from core_llm.notify.discord import (
    build_command_started_message,
    build_command_failure_message,
    build_command_success_message,
    build_failure_message,
    build_run_message,
    build_run_started_message,
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


def test_build_run_started_message_includes_key_fields():
    message = build_run_started_message(
        work_dir="data/runs/example",
        run_type="wiki_tiny_sample",
        mention="@here",
        payload={"model_config": "configs/model.yaml", "max_docs": 5000},
    )
    assert "@here" in message
    assert "Training started" in message
    assert "run: data/runs/example" in message
    assert "type: wiki_tiny_sample" in message
    assert "model_config: configs/model.yaml" in message
    assert "max_docs: 5000" in message


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


def test_build_command_success_message_includes_payload_fields():
    message = build_command_success_message(
        command_name="train",
        payload={"step": 100, "best_val_perplexity": 12.34},
        mention="@here",
    )
    assert "@here" in message
    assert "Command completed" in message
    assert "command: train" in message
    assert "step: 100" in message
    assert "best_val_perplexity: 12.34" in message


def test_build_command_started_message_includes_payload_fields():
    message = build_command_started_message(
        command_name="prepare_dataset",
        payload={"manifest": "data/manifests/train.jsonl", "output_dir": "data/prepared"},
        mention="@here",
    )
    assert "@here" in message
    assert "Command started" in message
    assert "command: prepare_dataset" in message
    assert "manifest: data/manifests/train.jsonl" in message
    assert "output_dir: data/prepared" in message


def test_build_command_failure_message_includes_error():
    message = build_command_failure_message(
        command_name="prepare_dataset",
        error="Tokenizer vocab size does not match",
        mention="@here",
    )
    assert "@here" in message
    assert "Command failed" in message
    assert "command: prepare_dataset" in message
    assert "error: Tokenizer vocab size does not match" in message
