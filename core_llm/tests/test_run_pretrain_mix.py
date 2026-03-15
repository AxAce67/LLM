from __future__ import annotations

import json
from pathlib import Path

import pytest

from core_llm.data.manifest_schema import ManifestRecord, write_manifest
from core_llm.pipeline.pretrain_mix import resolve_mix_run_paths, run_pretrain_mix_pipeline


def _write_configs(tmp_path: Path) -> tuple[Path, Path, Path]:
    tokenizer_cfg = tmp_path / "tokenizer.yaml"
    tokenizer_cfg.write_text(
        "\n".join(
            [
                "vocab_size: 128",
                "character_coverage: 0.9995",
                "model_type: bpe",
                "input_sentence_size: 1000",
                "shuffle_input_sentence: true",
                "special_tokens:",
                "  pad_id: 0",
                "  unk_id: 1",
                "  bos_id: 2",
                "  eos_id: 3",
            ]
        ),
        encoding="utf-8",
    )
    model_cfg = tmp_path / "model.yaml"
    model_cfg.write_text(
        "\n".join(
            [
                "vocab_size: 128",
                "block_size: 8",
                "n_layer: 2",
                "n_head: 2",
                "n_embd: 32",
                "dropout: 0.1",
                "bias: false",
            ]
        ),
        encoding="utf-8",
    )
    train_cfg = tmp_path / "train.yaml"
    train_cfg.write_text(
        "\n".join(
            [
                "batch_size: 1",
                "seq_len: 8",
                "learning_rate: 0.001",
                "weight_decay: 0.0",
                "grad_accum_steps: 1",
                "warmup_steps: 1",
                "total_steps: 2",
                "eval_every: 1",
                "save_every: 1",
                "seed: 42",
                "device: cpu",
                "amp: false",
                "min_lr_ratio: 0.1",
                "grad_clip: 1.0",
                "early_stopping_patience: 10",
            ]
        ),
        encoding="utf-8",
    )
    return tokenizer_cfg, model_cfg, train_cfg


def test_run_pretrain_mix_pipeline_end_to_end(tmp_path: Path):
    first = tmp_path / "wiki.jsonl"
    second = tmp_path / "notes.jsonl"
    write_manifest(
        first,
        [
            ManifestRecord(
                id="wiki-1",
                text="人工知能は計算機科学の分野であり、十分に長い日本語の本文です。学習用サンプルです。",
                lang="ja",
                source="wikipedia_ja",
                license="cc-by-sa-4.0",
            )
        ],
    )
    write_manifest(
        second,
        [
            ManifestRecord(
                id="notes-1",
                text="これは手元ノート由来の日本語テキストです。こちらも十分に長い本文として扱います。",
                lang="ja",
                source="local_notes_ja",
                license="permissive-user-provided",
            )
        ],
    )
    tokenizer_cfg, model_cfg, train_cfg = _write_configs(tmp_path)

    work_dir = tmp_path / "run"
    summary = run_pretrain_mix_pipeline(
        work_dir=work_dir,
        manifest_inputs=[first, second],
        tokenizer_config=tokenizer_cfg,
        model_config=model_cfg,
        train_config=train_cfg,
        min_chars=20,
    )

    paths = resolve_mix_run_paths(work_dir)
    assert paths["manifest"].exists()
    assert paths["manifest_report"].exists()
    assert (paths["prepared_dir"] / "metadata.json").exists()
    assert (paths["checkpoint_dir"] / "latest.pt").exists()
    assert paths["eval_path"].exists()
    assert summary["source_counts"] == {"local_notes_ja": 1, "wikipedia_ja": 1}
    report = json.loads(paths["manifest_report"].read_text(encoding="utf-8"))
    assert report["kept_docs"] == 2


def test_run_pretrain_mix_skip_merge_requires_existing_artifacts(tmp_path: Path):
    tokenizer_cfg, model_cfg, train_cfg = _write_configs(tmp_path)
    with pytest.raises(FileNotFoundError):
        run_pretrain_mix_pipeline(
            work_dir=tmp_path / "run",
            manifest_inputs=[],
            tokenizer_config=tokenizer_cfg,
            model_config=model_cfg,
            train_config=train_cfg,
            skip_merge=True,
            skip_tokenizer=True,
            skip_dataset=True,
            skip_train=True,
            skip_eval=True,
        )
