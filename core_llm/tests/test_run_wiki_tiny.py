from __future__ import annotations

import bz2
import json
from pathlib import Path

import pytest

from core_llm.pipeline.wiki_tiny import resolve_run_paths, run_wiki_tiny_pipeline


def _write_dump(path: Path) -> None:
    pages = []
    for idx in range(8):
        pages.append(
            f"""
            <page>
              <title>記事{idx}</title>
              <ns>0</ns>
              <id>{idx + 1}</id>
              <revision><id>{idx + 10}</id><text xml:space="preserve">人工知能は計算機科学の分野であり、日本語の学習用テキストです。これは小規模実験用の本文 {idx} です。</text></revision>
            </page>
            """
        )
    xml = '<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.10/">' + "".join(pages) + "</mediawiki>"
    with bz2.open(path, "wt", encoding="utf-8") as f:
        f.write(xml)


def test_run_wiki_tiny_pipeline_end_to_end(tmp_path: Path):
    dump_path = tmp_path / "jawiki-latest-pages-articles.xml.bz2"
    _write_dump(dump_path)
    work_dir = tmp_path / "run"
    tokenizer_cfg = tmp_path / "tokenizer.yaml"
    tokenizer_cfg.write_text(
        "\n".join(
            [
                "vocab_size: 128",
                "character_coverage: 0.9995",
                "model_type: bpe",
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
    train_cfg = tmp_path / "train.yaml"
    train_cfg.write_text(
        "\n".join(
            [
                "batch_size: 1",
                "seq_len: 16",
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

    summary = run_wiki_tiny_pipeline(
        work_dir=work_dir,
        raw_dir=tmp_path / "raw",
        dump_path=dump_path,
        max_docs=6,
        min_chars=40,
        tokenizer_config=tokenizer_cfg,
        model_config=model_cfg,
        train_config=train_cfg,
    )
    paths = resolve_run_paths(work_dir)
    assert Path(summary["manifest_path"]).exists()
    assert Path(summary["tokenizer_path"]).exists()
    assert (paths["prepared_dir"] / "metadata.json").exists()
    assert (paths["checkpoint_dir"] / "latest.pt").exists()
    assert paths["eval_path"].exists()
    assert paths["summary_path"].exists()


def test_run_wiki_tiny_skip_tokenizer_requires_existing_artifact(tmp_path: Path):
    dump_path = tmp_path / "jawiki-latest-pages-articles.xml.bz2"
    _write_dump(dump_path)
    cfg = tmp_path / "dummy.yaml"
    cfg.write_text(
        "\n".join(
            [
                "vocab_size: 128",
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
    train_cfg = tmp_path / "train.yaml"
    train_cfg.write_text(
        "\n".join(
            [
                "batch_size: 1",
                "seq_len: 16",
                "learning_rate: 0.001",
                "weight_decay: 0.0",
                "grad_accum_steps: 1",
                "warmup_steps: 1",
                "total_steps: 1",
                "eval_every: 1",
                "save_every: 1",
                "seed: 42",
                "device: cpu",
                "amp: false",
                "min_lr_ratio: 0.1",
                "grad_clip: 1.0",
                "early_stopping_patience: 2",
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(FileNotFoundError):
        run_wiki_tiny_pipeline(
            work_dir=tmp_path / "run",
            raw_dir=tmp_path / "raw",
            dump_path=dump_path,
            max_docs=2,
            min_chars=40,
            tokenizer_config=cfg,
            model_config=cfg,
            train_config=train_cfg,
            skip_manifest=True,
            skip_tokenizer=True,
            skip_dataset=True,
            skip_train=True,
            skip_eval=True,
        )


def test_run_wiki_tiny_zero_doc_manifest_fails(tmp_path: Path):
    dump_path = tmp_path / "jawiki-latest-pages-articles.xml.bz2"
    xml = '<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.10/"><page><title>x</title><ns>0</ns><id>1</id><revision><id>2</id><text xml:space="preserve">short</text></revision></page></mediawiki>'
    with bz2.open(dump_path, "wt", encoding="utf-8") as f:
        f.write(xml)
    tokenizer_cfg = tmp_path / "tokenizer.yaml"
    tokenizer_cfg.write_text(
        "\n".join(
            [
                "vocab_size: 128",
                "character_coverage: 0.9995",
                "model_type: bpe",
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
    train_cfg = tmp_path / "train.yaml"
    train_cfg.write_text(
        "\n".join(
            [
                "batch_size: 1",
                "seq_len: 16",
                "learning_rate: 0.001",
                "weight_decay: 0.0",
                "grad_accum_steps: 1",
                "warmup_steps: 1",
                "total_steps: 1",
                "eval_every: 1",
                "save_every: 1",
                "seed: 42",
                "device: cpu",
                "amp: false",
                "min_lr_ratio: 0.1",
                "grad_clip: 1.0",
                "early_stopping_patience: 2",
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit):
        run_wiki_tiny_pipeline(
            work_dir=tmp_path / "run",
            raw_dir=tmp_path / "raw",
            dump_path=dump_path,
            max_docs=2,
            min_chars=120,
            tokenizer_config=tokenizer_cfg,
            model_config=model_cfg,
            train_config=train_cfg,
        )
