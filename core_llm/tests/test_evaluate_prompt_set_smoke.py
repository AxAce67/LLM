import json
import sys
from pathlib import Path

from core_llm.config import ModelConfig, TokenizerConfig, TrainConfig
from core_llm.data.binary_writer import BinarySplitWriter, write_metadata
from core_llm.data.cleaning import normalize_text
from core_llm.data.manifest_schema import iter_manifest
from core_llm.data.split import assign_split
from core_llm.tokenizer.encode import load_tokenizer
from core_llm.tokenizer.special_tokens import DEFAULT_SPECIAL_TOKENS
from core_llm.tokenizer.trainer import train_tokenizer
from core_llm.train.loop import train_model


def _prepare_base_checkpoint(tmp_path: Path) -> tuple[Path, Path]:
    manifest_path = tmp_path / "base_manifest.jsonl"
    rows = [
        {
            "id": "doc-1",
            "text": "人工知能は人間の知的な作業を計算機で実現する技術です。" * 50,
            "lang": "ja",
            "source": "test",
            "license": "cc-by-4.0",
            "split_hint": "train",
        },
        {
            "id": "doc-2",
            "text": "機械学習はデータから規則性を学ぶ人工知能の一分野です。" * 50,
            "lang": "ja",
            "source": "test",
            "license": "cc-by-4.0",
            "split_hint": "val",
        },
    ]
    with manifest_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir()
    tokenizer_path = train_tokenizer(
        manifest_path=manifest_path,
        output_dir=tokenizer_dir,
        config=TokenizerConfig(
            vocab_size=128,
            character_coverage=0.9995,
            model_type="bpe",
            special_tokens=DEFAULT_SPECIAL_TOKENS,
            input_sentence_size=1000,
        ),
    )

    prepared_dir = tmp_path / "prepared"
    model_config = ModelConfig(
        vocab_size=128,
        block_size=64,
        n_layer=2,
        n_head=2,
        n_embd=64,
        dropout=0.0,
        bias=False,
    )
    prepared_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer(tokenizer_path)
    train_writer = BinarySplitWriter(prepared_dir / "train.bin")
    val_writer = BinarySplitWriter(prepared_dir / "val.bin")
    kept = 0
    for row in iter_manifest(manifest_path):
        tokens = tokenizer.encode_as_ids(normalize_text(row.text))
        tokens.append(tokenizer.eos_id())
        split = row.split_hint if row.split_hint in {"train", "val"} else assign_split(row.id)
        if split == "val":
            val_writer.write_tokens(tokens)
        else:
            train_writer.write_tokens(tokens)
        kept += 1
    train_writer.close()
    val_writer.close()
    write_metadata(
        prepared_dir / "metadata.json",
        {
            "kept_docs": kept,
            "filtered_docs": 0,
            "duplicate_docs": 0,
            "train_tokens": train_writer.token_count,
            "val_tokens": val_writer.token_count,
            "vocab_size": tokenizer.get_piece_size(),
        },
    )

    checkpoint_dir = tmp_path / "checkpoints"
    train_model(
        data_dir=prepared_dir,
        checkpoint_dir=checkpoint_dir,
        model_config=model_config,
        train_config=TrainConfig(
            batch_size=1,
            seq_len=64,
            learning_rate=1e-3,
            weight_decay=0.0,
            grad_accum_steps=1,
            warmup_steps=1,
            total_steps=3,
            eval_every=3,
            save_every=3,
            seed=42,
            device="cpu",
            amp=False,
        ),
    )
    return checkpoint_dir / "best.pt", tokenizer_path


def test_evaluate_prompt_set_smoke(tmp_path: Path):
    base_checkpoint, tokenizer_path = _prepare_base_checkpoint(tmp_path)
    questions = tmp_path / "questions.jsonl"
    with open(questions, "w", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "id": "eval-001",
                    "category": "definition",
                    "instruction": "人工知能とは何ですか？",
                    "input": "",
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        f.write(
            json.dumps(
                {
                    "id": "eval-002",
                    "category": "comparison",
                    "instruction": "機械学習とは何ですか？",
                    "input": "",
                },
                ensure_ascii=False,
            )
            + "\n"
        )

    output_path = tmp_path / "responses.jsonl"

    from core_llm.scripts.evaluate_prompt_set import main as eval_prompt_set_main

    old_argv = sys.argv
    sys.argv = [
        "evaluate_prompt_set",
        "--checkpoint",
        str(base_checkpoint),
        "--tokenizer",
        str(tokenizer_path),
        "--questions",
        str(questions),
        "--output",
        str(output_path),
        "--max-new-tokens",
        "8",
        "--device",
        "cpu",
    ]
    try:
        eval_prompt_set_main()
    finally:
        sys.argv = old_argv

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 2
    assert rows[0]["id"] == "eval-001"
    assert "response" in rows[0]
    assert "scores" in rows[0]
    assert "qa_ok" in rows[0]["scores"]

    summary_path = output_path.with_suffix(".summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["counts"]["total"] == 2
    assert "qa_ok_rate" in summary["counts"]
