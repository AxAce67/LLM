import json
from pathlib import Path

from core_llm.config import TokenizerConfig
from core_llm.data.sft_dataset import SFTDataset, format_sft_prompt
from core_llm.tokenizer.trainer import train_tokenizer


def _write_sft_manifest(path: Path) -> Path:
    rows = [
        {
            "id": "sft-001",
            "instruction": "人工知能とは何ですか？",
            "input": "",
            "output": "人工知能とは、人間の知的作業を機械で実現する技術分野です。",
        },
        {
            "id": "sft-002",
            "instruction": "機械学習を説明してください。",
            "input": "",
            "output": "機械学習は、データから規則性を学ぶ手法です。",
        },
        {
            "id": "sft-003",
            "instruction": "自然言語処理とは？",
            "input": "",
            "output": "自然言語処理は、人間の言葉を計算機で扱う技術です。",
        },
    ]
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def _write_tokenizer_manifest(path: Path) -> Path:
    rows = [
        {
            "id": "tok-001",
            "text": "人工知能とは、人間の知的作業を機械で実現する技術分野です。",
            "lang": "ja",
            "source": "test",
            "license": "test",
        },
        {
            "id": "tok-002",
            "text": "機械学習は、データから規則性を学ぶ手法です。",
            "lang": "ja",
            "source": "test",
            "license": "test",
        },
        {
            "id": "tok-003",
            "text": "自然言語処理は、人間の言葉を計算機で扱う技術です。",
            "lang": "ja",
            "source": "test",
            "license": "test",
        },
    ]
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def test_sft_dataset_masks_prompt_tokens(tmp_path: Path):
    manifest = _write_sft_manifest(tmp_path / "sft.jsonl")
    tokenizer_manifest = _write_tokenizer_manifest(tmp_path / "tokenizer_manifest.jsonl")
    tokenizer_path = train_tokenizer(
        tokenizer_manifest,
        tmp_path / "tokenizer",
        TokenizerConfig(vocab_size=64, input_sentence_size=1000),
    )
    dataset = SFTDataset(
        manifest_path=manifest,
        tokenizer_path=tokenizer_path,
        batch_size=1,
        seq_len=64,
        split="train",
        val_fraction=0.34,
        seed=42,
    )
    x, y = dataset.next_batch("cpu")
    assert x.shape == y.shape == (1, 64)
    assert (y == -1).any()
    assert (y != -1).any()


def test_format_sft_prompt():
    prompt = format_sft_prompt("質問", "補足")
    assert "### Instruction" in prompt
    assert "### Input" in prompt
    assert "### Response" in prompt
