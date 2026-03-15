import json
import sys
from pathlib import Path

from core_llm.config import ModelConfig, TokenizerConfig, TrainConfig
from core_llm.seed import set_seed
from core_llm.tokenizer.trainer import train_tokenizer
from core_llm.train.loop import train_model


def _prepare_base_checkpoint(tmp_path: Path) -> tuple[Path, Path]:
    manifest = tmp_path / "pretrain_manifest.jsonl"
    rows = [
        {
            "id": f"doc-{idx:03d}",
            "text": (
                "人工知能は計算機が知的な処理を行うための技術です。"
                "機械学習や自然言語処理の基盤になります。"
                f" これはサンプル文書 {idx} です。"
            ),
            "lang": "ja",
            "source": "fixture",
            "license": "cc-by-sa-4.0",
            "split_hint": "train" if idx < 6 else "val",
        }
        for idx in range(8)
    ]
    with open(manifest, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tokenizer_path = train_tokenizer(
        manifest,
        tmp_path / "tokenizer",
        TokenizerConfig(vocab_size=64, input_sentence_size=1000),
    )
    cfg_path = tmp_path / "model.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "vocab_size: 64",
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
    from core_llm.scripts.prepare_dataset import main as prepare_dataset_main

    old_argv = sys.argv
    sys.argv = [
        "prepare_dataset",
        "--config", str(cfg_path),
        "--manifest", str(manifest),
        "--tokenizer", str(tokenizer_path),
        "--output-dir", str(tmp_path / "prepared"),
        "--min-chars", "10",
    ]
    try:
        prepare_dataset_main()
    finally:
        sys.argv = old_argv
    set_seed(42)
    model_config = ModelConfig(vocab_size=64, block_size=16, n_layer=2, n_head=2, n_embd=32, dropout=0.1, bias=False)
    train_config = TrainConfig(
        batch_size=1,
        seq_len=16,
        learning_rate=1e-3,
        weight_decay=0.0,
        grad_accum_steps=1,
        warmup_steps=1,
        total_steps=1,
        eval_every=1,
        save_every=1,
        seed=42,
        device="cpu",
        amp=False,
        early_stopping_patience=10,
    )
    train_model(
        data_dir=tmp_path / "prepared",
        checkpoint_dir=tmp_path / "checkpoints",
        model_config=model_config,
        train_config=train_config,
    )
    return tmp_path / "checkpoints" / "latest.pt", tokenizer_path


def test_train_sft_smoke(tmp_path: Path):
    base_checkpoint, tokenizer_path = _prepare_base_checkpoint(tmp_path)
    sft_manifest = tmp_path / "sft.jsonl"
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
            "instruction": "自然言語処理とは何ですか？",
            "input": "",
            "output": "自然言語処理は、人間の言葉を計算機で扱う技術です。",
        },
    ]
    with open(sft_manifest, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    work_dir = tmp_path / "sft_run"
    train_cfg = tmp_path / "train_sft.yaml"
    train_cfg.write_text(
        "\n".join(
            [
                "batch_size: 1",
                "seq_len: 16",
                "learning_rate: 1e-4",
                "weight_decay: 0.0",
                "grad_accum_steps: 1",
                "warmup_steps: 1",
                "total_steps: 2",
                "eval_every: 1",
                "save_every: 1",
                "seed: 42",
                "device: cpu",
                "amp: false",
                "cpu_threads: 1",
                "interop_threads: 1",
                "min_lr_ratio: 0.1",
                "grad_clip: 1.0",
                "early_stopping_patience: 4",
            ]
        ),
        encoding="utf-8",
    )
    from core_llm.scripts.train_sft import main as train_sft_main

    old_argv = sys.argv
    sys.argv = [
        "train_sft",
        "--base-checkpoint", str(base_checkpoint),
        "--tokenizer", str(tokenizer_path),
        "--manifest", str(sft_manifest),
        "--train-config", str(train_cfg),
        "--work-dir", str(work_dir),
    ]
    try:
        train_sft_main()
    finally:
        sys.argv = old_argv
    summary_path = work_dir / "run_summary.json"
    if not summary_path.exists():
        candidates = list(tmp_path.rglob("run_summary.json"))
        summary_path = next((p for p in candidates if json.loads(p.read_text()).get("run_type") == "sft"), None)
        assert summary_path is not None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    final_dir = Path(summary["work_dir"])
    assert (final_dir / "checkpoints" / "latest.pt").exists()
    assert (final_dir / "checkpoints" / "train_metrics.jsonl").exists()
