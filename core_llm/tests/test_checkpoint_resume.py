from pathlib import Path

from core_llm.config import ModelConfig, TokenizerConfig, TrainConfig
from core_llm.seed import set_seed
from core_llm.tokenizer.trainer import train_tokenizer
from core_llm.train.loop import train_model


def _prepare_dataset(sample_manifest, tmp_path: Path, vocab_size: int = 64, block_size: int = 16):
    model_path = train_tokenizer(sample_manifest, tmp_path / "tokenizer", TokenizerConfig(vocab_size=vocab_size))
    cfg_path = tmp_path / "model.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                f"vocab_size: {vocab_size}",
                f"block_size: {block_size}",
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
    import sys

    old_argv = sys.argv
    sys.argv = [
        "prepare_dataset",
        "--config", str(cfg_path),
        "--manifest", str(sample_manifest),
        "--tokenizer", str(model_path),
        "--output-dir", str(tmp_path / "prepared"),
        "--min-chars", "10",
    ]
    try:
        prepare_dataset_main()
    finally:
        sys.argv = old_argv


def test_checkpoint_resume(sample_manifest, tmp_path: Path):
    _prepare_dataset(sample_manifest, tmp_path)
    set_seed(42)
    model_config = ModelConfig(vocab_size=64, block_size=16, n_layer=2, n_head=2, n_embd=32, dropout=0.1, bias=False)
    train_config = TrainConfig(
        batch_size=1,
        seq_len=16,
        learning_rate=1e-3,
        weight_decay=0.0,
        grad_accum_steps=1,
        warmup_steps=1,
        total_steps=2,
        eval_every=1,
        save_every=1,
        seed=42,
        device="cpu",
        amp=False,
        early_stopping_patience=10,
    )
    first = train_model(
        data_dir=tmp_path / "prepared",
        checkpoint_dir=tmp_path / "checkpoints",
        model_config=model_config,
        train_config=train_config,
    )
    train_config.total_steps = 3
    second = train_model(
        data_dir=tmp_path / "prepared",
        checkpoint_dir=tmp_path / "checkpoints",
        model_config=model_config,
        train_config=train_config,
    )
    assert first["step"] == 2
    assert second["step"] == 3
