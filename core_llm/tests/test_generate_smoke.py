from pathlib import Path

from core_llm.config import ModelConfig, TokenizerConfig, TrainConfig
from core_llm.inference.cli import generate_text
from core_llm.inference.runtime import load_runtime
from core_llm.seed import set_seed
from core_llm.tokenizer.trainer import train_tokenizer
from core_llm.train.loop import train_model


def _prepare_training_artifacts(sample_manifest, tmp_path: Path):
    model_path = train_tokenizer(sample_manifest, tmp_path / "data" / "tokenizer", TokenizerConfig(vocab_size=64))
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
    import sys

    old_argv = sys.argv
    sys.argv = [
        "prepare_dataset",
        "--config", str(cfg_path),
        "--manifest", str(sample_manifest),
        "--tokenizer", str(model_path),
        "--output-dir", str(tmp_path / "data" / "prepared"),
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
        data_dir=tmp_path / "data" / "prepared",
        checkpoint_dir=tmp_path / "data" / "checkpoints",
        model_config=model_config,
        train_config=train_config,
    )
    return tmp_path / "data" / "checkpoints" / "latest.pt"


def test_generate_smoke(sample_manifest, tmp_path: Path):
    checkpoint = _prepare_training_artifacts(sample_manifest, tmp_path)
    model, tokenizer, device = load_runtime(checkpoint, device="cpu")
    text = generate_text(model, tokenizer, "人工知能", max_new_tokens=4, device=device)
    assert isinstance(text, str)
