import json

from core_llm.config import ModelConfig, TokenizerConfig
from core_llm.data.dataset import BinaryTokenDataset
from core_llm.tokenizer.trainer import train_tokenizer


def test_binary_dataset_returns_expected_shapes(sample_manifest, tmp_path):
    model_path = train_tokenizer(
        sample_manifest,
        tmp_path / "tokenizer",
        TokenizerConfig(vocab_size=64, input_sentence_size=1000),
    )
    tokenizer_dir = model_path.parent
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
        "--tokenizer", str(tokenizer_dir / "tokenizer.model"),
        "--output-dir", str(tmp_path / "prepared"),
        "--min-chars", "10",
    ]
    try:
        prepare_dataset_main()
    finally:
        sys.argv = old_argv

    dataset = BinaryTokenDataset(tmp_path / "prepared" / "train.bin", batch_size=1, seq_len=16)
    x, y = dataset.next_batch("cpu")
    assert tuple(x.shape) == (1, 16)
    assert tuple(y.shape) == (1, 16)
    metadata = json.loads((tmp_path / "prepared" / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["train_tokens"] > 0
