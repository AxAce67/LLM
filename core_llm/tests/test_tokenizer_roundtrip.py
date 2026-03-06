from pathlib import Path

from core_llm.config import TokenizerConfig
from core_llm.tokenizer.encode import decode_ids, load_tokenizer
from core_llm.tokenizer.trainer import resolve_tokenizer_threads, train_tokenizer


def test_tokenizer_roundtrip(sample_manifest, tmp_path: Path):
    model_path = train_tokenizer(sample_manifest, tmp_path / "tokenizer", TokenizerConfig(vocab_size=64))
    tokenizer = load_tokenizer(model_path)
    text = "人工知能は面白い分野です。"
    ids = tokenizer.encode_as_ids(text)
    decoded = decode_ids(tokenizer, ids)
    assert ids
    assert "人工" in decoded


def test_tokenizer_threads_default_to_cpu_count():
    assert resolve_tokenizer_threads(TokenizerConfig(num_threads=0)) >= 1


def test_tokenizer_threads_respect_explicit_value():
    assert resolve_tokenizer_threads(TokenizerConfig(num_threads=8)) == 8
