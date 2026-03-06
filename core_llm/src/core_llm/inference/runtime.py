from __future__ import annotations

from pathlib import Path

from core_llm.config import ModelConfig
from core_llm.model.transformer import GPT
from core_llm.tokenizer.encode import load_tokenizer
from core_llm.train.checkpoint import load_checkpoint
from core_llm.train.loop import resolve_device


def load_runtime(checkpoint_path: str | Path, tokenizer_path: str | Path | None = None, device: str = "auto"):
    resolved_device = resolve_device(device)
    checkpoint = load_checkpoint(checkpoint_path, resolved_device)
    model_config = ModelConfig(**checkpoint["model_config"])
    model = GPT(model_config).to(resolved_device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    checkpoint_dir = Path(checkpoint_path).parent.parent
    tok_path = Path(tokenizer_path) if tokenizer_path else checkpoint_dir / "tokenizer" / "tokenizer.model"
    tokenizer = load_tokenizer(tok_path)
    return model, tokenizer, resolved_device
