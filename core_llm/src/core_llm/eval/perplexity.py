from __future__ import annotations

from pathlib import Path

from core_llm.config import ModelConfig
from core_llm.data.dataset import BinaryTokenDataset
from core_llm.model.transformer import GPT
from core_llm.train.checkpoint import load_checkpoint
from core_llm.train.loop import evaluate_loss, resolve_device
from core_llm.train.metrics import perplexity_from_loss


def evaluate_checkpoint_perplexity(
    checkpoint_path: str | Path,
    data_dir: str | Path,
    batch_size: int = 2,
    device: str = "auto",
) -> dict:
    resolved_device = resolve_device(device)
    checkpoint = load_checkpoint(checkpoint_path, resolved_device)
    model_config = ModelConfig(**checkpoint["model_config"])
    model = GPT(model_config).to(resolved_device)
    model.load_state_dict(checkpoint["model_state_dict"])
    val_ds = BinaryTokenDataset(Path(data_dir) / "val.bin", batch_size, model_config.block_size)
    val_loss = evaluate_loss(model, val_ds, resolved_device)
    return {
        "checkpoint": str(checkpoint_path),
        "val_loss": val_loss,
        "val_perplexity": perplexity_from_loss(val_loss),
        "device": resolved_device,
    }
