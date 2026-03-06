from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from core_llm.config import ModelConfig, TrainConfig, dump_dataclass_jsonable


def checkpoint_payload(
    *,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    model_config: ModelConfig,
    train_config: TrainConfig,
    latest_train_loss: float,
    best_val_perplexity: float,
) -> dict[str, Any]:
    return {
        "step": step,
        "model_config": dump_dataclass_jsonable(model_config),
        "train_config": dump_dataclass_jsonable(train_config),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": {},
        "best_val_perplexity": best_val_perplexity,
        "latest_train_loss": latest_train_loss,
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.random.get_rng_state(),
        },
    }


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, target)


def load_checkpoint(path: str | Path, device: str) -> dict[str, Any]:
    return torch.load(path, map_location=device, weights_only=False)
