from __future__ import annotations

import torch

from core_llm.config import TrainConfig


def create_optimizer(model: torch.nn.Module, config: TrainConfig) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
