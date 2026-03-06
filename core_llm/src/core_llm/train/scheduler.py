from __future__ import annotations

import math

from core_llm.config import TrainConfig


def lr_for_step(step: int, config: TrainConfig) -> float:
    if step < config.warmup_steps:
        return config.learning_rate * (step + 1) / max(1, config.warmup_steps)
    progress = (step - config.warmup_steps) / max(1, config.total_steps - config.warmup_steps)
    progress = max(0.0, min(1.0, progress))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.learning_rate * (config.min_lr_ratio + (1.0 - config.min_lr_ratio) * cosine)
