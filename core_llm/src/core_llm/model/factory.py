from __future__ import annotations

import torch.nn as nn

from core_llm.config import ModelConfig
from core_llm.model.llama import Llama
from core_llm.model.transformer import GPT


def build_model(config: ModelConfig) -> nn.Module:
    model_type = (config.model_type or "gpt").lower()
    if model_type in {"gpt", "gpt2", "nano", "nanogpt"}:
        return GPT(config)
    if model_type in {"llama", "llama2", "llama3"}:
        return Llama(config)
    raise ValueError(f"Unknown model_type: {config.model_type}")
