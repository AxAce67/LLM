from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_SPECIAL_TOKENS = {
    "pad_id": 0,
    "unk_id": 1,
    "bos_id": 2,
    "eos_id": 3,
}


@dataclass
class TokenizerConfig:
    vocab_size: int = 16000
    character_coverage: float = 0.9995
    model_type: str = "bpe"
    special_tokens: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_SPECIAL_TOKENS))


@dataclass
class ModelConfig:
    vocab_size: int = 16000
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.1
    bias: bool = False


@dataclass
class TrainConfig:
    batch_size: int = 2
    seq_len: int = 256
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    grad_accum_steps: int = 8
    warmup_steps: int = 100
    total_steps: int = 5000
    eval_every: int = 250
    save_every: int = 250
    seed: int = 42
    device: str = "auto"
    amp: bool = False
    min_lr_ratio: float = 0.1
    grad_clip: float = 1.0
    early_stopping_patience: int = 8


def _load_yaml(path: str | Path) -> dict[str, Any]:
    data: dict[str, Any] = {}
    current_parent: str | None = None
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip()
            if not line or line.lstrip().startswith("#"):
                continue
            if not line.startswith(" ") and line.endswith(":"):
                current_parent = line[:-1].strip()
                data[current_parent] = {}
                continue
            if ":" not in line:
                raise ValueError(f"Invalid config line: {line}")
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            parsed: Any
            if value.lower() in {"true", "false"}:
                parsed = value.lower() == "true"
            else:
                try:
                    parsed = int(value)
                except ValueError:
                    try:
                        parsed = float(value)
                    except ValueError:
                        parsed = value
            if raw_line.startswith("  "):
                if current_parent is None or not isinstance(data.get(current_parent), dict):
                    raise ValueError(f"Invalid nested config line: {line}")
                data[current_parent][key] = parsed
            else:
                current_parent = None
                data[key] = parsed
    return data


def load_tokenizer_config(path: str | Path) -> TokenizerConfig:
    return TokenizerConfig(**_load_yaml(path))


def load_model_config(path: str | Path) -> ModelConfig:
    return ModelConfig(**_load_yaml(path))


def load_train_config(path: str | Path) -> TrainConfig:
    return TrainConfig(**_load_yaml(path))


def dump_dataclass_jsonable(obj: Any) -> dict[str, Any]:
    return asdict(obj)
