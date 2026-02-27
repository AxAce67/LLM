import os
from dataclasses import dataclass

from runtime.auto_tuner import detect_runtime_profile

@dataclass
class TrainConfig:
    batch_size: int
    seq_len: int
    learning_rate: float
    vocab_size: int
    n_layer: int
    n_head: int
    n_embd: int
    dropout: float
    bias: bool
    weight_decay: float
    grad_accum_steps: int
    warmup_steps: int
    min_lr_ratio: float
    cpu_threads: int


MODEL_PRESETS = {
    "tiny": {"n_layer": 4, "n_head": 4, "n_embd": 256},
    "small": {"n_layer": 8, "n_head": 8, "n_embd": 512},
    "base": {"n_layer": 12, "n_head": 12, "n_embd": 768},
    "medium": {"n_layer": 16, "n_head": 16, "n_embd": 1024},
}


def load_train_config() -> TrainConfig:
    auto_tune = os.environ.get("AUTO_TUNE", "1") == "1"
    profile = detect_runtime_profile() if auto_tune else None
    preset_name = os.environ.get("MODEL_SIZE", (profile["model_size"] if profile else "small")).lower()
    preset = MODEL_PRESETS.get(preset_name, MODEL_PRESETS["small"])
    cpu_default = profile["pytorch_cpu_threads"] if profile else max(1, (os.cpu_count() or 2) - 1)

    return TrainConfig(
        batch_size=int(os.environ.get("TRAIN_BATCH_SIZE", str(profile["train_batch_size"] if profile else 4))),
        seq_len=int(os.environ.get("TRAIN_SEQ_LEN", str(profile["train_seq_len"] if profile else 512))),
        learning_rate=float(os.environ.get("TRAIN_LR", "3e-4")),
        vocab_size=int(os.environ.get("TRAIN_VOCAB_SIZE", "8000")),
        n_layer=int(os.environ.get("TRAIN_N_LAYER", str(preset["n_layer"]))),
        n_head=int(os.environ.get("TRAIN_N_HEAD", str(preset["n_head"]))),
        n_embd=int(os.environ.get("TRAIN_N_EMBD", str(preset["n_embd"]))),
        dropout=float(os.environ.get("TRAIN_DROPOUT", "0.0")),
        bias=os.environ.get("TRAIN_BIAS", "0") == "1",
        weight_decay=float(os.environ.get("TRAIN_WEIGHT_DECAY", "0.1")),
        grad_accum_steps=max(1, int(os.environ.get("TRAIN_GRAD_ACCUM_STEPS", "1"))),
        warmup_steps=max(1, int(os.environ.get("TRAIN_WARMUP_STEPS", "20"))),
        min_lr_ratio=float(os.environ.get("TRAIN_MIN_LR_RATIO", "0.1")),
        cpu_threads=max(1, int(os.environ.get("PYTORCH_CPU_THREADS", str(cpu_default)))),
    )
