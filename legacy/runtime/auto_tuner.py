import os
from typing import Dict

import psutil
import torch


def _detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def detect_runtime_profile() -> Dict[str, float]:
    cpu_cores = os.cpu_count() or 2
    vm = psutil.virtual_memory()
    total_ram_gb = vm.total / (1024 ** 3)
    available_ram_gb = vm.available / (1024 ** 3)
    device = _detect_device()

    if device == "cuda":
        model_size = "base"
        seq_len = 512
        batch_size = 8
        max_generate_tokens = 384
    elif total_ram_gb >= 24:
        model_size = "small"
        seq_len = 512
        batch_size = 6
        max_generate_tokens = 256
    elif total_ram_gb >= 12:
        model_size = "small"
        seq_len = 384
        batch_size = 4
        max_generate_tokens = 192
    elif total_ram_gb >= 8:
        model_size = "tiny"
        seq_len = 256
        batch_size = 3
        max_generate_tokens = 128
    else:
        model_size = "tiny"
        seq_len = 128
        batch_size = 2
        max_generate_tokens = 96

    # available RAMが厳しい時は保守的に落とす（リアルタイム適応）
    if available_ram_gb < 1.5:
        batch_size = max(1, batch_size - 2)
        seq_len = max(96, seq_len // 2)
        max_generate_tokens = max(64, max_generate_tokens // 2)
    elif available_ram_gb < 3:
        batch_size = max(1, batch_size - 1)
        seq_len = max(128, int(seq_len * 0.75))

    max_crawler_workers = max(1, min(16, cpu_cores * 2))
    pytorch_cpu_threads = max(1, cpu_cores - 1)

    return {
        "device": device,
        "cpu_cores": cpu_cores,
        "total_ram_gb": round(total_ram_gb, 2),
        "available_ram_gb": round(available_ram_gb, 2),
        "model_size": model_size,
        "train_seq_len": int(seq_len),
        "train_batch_size": int(batch_size),
        "max_generate_tokens": int(max_generate_tokens),
        "max_crawler_workers": int(max_crawler_workers),
        "pytorch_cpu_threads": int(pytorch_cpu_threads),
    }
