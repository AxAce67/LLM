from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class BinaryTokenDataset:
    def __init__(self, path: str | Path, batch_size: int, seq_len: int):
        self.path = Path(path)
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.position = 0
        if not self.path.exists() or self.path.stat().st_size == 0:
            self.data = np.array([], dtype=np.uint16)
        else:
            self.data = np.memmap(self.path, dtype=np.uint16, mode="r")
        self.total_batches = len(self.data) // max(1, (batch_size * seq_len))

    def __len__(self) -> int:
        return self.total_batches

    def next_batch(self, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        needed = self.batch_size * self.seq_len + 1
        if len(self.data) < needed:
            raise ValueError("Dataset does not contain enough tokens for a single batch")
        if self.position + needed > len(self.data):
            self.position = 0
        buf = np.asarray(self.data[self.position:self.position + needed], dtype=np.int64)
        self.position += self.batch_size * self.seq_len
        x = torch.from_numpy(buf[:-1].reshape(self.batch_size, self.seq_len)).to(device)
        y = torch.from_numpy(buf[1:].reshape(self.batch_size, self.seq_len)).to(device)
        return x, y
