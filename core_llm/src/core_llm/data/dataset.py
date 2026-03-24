from __future__ import annotations

import threading
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

        # Prefetch state: load next batch on CPU while GPU computes current
        self._prefetch: tuple[torch.Tensor, torch.Tensor] | None = None
        self._prefetch_thread: threading.Thread | None = None
        self._prefetch_error: BaseException | None = None
        if len(self.data) > 0:
            self._start_prefetch()

    def __len__(self) -> int:
        return self.total_batches

    def _load_batch_cpu(self) -> tuple[torch.Tensor, torch.Tensor]:
        needed = self.batch_size * self.seq_len + 1
        if self.position + needed > len(self.data):
            self.position = 0
        buf = np.asarray(self.data[self.position : self.position + needed], dtype=np.int64)
        self.position += self.batch_size * self.seq_len
        x = torch.from_numpy(buf[:-1].reshape(self.batch_size, self.seq_len))
        y = torch.from_numpy(buf[1:].reshape(self.batch_size, self.seq_len))
        return x, y

    def _start_prefetch(self) -> None:
        def _worker() -> None:
            try:
                self._prefetch = self._load_batch_cpu()
            except Exception as e:
                self._prefetch_error = e

        self._prefetch_thread = threading.Thread(target=_worker, daemon=True)
        self._prefetch_thread.start()

    def next_batch(self, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        needed = self.batch_size * self.seq_len + 1
        if len(self.data) < needed:
            raise ValueError("Dataset does not contain enough tokens for a single batch")

        if self._prefetch_thread is not None:
            self._prefetch_thread.join()
            if self._prefetch_error is not None:
                raise self._prefetch_error
            x_cpu, y_cpu = self._prefetch  # type: ignore[misc]
            self._prefetch = None
        else:
            x_cpu, y_cpu = self._load_batch_cpu()

        # Start loading next batch in background while GPU processes current
        self._start_prefetch()

        return x_cpu.to(device), y_cpu.to(device)
