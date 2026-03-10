from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch

from core_llm.tokenizer.encode import encode_text, load_tokenizer


PROMPT_PREFIX = "### Instruction\n"
INPUT_PREFIX = "\n\n### Input\n"
RESPONSE_PREFIX = "\n\n### Response\n"


@dataclass(frozen=True)
class SFTExample:
    id: str
    instruction: str
    input: str
    output: str


def format_sft_prompt(instruction: str, input_text: str) -> str:
    prompt = f"{PROMPT_PREFIX}{instruction.strip()}"
    if input_text.strip():
        prompt += f"{INPUT_PREFIX}{input_text.strip()}"
    prompt += RESPONSE_PREFIX
    return prompt


def iter_sft_examples(path: str | Path) -> Iterable[SFTExample]:
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            try:
                yield SFTExample(
                    id=str(row["id"]),
                    instruction=str(row["instruction"]).strip(),
                    input=str(row.get("input", "")).strip(),
                    output=str(row["output"]).strip(),
                )
            except KeyError as exc:
                raise ValueError(f"Missing required SFT field on line {lineno}: {exc}") from exc


def encode_sft_example(
    example: SFTExample,
    *,
    tokenizer_path: str | Path,
    seq_len: int,
    eos_id: int = 3,
) -> tuple[list[int], list[int]] | None:
    tokenizer = load_tokenizer(tokenizer_path)
    prompt_ids = encode_text(tokenizer, format_sft_prompt(example.instruction, example.input))
    response_ids = encode_text(tokenizer, example.output) + [eos_id]
    token_ids = prompt_ids + response_ids
    if len(token_ids) < 2:
        return None
    input_ids = token_ids[:-1]
    target_ids = token_ids[1:]
    prompt_target_count = max(0, len(prompt_ids) - 1)
    masked_targets = [-1] * prompt_target_count + response_ids
    masked_targets = masked_targets[: len(target_ids)]
    if all(token == -1 for token in masked_targets):
        return None
    if len(input_ids) > seq_len:
        input_ids = input_ids[-seq_len:]
        masked_targets = masked_targets[-seq_len:]
        if all(token == -1 for token in masked_targets):
            return None
    if len(input_ids) < seq_len:
        pad_len = seq_len - len(input_ids)
        input_ids = input_ids + [0] * pad_len
        masked_targets = masked_targets + [-1] * pad_len
    return input_ids, masked_targets


class SFTDataset:
    def __init__(
        self,
        *,
        manifest_path: str | Path,
        tokenizer_path: str | Path,
        batch_size: int,
        seq_len: int,
        split: str,
        val_fraction: float = 0.1,
        seed: int = 42,
    ):
        if split not in {"train", "val"}:
            raise ValueError(f"Unsupported SFT split: {split}")
        self.batch_size = batch_size
        self.seq_len = seq_len
        all_examples = list(iter_sft_examples(manifest_path))
        shuffled = list(all_examples)
        rng = random.Random(seed)
        rng.shuffle(shuffled)
        val_size = max(1, int(len(shuffled) * val_fraction)) if len(shuffled) > 1 else 0
        if split == "val":
            selected = shuffled[:val_size] if val_size else []
        else:
            selected = shuffled[val_size:] if val_size else shuffled
        self.rows: list[tuple[list[int], list[int]]] = []
        for example in selected:
            encoded = encode_sft_example(
                example,
                tokenizer_path=tokenizer_path,
                seq_len=seq_len,
            )
            if encoded is not None:
                self.rows.append(encoded)
        self.position = 0

    def __len__(self) -> int:
        return len(self.rows) // max(1, self.batch_size)

    def next_batch(self, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        if len(self.rows) < self.batch_size:
            raise ValueError("SFT dataset does not contain enough rows for a single batch")
        if self.position + self.batch_size > len(self.rows):
            self.position = 0
            random.shuffle(self.rows)
        batch = self.rows[self.position:self.position + self.batch_size]
        self.position += self.batch_size
        x = torch.tensor([row[0] for row in batch], dtype=torch.long, device=device)
        y = torch.tensor([row[1] for row in batch], dtype=torch.long, device=device)
        return x, y
