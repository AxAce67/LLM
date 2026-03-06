from __future__ import annotations

import torch


def sample_next_token(
    logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    repetition_penalty: float = 1.0,
    recent_tokens: list[int] | None = None,
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    logits = logits / temperature
    if repetition_penalty > 1.0 and recent_tokens:
        for tok in set(recent_tokens[-128:]):
            tok_logits = logits[:, tok]
            logits[:, tok] = torch.where(tok_logits < 0, tok_logits * repetition_penalty, tok_logits / repetition_penalty)
    if top_k is not None:
        values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < values[:, [-1]]] = -float("inf")
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cumulative_probs > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = 0
        indices_to_remove = remove.scatter(1, sorted_indices, remove)
        logits = logits.masked_fill(indices_to_remove, -float("inf"))
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
