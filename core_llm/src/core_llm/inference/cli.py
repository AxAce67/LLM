from __future__ import annotations

import torch

from core_llm.inference.sampler import sample_next_token


@torch.no_grad()
def generate_text(
    model,
    tokenizer,
    prompt: str,
    *,
    max_new_tokens: int = 40,
    temperature: float = 0.8,
    top_k: int | None = 40,
    top_p: float | None = 0.95,
    repetition_penalty: float = 1.05,
    device: str = "cpu",
) -> str:
    prompt_ids = tokenizer.encode_as_ids(prompt)
    if not prompt_ids:
        prompt_ids = [tokenizer.bos_id()]
    x = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
    for _ in range(max_new_tokens):
        cond = x if x.size(1) <= model.config.block_size else x[:, -model.config.block_size:]
        logits, _ = model(cond)
        next_idx = sample_next_token(
            logits[:, -1, :],
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            recent_tokens=x[0].tolist(),
        )
        x = torch.cat((x, next_idx), dim=1)
    new_ids = x[0].tolist()[len(prompt_ids):]
    return tokenizer.decode_ids(new_ids)
