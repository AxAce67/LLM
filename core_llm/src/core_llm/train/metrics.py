from __future__ import annotations

import math


def perplexity_from_loss(loss: float | None) -> float | None:
    if loss is None:
        return None
    return math.exp(loss)
