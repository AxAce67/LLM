import torch

from core_llm.config import ModelConfig
from core_llm.model.transformer import GPT


def test_transformer_forward_shapes():
    config = ModelConfig(vocab_size=64, block_size=16, n_layer=2, n_head=2, n_embd=32, dropout=0.1, bias=False)
    model = GPT(config)
    x = torch.randint(0, 64, (2, 16))
    y = torch.randint(0, 64, (2, 16))
    logits, loss = model(x, y)
    assert tuple(logits.shape) == (2, 16, 64)
    assert loss is not None
