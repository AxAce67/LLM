from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch

from core_llm.config import DEFAULT_SPECIAL_TOKENS
from core_llm.train.checkpoint import load_checkpoint


def _compute_intermediate_size(n_embd: int, ffn_multiplier: float, multiple_of: int) -> int:
    hidden = int(ffn_multiplier * n_embd)
    if multiple_of > 1:
        hidden = (hidden + multiple_of - 1) // multiple_of * multiple_of
    return hidden


def _load_special_tokens(tokenizer_path: Path) -> dict[str, int]:
    meta_path = tokenizer_path.parent / "tokenizer_meta.json"
    if not meta_path.exists():
        return dict(DEFAULT_SPECIAL_TOKENS)
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    return data.get("special_tokens", dict(DEFAULT_SPECIAL_TOKENS))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    checkpoint = load_checkpoint(args.checkpoint, device="cpu")
    model_config = checkpoint.get("model_config", {})
    if model_config.get("model_type") != "llama":
        raise ValueError(f"model_type must be 'llama' (got {model_config.get('model_type')})")
    if model_config.get("bias"):
        raise ValueError("LLaMA export requires bias=false to match HF/llama.cpp expectations.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_path = Path(args.tokenizer)
    shutil.copy2(tokenizer_path, output_dir / "tokenizer.model")
    special_tokens = _load_special_tokens(tokenizer_path)

    config = {
        "architectures": ["LlamaForCausalLM"],
        "bos_token_id": special_tokens.get("bos_id", 2),
        "eos_token_id": special_tokens.get("eos_id", 3),
        "pad_token_id": special_tokens.get("pad_id", 0),
        "hidden_size": model_config["n_embd"],
        "intermediate_size": _compute_intermediate_size(
            model_config["n_embd"],
            model_config.get("ffn_multiplier", 4.0),
            model_config.get("ffn_multiple_of", 256),
        ),
        "max_position_embeddings": model_config["block_size"],
        "model_type": "llama",
        "num_attention_heads": model_config["n_head"],
        "num_hidden_layers": model_config["n_layer"],
        "num_key_value_heads": model_config["n_head"],
        "rms_norm_eps": model_config.get("rms_norm_eps", 1e-5),
        "rope_theta": model_config.get("rope_theta", 10000.0),
        "tie_word_embeddings": False,
        "vocab_size": model_config["vocab_size"],
    }
    (output_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    tokenizer_config = {
        "bos_token": "<s>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
        "pad_token": "<pad>",
        "model_max_length": model_config["block_size"],
    }
    (output_dir / "tokenizer_config.json").write_text(
        json.dumps(tokenizer_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "special_tokens_map.json").write_text(
        json.dumps(
            {
                "bos_token": "<s>",
                "eos_token": "</s>",
                "unk_token": "<unk>",
                "pad_token": "<pad>",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    state = checkpoint["model_state_dict"]
    mapped = {}
    mapped["model.embed_tokens.weight"] = state["tok_embeddings.weight"]
    mapped["model.norm.weight"] = state["norm.weight"]
    mapped["lm_head.weight"] = state["lm_head.weight"]
    n_layer = model_config["n_layer"]
    for i in range(n_layer):
        prefix = f"layers.{i}"
        mapped[f"model.layers.{i}.input_layernorm.weight"] = state[f"{prefix}.attn_norm.weight"]
        mapped[f"model.layers.{i}.post_attention_layernorm.weight"] = state[f"{prefix}.ffn_norm.weight"]
        mapped[f"model.layers.{i}.self_attn.q_proj.weight"] = state[f"{prefix}.attn.q_proj.weight"]
        mapped[f"model.layers.{i}.self_attn.k_proj.weight"] = state[f"{prefix}.attn.k_proj.weight"]
        mapped[f"model.layers.{i}.self_attn.v_proj.weight"] = state[f"{prefix}.attn.v_proj.weight"]
        mapped[f"model.layers.{i}.self_attn.o_proj.weight"] = state[f"{prefix}.attn.o_proj.weight"]
        mapped[f"model.layers.{i}.mlp.gate_proj.weight"] = state[f"{prefix}.mlp.w1.weight"]
        mapped[f"model.layers.{i}.mlp.up_proj.weight"] = state[f"{prefix}.mlp.w3.weight"]
        mapped[f"model.layers.{i}.mlp.down_proj.weight"] = state[f"{prefix}.mlp.w2.weight"]

    torch.save(mapped, output_dir / "pytorch_model.bin")
    print(f"Exported HF LLaMA format to {output_dir}")


if __name__ == "__main__":
    main()
