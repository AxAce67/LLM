from __future__ import annotations

import argparse

from core_llm.inference.cli import generate_text
from core_llm.inference.runtime import load_runtime


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--tokenizer")
    ap.add_argument("--max-new-tokens", type=int, default=40)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--repetition-penalty", type=float, default=1.05)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    model, tokenizer, device = load_runtime(args.checkpoint, args.tokenizer, args.device)
    text = generate_text(
        model,
        tokenizer,
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        device=device,
    )
    print(text)


if __name__ == "__main__":
    main()
