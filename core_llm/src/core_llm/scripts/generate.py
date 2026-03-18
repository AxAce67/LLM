from __future__ import annotations

import argparse

from core_llm.inference.cli import generate_text
from core_llm.inference.runtime import load_runtime


def _apply_stop(text: str, stops: list[str] | None) -> str:
    if not stops:
        return text
    candidates = [s for s in stops if s]
    if not candidates:
        return text
    earliest_idx = None
    earliest_stop = ""
    for stop in candidates:
        idx = text.find(stop)
        if idx == -1:
            continue
        if earliest_idx is None or idx < earliest_idx:
            earliest_idx = idx
            earliest_stop = stop
    if earliest_idx is None:
        return text
    return text[: earliest_idx + len(earliest_stop)]


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
    ap.add_argument("--stop", action="append", default=[])
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
    print(_apply_stop(text, args.stop))


if __name__ == "__main__":
    main()
