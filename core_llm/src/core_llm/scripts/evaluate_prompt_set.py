from __future__ import annotations

import argparse
import json
from pathlib import Path

from core_llm.data.sft_dataset import format_sft_prompt
from core_llm.inference.cli import generate_text
from core_llm.inference.runtime import load_runtime


def _score_response(text: str) -> dict[str, float | int | bool]:
    stripped = text.strip()
    if not stripped:
        return {"response_len": 0, "unique_char_ratio": 0.0, "repeat_char_ratio": 1.0, "empty": True}
    total_chars = len(stripped)
    unique_chars = len(set(stripped))
    unique_ratio = unique_chars / total_chars if total_chars else 0.0
    return {
        "response_len": total_chars,
        "unique_char_ratio": round(unique_ratio, 4),
        "repeat_char_ratio": round(1.0 - unique_ratio, 4),
        "empty": False,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--tokenizer")
    ap.add_argument("--max-new-tokens", type=int, default=120)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--repetition-penalty", type=float, default=1.05)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    questions_path = Path(args.questions)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model, tokenizer, device = load_runtime(args.checkpoint, args.tokenizer, args.device)

    with open(questions_path, "r", encoding="utf-8") as src, open(output_path, "w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            instruction = str(row.get("instruction", "")).strip()
            input_text = str(row.get("input", "")).strip()
            prompt = format_sft_prompt(instruction, input_text)
            response = generate_text(
                model,
                tokenizer,
                prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                device=device,
            )
            payload = {
                "id": row.get("id"),
                "category": row.get("category", ""),
                "instruction": instruction,
                "input": input_text,
                "response": response.strip(),
                "scores": _score_response(response),
            }
            dst.write(json.dumps(payload, ensure_ascii=False) + "\n")

    print(
        {
            "checkpoint": args.checkpoint,
            "questions": str(questions_path),
            "output": str(output_path),
            "device": device,
        }
    )


if __name__ == "__main__":
    main()
