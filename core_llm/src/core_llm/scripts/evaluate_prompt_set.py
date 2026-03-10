from __future__ import annotations

import argparse
import json
from pathlib import Path

from core_llm.data.sft_dataset import format_sft_prompt
from core_llm.inference.cli import generate_text
from core_llm.inference.runtime import load_runtime


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
            prompt = format_sft_prompt(instruction, input_text) + "### Response\n"
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
