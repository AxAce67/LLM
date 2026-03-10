from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    with open(input_path, "r", encoding="utf-8") as src, open(output_path, "w", encoding="utf-8") as dst:
        for idx, line in enumerate(src, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            instruction = str(row.get("instruction", "")).strip()
            input_text = str(row.get("input", "")).strip()
            output = str(row.get("output", "")).strip()
            if not instruction or not output:
                continue
            payload = {
                "id": str(row.get("id", f"sft-{idx:06d}")),
                "instruction": instruction,
                "input": input_text,
                "output": output,
            }
            dst.write(json.dumps(payload, ensure_ascii=False) + "\n")
            kept += 1
    print({"input": str(input_path), "output": str(output_path), "kept_examples": kept})


if __name__ == "__main__":
    main()
