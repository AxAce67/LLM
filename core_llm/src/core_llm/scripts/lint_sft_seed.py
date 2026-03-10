from __future__ import annotations

import argparse
from pathlib import Path

from core_llm.data.sft_quality import lint_sft_seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    args = ap.parse_args()

    input_path = Path(args.input)
    issues, category_counts = lint_sft_seed(input_path)
    print({"input": str(input_path), "categories": dict(category_counts), "issues": issues})
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
