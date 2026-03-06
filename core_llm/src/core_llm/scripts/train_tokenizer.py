from __future__ import annotations

import argparse
from pathlib import Path

from core_llm.config import load_tokenizer_config
from core_llm.tokenizer.trainer import train_tokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output-dir", default="data/tokenizer")
    args = ap.parse_args()

    config = load_tokenizer_config(args.config)
    model_path = train_tokenizer(args.manifest, Path(args.output_dir), config)
    print(f"Tokenizer saved to {model_path}")


if __name__ == "__main__":
    main()
