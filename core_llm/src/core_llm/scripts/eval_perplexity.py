from __future__ import annotations

import argparse
import json
from pathlib import Path

from core_llm.eval.perplexity import evaluate_checkpoint_perplexity


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-dir", default="data/prepared")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--output", default="data/eval/perplexity.json")
    args = ap.parse_args()

    result = evaluate_checkpoint_perplexity(args.checkpoint, args.data_dir, args.batch_size, args.device)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
