from __future__ import annotations

import argparse
import json
from pathlib import Path

from core_llm.pipeline.run_registry import build_run_index


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--output")
    args = ap.parse_args()

    entries = build_run_index(args.runs_dir)
    if args.output:
        Path(args.output).write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(entries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
