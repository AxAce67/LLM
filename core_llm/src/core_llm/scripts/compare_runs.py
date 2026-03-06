from __future__ import annotations

import argparse
import json

from core_llm.pipeline.run_registry import compare_runs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", dest="runs", action="append", required=True)
    args = ap.parse_args()

    print(json.dumps(compare_runs(args.runs), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
