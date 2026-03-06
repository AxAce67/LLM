from __future__ import annotations

import argparse
import json
from pathlib import Path

from core_llm.data.government_fetch import fetch_government_corpus


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-file", default="data/seed_urls/government_ja.txt")
    ap.add_argument("--output-dir", default="data/raw/curated/government_ja")
    ap.add_argument("--min-chars", type=int, default=200)
    ap.add_argument("--timeout", type=int, default=20)
    ap.add_argument("--max-pages", type=int)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--report-path")
    args = ap.parse_args()

    report = fetch_government_corpus(
        seed_file=Path(args.seed_file),
        output_dir=Path(args.output_dir),
        min_chars=args.min_chars,
        timeout=args.timeout,
        max_pages=args.max_pages,
        refresh=args.refresh,
        report_path=Path(args.report_path) if args.report_path else None,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["saved_docs"] <= 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
