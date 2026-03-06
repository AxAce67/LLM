from __future__ import annotations

import argparse
from pathlib import Path

from core_llm.data.wiki_dump import build_wikipedia_manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="ja")
    ap.add_argument("--output", required=True)
    ap.add_argument("--raw-dir", default="data/raw/wikipedia")
    ap.add_argument("--dump-path")
    ap.add_argument("--min-chars", type=int, default=120)
    ap.add_argument("--max-docs", type=int)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--keep-dump", action="store_true")
    ap.add_argument("--report-path")
    args = ap.parse_args()

    report = build_wikipedia_manifest(
        lang=args.lang,
        output_path=Path(args.output),
        raw_dir=Path(args.raw_dir),
        dump_path=Path(args.dump_path) if args.dump_path else None,
        min_chars=args.min_chars,
        max_docs=args.max_docs,
        refresh=args.refresh,
        report_path=Path(args.report_path) if args.report_path else None,
    )
    print(report)
    if report["kept_docs"] <= 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
