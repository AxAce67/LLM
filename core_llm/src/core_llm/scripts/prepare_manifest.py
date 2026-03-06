from __future__ import annotations

import argparse
from pathlib import Path

from core_llm.data.local_corpus import build_manifest_from_directory


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--license", dest="license_name", required=True)
    ap.add_argument("--min-chars", type=int, default=80)
    ap.add_argument("--include-ext", dest="extensions", action="append")
    ap.add_argument("--split-hint", default="auto", choices=["train", "val", "auto"])
    ap.add_argument("--id-prefix", default="")
    ap.add_argument("--report-path")
    args = ap.parse_args()

    report = build_manifest_from_directory(
        input_dir=Path(args.input_dir),
        output_path=Path(args.output),
        source=args.source,
        license_name=args.license_name,
        min_chars=args.min_chars,
        extensions=args.extensions,
        split_hint=args.split_hint,
        id_prefix=args.id_prefix,
        report_path=Path(args.report_path) if args.report_path else None,
    )
    print(f"Wrote {report['kept_docs']} manifest rows to {args.output}")


if __name__ == "__main__":
    main()
