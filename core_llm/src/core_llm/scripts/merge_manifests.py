from __future__ import annotations

import argparse
import json
from pathlib import Path

from core_llm.data.manifest_ops import merge_manifests


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", dest="inputs", action="append", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--report-path")
    ap.add_argument("--min-chars", type=int, default=80)
    args = ap.parse_args()

    input_paths = [Path(path) for path in args.inputs]
    output_path = Path(args.output)
    report_path = Path(args.report_path) if args.report_path else output_path.with_suffix(".report.json")
    report = merge_manifests(inputs=input_paths, output=output_path, min_chars=args.min_chars)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Merged {report['kept_docs']} records into {output_path}")


if __name__ == "__main__":
    main()
