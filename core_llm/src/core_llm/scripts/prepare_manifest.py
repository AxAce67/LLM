from __future__ import annotations

import argparse
import json
from pathlib import Path

from core_llm.data.cleaning import is_usable_text, looks_japanese, normalize_text
from core_llm.data.dedup import is_duplicate
from core_llm.data.manifest_schema import ManifestRecord, write_manifest


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

    source_dir = Path(args.input_dir)
    extensions = args.extensions or ["txt", "md"]
    rows = []
    seen: set[str] = set()
    report = {
        "input_dir": str(source_dir),
        "output_path": str(args.output),
        "source": args.source,
        "license": args.license_name,
        "extensions": sorted(f".{ext.lstrip('.').lower()}" for ext in extensions),
        "input_files": 0,
        "kept_docs": 0,
        "filtered_short": 0,
        "filtered_non_japanese": 0,
        "filtered_duplicate": 0,
    }

    candidate_paths: list[Path] = []
    for extension in extensions:
        candidate_paths.extend(source_dir.rglob(f"*.{extension.lstrip('.')}"))
    for path in sorted(set(candidate_paths)):
        report["input_files"] += 1
        text = normalize_text(path.read_text(encoding="utf-8"))
        if not is_usable_text(text, min_chars=args.min_chars):
            report["filtered_short"] += 1
            continue
        if not looks_japanese(text):
            report["filtered_non_japanese"] += 1
            continue
        if is_duplicate(text, seen):
            report["filtered_duplicate"] += 1
            continue
        relative_stem = path.relative_to(source_dir).with_suffix("").as_posix().replace("/", "__")
        doc_id = f"{args.id_prefix}{relative_stem}" if args.id_prefix else relative_stem
        rows.append(
            ManifestRecord(
                id=doc_id,
                text=text,
                lang="ja",
                source=args.source,
                license=args.license_name,
                split_hint=args.split_hint,
            )
        )
    write_manifest(args.output, rows)
    report["kept_docs"] = len(rows)
    report_path = Path(args.report_path) if args.report_path else Path(args.output).with_suffix(".report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} manifest rows to {args.output}")


if __name__ == "__main__":
    main()
