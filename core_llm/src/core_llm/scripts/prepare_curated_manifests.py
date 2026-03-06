from __future__ import annotations

import argparse
import json
from pathlib import Path

from core_llm.data.local_corpus import build_manifest_from_directory
from core_llm.data.source_presets import CURATED_SOURCE_PRESETS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", default="data/raw/curated")
    ap.add_argument("--manifest-dir", default="data/manifests")
    ap.add_argument("--source", dest="sources", action="append")
    ap.add_argument("--min-chars", type=int, default=80)
    ap.add_argument("--summary-report")
    args = ap.parse_args()

    raw_root = Path(args.raw_root)
    manifest_dir = Path(args.manifest_dir)
    selected_sources = args.sources or list(CURATED_SOURCE_PRESETS.keys())
    reports: dict[str, dict] = {}

    for source_name in selected_sources:
        if source_name not in CURATED_SOURCE_PRESETS:
            raise ValueError(f"Unknown curated source preset: {source_name}")
        preset = CURATED_SOURCE_PRESETS[source_name]
        input_dir = raw_root / preset["relative_dir"]
        if not input_dir.exists():
            continue
        output_path = manifest_dir / f"{source_name}.jsonl"
        reports[source_name] = build_manifest_from_directory(
            input_dir=input_dir,
            output_path=output_path,
            source=source_name,
            license_name=preset["license"],
            min_chars=args.min_chars,
            extensions=list(preset["extensions"]),
            split_hint="auto",
            id_prefix=str(preset["id_prefix"]),
        )

    if not reports:
        raise SystemExit("No curated source directories were found")

    summary = {
        "raw_root": str(raw_root),
        "manifest_dir": str(manifest_dir),
        "sources": reports,
    }
    summary_path = Path(args.summary_report) if args.summary_report else manifest_dir / "curated_sources.report.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
