from __future__ import annotations

import argparse
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
    args = ap.parse_args()

    source_dir = Path(args.input_dir)
    rows = []
    seen: set[str] = set()
    for path in sorted(source_dir.rglob("*.txt")):
        text = normalize_text(path.read_text(encoding="utf-8"))
        if not is_usable_text(text, min_chars=args.min_chars):
            continue
        if not looks_japanese(text):
            continue
        if is_duplicate(text, seen):
            continue
        rows.append(
            ManifestRecord(
                id=path.stem,
                text=text,
                lang="ja",
                source=args.source,
                license=args.license_name,
                split_hint="auto",
            )
        )
    write_manifest(args.output, rows)
    print(f"Wrote {len(rows)} manifest rows to {args.output}")


if __name__ == "__main__":
    main()
