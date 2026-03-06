from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path

from core_llm.data.cleaning import is_usable_text, looks_japanese, normalize_text
from core_llm.data.dedup import is_duplicate
from core_llm.data.manifest_schema import ManifestRecord, iter_manifest, write_manifest


def merge_manifests(
    *,
    inputs: list[Path],
    output: Path,
    min_chars: int = 80,
) -> dict:
    merged_rows: list[ManifestRecord] = []
    seen_texts: set[str] = set()
    seen_ids: set[str] = set()
    source_counts: Counter[str] = Counter()
    license_counts: Counter[str] = Counter()
    report = {
        "input_paths": [str(path) for path in inputs],
        "output_path": str(output),
        "input_docs": 0,
        "kept_docs": 0,
        "filtered_short": 0,
        "filtered_non_japanese": 0,
        "filtered_duplicate_text": 0,
        "deduplicated_id_collisions": 0,
        "source_counts": {},
        "license_counts": {},
    }

    for path in inputs:
        for row in iter_manifest(path):
            report["input_docs"] += 1
            text = normalize_text(row.text)
            if not is_usable_text(text, min_chars=min_chars):
                report["filtered_short"] += 1
                continue
            if row.lang != "ja" or not looks_japanese(text):
                report["filtered_non_japanese"] += 1
                continue
            if is_duplicate(text, seen_texts):
                report["filtered_duplicate_text"] += 1
                continue
            record_id = row.id
            if record_id in seen_ids:
                report["deduplicated_id_collisions"] += 1
                suffix = hashlib.sha1(f"{row.source}:{row.id}".encode("utf-8")).hexdigest()[:8]
                record_id = f"{row.id}:{suffix}"
            seen_ids.add(record_id)
            merged_rows.append(
                ManifestRecord(
                    id=record_id,
                    text=text,
                    lang="ja",
                    source=row.source,
                    license=row.license,
                    split_hint=row.split_hint,
                )
            )
            source_counts[row.source] += 1
            license_counts[row.license] += 1

    write_manifest(output, merged_rows)
    report["kept_docs"] = len(merged_rows)
    report["source_counts"] = dict(sorted(source_counts.items()))
    report["license_counts"] = dict(sorted(license_counts.items()))
    return report
