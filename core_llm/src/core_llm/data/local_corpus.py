from __future__ import annotations

import json
from pathlib import Path

from core_llm.data.cleaning import is_usable_text, looks_japanese, normalize_text
from core_llm.data.dedup import is_duplicate
from core_llm.data.manifest_schema import ManifestRecord, write_manifest


def build_manifest_from_directory(
    *,
    input_dir: Path,
    output_path: Path,
    source: str,
    license_name: str,
    min_chars: int = 80,
    extensions: list[str] | None = None,
    split_hint: str = "auto",
    id_prefix: str = "",
    report_path: Path | None = None,
) -> dict:
    normalized_extensions = sorted(f".{ext.lstrip('.').lower()}" for ext in (extensions or ["txt", "md"]))
    rows: list[ManifestRecord] = []
    seen: set[str] = set()
    report = {
        "input_dir": str(input_dir),
        "output_path": str(output_path),
        "source": source,
        "license": license_name,
        "extensions": normalized_extensions,
        "input_files": 0,
        "kept_docs": 0,
        "filtered_short": 0,
        "filtered_non_japanese": 0,
        "filtered_duplicate": 0,
    }

    candidate_paths: list[Path] = []
    for extension in normalized_extensions:
        candidate_paths.extend(input_dir.rglob(f"*{extension}"))
    for path in sorted(set(candidate_paths)):
        report["input_files"] += 1
        text = normalize_text(path.read_text(encoding="utf-8"))
        if not is_usable_text(text, min_chars=min_chars):
            report["filtered_short"] += 1
            continue
        if not looks_japanese(text):
            report["filtered_non_japanese"] += 1
            continue
        if is_duplicate(text, seen):
            report["filtered_duplicate"] += 1
            continue
        relative_stem = path.relative_to(input_dir).with_suffix("").as_posix().replace("/", "__")
        doc_id = f"{id_prefix}{relative_stem}" if id_prefix else relative_stem
        rows.append(
            ManifestRecord(
                id=doc_id,
                text=text,
                lang="ja",
                source=source,
                license=license_name,
                split_hint=split_hint,
            )
        )

    write_manifest(output_path, rows)
    report["kept_docs"] = len(rows)
    target_report = report_path if report_path is not None else output_path.with_suffix(".report.json")
    target_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
