from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


ALLOWED_SPLITS = {"train", "val", "auto", ""}


@dataclass
class ManifestRecord:
    id: str
    text: str
    lang: str
    source: str
    license: str
    split_hint: str = "auto"

    @classmethod
    def from_dict(cls, row: dict) -> "ManifestRecord":
        required = ["id", "text", "lang", "source", "license"]
        missing = [key for key in required if key not in row]
        if missing:
            raise ValueError(f"Manifest record is missing required keys: {missing}")
        split_hint = (row.get("split_hint", "auto") or "auto").strip().lower()
        if split_hint not in ALLOWED_SPLITS:
            raise ValueError(f"Invalid split_hint: {split_hint}")
        text = str(row["text"]).strip()
        if not text:
            raise ValueError("Manifest text must not be empty")
        return cls(
            id=str(row["id"]).strip(),
            text=text,
            lang=str(row["lang"]).strip().lower(),
            source=str(row["source"]).strip(),
            license=str(row["license"]).strip(),
            split_hint=split_hint or "auto",
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "lang": self.lang,
            "source": self.source,
            "license": self.license,
            "split_hint": self.split_hint,
        }


def iter_manifest(path: str | Path) -> Iterator[ManifestRecord]:
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {lineno}: {exc}") from exc
            yield ManifestRecord.from_dict(payload)


def write_manifest(path: str | Path, rows: list[ManifestRecord]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")
