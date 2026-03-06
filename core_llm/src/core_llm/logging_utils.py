from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


def append_jsonl(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def log_event(path: str | Path, event: str, **kwargs: Any) -> None:
    payload = {"ts": time.time(), "event": event}
    payload.update(kwargs)
    append_jsonl(path, payload)
