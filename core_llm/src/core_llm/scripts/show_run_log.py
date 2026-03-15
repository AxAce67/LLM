from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-path", default="data/runs/run_log.jsonl")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--event", action="append", default=[])
    args = ap.parse_args()

    log_path = Path(args.log_path)
    if not log_path.exists():
        raise SystemExit(f"Log not found: {log_path}")
    events = set(args.event)
    rows = []
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if events and row.get("event") not in events:
                continue
            rows.append(row)
    rows = rows[-args.limit :] if args.limit > 0 else rows
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
