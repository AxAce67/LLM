from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-path", default="data/runs/run_log.jsonl")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--event", action="append", default=[])
    ap.add_argument("--since")
    ap.add_argument("--contains")
    args = ap.parse_args()

    log_path = Path(args.log_path)
    if not log_path.exists():
        raise SystemExit(f"Log not found: {log_path}")
    events = set(args.event)
    since = None
    if args.since:
        if args.since.endswith("Z"):
            since = datetime.fromisoformat(args.since.replace("Z", "+00:00"))
        else:
            since = datetime.fromisoformat(args.since)
    contains = args.contains
    rows = []
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if events and row.get("event") not in events:
                continue
            if since:
                ts = row.get("ts")
                if not ts:
                    continue
                ts_val = datetime.fromisoformat(str(ts))
                if ts_val < since:
                    continue
            if contains and contains not in json.dumps(row, ensure_ascii=False):
                continue
            rows.append(row)
    rows = rows[-args.limit :] if args.limit > 0 else rows
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
