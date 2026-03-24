"""Prepare manifest from livedoor news corpus."""
from __future__ import annotations

import argparse
from pathlib import Path

from core_llm.data.livedoor_news import build_livedoor_manifest
from core_llm.env import load_env_file


def main() -> None:
    load_env_file(".env.local")
    ap = argparse.ArgumentParser(description="Download and build livedoor news manifest")
    ap.add_argument("--output", required=True, help="Output manifest JSONL path")
    ap.add_argument("--raw-dir", default="data/raw/livedoor", help="Directory to store downloaded archive")
    ap.add_argument("--min-chars", type=int, default=120)
    ap.add_argument("--max-docs", type=int, default=None)
    ap.add_argument("--refresh", action="store_true", help="Re-download even if archive exists")
    args = ap.parse_args()

    build_livedoor_manifest(
        output_path=Path(args.output),
        raw_dir=Path(args.raw_dir),
        min_chars=args.min_chars,
        max_docs=args.max_docs,
        refresh=args.refresh,
    )


if __name__ == "__main__":
    main()
