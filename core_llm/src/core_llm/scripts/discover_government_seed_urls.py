from __future__ import annotations

import argparse

from core_llm.data.government_seed_discovery import discover_government_seed_urls, write_seed_urls


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="data/seed_urls/government_ja.txt")
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--timeout", type=int, default=20)
    args = ap.parse_args()

    urls = discover_government_seed_urls(limit=args.limit, timeout=args.timeout)
    write_seed_urls(args.output, urls)
    print(f"Wrote {len(urls)} URLs to {args.output}")


if __name__ == "__main__":
    main()
