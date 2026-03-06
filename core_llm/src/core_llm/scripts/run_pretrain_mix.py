from __future__ import annotations

import argparse
from pathlib import Path

from core_llm.pipeline.pretrain_mix import run_pretrain_mix_pipeline


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--manifest", dest="manifests", action="append", required=True)
    ap.add_argument("--min-chars", type=int, default=80)
    ap.add_argument("--tokenizer-config", default="configs/tokenizer_ja_tiny_sample.yaml")
    ap.add_argument("--model-config", default="configs/model_tiny_ja_sample.yaml")
    ap.add_argument("--train-config", default="configs/train_tiny_sample_cpu.yaml")
    ap.add_argument("--skip-merge", action="store_true")
    ap.add_argument("--skip-tokenizer", action="store_true")
    ap.add_argument("--skip-dataset", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-eval", action="store_true")
    args = ap.parse_args()

    summary = run_pretrain_mix_pipeline(
        work_dir=Path(args.work_dir),
        manifest_inputs=[Path(path) for path in args.manifests],
        tokenizer_config=Path(args.tokenizer_config),
        model_config=Path(args.model_config),
        train_config=Path(args.train_config),
        min_chars=args.min_chars,
        skip_merge=args.skip_merge,
        skip_tokenizer=args.skip_tokenizer,
        skip_dataset=args.skip_dataset,
        skip_train=args.skip_train,
        skip_eval=args.skip_eval,
    )
    print(summary)


if __name__ == "__main__":
    main()
