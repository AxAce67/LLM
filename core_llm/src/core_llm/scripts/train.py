from __future__ import annotations

import argparse
from pathlib import Path

from core_llm.config import load_model_config, load_train_config
from core_llm.seed import set_seed
from core_llm.train.loop import train_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--train-config", required=True)
    ap.add_argument("--data-dir", default="data/prepared")
    ap.add_argument("--checkpoint-dir", default="data/checkpoints")
    args = ap.parse_args()

    model_config = load_model_config(args.config)
    train_config = load_train_config(args.train_config)
    set_seed(train_config.seed)
    result = train_model(
        data_dir=Path(args.data_dir),
        checkpoint_dir=Path(args.checkpoint_dir),
        model_config=model_config,
        train_config=train_config,
    )
    print(result)


if __name__ == "__main__":
    main()
