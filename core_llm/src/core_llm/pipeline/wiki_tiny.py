from __future__ import annotations

import json
from pathlib import Path

from core_llm.config import load_model_config
from core_llm.eval.perplexity import evaluate_checkpoint_perplexity
from core_llm.scripts.prepare_dataset import main as prepare_dataset_main
from core_llm.scripts.prepare_wikipedia_manifest import main as prepare_wikipedia_manifest_main
from core_llm.scripts.train import main as train_main
from core_llm.scripts.train_tokenizer import main as train_tokenizer_main


def resolve_run_paths(work_dir: Path) -> dict[str, Path]:
    return {
        "manifest": work_dir / "manifests" / "wikipedia_ja.jsonl",
        "manifest_report": work_dir / "manifests" / "wikipedia_ja.report.json",
        "tokenizer_dir": work_dir / "tokenizer",
        "prepared_dir": work_dir / "prepared",
        "checkpoint_dir": work_dir / "checkpoints",
        "eval_path": work_dir / "eval" / "perplexity.json",
        "summary_path": work_dir / "run_summary.json",
    }


def _run_cli(main_fn, argv: list[str]) -> None:
    import sys

    old_argv = sys.argv
    sys.argv = argv
    try:
        main_fn()
    finally:
        sys.argv = old_argv


def _require_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required {label} does not exist: {path}")


def run_wiki_tiny_pipeline(
    *,
    work_dir: Path,
    lang: str = "ja",
    raw_dir: Path,
    dump_path: Path | None = None,
    max_docs: int = 5000,
    min_chars: int = 120,
    tokenizer_config: Path,
    model_config: Path,
    train_config: Path,
    skip_manifest: bool = False,
    skip_tokenizer: bool = False,
    skip_dataset: bool = False,
    skip_train: bool = False,
    skip_eval: bool = False,
    refresh_dump: bool = False,
) -> dict:
    if lang != "ja":
        raise ValueError("Only ja is supported in the sample pipeline")
    work_dir.mkdir(parents=True, exist_ok=True)
    paths = resolve_run_paths(work_dir)
    for key in ("manifest", "manifest_report", "eval_path", "summary_path"):
        paths[key].parent.mkdir(parents=True, exist_ok=True)
    for key in ("tokenizer_dir", "prepared_dir", "checkpoint_dir"):
        paths[key].mkdir(parents=True, exist_ok=True)

    steps: list[str] = []

    if skip_manifest:
        _require_exists(paths["manifest"], "manifest")
        _require_exists(paths["manifest_report"], "manifest report")
    else:
        argv = [
            "prepare_wikipedia_manifest",
            "--lang", lang,
            "--output", str(paths["manifest"]),
            "--raw-dir", str(raw_dir),
            "--min-chars", str(min_chars),
            "--max-docs", str(max_docs),
            "--report-path", str(paths["manifest_report"]),
        ]
        if dump_path is not None:
            argv.extend(["--dump-path", str(dump_path)])
        if refresh_dump:
            argv.append("--refresh")
        _run_cli(prepare_wikipedia_manifest_main, argv)
        steps.append("manifest")

    manifest_report = json.loads(paths["manifest_report"].read_text(encoding="utf-8"))
    if int(manifest_report.get("kept_docs", 0)) <= 0:
        raise ValueError("Wikipedia manifest produced zero documents")

    tokenizer_model_path = paths["tokenizer_dir"] / "tokenizer.model"
    if skip_tokenizer:
        _require_exists(tokenizer_model_path, "tokenizer model")
    else:
        _run_cli(
            train_tokenizer_main,
            [
                "train_tokenizer",
                "--config", str(tokenizer_config),
                "--manifest", str(paths["manifest"]),
                "--output-dir", str(paths["tokenizer_dir"]),
            ],
        )
        steps.append("tokenizer")

    if skip_dataset:
        _require_exists(paths["prepared_dir"] / "train.bin", "prepared train.bin")
        _require_exists(paths["prepared_dir"] / "metadata.json", "prepared metadata")
    else:
        _run_cli(
            prepare_dataset_main,
            [
                "prepare_dataset",
                "--config", str(model_config),
                "--manifest", str(paths["manifest"]),
                "--tokenizer", str(tokenizer_model_path),
                "--output-dir", str(paths["prepared_dir"]),
                "--min-chars", str(min_chars),
            ],
        )
        steps.append("dataset")

    metadata = json.loads((paths["prepared_dir"] / "metadata.json").read_text(encoding="utf-8"))
    if int(metadata.get("train_tokens", 0)) <= 0:
        raise ValueError("Prepared dataset has zero train tokens")

    latest_checkpoint = paths["checkpoint_dir"] / "latest.pt"
    if skip_train:
        _require_exists(latest_checkpoint, "latest checkpoint")
    else:
        _run_cli(
            train_main,
            [
                "train",
                "--config", str(model_config),
                "--train-config", str(train_config),
                "--data-dir", str(paths["prepared_dir"]),
                "--checkpoint-dir", str(paths["checkpoint_dir"]),
            ],
        )
        steps.append("train")

    eval_result = None
    if skip_eval:
        _require_exists(paths["eval_path"], "evaluation output")
        eval_result = json.loads(paths["eval_path"].read_text(encoding="utf-8"))
    else:
        _require_exists(latest_checkpoint, "latest checkpoint")
        batch_size = load_model_config(model_config).block_size
        # evaluation helper expects actual batch size, not sequence length; use train batch from metadata fallback
        eval_result = evaluate_checkpoint_perplexity(
            latest_checkpoint,
            paths["prepared_dir"],
            batch_size=2,
            device="auto",
        )
        paths["eval_path"].write_text(json.dumps(eval_result, ensure_ascii=False, indent=2), encoding="utf-8")
        steps.append("eval")

    summary = {
        "work_dir": str(work_dir),
        "manifest_path": str(paths["manifest"]),
        "tokenizer_path": str(tokenizer_model_path) if tokenizer_model_path.exists() else None,
        "prepared_dir": str(paths["prepared_dir"]) if paths["prepared_dir"].exists() else None,
        "checkpoint_dir": str(paths["checkpoint_dir"]) if paths["checkpoint_dir"].exists() else None,
        "eval_path": str(paths["eval_path"]) if paths["eval_path"].exists() else None,
        "kept_docs": int(manifest_report.get("kept_docs", 0)),
        "train_tokens": int(metadata.get("train_tokens", 0)),
        "best_val_perplexity": None if eval_result is None else eval_result.get("val_perplexity"),
        "steps": steps,
    }
    paths["summary_path"].write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
