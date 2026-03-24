"""Modal.com pretraining script for 104M LLaMA model.

Usage:
    # First time (downloads Wikipedia ~4.3GB):
    modal run modal_pretrain.py

    # Subsequent runs (reuses cached Wikipedia):
    modal run modal_pretrain.py

    # Download checkpoint after training:
    modal volume get llm-data-ja runs/best.pt ./
    modal volume get llm-data-ja runs/tokenizer.model ./

    # List volume contents:
    modal volume ls llm-data-ja
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# App & infrastructure
# ---------------------------------------------------------------------------

app = modal.App("llm-pretrain-ja")

# Persistent volume: caches Wikipedia dump and saves checkpoints
volume = modal.Volume.from_name("llm-data-ja", create_if_missing=True)
VOL_PATH = Path("/vol")

REPO_URL = "https://github.com/AxAce67/LLM.git"
WIKI_URL = "https://dumps.wikimedia.org/jawiki/latest/jawiki-latest-pages-articles.xml.bz2"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("wget", "git")
    .pip_install(
        "torch==2.6.0",
        "sentencepiece>=0.2.0",
        "numpy>=1.26,<3",
        "beautifulsoup4>=4.12",
        "requests>=2.32",
    )
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(cmd: list[str], cwd: str | None = None) -> None:
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=cwd)


def _ensure_wikipedia(data_dir: Path) -> Path:
    """Return path to Wikipedia dump, downloading if needed."""
    dump_path = data_dir / "raw" / "wikipedia" / "jawiki-latest-pages-articles.xml.bz2"
    dump_path.parent.mkdir(parents=True, exist_ok=True)

    # Check volume cache first
    vol_dump = VOL_PATH / "wikipedia" / "jawiki-latest-pages-articles.xml.bz2"
    if vol_dump.exists():
        print(f"Wikipedia dump found in volume cache ({vol_dump.stat().st_size // 1024 // 1024}MB)")
        if not dump_path.exists():
            shutil.copy(vol_dump, dump_path)
        return dump_path

    # Download from Wikimedia
    print("Downloading Wikipedia dump (~4.3GB)…")
    _run(["wget", "-q", "--show-progress", "-O", str(dump_path), WIKI_URL])

    # Cache to volume for next run
    vol_dump.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(dump_path, vol_dump)
    volume.commit()
    print("Wikipedia dump cached to volume.")
    return dump_path


def _save_results(work_dir: Path) -> None:
    """Copy checkpoint and tokenizer to Modal volume."""
    # Find best checkpoint
    checkpoint = work_dir / "checkpoints" / "best.pt"
    tokenizer = work_dir / "tokenizer" / "tokenizer.model"

    vol_runs = VOL_PATH / "runs"
    vol_runs.mkdir(parents=True, exist_ok=True)

    if checkpoint.exists():
        shutil.copy(checkpoint, vol_runs / "best.pt")
        print(f"Saved checkpoint → volume:/runs/best.pt")
    if tokenizer.exists():
        shutil.copy(tokenizer, vol_runs / "tokenizer.model")
        print(f"Saved tokenizer  → volume:/runs/tokenizer.model")

    # Also copy full run dir summary
    summary = work_dir / "run_summary.json"
    if summary.exists():
        shutil.copy(summary, vol_runs / "run_summary.json")

    volume.commit()


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

@app.function(
    gpu="A10G",
    image=image,
    volumes={str(VOL_PATH): volume},
    secrets=[modal.Secret.from_name("llm-secrets", required=False)],
    timeout=86400,  # 24 hours max
)
def pretrain_medium():
    # 1. Clone repo
    repo_dir = Path("/repo")
    if repo_dir.exists():
        shutil.rmtree(repo_dir)
    _run(["git", "clone", REPO_URL, str(repo_dir)])

    core_dir = repo_dir / "core_llm"
    os.chdir(core_dir)

    # 2. Install package
    _run(["pip", "install", "-e", "."], cwd=str(core_dir))

    # 3. Ensure Wikipedia dump
    data_dir = core_dir / "data"
    dump_path = _ensure_wikipedia(data_dir)

    # 4. Prepare livedoor news manifest
    livedoor_manifest = data_dir / "manifests" / "livedoor_ja.jsonl"
    _run([
        "python", "src/core_llm/scripts/prepare_livedoor_manifest.py",
        "--output", str(livedoor_manifest),
        "--raw-dir", str(data_dir / "raw" / "livedoor"),
    ], cwd=str(core_dir))

    # 5. Run pretraining pipeline (Wikipedia 800k + livedoor mix)
    #    Using run_wiki_tiny for Wikipedia base, then we'll add livedoor via merge
    train_cmd = [
        "python", "src/core_llm/scripts/run_wiki_tiny.py",
        "--dump-path", str(dump_path),
        "--max-docs", "800000",
        "--model-config", "../configs/model_llama_medium_ja_sample.yaml",
        "--tokenizer-config", "../configs/tokenizer_ja_medium_sample.yaml",
        "--train-config", "../configs/train_medium_100k_a10g.yaml",
    ]
    # Pass Discord credentials if available via Modal Secret
    discord_webhook = os.environ.get("DISCORD_WEBHOOK_URL")
    discord_mention = os.environ.get("DISCORD_MENTION")
    if discord_webhook:
        train_cmd += ["--discord-webhook-url", discord_webhook]
    if discord_mention:
        train_cmd += ["--discord-mention", discord_mention]

    result = subprocess.run(train_cmd, cwd=str(core_dir))

    # 6. Find latest run dir and save to volume
    runs_dir = data_dir / "runs"
    run_dirs = sorted(
        [d for d in runs_dir.iterdir() if d.is_dir() and "wiki_tiny" in d.name],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    if run_dirs:
        _save_results(run_dirs[0])
        print(f"\nTraining complete! Run: {run_dirs[0].name}")
    else:
        print("Warning: could not find run directory to save results.")


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main():
    print("Starting 104M LLaMA pretraining on Modal A10G...")
    print("This will take approximately 5-10 hours.")
    print("Monitor progress at: https://modal.com/apps")
    pretrain_medium.remote()
    print("\nDone! Download results with:")
    print("  modal volume get llm-data-ja runs/best.pt ./")
    print("  modal volume get llm-data-ja runs/tokenizer.model ./")
