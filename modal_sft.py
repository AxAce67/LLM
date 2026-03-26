"""Modal.com SFT script for 104M LLaMA model.

Usage:
    modal run modal_sft.py

    # Download SFT checkpoint after training:
    modal volume get llm-data-ja runs/sft_best.pt ./
    modal volume get llm-data-ja runs/tokenizer.model ./
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

app = modal.App("llm-sft-ja")

volume = modal.Volume.from_name("llm-data-ja", create_if_missing=True)
VOL_PATH = Path("/vol")

REPO_URL = "https://github.com/AxAce67/LLM.git"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "torch==2.6.0",
        "sentencepiece>=0.2.0",
        "numpy>=1.26,<3",
    )
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(cmd: list[str], cwd: str | None = None) -> None:
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=cwd)


def _merge_sft_data(raw_sft_dir: Path, output_path: Path) -> int:
    """Merge all SFT seed files into one manifest."""
    import json

    output_path.parent.mkdir(parents=True, exist_ok=True)
    files = [
        "qa_seed_core_ja.jsonl",
        "qa_seed_general_ja.jsonl",
        "qa_seed.jsonl",
    ]
    seen_ids: set[str] = set()
    kept = 0
    with open(output_path, "w", encoding="utf-8") as out:
        for fname in files:
            fpath = raw_sft_dir / fname
            if not fpath.exists():
                print(f"  skip (not found): {fname}")
                continue
            for line in fpath.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                rid = str(row.get("id", ""))
                if rid in seen_ids:
                    continue
                seen_ids.add(rid)
                instruction = str(row.get("instruction", "")).strip()
                output = str(row.get("output", "")).strip()
                if not instruction or not output:
                    continue
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                kept += 1
    print(f"SFT manifest: {kept} records → {output_path}")
    return kept


# ---------------------------------------------------------------------------
# Main SFT function
# ---------------------------------------------------------------------------

@app.function(
    gpu="A10G",
    image=image,
    volumes={str(VOL_PATH): volume},
    secrets=[modal.Secret.from_name("llm-secrets")],
    timeout=3600,  # 1 hour max
)
def sft_medium():
    # 1. Clone repo
    repo_dir = Path("/repo")
    if repo_dir.exists():
        shutil.rmtree(repo_dir)
    _run(["git", "clone", REPO_URL, str(repo_dir)])

    core_dir = repo_dir / "core_llm"
    os.chdir(core_dir)

    # 2. Install package
    _run(["pip", "install", "-e", "."], cwd=str(core_dir))

    # 3. Base checkpoint from Volume
    base_checkpoint = VOL_PATH / "runs" / "best.pt"
    if not base_checkpoint.exists():
        raise FileNotFoundError(
            f"Base checkpoint not found: {base_checkpoint}\n"
            "Run modal_pretrain.py first."
        )
    tokenizer = VOL_PATH / "runs" / "tokenizer.model"
    if not tokenizer.exists():
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer}")

    # 4. Build SFT manifest
    raw_sft_dir = core_dir / "data" / "raw" / "sft"
    manifest_path = Path("/tmp/sft_manifest.jsonl")
    n_records = _merge_sft_data(raw_sft_dir, manifest_path)
    if n_records < 10:
        raise ValueError(f"Too few SFT records: {n_records}")

    # 5. Run SFT
    sft_work_dir = Path("/tmp/sft_run")
    sft_work_dir.mkdir(parents=True, exist_ok=True)

    train_cmd = [
        "python", "src/core_llm/scripts/train_sft.py",
        "--base-checkpoint", str(base_checkpoint),
        "--tokenizer", str(tokenizer),
        "--manifest", str(manifest_path),
        "--train-config", "../configs/train_sft_medium_ja.yaml",
        "--work-dir", str(sft_work_dir),
    ]

    discord_webhook = os.environ.get("DISCORD_WEBHOOK_URL")
    discord_mention = os.environ.get("DISCORD_MENTION")
    if discord_webhook:
        train_cmd += ["--discord-webhook-url", discord_webhook]
    if discord_mention:
        train_cmd += ["--discord-mention", discord_mention]

    train_env = os.environ.copy()
    train_env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    subprocess.run(train_cmd, cwd=str(core_dir), env=train_env)

    # 6. Save SFT checkpoint to Volume
    # Find actual work dir (may have been renamed by pipeline)
    candidates = sorted(
        sft_work_dir.parent.glob(sft_work_dir.name + "*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    actual_work_dir = candidates[0] if candidates else sft_work_dir

    vol_runs = VOL_PATH / "runs"
    vol_runs.mkdir(parents=True, exist_ok=True)

    sft_best = actual_work_dir / "checkpoints" / "best.pt"
    if sft_best.exists():
        shutil.copy(sft_best, vol_runs / "sft_best.pt")
        print(f"Saved SFT checkpoint → volume:/runs/sft_best.pt")
    else:
        # Fallback to latest
        sft_latest = actual_work_dir / "checkpoints" / "latest.pt"
        if sft_latest.exists():
            shutil.copy(sft_latest, vol_runs / "sft_best.pt")
            print(f"Saved SFT latest → volume:/runs/sft_best.pt")
        else:
            print("WARNING: No SFT checkpoint found")

    volume.commit()
    print("\nSFT complete!")


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main():
    print("Starting SFT on Modal A10G...")
    sft_medium.remote()
    print("\nDone! Download results with:")
    print("  modal volume get llm-data-ja runs/sft_best.pt ./")
    print("  modal volume get llm-data-ja runs/tokenizer.model ./")
