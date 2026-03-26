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
        "datasets>=2.19",
    )
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(cmd: list[str], cwd: str | None = None) -> None:
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=cwd)


def _download_dolly_ja(output_path: Path) -> int:
    """Download kunishou/databricks-dolly-15k-ja from Hugging Face."""
    import json
    from datasets import load_dataset

    print("Downloading databricks-dolly-15k-ja...")
    ds = load_dataset("kunishou/databricks-dolly-15k-ja", split="train")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    with open(output_path, "w", encoding="utf-8") as out:
        for i, row in enumerate(ds):
            instruction = str(row.get("instruction", "")).strip()
            input_text = str(row.get("context", "")).strip()
            output = str(row.get("response", "")).strip()
            if not instruction or not output:
                continue
            payload = {
                "id": f"dolly-{i:05d}",
                "instruction": instruction,
                "input": input_text,
                "output": output,
            }
            out.write(json.dumps(payload, ensure_ascii=False) + "\n")
            kept += 1
    print(f"dolly-15k-ja: {kept} records → {output_path}")
    return kept


def _merge_sft_data(raw_sft_dir: Path, dolly_path: Path, output_path: Path) -> int:
    """Merge dolly-15k-ja + seed files into one manifest."""
    import json

    output_path.parent.mkdir(parents=True, exist_ok=True)
    seen_ids: set[str] = set()
    kept = 0

    with open(output_path, "w", encoding="utf-8") as out:
        # 1. dolly-15k-ja (main data)
        if dolly_path.exists():
            for line in dolly_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                rid = str(row.get("id", ""))
                if rid in seen_ids:
                    continue
                seen_ids.add(rid)
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                kept += 1

        # 2. Seed files (高品質な少量データ)
        for fname in ["qa_seed_core_ja.jsonl", "qa_seed_general_ja.jsonl", "qa_seed.jsonl"]:
            fpath = raw_sft_dir / fname
            if not fpath.exists():
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

    print(f"SFT manifest: {kept} records total → {output_path}")
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

    # 4. Build SFT manifest (dolly-15k-ja + seed files)
    raw_sft_dir = core_dir / "data" / "raw" / "sft"
    dolly_path = Path("/tmp/dolly_ja.jsonl")
    manifest_path = Path("/tmp/sft_manifest.jsonl")
    _download_dolly_ja(dolly_path)
    n_records = _merge_sft_data(raw_sft_dir, dolly_path, manifest_path)
    if n_records < 10:
        raise ValueError(f"Too few SFT records: {n_records}")

    # 5. Run SFT
    # Use a fixed checkpoint dir outside work_dir so renaming doesn't affect us
    sft_work_dir = Path("/tmp/sft_run")
    sft_checkpoint_dir = Path("/tmp/sft_checkpoints")
    sft_checkpoint_dir.mkdir(parents=True, exist_ok=True)
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
    # Search all possible locations: original + renamed work dirs
    vol_runs = VOL_PATH / "runs"
    vol_runs.mkdir(parents=True, exist_ok=True)

    candidates = sorted(
        Path("/tmp").glob("sft_run*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    print(f"Found work dir candidates: {[str(c) for c in candidates]}")

    saved = False
    for work_dir_candidate in candidates:
        sft_best = work_dir_candidate / "checkpoints" / "best.pt"
        sft_latest = work_dir_candidate / "checkpoints" / "latest.pt"
        for ckpt in [sft_best, sft_latest]:
            if ckpt.exists():
                shutil.copy(ckpt, vol_runs / "sft_best.pt")
                print(f"Saved {ckpt} → volume:/runs/sft_best.pt")
                saved = True
                break
        if saved:
            break

    if not saved:
        print("WARNING: No SFT checkpoint found in any candidate dir")

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
