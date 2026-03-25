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
import re
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


def _auto_batch_size() -> int:
    """Compute optimal batch_size based on available GPU VRAM.

    Calibrated for 104M LLaMA with seq_len=512 and AMP enabled.
    Empirical: batch=16→9681MB, batch=48→15245MB on A10G (23028MB).
      - Fixed overhead: ~6893MB
      - Per-sample activation: ~174MB
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return 4
        total_mb = torch.cuda.get_device_properties(0).total_memory // (1024 ** 2)
        gpu_name = torch.cuda.get_device_name(0)
        available = int(total_mb * 0.80) - 6893
        if available <= 0:
            return 1
        n = available // 174
        # Round down to nearest multiple of 8 (optimal for Tensor Cores)
        batch_size = max(1, (n // 8) * 8)
        print(f"GPU: {gpu_name} ({total_mb}MB VRAM) → auto batch_size={batch_size}")
        return batch_size
    except Exception as e:
        print(f"Auto batch_size failed ({e}), falling back to 16")
        return 16


def _make_train_config(base_config: Path, batch_size: int, out_path: Path) -> None:
    """Write a modified train config with the given batch_size."""
    text = base_config.read_text()
    text = re.sub(r"^batch_size:.*$", f"batch_size: {batch_size}", text, flags=re.MULTILINE)
    text = re.sub(r"^grad_accum_steps:.*$", "grad_accum_steps: 1", text, flags=re.MULTILINE)
    out_path.write_text(text)
    print(f"Train config written → {out_path} (batch_size={batch_size})")


def _build_merged_manifest(work_dir: Path, data_dir: Path, dump_path: Path, core_dir: Path) -> None:
    """Build Wikipedia manifest, then merge livedoor records into it."""
    manifest_dir = work_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    # Step A: Wikipedia manifest (800k docs)
    _run([
        "python", "src/core_llm/scripts/prepare_wikipedia_manifest.py",
        "--lang", "ja",
        "--output", str(manifest_dir / "wikipedia_ja.jsonl"),
        "--raw-dir", str(data_dir / "raw" / "wikipedia"),
        "--min-chars", "120",
        "--max-docs", "800000",
        "--report-path", str(manifest_dir / "wikipedia_ja.report.json"),
        "--dump-path", str(dump_path),
    ], cwd=str(core_dir))

    # Step B: Livedoor manifest (~7k articles)
    livedoor_tmp = Path("/tmp/livedoor_ja.jsonl")
    _run([
        "python", "src/core_llm/scripts/prepare_livedoor_manifest.py",
        "--output", str(livedoor_tmp),
        "--raw-dir", str(data_dir / "raw" / "livedoor"),
    ], cwd=str(core_dir))

    # Step C: Append livedoor into Wikipedia manifest
    if livedoor_tmp.exists():
        count = 0
        with open(manifest_dir / "wikipedia_ja.jsonl", "a", encoding="utf-8") as out:
            for line in livedoor_tmp.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    out.write(line + "\n")
                    count += 1
        print(f"Merged {count} livedoor records into Wikipedia manifest")


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


def _restore_preprocess_cache(work_dir: Path) -> list[str]:
    """Restore manifest/tokenizer/dataset from Volume cache.

    Returns list of --skip-* flags to pass to run_wiki_tiny.py.
    Cache key 'medium_800k' encodes: max_docs=800000, vocab=16k, seq=512.
    """
    vol_cache = VOL_PATH / "cache" / "medium_800k"
    skip_flags: list[str] = []

    for sub, flag in [
        ("manifests", "--skip-manifest"),
        ("tokenizer", "--skip-tokenizer"),
        ("prepared",  "--skip-dataset"),
    ]:
        src = vol_cache / sub
        if src.exists() and any(src.iterdir()):
            dst = work_dir / sub
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copytree(src, dst, dirs_exist_ok=True)
            skip_flags.append(flag)
            print(f"Restored {sub} from Volume cache")
        else:
            print(f"No Volume cache for {sub}, will build from scratch")

    return skip_flags


def _save_preprocess_cache(work_dir: Path) -> None:
    """Save manifest/tokenizer/dataset to Volume cache for future runs."""
    vol_cache = VOL_PATH / "cache" / "medium_800k"
    saved_any = False

    for sub in ("manifests", "tokenizer", "prepared"):
        src = work_dir / sub
        if src.exists():
            dst = vol_cache / sub
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copytree(src, dst, dirs_exist_ok=True)
            print(f"Cached {sub} → volume:/cache/medium_800k/{sub}")
            saved_any = True

    if saved_any:
        volume.commit()


def _find_actual_work_dir(work_dir: Path) -> Path:
    """Return actual work dir, accounting for pipeline rename (e.g. appending step info)."""
    if work_dir.exists():
        return work_dir
    candidates = sorted(
        work_dir.parent.glob(work_dir.name + "*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        print(f"Work dir was renamed → {candidates[0]}")
        return candidates[0]
    raise FileNotFoundError(f"Work dir not found: {work_dir}")


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
    secrets=[modal.Secret.from_name("llm-secrets")],
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

    # 4. Set up fixed work_dir with preprocessing cache
    work_dir = Path("/tmp/medium_800k_run")
    work_dir.mkdir(parents=True, exist_ok=True)

    # Clear previous checkpoints so training always starts fresh
    checkpoint_dir = work_dir / "checkpoints"
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)

    # Restore manifest/tokenizer/dataset from Volume cache if available
    skip_flags = _restore_preprocess_cache(work_dir)

    # If manifest not cached, build Wikipedia manifest and merge livedoor
    if "--skip-manifest" not in skip_flags:
        _build_merged_manifest(work_dir, data_dir, dump_path, core_dir)
        skip_flags.append("--skip-manifest")

    # 5. Run pretraining pipeline
    batch_size = _auto_batch_size()
    auto_train_cfg = Path("/tmp/train_auto.yaml")
    _make_train_config(
        core_dir / "../configs/train_medium_100k_a10g.yaml",
        batch_size,
        auto_train_cfg,
    )
    train_cmd = [
        "python", "src/core_llm/scripts/run_wiki_tiny.py",
        "--dump-path", str(dump_path),
        "--max-docs", "800000",
        "--model-config", "../configs/model_llama_medium_ja_sample.yaml",
        "--tokenizer-config", "../configs/tokenizer_ja_medium_sample.yaml",
        "--train-config", str(auto_train_cfg),
        "--work-dir", str(work_dir),
    ] + skip_flags

    # Pass Discord credentials if available via Modal Secret
    discord_webhook = os.environ.get("DISCORD_WEBHOOK_URL")
    discord_mention = os.environ.get("DISCORD_MENTION")
    if discord_webhook:
        train_cmd += ["--discord-webhook-url", discord_webhook]
    if discord_mention:
        train_cmd += ["--discord-mention", discord_mention]

    # PYTORCH_CUDA_ALLOC_CONF reduces OOM risk from memory fragmentation
    train_env = os.environ.copy()
    train_env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    subprocess.run(train_cmd, cwd=str(core_dir), env=train_env)

    # 6. Resolve actual work dir (pipeline may have renamed it)
    actual_work_dir = _find_actual_work_dir(work_dir)

    # 7. Save preprocessing artifacts to Volume cache for next run
    _save_preprocess_cache(actual_work_dir)

    # 8. Save checkpoint and tokenizer to Volume
    _save_results(actual_work_dir)
    print("\nTraining complete!")


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
