from __future__ import annotations

from pathlib import Path

import torch

from core_llm.config import ModelConfig, TrainConfig
from core_llm.data.dataset import BinaryTokenDataset
from core_llm.logging_utils import log_event
from core_llm.model.transformer import GPT
from core_llm.train.checkpoint import checkpoint_payload, load_checkpoint, save_checkpoint
from core_llm.train.metrics import perplexity_from_loss
from core_llm.train.optimizer import create_optimizer
from core_llm.train.scheduler import lr_for_step


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.no_grad()
def evaluate_loss(model: GPT, dataset: BinaryTokenDataset | None, device: str, eval_batches: int = 10) -> float | None:
    if dataset is None or len(dataset) == 0:
        return None
    was_training = model.training
    model.eval()
    losses = []
    for _ in range(min(eval_batches, len(dataset))):
        x, y = dataset.next_batch(device)
        _, loss = model(x, y)
        losses.append(float(loss.item()))
    if was_training:
        model.train()
    return sum(losses) / max(1, len(losses))


def train_model(
    *,
    data_dir: str | Path,
    checkpoint_dir: str | Path,
    model_config: ModelConfig,
    train_config: TrainConfig,
) -> dict:
    if model_config.vocab_size <= 0:
        raise ValueError("Model vocab_size must be positive")
    if train_config.seq_len != model_config.block_size:
        raise ValueError("Train seq_len must match model block_size")
    device = resolve_device(train_config.device)
    train_path = Path(data_dir) / "train.bin"
    val_path = Path(data_dir) / "val.bin"
    train_ds = BinaryTokenDataset(train_path, train_config.batch_size, train_config.seq_len)
    if len(train_ds.data) < (train_config.batch_size * train_config.seq_len + 1):
        raise ValueError("Prepared training dataset does not contain enough tokens")
    val_ds = BinaryTokenDataset(val_path, train_config.batch_size, train_config.seq_len) if val_path.exists() else None

    checkpoint_dir = Path(checkpoint_dir)
    metrics_path = checkpoint_dir / "train_metrics.jsonl"
    latest_path = checkpoint_dir / "latest.pt"
    best_path = checkpoint_dir / "best.pt"

    model = GPT(model_config).to(device)
    optimizer = create_optimizer(model, train_config)
    use_amp = device == "cuda" and train_config.amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_step = 0
    best_val_perplexity = float("inf")
    if latest_path.exists():
        checkpoint = load_checkpoint(latest_path, device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_step = int(checkpoint.get("step", 0))
        best_val_perplexity = float(checkpoint.get("best_val_perplexity", float("inf")))

    model.train()
    stale_evals = 0
    latest_train_loss = 0.0
    for step in range(start_step, train_config.total_steps):
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for _ in range(train_config.grad_accum_steps):
            x, y = train_ds.next_batch(device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                _, loss = model(x, y)
                loss = loss / train_config.grad_accum_steps
            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            accum_loss += float(loss.item())
        for group in optimizer.param_groups:
            group["lr"] = lr_for_step(step, train_config)
        if use_amp:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
            optimizer.step()
        latest_train_loss = accum_loss

        should_eval = ((step + 1) % train_config.eval_every == 0) or (step + 1 == train_config.total_steps)
        should_save = ((step + 1) % train_config.save_every == 0) or should_eval
        val_loss = None
        val_ppl = None
        if should_eval:
            val_loss = evaluate_loss(model, val_ds, device)
            val_ppl = perplexity_from_loss(val_loss)
            if val_ppl is not None and val_ppl < best_val_perplexity:
                best_val_perplexity = val_ppl
                stale_evals = 0
                save_checkpoint(
                    best_path,
                    checkpoint_payload(
                        step=step + 1,
                        model=model,
                        optimizer=optimizer,
                        model_config=model_config,
                        train_config=train_config,
                        latest_train_loss=latest_train_loss,
                        best_val_perplexity=best_val_perplexity,
                    ),
                )
            else:
                stale_evals += 1
                if stale_evals >= train_config.early_stopping_patience:
                    should_save = True
        if should_save:
            save_checkpoint(
                latest_path,
                checkpoint_payload(
                    step=step + 1,
                    model=model,
                    optimizer=optimizer,
                    model_config=model_config,
                    train_config=train_config,
                    latest_train_loss=latest_train_loss,
                    best_val_perplexity=best_val_perplexity,
                ),
            )
        log_event(
            metrics_path,
            "train_step",
            step=step + 1,
            train_loss=latest_train_loss,
            val_loss=val_loss,
            val_perplexity=val_ppl,
            lr=optimizer.param_groups[0]["lr"],
        )
        if stale_evals >= train_config.early_stopping_patience:
            break
    return {
        "step": step + 1,
        "latest_train_loss": latest_train_loss,
        "best_val_perplexity": best_val_perplexity,
        "device": device,
    }
