from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from core_llm.config import ModelConfig, TrainConfig, load_train_config
from core_llm.env import load_env_file
from core_llm.logging_utils import log_event
from core_llm.model.transformer import GPT
from core_llm.notify.discord import (
    build_command_failure_message,
    build_command_progress_message,
    build_command_started_message,
    build_command_success_message,
    collect_machine_status,
    format_duration,
    format_timestamp,
    resolve_discord_settings,
    send_discord_message,
)
from core_llm.seed import set_seed
from core_llm.train.checkpoint import checkpoint_payload, load_checkpoint, save_checkpoint
from core_llm.train.loop import (
    configure_torch_threads,
    resolve_amp,
    resolve_batch_size,
    resolve_device,
)
from core_llm.train.metrics import perplexity_from_loss
from core_llm.train.optimizer import create_optimizer
from core_llm.train.scheduler import lr_for_step
from core_llm.data.sft_dataset import SFTDataset


@torch.no_grad()
def evaluate_sft_loss(model: GPT, dataset: SFTDataset | None, device: str, eval_batches: int = 10) -> float | None:
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


def main() -> None:
    load_env_file(".env.local")
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-checkpoint", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--train-config", required=True)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--fresh", action="store_true", help="fail if latest.pt already exists")
    ap.add_argument("--discord-webhook-url")
    ap.add_argument("--discord-mention")
    args = ap.parse_args()
    webhook_url, mention = resolve_discord_settings(args.discord_webhook_url, args.discord_mention)

    work_dir = Path(args.work_dir)
    checkpoint_dir = work_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = checkpoint_dir / "train_metrics.jsonl"
    latest_path = checkpoint_dir / "latest.pt"
    best_path = checkpoint_dir / "best.pt"
    if args.fresh and latest_path.exists():
        raise SystemExit(f"Refusing to resume: latest.pt exists in {checkpoint_dir}")

    try:
        if webhook_url:
            started_payload = {
                "base_checkpoint": args.base_checkpoint,
                "manifest": args.manifest,
                "train_config": args.train_config,
                "work_dir": args.work_dir,
            }
            started_payload.update(collect_machine_status())
            send_discord_message(
                webhook_url,
                build_command_started_message(command_name="train_sft", payload=started_payload, mention=mention),
            )

        train_config = load_train_config(args.train_config)
        base_checkpoint = load_checkpoint(args.base_checkpoint, "cpu")
        model_config = ModelConfig(**base_checkpoint["model_config"])
        set_seed(train_config.seed)

        device = resolve_device(train_config.device)
        effective_batch_size = resolve_batch_size(train_config, model_config, device)
        effective_train_config = TrainConfig(**{**train_config.__dict__, "batch_size": effective_batch_size})
        configure_torch_threads(effective_train_config)
        use_amp = resolve_amp(effective_train_config, device)

        train_ds = SFTDataset(
            manifest_path=args.manifest,
            tokenizer_path=args.tokenizer,
            batch_size=effective_train_config.batch_size,
            seq_len=effective_train_config.seq_len,
            split="train",
            seed=effective_train_config.seed,
        )
        val_ds = SFTDataset(
            manifest_path=args.manifest,
            tokenizer_path=args.tokenizer,
            batch_size=effective_train_config.batch_size,
            seq_len=effective_train_config.seq_len,
            split="val",
            seed=effective_train_config.seed,
        )
        if len(train_ds.rows) < effective_train_config.batch_size:
            raise ValueError("SFT training dataset is too small for the configured batch size")

        model = GPT(model_config).to(device)
        optimizer = create_optimizer(model, effective_train_config)
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        start_step = 0
        best_val_perplexity = float("inf")
        if latest_path.exists():
            checkpoint = load_checkpoint(latest_path, device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_step = int(checkpoint.get("step", 0))
            best_val_perplexity = float(checkpoint.get("best_val_perplexity", float("inf")))
        else:
            model.load_state_dict(base_checkpoint["model_state_dict"])

        model.train()
        stale_evals = 0
        latest_train_loss = 0.0
        started_at = time.time()
        progress_notified = False

        for step in range(start_step, effective_train_config.total_steps):
            optimizer.zero_grad(set_to_none=True)
            accum_loss = 0.0
            for _ in range(effective_train_config.grad_accum_steps):
                x, y = train_ds.next_batch(device)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    _, loss = model(x, y)
                    loss = loss / effective_train_config.grad_accum_steps
                if use_amp:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                accum_loss += float(loss.item())
            for group in optimizer.param_groups:
                group["lr"] = lr_for_step(step, effective_train_config)
            if use_amp:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), effective_train_config.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), effective_train_config.grad_clip)
                optimizer.step()
            latest_train_loss = accum_loss

            should_eval = ((step + 1) % effective_train_config.eval_every == 0) or (
                step + 1 == effective_train_config.total_steps
            )
            should_save = ((step + 1) % effective_train_config.save_every == 0) or should_eval
            val_loss = None
            val_ppl = None
            if should_eval:
                val_loss = evaluate_sft_loss(model, val_ds, device)
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
                            train_config=effective_train_config,
                            latest_train_loss=latest_train_loss,
                            best_val_perplexity=best_val_perplexity,
                        ),
                    )
                else:
                    stale_evals += 1
                    if stale_evals >= effective_train_config.early_stopping_patience:
                        should_save = True
            if should_save:
                save_checkpoint(
                    latest_path,
                    checkpoint_payload(
                        step=step + 1,
                        model=model,
                        optimizer=optimizer,
                        model_config=model_config,
                        train_config=effective_train_config,
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
            elapsed_seconds = time.time() - started_at
            if webhook_url:
                should_notify_progress = False
                if not progress_notified and elapsed_seconds >= 180:
                    should_notify_progress = True
                    progress_notified = True
                elif should_eval and progress_notified:
                    should_notify_progress = True
                if should_notify_progress:
                    steps_completed = max(1, step + 1 - start_step)
                    seconds_per_step = elapsed_seconds / steps_completed
                    remaining_steps = max(0, effective_train_config.total_steps - (step + 1))
                    estimated_finish = time.time() + seconds_per_step * remaining_steps
                    progress_payload = {
                        "checkpoint_dir": str(checkpoint_dir),
                        "step": f"{step + 1}/{effective_train_config.total_steps}",
                        "latest_train_loss": latest_train_loss,
                        "best_val_perplexity": best_val_perplexity,
                        "val_perplexity": val_ppl,
                        "elapsed": format_duration(elapsed_seconds),
                        "sec_per_step": f"{seconds_per_step:.2f}",
                        "estimated_finish": format_timestamp(estimated_finish),
                    }
                    progress_payload.update(collect_machine_status())
                    send_discord_message(
                        webhook_url,
                        build_command_progress_message(
                            command_name="train_sft",
                            payload=progress_payload,
                            mention=mention,
                        ),
                    )
            if stale_evals >= effective_train_config.early_stopping_patience:
                break

        result = {
            "step": step + 1,
            "latest_train_loss": latest_train_loss,
            "best_val_perplexity": best_val_perplexity,
            "device": device,
            "batch_size": effective_train_config.batch_size,
            "amp": use_amp,
            "duration_seconds": time.time() - started_at,
        }
        if webhook_url:
            success_payload = {
                "work_dir": args.work_dir,
                "step": result.get("step", "-"),
                "latest_train_loss": result.get("latest_train_loss", "-"),
                "best_val_perplexity": result.get("best_val_perplexity", "-"),
                "duration": format_duration(result.get("duration_seconds")),
            }
            success_payload.update(collect_machine_status())
            send_discord_message(
                webhook_url,
                build_command_success_message(
                    command_name="train_sft",
                    payload=success_payload,
                    mention=mention,
                ),
            )
        print(result)
    except Exception as exc:
        if webhook_url:
            failure_message = build_command_failure_message(
                command_name="train_sft",
                error=str(exc),
                mention=mention,
            )
            machine_lines = "\n".join(f"{key}: {value}" for key, value in collect_machine_status().items())
            send_discord_message(webhook_url, f"{failure_message}\n{machine_lines}")
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
