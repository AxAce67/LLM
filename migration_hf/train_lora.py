import argparse
import os
import sys

import torch
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from runtime.auto_tuner import detect_runtime_profile


def load_lines(path: str) -> list[str]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if t:
                rows.append(t)
    return rows


class TokenDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, texts: list[str], max_length: int = 512):
        self.samples = []
        for t in texts:
            tok = tokenizer(
                t,
                max_length=max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            self.samples.append(
                {
                    "input_ids": tok["input_ids"][0],
                    "attention_mask": tok["attention_mask"][0],
                    "labels": tok["input_ids"][0].clone(),
                }
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def main():
    profile = detect_runtime_profile()
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True, help="HF model id, e.g. Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--train_text", default="dataset/hf/train.txt")
    ap.add_argument("--output_dir", default="models/hf_lora")
    ap.add_argument("--max_length", type=int, default=int(os.environ.get("HF_MAX_LENGTH", str(profile["train_seq_len"]))))
    ap.add_argument("--epochs", type=float, default=float(os.environ.get("HF_EPOCHS", "1.0")))
    ap.add_argument("--batch_size", type=int, default=int(os.environ.get("HF_BATCH_SIZE", str(profile["train_batch_size"]))))
    ap.add_argument("--grad_accum_steps", type=int, default=int(os.environ.get("HF_GRAD_ACCUM_STEPS", "1")))
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--auto_tune", action="store_true", default=os.environ.get("HF_AUTO_TUNE", "1") == "1")
    ap.add_argument("--cpu_threads", type=int, default=int(os.environ.get("HF_CPU_THREADS", str(profile["pytorch_cpu_threads"]))))
    args = ap.parse_args()

    if not os.path.exists(args.train_text):
        raise FileNotFoundError(f"train text not found: {args.train_text}")

    if args.auto_tune:
        # CPU環境ではスレッド調整、低メモリ時は保守的に設定
        if profile["device"] == "cpu":
            torch.set_num_threads(max(1, args.cpu_threads))
            if hasattr(torch, "set_num_interop_threads"):
                torch.set_num_interop_threads(max(1, args.cpu_threads // 2))
        if profile["available_ram_gb"] < 4:
            args.batch_size = min(args.batch_size, 1)
            args.max_length = min(args.max_length, 256)
            args.grad_accum_steps = max(args.grad_accum_steps, 2)

    print(
        "[HF AutoTune] "
        f"device={profile['device']} "
        f"cores={profile['cpu_cores']} "
        f"ram={profile['available_ram_gb']}/{profile['total_ram_gb']}GB "
        f"batch={args.batch_size} seq={args.max_length} grad_accum={args.grad_accum_steps}"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=dtype)

    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)

    texts = load_lines(args.train_text)
    train_ds = TokenDataset(tokenizer, texts=texts, max_length=args.max_length)

    targs = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=max(1, args.grad_accum_steps),
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        bf16=(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
        dataloader_num_workers=max(0, min(8, profile["cpu_cores"] - 1)),
        report_to=[],
    )
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds)
    trainer.train()

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"LoRA saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
