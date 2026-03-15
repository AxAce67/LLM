from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

from core_llm.data.sft_dataset import format_sft_prompt
from core_llm.inference.cli import generate_text
from core_llm.inference.runtime import load_runtime


CONTENT_CHAR_RE = re.compile(r"[A-Za-z0-9\u3040-\u30FF\u4E00-\u9FFF]")


def _repeat_ngram_ratio(text: str, n: int) -> float:
    if n <= 0:
        return 0.0
    total = len(text) - n + 1
    if total <= 0:
        return 0.0
    counts: dict[str, int] = {}
    for i in range(total):
        ng = text[i : i + n]
        counts[ng] = counts.get(ng, 0) + 1
    repeats = sum(count - 1 for count in counts.values() if count > 1)
    return round(repeats / total, 4)


def _char_ratios(text: str) -> dict[str, float]:
    total = len(text)
    if total == 0:
        return {"symbol_ratio": 0.0, "latin_ratio": 0.0, "digit_ratio": 0.0}
    symbol = 0
    latin = 0
    digit = 0
    for ch in text:
        if ch.isdigit():
            digit += 1
            continue
        name = unicodedata.name(ch, "")
        if "LATIN" in name:
            latin += 1
        category = unicodedata.category(ch)
        if category.startswith(("P", "S")):
            symbol += 1
    return {
        "symbol_ratio": round(symbol / total, 4),
        "latin_ratio": round(latin / total, 4),
        "digit_ratio": round(digit / total, 4),
    }


def _instruction_coverage(instruction: str, response: str) -> float:
    inst_chars = {ch for ch in instruction if CONTENT_CHAR_RE.match(ch)}
    if not inst_chars:
        return 0.0
    resp_chars = {ch for ch in response if CONTENT_CHAR_RE.match(ch)}
    overlap = len(inst_chars & resp_chars)
    return round(overlap / len(inst_chars), 4)


def _structure_ok(category: str, response: str, input_text: str) -> bool:
    head = response[:80]
    if category == "definition":
        return "とは" in head or "である" in head or "です" in head
    if category == "comparison":
        return any(term in response for term in ("違い", "一方", "それぞれ", "対して", "比較"))
    if category == "procedure":
        if re.search(r"(^|\n)\s*(\d+\.|[-*]|・)", response):
            return True
        return any(term in response for term in ("手順", "ステップ", "次に", "まず"))
    if category == "summary":
        return len(response) <= max(20, int(len(input_text) * 1.2))
    if category == "writing":
        return len(response) >= 20
    return True


def _score_response(
    instruction: str,
    input_text: str,
    category: str,
    text: str,
) -> dict[str, float | int | bool]:
    stripped = text.strip()
    if not stripped:
        return {
            "response_len": 0,
            "unique_char_ratio": 0.0,
            "repeat_char_ratio": 1.0,
            "repeat_bigram_ratio": 0.0,
            "repeat_trigram_ratio": 0.0,
            "symbol_ratio": 0.0,
            "latin_ratio": 0.0,
            "digit_ratio": 0.0,
            "prompt_leak": False,
            "instruction_coverage": 0.0,
            "structure_ok": False,
            "qa_ok": False,
            "empty": True,
        }
    total_chars = len(stripped)
    unique_chars = len(set(stripped))
    unique_ratio = unique_chars / total_chars if total_chars else 0.0
    ratios = _char_ratios(stripped)
    prompt_leak = bool(re.search(r"###\s*(Instruction|Response)", stripped))
    coverage = _instruction_coverage(instruction, stripped)
    structure_ok = _structure_ok(category, stripped, input_text)
    qa_ok = (
        coverage >= 0.2
        and structure_ok
        and not prompt_leak
        and total_chars >= 15
    )
    return {
        "response_len": total_chars,
        "unique_char_ratio": round(unique_ratio, 4),
        "repeat_char_ratio": round(1.0 - unique_ratio, 4),
        "repeat_bigram_ratio": _repeat_ngram_ratio(stripped, 2),
        "repeat_trigram_ratio": _repeat_ngram_ratio(stripped, 3),
        "symbol_ratio": ratios["symbol_ratio"],
        "latin_ratio": ratios["latin_ratio"],
        "digit_ratio": ratios["digit_ratio"],
        "prompt_leak": prompt_leak,
        "instruction_coverage": coverage,
        "structure_ok": structure_ok,
        "qa_ok": qa_ok,
        "empty": False,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--tokenizer")
    ap.add_argument("--max-new-tokens", type=int, default=120)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--repetition-penalty", type=float, default=1.05)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    questions_path = Path(args.questions)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model, tokenizer, device = load_runtime(args.checkpoint, args.tokenizer, args.device)

    total = 0
    empty = 0
    sum_len = 0
    sum_unique_ratio = 0.0
    sum_repeat_ratio = 0.0
    sum_repeat_bigram_ratio = 0.0
    sum_repeat_trigram_ratio = 0.0
    sum_symbol_ratio = 0.0
    sum_latin_ratio = 0.0
    sum_digit_ratio = 0.0
    sum_instruction_coverage = 0.0
    prompt_leak = 0
    qa_ok = 0

    with open(questions_path, "r", encoding="utf-8") as src, open(output_path, "w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            instruction = str(row.get("instruction", "")).strip()
            input_text = str(row.get("input", "")).strip()
            prompt = format_sft_prompt(instruction, input_text)
            response = generate_text(
                model,
                tokenizer,
                prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                device=device,
            )
            scores = _score_response(instruction, input_text, str(row.get("category", "")), response)
            payload = {
                "id": row.get("id"),
                "category": row.get("category", ""),
                "instruction": instruction,
                "input": input_text,
                "response": response.strip(),
                "scores": scores,
            }
            total += 1
            score = payload["scores"]
            if score["empty"]:
                empty += 1
            sum_len += int(score["response_len"])
            sum_unique_ratio += float(score["unique_char_ratio"])
            sum_repeat_ratio += float(score["repeat_char_ratio"])
            sum_repeat_bigram_ratio += float(score["repeat_bigram_ratio"])
            sum_repeat_trigram_ratio += float(score["repeat_trigram_ratio"])
            sum_symbol_ratio += float(score["symbol_ratio"])
            sum_latin_ratio += float(score["latin_ratio"])
            sum_digit_ratio += float(score["digit_ratio"])
            sum_instruction_coverage += float(score["instruction_coverage"])
            if score["prompt_leak"]:
                prompt_leak += 1
            if score["qa_ok"]:
                qa_ok += 1
            dst.write(json.dumps(payload, ensure_ascii=False) + "\n")

    summary = {
        "checkpoint": args.checkpoint,
        "questions": str(questions_path),
        "output": str(output_path),
        "tokenizer": args.tokenizer,
        "device": device,
        "generation": {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "repetition_penalty": args.repetition_penalty,
        },
        "counts": {
            "total": total,
            "empty": empty,
            "empty_rate": round(empty / total, 4) if total else 0.0,
            "prompt_leak": prompt_leak,
            "prompt_leak_rate": round(prompt_leak / total, 4) if total else 0.0,
            "qa_ok": qa_ok,
            "qa_ok_rate": round(qa_ok / total, 4) if total else 0.0,
        },
        "response_stats": {
            "avg_len": round(sum_len / total, 2) if total else 0.0,
            "avg_unique_char_ratio": round(sum_unique_ratio / total, 4) if total else 0.0,
            "avg_repeat_char_ratio": round(sum_repeat_ratio / total, 4) if total else 0.0,
            "avg_repeat_bigram_ratio": round(sum_repeat_bigram_ratio / total, 4) if total else 0.0,
            "avg_repeat_trigram_ratio": round(sum_repeat_trigram_ratio / total, 4) if total else 0.0,
            "avg_symbol_ratio": round(sum_symbol_ratio / total, 4) if total else 0.0,
            "avg_latin_ratio": round(sum_latin_ratio / total, 4) if total else 0.0,
            "avg_digit_ratio": round(sum_digit_ratio / total, 4) if total else 0.0,
            "avg_instruction_coverage": round(sum_instruction_coverage / total, 4) if total else 0.0,
        },
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        {
            "checkpoint": args.checkpoint,
            "questions": str(questions_path),
            "output": str(output_path),
            "device": device,
            "summary": str(summary_path),
        }
    )


if __name__ == "__main__":
    main()
