import json
import os
from typing import List, Dict

import torch

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import inference
from data_collector.db_manager import DBManager


def load_benchmarks(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def score_keywords(text: str, keywords: List[str]) -> float:
    if not keywords:
        return 0.0
    lower = text.lower()
    matched = sum(1 for k in keywords if k.lower() in lower)
    return matched / len(keywords)


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    bench_path = os.path.join(base_dir, "eval", "benchmark_prompts.jsonl")
    benches = load_benchmarks(bench_path)
    print(f"[Eval] loaded benchmarks={len(benches)} path={bench_path}")

    device = "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"

    sp = inference.load_tokenizer()
    model, _ = inference.load_model(device=device)

    results = []
    by_category = {}
    for b in benches:
        out = inference.generate_text(
            model,
            sp,
            b["prompt"],
            max_new_tokens=int(os.environ.get("EVAL_MAX_TOKENS", "120")),
            temperature=0.4,
            top_k=40,
            top_p=0.9,
            repetition_penalty=1.05,
            device=device,
        )
        score = score_keywords(out, b.get("expected_keywords", []))
        category = b.get("category", "misc")
        results.append({"id": b["id"], "category": category, "score": score})
        by_category.setdefault(category, []).append(score)

    avg = sum(r["score"] for r in results) / max(1, len(results))
    category_scores = {
        cat: (sum(vals) / max(1, len(vals)))
        for cat, vals in by_category.items()
    }
    summary = {
        "avg_score": avg,
        "category_scores": category_scores,
        "results": results,
    }

    if os.environ.get("EVAL_SAVE_TO_DB", "1") == "1":
        try:
            db = DBManager()
            db.insert_evaluation_run(
                model_tag=os.environ.get("EVAL_MODEL_TAG", os.environ.get("MODEL_SIZE", "default")),
                avg_score=avg,
                result_json=summary,
            )
        except Exception as e:
            print(f"[Eval] Failed to save evaluation run: {e}")

    print(
        "[Eval] summary "
        f"avg_score={avg:.4f} "
        + " ".join([f"{k}={v:.4f}" for k, v in sorted(category_scores.items())])
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
