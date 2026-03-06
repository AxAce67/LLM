import os
import shutil
import time

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_collector.db_manager import DBManager


def promote_best():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    checkpoints_dir = os.path.join(base_dir, "checkpoints")
    source = os.path.join(checkpoints_dir, "ckpt_best.pt")
    target = os.path.join(checkpoints_dir, "ckpt_production.pt")

    if not os.path.exists(source):
        raise FileNotFoundError(f"Best checkpoint not found: {source}")

    print(f"[Promote] source={source} target={target}")
    shutil.copy2(source, target)
    db = DBManager()
    latest_eval = db.get_latest_evaluation_run()
    min_score = float(os.environ.get("PROMOTION_MIN_SCORE", "0.72"))
    force = os.environ.get("FORCE_PROMOTE", "0") == "1"
    if latest_eval:
        print(
            f"[Promote] latest_eval model={latest_eval.get('model_tag')} "
            f"avg_score={latest_eval.get('avg_score', 0):.4f}"
        )
    else:
        print("[Promote] latest_eval not found")
    if not latest_eval and not force:
        raise RuntimeError("No evaluation result found. Run evaluation first, or set FORCE_PROMOTE=1.")
    avg_score = latest_eval["avg_score"] if latest_eval else 0.0
    if avg_score < min_score and not force:
        raise RuntimeError(
            f"Promotion blocked: avg_score={avg_score:.4f} is below PROMOTION_MIN_SCORE={min_score:.4f}. "
            "Set FORCE_PROMOTE=1 to override."
        )
    model_tag = os.environ.get("MODEL_SIZE", "default")

    db.insert_model_version(
        model_tag=model_tag,
        checkpoint_path=target,
        source_checkpoint=source,
        avg_score=avg_score,
        promoted=True,
        notes=f"Promoted at {time.strftime('%Y-%m-%d %H:%M:%S')}",
    )
    print(
        f"Promoted checkpoint: {source} -> {target} "
        f"(avg_score={avg_score:.4f}, min_score={min_score:.4f}, force={force})"
    )


if __name__ == "__main__":
    promote_best()
