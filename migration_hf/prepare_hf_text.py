import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_collector.db_manager import DBManager


def main():
    out_path = os.environ.get("HF_TRAIN_TEXT_PATH", "dataset/hf/train.txt")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    db = DBManager()
    count = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for row in db.stream_crawled_documents(batch_size=1000):
            if not row.get("allowed_for_training", True):
                continue
            text = (row.get("content") or "").strip()
            if len(text) < 80:
                continue
            f.write(text.replace("\n", " ").strip() + "\n")
            count += 1

    print(f"saved {count} rows -> {out_path}")


if __name__ == "__main__":
    main()
