import os
import sys
import sentencepiece as spm
import random
from array import array
import time

# ルートパス設定
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_collector.db_manager import DBManager
from data_preprocessor.tokenizer_builder import TokenizerBuilder, TOKENIZER_DIR, WIKI_DIR
from data_preprocessor.quality_filter import normalize_text, is_low_quality, is_duplicate

DATASET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset")

def _source_weight(source_type: str) -> float:
    key = (source_type or "web").lower()
    defaults = {
        "wikipedia": 1.4,
        "arxiv": 1.35,
        "docs": 1.3,
        "news": 1.25,
        "rss": 1.1,
        "web": 1.0,
    }
    env_key = f"SOURCE_WEIGHT_{key.upper()}"
    return max(0.0, float(os.environ.get(env_key, str(defaults.get(key, 1.0)))))


def _quality_weight(score: float) -> float:
    floor = float(os.environ.get("QUALITY_WEIGHT_FLOOR", "0.35"))
    boost = float(os.environ.get("QUALITY_WEIGHT_BOOST", "0.9"))
    s = max(0.0, min(1.0, float(score)))
    return max(0.0, floor + (boost * s))


def prepare_dataset(vocab_size=32000, val_ratio=0.05):
    """
    収集されたテキストをトークナイズし、巨大なIDの配列に変換してバイナリファイル(train.bin, val.bin)に保存する。
    これにより、PyTorch学習時のロード速度とメモリ効率が劇的に向上する（numpy memmapを使用予定のため）。
    """
    print("Starting dataset preparation (tokenization & binary encoding)...")
    
    model_path = os.path.join(TOKENIZER_DIR, "llm_tokenizer.model")
    
    # トークナイザモデルが存在しない場合は学習・生成する
    if not os.path.exists(model_path):
        print("Tokenizer model not found. Building a new one...")
        builder = TokenizerBuilder(vocab_size=vocab_size)
        model_path = builder.build_tokenizer()

    sp = spm.SentencePieceProcessor()
    sp.load(model_path)
    
    print(f"Loaded tokenizer '{model_path}' successfully (vocab_size: {sp.get_piece_size()})")

    db_manager = DBManager()
    train_bin_path = os.path.join(DATASET_DIR, "train.bin")
    val_bin_path = os.path.join(DATASET_DIR, "val.bin")

    if sp.get_piece_size() > 65535:
        raise ValueError("Tokenizer vocab is too large for uint16. Use vocab_size <= 65535.")

    print("Tokenizing data in streaming mode and writing train.bin / val.bin...")
    rng = random.Random(42)
    token_buffer_train = array("H")
    token_buffer_val = array("H")
    flush_threshold = 200000
    train_count = 0
    val_count = 0
    total_docs = 0
    expanded_docs = 0
    filtered_docs = 0
    duplicate_docs = 0
    seen_hashes = set()
    source_doc_counts = {}

    open(train_bin_path, "wb").close()
    open(val_bin_path, "wb").close()

    def flush_buffers(force=False):
        nonlocal token_buffer_train, token_buffer_val
        if force or len(token_buffer_train) >= flush_threshold:
            with open(train_bin_path, "ab") as f_train:
                token_buffer_train.tofile(f_train)
            token_buffer_train = array("H")
        if force or len(token_buffer_val) >= flush_threshold:
            with open(val_bin_path, "ab") as f_val:
                token_buffer_val.tofile(f_val)
            token_buffer_val = array("H")

    blocked_docs = 0

    def write_document_tokens(text: str, source_type: str = "web", quality: float = 0.5, allowed_for_training: bool = True):
        nonlocal train_count, val_count, total_docs, expanded_docs, filtered_docs, duplicate_docs
        nonlocal blocked_docs
        if not allowed_for_training:
            blocked_docs += 1
            return
        cleaned = normalize_text(text)
        if is_low_quality(cleaned, min_chars=int(os.environ.get("MIN_TRAIN_CHARS", "120"))):
            filtered_docs += 1
            return
        if os.environ.get("ENABLE_TEXT_DEDUP", "1") == "1" and is_duplicate(cleaned, seen_hashes):
            duplicate_docs += 1
            return

        tokens = sp.encode_as_ids(cleaned)
        if not tokens:
            return
        tokens.append(sp.eos_id())
        source_doc_counts[source_type] = source_doc_counts.get(source_type, 0) + 1

        weight = _source_weight(source_type) * _quality_weight(quality)
        copies_int = int(weight)
        copies = copies_int
        if rng.random() < (weight - copies_int):
            copies += 1
        if copies <= 0:
            return

        is_val = rng.random() < val_ratio
        for _ in range(copies):
            if is_val:
                token_buffer_val.extend(tokens)
                val_count += len(tokens)
            else:
                token_buffer_train.extend(tokens)
                train_count += len(tokens)
            expanded_docs += 1
        total_docs += 1
        flush_buffers()

    print("Loading text from DB...")
    try:
        for row in db_manager.stream_crawled_documents(batch_size=1000):
            write_document_tokens(
                row.get("content", ""),
                source_type=row.get("source_type", "web"),
                quality=row.get("quality_score", 0.5),
                allowed_for_training=row.get("allowed_for_training", True),
            )
    except Exception as e:
        print(f"Error reading DB: {e}")

    print("Loading text from local Wikipedia data...")
    if os.path.exists(WIKI_DIR):
        for root, dirs, files in os.walk(WIKI_DIR):
            for file in files:
                if file.endswith(".txt"):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            text = f.read().strip()
                            if text:
                                write_document_tokens(text, source_type="wikipedia", quality=0.95, allowed_for_training=True)
                    except Exception as e:
                        print(f"Error reading file {file}: {e}")

    flush_buffers(force=True)
    total_tokens = train_count + val_count
    print(
        f"Processed {total_docs:,} docs (expanded={expanded_docs:,}, blocked={blocked_docs:,}, filtered={filtered_docs:,}, duplicates={duplicate_docs:,}). "
        f"Total tokens: {total_tokens:,}"
    )

    if total_tokens == 0:
        print("[Warning] No data found to tokenize. Returning.")
        return False

    print(f"Saved {train_count:,} tokens to {train_bin_path}")
    print(f"Saved {val_count:,} tokens to {val_bin_path}")
    print("Dataset preparation completed successfully!")

    # データセット版をDBに記録（学習再現性）
    try:
        dataset_tag = f"dataset-{int(time.time())}"
        db_manager.insert_dataset_version(
            dataset_tag=dataset_tag,
            train_tokens=train_count,
            val_tokens=val_count,
            total_docs=total_docs,
            blocked_docs=blocked_docs,
            filtered_docs=filtered_docs,
            duplicate_docs=duplicate_docs,
            source_breakdown=source_doc_counts,
            metadata_json={
                "vocab_size": sp.get_piece_size(),
                "val_ratio": val_ratio,
                "dedup": os.environ.get("ENABLE_TEXT_DEDUP", "1"),
            },
        )
        print(f"Dataset version recorded: {dataset_tag}")
    except Exception as e:
        print(f"[Warning] Failed to record dataset version: {e}")

    return {
        "ok": True,
        "train_tokens": train_count,
        "val_tokens": val_count,
        "total_docs": total_docs,
        "blocked_docs": blocked_docs,
        "filtered_docs": filtered_docs,
        "duplicate_docs": duplicate_docs,
    }

if __name__ == "__main__":
    prepare_dataset(vocab_size=8000) # テスト実行用
