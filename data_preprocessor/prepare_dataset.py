import os
import sys
import numpy as np
import sentencepiece as spm
from tqdm import tqdm

# ルートパス設定
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_collector.db_manager import DBManager
from data_preprocessor.tokenizer_builder import TokenizerBuilder, TOKENIZER_DIR, WIKI_DIR

DATASET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset")

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

    # 全データをオンメモリでの処理は危険だが、初回の数十〜数百MBレベルであればリストで保持可能
    # より巨大なデータ（GB単位）になる場合は、逐次的にbinファイルへ追記書き込みする実装に切り替える
    all_tokens = []
    
    db_manager = DBManager()
    
    # 1. DBからテキストロード
    print("Loading text from Supabase DB...")
    try:
        response = db_manager.supabase.table("crawled_data").select("content").execute()
        for row in response.data:
            text = row.get("content", "").strip()
            if text:
                # 終端文字(</s>等)を入れて文の区切りを教える
                tokens = sp.encode_as_ids(text)
                all_tokens.extend(tokens)
                all_tokens.append(sp.eos_id())
    except Exception as e:
        print(f"Error reading DB: {e}")

    # 2. Wikipediaからローカルロード
    print("Loading text from Local Wikipedia data...")
    if os.path.exists(WIKI_DIR):
        for root, dirs, files in os.walk(WIKI_DIR):
            for file in files:
                if file.endswith(".txt"):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            text = f.read().strip()
                            if text:
                                tokens = sp.encode_as_ids(text)
                                all_tokens.extend(tokens)
                                all_tokens.append(sp.eos_id())
                    except Exception as e:
                        print(f"Error reading file {file}: {e}")

    total_tokens = len(all_tokens)
    print(f"Total tokens generated: {total_tokens:,}")
    
    if total_tokens == 0:
        print("[Warning] No data found to tokenize. Returning.")
        return False

    # 訓練(train)データと検証(val)データに分割
    val_size = int(total_tokens * val_ratio)
    train_size = total_tokens - val_size
    
    train_tokens = all_tokens[:train_size]
    val_tokens = all_tokens[train_size:]

    # バイナリ保存（np.uint16: 最大値65535なのでvocab_size=32000にはピッタリでサイズを半減できる）
    # 大規模言語モデルでは定番のテクニック
    train_bin_path = os.path.join(DATASET_DIR, "train.bin")
    val_bin_path = os.path.join(DATASET_DIR, "val.bin")
    
    print("Saving to binary files (train.bin, val.bin) as uint16...")
    train_array = np.array(train_tokens, dtype=np.uint16)
    train_array.tofile(train_bin_path)
    
    val_array = np.array(val_tokens, dtype=np.uint16)
    val_array.tofile(val_bin_path)

    print(f"Saved {len(train_tokens):,} tokens to {train_bin_path}")
    print(f"Saved {len(val_tokens):,} tokens to {val_bin_path}")
    print("Dataset preparation completed successfully!")
    return True

if __name__ == "__main__":
    prepare_dataset(vocab_size=8000) # テスト実行用
