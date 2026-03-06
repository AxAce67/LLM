import os
import sentencepiece as spm
import tempfile
import sys

# ルートディレクトリのモジュールをインポートするためのパス設定
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_collector.db_manager import DBManager

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
TOKENIZER_DIR = os.path.join(DATASET_DIR, "tokenizer")
WIKI_DIR = os.path.join(DATASET_DIR, "wikipedia")

class TokenizerBuilder:
    def __init__(self, vocab_size=32000):
        self.vocab_size = vocab_size
        self.db_manager = DBManager()
        os.makedirs(TOKENIZER_DIR, exist_ok=True)
        self.model_prefix = os.path.join(TOKENIZER_DIR, "llm_tokenizer")

    def _export_db_text(self, output_file_path):
        """Supabaseに貯まったWebクロールデータを抽出し、学習用の一時ファイルに書き出す"""
        print("Extracting text data from DB (streaming)...")
        try:
            doc_count = 0
            written_chars = 0
            with open(output_file_path, "a", encoding="utf-8") as f:
                for text in self.db_manager.stream_crawled_contents(batch_size=1000):
                    f.write(text + "\n\n")
                    written_chars += len(text)
                    doc_count += 1
            print(f"Extracted {doc_count} documents ({written_chars} characters) from DB.")
        except Exception as e:
            print(f"[Error] Failed to extract from DB: {e}")

    def _append_local_wiki_text(self, output_file_path):
        """ローカルにDLされたWikipediaの大容量テキストファイル群から学習用テキストを追加する"""
        print("Looking for local Wikipedia data...")
        if not os.path.exists(WIKI_DIR):
            print(f"Wikipedia directory not found: {WIKI_DIR}")
            return

        written_chars = 0
        with open(output_file_path, "a", encoding="utf-8") as out_f:
            for root, dirs, files in os.walk(WIKI_DIR):
                for file in files:
                    if file.endswith(".txt"):
                        file_path = os.path.join(root, file)
                        try:
                            with open(file_path, "r", encoding="utf-8") as in_f:
                                for line in in_f:
                                    out_f.write(line)
                                    written_chars += len(line)
                            out_f.write("\n\n")
                        except Exception as e:
                            print(f"Error reading wiki file {file}: {e}")
        
        print(f"Appended {written_chars} characters from local Wikipedia data.")

    def build_tokenizer(self):
        """
        データベースおよびローカルの全テキストデータをかき集め、
        SentencePieceを用いてBPEベースのトークナイザモデルを学習・生成する。
        """
        print(f"Building tokenizer with vocabulary size: {self.vocab_size}...")
        
        # 学習用の一時テキストファイルを作成（数GBになる可能性があるためテンポラリに作成して学習後に削除）
        fd, temp_path = tempfile.mkstemp(suffix=".txt")
        os.close(fd) # ファイルディスクリプタを閉じて通常のopenを使えるようにする
        
        try:
            # 1. DBから収集データを書き出し
            self._export_db_text(temp_path)
            
            # 2. ローカルからWikipediaデータを追記
            self._append_local_wiki_text(temp_path)
            
            # File size check
            size_mb = os.path.getsize(temp_path) / (1024 * 1024)
            print(f"Total training text size: {size_mb:.2f} MB")
            
            if size_mb < 0.1:
                print("[Warning] Not enough training data! Tokenizer quality will be poor. Assuming dummy run.")
                
            # 3. SentencePiece にかけて単語辞書を学習
            # --input: 元テキスト, --model_prefix: 出力ファイル名のプレフィックス
            # --vocab_size: 語彙数(32000), --model_type: bpe (Byte-Pair Encoding) もしくは unigram
            # --character_coverage: カバー率 (日本語等は0.9995等にする)
            print("Training SentencePiece model... (This may take a while depending on data size)")
            sp_threads = max(1, int(os.environ.get("SPM_NUM_THREADS", str(max(1, (os.cpu_count() or 2) - 1)))))
            spm.SentencePieceTrainer.train(
                input=temp_path,
                model_prefix=self.model_prefix,
                vocab_size=self.vocab_size,
                model_type="bpe",
                character_coverage=0.9995,
                pad_id=0,
                unk_id=1,
                bos_id=2, # Begin of sentence
                eos_id=3, # End of sentence
                num_threads=sp_threads,
            )
            
            print(f"Tokenizer training complete! Model saved to:\n  - {self.model_prefix}.model\n  - {self.model_prefix}.vocab")
            return f"{self.model_prefix}.model"
            
        finally:
            # 使い終わった学習用巨大テキストファイルは削除
            if os.path.exists(temp_path):
                os.remove(temp_path)

if __name__ == "__main__":
    builder = TokenizerBuilder(vocab_size=8000) # テスト時は時間を節約するため語彙数少なめ
    builder.build_tokenizer()
