import os
import urllib.request
import bz2
import re

DATASET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset", "wikipedia")

def _extract_text_from_xml(xml_text: str) -> list[str]:
    """
    Wikipedia XMLダンプ内の <text> タグからプレーンテキストを簡易的に抽出する。
    本格運用では wikiextractor ライブラリ等を使用するが、
    依存関係を最小限に抑えるため、正規表現による簡易実装とする。
    """
    articles = []
    # <text> タグのコンテンツを抽出
    text_blocks = re.findall(r'<text[^>]*>(.*?)</text>', xml_text, re.DOTALL)
    for raw_text in text_blocks:
        # Wikiマークアップの代表的なタグを除去
        cleaned = re.sub(r'\[\[ファイル:[^\]]*\]\]', '', raw_text)
        cleaned = re.sub(r'\[\[File:[^\]]*\]\]', '', cleaned)
        cleaned = re.sub(r'\[\[Image:[^\]]*\]\]', '', cleaned)
        cleaned = re.sub(r'\{\{[^}]*\}\}', '', cleaned)  # {{テンプレート}} を除去
        cleaned = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]+)\]\]', r'\1', cleaned)  # [[リンク|表示文字]] → 表示文字
        cleaned = re.sub(r'<[^>]+>', '', cleaned)  # HTMLタグ除去
        cleaned = re.sub(r'={2,}.*?={2,}', '', cleaned)  # == 見出し == 除去
        cleaned = re.sub(r'\|.*', '', cleaned, flags=re.MULTILINE)  # テーブルの残骸
        cleaned = re.sub(r'[ \t]+', ' ', cleaned)
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        cleaned = cleaned.strip()
        if len(cleaned) > 200:
            articles.append(cleaned)
    return articles

def download_wikipedia_dump(language: str = "ja") -> int:
    """
    Wikipediaの公式ダンプ（記事の塊）をダウンロードし、ローカルのtxtファイルに展開する。
    注意：数GBのファイルを扱うため、クラウドDB（Supabase）には保存せず
    各PC（コンテナのマウントボリューム）に直接保管するハイブリッド戦略を採用している。

    Returns: 新たに保存したファイル数（記事数の概算）
    """
    os.makedirs(DATASET_DIR, exist_ok=True)

    # 既に展開済みテキストがある場合はスキップ
    extracted_files = []
    for _, _, files in os.walk(DATASET_DIR):
        for f in files:
            if f.startswith("wiki_") and f.endswith(".txt"):
                extracted_files.append(f)
    if extracted_files:
        print(f"[WikiDL] Extracted Wikipedia text already exists ({len(extracted_files)} files). Skipping.")
        return len(extracted_files)

    # 最新のWikipedia圧縮ダンプのURL（全記事 / 単一bz2ファイル）
    dump_url = f"https://dumps.wikimedia.org/{language}wiki/latest/{language}wiki-latest-pages-articles.xml.bz2"
    dump_path = os.path.join(DATASET_DIR, f"{language}wiki-latest-pages-articles.xml.bz2")

    if not os.path.exists(dump_path):
        print(f"[WikiDL] Downloading Wikipedia dump from: {dump_url}")
        print("[WikiDL] This may take several minutes to hours depending on network speed...")

        try:
            # progress表示付きダウンロード
            def reporthook(count, block_size, total_size):
                if total_size > 0 and count % 500 == 0:
                    mb_done = count * block_size / (1024 * 1024)
                    mb_total = total_size / (1024 * 1024)
                    print(f"[WikiDL] Progress: {mb_done:.1f} / {mb_total:.1f} MB")

            urllib.request.urlretrieve(dump_url, dump_path, reporthook=reporthook)
            print(f"[WikiDL] Download complete: {dump_path}")

        except Exception as e:
            print(f"[WikiDL] Download failed: {e}")
            return 0
    else:
        print(f"[WikiDL] Found existing dump file. Skipping download and continuing extraction: {dump_path}")

    # bz2ストリームを分割で展開しながらテキスト抽出・保存する
    print("[WikiDL] Extracting and parsing articles...")
    saved_count = 0
    chunk_size = 1024 * 1024 * 5  # 5MB ずつ処理

    try:
        xml_buffer = ""
        with bz2.open(dump_path, "rt", encoding="utf-8") as bz_file:
            while True:
                chunk = bz_file.read(chunk_size)
                if not chunk:
                    break
                xml_buffer += chunk

                # バッファが大きくなったら記事を抽出して保存
                if len(xml_buffer) > chunk_size * 2:
                    articles = _extract_text_from_xml(xml_buffer)
                    for i, article in enumerate(articles):
                        fname = os.path.join(DATASET_DIR, f"wiki_{saved_count + i:07d}.txt")
                        with open(fname, "w", encoding="utf-8") as f:
                            f.write(article)
                    saved_count += len(articles)
                    # バッファをリセット（末尾側だけ残して前後の記事断片を防ぐ）
                    xml_buffer = xml_buffer[-5000:]
                    print(f"[WikiDL] Saved {saved_count} articles so far...")

            # 残りのバッファも処理
            articles = _extract_text_from_xml(xml_buffer)
            for i, article in enumerate(articles):
                fname = os.path.join(DATASET_DIR, f"wiki_{saved_count + i:07d}.txt")
                with open(fname, "w", encoding="utf-8") as f:
                    f.write(article)
            saved_count += len(articles)

    except Exception as e:
        print(f"[WikiDL] Extraction error: {e}")

    # ダウンロードした .bz2 圧縮ファイルを後始末（容量節約のため）
    if os.path.exists(dump_path):
        os.remove(dump_path)
        print(f"[WikiDL] Removed compressed dump file to save disk space.")

    print(f"[WikiDL] Complete! Total articles extracted: {saved_count}")
    return saved_count


if __name__ == "__main__":
    download_wikipedia_dump("ja")
