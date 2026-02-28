import os
import torch
import sys

# ルートディレクトリのモジュールをインポートするためのパス設定
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model.transformer import GPT, GPTConfig
import sentencepiece as spm

def load_tokenizer():
    """
    データ前処理で作成したSentencePieceモデルを読み込む
    """
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(BASE_DIR, "dataset", "tokenizer", "llm_tokenizer.model"),
        os.path.join(BASE_DIR, "dataset", "tokenizer.model"),
    ]
    spm_path = next((path for path in candidates if os.path.exists(path)), None)
    if spm_path is None:
        raise FileNotFoundError(
            "Tokenizer not found. Expected dataset/tokenizer/llm_tokenizer.model. Please run preprocessor first."
        )
        
    sp = spm.SentencePieceProcessor()
    sp.load(spm_path)
    return sp

def load_model(device='cpu'):
    """
    保存されたチェックポイントから学習済みの重みと設定を復元する
    """
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    use_production = os.environ.get("USE_PRODUCTION_MODEL", "1") == "1"
    production_path = os.path.join(BASE_DIR, "checkpoints", "ckpt_production.pt")
    latest_path = os.path.join(BASE_DIR, "checkpoints", "ckpt_latest.pt")
    checkpoint_path = production_path if use_production and os.path.exists(production_path) else latest_path
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}. Please train the model first.")
        
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # 保存時のコンフィグ(アーキテクチャ構造の設定)を復元
    config = checkpoint['config']
    
    # 復元した設定をもとに空のモデルを組み立てる
    model = GPT(config)
    
    # コンパイル済みのモデルから保存した場合の接頭辞 "_orig_mod." を取り除く
    state_dict = checkpoint['model_state_dict']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            
    # 空のモデルに学習済みの重みを注ぎ込む
    model.load_state_dict(state_dict)

    # CPU配布向け: 動的量子化 (Linear層)
    enable_quant = os.environ.get("ENABLE_CPU_QUANTIZATION", "0") == "1"
    if device == "cpu" and enable_quant:
        try:
            model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
            print("Enabled dynamic int8 quantization for CPU inference.")
        except Exception as e:
            print(f"[Inference] Quantization skipped: {e}")
    
    # 推論(おしゃべり)モードに設定
    model.eval()
    model.to(device)
    
    return model, config

@torch.no_grad()
def generate_text(
    model,
    sp,
    prompt,
    max_new_tokens=50,
    temperature=0.8,
    top_k=50,
    top_p=0.95,
    repetition_penalty=1.08,
    device='cpu'
):
    """
    与えられた文章(prompt)の続きを生成する
    """
    # 1. ユーザーの入力テキストをトークンIDのリスト(数字の列)に変換
    idx = sp.EncodeAsIds(prompt)
    if not idx:
        idx = [sp.bos_id()] if sp.bos_id() != -1 else [1]
        
    # PyTorchのテンソル(1 x T)に変換してデバイスへ送る
    x = torch.tensor(idx, dtype=torch.long, device=device).unsqueeze(0)
    
    # 生成ループ
    print(f"\n[Prompt]: {prompt}")
    print("[Generated]: ", end="", flush=True)
    
    for _ in range(max_new_tokens):
        # 現在のコンテキスト長がモデルの最大長を超えないようにクリップ
        cond_idx = x if x.size(1) <= model.config.block_size else x[:, -model.config.block_size:]
        
        # モデルに現在の文脈を渡し、次に来る単語の確率分布(logits)をもらう
        logits, _ = model(cond_idx)
        
        # 最後のステップ(次に来る単語)の予測値だけを取り出す (1, vocab_size)
        logits = logits[:, -1, :] / temperature

        # SentencePieceのunkトークン(⁇)は可読性を大きく下げるため、
        # 推論時は候補から除外する。
        unk_id = sp.unk_id() if hasattr(sp, "unk_id") else -1
        if isinstance(unk_id, int) and 0 <= unk_id < logits.size(-1):
            logits[:, unk_id] = -float("Inf")
        
        # 反復抑制
        if repetition_penalty > 1.0 and x.size(1) > 0:
            recent_tokens = set(x[0, -128:].tolist())
            for tok in recent_tokens:
                logits[:, tok] /= repetition_penalty

        # Top-Kサンプリング(確率の低い突拍子もない単語を除外して安定させる)
        if top_k is not None:
             v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
             logits[logits < v[:, [-1]]] = -float('Inf')

        # Nucleus (Top-p) sampling
        if top_p is not None and 0.0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits = logits.masked_fill(indices_to_remove, -float('Inf'))
             
        # 確率分布をSoftmaxで正規化
        probs = torch.nn.functional.softmax(logits, dim=-1)
        
        # 分布に従って次の単語をガチャ(サンプリング)で選ぶ
        idx_next = torch.multinomial(probs, num_samples=1)
        
        # 選ばれた単語のIDを元の文脈の後ろにくっつける
        x = torch.cat((x, idx_next), dim=1)
        
        # 新しく生成された1文字(1トークン)をデコードして画面に逐次表示
        next_token_str = sp.DecodeIds(idx_next[0].tolist())
        print(next_token_str, end="", flush=True)
        
    print("\n")
    
    # 最終的に生成された文字列全体を返す
    final_output = sp.DecodeIds(x[0].tolist())
    return final_output

if __name__ == "__main__":
    device = 'cpu'
    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'
        
    print(f"Using device: {device}")
    
    try:
        sp = load_tokenizer()
        model, config = load_model(device=device)
        
        test_prompts = [
            "Pythonでプログラムを書くときは、",
            "最新のAI技術について",
            "今日はとても天気が"
        ]
        
        for p in test_prompts:
            generate_text(model, sp, prompt=p, max_new_tokens=60, temperature=0.7, top_k=40, device=device)
            print("-" * 50)
            
    except Exception as e:
        print(f"Error during inference: {e}")
