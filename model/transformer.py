import math
import torch
import torch.nn as nn
from torch.nn import functional as F

class CausalSelfAttention(nn.Module):
    """
    自己注意機構（Self-Attention）: メッセージ（文脈）の集約を行うLLMの心臓部。
    Causal（因果的）マスクにより、未来の単語をカンニングしないようにする。
    """
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        # Q(Query), K(Key), V(Value) の3つのベクトルを一度に計算するための全結合層
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # 注意機構の結果を出力次元に戻す射影層
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        # 正規化やドロップアウト（過学習防止）の設定
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        # フラッシュアテンション（PyTorch 2.0以降の高効率Attention）が使える場合は使うフラグ
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            # 未来のトークンを見ないようにするための下三角行列マスク
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # Batch size, Sequence Length (Time), Channels (Embedding Dim)

        # q, k, v の計算
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        # マルチヘッドアテンションのため、ヘッド数ごとにテンソルを変形
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # アテンションの計算
        if self.flash:
            # 高速化されたPyTorch標準のAttention
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # 自前での計算 (Q * K^T / sqrt(d))
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            # 未来の単語へのアテンションを-infにしてマスク
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # 確率に応じてVを集計

        # マルチヘッドを元の次元に結合
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # 最終的な出力層とドロップアウト
        y = self.resid_dropout(self.c_proj(y))
        return y


class FeedForward(nn.Module):
    """
    位置ごとの情報を処理する2層のニューラルネットワーク（FFN）。
    Attentionが「他の単語との関係」を見るのに対し、FFNは「その単語の意味を深く考える」役割。
    """
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU() # ReLUに似た、より滑らかな活性化関数（BERTやGPTで標準）
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """
    Transformerの1つの層（ブロック）。
    LayerNorm -> Attention -> 残差接続 -> LayerNorm -> FFN -> 残差接続
    """
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = FeedForward(config)

    def forward(self, x):
        # 残差接続（x + ...）を使って、勾配消失を防ぎ深いネットワークを学習可能にする
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPTConfig:
    """LLMの各種ハイパーパラメータ（設計図）"""
    def __init__(self, vocab_size=32000, block_size=1024, n_layer=12, n_head=12, n_embd=768, dropout=0.0, bias=True):
        self.vocab_size = vocab_size # 単語辞書のサイズ（例：32000）
        self.block_size = block_size # 一度に読める最大文字数（コンテキスト長）
        self.n_layer = n_layer       # ブロック（層）の数
        self.n_head = n_head         # アテンションのヘッド数
        self.n_embd = n_embd         # ベクトルの次元数（思考の深さ）
        self.dropout = dropout       # 一部の神経を休ませて過学習を防ぐ割合
        self.bias = bias             # LinearやLayerNormでバイアス項を使うか


class GPT(nn.Module):
    """
    独自のLLM（Generative Pre-trained Transformer）本体。
    GPT-2に似たデコーダーのみのアーキテクチャ。
    """
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            # 単語IDをベクトルに変換する辞書（Token Embedding）
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            # 位置（何文字目か）をベクトルに変換する辞書（Positional Embedding）
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            # Transformer ブロックの積み重ね（n_layer個）
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            # 最終的な正規化
            ln_f = nn.LayerNorm(config.n_embd, bias=config.bias),
        ))
        
        # 最終的に次に来る単語の確率（vocab_size次元）を予測する層
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # 単語埋め込みと出力層の重みを共有する（Parameter Tieing - パラメータ数を削減し学習を安定させる）
        self.transformer.wte.weight = self.lm_head.weight

        # 重みの初期化（GPT-2の論文に基づく標準偏差調整）
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"入力シーケンス({t})が最大文脈長({self.config.block_size})を超えています"

        # 0, 1, 2... という位置の配列を作成
        pos = torch.arange(0, t, dtype=torch.long, device=device) 

        # トークン埋め込み + 位置埋め込み
        tok_emb = self.transformer.wte(idx) # (B, T, C)
        pos_emb = self.transformer.wpe(pos) # (T, C)
        x = self.transformer.drop(tok_emb + pos_emb)

        # Transformerの各層を通過
        for block in self.transformer.h:
            x = block(x)
            
        x = self.transformer.ln_f(x)

        if targets is not None:
            # 学習時：次に来る単語を予測し、正解(targets)との誤差(Loss)を計算
            logits = self.lm_head(x)
            # クロスエントロピー誤差を計算。平坦化して(B*T, vocab_size)にする必要がある
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # 推論時：最後のトークンの予測結果だけを返す（軽量化計算）
            logits = self.lm_head(x[:, [-1], :]) # (B, 1, vocab_size)
            loss = None

        return logits, loss

if __name__ == "__main__":
    # ちょっとした動作確認テスト
    print("Testing the LLM Architecture implementation...")
    # 超小型のLLM設定（デバッグや動作確認用。実際はもっと大きくする）
    config = GPTConfig(vocab_size=8000, block_size=128, n_layer=4, n_head=4, n_embd=128)
    model = GPT(config)
    
    # ダミーの入力データ (バッチサイズ2, 長さ10のシーケンス)
    dummy_x = torch.randint(0, 8000, (2, 10))
    dummy_y = torch.randint(0, 8000, (2, 10))
    
    # 順伝播のテスト
    logits, loss = model(dummy_x, dummy_y)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f} M")
    print(f"Logits shape (prediction for each token): {logits.shape}")
    print(f"Loss value: {loss.item():.4f}")
    print("LLM Architecture is beautifully constructed & running!")
