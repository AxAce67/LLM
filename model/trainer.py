import os
import time
import torch
import numpy as np
import sys

# ルートディレクトリのモジュールをインポートするためのパス設定
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.transformer import GPT, GPTConfig

class DataLoaderLite:
    """
    数GBに及ぶ巨大なバイナリデータ(train.binなど)からバッチサイズ分のデータを
    順番に（またはランダムに）切り出してPyTorchに渡す軽量かつ高速なデータローダー。
    """
    def __init__(self, data_path, B, T):
        self.B = B # Batch size (1回に処理する文章の数)
        self.T = T # Time / Sequence length (1つの文章の長さ・コンテキスト長)
        
        # memmapを使用して、数GBのファイルでもメモリに全展開せずにディスクから直接読み込む
        print(f"Loading data from {data_path}")
        self.data = np.memmap(data_path, dtype=np.uint16, mode='r')
        self.total_batches = len(self.data) // (B * T)
        self.current_position = 0
        print(f"Loaded {len(self.data):,} tokens. ({self.total_batches} batches available)")

    def next_batch(self, device):
        # データの終点に近づいたら最初に戻る (エポックの概念)
        if self.current_position + (self.B * self.T + 1) > len(self.data):
            self.current_position = 0
            
        # 連続したB*T個のトークンをX(入力)として取得
        buf = self.data[self.current_position : self.current_position + self.B * self.T + 1]
        
        # NumPy配列(uint16)からPyTorchテンソル(int64)へ変換して指定デバイス(GPU等)へ転送
        # 次の単語を予測するため、Xを1つずらしたものがY(正解)となる
        x = torch.tensor(buf[:-1], dtype=torch.long).view(self.B, self.T).to(device)
        y = torch.tensor(buf[1:], dtype=torch.long).view(self.B, self.T).to(device)
        
        # 次のバッチのために位置を進める
        self.current_position += self.B * self.T
        
        return x, y

def train_step(max_steps=50):
    """
    メインコントローラーから定期的に呼ばれる、指定ステップ数の学習を行う関数。
    途中でチェックポイント(重みデータ)を保存しながら進める。
    """
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATASET_DIR = os.path.join(BASE_DIR, "dataset")
    train_data_path = os.path.join(DATASET_DIR, "train.bin")
    
    CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    if not os.path.exists(train_data_path):
        print(f"[Warning] Training data not found at {train_data_path}. Please run preprocessor first.")
        return 0, 0.0 # epoch, loss
        
    # ハイパーパラメータの設定
    # ※ 本格的な学習時はもっと大きくする（VRAM容量と応相談）
    B = 4   # バッチサイズ
    T = 128 # コンテキスト長
    learning_rate = 6e-4
    vocab_size = 8000 # 今回は小規模な設定
    
    # デバイスの自動判別（MacならMPS、Windows/LinuxならCUDA、なければCPU）
    device = 'cpu'
    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'
            
    print(f"Starting training on device: {device}")
    
    # データローダーの準備
    train_loader = DataLoaderLite(train_data_path, B=B, T=T)
    if train_loader.total_batches == 0:
         print("[Warning] Not enough data to form a single batch.")
         return 0, 0.0

    # LLMモデルの生成
    config = GPTConfig(vocab_size=vocab_size, block_size=T, n_layer=4, n_head=4, n_embd=128)
    model = GPT(config)
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1)
    
    # 前回のチェックポイントがあれば読み込んで学習を再開する（レジューム機能）
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "ckpt_latest.pt")
    start_step = 0
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_step = checkpoint['step']
            print(f"Resumed from step {start_step}")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
        
    # PyTorch 2.0の高速化機能 (macのmpsでは動作が不安定な場合があるため除外)
    if hasattr(torch, 'compile') and device != 'mps' and sys.platform != 'win32':
        print("Compiling model using torch.compile() for speed up...")
        model = torch.compile(model)
        
    model.train()
    
    t0 = time.time()
    lossf = 0.0
    
    print(f"Running training loop for {max_steps} steps...")
    
    for step in range(max_steps):
        x, y = train_loader.next_batch(device)
        
        # 勾配の初期化
        optimizer.zero_grad(set_to_none=True)
        
        # 順伝播
        logits, loss = model(x, y)
        
        # 逆伝播（誤差計算）
        loss.backward()
        
        # 勾配クリッピング（学習の爆発・破綻を防ぐ安全装置）
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # モデルの重みを更新してAIを賢くする
        optimizer.step()
        
        lossf = loss.item()
        
        # 数十ステップごとにログ出力
        if step % 10 == 0 or step == max_steps - 1:
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            # 1ステップあたりの実行時間(ms)を計算して出力
            process_time_ms = (dt * 1000) if step == 0 else (dt * 1000) / 10 
            print(f"Step {start_step + step:5d} | Loss: {lossf:.4f} | Time: {process_time_ms:.2f}ms/step")
            
    # 指定ステップ終わったらチェックポイントを保存
    print(f"Saving checkpoint to {checkpoint_path}...")
    torch.save({
        'step': start_step + max_steps,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': lossf,
        'config': config # アーキテクチャ設定も保存しておく（ロード時に必要）
    }, checkpoint_path)
    
    print("Training cycle finished successfully.")
    
    # 現在の仮想エポック数（全体データを何週したか）と最終的なロスを返す
    current_epoch = (start_step + max_steps) // train_loader.total_batches if train_loader.total_batches > 0 else 0
    return current_epoch, lossf

if __name__ == "__main__":
    train_step(max_steps=100)
