import os
import time
import torch
import numpy as np
import sys
import math

# ルートディレクトリのモジュールをインポートするためのパス設定
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.transformer import GPT, GPTConfig
from model.train_config import load_train_config
from runtime.auto_tuner import detect_runtime_profile

_THREADS_TUNED = False


def _configure_torch_threads(cpu_threads: int):
    """
    Thread設定はプロセス初期化時に1回だけ行う。
    set_num_interop_threads は並列処理開始後に再設定すると RuntimeError になる。
    """
    global _THREADS_TUNED
    if _THREADS_TUNED:
        return
    torch.set_num_threads(cpu_threads)
    if hasattr(torch, "set_num_interop_threads"):
        try:
            torch.set_num_interop_threads(max(1, cpu_threads // 2))
        except RuntimeError:
            # 既に並列実行が始まっている場合は再設定不可のため無視
            pass
    _THREADS_TUNED = True


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


@torch.no_grad()
def estimate_val_loss(model, val_loader, device, eval_batches=20):
    if val_loader is None or val_loader.total_batches == 0:
        return None
    model_was_training = model.training
    model.eval()
    losses = []
    for _ in range(min(eval_batches, val_loader.total_batches)):
        x, y = val_loader.next_batch(device)
        _, loss = model(x, y)
        losses.append(loss.item())
    if model_was_training:
        model.train()
    return float(sum(losses) / max(1, len(losses)))

def train_step(max_steps=50):
    """
    メインコントローラーから定期的に呼ばれる、指定ステップ数の学習を行う関数。
    途中でチェックポイント(重みデータ)を保存しながら進める。
    """
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATASET_DIR = os.path.join(BASE_DIR, "dataset")
    train_data_path = os.path.join(DATASET_DIR, "train.bin")
    val_data_path = os.path.join(DATASET_DIR, "val.bin")
    
    CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    if not os.path.exists(train_data_path):
        print(f"[Warning] Training data not found at {train_data_path}. Please run preprocessor first.")
        return {"epoch": 0, "train_loss": 0.0, "val_loss": None, "best_val_loss": None, "steps_ran": 0}
        
    cfg = load_train_config()
    runtime_profile = detect_runtime_profile()
    seed = int(os.environ.get("TRAIN_SEED", "42"))
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    B = cfg.batch_size
    T = cfg.seq_len
    learning_rate = cfg.learning_rate
    vocab_size = cfg.vocab_size

    # CPU利用率改善: スレッド数設定（プロセスで1回のみ）
    _configure_torch_threads(cfg.cpu_threads)
    
    # デバイスの自動判別（MacならMPS、Windows/LinuxならCUDA、なければCPU）
    device = 'cpu'
    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'
            
    print(f"Starting training on device: {device}")
    print(
        f"[AutoTune] cores={runtime_profile['cpu_cores']} ram={runtime_profile['available_ram_gb']}/{runtime_profile['total_ram_gb']}GB "
        f"model={runtime_profile['model_size']} B={B} T={T} threads={cfg.cpu_threads}"
    )
    
    # データローダーの準備
    train_loader = DataLoaderLite(train_data_path, B=B, T=T)
    if train_loader.total_batches == 0:
         print("[Warning] Not enough data to form a single batch.")
         return {"epoch": 0, "train_loss": 0.0, "val_loss": None, "best_val_loss": None, "steps_ran": 0}
    val_loader = None
    if os.path.exists(val_data_path):
        val_loader = DataLoaderLite(val_data_path, B=B, T=T)

    config = GPTConfig(
        vocab_size=vocab_size,
        block_size=T,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_embd=cfg.n_embd,
        dropout=cfg.dropout,
        bias=cfg.bias
    )
    model = GPT(config)
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=cfg.weight_decay)
    
    # 前回のチェックポイントがあれば読み込んで学習を再開する（レジューム機能）
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "ckpt_latest.pt")
    best_checkpoint_path = os.path.join(CHECKPOINT_DIR, "ckpt_best.pt")
    start_step = 0
    best_val_loss = float("inf")
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            # チェックポイントのモデル設定を復元（アーキテクチャが変わった場合はスキップ）
            saved_config = checkpoint.get('config')
            if saved_config and saved_config.n_embd == config.n_embd and saved_config.n_layer == config.n_layer:
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_step = checkpoint['step']
                best_val_loss = checkpoint.get("best_val_loss", float("inf"))
                print(f"Resumed from step {start_step}")
            else:
                print("[Info] Architecture changed. Starting from scratch (old checkpoint ignored).")
        except Exception as e:
            print(f"Failed to load checkpoint (starting fresh): {e}")
        
    # PyTorch 2.0の高速化機能 (macのmpsでは動作が不安定な場合があるため除外)
    enable_compile = os.environ.get("ENABLE_TORCH_COMPILE", "0") == "1"
    if enable_compile and hasattr(torch, 'compile') and device != 'mps' and sys.platform != 'win32':
        print("Compiling model using torch.compile() for speed up...")
        model = torch.compile(model)
        
    model.train()
    
    t0 = time.time()
    lossf = 0.0
    
    print(f"Running training loop for {max_steps} steps...")
    
    use_amp = device == "cuda" and os.environ.get("TRAIN_USE_AMP", "1") == "1"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    eval_every = max(1, int(os.environ.get("VAL_EVAL_EVERY", "25")))
    eval_batches = max(1, int(os.environ.get("VAL_EVAL_BATCHES", "20")))
    early_stopping_patience = max(1, int(os.environ.get("EARLY_STOPPING_PATIENCE", "6")))
    stale_count = 0
    val_loss = None

    def get_lr(step_idx: int) -> float:
        warmup = cfg.warmup_steps
        if step_idx < warmup:
            return learning_rate * (step_idx + 1) / warmup
        progress = (step_idx - warmup) / max(1, max_steps - warmup)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return learning_rate * (cfg.min_lr_ratio + (1.0 - cfg.min_lr_ratio) * cosine)

    actual_steps = 0
    for step in range(max_steps):
        actual_steps += 1
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for _ in range(cfg.grad_accum_steps):
            x, y = train_loader.next_batch(device)

            with torch.cuda.amp.autocast(enabled=use_amp):
                _, loss = model(x, y)
                loss = loss / cfg.grad_accum_steps

            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            accum_loss += loss.item()

        for group in optimizer.param_groups:
            group["lr"] = get_lr(start_step + step)

        if use_amp:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        lossf = accum_loss
        
        # 数十ステップごとにログ出力
        if step % 10 == 0 or step == max_steps - 1:
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            # 1ステップあたりの実行時間(ms)を計算して出力
            process_time_ms = (dt * 1000) if step == 0 else (dt * 1000) / 10 
            print(f"Step {start_step + step:5d} | Loss: {lossf:.4f} | LR: {optimizer.param_groups[0]['lr']:.6e} | Time: {process_time_ms:.2f}ms/step")

        # 検証ロス評価とベスト更新
        if (step + 1) % eval_every == 0 or step == max_steps - 1:
            val_loss = estimate_val_loss(model, val_loader, device, eval_batches=eval_batches)
            if val_loss is not None:
                improved = val_loss < best_val_loss
                print(f"[Validation] step={start_step + step:5d} val_loss={val_loss:.4f} best={best_val_loss if best_val_loss != float('inf') else 'inf'}")
                if improved:
                    best_val_loss = val_loss
                    stale_count = 0
                    torch.save({
                        'step': start_step + step + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': lossf,
                        'val_loss': val_loss,
                        'best_val_loss': best_val_loss,
                        'config': config
                    }, best_checkpoint_path)
                    print(f"[Validation] New best model saved: {best_checkpoint_path}")
                else:
                    stale_count += 1
                    if stale_count >= early_stopping_patience:
                        print(f"[EarlyStopping] No val improvement for {stale_count} evals. Stopping early.")
                        break
            
    # 指定ステップ終わったらチェックポイントを保存
    print(f"Saving checkpoint to {checkpoint_path}...")
    torch.save({
        'step': start_step + actual_steps,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': lossf,
        'val_loss': val_loss,
        'best_val_loss': best_val_loss,
        'config': config # アーキテクチャ設定も保存しておく（ロード時に必要）
    }, checkpoint_path)
    
    print("Training cycle finished successfully.")
    
    # 現在の仮想エポック数（全体データを何週したか）と最終的なロスを返す
    current_epoch = (start_step + actual_steps) // train_loader.total_batches if train_loader.total_batches > 0 else 0
    return {
        "epoch": current_epoch,
        "train_loss": float(lossf),
        "val_loss": float(val_loss) if val_loss is not None else None,
        "best_val_loss": float(best_val_loss) if best_val_loss != float("inf") else None,
        "steps_ran": actual_steps,
    }

if __name__ == "__main__":
    train_step(max_steps=100)
