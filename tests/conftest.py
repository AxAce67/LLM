import os
import sys
import types
from pathlib import Path


# pytest実行時に、リポジトリルートをimportパスへ明示追加する。
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 一部モジュールが環境変数を前提にするため、テストでは安全なダミー値を設定。
os.environ.setdefault("DATABASE_URL", "postgresql://dummy:dummy@localhost:5432/dummy")

# CI軽量化のため、torch未導入環境では最小スタブを注入する。
# （今回のunit testsはGPU演算や学習を実行しない）
if "torch" not in sys.modules:
    try:
        __import__("torch")
    except Exception:
        torch_stub = types.SimpleNamespace(
            cuda=types.SimpleNamespace(is_available=lambda: False),
            backends=types.SimpleNamespace(
                mps=types.SimpleNamespace(is_available=lambda: False)
            ),
        )
        sys.modules["torch"] = torch_stub
