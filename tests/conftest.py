import os
import sys
from pathlib import Path


# pytest実行時に、リポジトリルートをimportパスへ明示追加する。
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 一部モジュールが環境変数を前提にするため、テストでは安全なダミー値を設定。
os.environ.setdefault("DATABASE_URL", "postgresql://dummy:dummy@localhost:5432/dummy")
