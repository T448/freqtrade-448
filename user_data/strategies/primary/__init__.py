"""Primary strategies package for Two-Tier Strategy architecture."""

import sys
from pathlib import Path

# user_dataディレクトリをパスに追加
user_data_path = Path(__file__).parent.parent.parent
if str(user_data_path) not in sys.path:
    sys.path.insert(0, str(user_data_path))

from strategies.primary.base import PrimaryStrategyBase

__all__ = ["PrimaryStrategyBase"]
