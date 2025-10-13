"""Primary strategy base class for Two-Tier Strategy architecture."""

from abc import ABC, abstractmethod
import pandas as pd


class PrimaryStrategyBase(ABC):
    """1次戦略の抽象基底クラス

    指値価格計算とリターン計算を担当する。
    各具体的な戦略はこのクラスを継承して実装する。
    """

    def __init__(self, params: dict):
        """Initialize primary strategy.

        Args:
            params: 戦略パラメータ辞書
                - execution_mode: "chase" または "one_candle"
                - その他、戦略固有のパラメータ
        """
        self.params = params
        self.execution_mode = params.get("execution_mode", "one_candle")

    @abstractmethod
    def calculate_prices(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """指値価格計算（buy_price, sell_priceカラムを追加）

        ML有効/無効に関わらず常に実行され、計算された指値価格は
        注文執行とML学習の両方で使用される。

        Args:
            dataframe: OHLCデータ

        Returns:
            buy_price, sell_priceが追加されたDataFrame
        """
        pass

    @abstractmethod
    def calculate_returns(self, dataframe: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        """指値戦略のリターン計算（ML学習用ラベル生成）

        約定シミュレーション方法はexecution_mode設定に基づいて切り替わる：
        - "chase": エントリー/エグジット両方で約定するまで追いかける方式
        - "one_candle": エントリーは1足限定、約定した場合のみリターン計算

        Args:
            dataframe: 価格計算済みのDataFrame (buy_price, sell_priceカラムが必要)

        Returns:
            (buy_return, sell_return): 買い/売りそれぞれの理論リターン
                - 買いと売りで独立したリターンを計算（両建て対応）
                - 計算されたリターンは、ML学習時にラベル化される（リターン > 0 で成功）

        Warning:
            このロジックは取引の根幹となるため、ミスがあると大きな損失に繋がる。
            必ず包括的なテストコードで検証すること。
        """
        pass
