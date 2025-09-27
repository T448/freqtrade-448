"""
ATRリターン計算システム

richmanbtcチュートリアルに基づくATR戦略の理論リターン計算機能を提供します。

Requirements implemented:
- 1.7: richmanbtcチュートリアルに準拠した決済方法の使用
- 1.8: Freqtradeの既存バックテストシステムとの統合
- 6.1: 1次モデル（ATR戦略）の独立性確保
- 6.4: ATR×0.5距離設定の忠実な再現
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, Tuple
import logging

from .atr_calculator import ATRCalculator

logger = logging.getLogger(__name__)


class ATRReturnCalculator:
    """ATR戦略の理論リターン計算を行うクラス"""

    def __init__(self, atr_period: int = 14, atr_multiplier: float = 0.5):
        """
        初期化

        Args:
            atr_period: ATR計算期間（デフォルト14）
            atr_multiplier: ATR乗数（デフォルト0.5）
        """
        self.atr_calculator = ATRCalculator(atr_period, atr_multiplier)
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        logger.info(f"ATRReturnCalculator初期化: period={atr_period}, multiplier={atr_multiplier}")

    def calculate_atr_returns(
        self,
        dataframe: pd.DataFrame,
        atr_period: Optional[int] = None,
        atr_multiplier: Optional[float] = None
    ) -> pd.Series:
        """
        ATR戦略の理論リターンを計算する

        richmanbtcチュートリアルに基づき、各期間でのATR指値戦略の
        理論的なリターンを計算します。

        Args:
            dataframe: OHLC市場データ
            atr_period: ATR計算期間（Noneの場合はインスタンス設定値）
            atr_multiplier: ATR乗数（Noneの場合はインスタンス設定値）

        Returns:
            各期間のATRリターンを含むSeries

        Raises:
            ValueError: データ不足またはカラム不足の場合
        """
        if dataframe.empty:
            raise ValueError("入力データが空です")

        calc_period = atr_period if atr_period is not None else self.atr_period
        calc_multiplier = atr_multiplier if atr_multiplier is not None else self.atr_multiplier

        # ATRと指値価格を計算
        df_with_atr = self.atr_calculator.calculate_atr_prices(
            dataframe, period=calc_period, multiplier=calc_multiplier
        )

        # リターン計算
        returns = self._calculate_period_returns(df_with_atr)

        return returns

    def _calculate_period_returns(self, df_with_atr: pd.DataFrame) -> pd.Series:
        """
        期間ごとのATRリターンを計算する（richmanbtc準拠）

        Args:
            df_with_atr: ATRと指値価格が計算済みのDataFrame

        Returns:
            各期間のリターンSeries
        """
        returns = pd.Series(index=df_with_atr.index, dtype=float, name='atr_returns')

        for i in range(1, len(df_with_atr)):
            current_price = df_with_atr['close'].iloc[i]
            prev_atr_buy = df_with_atr['atr_buy_price'].iloc[i-1]
            prev_atr_sell = df_with_atr['atr_sell_price'].iloc[i-1]

            # ATR指値戦略のリターン計算（richmanbtc準拠）
            if pd.notna(prev_atr_buy) and pd.notna(prev_atr_sell):
                if current_price <= prev_atr_buy:
                    # 買い指値約定: (current - buy_price) / buy_price
                    returns.iloc[i] = (current_price - prev_atr_buy) / prev_atr_buy
                elif current_price >= prev_atr_sell:
                    # 売り指値約定: (sell_price - current) / current
                    returns.iloc[i] = (prev_atr_sell - current_price) / current_price
                else:
                    # 約定なし
                    returns.iloc[i] = 0.0
            else:
                # ATR計算不可（データ不足）
                returns.iloc[i] = np.nan

        return returns

    def simulate_limit_execution(
        self,
        current_price: float,
        limit_price: float,
        side: str
    ) -> bool:
        """
        指値注文の約定シミュレーション

        Args:
            current_price: 現在価格
            limit_price: 指値価格
            side: 'buy' または 'sell'

        Returns:
            約定した場合True

        Raises:
            ValueError: 無効なside値の場合
        """
        if side == 'buy':
            return current_price <= limit_price
        elif side == 'sell':
            return current_price >= limit_price
        else:
            raise ValueError(f"無効なside値です: {side}. 'buy'または'sell'を指定してください")

    def calculate_theoretical_returns(self, dataframe: pd.DataFrame) -> pd.Series:
        """
        理論的なATRリターンを計算する（簡易版）

        Args:
            dataframe: OHLC市場データ

        Returns:
            理論リターンのSeries
        """
        return self.calculate_atr_returns(dataframe)

    def get_return_statistics(self, dataframe: pd.DataFrame) -> Dict[str, Any]:
        """
        ATRリターンの統計情報を取得する

        Args:
            dataframe: OHLC市場データ

        Returns:
            統計情報の辞書
        """
        returns = self.calculate_atr_returns(dataframe)
        valid_returns = returns.dropna()

        if len(valid_returns) == 0:
            return {
                'total_returns': 0,
                'mean_return': np.nan,
                'std_return': np.nan,
                'positive_returns': 0,
                'negative_returns': 0,
                'zero_returns': 0,
                'win_rate': np.nan
            }

        stats = {
            'total_returns': len(valid_returns),
            'mean_return': valid_returns.mean(),
            'std_return': valid_returns.std(),
            'positive_returns': (valid_returns > 0).sum(),
            'negative_returns': (valid_returns < 0).sum(),
            'zero_returns': (valid_returns == 0).sum(),
            'win_rate': (valid_returns > 0).mean()
        }

        return stats

    def validate_data_for_calculation(self, dataframe: pd.DataFrame) -> bool:
        """
        ATRリターン計算に必要なデータが揃っているかチェック

        Args:
            dataframe: チェック対象のデータ

        Returns:
            計算可能な場合True
        """
        return self.atr_calculator.validate_data_sufficiency(dataframe, self.atr_period)