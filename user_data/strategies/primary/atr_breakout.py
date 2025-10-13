"""ATR Breakout Strategy implementation."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# user_dataディレクトリをパスに追加
user_data_path = Path(__file__).parent.parent.parent
if str(user_data_path) not in sys.path:
    sys.path.insert(0, str(user_data_path))

from strategies.primary.base import PrimaryStrategyBase


class ATRBreakoutStrategy(PrimaryStrategyBase):
    """ATR指値戦略（richmanbtcチュートリアルの実装）

    ATR（Average True Range）に基づいて指値価格を計算し、
    約定シミュレーションによってリターンを計算する。
    """

    def __init__(self, params: dict):
        """Initialize ATR Breakout Strategy.

        Args:
            params: 戦略パラメータ辞書
                価格計算用:
                - period (int): ATR計算期間（デフォルト: 14）
                - multiplier (float): ATRに対する乗数（デフォルト: 0.5）

                ラベル生成用:
                - fee (float): シミュレーション用手数料（デフォルト: 0.00025）
                - exit_periods (int): N期間後のリターン計算（デフォルト: 24）
                - pips (float): 価格丸め精度（デフォルト: 0.5）
                - execution_mode (str): "chase" or "one_candle"（デフォルト: "one_candle"）
        """
        super().__init__(params)
        # 価格計算用パラメータ
        self.period = params.get("period", 14)
        self.multiplier = params.get("multiplier", 0.5)

        # ラベル生成用パラメータ
        self.fee = params.get("fee", 0.00025)
        self.exit_periods = params.get("exit_periods", 24)
        self.pips = params.get("pips", 0.5)

    def calculate_prices(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """ATRベースの指値価格計算

        現在価格からATRベースのオフセットを引いた/足した価格を指値として設定。

        Args:
            dataframe: OHLCデータ

        Returns:
            buy_price, sell_priceが追加されたDataFrame
        """
        # ATR計算
        atr = self._calculate_atr(dataframe, self.period)
        offset = atr * self.multiplier

        # 指値価格設定
        dataframe["buy_price"] = dataframe["close"] - offset
        dataframe["sell_price"] = dataframe["close"] + offset

        return dataframe

    def calculate_returns(self, dataframe: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        """指値戦略のリターン計算（ML学習用ラベル生成）

        約定シミュレーション方法はexecution_modeに基づいて切り替わる：
        - "chase": エントリー/エグジット両方で約定するまで追いかける方式
        - "one_candle": エントリーは1足限定、約定した場合のみリターン計算

        Args:
            dataframe: 価格計算済みのDataFrame (buy_price, sell_priceカラムが必要)

        Returns:
            (buy_return, sell_return): 買い/売りそれぞれの理論リターン

        Warning:
            このロジックは取引の根幹となるため、ミスがあると大きな損失に繋がる。
            必ず包括的なテストコードで検証すること。
        """
        if self.execution_mode == "chase":
            return self._calculate_chase_returns(dataframe)
        else:  # one_candle
            return self._calculate_one_candle_returns(dataframe)

    def _calculate_chase_returns(self, dataframe: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        """アプローチ1: エントリー追いかけ型（richmanbtc例1）

        エントリー/エグジット両方で約定するまで追いかける方式。
        Force Entry Price (FEP) を使用して実際の約定価格を計算する。

        Args:
            dataframe: 価格計算済みのDataFrame

        Returns:
            (buy_return, sell_return): 買い/売りそれぞれの理論リターン
        """
        # Force Entry Price (FEP) 計算
        buy_fep = self._calculate_force_entry_price(
            dataframe["buy_price"], dataframe["low"], self.pips, direction="buy"
        )
        sell_fep = self._calculate_force_entry_price(
            dataframe["sell_price"], dataframe["high"], self.pips, direction="sell"
        )

        # exit_periods足後のFEPでリターン計算
        future_sell_fep = sell_fep.shift(-self.exit_periods)
        future_buy_fep = buy_fep.shift(-self.exit_periods)

        # 買いリターン: buy_fep で買い → exit_periods後に sell_fep で売り
        buy_return = (future_sell_fep / buy_fep) - 1 - 2 * self.fee

        # 売りリターン: sell_fep で売り → exit_periods後に buy_fep で買い戻し
        sell_return = -(future_buy_fep / sell_fep - 1) - 2 * self.fee

        return buy_return, sell_return

    def _calculate_one_candle_returns(self, dataframe: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        """アプローチ2: エントリー1足限定型（richmanbtc例2、推奨）

        エントリーは1足限定で、約定した場合のみリターン計算。
        約定しない場合はリターン=0とする。

        Args:
            dataframe: 価格計算済みのDataFrame

        Returns:
            (buy_return, sell_return): 買い/売りそれぞれの理論リターン
        """
        # 次足での約定判定
        # 買い指値が次足のlowよりも高ければ約定
        buy_filled = (dataframe["buy_price"] / self.pips).round() > (
            dataframe["low"].shift(-1) / self.pips
        ).round()

        # 売り指値が次足のhighよりも低ければ約定
        sell_filled = (dataframe["sell_price"] / self.pips).round() < (
            dataframe["high"].shift(-1) / self.pips
        ).round()

        # エグジット用のFEP計算
        buy_fep = self._calculate_force_entry_price(
            dataframe["buy_price"], dataframe["low"], self.pips, direction="buy"
        )
        sell_fep = self._calculate_force_entry_price(
            dataframe["sell_price"], dataframe["high"], self.pips, direction="sell"
        )

        # 約定した場合のみリターン計算、約定しない場合は0
        future_sell_fep = sell_fep.shift(-self.exit_periods)
        future_buy_fep = buy_fep.shift(-self.exit_periods)

        # 買いリターン: buy_price で買い → exit_periods後に sell_fep で売り
        buy_return = np.where(
            buy_filled, (future_sell_fep / dataframe["buy_price"]) - 1 - 2 * self.fee, 0
        )

        # 売りリターン: sell_price で売り → exit_periods後に buy_fep で買い戻し
        sell_return = np.where(
            sell_filled, -(future_buy_fep / dataframe["sell_price"] - 1) - 2 * self.fee, 0
        )

        return pd.Series(buy_return, index=dataframe.index), pd.Series(
            sell_return, index=dataframe.index
        )

    def _calculate_force_entry_price(
        self,
        entry_price: pd.Series,
        extreme_price: pd.Series,
        pips: float,
        direction: str = "buy",
    ) -> pd.Series:
        """Force Entry Price (FEP) 計算

        買う（または売る）と決めてから約定するまで指値で追いかけた場合の
        実際の約定価格を計算する。

        Args:
            entry_price: 指値価格シリーズ
            extreme_price: low (買い) or high (売り) のシリーズ
            pips: 価格丸め精度
            direction: 'buy' または 'sell'

        Returns:
            各時点でのFEP
        """
        # numpy配列に変換して処理
        entry_arr = entry_price.values
        extreme_arr = extreme_price.values
        n = len(entry_arr)

        if direction == "buy":
            fep = _calculate_buy_fep(entry_arr, extreme_arr, pips, n)
        else:  # sell
            fep = _calculate_sell_fep(entry_arr, extreme_arr, pips, n)

        return pd.Series(fep, index=entry_price.index)

    def _calculate_atr(self, dataframe: pd.DataFrame, period: int) -> pd.Series:
        """ATR（Average True Range）計算

        Args:
            dataframe: OHLCデータ
            period: ATR計算期間

        Returns:
            ATRのシリーズ
        """
        high = dataframe["high"]
        low = dataframe["low"]
        close = dataframe["close"]
        prev_close = close.shift(1)

        # True Range計算
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # ATR = True Rangeの移動平均
        atr = tr.rolling(window=period).mean()

        return atr


def _calculate_buy_fep(entry_price: np.ndarray, low: np.ndarray, pips: float, n: int) -> np.ndarray:
    """買いFEP計算

    指値で買い注文を出し、約定するまで追いかける場合の実際の約定価格。

    Args:
        entry_price: 指値価格配列
        low: 最低価格配列
        pips: 価格丸め精度
        n: 配列長

    Returns:
        各時点でのFEP配列
    """
    fep = np.full(n, np.nan)

    for i in range(n):
        if np.isnan(entry_price[i]):
            continue

        # 指値価格を丸める
        limit_price = np.round(entry_price[i] / pips) * pips

        # 約定するまで追いかける
        for j in range(i, n):
            if np.isnan(low[j]):
                break

            # lowが指値以下なら約定
            if low[j] <= limit_price:
                fep[i] = limit_price
                break

            # 約定しなかった場合、次の足では指値を更新
            if j + 1 < n and not np.isnan(entry_price[j + 1]):
                limit_price = max(limit_price, np.round(entry_price[j + 1] / pips) * pips)

    return fep


def _calculate_sell_fep(
    entry_price: np.ndarray, high: np.ndarray, pips: float, n: int
) -> np.ndarray:
    """売りFEP計算

    指値で売り注文を出し、約定するまで追いかける場合の実際の約定価格。

    Args:
        entry_price: 指値価格配列
        high: 最高価格配列
        pips: 価格丸め精度
        n: 配列長

    Returns:
        各時点でのFEP配列
    """
    fep = np.full(n, np.nan)

    for i in range(n):
        if np.isnan(entry_price[i]):
            continue

        # 指値価格を丸める
        limit_price = np.round(entry_price[i] / pips) * pips

        # 約定するまで追いかける
        for j in range(i, n):
            if np.isnan(high[j]):
                break

            # highが指値以上なら約定
            if high[j] >= limit_price:
                fep[i] = limit_price
                break

            # 約定しなかった場合、次の足では指値を更新
            if j + 1 < n and not np.isnan(entry_price[j + 1]):
                limit_price = min(limit_price, np.round(entry_price[j + 1] / pips) * pips)

    return fep
