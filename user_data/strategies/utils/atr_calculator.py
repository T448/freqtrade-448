"""
ATR（Average True Range）計算機能

richmanbtcチュートリアルに基づくATR計算と指値価格算出機能を提供します。

Requirements implemented:
- 1.1: 市場データからATR（Average True Range）を計算する機能
- 1.2: 設定可能な期間パラメータ（デフォルト14期間）でATR値を算出
- 1.3: ATR乗数（デフォルト0.5）を使用した指値価格計算機能
- 1.4: データ不足時の適切なエラーハンドリングと期間スキップ機能
"""

import logging

import pandas as pd
import talib as ta


logger = logging.getLogger(__name__)


class ATRCalculator:
    """ATR計算と指値価格算出を行うクラス"""

    def __init__(self, atr_period: int = 14, atr_multiplier: float = 0.5):
        """
        初期化

        Args:
            atr_period: ATR計算期間（デフォルト14）
            atr_multiplier: ATR乗数（デフォルト0.5）

        Raises:
            ValueError: 無効なパラメータ値の場合
        """
        if atr_period <= 0:
            raise ValueError(f"ATR期間は正の値である必要があります: {atr_period}")
        if atr_multiplier < 0:
            raise ValueError(f"ATR乗数は非負の値である必要があります: {atr_multiplier}")

        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        logger.info(f"ATRCalculator初期化: period={atr_period}, multiplier={atr_multiplier}")

    def calculate_atr(self, dataframe: pd.DataFrame, period: int = None) -> pd.Series:
        """
        市場データからATRを計算する

        Args:
            dataframe: OHLC市場データ（high, low, close列必須）
            period: ATR計算期間（Noneの場合はインスタンス設定値を使用）

        Returns:
            ATR値のSeries

        Raises:
            ValueError: データ不足またはカラム不足の場合
        """
        if dataframe.empty:
            raise ValueError("入力データが空です")

        required_columns = ["high", "low", "close"]
        missing_columns = [col for col in required_columns if col not in dataframe.columns]
        if missing_columns:
            raise ValueError(f"必須カラムが不足しています: {missing_columns}")

        calc_period = period if period is not None else self.atr_period

        if len(dataframe) < calc_period:
            error_msg = f"データが不足しています。必要: {calc_period}行, 実際: {len(dataframe)}行"
            logger.warning(error_msg)
            raise ValueError(error_msg)

        atr_values = ta.ATR(
            dataframe["high"].astype(float).values,
            dataframe["low"].astype(float).values,
            dataframe["close"].astype(float).values,
            timeperiod=calc_period,
        )

        return pd.Series(atr_values, index=dataframe.index, name="atr")

    def calculate_limit_price(
        self, close_price: float, atr_value: float, side: str, multiplier: float = None
    ) -> float:
        """
        ATRを使用して指値価格を計算する

        Args:
            close_price: 終値
            atr_value: ATR値
            side: 'buy' または 'sell'
            multiplier: ATR乗数（Noneの場合はインスタンス設定値を使用）

        Returns:
            指値価格

        Raises:
            ValueError: 無効なside値の場合
        """
        if pd.isna(atr_value) or pd.isna(close_price):
            raise ValueError("ATR値または終値がNaNです")

        calc_multiplier = multiplier if multiplier is not None else self.atr_multiplier

        if side == "buy":
            # 買い指値: close - (ATR * multiplier)
            return close_price - (atr_value * calc_multiplier)
        elif side == "sell":
            # 売り指値: close + (ATR * multiplier)
            return close_price + (atr_value * calc_multiplier)
        else:
            raise ValueError(f"無効なside値です: {side}. 'buy'または'sell'を指定してください")

    def calculate_atr_prices(
        self, dataframe: pd.DataFrame, period: int = None, multiplier: float = None
    ) -> pd.DataFrame:
        """
        データフレーム全体に対してATRと指値価格を計算する

        Args:
            dataframe: OHLC市場データ
            period: ATR計算期間
            multiplier: ATR乗数

        Returns:
            atr, atr_buy_price, atr_sell_price列を追加したDataFrame
        """
        result_df = dataframe.copy()

        # ATR計算
        calc_period = period if period is not None else self.atr_period
        result_df["atr"] = self.calculate_atr(dataframe, period)

        # 指値価格計算
        calc_multiplier = multiplier if multiplier is not None else self.atr_multiplier

        result_df["atr_buy_price"] = result_df["close"] - (result_df["atr"] * calc_multiplier)
        result_df["atr_sell_price"] = result_df["close"] + (result_df["atr"] * calc_multiplier)

        # 詳細ログ出力（最新5行のサンプル）
        if len(result_df) >= 5:
            recent_data = result_df.tail(5)
            logger.info(f"🔢 ATR計算完了: period={calc_period}, multiplier={calc_multiplier}")
            logger.info(f"📊 最新ATR値: {recent_data['atr'].iloc[-1]:.8f}")
            logger.info(f"💰 最新買い価格: {recent_data['atr_buy_price'].iloc[-1]:.8f}")
            logger.info(f"💰 最新売り価格: {recent_data['atr_sell_price'].iloc[-1]:.8f}")
            logger.info(f"📈 クローズ価格: {recent_data['close'].iloc[-1]:.8f}")

            # ATR有効性チェック
            valid_atr_count = result_df["atr"].notna().sum()
            logger.info(f"✅ ATR有効レコード数: {valid_atr_count}/{len(result_df)}")

        return result_df

    def validate_data_sufficiency(self, dataframe: pd.DataFrame, period: int = None) -> bool:
        """
        ATR計算に十分なデータがあるかチェックする

        Args:
            dataframe: チェック対象のデータ
            period: 必要な期間

        Returns:
            十分なデータがある場合True
        """
        calc_period = period if period is not None else self.atr_period
        return len(dataframe) >= calc_period
