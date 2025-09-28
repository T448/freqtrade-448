"""
ATR計算エンジン - 統一ATR計算システム

richmanbtcチュートリアルに基づくATR計算と指値価格算出機能を提供します。
シングルトンパターンとキャッシュ機能により重複計算を排除します。

Requirements implemented:
- 1.1: 市場データからATR（Average True Range）を計算する機能
- 1.2: 設定可能な期間パラメータ（デフォルト14期間）でATR値を算出
- 1.3: ATR乗数（デフォルト0.5）を使用した指値価格計算機能
- 1.4: データ不足時の適切なエラーハンドリングと期間スキップ機能
- 統一エンジン: 重複するATR計算の一元化
- キャッシュ機能: 同一パラメータでの重複計算回避
"""

import hashlib
import logging
from typing import Dict, Optional, Tuple

import pandas as pd
import talib as ta


logger = logging.getLogger(__name__)


class ATRCalculatorEngine:
    """
    統一ATR計算エンジン（シングルトンパターン）

    全てのATR計算を一元化し、キャッシュ機能で重複計算を排除します。
    """

    _instance: Optional["ATRCalculatorEngine"] = None
    _cache: Dict[str, Tuple[pd.Series, pd.DataFrame]] = {}
    _cache_size_limit = 100  # キャッシュサイズ制限

    def __new__(cls, *args, **kwargs):
        """シングルトンパターンの実装"""
        if cls._instance is None:
            cls._instance = super(ATRCalculatorEngine, cls).__new__(cls)
        return cls._instance

    def __init__(self, atr_period: int = 14, atr_multiplier: float = 0.5):
        """
        初期化（シングルトンのため初回のみ実行）

        Args:
            atr_period: ATR計算期間（デフォルト14）
            atr_multiplier: ATR乗数（デフォルト0.5）

        Raises:
            ValueError: 無効なパラメータ値の場合
        """
        # シングルトンのため、既に初期化済みの場合はスキップ
        if hasattr(self, "initialized"):
            return

        if atr_period <= 0:
            raise ValueError(f"ATR period must be positive: {atr_period}")
        if atr_multiplier < 0:
            raise ValueError(f"ATR multiplier must be non-negative: {atr_multiplier}")

        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.initialized = True
        logger.debug(
            f"ATRCalculatorEngine initialized: period={atr_period}, multiplier={atr_multiplier}"
        )

    def _generate_cache_key(self, dataframe: pd.DataFrame, period: int, multiplier: float) -> str:
        """
        キャッシュキーを生成

        Args:
            dataframe: データフレーム
            period: ATR期間
            multiplier: ATR乗数

        Returns:
            キャッシュキー
        """
        # データフレームのハッシュ（最新数行のみ使用してパフォーマンス向上）
        sample_data = dataframe.tail(20) if len(dataframe) > 20 else dataframe
        data_hash = hashlib.md5(
            pd.util.hash_pandas_object(sample_data[["high", "low", "close"]], index=True).values
        ).hexdigest()[:16]

        return f"atr_{data_hash}_{period}_{multiplier}_{len(dataframe)}"

    def _manage_cache_size(self):
        """キャッシュサイズ管理"""
        if len(self._cache) > self._cache_size_limit:
            # 古いエントリを削除（最初の半分を削除）
            items_to_remove = list(self._cache.keys())[: self._cache_size_limit // 2]
            for key in items_to_remove:
                del self._cache[key]
            logger.debug(f"Cache cleaned: removed {len(items_to_remove)} entries")

    def calculate_atr(self, dataframe: pd.DataFrame, period: int = None) -> pd.Series:
        """
        市場データからATRを計算する（キャッシュ機能付き）

        Args:
            dataframe: OHLC市場データ（high, low, close列必須）
            period: ATR計算期間（Noneの場合はインスタンス設定値を使用）

        Returns:
            ATR値のSeries

        Raises:
            ValueError: データ不足またはカラム不足の場合
        """
        if dataframe.empty:
            raise ValueError("Input dataframe is empty")

        required_columns = ["high", "low", "close"]
        missing_columns = [col for col in required_columns if col not in dataframe.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        calc_period = period if period is not None else self.atr_period

        if len(dataframe) < calc_period:
            error_msg = f"Insufficient data: need {calc_period}, got {len(dataframe)}"
            logger.warning(error_msg)
            raise ValueError(error_msg)

        # キャッシュチェック
        cache_key = self._generate_cache_key(dataframe, calc_period, 0)  # ATR計算のみなので乗数は0

        if cache_key in self._cache:
            logger.debug("ATR cache hit")
            return self._cache[cache_key][0]  # キャッシュからATR Seriesを返す

        # ATR計算
        atr_values = ta.ATR(
            dataframe["high"].astype(float).values,
            dataframe["low"].astype(float).values,
            dataframe["close"].astype(float).values,
            timeperiod=calc_period,
        )

        atr_series = pd.Series(atr_values, index=dataframe.index, name="atr")

        # キャッシュに保存（価格データは後で計算）
        self._cache[cache_key] = (atr_series, None)
        self._manage_cache_size()

        logger.debug(f"ATR calculated: period={calc_period}, records={len(atr_series)}")
        return atr_series

    def calculate_limit_price(
        self, close_price: float, atr_value: float, side: str, multiplier: float = None
    ) -> float:
        """
        ATRを使用して指値価格を計算する

        Args:
            close_price: 終値
            atr_value: ATR値
            side: 'buy', 'sell', 'long', または 'short'
            multiplier: ATR乗数（Noneの場合はインスタンス設定値を使用）

        Returns:
            指値価格

        Raises:
            ValueError: 無効なside値の場合
        """
        if pd.isna(atr_value) or pd.isna(close_price):
            raise ValueError("ATR value or close price is NaN")

        calc_multiplier = multiplier if multiplier is not None else self.atr_multiplier

        # side値の正規化
        side_lower = side.lower()
        if side_lower in ["buy", "long"]:
            # 買い指値: close - (ATR * multiplier)
            return close_price - (atr_value * calc_multiplier)
        elif side_lower in ["sell", "short"]:
            # 売り指値: close + (ATR * multiplier)
            return close_price + (atr_value * calc_multiplier)
        else:
            raise ValueError(f"Invalid side value: {side}. Use 'buy'/'long' or 'sell'/'short'")

    def calculate_limit_price_for_pair(
        self, pair: str, side: str, period: int, multiplier: float
    ) -> Optional[float]:
        """
        ペア名を指定してATR指値価格を計算（戦略パターン用）

        Args:
            pair: 取引ペア
            side: 取引方向
            period: ATR期間
            multiplier: ATR乗数

        Returns:
            計算された指値価格（エラー時はNone）
        """
        try:
            # この機能は実際の戦略クラスで dataprovider を使用して実装される
            # ここでは抽象的なインターフェースのみ提供
            logger.debug(f"Limit price calculation requested for {pair} {side}")
            return None  # 実装は呼び出し元で行う

        except Exception as e:
            logger.error(f"Error calculating limit price for {pair}: {e}")
            return None

    def calculate_atr_prices(
        self, dataframe: pd.DataFrame, period: int = None, multiplier: float = None
    ) -> pd.DataFrame:
        """
        データフレーム全体に対してATRと指値価格を計算する（キャッシュ機能付き）

        Args:
            dataframe: OHLC市場データ
            period: ATR計算期間
            multiplier: ATR乗数

        Returns:
            atr, atr_buy_price, atr_sell_price列を追加したDataFrame
        """
        calc_period = period if period is not None else self.atr_period
        calc_multiplier = multiplier if multiplier is not None else self.atr_multiplier

        # 完全なキャッシュキー（乗数も含む）
        cache_key = self._generate_cache_key(dataframe, calc_period, calc_multiplier)

        if cache_key in self._cache and self._cache[cache_key][1] is not None:
            logger.debug("ATR prices cache hit")
            return self._cache[cache_key][1]

        result_df = dataframe.copy()

        # ATR計算
        result_df["atr"] = self.calculate_atr(dataframe, period)

        # 指値価格計算
        result_df["atr_buy_price"] = result_df["close"] - (result_df["atr"] * calc_multiplier)
        result_df["atr_sell_price"] = result_df["close"] + (result_df["atr"] * calc_multiplier)

        # 最適化されたログ出力
        if len(result_df) >= 5:
            recent_atr = result_df["atr"].iloc[-1]
            valid_atr_count = result_df["atr"].notna().sum()
            logger.debug(
                f"ATR calculation completed: period={calc_period}, multiplier={calc_multiplier}, "
                f"latest_atr={recent_atr:.8f}, valid_records={valid_atr_count}/{len(result_df)}"
            )

        # キャッシュに保存
        atr_series = result_df["atr"]
        self._cache[cache_key] = (atr_series, result_df)
        self._manage_cache_size()

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

    @classmethod
    def get_instance(cls, *args, **kwargs) -> "ATRCalculatorEngine":
        """
        インスタンス取得用メソッド

        Returns:
            ATRCalculatorEngineのシングルトンインスタンス
        """
        return cls(*args, **kwargs)

    @classmethod
    def clear_cache(cls):
        """キャッシュクリア"""
        cls._cache.clear()
        logger.debug("ATR cache cleared")


# 後方互換性のためのエイリアス
ATRCalculator = ATRCalculatorEngine
