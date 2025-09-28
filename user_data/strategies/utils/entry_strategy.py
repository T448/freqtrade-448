"""
Entry Strategy Pattern Implementation

ストラテジーパターンを使用してATR戦略とML統合戦略を分離します。
これによりforce_ml_positiveフラグによる複雑な分岐を排除します。
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


class EntryStrategy(ABC):
    """エントリー戦略の抽象基底クラス"""

    def __init__(self, atr_calculator=None):
        """
        初期化

        Args:
            atr_calculator: ATR計算エンジン（依存性注入）
        """
        self.atr_calculator = atr_calculator
        logger.debug("EntryStrategy initialized")

    @abstractmethod
    def generate_entry_signals(
        self, dataframe: pd.DataFrame, metadata: dict, config: dict
    ) -> pd.DataFrame:
        """
        エントリー信号を生成する

        Args:
            dataframe: 市場データ
            metadata: ペア情報
            config: 設定パラメータ

        Returns:
            エントリー信号が追加されたDataFrame
        """
        pass

    @abstractmethod
    def calculate_entry_price(
        self, pair: str, side: str, proposed_rate: float, config: dict
    ) -> float:
        """
        エントリー価格を計算する

        Args:
            pair: 取引ペア
            side: 取引方向（"long" or "short"）
            proposed_rate: 提案価格
            config: 設定パラメータ

        Returns:
            計算されたエントリー価格
        """
        pass

    def _validate_dataframe(self, dataframe: pd.DataFrame, min_length: int = 50) -> bool:
        """
        データフレームの妥当性チェック

        Args:
            dataframe: チェック対象
            min_length: 最小長さ

        Returns:
            妥当な場合True
        """
        if dataframe.empty:
            logger.warning("DataFrame is empty")
            return False

        if len(dataframe) < min_length:
            logger.warning(f"Insufficient data: {len(dataframe)} < {min_length}")
            return False

        required_columns = ["high", "low", "close", "atr"]
        missing_columns = [col for col in required_columns if col not in dataframe.columns]
        if missing_columns:
            logger.warning(f"Missing required columns: {missing_columns}")
            return False

        return True

    def _initialize_entry_signals(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        エントリー信号カラムを初期化

        Args:
            dataframe: 対象DataFrame

        Returns:
            初期化されたDataFrame
        """
        result = dataframe.copy()
        result["enter_long"] = 0
        result["enter_short"] = 0
        return result


class BasicATRStrategy(EntryStrategy):
    """基本ATR戦略（ML無効時）"""

    def generate_entry_signals(
        self, dataframe: pd.DataFrame, metadata: dict, config: dict
    ) -> pd.DataFrame:
        """
        基本ATR戦略でエントリー信号を生成

        前期間のATR価格と現在価格を比較して売買判定を行う
        """
        pair = metadata.get("pair", "unknown")
        result = self._initialize_entry_signals(dataframe)

        if not self._validate_dataframe(dataframe):
            logger.warning(f"Data validation failed for {pair}")
            return result

        try:
            # ATR価格データの確認
            required_atr_columns = ["atr", "atr_buy_price", "atr_sell_price"]
            if not all(col in dataframe.columns for col in required_atr_columns):
                logger.warning(f"ATR price data missing for {pair}")
                return result

            # ATR戦略の基本ロジック
            atr_buy_signal = pd.Series([False] * len(dataframe), index=dataframe.index)
            atr_sell_signal = pd.Series([False] * len(dataframe), index=dataframe.index)

            if len(dataframe) > 1:
                # 前期間のATR価格と現在価格の比較
                prev_atr_buy = dataframe["atr_buy_price"].shift(1)
                prev_atr_sell = dataframe["atr_sell_price"].shift(1)
                current_close = dataframe["close"]

                # ATR戦略: 価格がATR指値レベルに到達した時の取引判定
                atr_buy_signal = (current_close <= prev_atr_buy) & (prev_atr_buy > 0)
                atr_sell_signal = (current_close >= prev_atr_sell) & (prev_atr_sell > 0)

                buy_signals = atr_buy_signal.sum()
                sell_signals = atr_sell_signal.sum()
                logger.debug(f"ATR basic signals {pair}: long={buy_signals}, short={sell_signals}")

            # 有効性チェック
            atr_valid = dataframe["atr"] > 0
            price_valid = (dataframe["atr_buy_price"] > 0) & (dataframe["atr_sell_price"] > 0)

            # エントリー条件
            long_condition = atr_buy_signal & atr_valid & price_valid
            short_condition = atr_sell_signal & atr_valid & price_valid

            # 信号設定
            result.loc[long_condition, "enter_long"] = 1
            result.loc[short_condition, "enter_short"] = 1

            long_signals = long_condition.sum()
            short_signals = short_condition.sum()
            logger.info(
                f"Basic ATR entry signals {pair}: long={long_signals}, short={short_signals}"
            )

            return result

        except Exception as e:
            logger.error(f"Error generating basic ATR signals for {pair}: {e}")
            return result

    def calculate_entry_price(
        self, pair: str, side: str, proposed_rate: float, config: dict
    ) -> float:
        """
        ATR指値価格を計算
        """
        try:
            if not self.atr_calculator:
                logger.warning(f"ATR calculator not available for {pair}")
                return proposed_rate

            # ATR計算エンジンを使用して価格計算
            atr_price = self.atr_calculator.calculate_limit_price_for_pair(
                pair, side, config.get("entry_length", 14), config.get("entry_point", 0.5)
            )

            if pd.isna(atr_price):
                logger.warning(f"Invalid ATR price calculation for {pair}")
                return proposed_rate

            logger.debug(f"ATR entry price {pair} {side}: {atr_price:.8f}")
            return atr_price

        except Exception as e:
            logger.error(f"Error calculating ATR entry price for {pair}: {e}")
            return proposed_rate


class MLIntegratedStrategy(EntryStrategy):
    """ML統合戦略（ML有効時）"""

    def generate_entry_signals(
        self, dataframe: pd.DataFrame, metadata: dict, config: dict
    ) -> pd.DataFrame:
        """
        ML統合戦略でエントリー信号を生成

        ML予測と信頼度フィルタリングを使用
        """
        pair = metadata.get("pair", "unknown")
        result = self._initialize_entry_signals(dataframe)

        if not self._validate_dataframe(dataframe):
            logger.warning(f"Data validation failed for {pair}")
            return result

        try:
            # ML予測データの確認
            has_ml_prediction = "&-prediction" in dataframe.columns
            has_ml_probability = "&-probability" in dataframe.columns

            if not has_ml_prediction:
                logger.warning(f"ML prediction data missing for {pair}")
                return result

            # ML予測の取得
            ml_prediction = dataframe["&-prediction"] == 1

            # 信頼度フィルタリング
            confidence_filter = True
            confidence_threshold = config.get("confidence_threshold", 0.6)

            if has_ml_probability and confidence_threshold > 0:
                confidence_filter = dataframe["&-probability"] >= confidence_threshold
                high_confidence_count = confidence_filter.sum()
                logger.debug(
                    f"High confidence predictions {pair}: {high_confidence_count}/{len(dataframe)}"
                )

            # ATRデータの有効性チェック
            atr_valid = dataframe["atr"] > 0
            price_valid = (dataframe["atr_buy_price"] > 0) & (dataframe["atr_sell_price"] > 0)

            # ML統合条件
            long_condition = (
                ml_prediction  # ML予測=1
                & confidence_filter  # 信頼度フィルタ
                & atr_valid  # ATR有効
                & price_valid  # 価格データ有効
            )

            short_condition = (
                ~ml_prediction  # ML予測=0
                & confidence_filter  # 信頼度フィルタ
                & atr_valid  # ATR有効
                & price_valid  # 価格データ有効
            )

            # 信号設定
            result.loc[long_condition, "enter_long"] = 1
            result.loc[short_condition, "enter_short"] = 1

            long_signals = long_condition.sum()
            short_signals = short_condition.sum()
            logger.info(
                f"ML integrated entry signals {pair}: long={long_signals}, short={short_signals}"
            )

            # ML予測統計
            if has_ml_prediction:
                predictions = dataframe["&-prediction"].dropna()
                if len(predictions) > 0:
                    positive_predictions = (predictions == 1).sum()
                    prediction_ratio = positive_predictions / len(predictions)
                    logger.debug(
                        f"ML prediction stats {pair}: positive_ratio={prediction_ratio:.3f}"
                    )

            return result

        except Exception as e:
            logger.error(f"Error generating ML integrated signals for {pair}: {e}")
            return result

    def calculate_entry_price(
        self, pair: str, side: str, proposed_rate: float, config: dict
    ) -> float:
        """
        ML統合戦略のエントリー価格計算（基本ATRと同じ）
        """
        try:
            if not self.atr_calculator:
                logger.warning(f"ATR calculator not available for {pair}")
                return proposed_rate

            # ATR計算エンジンを使用
            atr_price = self.atr_calculator.calculate_limit_price_for_pair(
                pair, side, config.get("entry_length", 14), config.get("entry_point", 0.5)
            )

            if pd.isna(atr_price):
                logger.warning(f"Invalid ATR price calculation for {pair}")
                return proposed_rate

            logger.debug(f"ML integrated entry price {pair} {side}: {atr_price:.8f}")
            return atr_price

        except Exception as e:
            logger.error(f"Error calculating ML entry price for {pair}: {e}")
            return proposed_rate


class EntryStrategyFactory:
    """エントリー戦略のファクトリークラス"""

    @staticmethod
    def create_strategy(has_ml_prediction: bool, atr_calculator=None) -> EntryStrategy:
        """
        適切なエントリー戦略を作成

        Args:
            has_ml_prediction: ML予測データの有無
            atr_calculator: ATR計算エンジン

        Returns:
            適切なエントリー戦略インスタンス
        """
        if has_ml_prediction:
            logger.debug("Creating ML integrated strategy")
            return MLIntegratedStrategy(atr_calculator)
        else:
            logger.debug("Creating basic ATR strategy")
            return BasicATRStrategy(atr_calculator)
