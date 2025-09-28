"""
ATRMLStrategy - 2層統合トレーディングストラテジー

richmanbtcチュートリアルの概念に基づく2層システム：
1. ATRベースの指値戦略（1次モデル）
2. LightGBM機械学習分類（2次モデル）

要件:
- 5.1: FreqtradeストラテジーとのMLOps統合
- 5.2: フォールバック機能と運用安全性
"""

import logging
import os

# Import utilities using absolute import compatible with Freqtrade
import sys
from typing import Optional

import pandas as pd

from freqtrade.strategy import IStrategy


sys.path.append(os.path.dirname(__file__))

from utils.atr_calculator import ATRCalculatorEngine
from utils.entry_strategy import EntryStrategyFactory


logger = logging.getLogger(__name__)


class ATRMLStrategy(IStrategy):
    """
    ATR機械学習統合ストラテジー

    richmanbtcチュートリアルの2層トレーディングシステムを実装。
    ATR戦略（1次）とLightGBM分類（2次）を統合して取引判定を行う。
    """

    # ストラテジー基本設定
    INTERFACE_VERSION: int = 3
    minimal_roi = {"0": 0.05, "30": 0.03, "60": 0.01, "120": 0}
    stoploss = -0.10
    timeframe = "5m"
    process_only_new_candles = True

    # FreqAI統合設定
    can_short = True  # ショート取引有効
    use_exit_signal = True

    # ATR戦略パラメータ
    entry_length = 14  # ATR計算期間
    entry_point = 0.5  # ATR乗数

    # ML統合パラメータ
    confidence_threshold = 0.6  # ML予測信頼度閾値

    def __init__(self, config=None, **kwargs):
        """
        ストラテジー初期化

        Args:
            config: Freqtrade設定辞書
            **kwargs: 追加パラメータ
        """
        super().__init__(config, **kwargs)

        # カスタム設定の読み込み
        if config and "atr_ml_strategy" in config:
            strategy_config = config["atr_ml_strategy"]
            self.entry_length = strategy_config.get("entry_length", self.entry_length)
            self.entry_point = strategy_config.get("entry_point", self.entry_point)
            self.confidence_threshold = strategy_config.get(
                "confidence_threshold", self.confidence_threshold
            )

        # 統一ATR計算エンジンの初期化
        self.atr_engine = ATRCalculatorEngine.get_instance(self.entry_length, self.entry_point)

        # エントリー戦略は動的に決定（populate_entry_trendで初期化）
        self.entry_strategy = None

        logger.debug(
            f"ATRMLStrategy initialized: length={self.entry_length}, point={self.entry_point}"
        )

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        テクニカル指標とATR価格の計算 - 要件 5.1

        Args:
            dataframe: OHLCデータ
            metadata: ペア情報

        Returns:
            指標が追加されたDataFrame
        """
        try:
            # ATR関連価格の計算
            dataframe = self._calculate_atr_prices(dataframe)

            # FreqAI予測の取得（有効性チェック）
            freqai_enabled = (
                hasattr(self, "freqai")
                and self.freqai is not None
                and getattr(self.freqai, "enabled", False)
            )

            if freqai_enabled:
                try:
                    dataframe = self.freqai.start(dataframe, metadata, self)
                    logger.debug("FreqAI prediction data added successfully")
                except Exception as freqai_error:
                    logger.warning(f"FreqAI execution failed: {freqai_error}")
                    logger.debug("Continuing with ATR-only strategy")
            else:
                logger.debug("FreqAI disabled - running backtest with ATR only")

            pair = metadata.get("pair", "unknown")
            logger.debug(f"Indicators calculation completed: {pair}, records={len(dataframe)}")

            return dataframe

        except Exception as e:
            pair = metadata.get("pair", "unknown")
            logger.error(f"Indicators calculation error ({pair}): {e}")
            return dataframe

    def _calculate_atr_prices(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        ATR価格計算 - 要件 5.1（統一エンジン使用）

        Args:
            dataframe: OHLCデータ

        Returns:
            ATR価格が追加されたDataFrame
        """
        # 統一ATR計算エンジンを使用
        return self.atr_engine.calculate_atr_prices(
            dataframe, period=self.entry_length, multiplier=self.entry_point
        )

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        エントリートレンド生成 - 要件 5.1（新アーキテクチャ）

        ストラテジーパターンを使用してML予測有無に応じた適切な戦略を選択

        Args:
            dataframe: 指標付きDataFrame
            metadata: ペア情報

        Returns:
            エントリー信号が追加されたDataFrame
        """
        try:
            pair = metadata.get("pair", "unknown")

            # データ十分性チェック
            if len(dataframe) < self.entry_length + 10:
                logger.warning(
                    f"Insufficient data {pair}: need={self.entry_length + 10}, got={len(dataframe)}"
                )
                # エントリー信号カラムの初期化
                dataframe["enter_long"] = 0
                dataframe["enter_short"] = 0
                return dataframe

            # ML予測データの有無を確認
            has_ml_prediction = "&-prediction" in dataframe.columns

            # 適切なエントリー戦略を動的に選択
            self.entry_strategy = EntryStrategyFactory.create_strategy(
                has_ml_prediction, self.atr_engine
            )

            # 設定パラメータを準備
            config = {
                "entry_length": self.entry_length,
                "entry_point": self.entry_point,
                "confidence_threshold": self.confidence_threshold,
            }

            # 選択された戦略でエントリー信号を生成
            dataframe = self.entry_strategy.generate_entry_signals(dataframe, metadata, config)

            # 予測詳細ログ（簡略化）
            self._log_prediction_summary(dataframe, metadata)

            return dataframe

        except Exception as e:
            pair = metadata.get("pair", "unknown")
            logger.error(f"Entry trend generation error {pair}: {e}")
            # 安全なフォールバック
            dataframe["enter_long"] = 0
            dataframe["enter_short"] = 0
            return dataframe

    def _log_prediction_summary(self, dataframe: pd.DataFrame, metadata: dict) -> None:
        """
        予測結果の簡略サマリーログ - 要件 5.2

        Args:
            dataframe: データフレーム
            metadata: ペア情報
        """
        try:
            pair = metadata.get("pair", "unknown")

            # エントリー信号統計
            long_signals = dataframe["enter_long"].sum()
            short_signals = dataframe["enter_short"].sum()
            total_records = len(dataframe)

            logger.info(
                f"Entry signals {pair}: long={long_signals}, short={short_signals}, total={total_records}"
            )

            # ML予測統計（利用可能な場合のみ）
            if "&-prediction" in dataframe.columns:
                predictions = dataframe["&-prediction"].dropna()
                if len(predictions) > 0:
                    positive_predictions = (predictions == 1).sum()
                    prediction_ratio = positive_predictions / len(predictions)
                    logger.debug(f"ML prediction ratio {pair}: {prediction_ratio:.3f}")

        except Exception as e:
            pair = metadata.get("pair", "unknown")
            logger.error(f"Prediction summary log error {pair}: {e}")

    def custom_entry_price(
        self,
        pair: str,
        trade,
        current_time,
        proposed_rate: float,
        entry_tag: Optional[str],
        side: str,
        **kwargs,
    ) -> float:
        """
        ATR指値価格計算 - 要件 5.1（統一エンジン使用）

        Args:
            pair: 取引ペア
            side: 取引方向（"long" or "short"）

        Returns:
            ATRベース指値価格
        """
        try:
            # 最新のデータフレーム取得
            dataframe = self.dp.get_pair_dataframe(pair, self.timeframe)

            if dataframe.empty:
                logger.warning(f"Dataframe retrieval failed: {pair}")
                return proposed_rate

            # 統一ATR計算エンジンを使用してATR計算
            atr_series = self.atr_engine.calculate_atr(dataframe, period=self.entry_length)
            if atr_series.empty or pd.isna(atr_series.iloc[-1]):
                logger.warning(f"ATR calculation failed {pair}")
                return proposed_rate

            current_atr = atr_series.iloc[-1]
            current_close = dataframe["close"].iloc[-1]

            # 統一エンジンでATR指値価格計算
            atr_price = self.atr_engine.calculate_limit_price(
                current_close, current_atr, side, self.entry_point
            )

            # データ検証
            if pd.isna(atr_price) or current_atr <= 0:
                logger.warning(f"Invalid ATR calculation {pair}: ATR={current_atr}")
                return proposed_rate

            logger.debug(f"ATR limit price {pair} {side}: {atr_price:.8f}")
            return atr_price

        except Exception as e:
            logger.error(f"Custom entry price calculation error {pair}: {e}")
            return proposed_rate

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        エグジットトレンド生成

        Args:
            dataframe: データフレーム
            metadata: ペア情報

        Returns:
            エグジット信号が追加されたDataFrame
        """
        # 基本的なエグジット条件
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0

        # 利益確定・損切りはROIとstoplossに委譲
        return dataframe
