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
import pandas as pd
import numpy as np
from typing import Optional

from freqtrade.strategy import IStrategy, informative
import freqtrade.vendor.qtpylib.indicators as qtpylib

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
    can_short = True
    use_exit_signal = True

    # ATR戦略パラメータ
    entry_length = 14  # ATR計算期間
    entry_point = 0.5  # ATR乗数

    # ML統合パラメータ
    confidence_threshold = 0.6  # ML予測信頼度閾値
    fallback_mode = "skip_orders"  # フォールバック動作: "skip_orders" or "far_orders"
    log_predictions = True  # 予測詳細ログ

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
            self.fallback_mode = strategy_config.get("fallback_mode", self.fallback_mode)
            self.log_predictions = strategy_config.get("log_predictions", self.log_predictions)

        logger.info(f"ATRMLStrategy初期化: length={self.entry_length}, point={self.entry_point}")

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

            # FreqAI予測の取得（FreqAIが有効な場合）
            if hasattr(self, "freqai") and self.freqai:
                dataframe = self.freqai.start(dataframe, metadata, self)

            pair = metadata.get("pair", "unknown")
            logger.debug(f"指標計算完了: {pair}, レコード数={len(dataframe)}")

            return dataframe

        except Exception as e:
            pair = metadata.get("pair", "unknown")
            logger.error(f"指標計算エラー ({pair}): {e}")
            return dataframe

    def _calculate_atr_prices(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        ATR価格計算 - 要件 5.1

        Args:
            dataframe: OHLCデータ

        Returns:
            ATR価格が追加されたDataFrame
        """
        try:
            from user_data.strategies.utils.atr_calculator import ATRCalculator

            calculator = ATRCalculator(
                atr_period=self.entry_length, atr_multiplier=self.entry_point
            )

            return calculator.calculate_atr_prices(dataframe)

        except ImportError:
            logger.warning("ATRCalculatorが見つかりません。ATR価格計算をスキップします。")
            # フォールバック: 基本ATR計算
            dataframe["atr"] = qtpylib.atr(dataframe, timeperiod=self.entry_length)
            dataframe["atr_buy_price"] = dataframe["close"] - (dataframe["atr"] * self.entry_point)
            dataframe["atr_sell_price"] = dataframe["close"] + (dataframe["atr"] * self.entry_point)
            return dataframe

        except Exception as e:
            logger.error(f"ATR価格計算エラー: {e}")
            return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        2層判定エントリートレンド生成 - 要件 5.1

        1次モデル（ATR戦略）と2次モデル（ML分類）を統合した判定ロジック

        Args:
            dataframe: 指標付きDataFrame
            metadata: ペア情報

        Returns:
            エントリー信号が追加されたDataFrame
        """
        try:
            # エントリー信号カラムの初期化
            dataframe["enter_long"] = 0
            dataframe["enter_short"] = 0

            # データ十分性チェック
            if len(dataframe) < self.entry_length + 10:
                pair = metadata.get("pair", "unknown")
                logger.warning(
                    f"データ不足: {pair}, 必要={self.entry_length + 10}, 実際={len(dataframe)}"
                )
                return dataframe

            # ML予測の確認
            has_ml_prediction = "&-prediction" in dataframe.columns
            has_ml_probability = "&-probability" in dataframe.columns

            if not has_ml_prediction:
                pair = metadata.get("pair", "unknown")
                logger.warning(f"ML予測なし: {pair}, フォールバック={self.fallback_mode}")
                return self._handle_fallback_mode(dataframe, metadata)

            # 2層判定ロジック実行
            dataframe = self._apply_two_tier_logic(dataframe, metadata)

            # 予測詳細ログ
            if self.log_predictions:
                self._log_prediction_details(dataframe, metadata)

            return dataframe

        except Exception as e:
            pair = metadata.get("pair", "unknown")
            logger.error(f"エントリートレンド生成エラー ({pair}): {e}")
            # 安全なフォールバック
            dataframe["enter_long"] = 0
            dataframe["enter_short"] = 0
            return dataframe

    def _apply_two_tier_logic(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        2層判定ロジック適用 - 要件 5.1

        Args:
            dataframe: データフレーム
            metadata: ペア情報

        Returns:
            エントリー信号が設定されたDataFrame
        """
        # ATR価格データの存在確認
        required_columns = ["atr", "atr_buy_price", "atr_sell_price"]
        if not all(col in dataframe.columns for col in required_columns):
            pair = metadata.get("pair", "unknown")
            logger.warning(f"ATR価格データ不足: {pair}")
            return dataframe

        # ML予測条件
        ml_prediction = dataframe["&-prediction"] == 1

        # 信頼度フィルタリング（利用可能な場合）
        confidence_filter = True
        if "&-probability" in dataframe.columns and self.confidence_threshold > 0:
            confidence_filter = dataframe["&-probability"] >= self.confidence_threshold

        # ATR基本条件
        atr_valid = dataframe["atr"] > 0
        price_valid = (dataframe["atr_buy_price"] > 0) & (dataframe["atr_sell_price"] > 0)

        # 2層統合条件
        long_condition = (
            ml_prediction  # 2次モデル: ML予測=1
            & confidence_filter  # 信頼度フィルタ
            & atr_valid  # 1次モデル: ATR有効
            & price_valid  # 価格データ有効
        )

        short_condition = (
            ~ml_prediction  # 2次モデル: ML予測=0
            & confidence_filter  # 信頼度フィルタ
            & atr_valid  # 1次モデル: ATR有効
            & price_valid  # 価格データ有効
        )

        # エントリー信号設定
        dataframe.loc[long_condition, "enter_long"] = 1
        dataframe.loc[short_condition, "enter_short"] = 1

        return dataframe

    def _handle_fallback_mode(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        フォールバック動作処理 - 要件 5.2

        Args:
            dataframe: データフレーム
            metadata: ペア情報

        Returns:
            フォールバック処理されたDataFrame
        """
        if self.fallback_mode == "skip_orders":
            # 注文スキップモード：エントリー信号なし
            pair = metadata.get("pair", "unknown")
            logger.info(f"注文スキップモード実行: {pair}")
            dataframe["enter_long"] = 0
            dataframe["enter_short"] = 0

        elif self.fallback_mode == "far_orders":
            # 遠距離注文モード：ATRのみでエントリー
            pair = metadata.get("pair", "unknown")
            logger.info(f"遠距離注文モード実行: {pair}")

            # ATR基本条件のみ適用
            atr_valid = dataframe["atr"] > 0 if "atr" in dataframe.columns else False
            price_valid = (
                (
                    ("atr_buy_price" in dataframe.columns)
                    & ("atr_sell_price" in dataframe.columns)
                    & (dataframe["atr_buy_price"] > 0)
                    & (dataframe["atr_sell_price"] > 0)
                )
                if all(col in dataframe.columns for col in ["atr_buy_price", "atr_sell_price"])
                else False
            )

            # 基本的なロング条件（価格上昇トレンド）
            if atr_valid.any() and price_valid.any():
                basic_long = dataframe["close"] > dataframe["close"].shift(1)
                dataframe.loc[atr_valid & price_valid & basic_long, "enter_long"] = 1

        return dataframe

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
        ATR指値価格計算 - 要件 5.1

        Args:
            pair: 取引ペア
            trade: 取引オブジェクト
            current_time: 現在時刻
            proposed_rate: 提案価格
            entry_tag: エントリータグ
            side: 取引方向（"long" or "short"）
            **kwargs: 追加パラメータ

        Returns:
            ATRベース指値価格
        """
        try:
            # 最新のデータフレーム取得
            dataframe = self.dp.get_pair_dataframe(pair, self.timeframe)

            if dataframe.empty:
                logger.warning(f"データフレーム取得失敗: {pair}")
                return proposed_rate

            # 最新レコードのATR価格取得
            latest_idx = dataframe.index[-1]

            if side == "long":
                atr_price_col = "atr_buy_price"
            else:  # short
                atr_price_col = "atr_sell_price"

            if atr_price_col in dataframe.columns:
                atr_price = dataframe.loc[latest_idx, atr_price_col]

                if pd.isna(atr_price) or atr_price <= 0:
                    logger.warning(f"無効なATR価格: {pair}, {atr_price_col}={atr_price}")
                    return proposed_rate

                logger.info(
                    f"ATR指値価格: {pair} {side} {atr_price:.8f} (提案価格: {proposed_rate:.8f})"
                )
                return atr_price

            else:
                logger.warning(f"ATR価格カラム不足: {pair}, {atr_price_col}")
                return proposed_rate

        except Exception as e:
            logger.error(f"カスタムエントリー価格計算エラー ({pair}): {e}")
            return proposed_rate

    def _log_prediction_details(self, dataframe: pd.DataFrame, metadata: dict) -> None:
        """
        予測詳細ログ記録 - 要件 5.2

        Args:
            dataframe: データフレーム
            metadata: ペア情報
        """
        if not self.log_predictions:
            return

        try:
            pair = metadata["pair"]
            total_records = len(dataframe)

            # ML予測統計
            if "&-prediction" in dataframe.columns:
                predictions = dataframe["&-prediction"].dropna()
                if len(predictions) > 0:
                    positive_predictions = (predictions == 1).sum()
                    prediction_ratio = positive_predictions / len(predictions)

                    logger.info(
                        f"ML予測統計 ({pair}): 総数={len(predictions)}, "
                        f"正例={positive_predictions}, 正例率={prediction_ratio:.3f}"
                    )

            # エントリー信号統計
            long_signals = dataframe["enter_long"].sum()
            short_signals = dataframe["enter_short"].sum()

            logger.info(
                f"エントリー信号 ({pair}): ロング={long_signals}, "
                f"ショート={short_signals}, 総レコード={total_records}"
            )

            # 信頼度統計（利用可能な場合）
            if "&-probability" in dataframe.columns:
                probabilities = dataframe["&-probability"].dropna()
                if len(probabilities) > 0:
                    avg_confidence = probabilities.mean()
                    high_confidence = (probabilities >= self.confidence_threshold).sum()

                    logger.info(
                        f"信頼度統計 ({pair}): 平均={avg_confidence:.3f}, "
                        f"閾値以上={high_confidence}/{len(probabilities)}"
                    )

        except Exception as e:
            pair = metadata.get("pair", "unknown")
            logger.error(f"予測詳細ログエラー ({pair}): {e}")

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
