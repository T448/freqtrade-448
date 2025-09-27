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
            else:
                logger.info("FreqAI無効 - ATRのみでバックテスト実行")

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
        # 基本ATR計算を使用
        import talib as ta

        dataframe["atr"] = ta.ATR(
            dataframe["high"],
            dataframe["low"],
            dataframe["close"],
            timeperiod=self.entry_length,
        )
        dataframe["atr_buy_price"] = dataframe["close"] - (dataframe["atr"] * self.entry_point)
        dataframe["atr_sell_price"] = dataframe["close"] + (dataframe["atr"] * self.entry_point)
        logger.info(
            f"🔢 基本ATR計算完了: period={self.entry_length}, multiplier={self.entry_point}"
        )
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

            # 2層判定ロジック実行（FreqAI無効時はML予測を1に設定）
            dataframe = self._apply_two_tier_logic(
                dataframe, metadata, force_ml_positive=not has_ml_prediction
            )

            # 予測詳細ログ
            self._log_prediction_details(dataframe, metadata)

            return dataframe

        except Exception as e:
            pair = metadata.get("pair", "unknown")
            logger.error(f"エントリートレンド生成エラー ({pair}): {e}")
            # 安全なフォールバック
            dataframe["enter_long"] = 0
            dataframe["enter_short"] = 0
            return dataframe

    def _apply_two_tier_logic(
        self, dataframe: pd.DataFrame, metadata: dict, force_ml_positive: bool = False
    ) -> pd.DataFrame:
        """
        2層判定ロジック適用 - 要件 5.1

        Args:
            dataframe: データフレーム
            metadata: ペア情報

        Returns:
            エントリー信号が設定されたDataFrame
        """
        pair = metadata.get("pair", "unknown")

        # ATR価格データの存在確認
        required_columns = ["atr", "atr_buy_price", "atr_sell_price"]
        if not all(col in dataframe.columns for col in required_columns):
            logger.warning(f"🚨 ATR価格データ不足: {pair}")
            return dataframe

        # ML予測データの存在確認とログ
        has_ml_prediction = "&-prediction" in dataframe.columns
        has_ml_probability = "&-probability" in dataframe.columns

        if has_ml_prediction:
            recent_predictions = dataframe["&-prediction"].tail(5)
            positive_count = (recent_predictions == 1).sum()
            logger.info(f"🤖 ML予測状況 ({pair}): 最新5件中 {positive_count}件が買い予測")

            if has_ml_probability:
                recent_prob = dataframe["&-probability"].tail(5)
                avg_confidence = recent_prob.mean()
                logger.info(
                    f"🎯 ML信頼度 ({pair}): 平均 {avg_confidence:.3f}, 閾値 {self.confidence_threshold}"
                )

        # ATR基本条件とログ
        atr_valid = dataframe["atr"] > 0
        price_valid = (dataframe["atr_buy_price"] > 0) & (dataframe["atr_sell_price"] > 0)

        if len(dataframe) > 0:
            atr_valid_count = atr_valid.sum()
            price_valid_count = price_valid.sum()
            logger.info(
                f"📊 ATR有効性 ({pair}): ATR={atr_valid_count}/{len(dataframe)}, 価格={price_valid_count}/{len(dataframe)}"
            )

        # ML無効時は基本ATR戦略ロジックを使用
        if force_ml_positive:
            logger.info(f"🔧 FreqAI無効モード ({pair}): 基本ATR戦略で判定")

            # ATR戦略の基本ロジック：価格がATR指値に到達した場合の判定
            # 前の期間のATR価格と現在の価格を比較
            atr_buy_signal = pd.Series([False] * len(dataframe), index=dataframe.index)
            atr_sell_signal = pd.Series([False] * len(dataframe), index=dataframe.index)

            # 前期間のATR価格と現在価格の比較による売買判定
            if len(dataframe) > 1:
                # 現在価格が前期間のATR買い価格以下でlong signal
                prev_atr_buy = dataframe["atr_buy_price"].shift(1)
                prev_atr_sell = dataframe["atr_sell_price"].shift(1)
                current_close = dataframe["close"]

                # ATR戦略: 価格がATR指値レベルに到達した時の取引判定
                atr_buy_signal = (current_close <= prev_atr_buy) & (prev_atr_buy > 0)
                atr_sell_signal = (current_close >= prev_atr_sell) & (prev_atr_sell > 0)

                buy_signals = atr_buy_signal.sum()
                sell_signals = atr_sell_signal.sum()
                logger.info(f"📈 ATR基本判定 ({pair}): 買い={buy_signals}, 売り={sell_signals}")

            # 基本ATR戦略での条件
            long_condition = (
                atr_buy_signal  # ATR買いシグナル
                & atr_valid  # ATR有効
                & price_valid  # 価格データ有効
            )

            short_condition = (
                atr_sell_signal  # ATR売りシグナル
                & atr_valid  # ATR有効
                & price_valid  # 価格データ有効
            )

        else:
            # ML有効時は2層統合ロジック
            ml_prediction = dataframe["&-prediction"] == 1 if has_ml_prediction else False

            # 信頼度フィルタリング（利用可能な場合）
            confidence_filter = True
            if has_ml_probability and self.confidence_threshold > 0:
                confidence_filter = dataframe["&-probability"] >= self.confidence_threshold
                high_confidence_count = confidence_filter.sum()
                logger.info(f"✅ 高信頼度予測 ({pair}): {high_confidence_count}/{len(dataframe)}件")

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

        # エントリー信号設定とログ
        long_signals = long_condition.sum()
        short_signals = short_condition.sum()

        dataframe.loc[long_condition, "enter_long"] = 1
        dataframe.loc[short_condition, "enter_short"] = 1

        logger.info(
            f"🎯 エントリー信号生成 ({pair}): ロング={long_signals}, ショート={short_signals}"
        )

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
            side: 取引方向（"long" or "short"）

        Returns:
            ATRベース指値価格
        """
        try:
            # 最新のデータフレーム取得（ATR計算済み）
            dataframe = self.dp.get_pair_dataframe(pair, self.timeframe)

            if dataframe.empty:
                logger.warning(f"データフレーム取得失敗: {pair}")
                return proposed_rate

            # ATR計算（軽量版 - 最新データのみ）
            import talib as ta

            current_atr = ta.ATR(
                dataframe["high"],
                dataframe["low"],
                dataframe["close"],
                timeperiod=self.entry_length,
            ).iloc[-1]

            current_close = dataframe["close"].iloc[-1]

            # ATR価格計算
            if side == "long":
                atr_price = current_close - (current_atr * self.entry_point)
            else:  # short
                atr_price = current_close + (current_atr * self.entry_point)

            # データ検証
            if pd.isna(atr_price) or pd.isna(current_atr) or current_atr <= 0:
                logger.warning(f"🚨 無効なATR計算: {pair}, ATR={current_atr}")
                return proposed_rate

            logger.info(f"💰 ATR指値価格 ({pair} {side}): {atr_price:.8f}")
            return atr_price

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
