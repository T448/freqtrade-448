"""
リファクタリング済みATR機械学習統合戦略

設計文書に基づく2層戦略アーキテクチャの実装
- 設定駆動型による手法の柔軟な切り替え
- FreqAI機能の最大活用
- 統合ファクトリーパターンによる保守性向上
"""

import logging
import os
import sys
from typing import Optional

import pandas as pd
from freqtrade.strategy import IStrategy

# パス設定
sys.path.append(os.path.dirname(__file__))

# リファクタリング済みモジュールのインポート
from utils.strategy_factory import StrategyFactory, TwoTierStrategy
from utils.strategy_config import get_strategy_config
from utils.freqai_model_factory import FreqAIModelFactory

logger = logging.getLogger(__name__)


class ATRMLStrategyRefactored(IStrategy):
    """
    リファクタリング済みATR機械学習統合戦略

    設定駆動型により、コード変更なしで以下の切り替えが可能：
    - 1次モデル: ATR, ボリンジャーバンド, 移動平均等
    - 2次モデル: LightGBM, XGBoost, CatBoost等
    - 戦略設定: デフォルト, 高頻度, 保守的等
    """

    # 戦略基本設定
    INTERFACE_VERSION: int = 3
    minimal_roi = {"0": 0.05, "30": 0.03, "60": 0.01, "120": 0}
    stoploss = -0.10
    timeframe = "15m"
    process_only_new_candles = True

    # FreqAI統合設定
    can_short = True
    use_exit_signal = True

    def __init__(self, config=None, **kwargs):
        """
        戦略初期化（設定駆動型）

        Args:
            config: Freqtrade設定辞書
            **kwargs: 追加パラメータ
        """
        super().__init__(config, **kwargs)

        # 戦略設定の選択（環境変数・設定ファイルから選択可能）
        strategy_type = self._get_strategy_type(config)
        strategy_config = get_strategy_config(strategy_type)

        # 統合戦略ファクトリーで2層戦略を作成
        self.two_tier_strategy = StrategyFactory.create_two_tier_strategy(strategy_config)

        # 設定パラメータの展開（後方互換性維持）
        primary_config = strategy_config.get("primary_model", {})
        self.entry_length = primary_config.get("params", {}).get("period", 14)
        self.entry_point = primary_config.get("params", {}).get("multiplier", 0.5)

        entry_config = strategy_config.get("entry", {})
        self.confidence_threshold = entry_config.get("confidence_threshold", 0.6)

        # FreqAI設定の統合（自動生成）
        freqai_config = self.two_tier_strategy.get_freqai_config()
        if freqai_config and config:
            config["freqai"] = freqai_config
            logger.info("FreqAI configuration automatically integrated")

        logger.info(
            f"ATRMLStrategyRefactored initialized: "
            f"strategy_type={strategy_type}, "
            f"primary={primary_config.get('type', 'unknown')}, "
            f"secondary={'enabled' if strategy_config.get('secondary_model', {}).get('enabled') else 'disabled'}"
        )

    def _get_strategy_type(self, config) -> str:
        """戦略タイプの決定

        Args:
            config: Freqtrade設定

        Returns:
            戦略タイプ名
        """
        # 優先順位: 設定ファイル > 環境変数 > デフォルト
        strategy_type = None

        if config and "atr_ml_strategy" in config:
            strategy_type = config["atr_ml_strategy"].get("strategy_type")

        if not strategy_type:
            strategy_type = os.environ.get("STRATEGY_TYPE")

        if not strategy_type:
            strategy_type = "default"

        logger.info(f"Selected strategy type: {strategy_type}")
        return strategy_type

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        指標計算（統合アーキテクチャ）

        Args:
            dataframe: OHLC データ
            metadata: ペア情報

        Returns:
            指標が追加されたDataFrame
        """
        pair = metadata.get("pair", "unknown")

        try:
            # 統合戦略による価格計算
            dataframe = self.two_tier_strategy.primary_model.calculate_entry_prices(dataframe)

            # FreqAI予測の取得（自動統合）
            freqai_enabled = (
                hasattr(self, "freqai")
                and self.freqai is not None
                and getattr(self.freqai, "enabled", False)
            )

            if freqai_enabled:
                try:
                    dataframe = self.freqai.start(dataframe, metadata, self)
                    logger.debug("FreqAI prediction data integrated successfully")
                except Exception as freqai_error:
                    logger.warning(f"FreqAI execution failed: {freqai_error}")
                    logger.debug("Continuing with primary model only")
            else:
                logger.debug("FreqAI disabled - running with primary model only")

            logger.debug(f"Indicators calculation completed: {pair}, records={len(dataframe)}")
            return dataframe

        except Exception as e:
            logger.error(f"Indicators calculation error ({pair}): {e}")
            return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        エントリートレンド生成（統合戦略）

        Args:
            dataframe: 指標付きDataFrame
            metadata: ペア情報

        Returns:
            エントリー信号が追加されたDataFrame
        """
        try:
            # 統合戦略でエントリー信号生成
            return self.two_tier_strategy.generate_entry_signals(dataframe, metadata)

        except Exception as e:
            pair = metadata.get("pair", "unknown")
            logger.error(f"Entry trend generation error {pair}: {e}")
            # 安全なフォールバック
            dataframe["enter_long"] = 0
            dataframe["enter_short"] = 0
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
        エントリー価格計算（統合戦略）

        Args:
            pair: 取引ペア
            side: 取引方向

        Returns:
            統合戦略による指値価格
        """
        try:
            # 統合戦略でエントリー価格計算
            return self.two_tier_strategy.calculate_entry_price(pair, side, proposed_rate)

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

    def feature_engineering_expand_basic(self, dataframe: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        FreqAI基本特徴量生成

        Args:
            dataframe: OHLC データ

        Returns:
            基本特徴量が追加されたDataFrame
        """
        # FreqAI標準の基本特徴量生成
        dataframe["%-pct-change"] = dataframe["close"].pct_change()
        dataframe["%-raw_volume"] = dataframe["volume"]
        dataframe["%-raw_price"] = dataframe["close"]

        return dataframe

    def set_freqai_targets(self, dataframe: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        FreqAI目標設定

        Args:
            dataframe: データフレーム

        Returns:
            目標列が追加されたDataFrame
        """
        # ATRリターンベースのラベル生成
        try:
            # 統合戦略による2次モデル訓練
            training_result = self.two_tier_strategy.train_secondary_model(dataframe, kwargs)

            # 簡易ラベル生成（実際のFreqAI統合では更に詳細）
            look_ahead = 24
            future_return = dataframe["close"].pct_change(look_ahead).shift(-look_ahead)
            dataframe["&-target"] = (future_return > 0.001).astype(int)

        except Exception as e:
            logger.warning(f"FreqAI target generation error: {e}")
            # フォールバック: 中性ラベル
            dataframe["&-target"] = 0

        return dataframe

    def get_strategy_info(self) -> dict:
        """戦略情報の取得

        Returns:
            戦略情報辞書
        """
        return self.two_tier_strategy.get_strategy_info()

    def leverage(
        self,
        pair: str,
        current_time,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        entry_tag: Optional[str],
        side: str,
        **kwargs,
    ) -> float:
        """
        レバレッジ設定

        Returns:
            適用するレバレッジ倍率
        """
        return 1.0  # 現物取引相当
