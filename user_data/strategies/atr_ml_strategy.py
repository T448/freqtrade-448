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


class ATRMLStrategy(IStrategy):
    """
    ATR機械学習統合戦略

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
        戦略初期化

        Args:
            config: Freqtrade設定辞書
            **kwargs: 追加パラメータ
        """
        super().__init__(config, **kwargs)

        # 戦略設定の読み込み
        strategy_config = self._load_strategy_config(config)

        # 2層戦略の作成
        self.two_tier_strategy = StrategyFactory.create_two_tier_strategy(strategy_config)

        # パラメータの展開
        primary_config = strategy_config.get("primary_model", {})
        self.entry_length = primary_config.get("params", {}).get("period", 14)
        self.entry_point = primary_config.get("params", {}).get("multiplier", 0.5)

        secondary_config = strategy_config.get("secondary_model", {})
        self.confidence_threshold = secondary_config.get("confidence_threshold", 0.6)

        # FreqAI有効性の確認
        self.freqai_enabled = (
            config and "freqai" in config and config["freqai"].get("enabled", False)
        )

        logger.info(
            f"ATRMLStrategy initialized: "
            f"primary={primary_config.get('type', 'unknown')}, "
            f"secondary={'enabled' if secondary_config.get('enabled') else 'disabled'}, "
            f"freqai={'enabled' if self.freqai_enabled else 'disabled'}"
        )

    def _load_strategy_config(self, config) -> dict:
        """戦略設定の読み込み

        Args:
            config: Freqtrade設定辞書

        Returns:
            戦略設定辞書
        """
        if not config or "two_tier_strategy" not in config:
            logger.warning("two_tier_strategy not found in config, using default")
            return get_strategy_config("price_only")

        two_tier_config = config["two_tier_strategy"]

        # プリセット指定がある場合
        if "preset" in two_tier_config:
            preset_name = two_tier_config["preset"]
            logger.info(f"Loading preset: {preset_name}")
            base_config = get_strategy_config(preset_name)
        else:
            # プリセットなしの場合はデフォルト設定を基準
            base_config = get_strategy_config("default")

        # config.jsonの直接設定で上書き
        if "primary_model" in two_tier_config:
            base_config["primary_model"] = two_tier_config["primary_model"]

        if "secondary_model" in two_tier_config:
            base_config["secondary_model"].update(two_tier_config["secondary_model"])

        logger.info(
            f"Strategy config loaded: "
            f"primary={base_config['primary_model']['type']}, "
            f"ml_enabled={base_config.get('secondary_model', {}).get('enabled', False)}"
        )

        return base_config

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

            # FreqAI予測の取得と指値価格調整
            if self.freqai_enabled:
                try:
                    dataframe = self.freqai.start(dataframe, metadata, self)

                    # ML予測に基づく指値価格調整（richmanbtc方式）
                    if "&-prediction" in dataframe.columns:
                        # 大きなオフセット（取引を実質的に無効化）
                        large_offset = dataframe["close"] * 0.1  # 10%のオフセット

                        # 予測が0（損益が0以下）の場合、指値を大きく外す
                        buy_pred = dataframe["&-prediction"]
                        sell_pred = dataframe["&-prediction"]

                        dataframe["buy_price"] = dataframe["buy_price"].where(
                            buy_pred != 0, dataframe["buy_price"] - large_offset
                        )
                        dataframe["sell_price"] = dataframe["sell_price"].where(
                            sell_pred != 0, dataframe["sell_price"] + large_offset
                        )

                        logger.debug("ML-based price adjustment applied")

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
        """FreqAI基本特徴量生成（条件付き）"""
        if not self.freqai_enabled:
            return dataframe  # FreqAI無効時は何もしない

        # FreqAI有効時のみ基本特徴量を追加
        dataframe["%-pct-change"] = dataframe["close"].pct_change()
        dataframe["%-raw_volume"] = dataframe["volume"]
        dataframe["%-raw_price"] = dataframe["close"]

        return dataframe

    def set_freqai_targets(self, dataframe: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """FreqAI目標設定（条件付き）"""
        if not self.freqai_enabled:
            return dataframe  # FreqAI無効時は何もしない

        try:
            # ATRリターンベースのラベル生成
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
