"""TwoTierStrategy: Config駆動の2層取引戦略

このモジュールは、FreqtradeのIStrategyを継承し、config.jsonで指定された
1次戦略とFreqAIモデルを動的にロード・統合する戦略実装を提供します。

Phase 3: ML無効モード対応（基本統合）
- 1次戦略で価格計算のみ実行
- 常時エントリー戦略（ML予測なし）
- カスタム指値価格による注文
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
from freqtrade.strategy import IStrategy

# user_dataディレクトリをパスに追加
user_data_path = Path(__file__).parent.parent
if str(user_data_path) not in sys.path:
    sys.path.insert(0, str(user_data_path))

from strategies.utils.strategy_factory import PrimaryStrategyFactory

logger = logging.getLogger(__name__)


class TwoTierStrategy(IStrategy):
    """Config駆動の2層取引戦略（Freqtradeエントリーポイント）

    FreqtradeのIStrategyを継承し、config.jsonで指定された
    1次戦略と2次モデル（FreqAI）を動的にロード・統合する

    実行例:
        # ML無効モード
        freqtrade backtesting --strategy TwoTierStrategy --config config_ml_off.json

        # ML有効モード（Phase 4で実装）
        freqtrade backtesting --strategy TwoTierStrategy --config config_ml_on.json

    Attributes:
        primary_strategy: 1次戦略インスタンス（価格計算を担当）
        is_ml_enabled: FreqAI有効/無効フラグ
    """

    # Freqtradeの最小設定
    minimal_roi = {"0": 0.10}
    stoploss = -0.10
    timeframe = "5m"

    def __init__(self, config: dict):
        """TwoTierStrategyの初期化

        Config検証と1次戦略のロードを実行

        Args:
            config: Freqtrade設定辞書

        Raises:
            ValueError: Config検証エラー（freqai.enabledとsecondaryの不整合）
        """
        super().__init__(config)

        two_tier_config = config.get("two_tier_strategy", {})
        freqai_config = config.get("freqai", {})

        # Config validation: freqai.enabled と secondary の整合性チェック
        freqai_enabled = freqai_config.get("enabled", False)
        has_secondary = two_tier_config.get("secondary") is not None

        if has_secondary and not freqai_enabled:
            raise ValueError(
                "Invalid configuration: secondary model is specified but freqai.enabled is False. "
                "Please set freqai.enabled=true when using a secondary model."
            )

        if freqai_enabled and not has_secondary:
            raise ValueError(
                "Invalid configuration: freqai.enabled is True but no secondary model specified. "
                "Please set secondary to a model name (e.g., 'lightgbm_classifier') or disable FreqAI."
            )

        # 1次戦略をロード
        self.primary_strategy = PrimaryStrategyFactory.load_primary(two_tier_config)
        self.is_ml_enabled = freqai_enabled

        logger.info(
            f"TwoTierStrategy initialized: "
            f"primary={type(self.primary_strategy).__name__}, "
            f"freqai_enabled={freqai_enabled}"
        )

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """指標計算（Phase 3: ML無効モード - 価格計算のみ）

        1次戦略で指値価格（buy_price, sell_price）を計算する
        ML有効時の予測統合はPhase 4で実装

        Args:
            dataframe: OHLCV価格データ
            metadata: ペア情報等のメタデータ

        Returns:
            buy_price, sell_priceカラムが追加されたDataFrame
        """
        # 1次戦略: 指値価格計算
        dataframe = self.primary_strategy.calculate_prices(dataframe)

        # TODO Phase 4: FreqAI予測の統合
        # if self.is_ml_enabled:
        #     dataframe = self.freqai_buy.start(dataframe, metadata, self)
        #     dataframe.rename(columns={'&-prediction': '&-prediction_buy'}, inplace=True)
        #
        #     dataframe = self.freqai_sell.start(dataframe, metadata, self)
        #     dataframe.rename(columns={'&-prediction': '&-prediction_sell'}, inplace=True)

        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """エントリーシグナル生成（Phase 3: ML無効モード - 常時エントリー）

        ML無効時は常に両方向エントリー（指値価格があれば注文）
        両建て対応: buy/sellを独立して判定

        Args:
            dataframe: 指標計算済みDataFrame
            metadata: ペア情報

        Returns:
            enter_long, enter_shortシグナルが設定されたDataFrame
        """
        if self.is_ml_enabled:
            # TODO Phase 4: ML予測による判定
            # dataframe.loc[(dataframe['&-prediction_buy'] == 1), 'enter_long'] = 1
            # dataframe.loc[(dataframe['&-prediction_sell'] == 1), 'enter_short'] = 1
            pass
        else:
            # ML無効時は常に両方向エントリー（価格が有効な場合）
            # 価格有効性チェック: buy_price/sell_price > 0
            dataframe.loc[(dataframe["buy_price"] > 0), "enter_long"] = 1

            dataframe.loc[(dataframe["sell_price"] > 0), "enter_short"] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """エグジットシグナル生成

        Phase 3では明示的な決済シグナルは生成しない
        ROI/Stoplossによる自動決済のみ

        Phase 4でML予測に基づく決済シグナルを実装予定

        Args:
            dataframe: 指標計算済みDataFrame
            metadata: ペア情報

        Returns:
            exit_long, exit_shortシグナルが設定されたDataFrame
        """
        # Phase 3では明示的な決済シグナルなし
        # TODO Phase 4: ML予測による決済
        # if self.is_ml_enabled:
        #     dataframe.loc[(dataframe['&-prediction_sell'] == 1), 'exit_long'] = 1
        #     dataframe.loc[(dataframe['&-prediction_buy'] == 1), 'exit_short'] = 1

        return dataframe

    def custom_entry_price(
        self,
        pair: str,
        current_time,
        proposed_rate: float,
        entry_tag: Optional[str] = None,
        **kwargs,
    ) -> float:
        """エントリー指値価格（1次戦略の計算結果を使用）

        1次戦略で計算されたbuy_priceを指値として使用
        データが取得できない場合はproposed_rate（市場価格）を使用

        Args:
            pair: 通貨ペア
            current_time: 現在時刻
            proposed_rate: Freqtradeが提案する価格（現在の市場価格）
            entry_tag: エントリータグ（オプション）

        Returns:
            エントリー指値価格
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        if len(dataframe) > 0:
            latest_buy_price = dataframe.iloc[-1]["buy_price"]

            # 価格有効性チェック
            if latest_buy_price > 0:
                return latest_buy_price

        # フォールバック: 市場価格を使用
        logger.debug(f"Using proposed_rate for {pair}: {proposed_rate}")
        return proposed_rate

    def custom_exit_price(
        self, pair: str, trade, current_time, proposed_rate: float, **kwargs
    ) -> float:
        """エグジット指値価格（1次戦略の計算結果を使用）

        1次戦略で計算されたsell_priceを指値として使用
        データが取得できない場合はproposed_rate（市場価格）を使用

        Args:
            pair: 通貨ペア
            trade: トレード情報
            current_time: 現在時刻
            proposed_rate: Freqtradeが提案する価格（現在の市場価格）

        Returns:
            エグジット指値価格
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        if len(dataframe) > 0:
            latest_sell_price = dataframe.iloc[-1]["sell_price"]

            # 価格有効性チェック
            if latest_sell_price > 0:
                return latest_sell_price

        # フォールバック: 市場価格を使用
        logger.debug(f"Using proposed_rate for {pair}: {proposed_rate}")
        return proposed_rate

    # TODO Phase 4: FreqAIラベル生成実装
    # def set_freqai_targets(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    #     """FreqAI訓練用ラベル生成
    #
    #     1次戦略のリターン計算結果をラベル化
    #     buy/sell独立したラベルを生成（両建て対応）
    #     リターン > 0 で成功ラベル（1）、それ以外は失敗ラベル（0）
    #     """
    #     # 1次戦略: buy/sellそれぞれのリターン計算
    #     buy_return, sell_return = self.primary_strategy.calculate_returns(dataframe)
    #
    #     # リターン > 0 で成功ラベル
    #     dataframe['&-target_buy'] = (buy_return > 0).astype(int)
    #     dataframe['&-target_sell'] = (sell_return > 0).astype(int)
    #
    #     return dataframe
