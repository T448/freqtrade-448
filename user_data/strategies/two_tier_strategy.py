"""TwoTierStrategy: Config駆動の2層取引戦略

このモジュールは、FreqtradeのIStrategyを継承し、config.jsonで指定された
1次戦略とFreqAIモデルを動的にロード・統合する戦略実装を提供します。

Phase 3: ML無効モード対応（基本統合）
- 1次戦略で価格計算のみ実行
- 常時エントリー戦略（ML予測なし）
- カスタム指値価格による注文

Phase 4: ML有効モード対応（FreqAI統合）
- Buy/Sell独立したFreqAIモデル予測
- ML予測に基づくエントリー/エグジット判定
- 1次戦略のリターン計算に基づくラベル生成
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
        """指標計算（価格計算 + ML予測統合）

        1次戦略で指値価格（buy_price, sell_price）を計算し、
        ML有効時はBuy/Sell独立したFreqAI予測を統合する

        Args:
            dataframe: OHLCV価格データ
            metadata: ペア情報等のメタデータ

        Returns:
            buy_price, sell_priceカラム + ML予測カラムが追加されたDataFrame
        """
        # 1次戦略: 指値価格計算
        dataframe = self.primary_strategy.calculate_prices(dataframe)

        # FreqAI予測の統合（ML有効時のみ）
        if self.is_ml_enabled:
            # FreqAIモデルの特徴量生成 + 予測実行
            dataframe = self.freqai.start(dataframe, metadata, self)

            # &-prediction カラムを &-prediction_buy にリネーム
            if "&-prediction" in dataframe.columns:
                dataframe["&-prediction_buy"] = dataframe["&-prediction"]
                dataframe.drop(columns=["&-prediction"], inplace=True)

            # Note: マルチターゲット設定の場合、freqai_buy/freqai_sellとして
            # 2つの独立したFreqAIインスタンスを使用する
            # 現在はシングルモデル実装（Phase 4基本版）
            # Buy予測をSellにもコピー（同じモデルを使用）
            if "&-prediction_buy" in dataframe.columns:
                dataframe["&-prediction_sell"] = dataframe["&-prediction_buy"]

        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """エントリーシグナル生成（ML予測による判定）

        両建て対応: buy/sellを独立して判定
        - ML有効時: 各方向の予測=1の場合のみエントリー
        - ML無効時: 常に両方向エントリー（指値価格があれば注文）

        Args:
            dataframe: 指標計算済みDataFrame
            metadata: ペア情報

        Returns:
            enter_long, enter_shortシグナルが設定されたDataFrame
        """
        if self.is_ml_enabled:
            # ML予測が1の場合のみエントリー（buy/sell独立）
            dataframe.loc[(dataframe["&-prediction_buy"] == 1), "enter_long"] = 1
            dataframe.loc[(dataframe["&-prediction_sell"] == 1), "enter_short"] = 1
        else:
            # ML無効時は常に両方向エントリー（価格が有効な場合）
            # 価格有効性チェック: buy_price/sell_price > 0
            dataframe.loc[(dataframe["buy_price"] > 0), "enter_long"] = 1
            dataframe.loc[(dataframe["sell_price"] > 0), "enter_short"] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """エグジットシグナル生成（ML予測による決済）

        ML予測に基づく明示的な決済シグナル生成。
        Freqtradeは反対売買による自動決済をサポートしていないため、
        exit_long/exit_short で明示的に決済指示が必要。

        両建て状態（long + short 同時保有）では、両方のexitシグナルが
        同時に発生する可能性があり、その場合Freqtradeは両ポジションを決済する。

        Args:
            dataframe: 指標計算済みDataFrame
            metadata: ペア情報

        Returns:
            exit_long, exit_shortシグナルが設定されたDataFrame
        """
        if self.is_ml_enabled:
            # ロング決済: sell予測=1の場合
            dataframe.loc[(dataframe["&-prediction_sell"] == 1), "exit_long"] = 1

            # ショート決済: buy予測=1の場合
            dataframe.loc[(dataframe["&-prediction_buy"] == 1), "exit_short"] = 1

        # ML無効時は明示的な決済シグナルなし（ROI/Stoplossのみ）

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

    def set_freqai_targets(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """FreqAI訓練用ラベル生成

        1次戦略のリターン計算結果をラベル化する。
        buy/sell独立したラベルを生成（両建て対応）。
        リターン > 0 で成功ラベル（1）、それ以外は失敗ラベル（0）。

        Args:
            dataframe: 指標計算済みDataFrame
            metadata: ペア情報

        Returns:
            &-targetカラムが追加されたDataFrame

        Note:
            - 1次戦略のcalculate_returns()で約定シミュレーションを実行
            - execution_mode (chase/one_candle) に応じて異なるラベルが生成される
            - FreqAIフレームワークが&-targetカラムを訓練ラベルとして使用
        """
        # 1次戦略: buy/sellそれぞれのリターン計算
        buy_return, sell_return = self.primary_strategy.calculate_returns(dataframe)

        # リターン > 0 で成功ラベル（1）、それ以外は失敗ラベル（0）
        # 現在はシングルモデル実装のため、buyラベルのみ使用
        # マルチターゲット実装時はidentifierで判定する
        dataframe["&-target"] = (buy_return > 0).astype(int)

        # デバッグ用: ラベル分布をログ出力
        positive_ratio = dataframe["&-target"].mean()
        logger.info(
            f"Label generation for {metadata.get('pair', 'unknown')}: "
            f"positive_ratio={positive_ratio:.3f}, "
            f"total_samples={len(dataframe)}"
        )

        return dataframe
