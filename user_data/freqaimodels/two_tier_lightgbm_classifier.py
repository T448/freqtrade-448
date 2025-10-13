"""TwoTier LightGBM Classifier for FreqAI

2層戦略用のLightGBM Multi-target分類モデル。
任意の1次戦略と組み合わせ可能な汎用的なML実装を提供する。
"""

import logging
from typing import Any

import pandas as pd
from lightgbm import LGBMClassifier

from freqtrade.freqai.base_models.BaseClassifierModel import BaseClassifierModel
from freqtrade.freqai.base_models.FreqaiMultiOutputClassifier import FreqaiMultiOutputClassifier
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen

logger = logging.getLogger(__name__)


class TwoTierLightGBMClassifier(BaseClassifierModel):
    """2層戦略用LightGBM Multi-target分類モデル

    FreqAIフレームワークのBaseClassifierModelを継承し、
    任意の1次戦略と組み合わせ可能な汎用的なML実装を提供する。

    特徴:
        - 1次戦略（ATRBreakout, MeanReversion等）とは独立
        - configで1次戦略と2次モデルを自由に組み合わせ可能
        - Buy/Sell独立したモデルとして訓練・予測（Multi-target対応）
        - 各ターゲットごとに独立したestimatorを訓練

    Multi-target対応:
        - &-buy, &-sell の2つのラベルを使用
        - 各ターゲットに対して独立したLightGBMモデルを訓練
        - 予測結果も &-buy, &-sell として出力

    Note:
        - populate_indicators(): テクニカル指標の特徴量生成
        - set_freqai_targets(): TwoTierStrategyから呼ばれるため最小限実装
        - fit(): FreqaiMultiOutputClassifierを使用したMulti-target訓練
    """

    def fit(self, data_dictionary: dict, dk: FreqaiDataKitchen, **kwargs) -> Any:
        """LightGBM Multi-targetモデルの訓練

        Buy/Sell独立したラベル（&-buy, &-sell）に対して、
        それぞれ独立したLightGBMモデルを訓練する。

        Args:
            data_dictionary: 訓練・テストデータ、ラベル、ウェイトを含む辞書
            dk: 現在のペア/モデル用のDataKitchenオブジェクト

        Returns:
            訓練済みFreqaiMultiOutputClassifier（複数のestimatorを含む）
        """
        lgb = LGBMClassifier(**self.model_training_parameters)

        X = data_dictionary["train_features"]
        y = data_dictionary["train_labels"]
        sample_weight = data_dictionary["train_weights"]

        eval_weights = None
        eval_sets = [None] * y.shape[1]

        # テストセットの設定（各ターゲットごと）
        if self.freqai_info.get("data_split_parameters", {}).get("test_size", 0.1) != 0:
            eval_weights = [data_dictionary["test_weights"]]
            eval_sets = [(None, None)] * data_dictionary["test_labels"].shape[1]
            for i in range(data_dictionary["test_labels"].shape[1]):
                eval_sets[i] = (
                    data_dictionary["test_features"],
                    data_dictionary["test_labels"].iloc[:, i],
                )

        # 継続学習用の初期モデル取得
        init_model = self.get_init_model(dk.pair)
        if init_model:
            init_models = init_model.estimators_
        else:
            init_models = [None] * y.shape[1]

        # 各ターゲット用のfit_params準備
        fit_params = []
        for i in range(len(eval_sets)):
            fit_params.append(
                {
                    "eval_set": eval_sets[i],
                    "eval_sample_weight": eval_weights,
                    "init_model": init_models[i],
                }
            )

        # Multi-outputモデルの訓練
        model = FreqaiMultiOutputClassifier(estimator=lgb)
        thread_training = self.freqai_info.get("multitarget_parallel_training", False)
        if thread_training:
            model.n_jobs = y.shape[1]
        model.fit(X=X, y=y, sample_weight=sample_weight, fit_params=fit_params)

        return model

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """ML特徴量生成（テクニカル指標等）

        1次戦略に依存しない汎用的な特徴量を計算する。
        %プレフィックス付きカラムとしてFreqAIに認識される。

        Args:
            dataframe: OHLCV価格データ
            metadata: ペア情報等のメタデータ

        Returns:
            特徴量カラム（%プレフィックス付き）が追加されたDataFrame
        """
        # 移動平均
        for period in [10, 20, 50]:
            dataframe[f"%ema_{period}"] = dataframe["close"].ewm(span=period, adjust=False).mean()
            dataframe[f"%sma_{period}"] = dataframe["close"].rolling(window=period).mean()

        # RSI（相対力指数）
        for period in [6, 14, 21]:
            delta = dataframe["close"].diff()
            gain = delta.where(delta > 0, 0).rolling(window=period).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
            rs = gain / loss
            dataframe[f"%rsi_{period}"] = 100 - (100 / (1 + rs))

        # MACD（移動平均収束拡散）
        exp1 = dataframe["close"].ewm(span=12, adjust=False).mean()
        exp2 = dataframe["close"].ewm(span=26, adjust=False).mean()
        dataframe["%macd"] = exp1 - exp2
        dataframe["%macd_signal"] = dataframe["%macd"].ewm(span=9, adjust=False).mean()
        dataframe["%macd_diff"] = dataframe["%macd"] - dataframe["%macd_signal"]

        # ボリンジャーバンド
        for period in [20]:
            sma = dataframe["close"].rolling(window=period).mean()
            std = dataframe["close"].rolling(window=period).std()
            dataframe[f"%bb_upper_{period}"] = sma + (std * 2)
            dataframe[f"%bb_lower_{period}"] = sma - (std * 2)
            dataframe[f"%bb_width_{period}"] = (
                dataframe[f"%bb_upper_{period}"] - dataframe[f"%bb_lower_{period}"]
            ) / sma

        # ATR（Average True Range）
        for period in [14, 20]:
            high = dataframe["high"]
            low = dataframe["low"]
            close = dataframe["close"]
            prev_close = close.shift(1)

            tr1 = high - low
            tr2 = (high - prev_close).abs()
            tr3 = (low - prev_close).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            dataframe[f"%atr_{period}"] = tr.rolling(window=period).mean()

        # ボリューム関連
        dataframe["%volume_mean_20"] = dataframe["volume"].rolling(window=20).mean()
        dataframe["%volume_ratio"] = dataframe["volume"] / dataframe["%volume_mean_20"]

        # 価格変化率
        for period in [1, 5, 10]:
            dataframe[f"%price_change_{period}"] = dataframe["close"].pct_change(periods=period)

        return dataframe

    def set_freqai_targets(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """FreqAI訓練用ラベル生成（最小限実装）

        実際のラベル生成はTwoTierStrategy.set_freqai_targets()で実装されるため、
        このメソッドは空または最小限の実装。

        Args:
            dataframe: 指標計算済みDataFrame
            metadata: ペア情報

        Returns:
            dataframe（変更なし、TwoTierStrategyで処理される）

        Note:
            TwoTierStrategyが1次戦略のcalculate_returns()を呼び出し、
            リターンをラベル化する実装を行う。
        """
        # TwoTierStrategy.set_freqai_targets()で実装されるため、ここでは何もしない
        return dataframe
