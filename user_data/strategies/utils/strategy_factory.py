"""戦略ファクトリー統合モジュール

1次モデル、2次モデル、エントリー戦略の選択機能を統合
"""

import logging
from typing import Dict, Any, Optional, Union, Tuple
from abc import ABC, abstractmethod
import pandas as pd

from .price_calculator import PriceCalculatorBase, PriceCalculatorFactory
from .freqai_model_factory import FreqAIModelFactory

logger = logging.getLogger(__name__)


class MLTrainerBase(ABC):
    """機械学習訓練の抽象基底クラス

    Note: 現在のFreqAI統合実装では、訓練・予測はfreqai.start()により自動実行されるため、
    このクラスのメソッドは直接呼び出されません。
    将来的にFreqAI以外のML統合や、カスタムトレーニングロジックを実装する際の
    インターフェースとして保持されています。
    """

    def __init__(self, params: Dict[str, Any]):
        self.params = params
        self.model = None
        self.is_trained = False

    @abstractmethod
    def train(self, X, y) -> Dict[str, Any]:
        """モデル訓練

        Note: FreqAI統合時はfreqai.start()が自動的に訓練を実行するため、
        このメソッドは直接呼び出されません。将来の拡張性のために定義されています。

        Returns:
            訓練結果メトリクス
        """
        pass

    @abstractmethod
    def predict(self, X):
        """予測実行

        Note: FreqAI統合時はfreqai.start()が自動的に予測を実行するため、
        このメソッドは直接呼び出されません。将来の拡張性のために定義されています。
        """
        pass

    @abstractmethod
    def predict_proba(self, X):
        """確率予測

        Note: FreqAI統合時はfreqai.start()が自動的に確率予測を実行するため、
        このメソッドは直接呼び出されません。将来の拡張性のために定義されています。
        """
        pass


class FreqAIMLTrainer(MLTrainerBase):
    """FreqAI統合ML訓練器

    Note: 実際の訓練・予測はfreqai.start()により自動実行されます。
    このクラスは設定管理と将来の拡張性のために保持されています。
    """

    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)
        self.model_type = params.get("type", "lightgbm_classifier")
        self.freqai_params = FreqAIModelFactory.validate_model_config(self.model_type, params)

    def train(self, X, y) -> Dict[str, Any]:
        """FreqAI統合による訓練（プレースホルダー実装）

        Note: 実際の訓練はfreqai.start()により自動実行されるため、このメソッドは呼び出されません。
        """
        logger.info(f"Training FreqAI model: {self.model_type}")
        self.is_trained = True
        return {"status": "trained", "model_type": self.model_type}

    def predict(self, X):
        """FreqAI統合による予測（プレースホルダー実装）

        Note: 実際の予測はfreqai.start()により自動実行されるため、このメソッドは呼び出されません。
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        return [1] * len(X)

    def predict_proba(self, X):
        """FreqAI統合による確率予測（プレースホルダー実装）

        Note: 実際の確率予測はfreqai.start()により自動実行されるため、このメソッドは呼び出されません。
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        return [0.7] * len(X)


class StrategyFactory:
    """統合戦略ファクトリー"""

    # 登録済み実装の辞書
    _ml_trainers = {
        "freqai": FreqAIMLTrainer,
    }

    @classmethod
    def create_two_tier_strategy(cls, config: Dict[str, Any]) -> "TwoTierStrategy":
        """2層戦略の作成

        Args:
            config: 戦略設定

        Returns:
            設定済み2層戦略インスタンス
        """
        # 1次モデル（価格計算）の作成
        primary_config = config.get("primary_model", {})
        primary_model = cls.create_primary_model(primary_config)

        # 2次モデル（ML）の作成
        secondary_config = config.get("secondary_model", {})
        secondary_model = cls.create_secondary_model(secondary_config)

        # FreqAI設定の生成
        freqai_config = FreqAIModelFactory.create_freqai_config(config)

        # 2層戦略の作成
        strategy = TwoTierStrategy(
            primary_model=primary_model,
            secondary_model=secondary_model,
            freqai_config=freqai_config,
            config=config,
        )

        return strategy

    @classmethod
    def create_primary_model(cls, config: Dict[str, Any]) -> PriceCalculatorBase:
        """1次モデル（価格計算）の作成"""
        model_type = config.get("type", "atr")
        params = config.get("params", {})

        return PriceCalculatorFactory.create_calculator(model_type, params)

    @classmethod
    def create_secondary_model(cls, config: Dict[str, Any]) -> Optional[MLTrainerBase]:
        """2次モデル（ML）の作成"""
        if not config.get("enabled", True):
            return None

        model_type = config.get("type", "lightgbm_classifier")
        params = config.get("params", {})

        # FreqAI統合ML訓練器を使用
        trainer_params = {"type": model_type, **params}

        trainer_class = cls._ml_trainers.get("freqai", FreqAIMLTrainer)
        return trainer_class(trainer_params)

    @classmethod
    def register_ml_trainer(cls, name: str, trainer_class: type):
        """新しいML訓練器の登録"""
        if not issubclass(trainer_class, MLTrainerBase):
            raise ValueError("Trainer class must inherit from MLTrainerBase")

        cls._ml_trainers[name] = trainer_class
        logger.info(f"Registered ML trainer: {name}")

    @classmethod
    def list_available_models(cls) -> Dict[str, list]:
        """利用可能なモデル一覧"""
        return {
            "primary_models": PriceCalculatorFactory.list_available_calculators(),
            "secondary_models": FreqAIModelFactory.list_available_models(),
            "ml_trainers": list(cls._ml_trainers.keys()),
        }


class TwoTierStrategy:
    """2層戦略実行クラス"""

    def __init__(
        self,
        primary_model: PriceCalculatorBase,
        secondary_model: Optional[MLTrainerBase],
        freqai_config: Dict[str, Any],
        config: Dict[str, Any],
    ):
        self.primary_model = primary_model
        self.secondary_model = secondary_model
        self.freqai_config = freqai_config
        self.config = config
        self.is_ml_enabled = secondary_model is not None

        logger.info(
            f"TwoTierStrategy initialized: "
            f"primary={type(primary_model).__name__}, "
            f"secondary={type(secondary_model).__name__ if secondary_model else 'None'}, "
            f"freqai_enabled={bool(freqai_config)}"
        )

    def generate_entry_signals(self, dataframe, metadata: Dict[str, Any]):
        """エントリー信号生成"""
        pair = metadata.get("pair", "unknown")

        try:
            # 1次モデル: 価格計算
            price_data = self.primary_model.calculate_entry_prices(dataframe)

            if self.is_ml_enabled:
                # 2次モデル有効時: ML統合戦略
                return self._generate_ml_integrated_signals(price_data, metadata)
            else:
                # 2次モデル無効時: 基本価格戦略
                return self._generate_basic_price_signals(price_data, metadata)

        except Exception as e:
            logger.error(f"Signal generation error {pair}: {e}")
            # フォールバック: 信号なし
            result = dataframe.copy()
            result["enter_long"] = 0
            result["enter_short"] = 0
            return result

    def _generate_ml_integrated_signals(self, dataframe, metadata: Dict[str, Any]):
        """ML統合信号生成"""
        pair = metadata.get("pair", "unknown")
        result = dataframe.copy()
        result["enter_long"] = 0
        result["enter_short"] = 0

        # ML予測データの取得と検証
        ml_data = self._extract_ml_prediction_data(dataframe, pair)
        if ml_data is None:
            return result

        # ML統合条件の計算
        long_condition, short_condition = self._calculate_ml_entry_conditions(
            dataframe, ml_data, self.config
        )

        # 信号設定
        result.loc[long_condition, "enter_long"] = 1
        result.loc[short_condition, "enter_short"] = 1

        self._log_signal_summary(pair, long_condition, short_condition)
        return result

    def _extract_ml_prediction_data(
        self, dataframe: pd.DataFrame, pair: str
    ) -> Optional[Dict[str, pd.Series]]:
        """ML予測データの抽出と検証

        Returns:
            ML予測データ辞書、または予測データが不足している場合はNone
        """
        has_prediction = "&-prediction" in dataframe.columns
        has_probability = "&-probability" in dataframe.columns

        if not has_prediction:
            logger.warning(f"ML prediction data missing for {pair}")
            return None

        return {
            "prediction": dataframe["&-prediction"] == 1,
            "probability": dataframe["&-probability"] if has_probability else None,
        }

    def _calculate_ml_entry_conditions(
        self, dataframe: pd.DataFrame, ml_data: Dict[str, pd.Series], config: Dict[str, Any]
    ) -> Tuple[pd.Series, pd.Series]:
        """ML統合エントリー条件の計算

        Returns:
            (long_condition, short_condition)のタプル
        """
        ml_prediction = ml_data["prediction"]
        ml_probability = ml_data["probability"]

        # 信頼度フィルタリング
        confidence_threshold = config.get("entry", {}).get("confidence_threshold", 0.6)
        confidence_filter = self._apply_confidence_threshold(
            ml_probability, confidence_threshold, dataframe.index
        )

        # 価格データの有効性チェック
        price_valid = (dataframe["buy_price"] > 0) & (dataframe["sell_price"] > 0)

        # ML統合条件
        long_condition = ml_prediction & confidence_filter & price_valid
        short_condition = ~ml_prediction & confidence_filter & price_valid

        return long_condition, short_condition

    def _apply_confidence_threshold(
        self, probability: Optional[pd.Series], threshold: float, index: pd.Index
    ) -> pd.Series:
        """信頼度閾値フィルタの適用

        Args:
            probability: ML予測確率（0-1の範囲）
            threshold: 信頼度閾値
            index: 結果Seriesのインデックス

        Returns:
            信頼度フィルタの条件Series

        Raises:
            ValueError: thresholdが不正な値の場合
        """
        if not (0 <= threshold <= 1):
            raise ValueError(f"Threshold must be between 0 and 1, got {threshold}")

        # probabilityがNoneの場合は、信頼度フィルタリングを無効化（全てTrue）
        if probability is None:
            if threshold > 0:
                logger.warning("Probability data not available, confidence filtering disabled")
            return pd.Series(True, index=index)

        return probability >= threshold

    def _log_signal_summary(
        self, pair: str, long_condition: pd.Series, short_condition: pd.Series
    ) -> None:
        """信号生成サマリーのログ出力"""
        long_signals = long_condition.sum()
        short_signals = short_condition.sum()
        logger.info(f"ML integrated signals {pair}: long={long_signals}, short={short_signals}")

    def _generate_basic_price_signals(self, dataframe, metadata: Dict[str, Any]):
        """基本価格信号生成"""
        pair = metadata.get("pair", "unknown")
        result = dataframe.copy()
        result["enter_long"] = 0
        result["enter_short"] = 0

        if len(dataframe) <= 1:
            return result

        # 前期間の価格と現在価格の比較
        prev_buy_price = dataframe["buy_price"].shift(1)
        prev_sell_price = dataframe["sell_price"].shift(1)
        current_close = dataframe["close"]

        # 価格戦略: 価格が指値レベルに到達した時の取引判定
        buy_signal = (current_close <= prev_buy_price) & (prev_buy_price > 0)
        sell_signal = (current_close >= prev_sell_price) & (prev_sell_price > 0)

        # 価格データの有効性チェック
        price_valid = (dataframe["buy_price"] > 0) & (dataframe["sell_price"] > 0)

        # エントリー条件
        long_condition = buy_signal & price_valid
        short_condition = sell_signal & price_valid

        # 信号設定
        result.loc[long_condition, "enter_long"] = 1
        result.loc[short_condition, "enter_short"] = 1

        long_signals = long_condition.sum()
        short_signals = short_condition.sum()
        logger.info(f"Basic price signals {pair}: long={long_signals}, short={short_signals}")

        return result

    def calculate_entry_price(self, pair: str, side: str, proposed_rate: float):
        """エントリー価格計算"""
        try:
            return self.primary_model.calculate_limit_price(proposed_rate, side)
        except Exception as e:
            logger.error(f"Entry price calculation error {pair}: {e}")
            return proposed_rate

    def get_freqai_config(self) -> Dict[str, Any]:
        """FreqAI設定を取得"""
        return self.freqai_config

    def get_strategy_info(self) -> Dict[str, Any]:
        """戦略情報を取得"""
        return {
            "primary_model": {
                "type": type(self.primary_model).__name__,
                "params": self.primary_model.params,
            },
            "secondary_model": {
                "enabled": self.is_ml_enabled,
                "type": type(self.secondary_model).__name__ if self.secondary_model else None,
                "params": self.secondary_model.params if self.secondary_model else None,
            },
            "freqai_enabled": bool(self.freqai_config),
            "config": self.config,
        }
