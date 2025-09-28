"""戦略ファクトリー統合モジュール

1次モデル、2次モデル、エントリー戦略の選択機能を統合
"""

import logging
from typing import Dict, Any, Optional, Union
from abc import ABC, abstractmethod

from .price_calculator import PriceCalculatorBase, PriceCalculatorFactory
from .freqai_model_factory import FreqAIModelFactory

logger = logging.getLogger(__name__)


class MLTrainerBase(ABC):
    """機械学習訓練の抽象基底クラス"""

    def __init__(self, params: Dict[str, Any]):
        self.params = params
        self.model = None
        self.is_trained = False

    @abstractmethod
    def train(self, X, y) -> Dict[str, Any]:
        """モデル訓練

        Returns:
            訓練結果メトリクス
        """
        pass

    @abstractmethod
    def predict(self, X):
        """予測実行"""
        pass

    @abstractmethod
    def predict_proba(self, X):
        """確率予測"""
        pass


class FreqAIMLTrainer(MLTrainerBase):
    """FreqAI統合ML訓練器"""

    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)
        self.model_type = params.get("type", "lightgbm_classifier")
        self.freqai_params = FreqAIModelFactory.validate_model_config(self.model_type, params)

    def train(self, X, y) -> Dict[str, Any]:
        """FreqAI統合による訓練（プレースホルダー実装）"""
        # 実際の実装ではFreqAIのトレーニングパイプラインを使用
        logger.info(f"Training FreqAI model: {self.model_type}")
        self.is_trained = True
        return {"status": "trained", "model_type": self.model_type}

    def predict(self, X):
        """FreqAI統合による予測"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        # プレースホルダー実装
        return [1] * len(X)

    def predict_proba(self, X):
        """FreqAI統合による確率予測"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        # プレースホルダー実装
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
        confidence_threshold = self.config.get("entry", {}).get("confidence_threshold", 0.6)

        if has_ml_probability and confidence_threshold > 0:
            confidence_filter = dataframe["&-probability"] >= confidence_threshold

        # 価格データの有効性チェック
        price_valid = (dataframe["buy_price"] > 0) & (dataframe["sell_price"] > 0)

        # ML統合条件
        long_condition = ml_prediction & confidence_filter & price_valid
        short_condition = ~ml_prediction & confidence_filter & price_valid

        # 信号設定
        result.loc[long_condition, "enter_long"] = 1
        result.loc[short_condition, "enter_short"] = 1

        long_signals = long_condition.sum()
        short_signals = short_condition.sum()
        logger.info(f"ML integrated signals {pair}: long={long_signals}, short={short_signals}")

        return result

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

    def train_secondary_model(self, dataframe, metadata: Dict[str, Any]):
        """2次モデル訓練（必要時のみ）"""
        if not self.is_ml_enabled:
            logger.info("Secondary model disabled, skipping training")
            return None

        pair = metadata.get("pair", "unknown")
        logger.info(f"Training secondary model for {pair}")

        try:
            # 簡易特徴量生成（実際の実装ではより詳細な特徴量を使用）
            features_df = self._generate_basic_features(dataframe)

            # ラベル生成（実際の実装ではより精密なラベル生成を使用）
            labels = self._generate_basic_labels(dataframe)

            if len(features_df) < 100:  # 最小サンプル数チェック
                logger.warning(f"Insufficient training samples for {pair}: {len(features_df)}")
                return None

            # モデル訓練
            training_result = self.secondary_model.train(features_df, labels)

            logger.info(f"Secondary model training completed for {pair}: {training_result}")
            return training_result

        except Exception as e:
            logger.error(f"Secondary model training error {pair}: {e}")
            return None

    def _generate_basic_features(self, dataframe):
        """基本特徴量生成（プレースホルダー実装）"""
        # 実際の実装では技術指標エンジンを使用
        import talib as ta

        features = dataframe[["close", "volume"]].copy()
        features["rsi"] = ta.RSI(dataframe["close"])
        features["sma_20"] = ta.SMA(dataframe["close"], timeperiod=20)

        return features.dropna()

    def _generate_basic_labels(self, dataframe):
        """基本ラベル生成（プレースホルダー実装）"""
        # 簡易リターンベースラベル
        returns = dataframe["close"].pct_change(1).shift(-1)
        labels = (returns > 0.001).astype(int)

        return labels.dropna()

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
