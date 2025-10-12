"""戦略ファクトリー統合モジュール

1次モデル、2次モデル、エントリー戦略の選択機能を統合
"""

import importlib
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union

import pandas as pd

from .freqai_model_factory import FreqAIModelFactory
from .price_calculator import PriceCalculatorBase, PriceCalculatorFactory


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
        """エントリー信号生成

        1次モデルで価格計算を行い、2次モデルの有効状態に応じて
        適切な戦略で信号を生成

        Raises:
            ValueError: カラム不足など、呼び出し元で対処すべきエラー
            Exception: その他の予期しないエラー
        """
        # 1次モデル: 価格計算（エラーは呼び出し元に伝播）
        price_data = self.primary_model.calculate_entry_prices(dataframe)

        if self.is_ml_enabled:
            # 2次モデル有効時: ML予測を使用した信号生成
            return self._generate_ml_integrated_signals(price_data, metadata)
        else:
            # 2次モデル無効時: 価格のみの信号生成
            return self._generate_basic_price_signals(price_data, metadata)

    def _initialize_signal_dataframe(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """信号DataFrameの初期化

        enter_long, enter_shortカラムを追加（既存の場合は0にリセット）

        Args:
            dataframe: 初期化対象のDataFrame

        Returns:
            初期化されたDataFrame（同じオブジェクト）
        """
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        return dataframe

    def _generate_ml_integrated_signals(self, dataframe, metadata: Dict[str, Any]):
        """ML予測を使用した信号生成

        ML予測データを使用してエントリー信号を生成
        NaN検証とカラム存在確認を実施

        Raises:
            ValueError: 必要な価格カラムが存在しない場合
        """
        pair = metadata.get("pair", "unknown")
        result = self._initialize_signal_dataframe(dataframe)

        # ML予測データの取得と検証
        ml_data = self._extract_ml_prediction_data(dataframe, pair)
        if ml_data is None:
            logger.info(f"ML prediction unavailable for {pair}, returning empty signals")
            return result

        # エントリー条件の計算（ValueErrorは呼び出し元に伝播）
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
    ) -> Optional[Dict[str, Union[pd.Series, None]]]:
        """ML予測データの抽出とNaN検証

        Returns:
            ML予測データ辞書（prediction, probability, valid_mask）、
            または予測データが不足/全てNaNの場合はNone
        """
        has_prediction = "&-prediction" in dataframe.columns
        has_probability = "&-probability" in dataframe.columns

        if not has_prediction:
            logger.warning(f"ML prediction data missing for {pair}")
            return None

        # NaN検証
        prediction_col = dataframe["&-prediction"]
        prediction_nan_mask = prediction_col.isna()
        total_rows = len(dataframe)
        nan_count = prediction_nan_mask.sum()
        valid_count = total_rows - nan_count

        # 全てNaNの場合は処理不可
        if nan_count == total_rows:
            logger.error(
                f"ML prediction data all NaN for {pair}: "
                f"total={total_rows}, nan={nan_count}, valid={valid_count}. "
                f"Cannot proceed with ML-based trading."
            )
            return None

        # 部分的にNaNの場合は警告を出しつつ処理継続
        # リアルタイム取引では一時的なデータ欠損が発生することがあるため、
        # 有効なデータだけで取引を継続する
        if nan_count > 0:
            nan_ratio = (nan_count / total_rows) * 100
            logger.warning(
                f"ML prediction contains NaN values for {pair}: "
                f"total={total_rows}, nan={nan_count} ({nan_ratio:.2f}%), valid={valid_count}. "
                f"NaN rows will be excluded from signal generation."
            )

        # valid_maskを生成（NaNでない行を示すboolean mask）
        valid_mask = ~prediction_nan_mask

        # probabilityもNaNチェック（存在する場合）
        probability_series = None
        if has_probability:
            probability_col = dataframe["&-probability"]
            prob_nan_mask = probability_col.isna()
            prob_nan_count = prob_nan_mask.sum()

            if prob_nan_count > 0:
                logger.warning(
                    f"ML probability contains {prob_nan_count} NaN values for {pair}. "
                    f"These will be handled by confidence filtering."
                )

            probability_series = probability_col

        return {
            "prediction": prediction_col == 1,
            "probability": probability_series,
            "valid_mask": valid_mask,
        }

    def _calculate_ml_entry_conditions(
        self, dataframe: pd.DataFrame, ml_data: Dict[str, pd.Series], config: Dict[str, Any]
    ) -> Tuple[pd.Series, pd.Series]:
        """ML予測とATR価格を組み合わせたエントリー条件の計算

        Returns:
            (long_condition, short_condition)のタプル

        Raises:
            ValueError: 必要な価格カラムが存在しない場合
        """
        # 価格カラム存在確認
        required_price_columns = {"buy_price", "sell_price"}
        missing_price_columns = required_price_columns - set(dataframe.columns)

        if missing_price_columns:
            available_columns = sorted(dataframe.columns.tolist())
            raise ValueError(
                f"Required price columns missing: {sorted(missing_price_columns)}. "
                f"Available columns: {available_columns}. "
                f"Please ensure primary model calculated these columns in populate_indicators."
            )

        ml_prediction = ml_data["prediction"]
        ml_probability = ml_data["probability"]
        valid_mask = ml_data["valid_mask"]

        # 信頼度フィルタリング
        confidence_threshold = config.get("entry", {}).get("confidence_threshold", 0.6)
        confidence_filter = self._apply_confidence_threshold(
            ml_probability, confidence_threshold, dataframe.index
        )

        # 価格データの有効性チェック
        price_valid = (dataframe["buy_price"] > 0) & (dataframe["sell_price"] > 0)

        # エントリー条件（valid_maskでNaN行を除外）
        long_condition = ml_prediction & confidence_filter & price_valid & valid_mask
        short_condition = ~ml_prediction & confidence_filter & price_valid & valid_mask

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
        """価格ブレイクアウト信号生成

        ML予測なしの価格ブレイクアウト戦略
        前期間の指値価格に現在価格が到達した場合に信号を生成
        """
        pair = metadata.get("pair", "unknown")
        result = self._initialize_signal_dataframe(dataframe)

        if len(dataframe) <= 1:
            logger.debug(f"Insufficient data for {pair} (rows={len(dataframe)})")
            return result

        # 前期間の価格と現在価格の比較
        prev_buy_price = dataframe["buy_price"].shift(1)
        prev_sell_price = dataframe["sell_price"].shift(1)
        current_close = dataframe["close"]

        # 価格が指値レベルに到達した時の取引判定
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


class PrimaryStrategyFactory:
    """1次戦略（Primary Strategy）のファクトリークラス

    config.jsonの戦略名から具体的なPrimaryStrategyBaseクラスを動的にロードする。
    Phase 2の実装として、新しい設計ドキュメントに基づいて実装。
    """

    # 1次戦略の登録辞書
    # キー: config内で使用する戦略名
    # 値: モジュールパス.クラス名の文字列
    _primary_strategies = {
        "atr_breakout": "user_data.strategies.primary.atr_breakout.ATRBreakoutStrategy",
    }

    @classmethod
    def load_primary(cls, config: Dict[str, Any]) -> "PrimaryStrategyBase":
        """1次戦略をロードして返す

        Args:
            config: two_tier_strategy設定辞書
                - primary (str): 戦略名（必須）
                - primary_params (dict): 戦略パラメータ（オプション）

        Returns:
            PrimaryStrategyBaseインスタンス

        Raises:
            ValueError: primary名が指定されていない、または存在しない戦略名の場合

        Example:
            config = {
                "primary": "atr_breakout",
                "primary_params": {
                    "period": 14,
                    "multiplier": 0.5,
                    "execution_mode": "one_candle"
                }
            }
            strategy = PrimaryStrategyFactory.load_primary(config)
        """
        primary_name = config.get("primary")
        if not primary_name:
            raise ValueError(
                "Primary strategy name is required. "
                "Please specify 'primary' in config['two_tier_strategy']"
            )

        if primary_name not in cls._primary_strategies:
            available = ", ".join(cls._primary_strategies.keys())
            raise ValueError(
                f"Unknown primary strategy: '{primary_name}'. Available strategies: {available}"
            )

        # 戦略クラスをロード
        strategy_class = cls._load_class(cls._primary_strategies[primary_name])

        # パラメータを取得してインスタンス化
        primary_params = config.get("primary_params", {})
        strategy_instance = strategy_class(primary_params)

        logger.info(
            f"Loaded primary strategy: {primary_name} "
            f"({type(strategy_instance).__name__}) with params: {primary_params}"
        )

        return strategy_instance

    @classmethod
    def _load_class(cls, class_path: str):
        """モジュールパスからクラスを動的にロード

        Args:
            class_path: "module.path.ClassName"形式のクラスパス

        Returns:
            ロードされたクラスオブジェクト

        Raises:
            ImportError: モジュールのインポートに失敗した場合
            AttributeError: クラスが見つからない場合
        """
        try:
            # クラスパスを分割
            module_path, class_name = class_path.rsplit(".", 1)

            # モジュールをインポート
            module = importlib.import_module(module_path)

            # クラスを取得
            strategy_class = getattr(module, class_name)

            return strategy_class

        except (ImportError, AttributeError) as e:
            logger.error(f"Failed to load class from path '{class_path}': {e}")
            raise

    @classmethod
    def register_strategy(cls, name: str, class_path: str) -> None:
        """新しい1次戦略を登録する

        Phase 2以降で新しい戦略を追加する際に使用。

        Args:
            name: 戦略名（config内で使用する識別子）
            class_path: "module.path.ClassName"形式のクラスパス

        Example:
            PrimaryStrategyFactory.register_strategy(
                "bollinger_breakout",
                "user_data.strategies.primary.bollinger_breakout.BollingerBreakoutStrategy"
            )
        """
        if name in cls._primary_strategies:
            logger.warning(f"Overwriting existing strategy registration: {name}")

        cls._primary_strategies[name] = class_path
        logger.info(f"Registered primary strategy: {name} -> {class_path}")

    @classmethod
    def list_available_strategies(cls) -> list:
        """利用可能な1次戦略のリストを取得

        Returns:
            登録済み戦略名のリスト
        """
        return list(cls._primary_strategies.keys())
