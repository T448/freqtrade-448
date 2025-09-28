"""価格計算統合モジュール

1次モデル（ATR、ボリンジャーバンド等）の価格計算を統合管理
"""

import logging
import hashlib
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import pandas as pd
import talib as ta

logger = logging.getLogger(__name__)


class PriceCalculatorBase(ABC):
    """価格計算の抽象基底クラス"""

    def __init__(self, params: Dict[str, Any]):
        self.params = params
        self.validate_params()

    @abstractmethod
    def validate_params(self) -> None:
        """パラメータ検証"""
        pass

    @abstractmethod
    def calculate_entry_prices(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """エントリー価格計算

        Returns:
            buy_price, sell_price列を追加したDataFrame
        """
        pass

    @abstractmethod
    def calculate_limit_price(self, close_price: float, side: str) -> float:
        """個別指値価格計算"""
        pass

    @abstractmethod
    def get_calculation_columns(self) -> list:
        """計算に必要なカラム名リスト"""
        pass


class ATRPriceCalculator(PriceCalculatorBase):
    """ATR価格計算実装（既存ATRCalculatorEngineの抽象化）"""

    # キャッシュ機能
    _cache: Dict[str, pd.DataFrame] = {}
    _cache_size_limit = 50

    def validate_params(self) -> None:
        required = ["period", "multiplier"]
        for param in required:
            if param not in self.params:
                raise ValueError(f"Missing required parameter: {param}")

        if self.params["period"] <= 0:
            raise ValueError(f"ATR period must be positive: {self.params['period']}")
        if self.params["multiplier"] < 0:
            raise ValueError(f"ATR multiplier must be non-negative: {self.params['multiplier']}")

    def _generate_cache_key(self, dataframe: pd.DataFrame) -> str:
        """キャッシュキーを生成"""
        # データフレームのハッシュ（最新数行のみ使用してパフォーマンス向上）
        sample_data = dataframe.tail(20) if len(dataframe) > 20 else dataframe
        data_hash = hashlib.md5(
            pd.util.hash_pandas_object(sample_data[["high", "low", "close"]], index=True).values
        ).hexdigest()[:16]

        period = self.params["period"]
        multiplier = self.params["multiplier"]
        return f"atr_{data_hash}_{period}_{multiplier}_{len(dataframe)}"

    def _manage_cache_size(self):
        """キャッシュサイズ管理"""
        if len(self._cache) > self._cache_size_limit:
            # 古いエントリを削除（最初の半分を削除）
            items_to_remove = list(self._cache.keys())[: self._cache_size_limit // 2]
            for key in items_to_remove:
                del self._cache[key]
            logger.debug(f"ATR price cache cleaned: removed {len(items_to_remove)} entries")

    def calculate_entry_prices(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """ATRベースエントリー価格計算"""
        if dataframe.empty:
            raise ValueError("Input dataframe is empty")

        required_columns = ["high", "low", "close"]
        missing_columns = [col for col in required_columns if col not in dataframe.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        period = self.params["period"]
        multiplier = self.params["multiplier"]

        if len(dataframe) < period:
            error_msg = f"Insufficient data: need {period}, got {len(dataframe)}"
            logger.warning(error_msg)
            raise ValueError(error_msg)

        # キャッシュチェック
        cache_key = self._generate_cache_key(dataframe)
        if cache_key in self._cache:
            logger.debug("ATR price cache hit")
            return self._cache[cache_key]

        result = dataframe.copy()

        # ATR計算
        atr_values = ta.ATR(
            dataframe["high"].astype(float).values,
            dataframe["low"].astype(float).values,
            dataframe["close"].astype(float).values,
            timeperiod=period,
        )

        result["atr"] = pd.Series(atr_values, index=dataframe.index, name="atr")

        # エントリー価格計算
        result["buy_price"] = result["close"] - (result["atr"] * multiplier)
        result["sell_price"] = result["close"] + (result["atr"] * multiplier)

        # キャッシュに保存
        self._cache[cache_key] = result
        self._manage_cache_size()

        # ログ出力（最適化済み）
        if len(result) >= 5:
            recent_atr = result["atr"].iloc[-1]
            valid_atr_count = result["atr"].notna().sum()
            logger.debug(
                f"ATR price calculation completed: period={period}, multiplier={multiplier}, "
                f"latest_atr={recent_atr:.8f}, valid_records={valid_atr_count}/{len(result)}"
            )

        return result

    def calculate_limit_price(self, close_price: float, side: str) -> float:
        """ATR指値価格計算"""
        if pd.isna(close_price):
            raise ValueError("Close price is NaN")

        # この実装では単体でATR値を計算できないため、
        # 実際の使用時は呼び出し元でATR値を事前計算する必要がある
        # ここではプレースホルダー実装を提供
        multiplier = self.params["multiplier"]

        # 簡易ATR概算（実際の実装では事前計算されたATR値を使用）
        atr_estimate = close_price * 0.02  # 2%の簡易ATR概算

        side_lower = side.lower()
        if side_lower in ["buy", "long"]:
            return close_price - (atr_estimate * multiplier)
        elif side_lower in ["sell", "short"]:
            return close_price + (atr_estimate * multiplier)
        else:
            raise ValueError(f"Invalid side value: {side}. Use 'buy'/'long' or 'sell'/'short'")

    def get_calculation_columns(self) -> list:
        return ["high", "low", "close"]

    @classmethod
    def clear_cache(cls):
        """キャッシュクリア"""
        cls._cache.clear()
        logger.debug("ATR price cache cleared")


class BollingerPriceCalculator(PriceCalculatorBase):
    """ボリンジャーバンド価格計算実装"""

    def validate_params(self) -> None:
        required = ["period", "std_dev"]
        for param in required:
            if param not in self.params:
                raise ValueError(f"Missing required parameter: {param}")

        if self.params["period"] <= 0:
            raise ValueError(f"Bollinger period must be positive: {self.params['period']}")
        if self.params["std_dev"] <= 0:
            raise ValueError(f"Standard deviation must be positive: {self.params['std_dev']}")

    def calculate_entry_prices(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """ボリンジャーバンドベースエントリー価格計算"""
        if dataframe.empty:
            raise ValueError("Input dataframe is empty")

        if "close" not in dataframe.columns:
            raise ValueError("Missing required column: close")

        period = self.params["period"]
        std_dev = self.params["std_dev"]

        if len(dataframe) < period:
            error_msg = f"Insufficient data: need {period}, got {len(dataframe)}"
            logger.warning(error_msg)
            raise ValueError(error_msg)

        result = dataframe.copy()

        # ボリンジャーバンド計算
        result["bb_upper"], result["bb_middle"], result["bb_lower"] = ta.BBANDS(
            dataframe["close"], timeperiod=period, nbdevup=std_dev, nbdevdn=std_dev
        )

        # エントリー価格設定（下限で買い、上限で売り）
        result["buy_price"] = result["bb_lower"]
        result["sell_price"] = result["bb_upper"]

        logger.debug(f"Bollinger band calculation completed: period={period}, std_dev={std_dev}")
        return result

    def calculate_limit_price(self, close_price: float, side: str) -> float:
        """ボリンジャーバンド指値価格計算"""
        if pd.isna(close_price):
            raise ValueError("Close price is NaN")

        # 簡易ボリンジャーバンド概算
        period = self.params["period"]
        std_dev = self.params["std_dev"]

        # 簡易標準偏差概算（実際の実装では事前計算された値を使用）
        volatility_estimate = close_price * 0.03  # 3%のボラティリティ概算

        side_lower = side.lower()
        if side_lower in ["buy", "long"]:
            return close_price - (volatility_estimate * std_dev)
        elif side_lower in ["sell", "short"]:
            return close_price + (volatility_estimate * std_dev)
        else:
            raise ValueError(f"Invalid side value: {side}")

    def get_calculation_columns(self) -> list:
        return ["close"]


class MovingAveragePriceCalculator(PriceCalculatorBase):
    """移動平均価格計算実装"""

    def validate_params(self) -> None:
        required = ["fast_period", "slow_period", "offset_ratio"]
        for param in required:
            if param not in self.params:
                raise ValueError(f"Missing required parameter: {param}")

        if self.params["fast_period"] <= 0:
            raise ValueError(f"Fast period must be positive: {self.params['fast_period']}")
        if self.params["slow_period"] <= 0:
            raise ValueError(f"Slow period must be positive: {self.params['slow_period']}")
        if self.params["fast_period"] >= self.params["slow_period"]:
            raise ValueError("Fast period must be less than slow period")
        if self.params["offset_ratio"] < 0:
            raise ValueError(f"Offset ratio must be non-negative: {self.params['offset_ratio']}")

    def calculate_entry_prices(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """移動平均ベースエントリー価格計算"""
        if dataframe.empty:
            raise ValueError("Input dataframe is empty")

        if "close" not in dataframe.columns:
            raise ValueError("Missing required column: close")

        fast_period = self.params["fast_period"]
        slow_period = self.params["slow_period"]
        offset_ratio = self.params["offset_ratio"]

        if len(dataframe) < slow_period:
            error_msg = f"Insufficient data: need {slow_period}, got {len(dataframe)}"
            logger.warning(error_msg)
            raise ValueError(error_msg)

        result = dataframe.copy()

        # 移動平均計算
        result["ma_fast"] = ta.SMA(dataframe["close"], timeperiod=fast_period)
        result["ma_slow"] = ta.SMA(dataframe["close"], timeperiod=slow_period)

        # 移動平均の中間値をベースに指値価格計算
        ma_mid = (result["ma_fast"] + result["ma_slow"]) / 2
        offset = result["close"] * offset_ratio

        result["buy_price"] = ma_mid - offset
        result["sell_price"] = ma_mid + offset

        logger.debug(
            f"Moving average calculation completed: fast={fast_period}, slow={slow_period}, offset={offset_ratio}"
        )
        return result

    def calculate_limit_price(self, close_price: float, side: str) -> float:
        """移動平均指値価格計算"""
        if pd.isna(close_price):
            raise ValueError("Close price is NaN")

        offset_ratio = self.params["offset_ratio"]
        offset = close_price * offset_ratio

        # 移動平均の概算（実際の実装では事前計算された値を使用）
        ma_estimate = close_price  # 簡易概算として現在価格を使用

        side_lower = side.lower()
        if side_lower in ["buy", "long"]:
            return ma_estimate - offset
        elif side_lower in ["sell", "short"]:
            return ma_estimate + offset
        else:
            raise ValueError(f"Invalid side value: {side}")

    def get_calculation_columns(self) -> list:
        return ["close"]


class PriceCalculatorFactory:
    """価格計算器ファクトリー"""

    _calculators = {
        "atr": ATRPriceCalculator,
        "bollinger": BollingerPriceCalculator,
        "ma": MovingAveragePriceCalculator,
    }

    @classmethod
    def create_calculator(cls, calculator_type: str, params: Dict[str, Any]) -> PriceCalculatorBase:
        """価格計算器を作成

        Args:
            calculator_type: 計算器タイプ
            params: パラメータ辞書

        Returns:
            価格計算器インスタンス

        Raises:
            ValueError: 未知の計算器タイプの場合
        """
        if calculator_type not in cls._calculators:
            available = list(cls._calculators.keys())
            raise ValueError(f"Unknown calculator type: {calculator_type}. Available: {available}")

        calculator_class = cls._calculators[calculator_type]
        return calculator_class(params)

    @classmethod
    def register_calculator(cls, name: str, calculator_class: type):
        """新しい価格計算器を登録

        Args:
            name: 計算器名
            calculator_class: 計算器クラス

        Raises:
            ValueError: 無効な計算器クラスの場合
        """
        if not issubclass(calculator_class, PriceCalculatorBase):
            raise ValueError("Calculator class must inherit from PriceCalculatorBase")

        cls._calculators[name] = calculator_class
        logger.info(f"Registered price calculator: {name}")

    @classmethod
    def list_available_calculators(cls) -> list:
        """利用可能な計算器一覧

        Returns:
            計算器名のリスト
        """
        return list(cls._calculators.keys())

    @classmethod
    def get_calculator_info(cls, calculator_type: str) -> Dict[str, Any]:
        """計算器情報取得

        Args:
            calculator_type: 計算器タイプ

        Returns:
            計算器情報辞書
        """
        if calculator_type not in cls._calculators:
            raise ValueError(f"Unknown calculator type: {calculator_type}")

        calculator_class = cls._calculators[calculator_type]

        return {
            "class_name": calculator_class.__name__,
            "description": calculator_class.__doc__,
            "type": calculator_type,
        }
