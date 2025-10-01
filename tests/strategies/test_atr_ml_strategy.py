"""
ATRMLStrategy のユニットテスト

richmanbtcチュートリアルの2層統合システムの動作検証テスト。

テスト要件:
- 5.1: FreqtradeストラテジーとのMLOps統合
- 5.2: フォールバック機能と運用安全性
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from freqtrade.strategy.interface import IStrategy
from user_data.strategies.atr_ml_strategy import ATRMLStrategy


class TestATRMLStrategy:
    """ATRMLStrategy のテストクラス"""

    @pytest.fixture
    def config(self):
        """テスト用設定"""
        return {
            "freqai": {
                "enabled": True,
                "identifier": "test_atr_ml",
                "feature_parameters": {
                    "include_timeframes": ["5m", "15m", "1h"],
                    "include_corr_pairlist": ["BTC/USDT", "ETH/USDT"],
                },
                "data_split_parameters": {
                    "test_size": 0.25,
                    "shuffle": False,
                },
                "model_training_parameters": {
                    "n_estimators": 100,
                    "learning_rate": 0.1,
                    "max_depth": 7,
                },
            },
            "two_tier_strategy": {
                "preset": "price_only",
                "primary_model": {
                    "type": "atr",
                    "params": {"period": 14, "multiplier": 0.5},
                },
                "secondary_model": {
                    "enabled": False,
                    "confidence_threshold": 0.6,
                },
            },
        }

    @pytest.fixture
    def sample_dataframe(self):
        """サンプルOHLCデータ"""
        dates = pd.date_range("2023-01-01", periods=100, freq="5min")
        return pd.DataFrame(
            {
                "date": dates,
                "open": np.random.uniform(100, 110, 100),
                "high": np.random.uniform(110, 120, 100),
                "low": np.random.uniform(90, 100, 100),
                "close": np.random.uniform(95, 115, 100),
                "volume": np.random.uniform(1000, 5000, 100),
            }
        )

    @pytest.fixture
    def strategy(self, config):
        """ATRMLStrategyインスタンス"""
        strategy = ATRMLStrategy(config)
        # テスト用のモック属性を追加
        strategy.freqai = None
        strategy.dp = Mock()
        strategy.dp.get_pair_dataframe.return_value = pd.DataFrame()
        return strategy

    def test_strategy_inheritance(self, strategy):
        """IStrategyからの継承確認 - 要件 5.1"""
        assert isinstance(strategy, IStrategy)
        assert isinstance(strategy, ATRMLStrategy)

    def test_strategy_initialization(self, strategy):
        """ストラテジー初期化確認 - 要件 5.1"""
        # 基本属性の確認
        assert hasattr(strategy, "minimal_roi")
        assert hasattr(strategy, "stoploss")
        assert hasattr(strategy, "timeframe")
        assert hasattr(strategy, "process_only_new_candles")

        # ATR設定の確認
        assert hasattr(strategy, "entry_length")
        assert hasattr(strategy, "entry_point")
        assert strategy.entry_length == 14
        assert strategy.entry_point == 0.5

    def test_populate_indicators_basic(self, strategy, sample_dataframe):
        """基本指標生成テスト - 要件 5.1"""
        metadata = {"pair": "BTC/USDT"}
        result = strategy.populate_indicators(sample_dataframe, metadata)

        # ATR関連カラムの確認
        expected_columns = ["atr", "atr_buy_price", "atr_sell_price"]
        for col in expected_columns:
            assert col in result.columns

        # FreqAI統合の確認は実際の統合時に確認（テスト環境ではスキップ）
        assert len(result) == len(sample_dataframe)

    @patch("user_data.strategies.utils.atr_calculator.ATRCalculator")
    def test_populate_indicators_with_atr(self, mock_atr_calc, strategy, sample_dataframe):
        """ATR計算統合テスト - 要件 5.1"""
        # ATR計算器のモック設定
        mock_calculator = Mock()
        mock_calculator.calculate_atr_prices.return_value = sample_dataframe.copy()
        mock_atr_calc.return_value = mock_calculator

        metadata = {"pair": "BTC/USDT"}
        result = strategy.populate_indicators(sample_dataframe, metadata)

        # ATR計算器が呼び出されたことを確認
        mock_atr_calc.assert_called_once()
        mock_calculator.calculate_atr_prices.assert_called_once()

    def test_populate_entry_trend_with_ml_prediction(self, strategy, sample_dataframe):
        """2層判定ロジックテスト（ML予測あり） - 要件 5.1"""
        # MLからの予測結果をシミュレート
        dataframe = sample_dataframe.copy()
        dataframe["&-prediction"] = [1, 0, 1, 0, 1] * 20  # 正負交互
        dataframe["atr"] = np.random.uniform(0.5, 2.0, 100)
        dataframe["atr_buy_price"] = dataframe["close"] * 0.99
        dataframe["atr_sell_price"] = dataframe["close"] * 1.01

        metadata = {"pair": "BTC/USDT"}
        result = strategy.populate_entry_trend(dataframe, metadata)

        # エントリー条件の確認
        assert "enter_long" in result.columns
        assert "enter_short" in result.columns

        # 2層判定ロジックの確認（ML予測=1の場合のみエントリー）
        enter_long_indices = result[result["enter_long"] == 1].index
        for idx in enter_long_indices:
            assert result.loc[idx, "&-prediction"] == 1

    def test_populate_entry_trend_without_ml_prediction(self, strategy, sample_dataframe):
        """フォールバック動作テスト（ML予測なし） - 要件 5.2"""
        # ML予測なしのデータフレーム
        dataframe = sample_dataframe.copy()
        dataframe["atr"] = np.random.uniform(0.5, 2.0, 100)
        dataframe["atr_buy_price"] = dataframe["close"] * 0.99
        dataframe["atr_sell_price"] = dataframe["close"] * 1.01

        metadata = {"pair": "BTC/USDT"}
        result = strategy.populate_entry_trend(dataframe, metadata)

        # フォールバック設定に応じた動作確認
        if strategy.fallback_mode == "skip_orders":
            # 注文スキップモード：エントリー信号なし
            assert result["enter_long"].sum() == 0
            assert result["enter_short"].sum() == 0
        elif strategy.fallback_mode == "far_orders":
            # 遠距離注文モード：ATRのみでエントリー
            assert result["enter_long"].sum() > 0

    def test_custom_entry_price_with_atr(self, strategy, sample_dataframe):
        """ATR指値価格計算テスト - 要件 5.1"""
        # テストデータの準備
        dataframe = sample_dataframe.copy()
        dataframe["atr_buy_price"] = [100.5, 101.0, 99.8, 102.5, 98.9]

        # dp.get_pair_dataframeのモック設定
        strategy.dp.get_pair_dataframe.return_value = dataframe

        # ロング・ショートエントリー価格の確認
        for i in range(min(5, len(dataframe))):
            proposed_rate = dataframe.loc[i, "close"]
            buy_price = strategy.custom_entry_price(
                pair="BTC/USDT",
                trade=None,
                current_time=None,
                proposed_rate=proposed_rate,
                entry_tag="long",
                side="long",
                current_rate=proposed_rate,
            )

            expected_price = dataframe.loc[dataframe.index[-1], "atr_buy_price"]  # 最新レコード
            assert buy_price == expected_price

    def test_custom_entry_price_fallback(self, strategy):
        """ATR価格データなし時のフォールバック - 要件 5.2"""
        # ATR価格データが存在しない場合
        result = strategy.custom_entry_price(
            pair="BTC/USDT",
            trade=None,
            current_time=None,
            proposed_rate=100.0,
            entry_tag="long",
            side="long",
            current_rate=100.0,
        )

        # 提案価格をそのまま返すことを確認
        assert result == 100.0

    @patch("user_data.strategies.atr_ml_strategy.logger")
    def test_ml_prediction_logging(self, mock_logger, strategy, sample_dataframe):
        """ML予測詳細ログ機能テスト - 要件 5.2"""
        # ログ機能が有効な設定
        strategy.log_predictions = True

        dataframe = sample_dataframe.copy()
        dataframe["&-prediction"] = [1, 0, 1, 0, 1] * 20
        dataframe["atr"] = np.random.uniform(0.5, 2.0, 100)
        dataframe["atr_buy_price"] = dataframe["close"] * 0.99
        dataframe["atr_sell_price"] = dataframe["close"] * 1.01

        metadata = {"pair": "BTC/USDT"}
        strategy.populate_entry_trend(dataframe, metadata)

        # 予測詳細がログに記録されたことを確認
        mock_logger.info.assert_called()

    def test_error_handling_in_entry_trend(self, strategy, sample_dataframe):
        """エントリートレンド処理のエラーハンドリング - 要件 5.2"""
        # 不正なデータでエラーが発生する状況をシミュレート
        dataframe = sample_dataframe.copy()
        # ATRカラムを削除してエラーを誘発
        if "atr" in dataframe.columns:
            del dataframe["atr"]

        # エラーが発生してもクラッシュしないことを確認
        metadata = {"pair": "BTC/USDT"}
        result = strategy.populate_entry_trend(dataframe, metadata)

        # 安全なフォールバック（エントリー信号なし）
        assert "enter_long" in result.columns
        assert "enter_short" in result.columns
        assert result["enter_long"].sum() == 0
        assert result["enter_short"].sum() == 0

    def test_freqai_integration_config(self, strategy):
        """FreqAI統合設定確認 - 要件 5.1"""
        # FreqAI統合属性が存在することを確認（テスト環境ではNone）
        assert hasattr(strategy, "freqai")

        # プロセス専用新規キャンドル設定
        assert strategy.process_only_new_candles is True

    def test_confidence_threshold_filtering(self, strategy, sample_dataframe):
        """信頼度閾値フィルタリング機能テスト - 要件 5.1"""
        # 信頼度付き予測データの準備
        dataframe = sample_dataframe.copy()
        dataframe["&-prediction"] = [1, 0, 1, 0, 1] * 20
        dataframe["&-probability"] = [0.8, 0.3, 0.9, 0.2, 0.5] * 20  # 信頼度
        dataframe["atr"] = np.random.uniform(0.5, 2.0, 100)
        dataframe["atr_buy_price"] = dataframe["close"] * 0.99

        # 信頼度閾値を設定
        strategy.confidence_threshold = 0.6

        metadata = {"pair": "BTC/USDT"}
        result = strategy.populate_entry_trend(dataframe, metadata)

        # 閾値以上の信頼度を持つ予測のみがエントリー信号になることを確認
        enter_long_indices = result[result["enter_long"] == 1].index
        for idx in enter_long_indices:
            if "&-probability" in result.columns:
                assert result.loc[idx, "&-probability"] >= strategy.confidence_threshold
