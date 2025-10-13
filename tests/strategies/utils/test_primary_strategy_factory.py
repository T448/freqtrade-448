"""Tests for PrimaryStrategyFactory (Phase 2 implementation)."""

import pytest

from user_data.strategies.primary.atr_breakout import ATRBreakoutStrategy
from user_data.strategies.primary.base import PrimaryStrategyBase
from user_data.strategies.utils.strategy_factory import PrimaryStrategyFactory


class TestPrimaryStrategyFactory:
    """PrimaryStrategyFactoryのテストケース"""

    def test_load_primary_atr_breakout_success(self):
        """ATR戦略の正常ロードテスト"""
        # config設定
        config = {"primary": "atr_breakout", "primary_params": {"period": 14, "multiplier": 0.5}}

        # 戦略ロード
        strategy = PrimaryStrategyFactory.load_primary(config)

        # 検証
        assert isinstance(strategy, PrimaryStrategyBase)
        assert isinstance(strategy, ATRBreakoutStrategy)
        assert strategy.period == 14
        assert strategy.multiplier == 0.5

    def test_load_primary_with_custom_params(self):
        """カスタムパラメータでの戦略ロードテスト"""
        config = {
            "primary": "atr_breakout",
            "primary_params": {
                "period": 20,
                "multiplier": 1.0,
                "execution_mode": "chase",
                "fee": 0.0005,
                "exit_periods": 48,
            },
        }

        strategy = PrimaryStrategyFactory.load_primary(config)

        assert strategy.period == 20
        assert strategy.multiplier == 1.0
        assert strategy.execution_mode == "chase"
        assert strategy.fee == 0.0005
        assert strategy.exit_periods == 48

    def test_load_primary_with_default_params(self):
        """デフォルトパラメータでの戦略ロードテスト"""
        config = {"primary": "atr_breakout"}

        strategy = PrimaryStrategyFactory.load_primary(config)

        # デフォルト値確認
        assert strategy.period == 14
        assert strategy.multiplier == 0.5
        assert strategy.execution_mode == "one_candle"
        assert strategy.fee == 0.00025
        assert strategy.exit_periods == 24

    def test_load_primary_missing_name(self):
        """primary名が指定されていない場合のエラー"""
        config = {"primary_params": {"period": 14}}

        with pytest.raises(ValueError, match="Primary strategy name is required"):
            PrimaryStrategyFactory.load_primary(config)

    def test_load_primary_unknown_strategy(self):
        """存在しない戦略名のエラー"""
        config = {"primary": "unknown_strategy"}

        with pytest.raises(ValueError, match="Unknown primary strategy.*unknown_strategy"):
            PrimaryStrategyFactory.load_primary(config)

        # エラーメッセージに利用可能な戦略が含まれる
        with pytest.raises(ValueError, match="Available strategies"):
            PrimaryStrategyFactory.load_primary(config)

    def test_list_available_strategies(self):
        """利用可能な戦略リスト取得テスト"""
        strategies = PrimaryStrategyFactory.list_available_strategies()

        assert isinstance(strategies, list)
        assert "atr_breakout" in strategies
        assert len(strategies) >= 1

    def test_register_new_strategy(self):
        """新規戦略の登録テスト"""
        # 登録前の戦略数
        initial_count = len(PrimaryStrategyFactory.list_available_strategies())

        # 新規戦略を登録
        test_strategy_path = "user_data.strategies.primary.atr_breakout.ATRBreakoutStrategy"
        PrimaryStrategyFactory.register_strategy("test_strategy", test_strategy_path)

        # 登録確認
        strategies = PrimaryStrategyFactory.list_available_strategies()
        assert "test_strategy" in strategies
        assert len(strategies) == initial_count + 1

        # 登録した戦略をロード
        config = {"primary": "test_strategy", "primary_params": {"period": 10}}
        strategy = PrimaryStrategyFactory.load_primary(config)
        assert isinstance(strategy, ATRBreakoutStrategy)
        assert strategy.period == 10

        # クリーンアップ（他のテストへの影響を避けるため）
        del PrimaryStrategyFactory._primary_strategies["test_strategy"]

    def test_register_strategy_overwrite_warning(self, caplog):
        """既存戦略の上書き登録時の警告テスト"""
        test_path = "user_data.strategies.primary.atr_breakout.ATRBreakoutStrategy"

        # 既存の戦略名で登録（上書き）
        PrimaryStrategyFactory.register_strategy("atr_breakout", test_path)

        # 警告が出力されることを確認
        assert "Overwriting existing strategy registration" in caplog.text


class TestPrimaryStrategyFactoryIntegration:
    """PrimaryStrategyFactoryの統合テスト"""

    def test_load_and_execute_strategy(self):
        """戦略ロード後の実行テスト"""
        import pandas as pd

        config = {"primary": "atr_breakout", "primary_params": {"period": 14, "multiplier": 0.5}}

        strategy = PrimaryStrategyFactory.load_primary(config)

        # テストデータ作成（ATR計算に十分な期間を含む）
        dataframe = pd.DataFrame(
            {
                "high": [100.0, 101.0, 102.0, 103.0, 104.0] * 20,
                "low": [99.0, 100.0, 101.0, 102.0, 103.0] * 20,
                "close": [99.5, 100.5, 101.5, 102.5, 103.5] * 20,
            }
        )

        # 価格計算実行
        result = strategy.calculate_prices(dataframe)

        # 結果確認
        assert "buy_price" in result.columns
        assert "sell_price" in result.columns
        assert len(result) == len(dataframe)

        # NaNを除いて検証（ATRの計算期間前はNaN）
        valid_data = result.dropna(subset=["buy_price", "sell_price"])
        assert len(valid_data) > 0
        assert (valid_data["buy_price"] < valid_data["close"]).all()
        assert (valid_data["sell_price"] > valid_data["close"]).all()
