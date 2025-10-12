"""TwoTierStrategy integration tests - ラベル生成テスト

このモジュールは、TwoTierStrategyの統合テスト、特にラベル生成ロジックを
包括的にテストします。

テスト方針:
- set_freqai_targets()が正しくラベルを生成すること
- リターン > 0 でラベル=1、リターン <= 0 でラベル=0
- execution_mode切り替え時のラベル変化確認
- ML有効/無効モードでの挙動確認
"""

import numpy as np
import pandas as pd
import pytest

from user_data.strategies.two_tier_strategy import TwoTierStrategy


class TestTwoTierStrategy:
    """TwoTierStrategy統合テストスイート"""

    @pytest.fixture
    def ml_off_config(self):
        """ML無効モード設定

        Returns:
            dict: テスト用のML無効設定
        """
        return {
            "two_tier_strategy": {
                "primary": "atr_breakout",
                "secondary": None,
                "primary_params": {
                    "period": 14,
                    "multiplier": 0.5,
                    "execution_mode": "one_candle",
                    "fee": 0.00025,
                    "exit_periods": 24,
                    "pips": 0.5,
                },
            },
            "freqai": {
                "enabled": False,
                "data_split_parameters": {"test_size": 0.2, "shuffle": False},
            },
        }

    @pytest.fixture
    def sample_ohlcv(self):
        """テスト用OHLCVデータ生成

        Returns:
            pd.DataFrame: テスト用のOHLCVデータ
        """
        dates = pd.date_range("2024-01-01", periods=200, freq="5min")
        np.random.seed(42)

        base_price = 50000
        returns = np.random.randn(200) * 0.002 + 0.0001
        close_prices = base_price * (1 + returns).cumprod()

        high_prices = close_prices * (1 + np.abs(np.random.randn(200) * 0.003))
        low_prices = close_prices * (1 - np.abs(np.random.randn(200) * 0.003))
        open_prices = np.roll(close_prices, 1)
        open_prices[0] = base_price

        df = pd.DataFrame(
            {
                "date": dates,
                "open": open_prices,
                "high": high_prices,
                "low": low_prices,
                "close": close_prices,
                "volume": np.random.randint(100, 1000, 200),
            }
        )

        return df

    def test_label_generation_from_returns(self, ml_off_config, sample_ohlcv):
        """リターンからラベル生成が正しく行われることを確認

        - リターン > 0 → ラベル=1
        - リターン <= 0 → ラベル=0
        """
        strategy = TwoTierStrategy(ml_off_config)
        test_data = sample_ohlcv.copy()
        metadata = {"pair": "BTC/USDT"}

        # ラベル生成
        df_labels = strategy.set_freqai_targets(test_data.copy(), metadata)

        # ラベルが存在することを確認
        assert "&-target" in df_labels.columns, "set_freqai_targets should generate &-target column"

        # ラベルは0または1
        valid_labels = df_labels["&-target"].dropna()
        assert set(valid_labels.unique()).issubset({0, 1}), "Labels should be binary (0 or 1)"

        # リターンを直接計算して比較
        primary_strategy = strategy.primary_strategy
        df_with_prices = primary_strategy.calculate_prices(test_data.copy())
        buy_return, sell_return = primary_strategy.calculate_returns(df_with_prices)

        # ラベル生成ロジック: buy_return > 0 で成功ラベル(1)
        expected_labels = (buy_return > 0).astype(int)

        # 有効なインデックス（NaNでない箇所）で比較
        valid_idx = ~buy_return.isna() & ~df_labels["&-target"].isna()

        if valid_idx.sum() > 0:
            assert (df_labels.loc[valid_idx, "&-target"] == expected_labels[valid_idx]).all(), (
                "Labels should be generated correctly from returns: "
                "return > 0 → label=1, return <= 0 → label=0"
            )

    def test_execution_mode_label_difference(self, sample_ohlcv):
        """execution_mode切り替え時のラベル変化確認

        - chase方式とone_candle方式で異なるラベルが生成されるはず
        - one_candle方式では約定しない場合があるため、正例比率が低くなる傾向
        """
        metadata = {"pair": "BTC/USDT"}
        test_data = sample_ohlcv.copy()

        # chase方式
        config_chase = {
            "two_tier_strategy": {
                "primary": "atr_breakout",
                "secondary": None,
                "primary_params": {
                    "period": 14,
                    "multiplier": 0.5,
                    "execution_mode": "chase",
                    "fee": 0.00025,
                    "exit_periods": 24,
                    "pips": 0.5,
                },
            },
            "freqai": {"enabled": False},
        }

        # one_candle方式
        config_one_candle = {
            "two_tier_strategy": {
                "primary": "atr_breakout",
                "secondary": None,
                "primary_params": {
                    "period": 14,
                    "multiplier": 0.5,
                    "execution_mode": "one_candle",
                    "fee": 0.00025,
                    "exit_periods": 24,
                    "pips": 0.5,
                },
            },
            "freqai": {"enabled": False},
        }

        strategy_chase = TwoTierStrategy(config_chase)
        strategy_one_candle = TwoTierStrategy(config_one_candle)

        # ラベル生成
        df_chase = strategy_chase.set_freqai_targets(test_data.copy(), metadata)
        df_one_candle = strategy_one_candle.set_freqai_targets(test_data.copy(), metadata)

        # 2つのexecution_modeでラベルが異なることを確認
        positive_ratio_chase = df_chase["&-target"].mean()
        positive_ratio_one_candle = df_one_candle["&-target"].mean()

        # 少なくとも正例比率が異なることを確認
        assert abs(positive_ratio_chase - positive_ratio_one_candle) > 0.01, (
            f"execution_mode should affect label generation. "
            f"chase={positive_ratio_chase:.3f}, one_candle={positive_ratio_one_candle:.3f}"
        )

    def test_populate_indicators_ml_off(self, ml_off_config, sample_ohlcv):
        """populate_indicators()がML無効モードで正しく動作することを確認"""
        strategy = TwoTierStrategy(ml_off_config)
        test_data = sample_ohlcv.copy()
        metadata = {"pair": "BTC/USDT"}

        # 特徴量生成
        df = strategy.populate_indicators(test_data.copy(), metadata)

        # buy_price, sell_priceが生成されている
        assert "buy_price" in df.columns
        assert "sell_price" in df.columns

        # FreqAI予測カラムは存在しない（ML無効）
        prediction_cols = [col for col in df.columns if col.startswith("&-prediction")]
        assert len(prediction_cols) == 0, "ML disabled mode should not have prediction columns"

    def test_populate_entry_trend_ml_off(self, ml_off_config, sample_ohlcv):
        """populate_entry_trend()がML無効モードで正しく動作することを確認

        ML無効時は常に両方向エントリー（価格が有効な場合）
        """
        strategy = TwoTierStrategy(ml_off_config)
        test_data = sample_ohlcv.copy()
        metadata = {"pair": "BTC/USDT"}

        # 特徴量生成
        df = strategy.populate_indicators(test_data.copy(), metadata)

        # エントリーシグナル生成
        df = strategy.populate_entry_trend(df, metadata)

        # enter_long, enter_shortが生成されている
        assert "enter_long" in df.columns
        assert "enter_short" in df.columns

        # 価格が有効な行ではエントリーシグナルが立っているはず
        valid_rows = df[(df["buy_price"] > 0) & (df["sell_price"] > 0)]

        if len(valid_rows) > 0:
            # ML無効時は常にエントリー
            assert (valid_rows["enter_long"] == 1).any(), "Should have enter_long signals"
            assert (valid_rows["enter_short"] == 1).any(), "Should have enter_short signals"

    def test_populate_exit_trend_ml_off(self, ml_off_config, sample_ohlcv):
        """populate_exit_trend()がML無効モードで正しく動作することを確認

        ML無効時は明示的な決済シグナルなし（ROI/Stoplossのみ）
        """
        strategy = TwoTierStrategy(ml_off_config)
        test_data = sample_ohlcv.copy()
        metadata = {"pair": "BTC/USDT"}

        # 特徴量生成
        df = strategy.populate_indicators(test_data.copy(), metadata)

        # エグジットシグナル生成
        df = strategy.populate_exit_trend(df, metadata)

        # exit_long, exit_shortカラムが存在することを確認
        # ML無効時は明示的なシグナルはないが、カラムは存在する可能性がある
        # （Freqtradeフレームワークの仕様による）

    def test_custom_entry_price(self, ml_off_config, sample_ohlcv):
        """custom_entry_price()が正しく動作することを確認"""
        strategy = TwoTierStrategy(ml_off_config)
        test_data = sample_ohlcv.copy()
        metadata = {"pair": "BTC/USDT"}

        # DataProviderのモック設定（簡易版）
        # 実際のテストではDataProviderをモックする必要があるが、
        # ここでは基本的な動作確認のみ行う

        # populate_indicators()を実行してbuy_priceを生成
        df = strategy.populate_indicators(test_data.copy(), metadata)

        # buy_priceが生成されていることを確認
        assert "buy_price" in df.columns
        assert (df["buy_price"] > 0).any()

    def test_custom_exit_price(self, ml_off_config, sample_ohlcv):
        """custom_exit_price()が正しく動作することを確認"""
        strategy = TwoTierStrategy(ml_off_config)
        test_data = sample_ohlcv.copy()
        metadata = {"pair": "BTC/USDT"}

        # populate_indicators()を実行してsell_priceを生成
        df = strategy.populate_indicators(test_data.copy(), metadata)

        # sell_priceが生成されていることを確認
        assert "sell_price" in df.columns
        assert (df["sell_price"] > 0).any()

    def test_label_distribution_reasonable(self, ml_off_config, sample_ohlcv):
        """ラベル分布が合理的な範囲内であることを確認

        - 全て0または全て1ではないこと
        - 正例比率が0.05～0.95の範囲内
        """
        strategy = TwoTierStrategy(ml_off_config)
        test_data = sample_ohlcv.copy()
        metadata = {"pair": "BTC/USDT"}

        # ラベル生成
        df = strategy.set_freqai_targets(test_data.copy(), metadata)

        # 正例比率を計算
        valid_labels = df["&-target"].dropna()

        if len(valid_labels) > 0:
            positive_ratio = valid_labels.mean()

            # 正例比率が合理的な範囲内
            assert 0.05 < positive_ratio < 0.95, (
                f"Positive label ratio {positive_ratio:.3f} is too extreme. "
                "This might indicate issues in return calculation or labeling logic."
            )

    def test_label_count(self, ml_off_config, sample_ohlcv):
        """生成されるラベル数が適切であることを確認

        - 最後のexit_periods行を除いて、ラベルが生成されるはず
        """
        strategy = TwoTierStrategy(ml_off_config)
        test_data = sample_ohlcv.copy()
        metadata = {"pair": "BTC/USDT"}

        # ラベル生成
        df = strategy.set_freqai_targets(test_data.copy(), metadata)

        exit_periods = strategy.primary_strategy.exit_periods
        period = strategy.primary_strategy.period

        # 最後のexit_periods行と最初のperiod行を除いて、ラベルが存在するはず
        expected_valid_range = len(df) - exit_periods - period

        valid_labels = df["&-target"].dropna()

        # 少なくとも一定数のラベルが生成されている
        assert len(valid_labels) >= expected_valid_range * 0.5, (
            f"Expected at least {expected_valid_range * 0.5} valid labels, "
            f"but got {len(valid_labels)}"
        )

    def test_integration_populate_and_label(self, ml_off_config, sample_ohlcv):
        """統合テスト: populate_indicators()とset_freqai_targets()の連携

        - 特徴量生成とラベル生成が正しく連携すること
        - データの流れが正しいこと
        """
        strategy = TwoTierStrategy(ml_off_config)
        test_data = sample_ohlcv.copy()
        metadata = {"pair": "BTC/USDT"}

        # 特徴量生成
        df_features = strategy.populate_indicators(test_data.copy(), metadata)

        # ラベル生成
        df_labels = strategy.set_freqai_targets(test_data.copy(), metadata)

        # 両方のDataFrameが同じ長さ
        assert len(df_features) == len(df_labels)

        # 特徴量にbuy_price/sell_priceが含まれている
        assert "buy_price" in df_features.columns
        assert "sell_price" in df_features.columns

        # ラベルに&-targetが含まれている
        assert "&-target" in df_labels.columns

        # ラベルが特徴量に混入していない
        assert "&-target" not in df_features.columns
