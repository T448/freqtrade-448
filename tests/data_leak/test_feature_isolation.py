"""Feature isolation tests - データリーク検出（特徴量の未来データ依存性チェック）

このモジュールは、populate_indicators()で生成される特徴量が
未来のデータに依存していないことを検証するテストを提供します。

テスト方針:
- 時刻tの特徴量は時刻t+1以降のデータに依存してはいけない
- 部分的なデータで計算した特徴量と、全データで計算した特徴量が一致すること
- .shift(-n)の使用を静的に検出
"""

import inspect
import re

import numpy as np
import pandas as pd
import pytest

from user_data.strategies.two_tier_strategy import TwoTierStrategy


class TestFeatureIsolation:
    """特徴量の未来データ依存性チェックテストスイート"""

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

    def test_no_future_data_in_features(self, ml_off_config, sample_ohlcv):
        """特徴量が未来データに依存していないことを検証

        時刻tまでのデータで計算した特徴量と、全データで計算した特徴量が
        時刻tで一致することを確認する（未来データ不使用の証明）
        """
        strategy = TwoTierStrategy(ml_off_config)
        test_data = sample_ohlcv.copy()
        metadata = {"pair": "BTC/USDT"}

        # 全データで特徴量計算
        df_full = strategy.populate_indicators(test_data.copy(), metadata)

        # 複数の時点で部分データと比較
        test_indices = [50, 100, 150]

        for i in test_indices:
            # 時刻iまでのデータで特徴量計算
            df_partial = strategy.populate_indicators(test_data.iloc[: i + 1].copy(), metadata)

            # 時刻iの特徴量が一致することを確認
            for col in df_full.columns:
                # FreqAI予測カラムと元データは除外
                if col.startswith("&-prediction") or col.startswith("%"):
                    continue
                if col in ["open", "high", "low", "close", "volume", "date"]:
                    continue

                # 特徴量の値が一致するか確認（NaN同士も一致とみなす）
                full_val = df_full.iloc[i][col]
                partial_val = df_partial.iloc[-1][col]

                assert np.isclose(full_val, partial_val, equal_nan=True), (
                    f"Feature '{col}' at index {i} depends on future data: "
                    f"full={full_val}, partial={partial_val}"
                )

    def test_no_shift_negative_in_populate_indicators(self, ml_off_config):
        """populate_indicators()で.shift(-n)が使われていないことを確認

        静的解析により、populate_indicators()のソースコード内で
        .shift(-n)パターンを検出する
        """
        strategy = TwoTierStrategy(ml_off_config)

        # ソースコード検査
        source = inspect.getsource(strategy.populate_indicators)

        # .shift(-n)パターンを検出（負の値によるシフト）
        negative_shifts = re.findall(r"\.shift\s*\(\s*-\s*\d+", source)

        assert len(negative_shifts) == 0, (
            f"Found {len(negative_shifts)} .shift(-n) in populate_indicators: {negative_shifts}. "
            "This indicates potential data leakage (using future data)."
        )

    def test_primary_strategy_calculate_prices_no_future_data(self, ml_off_config, sample_ohlcv):
        """1次戦略のcalculate_prices()が未来データを使用していないことを確認

        指値価格計算は時刻tのデータのみを使用すべき
        """
        strategy = TwoTierStrategy(ml_off_config)
        primary_strategy = strategy.primary_strategy
        test_data = sample_ohlcv.copy()

        # 全データで価格計算
        df_full = primary_strategy.calculate_prices(test_data.copy())

        # 複数の時点で部分データと比較
        test_indices = [50, 100, 150]

        for i in test_indices:
            # 時刻iまでのデータで価格計算
            df_partial = primary_strategy.calculate_prices(test_data.iloc[: i + 1].copy())

            # buy_price, sell_priceが一致することを確認
            for col in ["buy_price", "sell_price"]:
                full_val = df_full.iloc[i][col]
                partial_val = df_partial.iloc[-1][col]

                assert np.isclose(full_val, partial_val, equal_nan=True), (
                    f"Price calculation '{col}' at index {i} depends on future data: "
                    f"full={full_val}, partial={partial_val}"
                )

    def test_rolling_calculations_have_min_periods(self, ml_off_config, sample_ohlcv):
        """rolling計算でmin_periodsが適切に設定されているか確認

        min_periodsが設定されていない場合、初期の値が不正確になる可能性がある
        このテストでは、ATR計算でrollingが使用されていることを確認
        """
        strategy = TwoTierStrategy(ml_off_config)
        test_data = sample_ohlcv.copy()
        metadata = {"pair": "BTC/USDT"}

        df = strategy.populate_indicators(test_data.copy(), metadata)

        # ATR期間(14)未満のインデックスでは、buy_price/sell_priceがNaNであるべき
        # ATRが計算できない期間は価格計算も不可能
        period = strategy.primary_strategy.period

        # 最初のperiod-1個はNaNまたは計算が不安定
        initial_rows = df.iloc[: period - 1]

        # buy_price/sell_priceのNaN数をチェック
        # 完全にNaNである必要はないが、多くがNaNであることを期待
        nan_ratio = initial_rows["buy_price"].isna().sum() / len(initial_rows)

        assert nan_ratio > 0.5, (
            f"Expected high NaN ratio in initial {period - 1} rows, got {nan_ratio:.2f}. "
            "This might indicate missing min_periods in rolling calculations."
        )

    def test_no_future_data_in_primary_strategy_source(self, ml_off_config):
        """1次戦略のcalculate_prices()内で.shift(-n)が使われていないことを確認"""
        strategy = TwoTierStrategy(ml_off_config)
        primary_strategy = strategy.primary_strategy

        # calculate_prices()のソースコード検査
        source = inspect.getsource(primary_strategy.calculate_prices)

        # .shift(-n)パターンを検出
        negative_shifts = re.findall(r"\.shift\s*\(\s*-\s*\d+", source)

        assert len(negative_shifts) == 0, (
            f"Found {len(negative_shifts)} .shift(-n) in calculate_prices: {negative_shifts}. "
            "Price calculation should not use future data."
        )

    def test_feature_consistency_across_runs(self, ml_off_config, sample_ohlcv):
        """同じデータで複数回特徴量計算を実行した場合、結果が一致することを確認

        ランダム性や時刻依存の処理がない限り、同じ入力で同じ出力が得られるべき
        """
        strategy = TwoTierStrategy(ml_off_config)
        test_data = sample_ohlcv.copy()
        metadata = {"pair": "BTC/USDT"}

        # 1回目の計算
        df1 = strategy.populate_indicators(test_data.copy(), metadata)

        # 2回目の計算
        df2 = strategy.populate_indicators(test_data.copy(), metadata)

        # 特徴量カラムが一致することを確認
        feature_cols = [
            col
            for col in df1.columns
            if col not in ["open", "high", "low", "close", "volume", "date"]
            and not col.startswith("&-prediction")
            and not col.startswith("%")
        ]

        for col in feature_cols:
            assert np.allclose(df1[col].values, df2[col].values, equal_nan=True), (
                f"Feature '{col}' is not consistent across runs"
            )

    def test_edge_case_single_row_dataframe(self, ml_off_config):
        """エッジケース: 1行だけのDataFrameでエラーにならないことを確認"""
        strategy = TwoTierStrategy(ml_off_config)
        metadata = {"pair": "BTC/USDT"}

        df = pd.DataFrame(
            {
                "date": [pd.Timestamp("2024-01-01")],
                "open": [50000.0],
                "high": [50100.0],
                "low": [49900.0],
                "close": [50050.0],
                "volume": [100],
            }
        )

        # エラーにならないことを確認
        result = strategy.populate_indicators(df.copy(), metadata)

        # 結果が返されることを確認
        assert len(result) == 1
        assert "buy_price" in result.columns
        assert "sell_price" in result.columns
