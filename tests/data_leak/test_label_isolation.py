"""Label isolation tests - ラベル隔離確認（ラベル生成での未来データ隔離）

このモジュールは、set_freqai_targets()で生成されるラベルが
特徴量DataFrameに混入していないことを検証するテストを提供します。

テスト方針:
- ラベル生成で使用する未来データ(&-target)が特徴量に含まれないこと
- リターン計算で未来データを正しく使用していること
- calculate_returns()が適切に未来データを参照していること
"""

import inspect
import re

import numpy as np
import pandas as pd
import pytest

from user_data.strategies.two_tier_strategy import TwoTierStrategy


class TestLabelIsolation:
    """ラベル隔離確認テストスイート"""

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

    def test_label_future_data_isolation(self, ml_off_config, sample_ohlcv):
        """ラベル生成で使用する未来データが特徴量に混入していないことを確認

        - populate_indicators()で生成される特徴量に&-targetが含まれないこと
        - set_freqai_targets()で生成されるラベルが特徴量と分離されていること
        """
        strategy = TwoTierStrategy(ml_off_config)
        test_data = sample_ohlcv.copy()
        metadata = {"pair": "BTC/USDT"}

        # 特徴量生成
        df_features = strategy.populate_indicators(test_data.copy(), metadata)

        # ラベル生成
        df_labels = strategy.set_freqai_targets(test_data.copy(), metadata)

        # ラベルカラム(&-target)が特徴量DataFrameに存在しないことを確認
        label_cols = [col for col in df_labels.columns if col.startswith("&-target")]

        assert len(label_cols) > 0, "set_freqai_targets should generate &-target columns"

        for col in label_cols:
            assert col not in df_features.columns, (
                f"Label column '{col}' leaked into features. "
                "Labels must be isolated from features to prevent data leakage."
            )

    def test_return_calculation_uses_future_data_correctly(self, ml_off_config, sample_ohlcv):
        """calculate_returns()が正しく未来データを使用していることを確認

        - 最後のexit_periods行はリターン計算不可（未来データ不足）
        - それ以前の行はリターンが計算されている
        """
        strategy = TwoTierStrategy(ml_off_config)
        primary_strategy = strategy.primary_strategy
        test_data = sample_ohlcv.copy()

        # 価格計算
        df = primary_strategy.calculate_prices(test_data.copy())

        # リターン計算
        buy_return, sell_return = primary_strategy.calculate_returns(df)

        exit_periods = primary_strategy.exit_periods

        # 最後のexit_periods行はリターン計算不可（未来データ不足）
        last_rows_buy = buy_return.iloc[-exit_periods:]
        last_rows_sell = sell_return.iloc[-exit_periods:]

        # NaNまたは0である（未来データがないため計算不可）
        assert last_rows_buy.isna().all() or (last_rows_buy == 0).all(), (
            f"Last {exit_periods} rows of buy_return should be NaN or 0 "
            "(no future data available for return calculation)"
        )

        assert last_rows_sell.isna().all() or (last_rows_sell == 0).all(), (
            f"Last {exit_periods} rows of sell_return should be NaN or 0 "
            "(no future data available for return calculation)"
        )

        # それ以前の行はリターンが計算されている（NaNでない行が存在）
        earlier_rows_buy = buy_return.iloc[:-exit_periods]
        earlier_rows_sell = sell_return.iloc[:-exit_periods]

        # ATR計算期間を考慮して、有効な行があることを確認
        assert not earlier_rows_buy.isna().all(), (
            "Earlier rows should have calculated returns (not all NaN)"
        )

        assert not earlier_rows_sell.isna().all(), (
            "Earlier rows should have calculated returns (not all NaN)"
        )

    def test_calculate_returns_uses_shift_negative(self, ml_off_config):
        """calculate_returns()が.shift(-n)を使用していることを確認

        リターン計算では未来データを参照する必要があるため、
        .shift(-n)の使用は正当である
        """
        strategy = TwoTierStrategy(ml_off_config)
        primary_strategy = strategy.primary_strategy

        # calculate_returns()のソースコード検査
        source = inspect.getsource(primary_strategy.calculate_returns)

        # .shift(-n)パターンを検出
        negative_shifts = re.findall(r"\.shift\s*\(\s*-\s*\d+", source)

        # calculate_returns()では未来データを使用するため、.shift(-n)の使用は正当
        assert len(negative_shifts) > 0, (
            "calculate_returns should use .shift(-n) to access future data for return calculation. "
            "No .shift(-n) found, which might indicate incorrect implementation."
        )

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

    def test_no_target_columns_in_populate_indicators(self, ml_off_config, sample_ohlcv):
        """populate_indicators()が&-targetカラムを生成しないことを確認

        ラベル生成はset_freqai_targets()の責務であり、
        populate_indicators()では特徴量のみを生成すべき
        """
        strategy = TwoTierStrategy(ml_off_config)
        test_data = sample_ohlcv.copy()
        metadata = {"pair": "BTC/USDT"}

        # 特徴量生成
        df = strategy.populate_indicators(test_data.copy(), metadata)

        # &-targetカラムが存在しないことを確認
        target_cols = [col for col in df.columns if col.startswith("&-target")]

        assert len(target_cols) == 0, (
            f"populate_indicators should not generate &-target columns. Found: {target_cols}"
        )

    def test_set_freqai_targets_source_code_check(self, ml_off_config):
        """set_freqai_targets()がcalculate_returns()を呼び出していることを確認

        静的解析により、ソースコード内でcalculate_returns()が
        呼び出されていることを確認
        """
        strategy = TwoTierStrategy(ml_off_config)

        # set_freqai_targets()のソースコード検査
        source = inspect.getsource(strategy.set_freqai_targets)

        # calculate_returns()の呼び出しを検出
        assert "calculate_returns" in source, (
            "set_freqai_targets should call calculate_returns() to generate labels from returns"
        )

    def test_label_positive_ratio(self, ml_off_config, sample_ohlcv):
        """ラベルの正例比率が合理的な範囲内であることを確認

        - ラベルが全て0または全て1の場合、学習が困難
        - 正例比率が0.1～0.9の範囲内であることを期待
        """
        strategy = TwoTierStrategy(ml_off_config)
        test_data = sample_ohlcv.copy()
        metadata = {"pair": "BTC/USDT"}

        # ラベル生成
        df_labels = strategy.set_freqai_targets(test_data.copy(), metadata)

        # 正例比率を計算
        valid_labels = df_labels["&-target"].dropna()

        if len(valid_labels) > 0:
            positive_ratio = valid_labels.mean()

            # 正例比率が合理的な範囲内（極端に偏っていない）
            assert 0.05 < positive_ratio < 0.95, (
                f"Positive label ratio {positive_ratio:.3f} is too extreme. "
                "This might indicate issues in return calculation or labeling logic."
            )

    def test_execution_mode_affects_labels(self, sample_ohlcv):
        """execution_mode切り替え時にラベルが変化することを確認

        - chase方式とone_candle方式で異なるラベルが生成されるはず
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
        # one_candle方式では約定しない場合があるため、正例比率が低くなる傾向
        positive_ratio_chase = df_chase["&-target"].mean()
        positive_ratio_one_candle = df_one_candle["&-target"].mean()

        # 少なくとも正例比率が異なることを確認（完全一致は考えにくい）
        assert abs(positive_ratio_chase - positive_ratio_one_candle) > 0.01, (
            f"execution_mode should affect label generation. "
            f"chase={positive_ratio_chase:.3f}, one_candle={positive_ratio_one_candle:.3f}"
        )
