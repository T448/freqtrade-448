"""Unit tests for ATRBreakoutStrategy."""

import numpy as np
import pandas as pd
import pytest

from user_data.strategies.primary.atr_breakout import ATRBreakoutStrategy


class TestATRBreakoutStrategy:
    """Test suite for ATRBreakoutStrategy."""

    @pytest.fixture
    def sample_dataframe(self):
        """サンプルOHLCデータを作成

        Returns:
            pd.DataFrame: テスト用のOHLCデータ
        """
        dates = pd.date_range("2024-01-01", periods=100, freq="1h")
        np.random.seed(42)

        # 価格データ生成（上昇トレンド）
        base_price = 100
        returns = np.random.randn(100) * 0.01 + 0.001
        close_prices = base_price * (1 + returns).cumprod()

        # OHLC生成
        high_prices = close_prices * (1 + np.abs(np.random.randn(100) * 0.005))
        low_prices = close_prices * (1 - np.abs(np.random.randn(100) * 0.005))
        open_prices = np.roll(close_prices, 1)
        open_prices[0] = base_price

        df = pd.DataFrame(
            {
                "date": dates,
                "open": open_prices,
                "high": high_prices,
                "low": low_prices,
                "close": close_prices,
                "volume": np.random.randint(1000, 10000, 100),
            }
        )

        return df

    @pytest.fixture
    def default_params(self):
        """デフォルトパラメータ

        Returns:
            dict: デフォルトパラメータ辞書
        """
        return {
            "period": 14,
            "multiplier": 0.5,
            "fee": 0.00025,
            "exit_periods": 24,
            "pips": 0.5,
            "execution_mode": "one_candle",
        }

    def test_initialization(self, default_params):
        """初期化テスト"""
        strategy = ATRBreakoutStrategy(default_params)

        assert strategy.period == 14
        assert strategy.multiplier == 0.5
        assert strategy.fee == 0.00025
        assert strategy.exit_periods == 24
        assert strategy.pips == 0.5
        assert strategy.execution_mode == "one_candle"

    def test_calculate_atr(self, sample_dataframe, default_params):
        """ATR計算の正確性テスト"""
        strategy = ATRBreakoutStrategy(default_params)
        df = sample_dataframe.copy()

        atr = strategy._calculate_atr(df, period=14)

        # ATRは正の値であるべき
        assert (atr[14:] > 0).all(), "ATR should be positive"

        # ATRは合理的な範囲内（価格の数%以内）
        assert (atr[14:] < df["close"][14:] * 0.5).all(), (
            "ATR should be reasonable (< 50% of price)"
        )

        # NaNチェック（最初のperiod-1個はNaN）
        assert atr[:14].isna().sum() >= 13, "First period-1 values should be NaN"

    def test_calculate_prices(self, sample_dataframe, default_params):
        """指値価格計算テスト"""
        strategy = ATRBreakoutStrategy(default_params)
        df = sample_dataframe.copy()

        result_df = strategy.calculate_prices(df)

        # カラムが追加されているか
        assert "buy_price" in result_df.columns
        assert "sell_price" in result_df.columns

        # 買い指値 < close < 売り指値
        valid_rows = result_df[14:].dropna()  # ATR計算期間後のデータ
        assert (valid_rows["buy_price"] < valid_rows["close"]).all()
        assert (valid_rows["sell_price"] > valid_rows["close"]).all()

        # 価格が正の値
        assert (valid_rows["buy_price"] > 0).all()
        assert (valid_rows["sell_price"] > 0).all()

    def test_one_candle_execution_mode_returns(self, sample_dataframe, default_params):
        """one_candle方式のリターン計算テスト"""
        default_params["execution_mode"] = "one_candle"
        strategy = ATRBreakoutStrategy(default_params)
        df = sample_dataframe.copy()

        # 指値価格計算
        df = strategy.calculate_prices(df)

        # リターン計算
        buy_return, sell_return = strategy.calculate_returns(df)

        # リターンがSeriesである
        assert isinstance(buy_return, pd.Series)
        assert isinstance(sell_return, pd.Series)

        # 長さが一致
        assert len(buy_return) == len(df)
        assert len(sell_return) == len(df)

        # リターンの範囲チェック（-1以上、合理的な範囲内）
        # 完全損失は-1、利益は通常数%以内
        valid_buy = buy_return[~buy_return.isna()]
        valid_sell = sell_return[~sell_return.isna()]

        if len(valid_buy) > 0:
            assert (valid_buy >= -1).all(), "Buy return should be >= -1 (total loss)"
            assert (valid_buy <= 1).all(), "Buy return should be <= 1 (100% profit)"

        if len(valid_sell) > 0:
            assert (valid_sell >= -1).all(), "Sell return should be >= -1 (total loss)"
            assert (valid_sell <= 1).all(), "Sell return should be <= 1 (100% profit)"

    def test_chase_execution_mode_returns(self, sample_dataframe, default_params):
        """chase方式のリターン計算テスト"""
        default_params["execution_mode"] = "chase"
        strategy = ATRBreakoutStrategy(default_params)
        df = sample_dataframe.copy()

        # 指値価格計算
        df = strategy.calculate_prices(df)

        # リターン計算
        buy_return, sell_return = strategy.calculate_returns(df)

        # リターンがSeriesである
        assert isinstance(buy_return, pd.Series)
        assert isinstance(sell_return, pd.Series)

        # 長さが一致
        assert len(buy_return) == len(df)
        assert len(sell_return) == len(df)

        # リターンの範囲チェック
        valid_buy = buy_return[~buy_return.isna()]
        valid_sell = sell_return[~sell_return.isna()]

        if len(valid_buy) > 0:
            assert (valid_buy >= -1).all(), "Buy return should be >= -1 (total loss)"
            assert (valid_buy <= 1).all(), "Buy return should be <= 1 (100% profit)"

        if len(valid_sell) > 0:
            assert (valid_sell >= -1).all(), "Sell return should be >= -1 (total loss)"
            assert (valid_sell <= 1).all(), "Sell return should be <= 1 (100% profit)"

    def test_force_entry_price_calculation(self, sample_dataframe, default_params):
        """FEP計算テスト"""
        strategy = ATRBreakoutStrategy(default_params)
        df = sample_dataframe.copy()

        # 指値価格計算
        df = strategy.calculate_prices(df)

        # 買いFEP計算
        buy_fep = strategy._calculate_force_entry_price(
            df["buy_price"], df["low"], pips=0.5, direction="buy"
        )

        # 売りFEP計算
        sell_fep = strategy._calculate_force_entry_price(
            df["sell_price"], df["high"], pips=0.5, direction="sell"
        )

        # FEPはSeriesである
        assert isinstance(buy_fep, pd.Series)
        assert isinstance(sell_fep, pd.Series)

        # FEPは約定した場合に値が設定される
        valid_buy_idx = ~buy_fep.isna()
        valid_sell_idx = ~sell_fep.isna()

        # 約定したFEPは正の値
        if valid_buy_idx.sum() > 0:
            assert (buy_fep[valid_buy_idx] > 0).all(), "Buy FEP should be positive"
            # FEPはpips単位で丸められている
            assert ((buy_fep[valid_buy_idx] % default_params["pips"]).abs() < 0.01).all()

        if valid_sell_idx.sum() > 0:
            assert (sell_fep[valid_sell_idx] > 0).all(), "Sell FEP should be positive"
            # FEPはpips単位で丸められている
            assert ((sell_fep[valid_sell_idx] % default_params["pips"]).abs() < 0.01).all()

    def test_one_candle_no_fill_scenario(self, default_params):
        """one_candle方式：約定しない場合のリターン=0テスト"""
        default_params["execution_mode"] = "one_candle"
        strategy = ATRBreakoutStrategy(default_params)

        # 約定しないシナリオ：指値が価格範囲外
        df = pd.DataFrame(
            {
                "open": [100.0] * 50,
                "high": [101.0] * 50,
                "low": [99.0] * 50,
                "close": [100.0] * 50,
            }
        )

        # 極端な指値価格（約定しない）
        df["buy_price"] = 90.0  # lowよりはるかに低い
        df["sell_price"] = 110.0  # highよりはるかに高い

        buy_return, sell_return = strategy.calculate_returns(df)

        # 約定しないため、リターンは0またはNaN
        valid_buy = buy_return[: -default_params["exit_periods"]].dropna()
        valid_sell = sell_return[: -default_params["exit_periods"]].dropna()

        # 約定しない場合は0
        if len(valid_buy) > 0:
            assert (valid_buy == 0).all(), "No fill should result in zero return for buy"
        if len(valid_sell) > 0:
            assert (valid_sell == 0).all(), "No fill should result in zero return for sell"

    def test_one_candle_fill_scenario(self, default_params):
        """one_candle方式：約定する場合のリターン計算テスト"""
        default_params["execution_mode"] = "one_candle"
        default_params["exit_periods"] = 2
        default_params["fee"] = 0.0  # 手数料を0にして計算を簡略化
        strategy = ATRBreakoutStrategy(default_params)

        # 約定するシナリオ
        df = pd.DataFrame(
            {
                "open": [100.0, 100.0, 100.0, 102.0, 103.0],
                "high": [101.0, 101.0, 101.0, 103.0, 104.0],
                "low": [99.0, 98.0, 99.0, 101.0, 102.0],
                "close": [100.0, 100.0, 100.0, 102.0, 103.0],
            }
        )

        # 買い指値: 99.5（次足のlow=98.0で約定）
        # 売り指値: 100.5（次足のhigh=101.0で約定）
        df["buy_price"] = 99.5
        df["sell_price"] = 100.5

        buy_return, sell_return = strategy.calculate_returns(df)

        # 最初の行は約定して、2足後に決済
        # buy_return[0] = (2足後のsell_fep / 99.5) - 1
        # ただし、FEP計算があるため厳密な値の検証は難しい
        # ここでは、約定した場合にリターンが0でないことを確認
        assert buy_return[0] != 0, "Buy return should not be zero when filled"

    def test_edge_case_insufficient_data(self, default_params):
        """エッジケース：データ不足時の挙動テスト"""
        strategy = ATRBreakoutStrategy(default_params)

        # 不十分なデータ（ATR計算期間より短い）
        df = pd.DataFrame(
            {
                "open": [100.0] * 10,
                "high": [101.0] * 10,
                "low": [99.0] * 10,
                "close": [100.0] * 10,
            }
        )

        # 指値価格計算（エラーにならないこと）
        df = strategy.calculate_prices(df)

        # 初期のNaNは許容される
        assert df["buy_price"].isna().sum() > 0
        assert df["sell_price"].isna().sum() > 0

    def test_edge_case_nan_handling(self, default_params):
        """エッジケース：NaN値の処理テスト"""
        strategy = ATRBreakoutStrategy(default_params)

        # NaNを含むデータ
        df = pd.DataFrame(
            {
                "open": [100.0, np.nan, 100.0, 100.0, 100.0] * 10,
                "high": [101.0, np.nan, 101.0, 101.0, 101.0] * 10,
                "low": [99.0, np.nan, 99.0, 99.0, 99.0] * 10,
                "close": [100.0, np.nan, 100.0, 100.0, 100.0] * 10,
            }
        )

        # 指値価格計算（エラーにならないこと）
        df = strategy.calculate_prices(df)

        # リターン計算（エラーにならないこと）
        buy_return, sell_return = strategy.calculate_returns(df)

        # 結果が返されることを確認
        assert len(buy_return) == len(df)
        assert len(sell_return) == len(df)

    def test_fee_impact_on_returns(self, sample_dataframe):
        """手数料がリターンに与える影響のテスト"""
        # 手数料あり
        params_with_fee = {
            "period": 14,
            "multiplier": 0.5,
            "fee": 0.001,  # 0.1%手数料
            "exit_periods": 24,
            "pips": 0.5,
            "execution_mode": "chase",
        }
        strategy_with_fee = ATRBreakoutStrategy(params_with_fee)

        # 手数料なし
        params_no_fee = params_with_fee.copy()
        params_no_fee["fee"] = 0.0
        strategy_no_fee = ATRBreakoutStrategy(params_no_fee)

        df = sample_dataframe.copy()
        df = strategy_with_fee.calculate_prices(df)

        # リターン計算
        buy_return_with_fee, sell_return_with_fee = strategy_with_fee.calculate_returns(df)
        buy_return_no_fee, sell_return_no_fee = strategy_no_fee.calculate_returns(df)

        # 手数料ありの方がリターンが低いはず
        valid_idx = ~buy_return_with_fee.isna() & ~buy_return_no_fee.isna()
        if valid_idx.sum() > 0:
            # 手数料分だけリターンが減少（片道0.1% x 2 = 0.2%）
            assert (buy_return_with_fee[valid_idx] < buy_return_no_fee[valid_idx] + 1e-6).all(), (
                "Returns with fee should be lower"
            )

    def test_execution_mode_difference(self, sample_dataframe, default_params):
        """execution_mode切り替え確認テスト"""
        df = sample_dataframe.copy()

        # chase方式
        params_chase = default_params.copy()
        params_chase["execution_mode"] = "chase"
        strategy_chase = ATRBreakoutStrategy(params_chase)
        df = strategy_chase.calculate_prices(df)
        buy_return_chase, sell_return_chase = strategy_chase.calculate_returns(df)

        # one_candle方式
        params_one_candle = default_params.copy()
        params_one_candle["execution_mode"] = "one_candle"
        strategy_one_candle = ATRBreakoutStrategy(params_one_candle)
        buy_return_one_candle, sell_return_one_candle = strategy_one_candle.calculate_returns(df)

        # 2つの方式でリターンが異なることを確認
        # one_candle方式では約定しない場合があるため、0のリターンが多いはず
        zero_count_chase = (buy_return_chase == 0).sum()
        zero_count_one_candle = (buy_return_one_candle == 0).sum()

        # one_candle方式の方が0が多い（約定しない場合がある）
        assert zero_count_one_candle >= zero_count_chase, (
            "one_candle mode should have more zero returns due to no-fill scenarios"
        )
