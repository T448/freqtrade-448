"""
TechnicalIndicatorEngineのテストケース

richmanbtcチュートリアルに基づく機械学習特徴量として使用する
10以上の標準テクニカル指標計算エンジンのテスト

Requirements tested:
- 2.1: 移動平均、RSI、MACD、ボリンジャーバンド等10以上の標準指標を計算
- 2.3: 欠損値と外れ値の適切な処理によるデータ品質確保
- 2.5: すべての指標の時間同期とデータポイント整合性確保
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from user_data.strategies.utils.technical_indicator_engine import TechnicalIndicatorEngine


class TestTechnicalIndicatorEngine:
    """TechnicalIndicatorEngineクラスのテストケース"""

    @pytest.fixture
    def sample_ohlcv_data(self):
        """テスト用のOHLCVデータを生成"""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100, freq="1H")

        # 現実的な価格変動を模擬
        base_price = 100.0
        price_changes = np.random.normal(0, 0.02, 100)  # 2%の標準偏差
        prices = []
        current_price = base_price

        for change in price_changes:
            current_price *= 1 + change
            prices.append(current_price)

        # OHLCV データ生成
        data = []
        for i, price in enumerate(prices):
            high = price * (1 + abs(np.random.normal(0, 0.01)))
            low = price * (1 - abs(np.random.normal(0, 0.01)))
            open_price = prices[i - 1] if i > 0 else price
            close = price
            volume = np.random.randint(1000, 10000)

            data.append(
                {
                    "date": dates[i],
                    "open": open_price,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                }
            )

        df = pd.DataFrame(data)
        df.set_index("date", inplace=True)
        return df

    @pytest.fixture
    def insufficient_data(self):
        """不十分なデータ（少ないサンプル数）"""
        dates = pd.date_range("2023-01-01", periods=5, freq="1H")
        data = {
            "open": [100, 101, 102, 103, 104],
            "high": [101, 102, 103, 104, 105],
            "low": [99, 100, 101, 102, 103],
            "close": [100.5, 101.5, 102.5, 103.5, 104.5],
            "volume": [1000, 1100, 1200, 1300, 1400],
        }
        df = pd.DataFrame(data, index=dates)
        return df

    @pytest.fixture
    def engine(self):
        """TechnicalIndicatorEngineインスタンス"""
        return TechnicalIndicatorEngine()

    def test_initialization(self, engine):
        """初期化テスト"""
        assert isinstance(engine, TechnicalIndicatorEngine)
        assert hasattr(engine, "calculate_all_indicators")

    def test_sma_calculation(self, engine, sample_ohlcv_data):
        """移動平均（SMA）計算テスト"""
        result = engine.calculate_sma(sample_ohlcv_data, period=14)

        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_ohlcv_data)
        assert result.name == "sma_14"

        # 最初の13個はNaN、14個目から値が入る
        assert pd.isna(result.iloc[0:13]).all()
        assert not pd.isna(result.iloc[13])

    def test_ema_calculation(self, engine, sample_ohlcv_data):
        """指数移動平均（EMA）計算テスト"""
        result = engine.calculate_ema(sample_ohlcv_data, period=12)

        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_ohlcv_data)
        assert result.name == "ema_12"

        # EMAは最初の値から計算されるため、最初の値は入る
        assert not pd.isna(result.iloc[11])

    def test_rsi_calculation(self, engine, sample_ohlcv_data):
        """RSI計算テスト"""
        result = engine.calculate_rsi(sample_ohlcv_data, period=14)

        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_ohlcv_data)
        assert result.name == "rsi_14"

        # RSI値の範囲チェック（0-100）
        valid_values = result.dropna()
        assert (valid_values >= 0).all()
        assert (valid_values <= 100).all()

    def test_macd_calculation(self, engine, sample_ohlcv_data):
        """MACD計算テスト"""
        result = engine.calculate_macd(sample_ohlcv_data)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_ohlcv_data)
        assert "macd" in result.columns
        assert "macd_signal" in result.columns
        assert "macd_histogram" in result.columns

    def test_bollinger_bands_calculation(self, engine, sample_ohlcv_data):
        """ボリンジャーバンド計算テスト"""
        result = engine.calculate_bollinger_bands(sample_ohlcv_data, period=20)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_ohlcv_data)
        assert "bb_upper" in result.columns
        assert "bb_middle" in result.columns
        assert "bb_lower" in result.columns

        # 上部バンド > 中部バンド > 下部バンド の関係チェック
        valid_rows = result.dropna()
        if len(valid_rows) > 0:
            assert (valid_rows["bb_upper"] >= valid_rows["bb_middle"]).all()
            assert (valid_rows["bb_middle"] >= valid_rows["bb_lower"]).all()

    def test_stochastic_calculation(self, engine, sample_ohlcv_data):
        """ストキャスティクス計算テスト"""
        result = engine.calculate_stochastic(sample_ohlcv_data, k_period=14, d_period=3)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_ohlcv_data)
        assert "stoch_k" in result.columns
        assert "stoch_d" in result.columns

        # 値の範囲チェック（0-100）
        for col in ["stoch_k", "stoch_d"]:
            valid_values = result[col].dropna()
            if len(valid_values) > 0:
                assert (valid_values >= 0).all()
                assert (valid_values <= 100).all()

    def test_williams_r_calculation(self, engine, sample_ohlcv_data):
        """ウィリアムズ%R計算テスト"""
        result = engine.calculate_williams_r(sample_ohlcv_data, period=14)

        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_ohlcv_data)
        assert result.name == "williams_r_14"

        # 値の範囲チェック（-100 ~ 0）
        valid_values = result.dropna()
        if len(valid_values) > 0:
            assert (valid_values >= -100).all()
            assert (valid_values <= 0).all()

    def test_atr_calculation(self, engine, sample_ohlcv_data):
        """ATR計算テスト"""
        result = engine.calculate_atr(sample_ohlcv_data, period=14)

        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_ohlcv_data)
        assert result.name == "atr_14"

        # ATR値は正の値
        valid_values = result.dropna()
        if len(valid_values) > 0:
            assert (valid_values > 0).all()

    def test_adx_calculation(self, engine, sample_ohlcv_data):
        """ADX計算テスト"""
        result = engine.calculate_adx(sample_ohlcv_data, period=14)

        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_ohlcv_data)
        assert result.name == "adx_14"

        # ADX値の範囲チェック（0-100）
        valid_values = result.dropna()
        if len(valid_values) > 0:
            assert (valid_values >= 0).all()
            assert (valid_values <= 100).all()

    def test_cci_calculation(self, engine, sample_ohlcv_data):
        """CCI計算テスト"""
        result = engine.calculate_cci(sample_ohlcv_data, period=20)

        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_ohlcv_data)
        assert result.name == "cci_20"

    def test_mfi_calculation(self, engine, sample_ohlcv_data):
        """MFI計算テスト"""
        result = engine.calculate_mfi(sample_ohlcv_data, period=14)

        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_ohlcv_data)
        assert result.name == "mfi_14"

        # MFI値の範囲チェック（0-100）
        valid_values = result.dropna()
        if len(valid_values) > 0:
            assert (valid_values >= 0).all()
            assert (valid_values <= 100).all()

    def test_calculate_all_indicators(self, engine, sample_ohlcv_data):
        """全指標一括計算テスト - 要件2.1の10以上の指標"""
        result = engine.calculate_all_indicators(sample_ohlcv_data)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_ohlcv_data)

        # 10以上の指標が計算されることを確認
        expected_indicators = [
            "sma_14",
            "sma_50",
            "ema_12",
            "ema_26",
            "rsi_14",
            "macd",
            "macd_signal",
            "macd_histogram",
            "bb_upper",
            "bb_middle",
            "bb_lower",
            "stoch_k",
            "stoch_d",
            "williams_r_14",
            "atr_14",
            "adx_14",
            "cci_20",
            "mfi_14",
        ]

        for indicator in expected_indicators:
            assert indicator in result.columns, f"指標 {indicator} が見つかりません"

        # 最低18の指標（10以上の要件を満たす）
        assert len(expected_indicators) >= 10

    def test_insufficient_data_handling(self, engine, insufficient_data):
        """データ不足時の処理テスト - 要件2.4"""
        with pytest.raises(ValueError, match="データが不足しています"):
            engine.calculate_all_indicators(insufficient_data)

    def test_empty_dataframe_handling(self, engine):
        """空のDataFrameの処理テスト"""
        empty_df = pd.DataFrame()

        with pytest.raises(ValueError, match="入力データが空です"):
            engine.calculate_all_indicators(empty_df)

    def test_missing_columns_handling(self, engine):
        """必須カラム不足時の処理テスト"""
        invalid_df = pd.DataFrame(
            {
                "open": [100, 101, 102],
                "high": [101, 102, 103],
                # 'low'と'close'が不足
            }
        )

        with pytest.raises(ValueError, match="必須カラムが不足しています"):
            engine.calculate_all_indicators(invalid_df)

    def test_data_quality_handling_nan_values(self, engine, sample_ohlcv_data):
        """NaN値を含むデータの品質管理テスト - 要件2.3"""
        # 一部にNaN値を挿入
        corrupted_data = sample_ohlcv_data.copy()
        corrupted_data.loc[corrupted_data.index[10], "close"] = np.nan
        corrupted_data.loc[corrupted_data.index[20], "high"] = np.nan

        result = engine.calculate_all_indicators(corrupted_data)

        # 結果が返されることを確認（エラーにならない）
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(corrupted_data)

    def test_data_synchronization(self, engine, sample_ohlcv_data):
        """データポイント整合性の確保テスト - 要件2.5"""
        result = engine.calculate_all_indicators(sample_ohlcv_data)

        # 全ての指標が同じインデックスを持つ
        assert result.index.equals(sample_ohlcv_data.index)

        # 全ての指標が同じ長さ
        for column in result.columns:
            assert len(result[column]) == len(sample_ohlcv_data)

    def test_validate_data_sufficiency(self, engine, sample_ohlcv_data, insufficient_data):
        """データ十分性検証テスト - 要件2.2"""
        # 十分なデータの場合
        assert engine.validate_data_sufficiency(sample_ohlcv_data, min_periods=50) == True

        # 不十分なデータの場合
        assert engine.validate_data_sufficiency(insufficient_data, min_periods=50) == False

    def test_outlier_handling(self, engine, sample_ohlcv_data):
        """外れ値処理テスト - 要件2.3"""
        # 外れ値を挿入
        outlier_data = sample_ohlcv_data.copy()
        outlier_data.loc[outlier_data.index[50], "close"] = 999999  # 外れ値

        result = engine.calculate_all_indicators(outlier_data)

        # 計算が完了し、極端な値が含まれていないことを確認
        assert isinstance(result, pd.DataFrame)

        # RSIは0-100の範囲内
        rsi_values = result["rsi_14"].dropna()
        if len(rsi_values) > 0:
            # 外れ値があってもRSIは正常範囲内
            assert (rsi_values >= 0).all()
            assert (rsi_values <= 100).all()
