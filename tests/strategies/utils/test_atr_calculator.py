"""
ATR計算機能のテストケース

Requirements tested:
- 1.1: 市場データからATR（Average True Range）を計算する機能
- 1.2: 設定可能な期間パラメータ（デフォルト14期間）でATR値を算出
- 1.3: ATR乗数（デフォルト0.5）を使用した指値価格計算機能
- 1.4: データ不足時の適切なエラーハンドリングと期間スキップ機能
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch
import sys
import os

# Add user_data/strategies to path for importing ATRCalculator
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../user_data/strategies"))

from utils.atr_calculator import ATRCalculator


class TestATRCalculator:
    def setup_method(self):
        """各テストメソッドの前に実行される設定"""
        self.calculator = ATRCalculator()

        # サンプル市場データの作成
        self.sample_data = pd.DataFrame(
            {
                "high": [
                    100,
                    105,
                    102,
                    108,
                    110,
                    107,
                    115,
                    112,
                    118,
                    120,
                    125,
                    122,
                    128,
                    130,
                    135,
                    132,
                    138,
                    140,
                ],
                "low": [
                    95,
                    98,
                    97,
                    102,
                    105,
                    102,
                    108,
                    107,
                    112,
                    115,
                    120,
                    117,
                    123,
                    125,
                    130,
                    127,
                    133,
                    135,
                ],
                "close": [
                    98,
                    102,
                    100,
                    106,
                    108,
                    105,
                    112,
                    110,
                    116,
                    118,
                    123,
                    120,
                    126,
                    128,
                    133,
                    130,
                    136,
                    138,
                ],
            }
        )

    def test_atr_calculation_default_period(self):
        """GREENテスト: デフォルト14期間でのATR計算が成功することを確認"""
        result = self.calculator.calculate_atr(self.sample_data)

        # ATRが計算されていることを確認
        assert isinstance(result, pd.Series)
        assert len(result) == len(self.sample_data)
        assert result.name == "atr"

        # TALibのATRでは最初の period 行がNaN
        assert pd.isna(result.iloc[:14]).all()

        # 15行目（0-indexed で14）以降はATR値が計算されている
        assert not pd.isna(result.iloc[14:]).any()

    def test_atr_calculation_custom_period(self):
        """GREENテスト: カスタム期間でのATR計算が成功することを確認"""
        result = self.calculator.calculate_atr(self.sample_data, period=10)

        # 10期間ATRが計算されていることを確認
        assert isinstance(result, pd.Series)
        assert len(result) == len(self.sample_data)
        assert result.name == "atr"

        # TALibのATRでは最初の period 行がNaN
        assert pd.isna(result.iloc[:10]).all()

        # 11行目（0-indexed で10）以降はATR値が計算されている
        assert not pd.isna(result.iloc[10:]).any()

    def test_atr_insufficient_data_handling(self):
        """GREENテスト: データ不足時の適切なエラーハンドリング"""
        insufficient_data = self.sample_data.head(5)  # 5行のみ

        with pytest.raises(ValueError, match="データが不足しています"):
            result = self.calculator.calculate_atr(insufficient_data, period=14)

    def test_limit_price_calculation_buy(self):
        """GREENテスト: 買い指値価格計算が成功することを確認"""
        close_price = 100.0
        atr_value = 2.0

        result = self.calculator.calculate_limit_price(
            close_price, atr_value, "buy", multiplier=0.5
        )

        # 買い指値価格 = close - (ATR * 0.5) = 100 - (2 * 0.5) = 99.0
        expected_price = 99.0
        assert result == expected_price

    def test_limit_price_calculation_sell(self):
        """GREENテスト: 売り指値価格計算が成功することを確認"""
        close_price = 100.0
        atr_value = 2.0

        result = self.calculator.calculate_limit_price(
            close_price, atr_value, "sell", multiplier=0.5
        )

        # 売り指値価格 = close + (ATR * 0.5) = 100 + (2 * 0.5) = 101.0
        expected_price = 101.0
        assert result == expected_price

    def test_atr_calculation_with_parameters(self):
        """GREENテスト: パラメータ設定可能なATR計算が成功することを確認"""
        calculator = ATRCalculator(atr_period=10, atr_multiplier=0.7)
        result = calculator.calculate_atr(self.sample_data)

        # パラメータが正しく設定されていることを確認
        assert calculator.atr_period == 10
        assert calculator.atr_multiplier == 0.7

        # ATRが正しく計算されていることを確認
        assert isinstance(result, pd.Series)
        assert len(result) == len(self.sample_data)

    def test_empty_dataframe_handling(self):
        """GREENテスト: 空のDataFrameの適切な処理"""
        empty_df = pd.DataFrame()

        with pytest.raises(ValueError, match="入力データが空です"):
            result = self.calculator.calculate_atr(empty_df)

    def test_invalid_column_handling(self):
        """GREENテスト: 無効な列名の適切な処理"""
        invalid_data = pd.DataFrame(
            {"invalid_high": [100, 105], "invalid_low": [95, 98], "invalid_close": [98, 102]}
        )

        with pytest.raises(ValueError, match="必須カラムが不足しています"):
            result = self.calculator.calculate_atr(invalid_data)


if __name__ == "__main__":
    pytest.main([__file__])
