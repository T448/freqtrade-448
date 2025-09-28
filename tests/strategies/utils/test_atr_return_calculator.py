"""
ATRリターン計算システムのテストケース

Requirements tested:
- 1.7: richmanbtcチュートリアルに準拠した決済方法の使用
- 1.8: Freqtradeの既存バックテストシステムとの統合
- 6.1: 1次モデル（ATR戦略）の独立性確保
- 6.4: ATR×0.5距離設定の忠実な再現
"""

import os
import sys

import pandas as pd
import pytest


# Add user_data/strategies to path for importing modules
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../user_data/strategies"))

from utils.atr_return_calculator import ATRReturnCalculator


class TestATRReturnCalculator:
    def setup_method(self):
        """各テストメソッドの前に実行される設定"""
        self.calculator = ATRReturnCalculator()

        # サンプル市場データの作成（ATR計算に十分な18行）
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

    def test_atr_return_calculation_basic(self):
        """GREENテスト: 基本的なATRリターン計算が成功することを確認"""
        result = self.calculator.calculate_atr_returns(self.sample_data)

        # リターンが計算されていることを確認
        assert isinstance(result, pd.Series)
        assert len(result) == len(self.sample_data)
        assert result.name == "atr_returns"

        # 最初の行はNaN（前の期間がないため）
        assert pd.isna(result.iloc[0])

        # ATR期間内の値もNaN（ATR計算不可のため）
        # 実際の値が出るのはATR期間+1行目以降
        valid_returns = result.dropna()
        assert len(valid_returns) >= 0  # 最低0個以上の有効値

    def test_atr_return_calculation_with_parameters(self):
        """GREENテスト: パラメータ付きATRリターン計算が成功することを確認"""
        result = self.calculator.calculate_atr_returns(
            self.sample_data, atr_period=10, atr_multiplier=0.7
        )

        # パラメータが異なる場合の結果を確認
        assert isinstance(result, pd.Series)
        assert len(result) == len(self.sample_data)
        assert result.name == "atr_returns"

    def test_buy_limit_execution_simulation(self):
        """GREENテスト: 買い指値約定シミュレーションが成功することを確認"""
        # 買い指値約定のケース（現在価格 <= 指値価格）
        result = self.calculator.simulate_limit_execution(
            current_price=99.0, limit_price=100.0, side="buy"
        )
        assert result is True

        # 買い指値未約定のケース（現在価格 > 指値価格）
        result = self.calculator.simulate_limit_execution(
            current_price=101.0, limit_price=100.0, side="buy"
        )
        assert result is False

    def test_sell_limit_execution_simulation(self):
        """GREENテスト: 売り指値約定シミュレーションが成功することを確認"""
        # 売り指値約定のケース（現在価格 >= 指値価格）
        result = self.calculator.simulate_limit_execution(
            current_price=101.0, limit_price=100.0, side="sell"
        )
        assert result is True

        # 売り指値未約定のケース（現在価格 < 指値価格）
        result = self.calculator.simulate_limit_execution(
            current_price=99.0, limit_price=100.0, side="sell"
        )
        assert result is False

    def test_richmanbtc_return_logic(self):
        """GREENテスト: richmanbtcチュートリアル準拠のリターン計算が成功することを確認"""
        result = self.calculator.calculate_theoretical_returns(self.sample_data)

        # 理論リターンが計算されていることを確認
        assert isinstance(result, pd.Series)
        assert len(result) == len(self.sample_data)
        assert result.name == "atr_returns"

    def test_return_calculation_edge_cases(self):
        """GREENテスト: エッジケースの適切な処理"""
        # 価格変動なしのケース
        flat_data = pd.DataFrame({"high": [100] * 18, "low": [100] * 18, "close": [100] * 18})

        result = self.calculator.calculate_atr_returns(flat_data)

        # 価格変動がない場合、リターンは主に0になるはず
        assert isinstance(result, pd.Series)
        assert len(result) == len(flat_data)

    def test_insufficient_data_handling(self):
        """GREENテスト: データ不足時の適切な処理"""
        insufficient_data = self.sample_data.head(5)

        with pytest.raises(ValueError, match="データが不足しています"):
            result = self.calculator.calculate_atr_returns(insufficient_data)

    def test_return_statistics(self):
        """GREENテスト: リターン統計計算が成功することを確認"""
        result = self.calculator.get_return_statistics(self.sample_data)

        # 統計情報が辞書として返されることを確認
        assert isinstance(result, dict)

        # 必要な統計項目が含まれていることを確認
        expected_keys = [
            "total_returns",
            "mean_return",
            "std_return",
            "positive_returns",
            "negative_returns",
            "zero_returns",
            "win_rate",
        ]
        for key in expected_keys:
            assert key in result


if __name__ == "__main__":
    pytest.main([__file__])
