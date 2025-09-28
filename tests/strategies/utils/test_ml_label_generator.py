"""
MLLabelGenerator のユニットテスト

TDD Phase: RED - 失敗するテストを作成

Requirements:
- 3.1: ATRリターンに基づくバイナリラベル生成（正のリターン=1、負・ゼロ=0）
- 3.4: 十分な訓練データの存在確認機能
- 訓練データの適切な前処理機能
- ラベル品質検証機能
"""

import numpy as np
import pandas as pd
import pytest

from user_data.strategies.utils.ml_label_generator import MLLabelGenerator


class TestMLLabelGenerator:
    """MLLabelGenerator のテストクラス"""

    def setup_method(self):
        """各テストメソッドの前に実行される初期化"""
        self.generator = MLLabelGenerator()

        # 有効なOHLCVデータ作成
        self.sample_ohlcv_data = pd.DataFrame(
            {
                "open": [100.0, 101.0, 102.0, 103.0, 104.0] * 20,  # 100行
                "high": [101.0, 102.0, 103.0, 104.0, 105.0] * 20,
                "low": [99.0, 100.0, 101.0, 102.0, 103.0] * 20,
                "close": [100.5, 101.5, 102.5, 103.5, 104.5] * 20,
                "volume": [1000.0, 1100.0, 1200.0, 1300.0, 1400.0] * 20,
            },
            index=pd.date_range("2023-01-01", periods=100, freq="1h"),
        )

        # ATRリターンデータ作成（正負の値を含む）
        np.random.seed(42)
        self.atr_returns = pd.Series(
            [
                0.02,
                -0.01,
                0.05,
                0.0,
                -0.03,  # 1, 0, 1, 0, 0
                0.01,
                0.07,
                -0.02,
                0.03,
                -0.01,  # 1, 1, 0, 1, 0
            ]
            * 10,
            index=self.sample_ohlcv_data.index,
            name="atr_returns",
        )

    def test_initialization(self):
        """初期化テスト - これは失敗するはず"""
        assert isinstance(self.generator, MLLabelGenerator)

    def test_generate_binary_labels_from_atr_returns(self):
        """ATRリターンからバイナリラベル生成 - 要件 3.1"""
        labels = self.generator.generate_binary_labels_from_atr_returns(self.atr_returns)

        # 期待する結果：正のリターン=1、負・ゼロ=0
        expected_labels = [1, 0, 1, 0, 0, 1, 1, 0, 1, 0] * 10

        assert isinstance(labels, pd.Series)
        assert len(labels) == len(self.atr_returns)
        assert labels.tolist() == expected_labels

    def test_generate_binary_labels_edge_cases(self):
        """エッジケースのテスト（NaN値、極端な値）"""
        edge_case_returns = pd.Series(
            [
                0.0,  # ゼロ -> 0
                np.nan,  # NaN -> 適切に処理
                0.001,  # 極小正数 -> 1
                -0.001,  # 極小負数 -> 0
                np.inf,  # 無限大 -> 適切に処理
                -np.inf,  # 負の無限大 -> 適切に処理
            ]
        )

        labels = self.generator.generate_binary_labels_from_atr_returns(edge_case_returns)

        assert isinstance(labels, pd.Series)
        assert not labels.isnull().all()  # すべてNaNではない

    def test_validate_training_data_sufficiency(self):
        """十分な訓練データの存在確認 - 要件 3.4"""
        sufficient_data = self.sample_ohlcv_data
        insufficient_data = self.sample_ohlcv_data.iloc[:10]

        # 十分なデータの場合
        result = self.generator.validate_training_data_sufficiency(sufficient_data, min_samples=50)
        assert result is True

        # 不十分なデータの場合
        result = self.generator.validate_training_data_sufficiency(
            insufficient_data, min_samples=50
        )
        assert result is False

    def test_preprocess_training_data(self):
        """訓練データの適切な前処理 - 要件 3.4"""
        # 特徴量データとラベルデータを準備
        features = pd.DataFrame(
            {
                "sma_14": [100.0] * 100,
                "rsi_14": [50.0] * 100,
                "macd": [0.5] * 100,
            },
            index=self.sample_ohlcv_data.index,
        )

        labels = self.generator.generate_binary_labels_from_atr_returns(self.atr_returns)

        processed_features, processed_labels = self.generator.preprocess_training_data(
            features, labels
        )

        assert isinstance(processed_features, pd.DataFrame)
        assert isinstance(processed_labels, pd.Series)
        assert len(processed_features) == len(processed_labels)

    def test_validate_label_quality(self):
        """ラベル品質検証 - 要件 3.4"""
        # 良質なラベル（適度なバランス）
        balanced_labels = pd.Series([0, 1, 0, 1, 0, 1] * 10)
        result = self.generator.validate_label_quality(balanced_labels)
        assert result["is_valid"] == True

        # 不均衡なラベル（すべて0）
        imbalanced_labels = pd.Series([0] * 60)
        result = self.generator.validate_label_quality(imbalanced_labels)
        assert result["is_valid"] == False

    def test_create_training_dataset(self):
        """完全な訓練データセット作成"""
        features = pd.DataFrame(
            {
                "sma_14": np.random.normal(100, 10, 100),
                "rsi_14": np.random.uniform(0, 100, 100),
                "macd": np.random.normal(0, 2, 100),
            },
            index=self.sample_ohlcv_data.index,
        )

        dataset = self.generator.create_training_dataset(features, self.atr_returns)

        assert "features" in dataset
        assert "labels" in dataset
        assert "metadata" in dataset
        assert isinstance(dataset["features"], pd.DataFrame)
        assert isinstance(dataset["labels"], pd.Series)

    def test_get_label_distribution_report(self):
        """ラベル分布レポート生成"""
        labels = self.generator.generate_binary_labels_from_atr_returns(self.atr_returns)
        report = self.generator.get_label_distribution_report(labels)

        assert "total_samples" in report
        assert "positive_ratio" in report
        assert "negative_ratio" in report
        assert "balance_score" in report

    def test_handle_insufficient_data_error(self):
        """データ不足時のエラーハンドリング"""
        insufficient_data = self.sample_ohlcv_data.iloc[:5]  # 5行のみ

        with pytest.raises(ValueError, match="訓練データが不足"):
            self.generator.validate_training_data_sufficiency(insufficient_data, min_samples=50)

    def test_handle_invalid_atr_returns(self):
        """無効なATRリターンデータの処理"""
        # 空のSeries
        empty_returns = pd.Series([], dtype=float)

        with pytest.raises(ValueError, match="ATRリターンデータが空"):
            self.generator.generate_binary_labels_from_atr_returns(empty_returns)

    def test_label_consistency_check(self):
        """ラベルの整合性チェック"""
        # 一定のリターンパターンでラベルの一貫性を確認
        consistent_returns = pd.Series([0.01, 0.02, 0.03, -0.01, -0.02, 0.0])
        labels = self.generator.generate_binary_labels_from_atr_returns(consistent_returns)

        # 期待される結果: [1, 1, 1, 0, 0, 0]
        expected = [1, 1, 1, 0, 0, 0]
        assert labels.tolist() == expected
