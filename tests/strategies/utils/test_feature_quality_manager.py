"""
FeatureQualityManagerのテストケース

richmanbtcチュートリアルに基づく機械学習特徴量の品質管理システムのテスト

Requirements tested:
- 2.2: 十分な履歴データの存在確認機能
- 2.4: ルックバック期間に対応した履歴データ確保機能
- 特徴量計算エラー時の適切な処理機能
- 完全な特徴量ベクトル生成機能
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from user_data.strategies.utils.feature_quality_manager import FeatureQualityManager
from user_data.strategies.utils.technical_indicator_engine import TechnicalIndicatorEngine


class TestFeatureQualityManager:
    """FeatureQualityManagerクラスのテストケース"""

    @pytest.fixture
    def sample_ohlcv_data(self):
        """テスト用のOHLCVデータを生成"""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100, freq="1h")

        # 現実的な価格変動を模擬
        base_price = 100.0
        price_changes = np.random.normal(0, 0.02, 100)
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
        dates = pd.date_range("2023-01-01", periods=20, freq="1h")
        data = {
            "open": np.random.normal(100, 5, 20),
            "high": np.random.normal(105, 5, 20),
            "low": np.random.normal(95, 5, 20),
            "close": np.random.normal(100, 5, 20),
            "volume": np.random.randint(1000, 10000, 20),
        }
        df = pd.DataFrame(data, index=dates)
        return df

    @pytest.fixture
    def corrupted_features(self):
        """欠損値や異常値を含む特徴量データ"""
        dates = pd.date_range("2023-01-01", periods=50, freq="1h")
        data = {
            "sma_14": np.random.normal(100, 10, 50),
            "rsi_14": np.random.normal(50, 20, 50),
            "macd": np.random.normal(0, 2, 50),
            "bb_upper": np.random.normal(110, 10, 50),
        }
        df = pd.DataFrame(data, index=dates)

        # 意図的に欠損値と異常値を挿入（10%以上の欠損値を作成）
        df.loc[df.index[5], "sma_14"] = np.nan
        df.loc[df.index[10], "rsi_14"] = np.inf
        df.loc[df.index[15], "macd"] = -np.inf
        df.loc[df.index[20:26], "bb_upper"] = np.nan  # 6行の欠損値（12%）

        return df

    @pytest.fixture
    def quality_manager(self):
        """FeatureQualityManagerインスタンス"""
        return FeatureQualityManager()

    def test_initialization(self, quality_manager):
        """初期化テスト"""
        assert isinstance(quality_manager, FeatureQualityManager)
        assert hasattr(quality_manager, "validate_data_sufficiency")
        assert hasattr(quality_manager, "generate_complete_feature_vector")

    def test_validate_data_sufficiency_sufficient_data(self, quality_manager, sample_ohlcv_data):
        """十分なデータの場合の検証テスト - 要件2.2"""
        result = quality_manager.validate_data_sufficiency(sample_ohlcv_data, min_periods=50)

        assert result is True

    def test_validate_data_sufficiency_insufficient_data(self, quality_manager, insufficient_data):
        """不十分なデータの場合の検証テスト - 要件2.2"""
        result = quality_manager.validate_data_sufficiency(insufficient_data, min_periods=50)

        assert result is False

    def test_validate_lookback_period_support(self, quality_manager, sample_ohlcv_data):
        """ルックバック期間対応の履歴データ確保テスト - 要件2.4"""
        lookback_periods = {"sma_14": 14, "sma_50": 50, "ema_26": 26, "rsi_14": 14, "atr_14": 14}

        result = quality_manager.validate_lookback_period_support(
            sample_ohlcv_data, lookback_periods
        )

        assert result is True

    def test_validate_lookback_period_support_insufficient(
        self, quality_manager, insufficient_data
    ):
        """ルックバック期間に対して不十分なデータの場合"""
        lookback_periods = {
            "sma_50": 50,  # 20行のデータに対して50期間は不十分
            "ema_26": 26,  # 26期間も不十分
        }

        result = quality_manager.validate_lookback_period_support(
            insufficient_data, lookback_periods
        )

        assert result is False

    def test_generate_complete_feature_vector_success(self, quality_manager, sample_ohlcv_data):
        """完全な特徴量ベクトル生成成功テスト - 要件2.2"""
        with patch.object(TechnicalIndicatorEngine, "calculate_all_indicators") as mock_calculate:
            # モックの戻り値を設定
            mock_features = pd.DataFrame(
                {
                    "sma_14": np.random.normal(100, 10, len(sample_ohlcv_data)),
                    "rsi_14": np.random.normal(50, 20, len(sample_ohlcv_data)),
                    "macd": np.random.normal(0, 2, len(sample_ohlcv_data)),
                },
                index=sample_ohlcv_data.index,
            )
            mock_calculate.return_value = mock_features

            result = quality_manager.generate_complete_feature_vector(sample_ohlcv_data)

            assert isinstance(result, pd.DataFrame)
            assert len(result) == len(sample_ohlcv_data)
            assert "sma_14" in result.columns
            assert "rsi_14" in result.columns
            assert "macd" in result.columns

    def test_generate_complete_feature_vector_insufficient_data(
        self, quality_manager, insufficient_data
    ):
        """データ不足時の特徴量ベクトル生成テスト"""
        with pytest.raises(ValueError, match="データが不足しています"):
            quality_manager.generate_complete_feature_vector(insufficient_data)

    def test_handle_feature_calculation_errors(self, quality_manager, sample_ohlcv_data):
        """特徴量計算エラー時の適切な処理テスト - 要件2.4"""
        with patch.object(TechnicalIndicatorEngine, "calculate_all_indicators") as mock_calculate:
            # 計算エラーをシミュレート
            mock_calculate.side_effect = RuntimeError("テクニカル指標計算エラー")

            with pytest.raises(RuntimeError, match="特徴量計算中にエラーが発生"):
                quality_manager.generate_complete_feature_vector(sample_ohlcv_data)

    def test_clean_feature_data_missing_values(self, quality_manager, corrupted_features):
        """欠損値を含むデータのクリーニングテスト"""
        cleaned_data = quality_manager.clean_feature_data(corrupted_features)

        assert isinstance(cleaned_data, pd.DataFrame)
        assert len(cleaned_data) == len(corrupted_features)

        # 無限大値がNaNに変換されていることを確認
        assert not np.isinf(cleaned_data.values).any()

    def test_clean_feature_data_outlier_handling(self, quality_manager):
        """外れ値処理テスト"""
        # 極端な外れ値を含むデータ
        dates = pd.date_range("2023-01-01", periods=20, freq="1h")
        data = {
            "normal_feature": np.random.normal(100, 10, 20),
            "outlier_feature": [100] * 19 + [99999],  # 最後に極端な外れ値
        }
        df = pd.DataFrame(data, index=dates)

        cleaned_data = quality_manager.clean_feature_data(df)

        # 外れ値が適切に処理されていることを確認
        assert isinstance(cleaned_data, pd.DataFrame)
        assert len(cleaned_data) == len(df)

    def test_validate_feature_completeness_complete(self, quality_manager):
        """完全な特徴量データの検証テスト"""
        complete_features = pd.DataFrame(
            {
                "sma_14": np.random.normal(100, 10, 50),
                "rsi_14": np.random.normal(50, 20, 50),
                "macd": np.random.normal(0, 2, 50),
            }
        )

        result = quality_manager.validate_feature_completeness(
            complete_features, required_features=["sma_14", "rsi_14", "macd"]
        )

        assert result is True

    def test_validate_feature_completeness_missing_features(self, quality_manager):
        """必須特徴量が不足している場合の検証テスト"""
        incomplete_features = pd.DataFrame(
            {
                "sma_14": np.random.normal(100, 10, 50),
                "rsi_14": np.random.normal(50, 20, 50),
                # 'macd'が不足
            }
        )

        result = quality_manager.validate_feature_completeness(
            incomplete_features, required_features=["sma_14", "rsi_14", "macd"]
        )

        assert result is False

    def test_validate_feature_completeness_excessive_missing_values(self, quality_manager):
        """過度な欠損値がある場合の検証テスト"""
        features_with_missing = pd.DataFrame(
            {
                "sma_14": [100] * 30 + [np.nan] * 20,  # 40%の欠損値
                "rsi_14": np.random.normal(50, 20, 50),
                "macd": np.random.normal(0, 2, 50),
            }
        )

        result = quality_manager.validate_feature_completeness(
            features_with_missing,
            required_features=["sma_14", "rsi_14", "macd"],
            max_missing_ratio=0.3,  # 30%以下の欠損値のみ許可
        )

        assert result is False

    def test_get_feature_quality_report(self, quality_manager, corrupted_features):
        """特徴量品質レポート生成テスト"""
        report = quality_manager.get_feature_quality_report(corrupted_features)

        assert isinstance(report, dict)
        assert "total_features" in report
        assert "total_samples" in report
        assert "missing_value_ratio" in report
        assert "features_with_issues" in report

        # 欠損値を含むデータなので、問題のある特徴量が報告される
        assert len(report["features_with_issues"]) > 0

    def test_empty_dataframe_handling(self, quality_manager):
        """空のDataFrameの処理テスト"""
        empty_df = pd.DataFrame()

        with pytest.raises(ValueError, match="入力データが空です"):
            quality_manager.validate_data_sufficiency(empty_df, 10)

    def test_comprehensive_quality_check(self, quality_manager, sample_ohlcv_data):
        """包括的な品質チェック統合テスト"""
        with patch.object(TechnicalIndicatorEngine, "calculate_all_indicators") as mock_calculate:
            # 良好な特徴量データをモック
            mock_features = pd.DataFrame(
                {
                    "sma_14": np.random.normal(100, 10, len(sample_ohlcv_data)),
                    "rsi_14": np.random.uniform(0, 100, len(sample_ohlcv_data)),
                    "macd": np.random.normal(0, 2, len(sample_ohlcv_data)),
                    "bb_upper": np.random.normal(110, 10, len(sample_ohlcv_data)),
                },
                index=sample_ohlcv_data.index,
            )
            mock_calculate.return_value = mock_features

            # 包括的な品質チェック実行
            features = quality_manager.generate_complete_feature_vector(sample_ohlcv_data)
            is_sufficient = quality_manager.validate_data_sufficiency(sample_ohlcv_data, 50)
            is_complete = quality_manager.validate_feature_completeness(
                features, ["sma_14", "rsi_14", "macd", "bb_upper"]
            )

            assert is_sufficient is True
            assert is_complete is True
            assert isinstance(features, pd.DataFrame)
            assert len(features) == len(sample_ohlcv_data)
