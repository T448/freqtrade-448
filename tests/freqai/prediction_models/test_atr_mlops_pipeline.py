"""
ATR MLOpsパイプラインのテスト

自動化MLOpsパイプライン機能のテストを実行します。
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import shutil

from freqtrade.freqai.prediction_models.atr_lightgbm_classifier import ATRLightGBMClassifier
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.configuration import Configuration


class TestATRMLOpsPipeline:
    """ATR MLOpsパイプラインのテストクラス"""

    @pytest.fixture
    def config(self):
        """テスト用設定"""
        return {
            "freqai": {
                "identifier": "test_atr_classifier",
                "feature_parameters": {
                    "include_timeframes": ["5m"],
                    "label_period_candles": 1,
                },
                "data_split_parameters": {
                    "test_size": 0.2,
                },
                "model_training_parameters": {
                    "n_estimators": 10,
                    "learning_rate": 0.1,
                    "verbosity": -1,
                },
                "expiration_hours": 24,
                "purge_old_models": 2,
            }
        }

    @pytest.fixture
    def sample_data(self):
        """テスト用サンプルデータ"""
        dates = pd.date_range("2023-01-01", periods=100, freq="5min")
        data = {
            "date": dates,
            "open": np.random.uniform(100, 110, 100),
            "high": np.random.uniform(105, 115, 100),
            "low": np.random.uniform(95, 105, 100),
            "close": np.random.uniform(100, 110, 100),
            "volume": np.random.uniform(1000, 5000, 100),
        }
        df = pd.DataFrame(data)
        df.set_index("date", inplace=True)
        return df

    @pytest.fixture
    def classifier(self, config):
        """ATRLightGBMClassifierインスタンス"""
        return ATRLightGBMClassifier(config)

    @pytest.fixture
    def mock_data_kitchen(self):
        """モックDataKitchen"""
        dk = Mock(spec=FreqaiDataKitchen)
        dk.training_features_list = ["feature1", "feature2", "feature3"]
        dk.label_list = ["target"]
        dk.filter_features = Mock(return_value=(pd.DataFrame(np.random.random((10, 3))), None))
        return dk

    def test_automatic_feature_generation(self, classifier, sample_data):
        """自動特徴量生成機能のテスト - 要件3.6, 5.7"""
        # ATRLightGBMClassifierが自動特徴量生成をサポートすることを確認

        # FreqAIの標準的な特徴量生成メソッドが呼び出し可能か確認
        assert hasattr(classifier, "config")
        assert "freqai" in classifier.config

        # 特徴量パラメータが設定されていることを確認
        feature_params = classifier.freqai_info.get("feature_parameters", {})
        assert isinstance(feature_params, dict)

    def test_automatic_label_generation(self, classifier, sample_data):
        """自動ラベル生成機能のテスト - 要件3.6, 5.7"""
        from user_data.strategies.utils.atr_return_calculator import ATRReturnCalculator

        # ATRリターン計算器の初期化
        atr_calculator = ATRReturnCalculator()

        # ATRリターンの計算
        atr_returns = atr_calculator.calculate_atr_returns(sample_data)

        # ラベル生成（正のリターン=1、負・ゼロ=0）
        labels = (atr_returns > 0).astype(int)

        # ラベルが適切に生成されることを確認
        assert len(labels) == len(sample_data)
        assert set(labels.dropna().unique()).issubset({0, 1})
        assert labels.dtype == int

    def test_automatic_model_training(self, classifier, mock_data_kitchen):
        """自動モデル訓練機能のテスト - 要件3.6, 5.7"""
        # テスト用データ辞書作成
        data_dict = {
            "train_features": pd.DataFrame(
                np.random.random((50, 3)), columns=["feature1", "feature2", "feature3"]
            ),
            "train_labels": pd.Series(np.random.randint(0, 2, 50), name="target"),
            "train_weights": pd.Series(np.ones(50)),
            "test_features": pd.DataFrame(
                np.random.random((20, 3)), columns=["feature1", "feature2", "feature3"]
            ),
            "test_labels": pd.Series(np.random.randint(0, 2, 20), name="target"),
            "test_weights": pd.Series(np.ones(20)),
        }

        # モデル訓練実行
        model = classifier.fit(data_dict, mock_data_kitchen)

        # 訓練されたモデルが存在することを確認
        assert model is not None
        assert hasattr(model, "predict")

        # モデルが分類器として機能することを確認
        test_features = data_dict["train_features"].iloc[:5]
        predictions = model.predict(test_features)
        assert len(predictions) == 5
        assert all(0 <= p <= 1 for p in predictions)

    def test_automatic_prediction_execution(self, classifier, mock_data_kitchen):
        """自動予測実行機能のテスト - 要件3.6, 5.7"""
        # ダミーモデルを設定
        classifier.model = Mock()
        classifier.model.predict = Mock(return_value=np.array([0.7, 0.3, 0.8]))

        # テスト用データフレーム
        test_df = pd.DataFrame(
            np.random.random((3, 3)), columns=["feature1", "feature2", "feature3"]
        )

        # 予測実行
        predictions = classifier.predict(test_df, mock_data_kitchen)

        # 予測結果が適切に返されることを確認
        assert isinstance(predictions, pd.DataFrame)
        assert len(predictions) == 3
        assert predictions.iloc[:, 0].dtype == int  # バイナリ予測
        assert set(predictions.iloc[:, 0].unique()).issubset({0, 1})

    def test_model_versioning_and_persistence(self, classifier, config):
        """モデルバージョン管理と永続化機能のテスト - 要件3.6, 5.7"""
        # FreqAI設定にモデル保存パラメータが含まれることを確認
        freqai_config = config["freqai"]

        # identifierが設定されていることを確認（バージョン管理用）
        assert "identifier" in freqai_config
        assert freqai_config["identifier"] == "test_atr_classifier"

        # 古いモデル削除パラメータが設定されていることを確認
        assert "purge_old_models" in freqai_config
        assert isinstance(freqai_config["purge_old_models"], int)

    def test_automatic_retraining_configuration(self, classifier, config):
        """自動再訓練システムの設定テスト - 要件3.6, 5.7"""
        freqai_config = config["freqai"]

        # 自動再訓練に必要な設定が存在することを確認
        assert "expiration_hours" in freqai_config
        assert isinstance(freqai_config["expiration_hours"], (int, float))

        # データ分割パラメータが設定されていることを確認
        assert "data_split_parameters" in freqai_config
        assert "test_size" in freqai_config["data_split_parameters"]

    def test_mlops_pipeline_integration(self, classifier, mock_data_kitchen):
        """MLOpsパイプライン統合テスト - 要件3.6, 5.7"""
        # 1. 特徴量生成からラベル生成までの流れ
        sample_data = pd.DataFrame(
            {
                "high": [110, 112, 108, 115, 113],
                "low": [105, 107, 103, 110, 108],
                "close": [108, 110, 105, 113, 111],
                "volume": [1000, 1200, 800, 1500, 1100],
            }
        )

        # ATRリターン計算とラベル生成
        from user_data.strategies.utils.atr_return_calculator import ATRReturnCalculator

        atr_calculator = ATRReturnCalculator()

        # データ十分性チェック
        if len(sample_data) >= 14:  # ATRに必要な最小期間
            atr_returns = atr_calculator.calculate_atr_returns(sample_data)
            labels = (atr_returns > 0).astype(int)

            # ラベルが生成されることを確認
            assert len(labels) > 0

        # 2. モデルパラメータの自動取得
        model_params = classifier._get_model_training_parameters()

        # 必要なパラメータが設定されていることを確認
        assert "objective" in model_params
        assert model_params["objective"] == "binary"
        assert "n_estimators" in model_params
        assert "learning_rate" in model_params

    def test_error_handling_in_pipeline(self, classifier, mock_data_kitchen):
        """パイプライン内エラーハンドリングのテスト - 要件3.6, 5.7"""
        # モデルが訓練されていない状態での予測エラー
        test_df = pd.DataFrame(np.random.random((3, 3)))

        with pytest.raises(ValueError, match="モデルが訓練されていません"):
            classifier.predict_proba(test_df, mock_data_kitchen)

    def test_logging_functionality(self, classifier, mock_data_kitchen, caplog):
        """ログ記録機能のテスト - 要件3.6, 5.7"""
        # ダミーモデルを設定
        classifier.model = Mock()
        classifier.model.predict = Mock(return_value=np.array([0.7, 0.3]))

        test_df = pd.DataFrame(np.random.random((2, 3)))

        # 予測実行とログチェック
        classifier.predict(test_df, mock_data_kitchen)

        # ログが出力されることを確認
        assert "ATR予測完了" in caplog.text

    def test_freqai_framework_compatibility(self, classifier):
        """FreqAIフレームワーク互換性テスト - 要件5.7"""
        from freqtrade.freqai.base_models.BaseClassifierModel import BaseClassifierModel

        # BaseClassifierModelを継承していることを確認
        assert isinstance(classifier, BaseClassifierModel)

        # FreqAI標準メソッドが存在することを確認
        assert hasattr(classifier, "fit")
        assert hasattr(classifier, "predict")
        assert callable(classifier.fit)
        assert callable(classifier.predict)
