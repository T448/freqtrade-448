"""
RealtimePredictor のユニットテスト

TDD Phase: RED - 失敗するテストを作成

Requirements:
- 3.3: 訓練済みモデルによるバイナリ分類予測（0または1）
- 3.6: モデル再訓練とデプロイ前のパフォーマンス検証
- 予測エラー時の適切な処理機能
- 予測結果のログ記録機能
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import Mock, patch, MagicMock
import lightgbm as lgb

from user_data.strategies.utils.realtime_predictor import RealtimePredictor


class TestRealtimePredictor:
    """RealtimePredictor のテストクラス"""

    def setup_method(self):
        """各テストメソッドの前に実行される初期化"""
        self.predictor = RealtimePredictor()

        # サンプル特徴量データ作成
        self.sample_features = pd.DataFrame(
            {
                "sma_14": [100.0, 101.0, 102.0],
                "rsi_14": [50.0, 55.0, 60.0],
                "macd": [0.5, 0.6, 0.7],
                "bb_upper": [105.0, 106.0, 107.0],
                "atr_14": [2.0, 2.1, 2.2],
            }
        )

        # モックモデル作成
        self.mock_model = Mock(spec=lgb.Booster)
        self.mock_model.predict.return_value = np.array([0.7, 0.3, 0.8])

    def test_initialization(self):
        """初期化テスト - これは失敗するはず"""
        assert isinstance(self.predictor, RealtimePredictor)
        assert hasattr(self.predictor, "predict_binary")
        assert hasattr(self.predictor, "load_model")

    def test_predict_binary_classification_success(self):
        """バイナリ分類予測成功テスト - 要件 3.3"""
        predictions = self.predictor.predict_binary(self.mock_model, self.sample_features)

        # 期待される結果：閾値0.5を超える場合は1、以下は0
        expected = [1, 0, 1]  # [0.7, 0.3, 0.8]

        assert isinstance(predictions, list)
        assert len(predictions) == len(self.sample_features)
        assert predictions == expected

    def test_predict_binary_with_custom_threshold(self):
        """カスタム閾値を使用したバイナリ予測テスト"""
        predictions = self.predictor.predict_binary(
            self.mock_model, self.sample_features, threshold=0.6
        )

        # 閾値0.6での期待される結果
        expected = [1, 0, 1]  # [0.7, 0.3, 0.8]

        assert predictions == expected

    def test_predict_with_probability_output(self):
        """確率出力付き予測テスト"""
        predictions, probabilities = self.predictor.predict_with_probabilities(
            self.mock_model, self.sample_features
        )

        assert isinstance(predictions, list)
        assert isinstance(probabilities, list)
        assert len(predictions) == len(probabilities)
        assert all(0 <= p <= 1 for p in probabilities)

    def test_model_loading_success(self):
        """モデル読み込み成功テスト"""
        model_path = "/tmp/test_model.txt"

        with patch("os.path.exists") as mock_exists:
            mock_exists.return_value = True
            with patch("lightgbm.Booster") as mock_booster:
                mock_instance = Mock()
                mock_booster.return_value = mock_instance

                model = self.predictor.load_model(model_path)

                assert model is not None
                mock_booster.assert_called_once_with(model_file=model_path)

    def test_model_loading_failure(self):
        """モデル読み込み失敗テスト"""
        invalid_path = "/nonexistent/model.txt"

        with pytest.raises(FileNotFoundError, match="モデルファイルが見つかりません"):
            self.predictor.load_model(invalid_path)

    def test_prediction_error_handling(self):
        """予測エラー時の適切な処理テスト - 要件 3.6"""
        # エラーを発生させるモックモデル
        error_model = Mock()
        error_model.predict.side_effect = RuntimeError("予測エラー")

        with pytest.raises(RuntimeError, match="予測実行中にエラーが発生"):
            self.predictor.predict_binary(error_model, self.sample_features)

    def test_invalid_features_handling(self):
        """無効な特徴量データの処理テスト"""
        # 空のDataFrame
        empty_features = pd.DataFrame()

        with pytest.raises(ValueError, match="特徴量データが空です"):
            self.predictor.predict_binary(self.mock_model, empty_features)

        # NaNを含むデータ
        nan_features = pd.DataFrame({"feature1": [1.0, np.nan, 3.0], "feature2": [4.0, 5.0, 6.0]})

        with pytest.raises(ValueError, match="特徴量データに欠損値が含まれています"):
            self.predictor.predict_binary(self.mock_model, nan_features)

    def test_prediction_logging(self):
        """予測結果のログ記録テスト - 要件 3.6"""
        with patch("user_data.strategies.utils.realtime_predictor.logger") as mock_logger:
            self.predictor.predict_binary(self.mock_model, self.sample_features)

            # ログ出力が呼ばれることを確認
            mock_logger.info.assert_called()

    def test_model_performance_validation_success(self):
        """モデル性能検証成功テスト - 要件 3.6"""
        # 検証用データ
        X_val = self.sample_features
        y_val = pd.Series([1, 0, 1])

        validation_result = self.predictor.validate_model_performance(self.mock_model, X_val, y_val)

        assert isinstance(validation_result, dict)
        assert "accuracy" in validation_result
        assert "precision" in validation_result
        assert "recall" in validation_result
        assert "f1_score" in validation_result

    def test_model_performance_validation_failure(self):
        """モデル性能検証失敗テスト"""
        # 性能が悪いモックモデル
        bad_model = Mock()
        bad_model.predict.return_value = np.array([0.1, 0.1, 0.1])  # すべて低い確率

        X_val = self.sample_features
        y_val = pd.Series([1, 1, 1])  # すべて正例

        validation_result = self.predictor.validate_model_performance(
            bad_model, X_val, y_val, min_accuracy=0.8
        )

        assert validation_result["is_valid"] is False

    def test_model_retraining_trigger(self):
        """モデル再訓練トリガーテスト - 要件 3.6"""
        performance_metrics = {
            "accuracy": 0.6,  # 低い精度
            "precision": 0.5,
            "recall": 0.4,
        }

        should_retrain = self.predictor.should_retrain_model(performance_metrics, min_accuracy=0.8)

        assert should_retrain is True

    def test_prediction_batch_processing(self):
        """バッチ予測処理テスト"""
        # 大きなデータセット
        large_features = pd.DataFrame(
            {
                "sma_14": np.random.normal(100, 10, 1000),
                "rsi_14": np.random.uniform(0, 100, 1000),
                "macd": np.random.normal(0, 2, 1000),
            }
        )

        # バッチサイズ指定での予測
        mock_model_large = Mock()
        # side_effectでバッチサイズに応じた確率を返す
        mock_model_large.predict.side_effect = lambda x: np.random.random(len(x))

        predictions = self.predictor.predict_binary_batch(
            mock_model_large, large_features, batch_size=100
        )

        assert len(predictions) == 1000
        assert all(p in [0, 1] for p in predictions)

    def test_prediction_confidence_calculation(self):
        """予測信頼度計算テスト"""
        probabilities = [0.9, 0.3, 0.7, 0.1, 0.8]

        confidence_scores = self.predictor.calculate_prediction_confidence(probabilities)

        assert len(confidence_scores) == len(probabilities)
        assert all(0 <= score <= 1 for score in confidence_scores)

    def test_model_version_tracking(self):
        """モデルバージョン追跡テスト"""
        model_path = "/tmp/model_v1.0.txt"

        with patch.object(self.predictor, "load_model") as mock_load:
            mock_model = Mock()
            mock_load.return_value = mock_model

            loaded_model = self.predictor.load_model_with_version(model_path, version="1.0")

            assert loaded_model is not None
            mock_load.assert_called_once_with(model_path)

    def test_prediction_result_formatting(self):
        """予測結果フォーマットテスト"""
        predictions = [1, 0, 1]
        probabilities = [0.7, 0.3, 0.8]

        formatted_result = self.predictor.format_prediction_result(
            predictions, probabilities, timestamps=pd.date_range("2023-01-01", periods=3, freq="1h")
        )

        assert isinstance(formatted_result, pd.DataFrame)
        assert "prediction" in formatted_result.columns
        assert "probability" in formatted_result.columns
        assert "timestamp" in formatted_result.columns

    def test_concurrent_prediction_safety(self):
        """並行予測処理の安全性テスト"""
        import threading

        results = []

        def predict_worker():
            try:
                prediction = self.predictor.predict_binary(self.mock_model, self.sample_features)
                results.append(prediction)
            except Exception as e:
                results.append(e)

        # 複数スレッドで同時予測
        threads = [threading.Thread(target=predict_worker) for _ in range(5)]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # すべての予測が成功することを確認
        assert len(results) == 5
        assert all(isinstance(result, list) for result in results)
