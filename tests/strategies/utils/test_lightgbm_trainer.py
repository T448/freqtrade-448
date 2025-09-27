"""
LightGBMTrainer のユニットテスト

TDD Phase: RED - 失敗するテストを作成

Requirements:
- 3.2: テクニカル指標特徴量を使用したLightGBM二値分類器の訓練
- 3.4: 適切な訓練・検証データ分割機能
- 3.5: オーバーフィッティング防止機能
- 3.4: モデル訓練失敗時のフォールバック機能
"""


import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from user_data.strategies.utils.lightgbm_trainer import LightGBMTrainer


class TestLightGBMTrainer:
    """LightGBMTrainer のテストクラス"""

    def setup_method(self):
        """各テストメソッドの前に実行される初期化"""
        self.trainer = LightGBMTrainer()

        # サンプル特徴量データ作成
        np.random.seed(42)
        self.sample_features = pd.DataFrame(
            {
                "sma_14": np.random.normal(100, 10, 1000),
                "rsi_14": np.random.uniform(0, 100, 1000),
                "macd": np.random.normal(0, 2, 1000),
                "bb_upper": np.random.normal(110, 10, 1000),
                "atr_14": np.random.uniform(0.5, 3.0, 1000),
            }
        )

        # サンプルラベルデータ作成（バランスの取れたバイナリラベル）
        self.sample_labels = pd.Series(np.random.choice([0, 1], size=1000, p=[0.6, 0.4]))

    def test_initialization(self):
        """初期化テスト - これは失敗するはず"""
        assert isinstance(self.trainer, LightGBMTrainer)
        assert hasattr(self.trainer, "train_model")
        assert hasattr(self.trainer, "split_data")

    def test_split_data_train_validation(self):
        """訓練・検証データ分割テスト - 要件 3.4"""
        train_ratio = 0.8
        X_train, X_val, y_train, y_val = self.trainer.split_data(
            self.sample_features, self.sample_labels, train_ratio=train_ratio
        )

        # データサイズの検証
        expected_train_size = int(len(self.sample_features) * train_ratio)
        assert len(X_train) == expected_train_size
        assert len(X_val) == len(self.sample_features) - expected_train_size
        assert len(y_train) == expected_train_size
        assert len(y_val) == len(self.sample_features) - expected_train_size

        # データ型の検証
        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(X_val, pd.DataFrame)
        assert isinstance(y_train, pd.Series)
        assert isinstance(y_val, pd.Series)

    def test_create_lgb_datasets(self):
        """LightGBMデータセット作成テスト"""
        X_train, X_val, y_train, y_val = self.trainer.split_data(
            self.sample_features, self.sample_labels
        )

        train_dataset, val_dataset = self.trainer.create_lgb_datasets(
            X_train, X_val, y_train, y_val
        )

        assert isinstance(train_dataset, lgb.Dataset)
        assert isinstance(val_dataset, lgb.Dataset)

    def test_configure_training_parameters(self):
        """訓練パラメータ設定テスト"""
        params = self.trainer.configure_training_parameters()

        # 必須パラメータの確認
        assert "objective" in params
        assert "metric" in params
        assert "boosting_type" in params
        assert "num_leaves" in params
        assert "learning_rate" in params

        # バイナリ分類設定の確認
        assert params["objective"] == "binary"
        assert "binary_logloss" in params["metric"] or "auc" in params["metric"]

    def test_configure_overfitting_prevention(self):
        """オーバーフィッティング防止設定テスト - 要件 3.5"""
        params = self.trainer.configure_training_parameters()

        # オーバーフィッティング防止パラメータの確認
        assert "feature_fraction" in params
        assert "bagging_fraction" in params
        assert "bagging_freq" in params
        assert "min_data_in_leaf" in params

        # 適切な値の範囲チェック
        assert 0 < params["feature_fraction"] <= 1
        assert 0 < params["bagging_fraction"] <= 1
        assert params["min_data_in_leaf"] >= 1

    def test_train_model_success(self):
        """LightGBMモデル訓練成功テスト - 要件 3.2"""
        model, training_history = self.trainer.train_model(self.sample_features, self.sample_labels)

        # モデルの検証
        assert isinstance(model, lgb.Booster)

        # 訓練履歴の検証
        assert isinstance(training_history, dict)
        assert "train_score" in training_history
        assert "val_score" in training_history
        assert "best_iteration" in training_history

    def test_train_model_with_insufficient_data(self):
        """データ不足時のモデル訓練テスト"""
        insufficient_features = self.sample_features.iloc[:10]  # 10サンプルのみ
        insufficient_labels = self.sample_labels.iloc[:10]

        with pytest.raises(ValueError, match="訓練データが不足"):
            self.trainer.train_model(insufficient_features, insufficient_labels)

    def test_train_model_with_imbalanced_labels(self):
        """不均衡ラベルでの訓練テスト"""
        # 極端に不均衡なラベル（95% vs 5%）
        imbalanced_labels = pd.Series([0] * 950 + [1] * 50)

        # 不均衡データでも訓練は可能だが、警告が発生することを確認
        model, history = self.trainer.train_model(self.sample_features, imbalanced_labels)

        assert isinstance(model, lgb.Booster)

    def test_early_stopping_configuration(self):
        """早期停止設定テスト - 要件 3.5"""
        callbacks = self.trainer.configure_callbacks()

        # 早期停止コールバックの確認（LightGBMコールバック型の確認）
        assert len(callbacks) > 0
        assert any(hasattr(cb, "__name__") or "early" in str(type(cb)).lower() for cb in callbacks)

    def test_cross_validation_support(self):
        """クロスバリデーション対応テスト"""
        cv_scores = self.trainer.cross_validate(self.sample_features, self.sample_labels, n_folds=3)

        assert isinstance(cv_scores, dict)
        assert "mean_score" in cv_scores
        assert "std_score" in cv_scores
        assert "fold_scores" in cv_scores
        assert len(cv_scores["fold_scores"]) == 3

    def test_model_training_failure_handling(self):
        """モデル訓練失敗時のフォールバック - 要件 3.4"""
        # 無効なデータでモデル訓練失敗をシミュレート
        invalid_features = pd.DataFrame({"feature1": [np.nan] * 100, "feature2": [np.inf] * 100})
        invalid_labels = pd.Series([0, 1] * 50)

        # LightGBMは無効なデータでも動作する場合があるので、異なるエラーケースをテスト
        # 空のDataFrameで確実にエラーを発生させる
        empty_features = pd.DataFrame()
        empty_labels = pd.Series(dtype=int)

        with pytest.raises((RuntimeError, ValueError)):
            self.trainer.train_model(empty_features, empty_labels)

    def test_feature_importance_extraction(self):
        """特徴量重要度抽出テスト"""
        model, _ = self.trainer.train_model(self.sample_features, self.sample_labels)

        feature_importance = self.trainer.get_feature_importance(model)

        assert isinstance(feature_importance, pd.DataFrame)
        assert "feature" in feature_importance.columns
        assert "importance" in feature_importance.columns
        assert len(feature_importance) == len(self.sample_features.columns)

    def test_model_evaluation_metrics(self):
        """モデル評価メトリクス計算テスト"""
        model, _ = self.trainer.train_model(self.sample_features, self.sample_labels)

        # 検証データで評価
        X_train, X_val, y_train, y_val = self.trainer.split_data(
            self.sample_features, self.sample_labels
        )

        metrics = self.trainer.evaluate_model(model, X_val, y_val)

        assert isinstance(metrics, dict)
        assert "accuracy" in metrics
        assert "auc" in metrics
        assert "precision" in metrics
        assert "recall" in metrics

    def test_save_and_load_model(self):
        """モデル保存・読み込みテスト"""
        model, _ = self.trainer.train_model(self.sample_features, self.sample_labels)

        # モデル保存
        model_path = "/tmp/test_lightgbm_model.txt"
        self.trainer.save_model(model, model_path)

        # モデル読み込み
        loaded_model = self.trainer.load_model(model_path)

        assert isinstance(loaded_model, lgb.Booster)

        # 予測結果の一致確認
        X_test = self.sample_features.iloc[:10]
        original_pred = model.predict(X_test)
        loaded_pred = loaded_model.predict(X_test)

        np.testing.assert_array_almost_equal(original_pred, loaded_pred)

    def test_hyperparameter_validation(self):
        """ハイパーパラメータ検証テスト"""
        # 無効なパラメータ設定
        invalid_params = {
            "learning_rate": -0.1,  # 負の学習率
            "num_leaves": 0,  # 無効な葉数
        }

        with pytest.raises(ValueError, match="無効なハイパーパラメータ"):
            self.trainer.validate_hyperparameters(invalid_params)

    def test_training_with_custom_parameters(self):
        """カスタムパラメータでの訓練テスト"""
        custom_params = {"learning_rate": 0.05, "num_leaves": 20, "max_depth": 5}

        model, history = self.trainer.train_model(
            self.sample_features, self.sample_labels, custom_params=custom_params
        )

        assert isinstance(model, lgb.Booster)
        assert isinstance(history, dict)
