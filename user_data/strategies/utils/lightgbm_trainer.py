"""
LightGBMTrainer - LightGBM二値分類器訓練システム

richmanbtcチュートリアルに基づくLightGBM機械学習分類器の訓練システム

Requirements implemented:
- 3.2: テクニカル指標特徴量を使用したLightGBM二値分類器の訓練
- 3.4: 適切な訓練・検証データ分割機能
- 3.5: オーバーフィッティング防止機能
- 3.4: モデル訓練失敗時のフォールバック機能
"""

import logging
import os
from typing import Any

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split


logger = logging.getLogger(__name__)


class LightGBMTrainer:
    """
    LightGBM二値分類器訓練システム

    richmanbtcチュートリアルの2次モデル（機械学習分類）として使用する
    LightGBM分類器を訓練・評価・保存するシステムです。
    """

    def __init__(self):
        """初期化"""
        self.default_params = self._get_default_parameters()
        logger.info("LightGBMTrainer初期化")

    def _get_default_parameters(self) -> dict[str, Any]:
        """
        デフォルトのLightGBMパラメータを取得

        Returns:
            デフォルトパラメータの辞書
        """
        return {
            "objective": "binary",
            "metric": ["binary_logloss", "auc"],
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.1,
            "feature_fraction": 0.8,  # オーバーフィッティング防止
            "bagging_fraction": 0.8,  # オーバーフィッティング防止
            "bagging_freq": 5,  # オーバーフィッティング防止
            "min_data_in_leaf": 20,  # オーバーフィッティング防止
            "max_depth": -1,
            "verbose": -1,
            "seed": 42,
        }

    def split_data(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        train_ratio: float = 0.8,
        random_state: int = 42,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        訓練・検証データ分割 - 要件3.4

        Args:
            features: 特徴量DataFrame
            labels: ラベルSeries
            train_ratio: 訓練データの比率
            random_state: 乱数シード

        Returns:
            (X_train, X_val, y_train, y_val)のタプル
        """
        X_train, X_val, y_train, y_val = train_test_split(
            features,
            labels,
            train_size=train_ratio,
            random_state=random_state,
            stratify=labels,  # ラベルの分布を保持
        )

        logger.debug(f"データ分割完了: 訓練{len(X_train)}サンプル, 検証{len(X_val)}サンプル")
        return X_train, X_val, y_train, y_val

    def create_lgb_datasets(
        self, X_train: pd.DataFrame, X_val: pd.DataFrame, y_train: pd.Series, y_val: pd.Series
    ) -> tuple[lgb.Dataset, lgb.Dataset]:
        """
        LightGBMデータセット作成

        Args:
            X_train: 訓練特徴量
            X_val: 検証特徴量
            y_train: 訓練ラベル
            y_val: 検証ラベル

        Returns:
            (train_dataset, val_dataset)のタプル
        """
        train_dataset = lgb.Dataset(X_train, label=y_train)
        val_dataset = lgb.Dataset(X_val, label=y_val, reference=train_dataset)

        return train_dataset, val_dataset

    def configure_training_parameters(
        self, custom_params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        訓練パラメータ設定

        Args:
            custom_params: カスタムパラメータ

        Returns:
            最終的な訓練パラメータ
        """
        params = self.default_params.copy()

        if custom_params:
            params.update(custom_params)

        return params

    def configure_callbacks(self) -> list[Any]:
        """
        コールバック設定 - 要件3.5（オーバーフィッティング防止）

        Returns:
            コールバックのリスト
        """
        callbacks = [lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=100)]

        return callbacks

    def validate_hyperparameters(self, params: dict[str, Any]) -> None:
        """
        ハイパーパラメータ検証

        Args:
            params: 検証対象のパラメータ

        Raises:
            ValueError: 無効なパラメータの場合
        """
        if "learning_rate" in params and params["learning_rate"] <= 0:
            raise ValueError("無効なハイパーパラメータ: learning_rateは正の値である必要があります")

        if "num_leaves" in params and params["num_leaves"] <= 0:
            raise ValueError("無効なハイパーパラメータ: num_leavesは正の値である必要があります")

    def train_model(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        custom_params: dict[str, Any] | None = None,
    ) -> tuple[lgb.Booster, dict[str, Any]]:
        """
        LightGBMモデル訓練 - 要件3.2

        Args:
            features: 特徴量DataFrame
            labels: ラベルSeries
            custom_params: カスタム訓練パラメータ

        Returns:
            (trained_model, training_history)のタプル

        Raises:
            ValueError: 訓練データが不足している場合
            RuntimeError: モデル訓練中にエラーが発生した場合
        """
        # データ十分性チェック
        if len(features) < 100:
            raise ValueError(
                f"訓練データが不足しています。現在{len(features)}サンプル、最低100サンプル必要です。"
            )

        try:
            # データ分割
            X_train, X_val, y_train, y_val = self.split_data(features, labels)

            # LightGBMデータセット作成
            train_dataset, val_dataset = self.create_lgb_datasets(X_train, X_val, y_train, y_val)

            # パラメータ設定
            params = self.configure_training_parameters(custom_params)

            # パラメータ検証
            if custom_params:
                self.validate_hyperparameters(custom_params)

            # コールバック設定
            callbacks = self.configure_callbacks()

            # モデル訓練
            model = lgb.train(
                params,
                train_dataset,
                valid_sets=[train_dataset, val_dataset],
                valid_names=["train", "eval"],
                num_boost_round=1000,
                callbacks=callbacks,
            )

            # 訓練履歴作成
            training_history = {
                "train_score": model.best_score["train"],
                "val_score": model.best_score["eval"],
                "best_iteration": model.best_iteration,
            }

            logger.info(
                f"モデル訓練完了: 最良反復{model.best_iteration}, 検証AUC: {model.best_score['eval']['auc']:.4f}"
            )
            return model, training_history

        except Exception as e:
            logger.error(f"モデル訓練中にエラーが発生: {e}")
            raise RuntimeError(f"モデル訓練中にエラーが発生: {e}")

    def cross_validate(
        self, features: pd.DataFrame, labels: pd.Series, n_folds: int = 5
    ) -> dict[str, float | list[float]]:
        """
        クロスバリデーション実行

        Args:
            features: 特徴量DataFrame
            labels: ラベルSeries
            n_folds: フォールド数

        Returns:
            クロスバリデーション結果
        """
        # LightGBMClassifierを使用してクロスバリデーション実行
        lgb_classifier = lgb.LGBMClassifier(**self.default_params)

        cv_scores = cross_val_score(
            lgb_classifier,
            features,
            labels,
            cv=StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42),
            scoring="roc_auc",
        )

        results = {
            "mean_score": cv_scores.mean(),
            "std_score": cv_scores.std(),
            "fold_scores": cv_scores.tolist(),
        }

        logger.info(
            f"クロスバリデーション完了: 平均AUC {results['mean_score']:.4f} ± {results['std_score']:.4f}"
        )
        return results

    def get_feature_importance(self, model: lgb.Booster) -> pd.DataFrame:
        """
        特徴量重要度抽出

        Args:
            model: 訓練済みLightGBMモデル

        Returns:
            特徴量重要度DataFrame
        """
        feature_names = model.feature_name()
        importance_values = model.feature_importance(importance_type="gain")

        feature_importance = pd.DataFrame(
            {"feature": feature_names, "importance": importance_values}
        ).sort_values("importance", ascending=False)

        return feature_importance

    def evaluate_model(
        self, model: lgb.Booster, X_test: pd.DataFrame, y_test: pd.Series
    ) -> dict[str, float]:
        """
        モデル評価メトリクス計算

        Args:
            model: 評価対象のモデル
            X_test: テスト特徴量
            y_test: テストラベル

        Returns:
            評価メトリクス辞書
        """
        # 予測実行
        y_pred_proba = model.predict(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int)

        # メトリクス計算
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "auc": roc_auc_score(y_test, y_pred_proba),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
        }

        return metrics

    def save_model(self, model: lgb.Booster, model_path: str) -> None:
        """
        モデル保存

        Args:
            model: 保存対象のモデル
            model_path: 保存先パス
        """
        # ディレクトリが存在しない場合は作成
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        model.save_model(model_path)
        logger.info(f"モデル保存完了: {model_path}")

    def load_model(self, model_path: str) -> lgb.Booster:
        """
        モデル読み込み

        Args:
            model_path: モデルファイルパス

        Returns:
            読み込み済みモデル
        """
        model = lgb.Booster(model_file=model_path)
        logger.info(f"モデル読み込み完了: {model_path}")
        return model
