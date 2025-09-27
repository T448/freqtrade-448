"""
RealtimePredictor - リアルタイム予測システム

richmanbtcチュートリアルに基づく機械学習予測システム
訓練済みLightGBMモデルによるバイナリ分類予測

Requirements implemented:
- 3.3: 訓練済みモデルによるバイナリ分類予測（0または1）
- 3.6: モデル再訓練とデプロイ前のパフォーマンス検証
- 予測エラー時の適切な処理機能
- 予測結果のログ記録機能
"""

import logging
from typing import List, Tuple, Dict, Any, Optional
import os
import threading

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

logger = logging.getLogger(__name__)


class RealtimePredictor:
    """
    リアルタイム予測システム

    訓練済みLightGBMモデルを使用して、特徴量データから
    バイナリ分類予測（買い=1、売り/待機=0）を実行します。
    """

    def __init__(self):
        """初期化"""
        self._lock = threading.Lock()
        logger.info("RealtimePredictor初期化")

    def predict_binary(
        self, model: lgb.Booster, features: pd.DataFrame, threshold: float = 0.5
    ) -> List[int]:
        """
        バイナリ分類予測を実行 - 要件 3.3

        Args:
            model: 訓練済みLightGBMモデル
            features: 特徴量データ
            threshold: 分類閾値（デフォルト0.5）

        Returns:
            バイナリ予測結果のリスト（0または1）

        Raises:
            ValueError: 入力データが無効な場合
            RuntimeError: 予測実行中にエラーが発生した場合
        """
        try:
            # 入力データ検証
            self._validate_features(features)

            # 予測実行
            probabilities = model.predict(features)

            # バイナリ変換
            predictions = [1 if prob > threshold else 0 for prob in probabilities]

            # ログ記録 - 要件 3.6
            logger.info(f"予測完了: {len(predictions)}件, 正例: {sum(predictions)}件")

            return predictions

        except ValueError as e:
            # ValueErrorはそのまま再発生
            raise e
        except Exception as e:
            logger.error(f"予測実行中にエラーが発生: {e}")
            raise RuntimeError(f"予測実行中にエラーが発生: {e}")

    def predict_with_probabilities(
        self, model: lgb.Booster, features: pd.DataFrame
    ) -> Tuple[List[int], List[float]]:
        """
        確率値付きでバイナリ予測を実行

        Args:
            model: 訓練済みLightGBMモデル
            features: 特徴量データ

        Returns:
            (予測結果, 予測確率) のタプル
        """
        self._validate_features(features)

        probabilities = model.predict(features).tolist()
        predictions = [1 if prob > 0.5 else 0 for prob in probabilities]

        return predictions, probabilities

    def load_model(self, model_path: str) -> lgb.Booster:
        """
        LightGBMモデルを読み込み

        Args:
            model_path: モデルファイルパス

        Returns:
            読み込まれたLightGBMモデル

        Raises:
            FileNotFoundError: モデルファイルが見つからない場合
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"モデルファイルが見つかりません: {model_path}")

        try:
            model = lgb.Booster(model_file=model_path)
            logger.info(f"モデル読み込み完了: {model_path}")
            return model
        except Exception as e:
            logger.error(f"モデル読み込み失敗: {e}")
            raise

    def validate_model_performance(
        self, model: lgb.Booster, X_val: pd.DataFrame, y_val: pd.Series, min_accuracy: float = 0.7
    ) -> Dict[str, Any]:
        """
        モデル性能検証 - 要件 3.6

        Args:
            model: 検証対象モデル
            X_val: 検証用特徴量
            y_val: 検証用ラベル
            min_accuracy: 最低精度要件

        Returns:
            検証結果の辞書
        """
        predictions = self.predict_binary(model, X_val)

        metrics = {
            "accuracy": accuracy_score(y_val, predictions),
            "precision": precision_score(y_val, predictions, zero_division=0),
            "recall": recall_score(y_val, predictions, zero_division=0),
            "f1_score": f1_score(y_val, predictions, zero_division=0),
        }

        metrics["is_valid"] = metrics["accuracy"] >= min_accuracy

        logger.info(f"モデル性能検証: 精度={metrics['accuracy']:.3f}")

        return metrics

    def should_retrain_model(
        self, performance_metrics: Dict[str, float], min_accuracy: float = 0.7
    ) -> bool:
        """
        モデル再訓練の必要性判定 - 要件 3.6

        Args:
            performance_metrics: 性能メトリクス
            min_accuracy: 最低精度要件

        Returns:
            再訓練が必要な場合True
        """
        current_accuracy = performance_metrics.get("accuracy", 0.0)
        return current_accuracy < min_accuracy

    def predict_binary_batch(
        self, model: lgb.Booster, features: pd.DataFrame, batch_size: int = 1000
    ) -> List[int]:
        """
        バッチ処理での予測実行

        Args:
            model: 訓練済みモデル
            features: 特徴量データ
            batch_size: バッチサイズ

        Returns:
            予測結果のリスト
        """
        self._validate_features(features)

        all_predictions = []

        for i in range(0, len(features), batch_size):
            batch_features = features.iloc[i : i + batch_size]
            # バッチ単位で直接予測して結果を取得
            probabilities = model.predict(batch_features)
            batch_predictions = [1 if prob > 0.5 else 0 for prob in probabilities]
            all_predictions.extend(batch_predictions)

        logger.info(f"バッチ予測完了: {len(all_predictions)}件")
        return all_predictions

    def calculate_prediction_confidence(self, probabilities: List[float]) -> List[float]:
        """
        予測信頼度計算

        Args:
            probabilities: 予測確率のリスト

        Returns:
            信頼度スコアのリスト（0.5からの距離ベース）
        """
        return [abs(prob - 0.5) * 2 for prob in probabilities]

    def load_model_with_version(self, model_path: str, version: str) -> lgb.Booster:
        """
        バージョン追跡付きモデル読み込み

        Args:
            model_path: モデルファイルパス
            version: モデルバージョン

        Returns:
            読み込まれたモデル
        """
        model = self.load_model(model_path)
        logger.info(f"モデルバージョン {version} を読み込み")
        return model

    def format_prediction_result(
        self,
        predictions: List[int],
        probabilities: List[float],
        timestamps: Optional[pd.DatetimeIndex] = None,
    ) -> pd.DataFrame:
        """
        予測結果のフォーマット

        Args:
            predictions: 予測結果
            probabilities: 予測確率
            timestamps: タイムスタンプ

        Returns:
            フォーマット済み結果のDataFrame
        """
        result_data = {
            "prediction": predictions,
            "probability": probabilities,
        }

        if timestamps is not None:
            result_data["timestamp"] = timestamps

        return pd.DataFrame(result_data)

    def _validate_features(self, features: pd.DataFrame) -> None:
        """
        特徴量データの検証

        Args:
            features: 特徴量データ

        Raises:
            ValueError: データが無効な場合
        """
        if features.empty:
            raise ValueError("特徴量データが空です")

        if features.isnull().any().any():
            raise ValueError("特徴量データに欠損値が含まれています")
