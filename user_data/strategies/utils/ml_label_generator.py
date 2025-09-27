"""
MLLabelGenerator - 機械学習ラベル生成システム

richmanbtcチュートリアルに基づく機械学習用バイナリラベル生成システム

Requirements implemented:
- 3.1: ATRリターンに基づくバイナリラベル生成（正のリターン=1、負・ゼロ=0）
- 3.4: 十分な訓練データの存在確認機能
- 訓練データの適切な前処理機能
- ラベル品質検証機能
"""

import logging
from typing import Any

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


class MLLabelGenerator:
    """
    機械学習用バイナリラベル生成システム

    richmanbtcチュートリアルの2次モデル（機械学習分類）で使用する
    バイナリラベル（0または1）をATRリターンから生成します。
    """

    def __init__(self):
        """初期化"""
        logger.info("MLLabelGenerator初期化")

    def generate_binary_labels_from_atr_returns(self, atr_returns: pd.Series) -> pd.Series:
        """
        ATRリターンからバイナリラベル生成 - 要件3.1

        Args:
            atr_returns: ATRリターンのSeries

        Returns:
            バイナリラベルのSeries（正のリターン=1、負・ゼロ=0）

        Raises:
            ValueError: ATRリターンデータが空の場合
        """
        if atr_returns.empty:
            raise ValueError("ATRリターンデータが空です")

        # 正のリターン=1、負・ゼロ=0のバイナリラベル生成
        labels = (atr_returns > 0).astype(int)

        # 無限大やNaN値の処理
        labels = labels.replace([np.inf, -np.inf], np.nan)

        logger.debug(
            f"バイナリラベル生成完了: {len(labels)}サンプル, 正ラベル比率: {labels.mean():.2%}"
        )
        return labels

    def validate_training_data_sufficiency(self, data: pd.DataFrame, min_samples: int = 50) -> bool:
        """
        十分な訓練データの存在確認 - 要件3.4

        Args:
            data: 訓練データ
            min_samples: 最低限必要なサンプル数

        Returns:
            十分なデータがある場合True

        Raises:
            ValueError: データ不足の場合
        """
        if len(data) < min_samples:
            if len(data) < 10:  # 極端に少ない場合はエラー
                raise ValueError(
                    f"訓練データが不足しています。現在{len(data)}サンプル、最低{min_samples}サンプル必要です。"
                )
            return False

        logger.debug(f"訓練データ十分性確認: {len(data)}サンプル >= {min_samples}サンプル")
        return True

    def preprocess_training_data(
        self, features: pd.DataFrame, labels: pd.Series
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        訓練データの適切な前処理 - 要件3.4

        Args:
            features: 特徴量DataFrame
            labels: ラベルSeries

        Returns:
            前処理済みの特徴量とラベルのタプル
        """
        # インデックスの整合性確認
        common_index = features.index.intersection(labels.index)

        processed_features = features.loc[common_index].copy()
        processed_labels = labels.loc[common_index].copy()

        # NaN値の除去
        valid_rows = ~(processed_features.isnull().any(axis=1) | processed_labels.isnull())
        processed_features = processed_features[valid_rows]
        processed_labels = processed_labels[valid_rows]

        logger.debug(
            f"前処理完了: {len(processed_features)}サンプル、{len(processed_features.columns)}特徴量"
        )
        return processed_features, processed_labels

    def validate_label_quality(self, labels: pd.Series) -> dict[str, bool | float]:
        """
        ラベル品質検証 - 要件3.4

        Args:
            labels: 検証対象のラベル

        Returns:
            品質検証結果の辞書
        """
        total_samples = len(labels)
        positive_count = labels.sum()
        positive_ratio = positive_count / total_samples if total_samples > 0 else 0

        # バランス評価（極端な不均衡を検出）
        min_class_ratio = 0.1  # 10%未満の場合は不均衡とする
        is_balanced = min_class_ratio <= positive_ratio <= (1 - min_class_ratio)

        result = {
            "is_valid": is_balanced and total_samples >= 10,
            "positive_ratio": positive_ratio,
            "negative_ratio": 1 - positive_ratio,
            "balance_score": min(positive_ratio, 1 - positive_ratio) * 2,  # 0-1の範囲
        }

        logger.debug(
            f"ラベル品質: 総数{total_samples}, 正比率{positive_ratio:.2%}, バランス{is_balanced}"
        )
        return result

    def create_training_dataset(
        self, features: pd.DataFrame, atr_returns: pd.Series
    ) -> dict[str, Any]:
        """
        完全な訓練データセット作成

        Args:
            features: 特徴量DataFrame
            atr_returns: ATRリターンSeries

        Returns:
            訓練データセットの辞書
        """
        # ラベル生成
        labels = self.generate_binary_labels_from_atr_returns(atr_returns)

        # データ前処理
        processed_features, processed_labels = self.preprocess_training_data(features, labels)

        # データセット作成
        dataset = {
            "features": processed_features,
            "labels": processed_labels,
            "metadata": {
                "total_samples": len(processed_labels),
                "feature_count": len(processed_features.columns),
                "positive_ratio": processed_labels.mean(),
                "created_at": pd.Timestamp.now(),
            },
        }

        logger.info(
            f"訓練データセット作成完了: {len(processed_labels)}サンプル、{len(processed_features.columns)}特徴量"
        )
        return dataset

    def get_label_distribution_report(self, labels: pd.Series) -> dict[str, int | float]:
        """
        ラベル分布レポート生成

        Args:
            labels: 分析対象のラベル

        Returns:
            分布レポートの辞書
        """
        total_samples = len(labels)
        positive_count = int(labels.sum())
        negative_count = total_samples - positive_count

        positive_ratio = positive_count / total_samples if total_samples > 0 else 0
        negative_ratio = negative_count / total_samples if total_samples > 0 else 0

        # バランススコア（完全均衡=1, 完全不均衡=0）
        balance_score = min(positive_ratio, negative_ratio) * 2

        report = {
            "total_samples": total_samples,
            "positive_count": positive_count,
            "negative_count": negative_count,
            "positive_ratio": positive_ratio,
            "negative_ratio": negative_ratio,
            "balance_score": balance_score,
        }

        logger.info(
            f"ラベル分布: 総数{total_samples}, 正{positive_count}({positive_ratio:.1%}), 負{negative_count}({negative_ratio:.1%})"
        )
        return report
