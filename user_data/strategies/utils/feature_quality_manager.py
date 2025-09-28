"""
FeatureQualityManager - 特徴量品質管理システム

richmanbtcチュートリアルに基づく機械学習特徴量の品質管理システム

Requirements implemented:
- 2.2: 十分な履歴データの存在確認機能
- 2.4: ルックバック期間に対応した履歴データ確保機能
- 特徴量計算エラー時の適切な処理機能
- 完全な特徴量ベクトル生成機能
"""

import logging

import numpy as np
import pandas as pd

from user_data.strategies.utils.technical_indicator_engine import TechnicalIndicatorEngine


logger = logging.getLogger(__name__)


class FeatureQualityManager:
    """
    機械学習特徴量の品質管理システム

    richmanbtcチュートリアルの2次モデル（機械学習分類）で使用する
    特徴量の品質管理と検証を行います。
    """

    def __init__(self):
        """初期化"""
        self.technical_engine = TechnicalIndicatorEngine()
        logger.info("FeatureQualityManager初期化")

    def validate_data_sufficiency(self, dataframe: pd.DataFrame, min_periods: int = 50) -> bool:
        """
        テクニカル指標計算に十分なデータがあるかチェック - 要件2.2

        Args:
            dataframe: チェック対象のデータ
            min_periods: 最低必要期間

        Returns:
            十分なデータがある場合True

        Raises:
            ValueError: データが空の場合
        """
        if dataframe.empty:
            raise ValueError("入力データが空です")

        is_sufficient = len(dataframe) >= min_periods
        logger.debug(
            f"データ十分性チェック: {len(dataframe)}行 >= {min_periods}期間 = {is_sufficient}"
        )

        return is_sufficient

    def validate_lookback_period_support(
        self, dataframe: pd.DataFrame, lookback_periods: dict[str, int]
    ) -> bool:
        """
        ルックバック期間要件への対応確認 - 要件2.4

        Args:
            dataframe: 対象データ
            lookback_periods: 各指標に必要なルックバック期間

        Returns:
            すべての指標について十分なデータがある場合True
        """
        if dataframe.empty:
            return False

        max_lookback = max(lookback_periods.values()) if lookback_periods else 0
        is_supported = len(dataframe) >= max_lookback

        logger.debug(
            f"ルックバック期間サポートチェック: {len(dataframe)}行 >= {max_lookback}期間 = {is_supported}"
        )

        return is_supported

    def generate_complete_feature_vector(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        完全な特徴量ベクトル生成 - 要件2.2

        Args:
            dataframe: OHLCV市場データ

        Returns:
            すべての特徴量を含むDataFrame

        Raises:
            ValueError: データが不足している場合
            RuntimeError: 特徴量計算中にエラーが発生した場合
        """
        min_required_data = 50  # 最低限必要なデータ数

        if not self.validate_data_sufficiency(dataframe, min_required_data):
            raise ValueError(f"データが不足しています。最低{min_required_data}行必要です。")

        try:
            # テクニカル指標の計算
            features = self.technical_engine.calculate_all_indicators(dataframe)

            # 特徴量データのクリーニング
            cleaned_features = self.clean_feature_data(features)

            logger.info(
                f"特徴量ベクトル生成完了: {len(cleaned_features.columns)}特徴量, {len(cleaned_features)}行"
            )
            return cleaned_features

        except Exception as e:
            logger.error(f"特徴量計算中にエラーが発生: {e}")
            raise RuntimeError(f"特徴量計算中にエラーが発生: {e}")

    def clean_feature_data(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        特徴量データのクリーニング処理

        Args:
            dataframe: クリーニング対象のDataFrame

        Returns:
            クリーニング後のDataFrame
        """
        result = dataframe.copy()

        # 数値カラムのみ対象
        numeric_columns = result.select_dtypes(include=[np.number]).columns

        # 無限大値をNaNに変換
        for column in numeric_columns:
            result[column] = result[column].replace([np.inf, -np.inf], np.nan)

        logger.debug(f"特徴量データクリーニング完了: {len(numeric_columns)}列")
        return result

    def validate_feature_completeness(
        self, features: pd.DataFrame, required_features: list[str], max_missing_ratio: float = 0.3
    ) -> bool:
        """
        特徴量の完全性検証

        Args:
            features: 検証対象の特徴量DataFrame
            required_features: 必須特徴量のリスト
            max_missing_ratio: 許容する最大欠損値比率

        Returns:
            すべての必須特徴量が存在し、欠損値が許容範囲内の場合True
        """
        # 必須特徴量の存在確認
        missing_features = [feat for feat in required_features if feat not in features.columns]
        if missing_features:
            logger.warning(f"必須特徴量が不足: {missing_features}")
            return False

        # 欠損値比率チェック
        for feature in required_features:
            missing_ratio = features[feature].isnull().sum() / len(features)
            if missing_ratio > max_missing_ratio:
                logger.warning(f"特徴量{feature}の欠損値比率が高すぎます: {missing_ratio:.2%}")
                return False

        return True

    def get_feature_quality_report(
        self, features: pd.DataFrame
    ) -> dict[str, int | float | list[str]]:
        """
        特徴量品質レポートの生成

        Args:
            features: 分析対象の特徴量DataFrame

        Returns:
            品質レポートの辞書
        """
        total_features = len(features.columns)
        total_samples = len(features)

        # 全体の欠損値比率
        total_missing = features.isnull().sum().sum()
        total_values = total_features * total_samples
        missing_ratio = total_missing / total_values if total_values > 0 else 0

        # 問題のある特徴量の特定
        features_with_issues = []
        for column in features.columns:
            column_missing_ratio = features[column].isnull().sum() / len(features)
            if column_missing_ratio > 0.1:  # 10%以上の欠損値
                features_with_issues.append(column)

        report = {
            "total_features": total_features,
            "total_samples": total_samples,
            "missing_value_ratio": missing_ratio,
            "features_with_issues": features_with_issues,
        }

        logger.info(
            f"特徴量品質レポート生成完了: {total_features}特徴量, 問題のある特徴量: {len(features_with_issues)}"
        )
        return report
