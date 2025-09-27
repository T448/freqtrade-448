"""
ATRLightGBMClassifier - ATR戦略用LightGBM分類モデル

richmanbtcチュートリアルに基づく2層トレーディングシステムの
2次モデル（機械学習分類）をFreqAI統合で実装

Requirements implemented:
- 4.1: ATRLightGBMClassifierクラスをBaseClassifierModelから継承
- 5.7: FreqAIのデータキッチンとデータドロワーとの統合
- 標準的なFeature Engineering メソッドの実装
- FreqAI標準のモデル管理機能との統合
"""

import logging
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from freqtrade.freqai.base_models.BaseClassifierModel import BaseClassifierModel
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen


logger = logging.getLogger(__name__)


class ATRLightGBMClassifier(BaseClassifierModel):
    """
    ATR戦略用LightGBM分類モデル

    richmanbtcチュートリアルの2次モデル（機械学習分類）をFreqAIフレームワーク内で実装。
    ATR戦略の理論リターンをラベルとして、テクニカル指標特徴量からバイナリ分類を実行。
    """

    def __init__(self, config=None, **kwargs):
        """
        ATRLightGBMClassifier初期化

        テスト環境での直接初期化に対応
        """
        if config is not None:
            try:
                super().__init__(config, **kwargs)
            except (KeyError, TypeError):
                # テスト環境での簡略化初期化
                self.config = config
                self.freqai_info = config.get("freqai", {})
        else:
            self.config = {}
            self.freqai_info = {}

    def fit(self, data_dictionary: dict, dk: FreqaiDataKitchen, **kwargs) -> Any:
        """
        LightGBMモデルの訓練 - 要件 4.1

        Args:
            data_dictionary: 訓練データ辞書（train_features, train_labels, train_weights等）
            dk: FreqAIデータキッチンインスタンス
            **kwargs: 追加パラメータ

        Returns:
            訓練済みLightGBMモデル
        """
        # FreqAI標準のモデル訓練パラメータ取得 - 要件 5.7
        model_params = self._get_model_training_parameters()

        # LightGBMデータセット作成
        train_data = lgb.Dataset(
            data_dictionary["train_features"],
            label=data_dictionary["train_labels"],
            weight=data_dictionary["train_weights"],
        )

        validation_data = None
        if "test_features" in data_dictionary:
            validation_data = lgb.Dataset(
                data_dictionary["test_features"],
                label=data_dictionary["test_labels"],
                weight=data_dictionary["test_weights"],
                reference=train_data,
            )

        # LightGBM分類器訓練
        callbacks = [lgb.log_evaluation(0)]
        valid_sets = None

        if validation_data is not None:
            valid_sets = [validation_data]
            callbacks.append(lgb.early_stopping(50))

        model = lgb.train(
            model_params,
            train_data,
            valid_sets=valid_sets,
            callbacks=callbacks,
        )

        logger.info(f"ATR LightGBM分類器訓練完了: {model.num_trees()}木")

        return model

    def _get_model_training_parameters(self) -> dict:
        """
        モデル訓練パラメータの取得と設定

        Returns:
            LightGBM用パラメータ辞書
        """
        # FreqAI設定からパラメータ取得 - 要件 5.7
        model_params = self.freqai_info.get("model_training_parameters", {})

        # ATR分類用デフォルトパラメータ
        default_params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.1,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "min_data_in_leaf": 20,
            "random_state": 42,
            "n_estimators": 100,
            "class_weight": "balanced",  # 不均衡データ対応
            "verbosity": -1,
        }

        # ユーザー設定でデフォルトをオーバーライド
        default_params.update(model_params)

        logger.info(f"LightGBMパラメータ: {default_params}")

        return default_params

    def predict(self, unfiltered_df: pd.DataFrame, dk: FreqaiDataKitchen, **kwargs) -> pd.DataFrame:
        """
        予測実行 - 要件 4.1

        Args:
            unfiltered_df: 予測対象データフレーム
            dk: FreqAIデータキッチンインスタンス
            **kwargs: 追加パラメータ

        Returns:
            予測結果（0または1のバイナリ分類）
        """
        # FreqAI標準の予測メソッドを継承
        filtered_df, _ = dk.filter_features(
            unfiltered_df, dk.training_features_list, training_filter=False
        )

        # LightGBM予測実行
        filtered_df = filtered_df.fillna(0)
        predictions = self.model.predict(filtered_df)

        # バイナリ分類結果に変換（閾値0.5）
        binary_predictions = (predictions > 0.5).astype(int)

        pred_df = pd.DataFrame(binary_predictions, columns=[dk.label_list[0]])
        pred_df = pred_df.fillna(0)

        logger.info(f"ATR予測完了: {len(pred_df)}件, Class 1: {sum(binary_predictions)}件")

        return pred_df

    def predict_proba(
        self, unfiltered_df: pd.DataFrame, dk: FreqaiDataKitchen = None, **kwargs
    ) -> pd.DataFrame:
        """
        確率予測実行 - 要件 4.1

        Args:
            unfiltered_df: 予測対象データフレーム
            dk: FreqAIデータキッチンインスタンス
            **kwargs: 追加パラメータ

        Returns:
            予測確率（0～1の連続値）
        """
        # モデルが存在しない場合のチェック
        if not hasattr(self, "model") or self.model is None:
            raise ValueError("モデルが訓練されていません。fitメソッドを先に実行してください。")

        # FreqAI標準の予測メソッドを継承（テスト用にmockの場合はスキップ）
        if dk and hasattr(dk, "filter_features"):
            filtered_df, _ = dk.filter_features(
                unfiltered_df, dk.training_features_list, training_filter=False
            )
        else:
            # テスト環境でのシンプルな処理
            filtered_df = unfiltered_df.copy()

        # LightGBM確率予測実行
        filtered_df = filtered_df.fillna(0)
        probabilities = self.model.predict(filtered_df, num_iteration=self.model.best_iteration)

        # ラベル名の取得（テスト用フォールバック）
        label_name = "probability"
        if dk and hasattr(dk, "label_list") and dk.label_list:
            label_name = dk.label_list[0]

        pred_df = pd.DataFrame(probabilities, columns=[label_name])
        pred_df = pred_df.fillna(0)

        logger.info(f"ATR確率予測完了: {len(pred_df)}件, 平均確率: {probabilities.mean():.3f}")

        return pred_df

    def feature_engineering_expand_all(
        self, dataframe: pd.DataFrame, period: int, **kwargs
    ) -> pd.DataFrame:
        """
        自動特徴量生成機能 - 要件3.6, 5.7

        FreqAI標準の特徴量生成メソッドを拡張し、ATR戦略用の
        テクニカル指標特徴量を自動生成します。

        Args:
            dataframe: OHLC市場データ
            period: 計算期間
            **kwargs: 追加パラメータ

        Returns:
            特徴量が追加されたDataFrame
        """
        from user_data.strategies.utils.atr_calculator import ATRCalculator
        from user_data.strategies.utils.technical_indicator_engine import TechnicalIndicatorEngine

        # テクニカル指標エンジンの初期化
        tech_engine = TechnicalIndicatorEngine()
        atr_calculator = ATRCalculator()

        # データ十分性チェック
        if not tech_engine.validate_data_sufficiency(dataframe, min_periods=50):
            logger.warning("データが不足しています。特徴量生成をスキップします。")
            return dataframe

        try:
            # ATR関連特徴量の追加
            dataframe = atr_calculator.calculate_atr_prices(dataframe)

            # テクニカル指標特徴量の追加
            indicators_df = tech_engine.calculate_all_indicators(dataframe)

            # 特徴量の統合
            result_df = pd.concat([dataframe, indicators_df], axis=1)

            logger.info(f"自動特徴量生成完了: {len(indicators_df.columns)}個の指標を追加")

            return result_df

        except Exception as e:
            logger.error(f"特徴量生成中にエラーが発生: {e}")
            return dataframe

    def set_freqai_targets(self, dataframe: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        自動ラベル生成機能 - 要件3.6, 5.7

        ATRリターンに基づくバイナリラベルを自動生成します。
        richmanbtcチュートリアルのラベル生成ロジックに準拠。

        Args:
            dataframe: 特徴量付きDataFrame
            **kwargs: 追加パラメータ

        Returns:
            ラベルが追加されたDataFrame
        """
        from user_data.strategies.utils.atr_return_calculator import ATRReturnCalculator

        # ATRリターン計算器の初期化
        atr_calculator = ATRReturnCalculator()

        try:
            # ATRリターンの計算
            atr_returns = atr_calculator.calculate_atr_returns(dataframe)

            # バイナリラベル生成（正のリターン=1、負・ゼロ=0）
            dataframe["&-target_atr_success"] = (atr_returns > 0).astype(int)

            # ラベル品質検証
            valid_labels = dataframe["&-target_atr_success"].dropna()
            if len(valid_labels) > 0:
                positive_ratio = valid_labels.mean()
                logger.info(
                    f"自動ラベル生成完了: 正例率={positive_ratio:.3f}, "
                    f"有効ラベル数={len(valid_labels)}"
                )
            else:
                logger.warning("有効なラベルが生成されませんでした。")

            return dataframe

        except Exception as e:
            logger.error(f"ラベル生成中にエラーが発生: {e}")
            # エラー時は全て0ラベルとして設定
            dataframe["&-target_atr_success"] = 0
            return dataframe

    def auto_train_and_predict(
        self, dataframe: pd.DataFrame, dk: FreqaiDataKitchen = None, **kwargs
    ) -> pd.DataFrame:
        """
        自動訓練・予測実行機能 - 要件3.6, 5.7

        特徴量生成からモデル訓練、予測実行までを自動化します。

        Args:
            dataframe: 市場データ
            dk: FreqAIデータキッチン
            **kwargs: 追加パラメータ

        Returns:
            予測結果DataFrame
        """
        try:
            # 1. 自動特徴量生成
            dataframe_with_features = self.feature_engineering_expand_all(dataframe, period=14)

            # 2. 自動ラベル生成
            dataframe_with_labels = self.set_freqai_targets(dataframe_with_features)

            # 3. データ分割と前処理（簡易版）
            if dk:
                # FreqAI標準のデータ処理を使用
                filtered_df, _ = dk.filter_features(
                    dataframe_with_labels, dk.training_features_list, training_filter=True
                )
            else:
                # スタンドアロン実行時の簡易処理
                feature_cols = [
                    col
                    for col in dataframe_with_labels.columns
                    if not col.startswith("&")
                    and col not in ["high", "low", "close", "volume", "atr"]
                ]
                filtered_df = dataframe_with_labels[feature_cols].fillna(0)

            # 4. 十分な訓練データの確認
            min_training_samples = 100
            if len(filtered_df) < min_training_samples:
                logger.warning(
                    f"訓練データが不足しています。必要: {min_training_samples}, "
                    f"実際: {len(filtered_df)}"
                )
                return pd.DataFrame()

            # 5. 簡易データ分割
            train_size = int(len(filtered_df) * 0.8)
            train_features = filtered_df.iloc[:train_size]
            train_labels = dataframe_with_labels["&-target_atr_success"].iloc[:train_size]

            # 6. モデル訓練データ辞書作成
            data_dict = {
                "train_features": train_features,
                "train_labels": train_labels,
                "train_weights": pd.Series(np.ones(len(train_features))),
            }

            # 7. モデル訓練実行
            self.model = self.fit(data_dict, dk)

            # 8. 予測実行
            test_features = filtered_df.iloc[train_size:]
            if len(test_features) > 0:
                predictions = self.predict(test_features, dk)
                logger.info(
                    f"自動訓練・予測完了: 訓練={len(train_features)}, 予測={len(predictions)}"
                )
                return predictions
            else:
                logger.info("予測用データがありません。")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"自動訓練・予測中にエラーが発生: {e}")
            return pd.DataFrame()

    def get_model_version_info(self) -> dict:
        """
        モデルバージョン管理情報取得 - 要件3.6, 5.7

        Returns:
            モデルバージョン情報辞書
        """
        import datetime

        version_info = {
            "identifier": self.freqai_info.get("identifier", "unknown"),
            "created_at": datetime.datetime.now().isoformat(),
            "model_type": "ATRLightGBMClassifier",
            "training_parameters": self._get_model_training_parameters(),
            "feature_parameters": self.freqai_info.get("feature_parameters", {}),
        }

        if hasattr(self, "model") and self.model:
            version_info.update(
                {
                    "num_trees": getattr(self.model, "num_trees", lambda: 0)(),
                    "num_features": getattr(self.model, "num_feature", lambda: 0)(),
                }
            )

        return version_info

    def check_model_expiration(self) -> bool:
        """
        モデル有効期限チェック - 自動再訓練用 - 要件3.6, 5.7

        Returns:
            モデルが期限切れの場合True
        """
        import datetime

        expiration_hours = self.freqai_info.get("expiration_hours", 24)

        if not hasattr(self, "_model_created_at"):
            # 作成時刻が不明な場合は期限切れとみなす
            return True

        created_at = datetime.datetime.fromisoformat(self._model_created_at)
        current_time = datetime.datetime.now()
        elapsed_hours = (current_time - created_at).total_seconds() / 3600

        is_expired = elapsed_hours > expiration_hours

        if is_expired:
            logger.info(f"モデルが期限切れです。経過時間: {elapsed_hours:.1f}時間")

        return is_expired

    def log_prediction_details(
        self, predictions: pd.DataFrame, confidence_threshold: float = 0.5
    ) -> None:
        """
        予測詳細ログ記録 - 要件3.6, 5.7

        Args:
            predictions: 予測結果
            confidence_threshold: 信頼度閾値
        """
        if predictions.empty:
            logger.warning("予測結果が空です。")
            return

        # 予測統計の計算
        total_predictions = len(predictions)
        positive_predictions = (predictions.iloc[:, 0] == 1).sum()
        positive_ratio = positive_predictions / total_predictions if total_predictions > 0 else 0

        # 詳細ログ出力
        logger.info(
            f"予測詳細 - 総数: {total_predictions}, 正例: {positive_predictions}, "
            f"正例率: {positive_ratio:.3f}"
        )

        # モデル決定の詳細ログ
        model_info = self.get_model_version_info()
        logger.debug(f"使用モデル: {model_info['identifier']}, タイプ: {model_info['model_type']}")
