"""FreqAI既存機能活用ファクトリー

FreqAIの既存分類器・回帰器を設定駆動型で統合選択
"""

import logging
from typing import Dict, Any, Optional, Type

logger = logging.getLogger(__name__)


class FreqAIModelFactory:
    """FreqAI既存モデル統合ファクトリー"""

    # FreqAI標準提供モデルの辞書
    _available_models = {
        # 分類器
        "lightgbm_classifier": "LightGBMClassifier",
        "xgboost_classifier": "XGBoostClassifier",
        "catboost_classifier": "CatboostClassifier",
        "sklearn_rf_classifier": "SKLearnRandomForestClassifier",
        # 回帰器
        "lightgbm_regressor": "LightGBMRegressor",
        "xgboost_regressor": "XGBoostRegressor",
        "catboost_regressor": "CatboostRegressor",
        # PyTorch（高度なユーザー向け）
        "pytorch_mlp_classifier": "PyTorchMLPClassifier",
        "pytorch_mlp_regressor": "PyTorchMLPRegressor",
        "pytorch_transformer_regressor": "PyTorchTransformerRegressor",
        # 強化学習
        "reinforcement_learner": "ReinforcementLearner",
        "reinforcement_learner_multiproc": "ReinforcementLearner_multiproc",
    }

    @classmethod
    def get_model_name(cls, model_type: str) -> str:
        """設定からFreqAIモデル名取得

        Args:
            model_type: 設定での簡易名

        Returns:
            FreqAI標準モデル名

        Raises:
            ValueError: 未対応モデルタイプの場合
        """
        if model_type not in cls._available_models:
            available = list(cls._available_models.keys())
            raise ValueError(f"Unknown model type: {model_type}. Available: {available}")

        return cls._available_models[model_type]

    @classmethod
    def create_freqai_config(cls, strategy_config: Dict[str, Any]) -> Dict[str, Any]:
        """戦略設定からFreqAI設定生成

        Args:
            strategy_config: 2層戦略設定

        Returns:
            FreqAI設定辞書
        """
        secondary_config = strategy_config.get("secondary_model", {})

        if not secondary_config.get("enabled", False):
            return {}

        model_type = secondary_config.get("type", "lightgbm_classifier")
        model_params = secondary_config.get("params", {})

        # FreqAIモデル名取得
        freqai_model = cls.get_model_name(model_type)

        # FreqAI標準設定構造
        freqai_config = {
            "enabled": True,
            "model_training_parameters": {
                **model_params,
                # 共通パラメータ
                "train_period_days": strategy_config.get("train_period_days", 30),
                "backtest_period_days": strategy_config.get("backtest_period_days", 7),
                "identifier": f"2tier_{model_type}",
            },
            "feature_parameters": {
                "include_timeframes": strategy_config.get(
                    "include_timeframes", ["5m", "15m", "1h"]
                ),
                "include_corr_pairlist": strategy_config.get("include_corr_pairs", []),
                "label_period_candles": strategy_config.get("label_period_candles", 24),
                "include_shifted_candles": strategy_config.get("include_shifted_candles", 2),
                "DI_threshold": strategy_config.get("DI_threshold", 0.9),
                "weight_factor": strategy_config.get("weight_factor", 0),
                "principal_component_analysis": strategy_config.get("use_pca", False),
                "use_SVM_to_remove_outliers": strategy_config.get("use_svm_outlier_removal", True),
                "indicator_periods_candles": strategy_config.get("indicator_periods", [10, 20, 50]),
            },
            "data_split_parameters": {
                "test_size": strategy_config.get("test_size", 0.33),
                "shuffle": strategy_config.get("shuffle", False),
            },
        }

        return freqai_config

    @classmethod
    def validate_model_config(cls, model_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """モデル設定の検証と標準化

        Args:
            model_type: モデルタイプ
            params: パラメータ辞書

        Returns:
            検証済みパラメータ辞書
        """
        validated_params = params.copy()

        # モデル別のデフォルトパラメータ設定
        if "lightgbm" in model_type:
            defaults = {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "num_leaves": 31,
                "feature_fraction": 0.8,
                "bagging_fraction": 0.8,
                "bagging_freq": 5,
            }
        elif "xgboost" in model_type:
            defaults = {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 6,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
            }
        elif "catboost" in model_type:
            defaults = {"iterations": 100, "learning_rate": 0.1, "depth": 6, "l2_leaf_reg": 3.0}
        elif "sklearn" in model_type:
            defaults = {
                "n_estimators": 100,
                "max_depth": None,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
            }
        elif "pytorch" in model_type:
            defaults = {"learning_rate": 3e-4, "model_kwargs": {}, "trainer_kwargs": {}}
        else:
            defaults = {}

        # デフォルト値の適用
        for key, default_value in defaults.items():
            if key not in validated_params:
                validated_params[key] = default_value

        return validated_params

    @classmethod
    def list_available_models(cls) -> Dict[str, str]:
        """利用可能なFreqAIモデル一覧

        Returns:
            モデルタイプ→FreqAIクラス名の辞書
        """
        return cls._available_models.copy()

    @classmethod
    def get_model_info(cls, model_type: str) -> Dict[str, Any]:
        """モデル情報取得

        Args:
            model_type: モデルタイプ

        Returns:
            モデル情報辞書
        """
        if model_type not in cls._available_models:
            raise ValueError(f"Unknown model type: {model_type}")

        freqai_class = cls._available_models[model_type]

        # モデル分類
        if "classifier" in model_type:
            task_type = "classification"
        elif "regressor" in model_type:
            task_type = "regression"
        elif "reinforcement" in model_type:
            task_type = "reinforcement_learning"
        else:
            task_type = "unknown"

        # フレームワーク分類
        if "lightgbm" in model_type:
            framework = "lightgbm"
        elif "xgboost" in model_type:
            framework = "xgboost"
        elif "catboost" in model_type:
            framework = "catboost"
        elif "sklearn" in model_type:
            framework = "sklearn"
        elif "pytorch" in model_type:
            framework = "pytorch"
        else:
            framework = "unknown"

        return {
            "freqai_class": freqai_class,
            "task_type": task_type,
            "framework": framework,
            "description": f"FreqAI {framework} {task_type} model",
        }
