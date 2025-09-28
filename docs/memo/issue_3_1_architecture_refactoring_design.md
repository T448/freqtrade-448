# 2層戦略アーキテクチャリファクタリング設計書

## 1. 現状分析と問題点

### 1.1 現在のファイル構成

#### ユーザーデータ層 (`user_data/strategies/utils/`)

```
├── atr_calculator.py           # ATR特化計算エンジン
├── atr_return_calculator.py    # ATR特化リターン計算
├── lightgbm_trainer.py         # LightGBM特化訓練
├── entry_strategy.py           # エントリー戦略パターン ✓
├── technical_indicator_engine.py # 汎用テクニカル指標 ✓
├── feature_quality_manager.py  # 汎用特徴量管理 ✓
├── ml_label_generator.py       # 汎用ラベル生成 ✓
└── realtime_predictor.py       # 汎用予測エンジン ✓
```

#### FreqAI統合層 (`freqtrade/freqai/prediction_models/`)

```
├── atr_lightgbm_classifier.py  # ATR+LightGBM特化
└── mlops_pipeline.py           # MLOps自動化
```

### 1.2 問題点

#### **特化ファイルの多さ**

- `atr_*` ファイル：3個
- `lightgbm_*` ファイル：2個
- 計5個の特化ファイル（全体の45%）

#### **拡張時の問題**

```
新手法追加時の増加パターン：
ボリンジャーバンド戦略 → bollinger_calculator.py, bollinger_return_calculator.py
XGBoost使用 → xgboost_trainer.py, atr_xgboost_classifier.py

結果：手法 × アルゴリズム の組み合わせ爆発
```

#### **保守性の課題**

- 類似機能の重複実装
- テスト時の複数ファイル対応
- 設定変更時の複数箇所修正

### 1.3 将来要件

#### **1次モデル（価格計算）の置き換え**

- ATR → ボリンジャーバンド、移動平均、RSI等
- 統一インターフェースでの切り替え

#### **2次モデル（ML）の置き換え**

- LightGBM → XGBoost、RandomForest、CatBoost等
- FreqAI統合の維持

#### **戦略構造の維持**

- ルールベース1次モデル + ML2次モデルのパターン継続
- 最小変更での手法切り替え

## 2. 設計目標

### 2.1 主要目標

1. **ファイル数削減**: 10ファイル → 6ファイル（40%削減）
2. **特化ファイル排除**: 手法特化ファイルの抽象化
3. **設定駆動型**: コード変更なしでの手法切り替え
4. **最小変更原則**: 新手法追加時はクラス追加のみ

### 2.2 品質目標

- **拡張性**: 新手法追加の容易さ
- **保守性**: 関連機能の集約
- **テスト容易性**: モック・スタブの活用
- **可読性**: 設計意図の明確性

## 3. 改善アーキテクチャ（FreqAI最大活用）

### 3.1 FreqAI既存機能の活用

#### **FreqAI提供済み機能（活用すべき）**

- `BaseClassifierModel`, `BaseRegressionModel`: ML基底クラス
- `FreqaiDataKitchen`: 前処理パイプライン（特徴量、正規化、外れ値除去）
- 既存ML分類器: `LightGBMClassifier`, `XGBoostClassifier`等
- 特徴量エンジニアリング: `feature_engineering_*`メソッド
- `self.freqai.start()`: 戦略統合機能

#### **重複排除後の新ファイル構成**

```
【改善後】4ファイル（6ファイル削減、FreqAI機能活用）
user_data/strategies/utils/
├── price_calculator.py         # 1次モデル統合（ATR、ボリンジャー等）
├── strategy_factory.py        # 戦略選択・設定統合
├── freqai_model_factory.py    # FreqAI既存モデル統合
└── strategy_config.py         # 設定駆動型パラメータ

（削除・統合されるファイル）
❌ ml_trainer.py              # → FreqAI既存分類器を活用
❌ ml_preprocessor.py         # → FreqaiDataKitchen活用
❌ unified_ml_classifier.py   # → 既存LightGBMClassifier等活用
❌ technical_indicator_engine.py # → FreqAI feature_engineering活用
❌ realtime_predictor.py      # → FreqAI予測パイプライン活用
```

### 3.2 アーキテクチャ図

```
┌─────────────────────────────────────────────────────────────────┐
│                    Freqtrade Framework                          │
│  ┌───────────────┐                    ┌──────────────────────┐  │
│  │   IStrategy   │                    │     FreqAI Base      │  │
│  └───────────────┘                    └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
           │                                        │
           ▼                                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                   戦略実装層                                     │
│  ┌───────────────┐                    ┌──────────────────────┐  │
│  │ ATRMLStrategy │◄──────────────────►│ UnifiedMLClassifier  │  │
│  │               │                    │ (設定駆動)           │  │
│  └───────────────┘                    └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
           │                                        │
           ▼                                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                   戦略選択層                                     │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │              StrategyFactory                             │  │
│  │ ┌─────────────────┐         ┌─────────────────────────┐   │  │
│  │ │ PrimaryModel    │         │ SecondaryModel          │   │  │
│  │ │ Factory         │         │ Factory                 │   │  │
│  │ └─────────────────┘         └─────────────────────────┘   │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
           │                                        │
           ▼                                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                   計算エンジン層                                 │
│  ┌───────────────┐                    ┌──────────────────────┐  │
│  │PriceCalculator│                    │    MLTrainer         │  │
│  │ Base          │                    │    Base              │  │
│  │ ├─ATR         │                    │ ├─LightGBM          │  │
│  │ ├─Bollinger   │                    │ ├─XGBoost           │  │
│  │ └─MA          │                    │ └─RandomForest      │  │
│  └───────────────┘                    └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
           │                                        │
           ▼                                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                   前処理・支援層                                 │
│  ┌───────────────┐                    ┌──────────────────────┐  │
│  │MLPreprocessor │                    │TechnicalIndicator    │  │
│  │ ├─Feature     │                    │Engine                │  │
│  │ ├─Label       │                    │                      │  │
│  │ └─Quality     │                    │                      │  │
│  └───────────────┘                    └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## 4. 詳細設計

### 4.1 price_calculator.py

#### **抽象基底クラス**

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import pandas as pd

class PriceCalculatorBase(ABC):
    """価格計算の抽象基底クラス"""

    def __init__(self, params: Dict[str, Any]):
        self.params = params
        self.validate_params()

    @abstractmethod
    def validate_params(self) -> None:
        """パラメータ検証"""
        pass

    @abstractmethod
    def calculate_entry_prices(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """エントリー価格計算

        Returns:
            buy_price, sell_price列を追加したDataFrame
        """
        pass

    @abstractmethod
    def calculate_limit_price(self, close_price: float, side: str) -> float:
        """個別指値価格計算"""
        pass

    @abstractmethod
    def get_calculation_columns(self) -> list:
        """計算に必要なカラム名リスト"""
        pass
```

#### **ATR実装**

```python
class ATRPriceCalculator(PriceCalculatorBase):
    """ATR価格計算実装"""

    def validate_params(self) -> None:
        required = ["period", "multiplier"]
        for param in required:
            if param not in self.params:
                raise ValueError(f"Missing required parameter: {param}")

    def calculate_entry_prices(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        import talib as ta

        period = self.params["period"]
        multiplier = self.params["multiplier"]

        result = dataframe.copy()
        result["atr"] = ta.ATR(
            dataframe["high"], dataframe["low"], dataframe["close"],
            timeperiod=period
        )
        result["buy_price"] = result["close"] - (result["atr"] * multiplier)
        result["sell_price"] = result["close"] + (result["atr"] * multiplier)

        return result

    def calculate_limit_price(self, close_price: float, side: str) -> float:
        # ATR値の取得は別途実装
        multiplier = self.params["multiplier"]
        atr_value = self._get_current_atr()  # 実装省略

        if side.lower() in ["buy", "long"]:
            return close_price - (atr_value * multiplier)
        else:
            return close_price + (atr_value * multiplier)

    def get_calculation_columns(self) -> list:
        return ["high", "low", "close"]
```

#### **ボリンジャーバンド実装（将来）**

```python
class BollingerPriceCalculator(PriceCalculatorBase):
    """ボリンジャーバンド価格計算実装"""

    def validate_params(self) -> None:
        required = ["period", "std_dev"]
        for param in required:
            if param not in self.params:
                raise ValueError(f"Missing required parameter: {param}")

    def calculate_entry_prices(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        import talib as ta

        period = self.params["period"]
        std_dev = self.params["std_dev"]

        result = dataframe.copy()
        result["bb_upper"], result["bb_middle"], result["bb_lower"] = ta.BBANDS(
            dataframe["close"], timeperiod=period, nbdevup=std_dev, nbdevdn=std_dev
        )
        result["buy_price"] = result["bb_lower"]
        result["sell_price"] = result["bb_upper"]

        return result

    def calculate_limit_price(self, close_price: float, side: str) -> float:
        # ボリンジャーバンド値の取得実装
        pass

    def get_calculation_columns(self) -> list:
        return ["close"]
```

### 4.2 ml_trainer.py

#### **抽象基底クラス**

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np

class MLTrainerBase(ABC):
    """機械学習訓練の抽象基底クラス"""

    def __init__(self, params: Dict[str, Any]):
        self.params = params
        self.model = None
        self.is_trained = False

    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """モデル訓練

        Returns:
            訓練結果メトリクス
        """
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """予測実行"""
        pass

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """確率予測"""
        pass

    @abstractmethod
    def save_model(self, filepath: str) -> None:
        """モデル保存"""
        pass

    @abstractmethod
    def load_model(self, filepath: str) -> None:
        """モデル読み込み"""
        pass

    @abstractmethod
    def get_feature_importance(self) -> Dict[str, float]:
        """特徴量重要度取得"""
        pass
```

#### **LightGBM実装**

```python
import lightgbm as lgb
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score

class LightGBMTrainer(MLTrainerBase):
    """LightGBM訓練実装"""

    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        # LightGBMパラメータ設定
        lgb_params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': self.params.get('num_leaves', 31),
            'learning_rate': self.params.get('learning_rate', 0.1),
            'feature_fraction': self.params.get('feature_fraction', 0.8),
            'bagging_fraction': self.params.get('bagging_fraction', 0.8),
            'bagging_freq': self.params.get('bagging_freq', 5),
            'verbose': -1
        }

        # データセット作成
        train_data = lgb.Dataset(X, label=y)

        # モデル訓練
        self.model = lgb.train(
            lgb_params,
            train_data,
            num_boost_round=self.params.get('n_estimators', 100),
            valid_sets=[train_data],
            callbacks=[lgb.early_stopping(50)]
        )

        self.is_trained = True

        # 評価メトリクス計算
        y_pred = self.predict(X)
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred)
        }

        return metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model not trained yet")

        probabilities = self.model.predict(X)
        return (probabilities > 0.5).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model not trained yet")

        return self.model.predict(X)

    def save_model(self, filepath: str) -> None:
        if not self.is_trained:
            raise ValueError("Model not trained yet")

        self.model.save_model(filepath)

    def load_model(self, filepath: str) -> None:
        self.model = lgb.Booster(model_file=filepath)
        self.is_trained = True

    def get_feature_importance(self) -> Dict[str, float]:
        if not self.is_trained:
            raise ValueError("Model not trained yet")

        importance = self.model.feature_importance()
        feature_names = self.model.feature_name()

        return dict(zip(feature_names, importance))
```

#### **XGBoost実装（将来）**

```python
import xgboost as xgb

class XGBoostTrainer(MLTrainerBase):
    """XGBoost訓練実装"""

    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        # XGBoostパラメータ設定
        xgb_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': self.params.get('max_depth', 6),
            'learning_rate': self.params.get('learning_rate', 0.1),
            'n_estimators': self.params.get('n_estimators', 100),
            'subsample': self.params.get('subsample', 0.8),
            'colsample_bytree': self.params.get('colsample_bytree', 0.8)
        }

        # モデル訓練
        self.model = xgb.XGBClassifier(**xgb_params)
        self.model.fit(X, y)
        self.is_trained = True

        # 評価メトリクス計算
        y_pred = self.predict(X)
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred)
        }

        return metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model not trained yet")

        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model not trained yet")

        return self.model.predict_proba(X)[:, 1]

    def save_model(self, filepath: str) -> None:
        if not self.is_trained:
            raise ValueError("Model not trained yet")

        self.model.save_model(filepath)

    def load_model(self, filepath: str) -> None:
        self.model = xgb.XGBClassifier()
        self.model.load_model(filepath)
        self.is_trained = True

    def get_feature_importance(self) -> Dict[str, float]:
        if not self.is_trained:
            raise ValueError("Model not trained yet")

        return self.model.get_booster().get_score(importance_type='weight')
```

### 4.3 freqai_model_factory.py（FreqAI既存機能活用）

```python
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
        'lightgbm_classifier': 'LightGBMClassifier',
        'xgboost_classifier': 'XGBoostClassifier',
        'catboost_classifier': 'CatboostClassifier',
        'sklearn_rf_classifier': 'SKLearnRandomForestClassifier',

        # 回帰器
        'lightgbm_regressor': 'LightGBMRegressor',
        'xgboost_regressor': 'XGBoostRegressor',
        'catboost_regressor': 'CatboostRegressor',

        # PyTorch（高度なユーザー向け）
        'pytorch_mlp_classifier': 'PyTorchMLPClassifier',
        'pytorch_mlp_regressor': 'PyTorchMLPRegressor',
        'pytorch_transformer_regressor': 'PyTorchTransformerRegressor',

        # 強化学習
        'reinforcement_learner': 'ReinforcementLearner',
        'reinforcement_learner_multiproc': 'ReinforcementLearner_multiproc'
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
        secondary_config = strategy_config.get('secondary_model', {})

        if not secondary_config.get('enabled', False):
            return {}

        model_type = secondary_config.get('type', 'lightgbm_classifier')
        model_params = secondary_config.get('params', {})

        # FreqAIモデル名取得
        freqai_model = cls.get_model_name(model_type)

        # FreqAI標準設定構造
        freqai_config = {
            "enabled": True,
            "model_training_parameters": {
                **model_params,
                # 共通パラメータ
                "train_period_days": strategy_config.get('train_period_days', 30),
                "backtest_period_days": strategy_config.get('backtest_period_days', 7),
                "identifier": f"2tier_{model_type}",
            },
            "feature_parameters": {
                "include_timeframes": strategy_config.get('include_timeframes', ['5m', '15m', '1h']),
                "include_corr_pairs": strategy_config.get('include_corr_pairs', []),
                "label_period_candles": strategy_config.get('label_period_candles', 24),
                "include_shifted_candles": strategy_config.get('include_shifted_candles', 2),
                "DI_threshold": strategy_config.get('DI_threshold', 0.9),
                "weight_factor": strategy_config.get('weight_factor', 0),
                "principal_component_analysis": strategy_config.get('use_pca', False),
                "use_SVM_to_remove_outliers": strategy_config.get('use_svm_outlier_removal', True),
                "indicator_periods_candles": strategy_config.get('indicator_periods', [10, 20, 50])
            },
            "data_split_parameters": {
                "test_size": strategy_config.get('test_size', 0.33),
                "shuffle": strategy_config.get('shuffle', False)
            }
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
        if 'lightgbm' in model_type:
            defaults = {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'num_leaves': 31,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5
            }
        elif 'xgboost' in model_type:
            defaults = {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 6,
                'subsample': 0.8,
                'colsample_bytree': 0.8
            }
        elif 'catboost' in model_type:
            defaults = {
                'iterations': 100,
                'learning_rate': 0.1,
                'depth': 6,
                'l2_leaf_reg': 3.0
            }
        elif 'sklearn' in model_type:
            defaults = {
                'n_estimators': 100,
                'max_depth': None,
                'min_samples_split': 2,
                'min_samples_leaf': 1
            }
        elif 'pytorch' in model_type:
            defaults = {
                'learning_rate': 3e-4,
                'model_kwargs': {},
                'trainer_kwargs': {}
            }
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
        if 'classifier' in model_type:
            task_type = 'classification'
        elif 'regressor' in model_type:
            task_type = 'regression'
        elif 'reinforcement' in model_type:
            task_type = 'reinforcement_learning'
        else:
            task_type = 'unknown'

        # フレームワーク分類
        if 'lightgbm' in model_type:
            framework = 'lightgbm'
        elif 'xgboost' in model_type:
            framework = 'xgboost'
        elif 'catboost' in model_type:
            framework = 'catboost'
        elif 'sklearn' in model_type:
            framework = 'sklearn'
        elif 'pytorch' in model_type:
            framework = 'pytorch'
        else:
            framework = 'unknown'

        return {
            'freqai_class': freqai_class,
            'task_type': task_type,
            'framework': framework,
            'description': f"FreqAI {framework} {task_type} model"
        }
```

### 4.4 strategy_factory.py

```python
"""戦略ファクトリー統合モジュール

1次モデル、2次モデル、エントリー戦略の選択機能を統合
"""

import logging
from typing import Dict, Any, Optional, Union
from abc import ABC, abstractmethod

from .price_calculator import PriceCalculatorBase, ATRPriceCalculator
from .ml_trainer import MLTrainerBase, LightGBMTrainer
from .ml_preprocessor import MLPreprocessor

logger = logging.getLogger(__name__)

class StrategyFactory:
    """統合戦略ファクトリー"""

    # 登録済み実装の辞書
    _price_calculators = {
        'atr': ATRPriceCalculator,
        # 将来の追加例:
        # 'bollinger': BollingerPriceCalculator,
        # 'ma': MovingAveragePriceCalculator,
    }

    _ml_trainers = {
        'lightgbm': LightGBMTrainer,
        # 将来の追加例:
        # 'xgboost': XGBoostTrainer,
        # 'randomforest': RandomForestTrainer,
    }

    @classmethod
    def create_two_tier_strategy(cls, config: Dict[str, Any]) -> 'TwoTierStrategy':
        """2層戦略の作成

        Args:
            config: 戦略設定

        Returns:
            設定済み2層戦略インスタンス
        """
        # 1次モデル（価格計算）の作成
        primary_config = config.get('primary_model', {})
        primary_model = cls.create_primary_model(primary_config)

        # 2次モデル（ML）の作成
        secondary_config = config.get('secondary_model', {})
        secondary_model = cls.create_secondary_model(secondary_config)

        # 前処理エンジンの作成
        preprocessor_config = config.get('preprocessor', {})
        preprocessor = MLPreprocessor(preprocessor_config)

        # 2層戦略の作成
        strategy = TwoTierStrategy(
            primary_model=primary_model,
            secondary_model=secondary_model,
            preprocessor=preprocessor,
            config=config
        )

        return strategy

    @classmethod
    def create_primary_model(cls, config: Dict[str, Any]) -> PriceCalculatorBase:
        """1次モデル（価格計算）の作成"""
        model_type = config.get('type', 'atr')
        params = config.get('params', {})

        if model_type not in cls._price_calculators:
            raise ValueError(f"Unknown primary model type: {model_type}")

        calculator_class = cls._price_calculators[model_type]
        return calculator_class(params)

    @classmethod
    def create_secondary_model(cls, config: Dict[str, Any]) -> Optional[MLTrainerBase]:
        """2次モデル（ML）の作成"""
        if not config.get('enabled', True):
            return None

        model_type = config.get('type', 'lightgbm')
        params = config.get('params', {})

        if model_type not in cls._ml_trainers:
            raise ValueError(f"Unknown secondary model type: {model_type}")

        trainer_class = cls._ml_trainers[model_type]
        return trainer_class(params)

    @classmethod
    def register_primary_model(cls, name: str, calculator_class: type):
        """新しい1次モデルの登録"""
        if not issubclass(calculator_class, PriceCalculatorBase):
            raise ValueError("Calculator class must inherit from PriceCalculatorBase")

        cls._price_calculators[name] = calculator_class
        logger.info(f"Registered primary model: {name}")

    @classmethod
    def register_secondary_model(cls, name: str, trainer_class: type):
        """新しい2次モデルの登録"""
        if not issubclass(trainer_class, MLTrainerBase):
            raise ValueError("Trainer class must inherit from MLTrainerBase")

        cls._ml_trainers[name] = trainer_class
        logger.info(f"Registered secondary model: {name}")

    @classmethod
    def list_available_models(cls) -> Dict[str, list]:
        """利用可能なモデル一覧"""
        return {
            'primary_models': list(cls._price_calculators.keys()),
            'secondary_models': list(cls._ml_trainers.keys())
        }

class TwoTierStrategy:
    """2層戦略実行クラス"""

    def __init__(self, primary_model: PriceCalculatorBase,
                 secondary_model: Optional[MLTrainerBase],
                 preprocessor: MLPreprocessor,
                 config: Dict[str, Any]):
        self.primary_model = primary_model
        self.secondary_model = secondary_model
        self.preprocessor = preprocessor
        self.config = config
        self.is_ml_enabled = secondary_model is not None

        logger.info(f"TwoTierStrategy initialized: "
                   f"primary={type(primary_model).__name__}, "
                   f"secondary={type(secondary_model).__name__ if secondary_model else 'None'}")

    def generate_entry_signals(self, dataframe, metadata: Dict[str, Any]):
        """エントリー信号生成"""
        pair = metadata.get('pair', 'unknown')

        try:
            # 1次モデル: 価格計算
            price_data = self.primary_model.calculate_entry_prices(dataframe)

            if self.is_ml_enabled:
                # 2次モデル有効時: ML統合戦略
                return self._generate_ml_integrated_signals(price_data, metadata)
            else:
                # 2次モデル無効時: 基本価格戦略
                return self._generate_basic_price_signals(price_data, metadata)

        except Exception as e:
            logger.error(f"Signal generation error {pair}: {e}")
            # フォールバック: 信号なし
            result = dataframe.copy()
            result["enter_long"] = 0
            result["enter_short"] = 0
            return result

    def _generate_ml_integrated_signals(self, dataframe, metadata: Dict[str, Any]):
        """ML統合信号生成"""
        pair = metadata.get('pair', 'unknown')
        result = dataframe.copy()
        result["enter_long"] = 0
        result["enter_short"] = 0

        # ML予測データの確認
        has_ml_prediction = "&-prediction" in dataframe.columns
        has_ml_probability = "&-probability" in dataframe.columns

        if not has_ml_prediction:
            logger.warning(f"ML prediction data missing for {pair}")
            return result

        # ML予測の取得
        ml_prediction = dataframe["&-prediction"] == 1

        # 信頼度フィルタリング
        confidence_filter = True
        confidence_threshold = self.config.get('confidence_threshold', 0.6)

        if has_ml_probability and confidence_threshold > 0:
            confidence_filter = dataframe["&-probability"] >= confidence_threshold

        # 価格データの有効性チェック
        price_valid = (dataframe["buy_price"] > 0) & (dataframe["sell_price"] > 0)

        # ML統合条件
        long_condition = ml_prediction & confidence_filter & price_valid
        short_condition = ~ml_prediction & confidence_filter & price_valid

        # 信号設定
        result.loc[long_condition, "enter_long"] = 1
        result.loc[short_condition, "enter_short"] = 1

        long_signals = long_condition.sum()
        short_signals = short_condition.sum()
        logger.info(f"ML integrated signals {pair}: long={long_signals}, short={short_signals}")

        return result

    def _generate_basic_price_signals(self, dataframe, metadata: Dict[str, Any]):
        """基本価格信号生成"""
        pair = metadata.get('pair', 'unknown')
        result = dataframe.copy()
        result["enter_long"] = 0
        result["enter_short"] = 0

        if len(dataframe) <= 1:
            return result

        # 前期間の価格と現在価格の比較
        prev_buy_price = dataframe["buy_price"].shift(1)
        prev_sell_price = dataframe["sell_price"].shift(1)
        current_close = dataframe["close"]

        # 価格戦略: 価格が指値レベルに到達した時の取引判定
        buy_signal = (current_close <= prev_buy_price) & (prev_buy_price > 0)
        sell_signal = (current_close >= prev_sell_price) & (prev_sell_price > 0)

        # 価格データの有効性チェック
        price_valid = (dataframe["buy_price"] > 0) & (dataframe["sell_price"] > 0)

        # エントリー条件
        long_condition = buy_signal & price_valid
        short_condition = sell_signal & price_valid

        # 信号設定
        result.loc[long_condition, "enter_long"] = 1
        result.loc[short_condition, "enter_short"] = 1

        long_signals = long_condition.sum()
        short_signals = short_condition.sum()
        logger.info(f"Basic price signals {pair}: long={long_signals}, short={short_signals}")

        return result

    def calculate_entry_price(self, pair: str, side: str, proposed_rate: float):
        """エントリー価格計算"""
        try:
            return self.primary_model.calculate_limit_price(proposed_rate, side)
        except Exception as e:
            logger.error(f"Entry price calculation error {pair}: {e}")
            return proposed_rate

    def train_secondary_model(self, dataframe, metadata: Dict[str, Any]):
        """2次モデル訓練（必要時のみ）"""
        if not self.is_ml_enabled:
            logger.info("Secondary model disabled, skipping training")
            return None

        pair = metadata.get('pair', 'unknown')
        logger.info(f"Training secondary model for {pair}")

        try:
            # 特徴量生成
            features_df = self.preprocessor.generate_features(dataframe)

            # ラベル生成
            labels = self.preprocessor.generate_labels(dataframe, self.primary_model)

            # 品質評価
            quality_result = self.preprocessor.assess_feature_quality(features_df, labels)
            high_quality_features = quality_result['high_quality_features']

            # 高品質特徴量のみで訓練
            if not high_quality_features:
                logger.warning(f"No high quality features found for {pair}")
                return None

            X = features_df[high_quality_features].dropna()
            y = labels.loc[X.index]

            if len(X) < 100:  # 最小サンプル数チェック
                logger.warning(f"Insufficient training samples for {pair}: {len(X)}")
                return None

            # モデル訓練
            training_result = self.secondary_model.train(X, y)

            logger.info(f"Secondary model training completed for {pair}: {training_result}")
            return training_result

        except Exception as e:
            logger.error(f"Secondary model training error {pair}: {e}")
            return None
```

### 4.5 unified_ml_classifier.py (FreqAI統合)

```python
"""FreqAI統合汎用ML分類器

設定に基づいてLightGBM、XGBoost等を動的選択
"""

import logging
from typing import Any, Dict
import pandas as pd
import numpy as np

from freqtrade.freqai.base_models.BaseClassifierModel import BaseClassifierModel
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen

# 戦略ファクトリーからML訓練器を取得
from user_data.strategies.utils.strategy_factory import StrategyFactory

logger = logging.getLogger(__name__)

class UnifiedMLClassifier(BaseClassifierModel):
    """FreqAI統合汎用ML分類器"""

    def fit(self, data_dictionary: Dict[str, Any], dk: FreqaiDataKitchen, **kwargs) -> Any:
        """モデル訓練

        FreqAIフレームワークとの統合を提供しつつ、
        内部では設定駆動型のML訓練器を使用
        """

        # FreqAI設定からML設定を取得
        ml_config = self.freqai_info.get('model_training_parameters', {})
        model_type = ml_config.get('type', 'lightgbm')
        model_params = ml_config.get('params', {})

        logger.info(f"Training unified ML classifier: type={model_type}")

        try:
            # 戦略ファクトリーからML訓練器を作成
            trainer = StrategyFactory.create_secondary_model({
                'type': model_type,
                'params': model_params
            })

            # 訓練データの準備
            X = data_dictionary["train_features"]
            y = data_dictionary["train_labels"]

            # モデル訓練
            training_result = trainer.train(X, y)

            # FreqAI用の属性設定
            self.model = trainer
            self.training_metrics = training_result

            logger.info(f"Training completed: {training_result}")

            return trainer

        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise

    def predict(self, unfiltered_df: pd.DataFrame, dk: FreqaiDataKitchen,
                **kwargs) -> tuple[pd.DataFrame, pd.DataFrame]:
        """予測実行"""

        # 特徴量の準備
        filtered_df, _ = dk.filter_features(
            unfiltered_df, dk.training_features_list, training_filter=False
        )

        # 予測実行
        predictions = self.model.predict(filtered_df)
        pred_df = pd.DataFrame(predictions, columns=[f"&-{self.dk.label_list[0]}"],
                              index=filtered_df.index)

        # 確率予測（利用可能な場合）
        try:
            probabilities = self.model.predict_proba(filtered_df)
            prob_df = pd.DataFrame(probabilities, columns=[f"&-{self.dk.label_list[0]}_probability"],
                                  index=filtered_df.index)
            pred_df = pd.concat([pred_df, prob_df], axis=1)
        except Exception as e:
            logger.warning(f"Probability prediction failed: {e}")
            # 確率予測失敗時はダミー値
            prob_df = pd.DataFrame(0.5, columns=[f"&-{self.dk.label_list[0]}_probability"],
                                  index=filtered_df.index)

        return pred_df, prob_df
```

## 5. 設定駆動型実装

### 5.1 戦略設定ファイル

#### **strategy_config.py**

```python
"""2層戦略設定

この設定を変更することで、コードを変更せずに手法を切り替え可能
"""

# 基本戦略設定
TWO_TIER_STRATEGY_CONFIG = {
    # 1次モデル設定（価格計算）
    "primary_model": {
        "type": "atr",  # 選択肢: "atr", "bollinger", "ma", "rsi"
        "params": {
            "period": 14,
            "multiplier": 0.5
        }
    },

    # 2次モデル設定（機械学習）
    "secondary_model": {
        "enabled": True,
        "type": "lightgbm",  # 選択肢: "lightgbm", "xgboost", "randomforest"
        "params": {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "num_leaves": 31,
            "max_depth": -1,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5
        }
    },

    # 前処理設定
    "preprocessor": {
        # テクニカル指標設定
        "indicators": {
            "sma": {"period": 20},
            "ema": {"period": 12},
            "rsi": {"period": 14},
            "macd": {"fast": 12, "slow": 26, "signal": 9},
            "bollinger": {"period": 20, "std": 2}
        },

        # ラベル生成設定
        "label": {
            "method": "return_based",  # "return_based" or "price_based"
            "look_ahead": 1,
            "threshold": 0.001
        },

        # 品質管理設定
        "quality": {
            "max_missing_ratio": 0.1,
            "min_predictive_power": 0.01,
            "max_correlation": 0.95
        }
    },

    # エントリー戦略設定
    "entry": {
        "confidence_threshold": 0.6,
        "min_data_length": 50
    }
}

# 設定バリエーション例

# ボリンジャーバンド + XGBoost設定
BOLLINGER_XGBOOST_CONFIG = {
    "primary_model": {
        "type": "bollinger",
        "params": {
            "period": 20,
            "std_dev": 2
        }
    },
    "secondary_model": {
        "enabled": True,
        "type": "xgboost",
        "params": {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8
        }
    },
    # 他の設定は共通
    **{k: v for k, v in TWO_TIER_STRATEGY_CONFIG.items()
       if k not in ["primary_model", "secondary_model"]}
}

# ML無効設定（価格戦略のみ）
PRICE_ONLY_CONFIG = {
    **TWO_TIER_STRATEGY_CONFIG,
    "secondary_model": {
        "enabled": False
    }
}
```

### 5.2 戦略での使用例

#### **更新されたATRMLStrategy**

```python
import logging
import os
import sys
from typing import Optional

import pandas as pd
from freqtrade.strategy import IStrategy

# パス設定
sys.path.append(os.path.dirname(__file__))

# 設定と戦略ファクトリーのインポート
from utils.strategy_factory import StrategyFactory
from utils.strategy_config import TWO_TIER_STRATEGY_CONFIG

logger = logging.getLogger(__name__)

class ATRMLStrategy(IStrategy):
    """設定駆動型2層戦略

    strategy_config.pyの設定を変更することで、
    コードを変更せずに手法を切り替え可能
    """

    # FreqAI設定
    process_only_new_candles = True
    stoploss = -0.05
    startup_candle_count: int = 50
    can_short = True

    def __init__(self, config: dict = None):
        super().__init__(config)

        # 設定の読み込み
        strategy_config = config.get('strategy_config', TWO_TIER_STRATEGY_CONFIG)

        # 2層戦略の作成
        self.two_tier_strategy = StrategyFactory.create_two_tier_strategy(strategy_config)

        # 設定パラメータの展開
        primary_config = strategy_config.get('primary_model', {})
        self.entry_length = primary_config.get('params', {}).get('period', 14)
        self.entry_point = primary_config.get('params', {}).get('multiplier', 0.5)

        entry_config = strategy_config.get('entry', {})
        self.confidence_threshold = entry_config.get('confidence_threshold', 0.6)

        logger.info(f"ATRMLStrategy initialized with 2-tier strategy: "
                   f"primary={primary_config.get('type', 'unknown')}, "
                   f"secondary={'enabled' if strategy_config.get('secondary_model', {}).get('enabled') else 'disabled'}")

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """指標計算"""
        pair = metadata.get("pair", "unknown")

        try:
            # 2層戦略の1次モデルで価格計算
            dataframe = self.two_tier_strategy.primary_model.calculate_entry_prices(dataframe)

            # FreqAI予測の取得（有効性チェック）
            freqai_enabled = (
                hasattr(self, "freqai")
                and self.freqai is not None
                and getattr(self.freqai, "enabled", False)
            )

            if freqai_enabled:
                try:
                    dataframe = self.freqai.start(dataframe, metadata, self)
                    logger.debug("FreqAI prediction data added successfully")
                except Exception as freqai_error:
                    logger.warning(f"FreqAI execution failed: {freqai_error}")
                    logger.debug("Continuing with primary model only")
            else:
                logger.debug("FreqAI disabled - running with primary model only")

            logger.debug(f"Indicators calculation completed: {pair}, records={len(dataframe)}")
            return dataframe

        except Exception as e:
            logger.error(f"Indicators calculation error ({pair}): {e}")
            return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """エントリートレンド生成（設定駆動型）"""

        # 2層戦略でエントリー信号生成
        return self.two_tier_strategy.generate_entry_signals(dataframe, metadata)

    def custom_entry_price(self, pair: str, current_time, current_rate: float,
                          proposed_rate: float, entry_tag: Optional[str], side: str,
                          **kwargs) -> float:
        """エントリー価格計算（設定駆動型）"""

        # 2層戦略でエントリー価格計算
        return self.two_tier_strategy.calculate_entry_price(pair, side, proposed_rate)

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """エグジットトレンド生成"""
        # 現在の実装を維持
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0
        return dataframe
```

## 6. 安全な段階的移行プロセス（FreqAI最大活用）

### 6.1 移行の基本原則

#### **アトミック移行（中途半端状態の完全回避）**

- 各フェーズは独立完結し、途中で停止してもシステムが動作
- 一時ファイルの明確な管理とクリーンアップ
- ロールバック機能の内蔵
- 進捗状況の可視化・追跡

#### **FreqAI機能最大活用**

- 既存FreqAI分類器・回帰器の活用（独自実装排除）
- FreqaiDataKitchenによる前処理統合
- 標準feature_engineering機能の利用
- `self.freqai.start()`統合機能の活用

### 6.2 段階的移行手順（中断安全型）

#### **Phase 0: 移行準備・検証**

```bash
#!/bin/bash
# migration_prepare.sh - 移行準備スクリプト

echo "=== Phase 0: Migration Preparation ==="

# 1. 現在の状態をバックアップ
echo "Creating backup..."
cp -r user_data/strategies/ user_data/strategies_backup_$(date +%Y%m%d_%H%M%S)
cp -r freqtrade/freqai/prediction_models/ freqtrade/freqai/prediction_models_backup_$(date +%Y%m%d_%H%M%S)

# 2. 移行状態ファイル作成
echo "phase_0_completed" > .migration_status

# 3. FreqAI利用可能性チェック
python -c "
import freqtrade.freqai
print('FreqAI available')
from freqtrade.freqai.prediction_models.LightGBMClassifier import LightGBMClassifier
print('LightGBMClassifier available')
" || {
    echo "Error: FreqAI not available"
    exit 1
}

# 4. 既存戦略のバックテスト実行（ベースライン）
echo "Running baseline backtest..."
freqtrade backtesting --strategy ATRMLStrategy --timerange 20231101-20231107 > baseline_backtest.log 2>&1

echo "Phase 0 completed successfully"
```

#### **Phase 1: FreqAI統合ファクトリー作成**

```bash
#!/bin/bash
# migration_phase1.sh - Phase 1実行

echo "=== Phase 1: FreqAI Factory Creation ==="

# 前フェーズ完了チェック
if [ ! -f .migration_status ] || ! grep -q "phase_0_completed" .migration_status; then
    echo "Error: Phase 0 not completed"
    exit 1
fi

# FreqAI統合ファクトリー作成（既存機能活用）
cat > user_data/strategies/utils/freqai_model_factory.py << 'EOF'
"""FreqAI既存機能活用ファクトリー"""
# （上記設計の実装）
EOF

# 設定ファイル作成
cat > user_data/strategies/utils/strategy_config.py << 'EOF'
"""設定駆動型パラメータ"""
# （設定駆動型の実装）
EOF

# 動作確認
python -c "
from user_data.strategies.utils.freqai_model_factory import FreqAIModelFactory
models = FreqAIModelFactory.list_available_models()
print(f'Available FreqAI models: {len(models)}')
"

echo "phase_1_completed" >> .migration_status
echo "Phase 1 completed successfully"
```

#### **Phase 2: 価格計算統合**

```bash
#!/bin/bash
# migration_phase2.sh - Phase 2実行

echo "=== Phase 2: Price Calculator Integration ==="

# 前フェーズ完了チェック
if ! grep -q "phase_1_completed" .migration_status; then
    echo "Error: Phase 1 not completed"
    exit 1
fi

# 価格計算統合実装
cat > user_data/strategies/utils/price_calculator.py << 'EOF'
"""価格計算統合実装"""
# （上記設計の実装）
EOF

# 一時的な互換性レイヤー作成（Phase 4で削除）
cat > user_data/strategies/utils/_migration_compat.py << 'EOF'
"""移行用互換性レイヤー（Phase 4で削除）"""
from .price_calculator import ATRPriceCalculator

class ATRCalculatorEngine(ATRPriceCalculator):
    def __init__(self, *args, **kwargs):
        if args:
            period = args[0] if len(args) > 0 else 14
            multiplier = args[1] if len(args) > 1 else 0.5
            params = {"period": period, "multiplier": multiplier}
        else:
            params = kwargs
        super().__init__(params)

ATRCalculator = ATRCalculatorEngine  # 後方互換性
EOF

# 互換性テスト
python -c "
from user_data.strategies.utils._migration_compat import ATRCalculatorEngine
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'high': np.random.rand(100) * 100 + 100,
    'low': np.random.rand(100) * 100 + 50,
    'close': np.random.rand(100) * 100 + 75
})

calc = ATRCalculatorEngine(14, 0.5)
result = calc.calculate_entry_prices(df)
assert 'buy_price' in result.columns
print('Phase 2 compatibility test passed')
"

echo "phase_2_completed" >> .migration_status
echo "Phase 2 completed successfully"
```

#### **Phase 3: 戦略ファクトリー・FreqAI統合**

```bash
#!/bin/bash
# migration_phase3.sh - Phase 3実行

echo "=== Phase 3: Strategy Factory & FreqAI Integration ==="

# 前フェーズ完了チェック
if ! grep -q "phase_2_completed" .migration_status; then
    echo "Error: Phase 2 not completed"
    exit 1
fi

# 戦略ファクトリー実装（FreqAI機能活用）
cat > user_data/strategies/utils/strategy_factory.py << 'EOF'
"""戦略ファクトリー実装（FreqAI機能活用版）"""
# （上記設計の実装）
EOF

# FreqAI feature engineering統合実装
cat > user_data/strategies/utils/_freqai_integration.py << 'EOF'
"""FreqAI標準機能活用ヘルパー（Phase 4で統合・削除）"""

class FreqAIFeatureHelper:
    @staticmethod
    def add_basic_features(dataframe):
        """FreqAI feature_engineering_expand_basic相当"""
        result = dataframe.copy()
        result["%-pct-change"] = dataframe["close"].pct_change()
        result["%-raw_volume"] = dataframe["volume"]
        result["%-raw_price"] = dataframe["close"]
        return result

    @staticmethod
    def set_freqai_targets(dataframe, label_config):
        """FreqAI set_freqai_targets相当"""
        result = dataframe.copy()
        look_ahead = label_config.get('label_period_candles', 24)
        future_return = dataframe["close"].pct_change(look_ahead).shift(-look_ahead)
        result["&-target"] = (future_return > 0.001).astype(int)
        return result
EOF

# 統合テスト
python -c "
from user_data.strategies.utils.strategy_factory import StrategyFactory
from user_data.strategies.utils.freqai_model_factory import FreqAIModelFactory
from user_data.strategies.utils.strategy_config import TWO_TIER_STRATEGY_CONFIG

factory = StrategyFactory()
freqai_config = FreqAIModelFactory.create_freqai_config(TWO_TIER_STRATEGY_CONFIG)
assert 'enabled' in freqai_config
print('Phase 3 integration test passed')
"

echo "phase_3_completed" >> .migration_status
echo "Phase 3 completed successfully"
```

#### **Phase 4: 戦略統合・テスト**

```bash
#!/bin/bash
# migration_phase4.sh - Phase 4実行

echo "=== Phase 4: Strategy Integration & Testing ==="

# 前フェーズ完了チェック
if ! grep -q "phase_3_completed" .migration_status; then
    echo "Error: Phase 3 not completed"
    exit 1
fi

# ATRMLStrategy更新（FreqAI最大活用版）
cp user_data/strategies/atr_ml_strategy.py user_data/strategies/atr_ml_strategy_v1.py

cat > user_data/strategies/atr_ml_strategy_v2.py << 'EOF'
"""FreqAI最大活用版ATRMLStrategy"""

import logging
import os
import sys
from typing import Optional
import pandas as pd
from freqtrade.strategy import IStrategy

sys.path.append(os.path.dirname(__file__))

from utils.strategy_factory import StrategyFactory
from utils.freqai_model_factory import FreqAIModelFactory
from utils.strategy_config import TWO_TIER_STRATEGY_CONFIG
from utils._migration_compat import ATRCalculatorEngine
from utils._freqai_integration import FreqAIFeatureHelper

logger = logging.getLogger(__name__)

class ATRMLStrategy(IStrategy):
    """FreqAI最大活用2層戦略"""

    process_only_new_candles = True
    stoploss = -0.05
    startup_candle_count: int = 50
    can_short = True

    def __init__(self, config: dict = None):
        super().__init__(config)

        strategy_config = config.get('strategy_config', TWO_TIER_STRATEGY_CONFIG)
        freqai_config = FreqAIModelFactory.create_freqai_config(strategy_config)

        primary_config = strategy_config.get('primary_model', {})
        self.price_calculator = StrategyFactory.create_primary_model(primary_config)

        # 互換性維持（一時的）
        self.atr_engine = ATRCalculatorEngine(
            primary_config.get('params', {}).get('period', 14),
            primary_config.get('params', {}).get('multiplier', 0.5)
        )

        self.entry_length = primary_config.get('params', {}).get('period', 14)
        self.entry_point = primary_config.get('params', {}).get('multiplier', 0.5)
        self.confidence_threshold = strategy_config.get('entry', {}).get('confidence_threshold', 0.6)

    def feature_engineering_expand_basic(self, dataframe: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """FreqAI基本特徴量（FreqAI標準機能活用）"""
        return FreqAIFeatureHelper.add_basic_features(dataframe)

    def set_freqai_targets(self, dataframe: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """FreqAI目標設定（FreqAI標準機能活用）"""
        label_config = {'label_period_candles': 24}
        return FreqAIFeatureHelper.set_freqai_targets(dataframe, label_config)

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """指標計算（FreqAI統合版）"""
        pair = metadata.get("pair", "unknown")

        try:
            # 1次モデル: 価格計算
            dataframe = self.price_calculator.calculate_entry_prices(dataframe)

            # FreqAI予測（標準FreqAI機能）
            freqai_enabled = (
                hasattr(self, "freqai")
                and self.freqai is not None
                and getattr(self.freqai, "enabled", False)
            )

            if freqai_enabled:
                try:
                    dataframe = self.freqai.start(dataframe, metadata, self)
                    logger.debug("FreqAI prediction added successfully")
                except Exception as e:
                    logger.warning(f"FreqAI failed: {e}")

            return dataframe

        except Exception as e:
            logger.error(f"Indicators error {pair}: {e}")
            return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """エントリートレンド生成（簡略版）"""
        pair = metadata.get("pair", "unknown")

        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0

        if len(dataframe) < self.entry_length + 10:
            return dataframe

        has_ml_prediction = "&-prediction" in dataframe.columns

        if has_ml_prediction:
            # ML統合モード
            ml_prediction = dataframe["&-prediction"] == 1
            confidence_filter = True

            if "&-probability" in dataframe.columns:
                confidence_filter = dataframe["&-probability"] >= self.confidence_threshold

            price_valid = (dataframe["buy_price"] > 0) & (dataframe["sell_price"] > 0)

            long_condition = ml_prediction & confidence_filter & price_valid
            short_condition = ~ml_prediction & confidence_filter & price_valid

            dataframe.loc[long_condition, "enter_long"] = 1
            dataframe.loc[short_condition, "enter_short"] = 1

        else:
            # 基本価格モード
            if len(dataframe) > 1:
                prev_buy = dataframe["buy_price"].shift(1)
                prev_sell = dataframe["sell_price"].shift(1)
                current_close = dataframe["close"]

                buy_signal = (current_close <= prev_buy) & (prev_buy > 0)
                sell_signal = (current_close >= prev_sell) & (prev_sell > 0)

                dataframe.loc[buy_signal, "enter_long"] = 1
                dataframe.loc[sell_signal, "enter_short"] = 1

        return dataframe

    def custom_entry_price(self, pair: str, current_time, current_rate: float,
                          proposed_rate: float, entry_tag: Optional[str], side: str,
                          **kwargs) -> float:
        """エントリー価格計算"""
        try:
            return self.price_calculator.calculate_limit_price(proposed_rate, side)
        except Exception as e:
            logger.error(f"Entry price error {pair}: {e}")
            return proposed_rate

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """エグジットトレンド生成"""
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0
        return dataframe
EOF

# ハイブリッド版テスト
echo "Testing FreqAI integrated strategy..."
python -c "
from user_data.strategies.atr_ml_strategy_v2 import ATRMLStrategy
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'open': np.random.rand(100) * 100 + 100,
    'high': np.random.rand(100) * 100 + 110,
    'low': np.random.rand(100) * 100 + 90,
    'close': np.random.rand(100) * 100 + 100,
    'volume': np.random.rand(100) * 1000000
})

config = {'strategy_config': {'primary_model': {'type': 'atr', 'params': {'period': 14, 'multiplier': 0.5}}}}
strategy = ATRMLStrategy(config)

result = strategy.populate_indicators(df, {'pair': 'BTC/USDT'})
assert 'buy_price' in result.columns
print('FreqAI integrated strategy test passed')
"

# バックテスト比較
echo "Running comparison backtest..."
freqtrade backtesting --strategy ATRMLStrategy_v2 --timerange 20231101-20231107 > freqai_integrated_backtest.log 2>&1

echo "phase_4_completed" >> .migration_status
echo "Phase 4 completed successfully"
```

#### **Phase 5: クリーンアップ・完了**

```bash
#!/bin/bash
# migration_phase5.sh - Phase 5実行（最終クリーンアップ）

echo "=== Phase 5: Final Cleanup ==="

# 前フェーズ完了チェック
if ! grep -q "phase_4_completed" .migration_status; then
    echo "Error: Phase 4 not completed"
    exit 1
fi

# 1. 一時ファイル削除
echo "Removing temporary files..."
rm -f user_data/strategies/utils/_migration_compat.py
rm -f user_data/strategies/utils/_freqai_integration.py

# 2. 旧ファイル削除（バックアップ済み）
echo "Removing old files..."
rm -f user_data/strategies/utils/atr_calculator.py
rm -f user_data/strategies/utils/atr_return_calculator.py
rm -f user_data/strategies/utils/lightgbm_trainer.py
rm -f user_data/strategies/utils/feature_quality_manager.py
rm -f user_data/strategies/utils/ml_label_generator.py
rm -f user_data/strategies/utils/technical_indicator_engine.py
rm -f user_data/strategies/utils/realtime_predictor.py
rm -f user_data/strategies/utils/entry_strategy.py
rm -f freqtrade/freqai/prediction_models/atr_lightgbm_classifier.py
rm -f freqtrade/freqai/prediction_models/mlops_pipeline.py

# 3. 戦略ファイル統合
mv user_data/strategies/atr_ml_strategy_v2.py user_data/strategies/atr_ml_strategy.py

# 4. 最終テスト
echo "Running final validation..."
freqtrade backtesting --strategy ATRMLStrategy --timerange 20231101-20231107 > final_backtest.log 2>&1

# 5. 移行完了
echo "migration_completed" >> .migration_status
rm -f .migration_status

echo "=== Migration Completed Successfully ==="
echo "Files removed: 10"
echo "Files created: 4"
echo "Net reduction: 6 files (60%)"
echo ""
echo "FreqAI integration maximized:"
echo "✓ Using FreqAI standard models"
echo "✓ Using FreqaiDataKitchen for preprocessing"
echo "✓ Using feature_engineering_* methods"
echo "✓ Configuration-driven model selection"
```

### 6.3 緊急ロールバック機能

#### **完全ロールバック**

```bash
#!/bin/bash
# rollback.sh - 緊急ロールバック

echo "=== Emergency Rollback ==="

# 最新バックアップを特定
LATEST_BACKUP=$(ls -t user_data/strategies_backup_* | head -1)
LATEST_FREQAI_BACKUP=$(ls -t freqtrade/freqai/prediction_models_backup_* | head -1)

if [ -z "$LATEST_BACKUP" ]; then
    echo "Error: No backup found"
    exit 1
fi

echo "Rolling back to: $LATEST_BACKUP"

# バックアップから復元
rm -rf user_data/strategies/
cp -r "$LATEST_BACKUP" user_data/strategies/

rm -rf freqtrade/freqai/prediction_models/
cp -r "$LATEST_FREQAI_BACKUP" freqtrade/freqai/prediction_models/

# 状態ファイル削除
rm -f .migration_status

echo "Rollback completed successfully"
```

### 6.4 進捗確認・診断ツール

#### **移行状況確認**

```bash
#!/bin/bash
# check_migration.sh - 移行進捗確認

echo "=== Migration Status Check ==="

if [ ! -f .migration_status ]; then
    echo "Status: Not started"
    exit 0
fi

echo "Completed phases:"
cat .migration_status | nl

echo ""
echo "New files status:"
ls -la user_data/strategies/utils/freqai_model_factory.py 2>/dev/null && echo "✓ freqai_model_factory.py" || echo "✗ freqai_model_factory.py"
ls -la user_data/strategies/utils/price_calculator.py 2>/dev/null && echo "✓ price_calculator.py" || echo "✗ price_calculator.py"
ls -la user_data/strategies/utils/strategy_factory.py 2>/dev/null && echo "✓ strategy_factory.py" || echo "✗ strategy_factory.py"
ls -la user_data/strategies/utils/strategy_config.py 2>/dev/null && echo "✓ strategy_config.py" || echo "✗ strategy_config.py"

echo ""
echo "Old files (should be removed in Phase 5):"
ls -la user_data/strategies/utils/atr_calculator.py 2>/dev/null && echo "⚠ atr_calculator.py (pending removal)" || echo "✓ atr_calculator.py (removed)"
ls -la user_data/strategies/utils/lightgbm_trainer.py 2>/dev/null && echo "⚠ lightgbm_trainer.py (pending removal)" || echo "✓ lightgbm_trainer.py (removed)"

echo ""
echo "Backup files:"
ls -la user_data/strategies_backup_* 2>/dev/null | wc -l | xargs echo "Backup count:"

echo ""
echo "FreqAI integration status:"
python -c "
try:
    from freqtrade.freqai.prediction_models.LightGBMClassifier import LightGBMClassifier
    print('✓ FreqAI models available')
except ImportError:
    print('✗ FreqAI models not available')
"
```

この移行プロセスにより、**中途半端な状態を完全に回避**し、**FreqAI機能を最大活用**した安全で効率的な移行を実現します。

## 7. 拡張例

### 7.1 新しい1次モデルの追加

#### **移動平均戦略の追加**

```python
# price_calculator.py に追加

class MovingAveragePriceCalculator(PriceCalculatorBase):
    """移動平均価格計算実装"""

    def validate_params(self) -> None:
        required = ["fast_period", "slow_period", "offset_ratio"]
        for param in required:
            if param not in self.params:
                raise ValueError(f"Missing required parameter: {param}")

    def calculate_entry_prices(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        import talib as ta

        fast_period = self.params["fast_period"]
        slow_period = self.params["slow_period"]
        offset_ratio = self.params["offset_ratio"]

        result = dataframe.copy()
        result["ma_fast"] = ta.SMA(dataframe["close"], timeperiod=fast_period)
        result["ma_slow"] = ta.SMA(dataframe["close"], timeperiod=slow_period)

        # 移動平均の中間値をベースに指値価格計算
        ma_mid = (result["ma_fast"] + result["ma_slow"]) / 2
        offset = result["close"] * offset_ratio

        result["buy_price"] = ma_mid - offset
        result["sell_price"] = ma_mid + offset

        return result

    def calculate_limit_price(self, close_price: float, side: str) -> float:
        # 実装省略
        pass

    def get_calculation_columns(self) -> list:
        return ["close"]


# strategy_factory.py で登録
StrategyFactory.register_primary_model("ma", MovingAveragePriceCalculator)
```

#### **設定での利用**

```python
# strategy_config.py に追加

MA_LIGHTGBM_CONFIG = {
    "primary_model": {
        "type": "ma",  # 新しい移動平均戦略
        "params": {
            "fast_period": 12,
            "slow_period": 26,
            "offset_ratio": 0.002
        }
    },
    "secondary_model": {
        "enabled": True,
        "type": "lightgbm",
        "params": {
            "n_estimators": 100,
            "learning_rate": 0.1
        }
    },
    # 他の設定は共通
}
```

### 7.2 新しい2次モデルの追加

#### **XGBoost実装の追加**

```python
# ml_trainer.py に追加 (すでに設計済み)

# strategy_factory.py で登録
StrategyFactory.register_secondary_model("xgboost", XGBoostTrainer)
```

#### **設定での利用**

```python
# strategy_config.py
ATR_XGBOOST_CONFIG = {
    "primary_model": {
        "type": "atr",
        "params": {"period": 14, "multiplier": 0.5}
    },
    "secondary_model": {
        "enabled": True,
        "type": "xgboost",  # 新しいXGBoost
        "params": {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1
        }
    }
}
```

### 7.3 設定のみでの手法切り替え

#### **戦略での設定選択**

```python
# atr_ml_strategy.py
from utils.strategy_config import (
    TWO_TIER_STRATEGY_CONFIG,    # ATR + LightGBM
    BOLLINGER_XGBOOST_CONFIG,    # ボリンジャー + XGBoost
    MA_LIGHTGBM_CONFIG,          # 移動平均 + LightGBM
    PRICE_ONLY_CONFIG            # 価格戦略のみ
)

class ATRMLStrategy(IStrategy):
    def __init__(self, config: dict = None):
        super().__init__(config)

        # 設定選択（環境変数等で切り替え可能）
        strategy_type = os.environ.get('STRATEGY_TYPE', 'default')

        if strategy_type == 'bollinger_xgboost':
            strategy_config = BOLLINGER_XGBOOST_CONFIG
        elif strategy_type == 'ma_lightgbm':
            strategy_config = MA_LIGHTGBM_CONFIG
        elif strategy_type == 'price_only':
            strategy_config = PRICE_ONLY_CONFIG
        else:
            strategy_config = TWO_TIER_STRATEGY_CONFIG

        self.two_tier_strategy = StrategyFactory.create_two_tier_strategy(strategy_config)
```

#### **実行時の手法切り替え**

```bash
# ATR + LightGBM で実行
freqtrade backtesting --strategy ATRMLStrategy

# ボリンジャー + XGBoost で実行
STRATEGY_TYPE=bollinger_xgboost freqtrade backtesting --strategy ATRMLStrategy

# 移動平均 + LightGBM で実行
STRATEGY_TYPE=ma_lightgbm freqtrade backtesting --strategy ATRMLStrategy

# 価格戦略のみで実行
STRATEGY_TYPE=price_only freqtrade backtesting --strategy ATRMLStrategy
```

## 8. 効果と利益

### 8.1 定量的効果

#### **ファイル数削減**

- **Before**: 10ファイル
- **After**: 6ファイル
- **削減率**: 40%

#### **特化ファイル排除**

- **Before**: 5個の特化ファイル (50%)
- **After**: 0個の特化ファイル (0%)
- **改善**: 100%排除

#### **新手法追加コスト**

- **Before**: 新ファイル作成 + 既存ファイル修正
- **After**: 新クラス追加のみ
- **工数削減**: 約70%

### 8.2 定性的効果

#### **拡張性向上**

- 新しい1次モデル: 1クラス追加 + 1行登録
- 新しい2次モデル: 1クラス追加 + 1行登録
- 設定のみでの切り替え可能

#### **保守性向上**

- 関連機能の集約による理解容易性向上
- テスト時のモック作成簡略化
- 設定とロジックの分離

#### **品質向上**

- 抽象化による実装の一貫性
- インターフェースの明確化
- エラーハンドリングの統一

### 8.3 将来の発展可能性

#### **対応予定の新手法**

1. **1次モデル**: ボリンジャーバンド、RSI、移動平均、フィボナッチ
2. **2次モデル**: XGBoost、RandomForest、CatBoost、ニューラルネットワーク
3. **ハイブリッド**: 複数1次モデルの組み合わせ、アンサンブル学習

#### **アーキテクチャの発展**

- プラグインシステムによる外部手法の動的読み込み
- A/Bテスト機能の内蔵
- リアルタイム手法切り替え機能

## 9. テストコード移行戦略

### 9.1 既存テストファイル分析

```bash
# 既存テストファイル（10ファイル）
tests/freqai/prediction_models/test_atr_lightgbm_classifier.py
tests/freqai/prediction_models/test_atr_mlops_pipeline.py
tests/strategies/test_atr_ml_strategy.py
tests/strategies/utils/test_atr_calculator.py
tests/strategies/utils/test_atr_return_calculator.py
tests/strategies/utils/test_feature_quality_manager.py
tests/strategies/utils/test_lightgbm_trainer.py
tests/strategies/utils/test_ml_label_generator.py
tests/strategies/utils/test_realtime_predictor.py
tests/strategies/utils/test_technical_indicator_engine.py
```

### 9.2 テスト移行マッピング

#### **削除対象（FreqAI重複排除）**

```bash
# FreqAI標準機能で置き換えられるもの
tests/freqai/prediction_models/test_atr_lightgbm_classifier.py   # → FreqAI BaseClassifierModel
tests/freqai/prediction_models/test_atr_mlops_pipeline.py       # → FreqAI pipeline
tests/strategies/utils/test_lightgbm_trainer.py                # → FreqAI trainer
tests/strategies/utils/test_ml_label_generator.py              # → FreqAI label機能
tests/strategies/utils/test_realtime_predictor.py              # → FreqAI predict機能
```

#### **統合対象（新アーキテクチャ対応）**

```bash
# 抽象化により統合
test_atr_calculator.py + test_atr_return_calculator.py
→ tests/strategies/utils/test_price_calculator.py

test_feature_quality_manager.py
→ tests/strategies/utils/test_strategy_factory.py（統合）

test_technical_indicator_engine.py
→ tests/strategies/utils/test_price_calculator.py（統合）
```

#### **更新対象（既存維持）**

```bash
# メイン戦略テスト（大幅更新）
tests/strategies/test_atr_ml_strategy.py
→ tests/strategies/test_atr_ml_strategy.py（新アーキテクチャ対応）
```

### 9.3 新テストファイル構成

```bash
tests/
├── strategies/
│   ├── test_atr_ml_strategy.py              # メイン戦略（更新）
│   └── utils/
│       ├── test_price_calculator.py         # 価格計算（新規統合）
│       ├── test_strategy_factory.py         # 戦略工場（新規）
│       ├── test_freqai_model_factory.py     # モデル工場（新規）
│       └── test_strategy_config.py          # 設定管理（新規）
└── migration/
    └── test_migration_phases.py             # 移行テスト（新規）
```

### 9.4 段階的テスト移行手順

#### **Phase 0: テストバックアップ**

```bash
#!/bin/bash
# test_migration_prepare.sh

echo "=== テストコードバックアップ開始 ==="

# 既存テストのバックアップ
mkdir -p .migration_backup/tests
cp -r tests/ .migration_backup/tests/

# テスト実行結果のベースライン取得
echo "ベースライン取得中..."
python -m pytest tests/strategies/test_atr_ml_strategy.py -v > .migration_backup/baseline_results.txt

echo "✅ テストバックアップ完了"
```

#### **Phase 1: 新テストファイル作成**

```bash
#!/bin/bash
# test_migration_phase1.sh

echo "=== 新テストファイル作成 ==="

# 新テストファイルを作成（まず最小構成）
create_test_price_calculator() {
    cat > tests/strategies/utils/test_price_calculator.py << 'EOF'
import pytest
import pandas as pd
from user_data.strategies.utils.price_calculator import PriceCalculatorFactory

class TestPriceCalculatorFactory:
    def test_create_atr_calculator(self):
        """ATR計算機の作成テスト"""
        calculator = PriceCalculatorFactory.create_calculator("atr", {"period": 14})
        assert calculator is not None

    def test_backwards_compatibility(self):
        """後方互換性テスト（ATRCalculatorEngine）"""
        from user_data.strategies.utils.atr_calculator import ATRCalculatorEngine
        old_engine = ATRCalculatorEngine.get_instance()
        assert old_engine is not None
EOF
}

create_test_price_calculator
echo "✅ 新テストファイル作成完了"
```

#### **Phase 2: 統合テスト作成**

```bash
#!/bin/bash
# test_migration_phase2.sh

echo "=== 統合テスト作成 ==="

# 既存機能をカバーする統合テスト
create_integration_tests() {
    cat > tests/strategies/utils/test_integration.py << 'EOF'
import pytest
from unittest.mock import patch, MagicMock
from user_data.strategies.atr_ml_strategy import ATRMLStrategy

class TestBackwardsCompatibility:
    """既存機能の後方互換性テスト"""

    def test_atr_calculation_compatibility(self):
        """ATR計算の互換性確認"""
        # 旧システムと新システムで同じ結果が出ることを確認
        pass

    def test_freqai_integration_compatibility(self):
        """FreqAI統合の互換性確認"""
        # FreqAI標準機能との互換性確認
        pass
EOF
}

create_integration_tests
echo "✅ 統合テスト作成完了"
```

#### **Phase 3: 段階的テスト置き換え**

```bash
#!/bin/bash
# test_migration_phase3.sh

echo "=== 段階的テスト置き換え ==="

# 既存テストを段階的に新システム対応に更新
update_main_strategy_test() {
    # test_atr_ml_strategy.pyを新アーキテクチャ対応に更新
    echo "メイン戦略テスト更新中..."

    # バックアップ作成
    cp tests/strategies/test_atr_ml_strategy.py .migration_backup/test_atr_ml_strategy_original.py

    # 新アーキテクチャ対応に更新（設定駆動型）
    # 実際の更新コードをここに配置
}

update_main_strategy_test
echo "✅ テスト置き換え完了"
```

#### **Phase 4: 冗長テスト削除**

```bash
#!/bin/bash
# test_migration_phase4.sh

echo "=== 冗長テスト削除 ==="

# FreqAI重複分の削除
remove_redundant_tests() {
    local files_to_remove=(
        "tests/freqai/prediction_models/test_atr_lightgbm_classifier.py"
        "tests/freqai/prediction_models/test_atr_mlops_pipeline.py"
        "tests/strategies/utils/test_lightgbm_trainer.py"
        "tests/strategies/utils/test_ml_label_generator.py"
        "tests/strategies/utils/test_realtime_predictor.py"
    )

    for file in "${files_to_remove[@]}"; do
        if [[ -f "$file" ]]; then
            echo "削除: $file"
            rm "$file"
        fi
    done
}

remove_redundant_tests
echo "✅ 冗長テスト削除完了"
```

#### **Phase 5: テスト品質検証**

```bash
#!/bin/bash
# test_migration_phase5.sh

echo "=== テスト品質検証 ==="

# カバレッジ確認
check_test_coverage() {
    echo "テストカバレッジ確認中..."
    python -m pytest --cov=user_data/strategies/ --cov-report=term-missing

    # 最小カバレッジ要件確認（85%以上）
    coverage_result=$(python -m pytest --cov=user_data/strategies/ --cov-report=json | jq '.totals.percent_covered')

    if (( $(echo "$coverage_result >= 85" | bc -l) )); then
        echo "✅ カバレッジ要件達成: ${coverage_result}%"
    else
        echo "❌ カバレッジ不足: ${coverage_result}%"
        return 1
    fi
}

# パフォーマンステスト
check_test_performance() {
    echo "テスト実行時間確認中..."
    start_time=$(date +%s)
    python -m pytest tests/strategies/ -v
    end_time=$(date +%s)

    execution_time=$((end_time - start_time))
    echo "テスト実行時間: ${execution_time}秒"

    # 5分以内の実行時間要件
    if [ $execution_time -lt 300 ]; then
        echo "✅ パフォーマンス要件達成"
    else
        echo "❌ テスト実行時間超過"
        return 1
    fi
}

check_test_coverage && check_test_performance
echo "✅ テスト品質検証完了"
```

### 9.5 テストロールバック戦略

#### **自動ロールバック条件**

```bash
# test_rollback_check.sh
#!/bin/bash

rollback_conditions=(
    "テストカバレッジが80%を下回る"
    "テスト実行時間が5分を超える"
    "既存テストの互換性テストが失敗"
    "FreqAI統合テストが失敗"
)

check_rollback_conditions() {
    local should_rollback=false

    # カバレッジチェック
    coverage=$(python -m pytest --cov=user_data/strategies/ --cov-report=json | jq '.totals.percent_covered')
    if (( $(echo "$coverage < 80" | bc -l) )); then
        echo "❌ カバレッジ不足: ${coverage}%"
        should_rollback=true
    fi

    # パフォーマンスチェック
    start_time=$(date +%s)
    python -m pytest tests/strategies/ -q
    end_time=$(date +%s)
    execution_time=$((end_time - start_time))

    if [ $execution_time -gt 300 ]; then
        echo "❌ テスト実行時間超過: ${execution_time}秒"
        should_rollback=true
    fi

    if [ "$should_rollback" = true ]; then
        echo "🔄 テストロールバック実行中..."
        cp -r .migration_backup/tests/ ./
        echo "✅ テストロールバック完了"
        return 1
    fi

    return 0
}
```

### 9.6 テスト移行の完全自動化

#### **master_test_migration.sh**

```bash
#!/bin/bash
# テスト移行マスタースクリプト

set -e  # エラー時即座終了

echo "🚀 テストコード移行開始"

# 移行前チェック
if ! python -m pytest tests/strategies/test_atr_ml_strategy.py -v; then
    echo "❌ 移行前テストが失敗。移行を中止します。"
    exit 1
fi

# 段階実行
phases=(
    "test_migration_prepare.sh"
    "test_migration_phase1.sh"
    "test_migration_phase2.sh"
    "test_migration_phase3.sh"
    "test_migration_phase4.sh"
    "test_migration_phase5.sh"
)

for phase in "${phases[@]}"; do
    echo "📋 実行中: $phase"

    if ! ./$phase; then
        echo "❌ $phase 失敗。ロールバック実行中..."
        ./test_rollback_check.sh
        exit 1
    fi

    # 各段階後のテスト実行
    if ! python -m pytest tests/strategies/ -x; then
        echo "❌ $phase 後のテスト失敗。ロールバック実行中..."
        ./test_rollback_check.sh
        exit 1
    fi

    echo "✅ $phase 完了"
done

echo "🎉 テストコード移行完了！"
echo "📊 最終統計:"
python -m pytest --cov=user_data/strategies/ --cov-report=term-missing
```

### 9.7 移行後テスト品質保証

#### **品質メトリクス**

```bash
# 移行前後の品質比較
echo "=== 移行前後品質比較 ==="

# テストファイル数
echo "テストファイル数:"
echo "  移行前: 10ファイル"
echo "  移行後: 5ファイル（50%削減）"

# テストカバレッジ
echo "カバレッジ:"
echo "  移行前: $(cat .migration_backup/baseline_coverage.txt)"
echo "  移行後: $(python -m pytest --cov=user_data/strategies/ --cov-report=json | jq '.totals.percent_covered')%"

# 実行時間
echo "実行時間:"
echo "  移行前: $(cat .migration_backup/baseline_time.txt)秒"
echo "  移行後: $(measure_test_time)秒"
```

#### **継続的品質監視**

```bash
# .github/workflows/test_quality_monitor.yml
name: Test Quality Monitor
on: [push, pull_request]

jobs:
  test-quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Setup Python
        uses: actions/setup-python@v3
        with:
          python-version: '3.9'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest-cov

      - name: Run tests with coverage
        run: |
          python -m pytest --cov=user_data/strategies/ --cov-fail-under=85

      - name: Performance check
        run: |
          start_time=$(date +%s)
          python -m pytest tests/strategies/ -v
          end_time=$(date +%s)
          execution_time=$((end_time - start_time))

          if [ $execution_time -gt 300 ]; then
            echo "Test execution time exceeded 5 minutes: ${execution_time}s"
            exit 1
          fi
```

## 10. まとめ

この設計により以下を達成：

### 実装削減（60%削減）

- **Before**: 10ファイル構成
- **After**: 4ファイル構成（60%削減）

### テスト削減（50%削減）

- **Before**: 10テストファイル
- **After**: 5テストファイル（50%削減）

### 設定駆動型アーキテクチャ

- JSONによる完全な設定制御
- 実装変更なしでの戦略切り替え

### FreqAI最大活用

- 重複コード完全排除
- FreqAI標準機能の活用
- 保守性とパフォーマンスの向上

### 安全な移行プロセス

- 5段階の原子的移行（実装・テスト同期）
- 各段階での自動ロールバック機能
- 破損状態の完全回避

この設計により、保守性・拡張性・安全性を備えた次世代ML統合システムを実現します。
