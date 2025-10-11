# 設計上の課題: Strategy Factory アーキテクチャ

## メタ情報

- **作成日**: 2025-10-11
- **最終更新**: 2025-10-11
- **コミットハッシュ**: 3ddd09b9dbb4929e14299fa2eab04c44229bd7f1
- **関連ファイル**:
  - `user_data/strategies/utils/strategy_factory.py`
  - `user_data/strategies/atr_ml_strategy.py`
  - `config.json`

## 概要

コードレビュー対応中に、`TwoTierStrategy`クラスにおける具体的な戦略ロジックのハードコーディングに関する設計上の課題が指摘された。本ドキュメントでは、より拡張性と柔軟性の高いアーキテクチャを提案する。

## 現状の実装

### 設定構造（config.json）

現在のFreqtrade設定では、2層戦略の設定は以下のように構造化されている:

```json
{
  "two_tier_strategy": {
    "preset": "price_only",
    "primary_model": {
      "type": "atr",
      "params": {
        "period": 14,
        "multiplier": 0.5
      }
    },
    "secondary_model": {
      "enabled": false,
      "confidence_threshold": 0.6
    }
  },
  "freqai": {
    "enabled": false,
    // ... FreqAI設定 ...
  }
}
```

**設定の特徴**:

- `preset`: 事前定義された戦略設定（`price_only`, `default`等）を選択可能
- `primary_model`: 1次モデル（価格計算）の選択とパラメータ設定
- `secondary_model`: 2次モデル（ML）の有効/無効とパラメータ設定

### 問題箇所

**ファイル**: `user_data/strategies/utils/strategy_factory.py:417-453`

```python
def _generate_basic_price_signals(self, dataframe, metadata: Dict[str, Any]):
    """価格ブレイクアウト信号生成

    ML予測なしの価格ブレイクアウト戦略
    前期間の指値価格に現在価格が到達した場合に信号を生成
    """
    # ... 価格ブレイクアウト戦略の具体的な実装 ...
```

### 問題点

1. **util層に具体的な戦略ロジックが存在**
   - `_generate_basic_price_signals`は「価格ブレイクアウト」という具体的なトレーディングロジックを実装
   - util層に特定の戦略がハードコードされている

2. **拡張性の課題**
   - ML無効時の戦略を変更する場合、`TwoTierStrategy`クラスを直接修正する必要がある
   - 新しい戦略を追加する際の柔軟性が低い

3. **設定との不整合**
   - `config.json`では`primary_model`の価格計算器を選択可能だが、信号生成戦略は選択できない
   - `secondary_model.enabled=false`の場合、常に価格ブレイクアウト戦略が使用される

## 設計原則

### 要求事項

1. **config.jsonで名前による戦略指定**
   - primary, secondaryを名前で指定可能
   - 名前=戦略ファイル/クラスに対応

2. **secondary=nullで1次モデルのみ**
   - secondaryをnullに設定することで、1次モデルのみで動作

3. **2次モデルのみは考えない**
   - 1次モデルは2次モデルのラベル生成にも使用される
   - 次の足のclose上げ下げを見るだけのロジックも1次モデルとして実装

4. **util層に具体的な処理を書かない**
   - util層はファクトリーと抽象化のみ
   - 具体的な戦略ロジックは別ディレクトリに配置

### richmanbtc概念との整合性

richmanbtcチュートリアルの概念:

- **1次モデル**: 指値注文戦略（価格計算＋信号生成を含む完全な戦略）
- **2次モデル**: ML分類によるフィルタリング（1次モデルの信号を選別）

この概念に基づき、1次戦略は単なる価格計算器ではなく、価格計算と信号生成の両方を持つ完全な戦略として設計する。

**現在の実装状況**:

- ATR戦略（primary="atr_breakout" + secondary="lightgbm_classifier"）はrichmanbtcチュートリアルの再現が目的
- 他の戦略の組み合わせも選択可能な設計

### 信号（ラベル）の定義

**2次モデル（ML）の学習ラベル**:

指値注文が執行されたときに、その取引価格を起点とした利益がプラスかマイナスかの2値分類を基本とする:

- **ラベル = 1**: 指値注文が執行された後、利益がプラスになる取引
- **ラベル = 0**: 指値注文が執行されても損失になる、または執行されない取引

**ラベル生成の2つのアプローチ**:

richmanbtcチュートリアルでは、以下の2通りの約定シミュレーション方法が提示されている:

#### アプローチ1: エントリー追いかけ型（約定シミュレーション例1）

**エントリールール**: 買うと決めたら約定するまで指値で追いかける
**エグジットルール**: t足後に売り始める。売ると決めたら約定するまで追いかける

```python
df['y_buy'] = df['sell_fep'].shift(-t) / df['buy_fep'] - 1 - 2 * fee
df['y_sell'] = -(df['buy_fep'].shift(-t) / df['sell_fep'] - 1) - 2 * fee
```

**特徴**:

- エントリーもエグジットも約定するまで追いかける
- Force Entry Price（FEP）を使用して実際の約定価格を計算
- 確実に取引が成立する前提
- より積極的な取引戦略

#### アプローチ2: エントリー1足限定型（約定シミュレーション例2、推奨）

**エントリールール**: 指値を出して次の足でキャンセル
**エグジットルール**: エントリーが約定した場合のみ、t足後に売り始める。売ると決めたら約定するまで追いかける

```python
df['y_buy'] = np.where(
    (df['buy_price'] / pips).round() > (df['lo'].shift(-1) / pips).round(),
    df['sell_fep'].shift(-t) / df['buy_price'] - 1 - 2 * fee,
    0
)
df['y_sell'] = np.where(
    (df['sell_price'] / pips).round() < (df['hi'].shift(-1) / pips).round(),
    -(df['buy_fep'].shift(-t) / df['sell_price'] - 1) - 2 * fee,
    0
)
```

**特徴**:

- エントリーは1足限定（約定しなければキャンセル）
- 約定した取引のみをラベル化（約定しない場合は0）
- より現実的な取引戦略
- リスク管理が容易

**Force Entry Price（FEP）の概念**:

買うと決めてから約定するまで指値で追いかけた場合に、実際に約定する価格。各足で指値を更新しながら、価格が指値に到達するまで待つ。

**戦略によるバリエーション**:

- 約定シミュレーション方法: 追いかけ型 / 1足限定型
- 利益計算期間（t）: 24足後、48足後など
- 手数料（fee）: maker手数料（例: -0.00025）、taker手数料など
- 指値の出し方: ATRベース、最良気配近似（close ± pips）など

**実装への反映**:

- 1次戦略は指値価格の計算とラベル生成ロジックを提供
- 2次モデルは学習済みMLで「この指値注文は利益になるか」を予測
- 予測結果に基づいて注文の執行を制御

参考: [richmanbtcチュートリアル](https://note.com/btcml/n/n9f730e59848c)

## 提案する設計

### ディレクトリ構造

```
user_data/strategies/
├── primary/                    # 1次戦略（価格計算＋信号生成）
│   ├── __init__.py
│   ├── base.py                # PrimaryStrategyBase抽象クラス
│   ├── atr_breakout.py        # ATR価格＋ブレイクアウト信号
│   ├── atr_mean_reversion.py  # ATR価格＋平均回帰信号
│   ├── bollinger_breakout.py  # ボリンジャーバンド＋ブレイクアウト
│   └── simple_close.py        # 次足close予測用（ラベル生成用）
├── secondary/                  # 2次モデル（ML予測＋フィルタリング）
│   ├── __init__.py
│   ├── base.py                # SecondaryModelBase抽象クラス
│   ├── lightgbm_classifier.py
│   ├── xgboost_classifier.py
│   └── catboost_classifier.py
├── utils/                      # ファクトリーと抽象化のみ
│   ├── strategy_factory.py    # TwoTierStrategyとファクトリー
│   ├── price_calculator.py    # 既存の価格計算器（後方互換性用）
│   └── freqai_model_factory.py
└── atr_ml_strategy.py         # Freqtradeエントリーポイント
```

**ポイント**:

- `primary/`: 具体的な1次戦略実装（util層から分離）
- `secondary/`: 具体的な2次モデル実装（util層から分離）
- `utils/`: ファクトリーと抽象基底クラスのみ（具体的な戦略実装なし）

### クラス設計

#### 1次戦略（Primary Strategy）

```python
# primary/base.py
from abc import ABC, abstractmethod
import pandas as pd

class PrimaryStrategyBase(ABC):
    """1次戦略の抽象基底クラス

    価格計算と信号生成の両方を担当する完全な戦略
    """

    def __init__(self, params: dict):
        self.params = params

    @abstractmethod
    def calculate_prices(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """価格計算（buy_price, sell_priceカラムを追加）

        Args:
            dataframe: OHLCデータ

        Returns:
            buy_price, sell_priceが追加されたDataFrame
        """
        pass

    @abstractmethod
    def generate_signals(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """エントリー信号生成（enter_long, enter_shortカラムを追加）

        Args:
            dataframe: 価格計算済みのDataFrame
            metadata: ペア情報

        Returns:
            enter_long, enter_shortが追加されたDataFrame
        """
        pass
```

```python
# primary/atr_breakout.py
class ATRBreakoutStrategy(PrimaryStrategyBase):
    """ATR価格計算＋ブレイクアウト信号生成戦略"""

    def __init__(self, params: dict):
        super().__init__(params)
        self.period = params.get("period", 14)
        self.multiplier = params.get("multiplier", 0.5)
        self.lookback = params.get("lookback", 1)

    def calculate_prices(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """ATRベースの指値価格計算"""
        # ATR計算
        atr = self._calculate_atr(dataframe, self.period)
        offset = atr * self.multiplier

        # 指値価格設定
        dataframe["buy_price"] = dataframe["close"] - offset
        dataframe["sell_price"] = dataframe["close"] + offset

        return dataframe

    def generate_signals(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """価格ブレイクアウト信号生成

        前期間の指値価格に現在価格が到達した場合に信号を生成
        """
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0

        if len(dataframe) <= self.lookback:
            return dataframe

        # 前期間の指値価格
        prev_buy_price = dataframe["buy_price"].shift(self.lookback)
        prev_sell_price = dataframe["sell_price"].shift(self.lookback)
        current_close = dataframe["close"]

        # ブレイクアウト条件
        buy_signal = (current_close <= prev_buy_price) & (prev_buy_price > 0)
        sell_signal = (current_close >= prev_sell_price) & (prev_sell_price > 0)

        # 価格有効性チェック
        price_valid = (dataframe["buy_price"] > 0) & (dataframe["sell_price"] > 0)

        dataframe.loc[buy_signal & price_valid, "enter_long"] = 1
        dataframe.loc[sell_signal & price_valid, "enter_short"] = 1

        return dataframe
```

```python
# primary/simple_close.py
class SimpleCloseStrategy(PrimaryStrategyBase):
    """次足close予測用シンプル戦略（ラベル生成用）

    ML学習時のラベル生成に使用
    次の足のclose上げ下げを予測するためのシンプルなロジック
    """

    def __init__(self, params: dict):
        super().__init__(params)
        self.threshold = params.get("threshold", 0.001)

    def calculate_prices(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """現在価格をそのまま使用"""
        dataframe["buy_price"] = dataframe["close"]
        dataframe["sell_price"] = dataframe["close"]
        return dataframe

    def generate_signals(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """次足closeの上げ下げで信号生成"""
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0

        # 次足のclose変化率
        future_return = dataframe["close"].pct_change().shift(-1)

        dataframe.loc[future_return > self.threshold, "enter_long"] = 1
        dataframe.loc[future_return < -self.threshold, "enter_short"] = 1

        return dataframe
```

#### 2次モデル（Secondary Model）

```python
# secondary/base.py
from abc import ABC, abstractmethod
import pandas as pd

class SecondaryModelBase(ABC):
    """2次モデル（ML）の抽象基底クラス"""

    def __init__(self, params: dict):
        self.params = params

    @abstractmethod
    def integrate_with_freqai(self, dataframe: pd.DataFrame, metadata: dict, strategy) -> pd.DataFrame:
        """FreqAI予測データの統合

        Args:
            dataframe: DataFrame
            metadata: ペア情報
            strategy: Freqtradeストラテジーインスタンス（freqai.start()実行用）

        Returns:
            &-predictionカラムが追加されたDataFrame
        """
        pass

    @abstractmethod
    def filter_signals(self, dataframe: pd.DataFrame, signals: pd.DataFrame, config: dict) -> pd.DataFrame:
        """ML予測に基づいて信号をフィルタリング

        Args:
            dataframe: 予測データを含むDataFrame
            signals: 1次戦略からの信号
            config: フィルタリング設定

        Returns:
            フィルタリングされた信号
        """
        pass
```

```python
# secondary/lightgbm_classifier.py
class LightGBMClassifier(SecondaryModelBase):
    """LightGBM二値分類モデル"""

    def __init__(self, params: dict):
        super().__init__(params)
        self.confidence_threshold = params.get("confidence_threshold", 0.6)

    def integrate_with_freqai(self, dataframe: pd.DataFrame, metadata: dict, strategy) -> pd.DataFrame:
        """FreqAI統合でLightGBM予測を取得"""
        # FreqAIのstart()を呼び出して予測データを取得
        dataframe = strategy.freqai.start(dataframe, metadata, strategy)
        return dataframe

    def filter_signals(self, dataframe: pd.DataFrame, signals: pd.DataFrame, config: dict) -> pd.DataFrame:
        """ML予測（0/1）で信号をフィルタリング"""
        if "&-prediction" not in dataframe.columns:
            logger.warning("ML prediction data not available")
            return signals

        # 予測が1の場合のみ信号を通す
        ml_filter = dataframe["&-prediction"] == 1

        # 信頼度フィルタリング
        if "&-probability" in dataframe.columns:
            confidence_filter = dataframe["&-probability"] >= self.confidence_threshold
            ml_filter = ml_filter & confidence_filter

        signals["enter_long"] = signals["enter_long"] & ml_filter
        signals["enter_short"] = signals["enter_short"] & ml_filter

        return signals
```

#### TwoTierStrategy（統合クラス）

```python
# utils/strategy_factory.py
class TwoTierStrategy:
    """2層戦略実行クラス

    1次戦略と2次モデルを統合して信号を生成
    """

    def __init__(
        self,
        primary_strategy: PrimaryStrategyBase,
        secondary_model: Optional[SecondaryModelBase],
        config: dict
    ):
        self.primary_strategy = primary_strategy
        self.secondary_model = secondary_model
        self.config = config
        self.is_ml_enabled = secondary_model is not None

        logger.info(
            f"TwoTierStrategy initialized: "
            f"primary={type(primary_strategy).__name__}, "
            f"secondary={type(secondary_model).__name__ if secondary_model else 'None'}"
        )

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict, strategy) -> pd.DataFrame:
        """指標計算（価格計算＋ML統合）"""
        # 1次戦略: 価格計算
        dataframe = self.primary_strategy.calculate_prices(dataframe)

        # 2次モデル: FreqAI統合（有効時のみ）
        if self.is_ml_enabled and hasattr(strategy, 'freqai'):
            dataframe = self.secondary_model.integrate_with_freqai(dataframe, metadata, strategy)

        return dataframe

    def generate_entry_signals(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """エントリー信号生成

        1次戦略で信号生成し、2次モデルでフィルタリング
        """
        # 1次戦略: 信号生成
        signals = self.primary_strategy.generate_signals(dataframe, metadata)

        # 2次モデル: MLフィルタリング（有効時のみ）
        if self.is_ml_enabled:
            signals = self.secondary_model.filter_signals(dataframe, signals, self.config)

        return signals
```

#### ファクトリー

```python
# utils/strategy_factory.py
class StrategyFactory:
    """戦略ファクトリー

    config.jsonの名前から具体的な戦略クラスをロード
    """

    # 1次戦略の登録
    _primary_strategies = {
        "atr_breakout": "user_data.strategies.primary.atr_breakout.ATRBreakoutStrategy",
        "atr_mean_reversion": "user_data.strategies.primary.atr_mean_reversion.ATRMeanReversionStrategy",
        "bollinger_breakout": "user_data.strategies.primary.bollinger_breakout.BollingerBreakoutStrategy",
        "simple_close": "user_data.strategies.primary.simple_close.SimpleCloseStrategy",
    }

    # 2次モデルの登録
    _secondary_models = {
        "lightgbm_classifier": "user_data.strategies.secondary.lightgbm_classifier.LightGBMClassifier",
        "xgboost_classifier": "user_data.strategies.secondary.xgboost_classifier.XGBoostClassifier",
        "catboost_classifier": "user_data.strategies.secondary.catboost_classifier.CatBoostClassifier",
    }

    @classmethod
    def create_two_tier_strategy(cls, config: dict) -> TwoTierStrategy:
        """2層戦略の作成

        Args:
            config: 戦略設定

        Returns:
            設定済み2層戦略インスタンス
        """
        primary_name = config.get("primary")
        secondary_name = config.get("secondary")

        if not primary_name:
            raise ValueError("primary strategy name is required")

        # 1次戦略のロード
        primary_class = cls._load_class(cls._primary_strategies[primary_name])
        primary_params = config.get("primary_params", {})
        primary_strategy = primary_class(primary_params)

        # 2次モデルのロード（nullの場合はNone）
        secondary_model = None
        if secondary_name:
            secondary_class = cls._load_class(cls._secondary_models[secondary_name])
            secondary_params = config.get("secondary_params", {})
            secondary_model = secondary_class(secondary_params)

        return TwoTierStrategy(
            primary_strategy=primary_strategy,
            secondary_model=secondary_model,
            config=config
        )

    @classmethod
    def _load_class(cls, class_path: str):
        """モジュールパスからクラスをロード"""
        module_path, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)

    @classmethod
    def register_primary_strategy(cls, name: str, class_path: str):
        """新しい1次戦略を登録"""
        cls._primary_strategies[name] = class_path
        logger.info(f"Registered primary strategy: {name}")

    @classmethod
    def register_secondary_model(cls, name: str, class_path: str):
        """新しい2次モデルを登録"""
        cls._secondary_models[name] = class_path
        logger.info(f"Registered secondary model: {name}")
```

### config.jsonの設計

#### 基本形式（2層戦略）

```json
{
  "two_tier_strategy": {
    "primary": "atr_breakout",
    "secondary": "lightgbm_classifier",
    "primary_params": {
      "period": 14,
      "multiplier": 0.5,
      "lookback": 1
    },
    "secondary_params": {
      "confidence_threshold": 0.6
    }
  },
  "freqai": {
    "enabled": true,
    "identifier": "atr_lightgbm_v1",
    // ... FreqAI設定 ...
  }
}
```

#### 1次モデルのみ（secondary=null）

```json
{
  "two_tier_strategy": {
    "primary": "atr_breakout",
    "secondary": null,
    "primary_params": {
      "period": 14,
      "multiplier": 0.5,
      "lookback": 1
    }
  },
  "freqai": {
    "enabled": false
  }
}
```

#### 平均回帰戦略の例

```json
{
  "two_tier_strategy": {
    "primary": "atr_mean_reversion",
    "secondary": "xgboost_classifier",
    "primary_params": {
      "period": 20,
      "multiplier": 1.0,
      "reversion_threshold": 0.02
    },
    "secondary_params": {
      "confidence_threshold": 0.7
    }
  }
}
```

#### ラベル生成用シンプル戦略

```json
{
  "two_tier_strategy": {
    "primary": "simple_close",
    "secondary": "lightgbm_classifier",
    "primary_params": {
      "threshold": 0.001
    },
    "secondary_params": {
      "confidence_threshold": 0.6
    }
  }
}
```

**設定の特徴**:

- `primary`: 1次戦略名（strategies/primary/配下のファイル/クラスに対応）
- `secondary`: 2次モデル名（strategies/secondary/配下のファイル/クラスに対応、nullで無効化）
- `primary_params`: 1次戦略のパラメータ
- `secondary_params`: 2次モデルのパラメータ

### Optuna最適化への対応（Phase 2）

Phase 2で実装予定のOptuna最適化を考慮した設計:

**最適化対象パラメータの定義**:

```python
# primary/atr_breakout.py
class ATRBreakoutStrategy(PrimaryStrategyBase):
    """ATR価格計算＋ブレイクアウト信号生成戦略"""

    # Optuna最適化用のパラメータ空間定義
    PARAM_SPACE = {
        "period": {"type": "int", "low": 7, "high": 28, "default": 14},
        "multiplier": {"type": "float", "low": 0.1, "high": 2.0, "default": 0.5},
        "lookback": {"type": "int", "low": 1, "high": 5, "default": 1}
    }

    def __init__(self, params: dict):
        super().__init__(params)
        self.period = params.get("period", self.PARAM_SPACE["period"]["default"])
        self.multiplier = params.get("multiplier", self.PARAM_SPACE["multiplier"]["default"])
        self.lookback = params.get("lookback", self.PARAM_SPACE["lookback"]["default"])
```

**Optuna統合イメージ**:

```python
# Phase 2で実装
def optimize_strategy_with_optuna(strategy_name: str, n_trials: int = 100):
    """Optunaで戦略パラメータを最適化"""

    def objective(trial):
        # 戦略クラスからパラメータ空間を取得
        strategy_class = StrategyFactory.get_strategy_class(strategy_name)
        param_space = strategy_class.PARAM_SPACE

        # Optunaでパラメータをサンプリング
        params = {}
        for param_name, space in param_space.items():
            if space["type"] == "int":
                params[param_name] = trial.suggest_int(param_name, space["low"], space["high"])
            elif space["type"] == "float":
                params[param_name] = trial.suggest_float(param_name, space["low"], space["high"])

        # バックテスト実行
        result = run_backtest(strategy_name, params)
        return result["profit"]

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params
```

**利点**:

- 各戦略クラスがパラメータ空間を定義
- config.jsonで戦略を指定すれば、その戦略のパラメータ空間が自動的に使用される
- 新しい戦略を追加しても、Optuna最適化ロジックの変更は不要

## 実装タイミング

### Phase 2以降で実装すべき理由

1. **Phase 1の目標達成を優先**
   - 現在のリファクタリング目標: ML統合ロジックの分離と品質向上
   - 基本戦略の抽象化は追加スコープ

2. **要件の明確化が必要**
   - どのような戦略バリエーションが必要か、実運用での経験が必要
   - 過度な抽象化を避け、実際のニーズに基づいた設計が望ましい

3. **段階的な改善の方針**
   - Phase 1: 基本機能の安定化（現在のリファクタリング完了）
   - Phase 2: ハイパーパラメータ最適化とアーキテクチャ改善
   - Phase 3: 性能検証と最終統合

## 具体的な実装ステップ（Phase 2以降）

### ステップ1: ディレクトリ構造の作成

```bash
mkdir -p user_data/strategies/primary
mkdir -p user_data/strategies/secondary
```

### ステップ2: 抽象基底クラスの実装

1. `primary/base.py`: PrimaryStrategyBase抽象クラス
2. `secondary/base.py`: SecondaryModelBase抽象クラス

### ステップ3: 既存戦略の移行

1. 既存の価格ブレイクアウト戦略を`primary/atr_breakout.py`として独立化
2. 既存のFreqAI統合を`secondary/lightgbm_classifier.py`として独立化

### ステップ4: ファクトリーの改修

1. StrategyFactoryに戦略ロード機能を追加
2. 名前解決とクラスロード機能の実装
3. 既存のTwoTierStrategyを新しいアーキテクチャに対応

### ステップ5: config.jsonの拡張

1. 新しい設定形式のサポート追加
2. バリデーション機能の追加
3. パラメータ空間定義の実装（Optuna対応）

### ステップ6: 新戦略の追加

1. 平均回帰戦略（primary/atr_mean_reversion.py）
2. ボリンジャーバンド戦略（primary/bollinger_breakout.py）
3. シンプルclose予測戦略（primary/simple_close.py）

### ステップ7: テストとドキュメント

1. 各戦略の単体テスト
2. 統合テスト（2層戦略全体）
3. 新戦略追加ガイドのドキュメント化
4. マイグレーションガイドの作成

## 利点

### 拡張性

1. **新戦略の追加が容易**
   - 新しいファイルとクラスを作成するだけ
   - ファクトリーへの登録は1行
   - utils層の変更不要

2. **戦略の組み合わせが自由**
   - 任意の1次戦略×任意の2次モデルの組み合わせ
   - config.jsonで簡単に切り替え

3. **実験が容易**
   - 複数の戦略バリエーションを並行して開発可能
   - バックテストで簡単に比較

### 保守性

1. **責務の明確化**
   - util層: ファクトリーと抽象化のみ
   - primary/: 1次戦略の具体的実装
   - secondary/: 2次モデルの具体的実装

2. **テストが容易**
   - 各戦略を独立してテスト可能
   - モックを使った単体テストが簡単

3. **コードの見通しが良い**
   - 1ファイル1クラスの原則
   - ディレクトリ構造が役割を表現

### 柔軟性

1. **設定駆動**
   - コード変更なしで戦略切り替え
   - パラメータ調整が容易

2. **最適化への親和性**
   - 各戦略クラスがパラメータ空間を定義
   - Optuna等の最適化ツールとの統合が容易

3. **カスタマイズ性**
   - ユーザーが独自の戦略を追加可能
   - 登録機能で拡張が簡単

## 結論

現状の実装は設計上の課題を含むものの、Phase 1の目標達成には十分であり、動作も安定している。

提案する設計は以下の利点を提供:

- **util層に具体的な戦略実装を書かない** - ファクトリーと抽象化のみ
- **config.jsonで名前による戦略指定** - 拡張性と柔軟性の向上
- **secondary=nullで1次モデルのみ** - シンプルな構成も可能
- **richmanbtc概念との整合性** - 1次戦略=完全な戦略として設計

Phase 2以降で本設計を実装することで、保守性・拡張性・柔軟性を大幅に向上させることができる。

## 参考情報

### 関連する仕様書

- `.kiro/specs/freqtrade-ml-atr-strategy/tasks.md` - Phase 1-3の実装計画
- `.kiro/specs/freqtrade-ml-atr-strategy/requirements.md` - 要件定義

### 関連コミット

- `61f040c1c`: FreqAI設定パラメータをdataclassで型安全化
- `3ddd09b9d`: コメント品質改善とエラーハンドリング強化

### 参考資料

- richmanbtcチュートリアル: 2層トレーディングシステムの概念
- Freqtrade Strategy開発ガイド: カスタム戦略の実装方法
