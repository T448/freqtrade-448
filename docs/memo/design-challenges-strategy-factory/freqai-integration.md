# FreqAI統合

[⬅️ README に戻る](./README.md)

このドキュメントでは、FreqAIフレームワークとの統合方法、ラベル生成フロー、マルチターゲット実装について説明します。

## 目次

- [FreqAI統合アーキテクチャ](#freqai統合アーキテクチャ)
- [TwoTierStrategy統合クラス](#twotierstrategy統合クラス)
- [FreqAIマルチターゲット実装](#freqaiマルチターゲット実装)
- [ラベル生成フロー](#ラベル生成フロー)

## FreqAI統合アーキテクチャ

### 全体構成

```
TwoTierStrategy(IStrategy) ← Freqtradeエントリーポイント
├── PrimaryStrategyBase (例: ATRBreakoutStrategy)
│   ├── calculate_prices() → buy_price, sell_price
│   └── calculate_returns() → ラベル生成用リターン計算
│
└── FreqAIフレームワーク経由で呼び出し
    └── TwoTierLightGBMClassifier(BaseClassifierModel) ← 実際のMLモデル
        ├── populate_indicators() → 特徴量生成
        ├── set_freqai_targets() → ラベル生成（TwoTierStrategyから呼ばれる）
        ├── fit() → モデル訓練
        └── predict() → 予測実行
```

### FreqAIモデルの実装場所

実際のMLモデルは、**FreqAIモデルとして別途実装**します。
1次戦略と2次モデルは独立して選択可能なため、FreqAIモデル名は2次モデルの種類のみを反映します。

**実装場所**: `user_data/freqaimodels/two_tier_lightgbm_classifier.py`

```python
# user_data/freqaimodels/two_tier_lightgbm_classifier.py
from freqtrade.freqai.base_models.BaseClassifierModel import BaseClassifierModel

class TwoTierLightGBMClassifier(BaseClassifierModel):
    """2層戦略用LightGBM二値分類モデル（FreqAIモデル本体）

    FreqAIフレームワークのBaseClassifierModelを継承し、
    任意の1次戦略と組み合わせ可能な汎用的なML実装を提供

    Note:
        - 1次戦略（ATRBreakout, MeanReversion等）とは独立
        - configで1次戦略と2次モデルを自由に組み合わせ可能
    """

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """特徴量生成（テクニカル指標等）

        Returns:
            %で始まる特徴量カラムが追加されたDataFrame
        """
        # 移動平均、RSI、MACD等のテクニカル指標を計算
        # 1次戦略に依存しない汎用的な特徴量
        # ...
        return dataframe

    def set_freqai_targets(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """訓練用ラベル生成

        TwoTierStrategy.set_freqai_targets()から間接的に呼ばれる

        Note:
            実際のラベル生成はTwoTierStrategyで実装されるため、
            このメソッドは空または最小限の実装
        """
        return dataframe
```

**Phase 2で追加される他のFreqAIモデル例**:

- `user_data/freqaimodels/two_tier_xgboost_classifier.py` - XGBoost実装
- `user_data/freqaimodels/two_tier_catboost_classifier.py` - CatBoost実装

### Config設定での組み合わせ例

`config.json`で1次戦略と2次モデルを独立して指定します：

```json
{
  "two_tier_strategy": {
    "primary": "atr_breakout",           // 1次戦略: ATRBreakout
    "secondary": "lightgbm_classifier"   // 2次モデル: LightGBM
  },
  "freqai": {
    "enabled": true,
    "model_name": "TwoTierLightGBMClassifier",  // FreqAIモデル
    "model_training_parameters": {
      "n_estimators": 100,
      "learning_rate": 0.1
    }
  }
}
```

**別の組み合わせ例（Phase 2以降）**:

```json
{
  "two_tier_strategy": {
    "primary": "bollinger_breakout",      // 1次戦略を変更
    "secondary": "xgboost_classifier"     // 2次モデルも変更
  },
  "freqai": {
    "enabled": true,
    "model_name": "TwoTierXGBoostClassifier"  // 対応するFreqAIモデル
  }
}
```

## TwoTierStrategy統合クラス

```python
# two_tier_strategy.py
from freqtrade.strategy import IStrategy
import pandas as pd
from typing import Optional

class TwoTierStrategy(IStrategy):
    """Config駆動の2層取引戦略（Freqtradeエントリーポイント）

    FreqtradeのIStrategyを継承し、config.jsonで指定された
    1次戦略と2次モデルを動的にロード・統合する

    実行例:
        freqtrade backtesting --strategy TwoTierStrategy --config config.json
    """

    def __init__(self, config: dict):
        super().__init__(config)
        two_tier_config = config.get('two_tier_strategy', {})
        freqai_config = config.get('freqai', {})

        # Config validation: freqai.enabled and secondary must be consistent
        freqai_enabled = freqai_config.get('enabled', False)
        has_secondary = two_tier_config.get('secondary') is not None

        if has_secondary and not freqai_enabled:
            raise ValueError(
                "Invalid configuration: secondary model is specified but freqai.enabled is False. "
                "Please set freqai.enabled=true when using a secondary model."
            )

        if freqai_enabled and not has_secondary:
            raise ValueError(
                "Invalid configuration: freqai.enabled is True but no secondary model specified. "
                "Please set secondary to a model name (e.g., 'lightgbm_classifier') or disable FreqAI."
            )

        # StrategyFactoryで1次戦略をロード
        from user_data.strategies.utils.strategy_factory import StrategyFactory
        self.primary_strategy = StrategyFactory.load_primary(two_tier_config)
        self.is_ml_enabled = freqai_enabled

        logger.info(
            f"TwoTierStrategy initialized: "
            f"primary={type(self.primary_strategy).__name__}, "
            f"freqai_enabled={freqai_enabled}"
        )

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """指標計算（価格計算＋ML統合）

        1次戦略で指値価格を計算し、ML有効時はFreqAI予測を統合
        """
        # 1次戦略: 指値価格計算
        dataframe = self.primary_strategy.calculate_prices(dataframe)

        # FreqAI直接呼び出し（ML有効時のみ）
        if self.is_ml_enabled:
            # Buy モデル予測
            dataframe = self.freqai_buy.start(dataframe, metadata, self)
            dataframe.rename(columns={'&-prediction': '&-prediction_buy'}, inplace=True)

            # Sell モデル予測
            dataframe = self.freqai_sell.start(dataframe, metadata, self)
            dataframe.rename(columns={'&-prediction': '&-prediction_sell'}, inplace=True)

        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """エントリーシグナル生成

        両建て対応: buy/sellを独立して判定
        ML有効時: 各方向の予測=1の場合のみエントリー
        ML無効時: 常に両方向エントリー（指値価格があれば注文）
        """
        if self.is_ml_enabled:
            # ML予測が1の場合のみエントリー（buy/sell独立）
            dataframe.loc[(dataframe['&-prediction_buy'] == 1), 'enter_long'] = 1
            dataframe.loc[(dataframe['&-prediction_sell'] == 1), 'enter_short'] = 1
        else:
            # ML無効時は常に両方向エントリー
            dataframe.loc[:, 'enter_long'] = 1
            dataframe.loc[:, 'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """エグジットシグナル生成

        ML予測に基づく明示的な決済シグナル生成

        Freqtradeは反対売買による自動決済をサポートしていないため、
        exit_long/exit_short で明示的に決済指示が必要

        両建て状態（long + short 同時保有）では、両方のexitシグナルが
        同時に発生する可能性があり、その場合Freqtradeは両ポジションを決済する
        """
        if self.is_ml_enabled:
            # ロング決済: sell予測=1の場合
            dataframe.loc[
                (dataframe['&-prediction_sell'] == 1),
                'exit_long'
            ] = 1

            # ショート決済: buy予測=1の場合
            dataframe.loc[
                (dataframe['&-prediction_buy'] == 1),
                'exit_short'
            ] = 1

        return dataframe

    def custom_entry_price(
        self,
        pair: str,
        current_time,
        proposed_rate: float,
        entry_tag: Optional[str] = None,
        **kwargs
    ) -> float:
        """エントリー指値価格（1次戦略の計算結果を使用）"""
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if len(dataframe) > 0:
            return dataframe.iloc[-1]['buy_price']
        return proposed_rate

    def custom_exit_price(
        self,
        pair: str,
        trade,
        current_time,
        proposed_rate: float,
        **kwargs
    ) -> float:
        """エグジット指値価格（1次戦略の計算結果を使用）"""
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if len(dataframe) > 0:
            return dataframe.iloc[-1]['sell_price']
        return proposed_rate

    def set_freqai_targets(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """FreqAI訓練用ラベル生成

        1次戦略のリターン計算結果をラベル化
        buy/sell独立したラベルを生成（両建て対応）
        リターン > 0 で成功ラベル（1）、それ以外は失敗ラベル（0）
        """
        # 1次戦略: buy/sellそれぞれのリターン計算
        buy_return, sell_return = self.primary_strategy.calculate_returns(dataframe)

        # リターン > 0 で成功ラベル
        dataframe['&-target_buy'] = (buy_return > 0).astype(int)
        dataframe['&-target_sell'] = (sell_return > 0).astype(int)

        return dataframe
```

## FreqAIマルチターゲット実装

### アーキテクチャ（方式A: 2つの独立モデル）

**実装方針**: Buy/Sell それぞれに独立した FreqAI モデルを訓練

- **Buy モデル**: `TwoTierLightGBMClassifier_Buy(BaseClassifierModel)`
- **Sell モデル**: `TwoTierLightGBMClassifier_Sell(BaseClassifierModel)`
- 各モデルは独立した FreqAI インスタンスとして訓練・予測

### Config 設定例

```json
{
  "freqai_buy": {
    "enabled": true,
    "identifier": "two_tier_buy_v1",
    "model_name": "TwoTierLightGBMClassifier",
    "model_training_parameters": {
      "n_estimators": 100,
      "learning_rate": 0.1
    }
  },
  "freqai_sell": {
    "enabled": true,
    "identifier": "two_tier_sell_v1",
    "model_name": "TwoTierLightGBMClassifier",
    "model_training_parameters": {
      "n_estimators": 100,
      "learning_rate": 0.1
    }
  }
}
```

### TwoTierStrategy での統合

```python
def populate_indicators(self, dataframe, metadata):
    """2つの独立した FreqAI モデルで予測"""
    # Buy モデル予測
    dataframe = self.freqai_buy.start(dataframe, metadata, self)
    dataframe.rename(columns={'&-prediction': '&-prediction_buy'}, inplace=True)

    # Sell モデル予測
    dataframe = self.freqai_sell.start(dataframe, metadata, self)
    dataframe.rename(columns={'&-prediction': '&-prediction_sell'}, inplace=True)

    return dataframe
```

### 利点

- ✅ FreqAI の標準パターンを使用（カスタマイズ不要）
- ✅ 各方向の特性を独立して学習
- ✅ モデル管理・監視が FreqAI 標準機能で可能

## ラベル生成フロー

ML学習時のラベル生成は以下のフローで行われます：

```
┌─────────────────────────────────────────────────────┐
│ 1. FreqAIフレームワークが訓練開始                  │
│    - TwoTierStrategy.set_freqai_targets()を呼び出し│
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│ 2. TwoTierStrategy.set_freqai_targets()             │
│    - primary_strategy.calculate_returns()を実行     │
│    - リターン計算結果をラベル化                     │
│      labels = (returns > 0).astype(int)             │
│    - dataframe['&-target'] = labels                 │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│ 3. FreqAIがラベル付きデータで訓練実行              │
│    - TwoTierLightGBMClassifier.fit()                │
│    - 特徴量（テクニカル指標）→ ラベル（0/1）      │
└─────────────────────────────────────────────────────┘
```

### ラベル生成の実装例（TwoTierStrategy内）

```python
def set_freqai_targets(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    """FreqAI訓練用ラベル生成

    1次戦略のリターン計算結果をラベル化し、FreqAIに渡す
    どの1次戦略と組み合わせても動作する汎用的な実装
    """
    # 1次戦略でリターン計算（約定シミュレーション）
    # ATRBreakout、MeanReversion等、任意の1次戦略が使用可能
    buy_return, sell_return = self.primary_strategy.calculate_returns(dataframe)

    # リターン > 0 で成功ラベル（1）、それ以外は失敗ラベル（0）
    # freqai.identifier でどちらのモデルかを判定
    if self.freqai.identifier.endswith('_buy'):
        dataframe['&-target'] = (buy_return > 0).astype(int)
    else:  # _sell
        dataframe['&-target'] = (sell_return > 0).astype(int)

    return dataframe
```

### ラベル生成のポイント

1. **1次戦略の独立性**: どの1次戦略でも同じラベル生成ロジックが使用可能
2. **Buy/Sell分離**: 両建て対応のため、Buy/Sell独立したラベルを生成
3. **約定シミュレーション**: 1次戦略の`calculate_returns()`が約定シミュレーションを実行
4. **2値分類**: リターン > 0 で成功（1）、それ以外は失敗（0）

## 関連ドキュメント

- [アーキテクチャ設計](./architecture.md) - PrimaryStrategyBaseとTwoTierStrategyの詳細
- [設定管理](./configuration.md) - config.jsonでのFreqAI設定
- [テスト戦略](./testing.md) - ラベル生成の正確性テスト
- [設計判断](./decisions.md) - SecondaryModelBase削除の背景

[⬅️ README に戻る](./README.md)
