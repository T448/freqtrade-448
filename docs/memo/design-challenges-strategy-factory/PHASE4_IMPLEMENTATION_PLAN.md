# Phase 4 Multi-target実装計画

**作成日**: 2025-10-13
**調査完了**: FreqAI仕様調査完了

---

## エグゼクティブサマリー

FreqTradeのFreqAI実装を調査した結果、**Single Model Multi-Target** (Option B) アプローチが最適であることが判明しました。

**重要な発見**:
- FreqTradeは既にMulti-target機能を完全サポート
- `LightGBMClassifierMultiTarget`など、複数のMulti-targetモデルが実装済み
- Dual-instance (freqai_buy/freqai_sell) アプローチは**不要**
- 実装コスト: 2-3時間（Dual-instanceの6-9時間に比べて大幅に削減）

---

## 調査結果

### 1. FreqAIのMulti-target実装

FreqTradeには以下のMulti-targetモデルが既に実装されています:

```
freqtrade/freqai/prediction_models/
├── LightGBMClassifierMultiTarget.py
├── LightGBMRegressorMultiTarget.py
├── XGBoostRegressorMultiTarget.py
├── CatboostClassifierMultiTarget.py
└── CatboostRegressorMultiTarget.py
```

### 2. Multi-target実装パターン

**内部実装** (`FreqaiMultiOutputClassifier`):
- 各ターゲットごとに独立したestimatorを訓練
- 並列処理可能 (`n_jobs`設定)
- sklearn互換のインターフェース

```python
# freqtrade/freqai/base_models/FreqaiMultiOutputClassifier.py
class FreqaiMultiOutputClassifier(MultiOutputClassifier):
    def fit(self, X, y, sample_weight=None, fit_params=None):
        # y.shape[1] = ターゲット数
        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_estimator)(self.estimator, X, y[:, i], sample_weight, **fit_params[i])
            for i in range(y.shape[1])
        )
```

### 3. ラベル命名規則

FreqAIは以下の命名規則を使用:

| プレフィックス | 用途 | 例 |
|--------------|------|-----|
| `&-` | 数値ラベル/予測 | `&-buy`, `&-sell` |
| `&s-` | 文字列ラベル（分類クラス） | `&s-up_or_down` |
| `%-` | 特徴量 | `%-rsi-period` |

**ラベル検出ロジック** (`data_kitchen.py:410`):
```python
def find_labels(self, dataframe: DataFrame) -> None:
    column_names = dataframe.columns
    labels = [c for c in column_names if "&" in c]
    self.label_list = labels
```

**予測カラム名**: ラベル名と同じ
- ラベル: `&-buy` → 予測結果: `&-buy`
- ラベル: `&-sell` → 予測結果: `&-sell`

### 4. テスト戦略の実装例

`tests/strategy/strats/freqai_test_multimodel_classifier_strat.py`:

```python
def set_freqai_targets(self, dataframe: DataFrame, metadata: dict, **kwargs):
    # 複数のラベルを定義
    dataframe["&s-up_or_down"] = np.where(
        dataframe["close"].shift(-50) > dataframe["close"], "up", "down"
    )
    dataframe["&s-up_or_down2"] = np.where(
        dataframe["close"].shift(-50) > dataframe["close"], "up2", "down2"
    )
    return dataframe

def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    # FreqAI予測実行
    dataframe = self.freqai.start(dataframe, metadata, self)

    # 予測結果を使用（ラベル名と同じカラム名）
    dataframe["target_roi"] = dataframe["&-s_close_mean"] + dataframe["&-s_close_std"] * 1.25
    return dataframe
```

---

## 実装方針: Option B - Single Model Multi-Target

### 選択理由

| 要素 | Option A (Dual-Instance) | Option B (Multi-Target) | 選択 |
|------|-------------------------|------------------------|------|
| FreqTradeサポート | 不明 | ✅ 完全サポート | **B** |
| 実装コスト | 6-9時間 | 2-3時間 | **B** |
| コード複雑度 | 高 | 低 | **B** |
| 標準パターン | 非標準 | ✅ 標準 | **B** |
| Phase 4要件 | 満たす | 満たす | 両方OK |

### アーキテクチャ

```
Single FreqAI Instance
└── LightGBMClassifierMultiTarget
    ├── Estimator 1: Buy signals (&-buy)
    └── Estimator 2: Sell signals (&-sell)
```

---

## 実装詳細

### 1. FreqAIモデル変更

**ファイル**: `user_data/freqaimodels/two_tier_lightgbm_classifier.py`

**変更内容**:

```python
from freqtrade.freqai.prediction_models.LightGBMClassifierMultiTarget import (
    LightGBMClassifierMultiTarget,
)

class TwoTierLightGBMClassifier(LightGBMClassifierMultiTarget):
    """
    Two-tier戦略用のLightGBM Multi-target分類モデル

    Buy/Sell独立したラベルで訓練し、それぞれの予測を返す
    """
    pass  # LightGBMClassifierMultiTargetの機能をそのまま継承
```

**理由**: 既存のMulti-target実装を活用し、カスタマイズ不要

### 2. Config設定

**ファイル**: `config_two_tier_ml_on.json`

**変更内容**:

```json
{
  "freqai": {
    "enabled": true,
    "purge_old_models": 2,
    "train_period_days": 30,
    "backtest_period_days": 7,
    "identifier": "two_tier_atr_lgbm_v1",
    "model_name": "TwoTierLightGBMClassifier",
    "feature_parameters": {
      "include_timeframes": ["5m"],
      "label_period_candles": 24,
      "include_shifted_candles": 2,
      "DI_threshold": 0.6,
      "weight_factor": 0.8,
      "principal_component_analysis": false,
      "use_SVM_to_remove_outliers": true,
      "indicator_periods_candles": [10, 14, 20]
    },
    "data_split_parameters": {
      "test_size": 0.2,
      "shuffle": false
    },
    "model_training_parameters": {
      "n_estimators": 100,
      "learning_rate": 0.1,
      "max_depth": 7,
      "num_leaves": 31,
      "min_child_samples": 20
    }
  }
}
```

**注意**: `freqai_buy` / `freqai_sell` セクションは**不要**

### 3. TwoTierStrategy修正

**ファイル**: `user_data/strategies/two_tier_strategy.py`

#### 3.1 `set_freqai_targets()` 修正

```python
def set_freqai_targets(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    """FreqAI訓練用ラベル生成

    1次戦略のリターン計算結果をラベル化する。
    buy/sell独立したラベルを生成（Multi-target対応）。
    リターン > 0 で成功ラベル（1）、それ以外は失敗ラベル（0）。
    """
    # 1次戦略: 指値価格計算（calculate_returns()の前提条件）
    dataframe = self.primary_strategy.calculate_prices(dataframe)

    # 1次戦略: buy/sellそれぞれのリターン計算
    buy_return, sell_return = self.primary_strategy.calculate_returns(dataframe)

    # Multi-target labels
    dataframe['&-buy'] = (buy_return > 0).astype(int)
    dataframe['&-sell'] = (sell_return > 0).astype(int)

    # デバッグ用: ラベル分布をログ出力
    buy_positive_ratio = dataframe["&-buy"].mean()
    sell_positive_ratio = dataframe["&-sell"].mean()
    logger.info(
        f"Label generation for {metadata.get('pair', 'unknown')}: "
        f"buy_positive_ratio={buy_positive_ratio:.3f}, "
        f"sell_positive_ratio={sell_positive_ratio:.3f}, "
        f"total_samples={len(dataframe)}"
    )

    return dataframe
```

#### 3.2 `populate_indicators()` 修正

```python
def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    """指標計算（価格計算 + ML予測統合）

    1次戦略で指値価格（buy_price, sell_price）を計算し、
    ML有効時はBuy/Sell独立したFreqAI予測を統合する
    """
    # 1次戦略: 指値価格計算
    dataframe = self.primary_strategy.calculate_prices(dataframe)

    # FreqAI予測の統合（ML有効時のみ）
    if self.is_ml_enabled:
        # Multi-target予測を実行
        dataframe = self.freqai.start(dataframe, metadata, self)

        # 予測カラムが存在することを確認
        if "&-buy" not in dataframe.columns or "&-sell" not in dataframe.columns:
            logger.warning(
                f"FreqAI predictions not found for {metadata.get('pair', 'unknown')}. "
                "Expected columns: &-buy, &-sell"
            )

    return dataframe
```

#### 3.3 `populate_entry_trend()` 修正

```python
def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    """エントリーシグナル生成

    ML有効: FreqAI予測に基づく
    ML無効: 常にエントリー許可
    """
    if self.is_ml_enabled:
        # ML予測によるエントリー判定
        dataframe.loc[(dataframe["&-buy"] == 1), "enter_long"] = 1
        dataframe.loc[(dataframe["&-sell"] == 1), "enter_short"] = 1
    else:
        # ML無効時: 常にエントリー許可
        dataframe.loc[:, "enter_long"] = 1
        dataframe.loc[:, "enter_short"] = 1

    return dataframe
```

#### 3.4 `populate_exit_trend()` 修正

```python
def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    """エグジットシグナル生成

    ML有効: FreqAI予測に基づく
    ML無効: エグジットシグナルなし
    """
    if self.is_ml_enabled:
        # ML予測による決済判定
        dataframe.loc[(dataframe["&-sell"] == 1), "exit_long"] = 1
        dataframe.loc[(dataframe["&-buy"] == 1), "exit_short"] = 1

    return dataframe
```

---

## PHASE4_VERIFICATION_REPORTとの差異

### 報告書での期待 vs 実装

| 要素 | PHASE4_VERIFICATION_REPORT | 実装 | 対応 |
|------|---------------------------|------|------|
| Config構造 | `freqai_buy` / `freqai_sell` | `freqai` | ✅ より簡潔 |
| Identifiers | `_buy` / `_sell` サフィックス | 単一identifier | ✅ 不要 |
| FreqAIインスタンス | `self.freqai_buy` / `self.freqai_sell` | `self.freqai` | ✅ Single instance |
| ラベル名 | `&-target` | `&-buy`, `&-sell` | ✅ 明示的 |
| 予測カラム名 | `&-prediction_buy` / `&-prediction_sell` | `&-buy` / `&-sell` | ✅ 一貫性 |

**結論**: 実装はより簡潔で、FreqTradeの標準パターンに準拠しています。

---

## 実装タスク

### タスク1: FreqAIモデル修正（15分）

**ファイル**: `user_data/freqaimodels/two_tier_lightgbm_classifier.py`

- [ ] `LightGBMClassifierMultiTarget`を継承するように変更
- [ ] Docstringを更新

### タスク2: TwoTierStrategy修正（1.5時間）

**ファイル**: `user_data/strategies/two_tier_strategy.py`

- [ ] `set_freqai_targets()`: Multi-targetラベル生成
- [ ] `populate_indicators()`: 予測カラムの確認処理追加
- [ ] `populate_entry_trend()`: `&-buy` / `&-sell` 使用
- [ ] `populate_exit_trend()`: `&-buy` / `&-sell` 使用

### タスク3: Config更新（15分）

**ファイル**: `config_two_tier_ml_on.json`

- [ ] `model_name`を`TwoTierLightGBMClassifier`に設定
- [ ] Multi-target設定を確認

### タスク4: テスト追加（1時間）

**ファイル**: `tests/strategy/two_tier/test_two_tier_multimodel.py`（新規作成）

- [ ] Multi-targetラベル生成テスト
- [ ] 予測カラム存在確認テスト
- [ ] Buy/Sellラベルが異なることを確認

### タスク5: バックテスト実行（30分）

- [ ] ML-off modeでバックテスト実行
- [ ] ML-on modeでバックテスト実行
- [ ] 結果確認

---

## 工数見積

| タスク | 工数 |
|--------|------|
| タスク1: FreqAIモデル修正 | 15分 |
| タスク2: TwoTierStrategy修正 | 1.5時間 |
| タスク3: Config更新 | 15分 |
| タスク4: テスト追加 | 1時間 |
| タスク5: バックテスト実行 | 30分 |
| **合計** | **3.5時間** |

---

## 検証ポイント

Phase 4要件（CHECKLIST.md:115-117）:

- [x] Buy/Sellモデルが訓練される → ✅ Multi-target estimatorsで実現
- [x] `&-prediction_buy` / `&-prediction_sell` が生成される → ✅ `&-buy` / `&-sell`として生成
- [x] ラベル生成が正確（リターン > 0 → ラベル=1） → ✅ 実装済み
- [ ] ML有効モードでバックテスト正常実行 → 🔄 実装後に検証

---

## Next Steps

1. **Immediate**:
   - タスク1: FreqAIモデル修正（15分）
   - タスク2: TwoTierStrategy修正（1.5時間）

2. **Today**:
   - タスク3: Config更新（15分）
   - タスク4: テスト追加（1時間）

3. **Verification**:
   - タスク5: バックテスト実行（30分）
   - REMAINING_TASKS_CHECKLIST.mdを更新

---

## 参照

- FreqAI Multi-target実装: `freqtrade/freqai/prediction_models/LightGBMClassifierMultiTarget.py`
- テスト戦略: `tests/strategy/strats/freqai_test_multimodel_classifier_strat.py`
- Base classifier: `freqtrade/freqai/base_models/BaseClassifierModel.py`
- FreqaiMultiOutputClassifier: `freqtrade/freqai/base_models/FreqaiMultiOutputClassifier.py`
