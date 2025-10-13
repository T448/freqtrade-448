# 残タスク実装チェックリスト

## 概要

Phase 1-5の検証レポートから抽出した**単純実装タスク**のチェックリストです。これらは追加の判断を必要とせず、仕様に従って実装できるタスクです。

判断が必要なタスクについては `REMAINING_TASKS_DECISIONS.md` を参照してください。

詳細は以下のファイルを確認してください。

- docs/memo/design-challenges-strategy-factory/PHASE1_VERIFICATION_REPORT.md
- docs/memo/design-challenges-strategy-factory/PHASE2_VERIFICATION_REPORT.md
- docs/memo/design-challenges-strategy-factory/PHASE3_VERIFICATION_REPORT.md
- docs/memo/design-challenges-strategy-factory/PHASE4_VERIFICATION_REPORT.md
- docs/memo/design-challenges-strategy-factory/PHASE5_VERIFICATION_REPORT.md

---

## 完成度サマリー

| Phase | 完成度 | 主な残タスク |
|-------|--------|-------------|
| Phase 1 | 95% | Config検証修正、テスト調整 |
| Phase 2 | 95% | モジュールパス修正 |
| Phase 3 | 75% | **config.json構造修正 (Critical)** |
| Phase 4 | 40% | Multi-target実装 (→DECISIONS.md) |
| Phase 5 | 75% | MLLabelGenerator実装 |

**総合完成度: 76%**

---

## Priority 1: Critical (ブロッカー)

これらのタスクはバックテスト実行をブロックしているため、最優先で対応が必要です。

### ✅ Task 1.1: Config.json構造をPhase 3仕様に更新

**参照**: `PHASE3_VERIFICATION_REPORT.md`

**問題**: 現在のconfig.jsonがPhase 3仕様と一致しない (Config compliance: 0%)

**現在の構造**:

```json
{
  "strategy": "TwoTierStrategy",
  "two_tier": {
    "primary_model": {
      "type": "atr",
      "params": {...}
    }
  }
}
```

**Phase 3仕様**:

```json
{
  "strategy": "TwoTierStrategy",
  "two_tier": {
    "primary": "atr_breakout",
    "atr_breakout": {
      "atr_period": 14,
      "atr_multiplier": 2.0,
      "volume_ma_period": 20,
      "volatility_threshold": 0.02
    },
    "fee": 0.001,
    "exit_periods": [12, 24, 48],
    "pips": [0.01, 0.02, 0.03],
    "execution_mode": "price_only"
  }
}
```

**修正内容**:

1. `primary_model` → `primary`
2. `type: "atr"` → `primary: "atr_breakout"`
3. 以下のパラメータを追加:
   - `fee: 0.001`
   - `exit_periods: [12, 24, 48]`
   - `pips: [0.01, 0.02, 0.03]`
   - `execution_mode: "price_only"`

**ファイル**: `config.json`

**所要時間**: 15分

**検証方法**:

```bash
# Config読み込みテストを実行
pytest tests/strategy/two_tier/ -v -k config
```

- [ ] config.json構造を修正
- [ ] テストで検証
- [ ] バックテスト実行可能であることを確認

---

### ✅ Task 1.2: Config検証の修正 (Phase 1)

**参照**: `PHASE1_VERIFICATION_REPORT.md`

**問題**: 以下の2つのテストが失敗

1. `test_invalid_config_missing_freqai` - FreqAIセクション存在チェック未実装
2. `test_config_with_secondary_as_empty_string` - 空文字列ハンドリング未実装

**修正箇所**: `user_data/strategies/two_tier/config_validator.py`

**修正内容**:

```python
def validate_two_tier_config(config: Dict[str, Any]) -> None:
    """Two-tier戦略のconfig検証"""

    # 既存のチェック...

    # 🆕 FreqAIセクション存在チェックを追加
    if config["two_tier"].get("execution_mode") == "ml_enhanced":
        if "freqai" not in config:
            raise ValueError(
                "execution_mode='ml_enhanced' requires 'freqai' section in config"
            )

    # 🆕 Secondary戦略の空文字列チェックを追加
    secondary = config["two_tier"].get("secondary")
    if secondary is not None and secondary == "":
        raise ValueError(
            "two_tier.secondary cannot be empty string. "
            "Use null/None to disable secondary strategy."
        )
```

**所要時間**: 30分

**検証方法**:

```bash
pytest tests/strategy/two_tier/test_config_validator.py -v
```

- [ ] FreqAI存在チェックを実装
- [ ] 空文字列ハンドリングを実装
- [ ] テストで検証

---

### ✅ Task 1.3: Config検証の修正 (Phase 5)

**参照**: `PHASE5_VERIFICATION_REPORT.md`

**問題**: Phase 1と同じく、config検証テストが2つ失敗

**修正箇所**: Task 1.2と同じファイル

**所要時間**: Task 1.2に含まれる

- [ ] Task 1.2の修正で対応完了

---

## Priority 2: High (機能実装)

### ✅ Task 2.1: MLLabelGenerator実装 (Phase 5)

**参照**: `PHASE5_VERIFICATION_REPORT.md`

**問題**: MLLabelGenerator moduleが未実装 (TDD RED phase)

**実装内容**:
テストファイル `tests/strategy/two_tier/test_ml_label_generator.py` に基づいて実装

**必要な機能**:

1. 価格ベース特徴量生成
2. テクニカル指標ベース特徴量生成
3. リターン計算
4. ラベル生成 (buy/sell signals)

**ファイル**: `user_data/strategies/two_tier/ml/label_generator.py`

**実装の参考**:

```python
from typing import Dict, List
import pandas as pd

class MLLabelGenerator:
    """ML用のラベル生成"""

    def __init__(self, config: Dict):
        self.label_period = config.get("label_period_candles", 24)
        self.buy_threshold = config.get("buy_threshold", 0.02)
        self.sell_threshold = config.get("sell_threshold", -0.02)

    def generate_labels(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """ラベル生成のメイン処理"""
        # 実装の詳細はテストを参照
        pass

    def calculate_future_returns(
        self,
        dataframe: pd.DataFrame,
        period: int
    ) -> pd.Series:
        """将来リターンの計算"""
        pass
```

**所要時間**: 2-3時間

**検証方法**:

```bash
pytest tests/strategy/two_tier/test_ml_label_generator.py -v
```

- [ ] MLLabelGeneratorクラスを実装
- [ ] 価格ベース特徴量生成を実装
- [ ] テクニカル指標ベース特徴量生成を実装
- [ ] リターン計算を実装
- [ ] ラベル生成を実装
- [ ] テストで検証

---

### ✅ Task 2.2: Return計算の実装修正 (.shift(-n) を使用) 🆕

**参照**: `PHASE5_VERIFICATION_REPORT.md`, `REMAINING_TASKS_DECISIONS.md` Item 1

**問題**: `test_calculate_returns_uses_shift_negative` が失敗

- テストが期待: `.shift(-n)` を使用したreturn計算
- 現在の実装: `.iloc` スライシングを使用

**決定事項**: ✅ **Option A採用** - 実装を `.shift(-n)` に変更

- 理由: `.shift(-n)` はindexを維持するため、`.values` でindexを落とすより安全性が高い

**修正箇所**: `user_data/strategies/two_tier/data_leak_detector.py` (または関連ファイル)

**修正内容**:

```python
# Before
future_prices = prices.iloc[n:].values
current_prices = prices.iloc[:-n].values
returns = (future_prices - current_prices) / current_prices

# After
future_prices = prices.shift(-n)
returns = (future_prices - prices) / prices
```

**所要時間**: 30分

**検証方法**:

```bash
pytest tests/strategy/two_tier/test_data_leak_detector.py::test_calculate_returns_uses_shift_negative -v
```

- [ ] 実装を `.shift(-n)` に変更
- [ ] テストで検証

---

### ✅ Task 2.3: 最後N行の動作修正 (NaN/0統一) 🆕

**参照**: `PHASE5_VERIFICATION_REPORT.md`, `REMAINING_TASKS_DECISIONS.md` Item 2

**問題**: `test_return_calculation_uses_future_data_correctly` が失敗

- テストが期待: 最後N行は「すべてNaN」または「すべて0」
- 現在の実装: NaNと0が混在

**決定事項**: ✅ **Option A採用** - 実装を修正してすべてNaN/0に統一

- 調査: 実装コードを読んで動作理由を特定
- 仕様確認: 元の設計意図を確認
- 決定: すべてNaN or すべて0に統一

**修正箇所**: `user_data/strategies/two_tier/data_leak_detector.py`

**修正内容**:

1. 実装コードを読んで0とNaNが混在する理由を特定
2. 設計意図に基づいて、以下のいずれかに統一:
   - すべてNaN (データ不足を明示)
   - すべて0 (デフォルト値として扱う)

**所要時間**: 1-2時間 (調査込み)

**検証方法**:

```bash
pytest tests/strategy/two_tier/test_data_leak_detector.py::test_return_calculation_uses_future_data_correctly -v
```

- [ ] 実装コードを読んで動作理由を特定
- [ ] 元の設計仕様を確認
- [ ] すべてNaN or すべて0に統一する実装に修正
- [ ] テストで検証

---

## Priority 3: Medium (検証・テスト)

### ✅ Task 3.1: ML-off Backtest実行 (Phase 3)

**参照**: `PHASE3_VERIFICATION_REPORT.md`

**目的**: Phase 3実装の動作確認

**前提条件**:

- Task 1.1 (config.json修正) 完了
- Task 1.2 (config検証) 完了

**実行コマンド**:

```bash
STRATEGY_TYPE=price_only freqtrade backtesting \
  --config config.json \
  --strategy TwoTierStrategy \
  --timerange 20240101-20240131
```

**検証項目**:

1. ✅ Primary strategyが正しく動作
2. ✅ ATR breakout orderが正しく生成
3. ✅ Config validationがパス
4. ✅ エラーなく完了

**所要時間**: 30分 (実行 + 結果確認)

- [ ] 実行環境を確認
- [ ] ML-off backtestを実行
- [ ] 結果を確認
- [ ] エラーがあれば修正

---

### ✅ Task 3.2: Phase 1-2 統合テスト実行

**参照**:

- `PHASE1_VERIFICATION_REPORT.md`
- `PHASE2_VERIFICATION_REPORT.md`

**目的**: すべてのPhase 1-2テストがパスすることを確認

**実行コマンド**:

```bash
# Phase 1 tests
pytest tests/strategy/two_tier/test_data_leak_detector.py -v
pytest tests/strategy/two_tier/test_config_validator.py -v

# Phase 2 tests
pytest tests/strategy/two_tier/test_strategy_factory.py -v
```

**前提条件**:

- Task 1.2 (config検証) 完了
- Task 2.2 (data leak検証) 完了

**所要時間**: 30分

- [ ] Phase 1テストを実行
- [ ] Phase 2テストを実行
- [ ] すべてのテストがパスすることを確認

---

### ✅ Task 3.3: Phase 5 統合テスト実行

**参照**: `PHASE5_VERIFICATION_REPORT.md`

**目的**: すべてのPhase 5テストがパスすることを確認

**実行コマンド**:

```bash
pytest tests/strategy/two_tier/test_ml_label_generator.py -v
pytest tests/strategy/two_tier/test_data_leak_detector.py -v
pytest tests/strategy/two_tier/test_config_validator.py -v
```

**前提条件**:

- Task 2.1 (MLLabelGenerator) 完了
- Task 1.2 (config検証) 完了
- Task 2.2 (data leak検証) 完了

**所要時間**: 30分

- [ ] MLLabelGeneratorテストを実行
- [ ] Data leak検証テストを実行
- [ ] Config検証テストを実行
- [ ] すべてのテストがパスすることを確認

---

## Priority 4: Low (マイナー修正)

### ✅ Task 4.1: モジュールパス修正 (Phase 2)

**参照**: `PHASE2_VERIFICATION_REPORT.md`

**問題**: `test_load_primary_atr_breakout_success` が失敗

- 期待: `user_data.strategies.primary...`
- 実際: `strategies.primary...`

**修正箇所**: `user_data/strategies/two_tier/strategy_factory.py`

**修正内容**:

```python
# Before
module_path = f"strategies.primary.{strategy_type}"

# After
module_path = f"user_data.strategies.primary.{strategy_type}"
```

**注意**: この修正は機能に影響しない可能性があるため、優先度は低い

**所要時間**: 5分

**検証方法**:

```bash
pytest tests/strategy/two_tier/test_strategy_factory.py::test_load_primary_atr_breakout_success -v
```

- [ ] モジュールパスを修正
- [ ] テストで検証

---

### ✅ Task 4.2: CHECKLIST.md更新

**目的**: 元のチェックリストを更新し、完了済み項目をマーク

**ファイル**: `docs/memo/design-challenges-strategy-factory/CHECKLIST.md`

**更新内容**:

1. Phase 1-5の完了項目にチェックマーク
2. 残タスクセクションを追加
3. 検証レポートへの参照を追加

**所要時間**: 15分

- [ ] CHECKLISTを確認
- [ ] 完了項目をマーク
- [ ] 残タスクセクションを追加

---

## Priority 5: Deferred (判断待ち)

### ✅ Task 5.1: Phase 4 Multi-target実装 🆕

**参照**:

- `PHASE4_VERIFICATION_REPORT.md`
- `REMAINING_TASKS_DECISIONS.md` の Item 3
- `PHASE4_IMPLEMENTATION_PLAN.md` ✅ **調査完了**

**調査完了**: ✅ 2025-10-13

**決定事項**: ✅ **Option B - Single Model Multi-Target**

**調査結果**:

- ✅ FreqTradeは既にMulti-target機能を完全サポート
- ✅ `LightGBMClassifierMultiTarget`など、複数のMulti-targetモデルが実装済み
- ✅ Dual-instance (freqai_buy/freqai_sell) アプローチは不要
- ✅ 実装コスト: **3.5時間** (Dual-instanceの6-9時間から大幅削減)

**実装方針**:

```
Single FreqAI Instance
└── LightGBMClassifierMultiTarget
    ├── Estimator 1: Buy signals (&-buy)
    └── Estimator 2: Sell signals (&-sell)
```

**実装タスク** (合計: 3.5時間):

1. **FreqAIモデル修正** (15分)
   - [ ] `LightGBMClassifierMultiTarget`を継承

2. **TwoTierStrategy修正** (1.5時間)
   - [ ] `set_freqai_targets()`: Multi-targetラベル生成 (`&-buy`, `&-sell`)
   - [ ] `populate_indicators()`: 予測カラム確認処理追加
   - [ ] `populate_entry_trend()`: `&-buy` / `&-sell` 使用
   - [ ] `populate_exit_trend()`: `&-buy` / `&-sell` 使用

3. **Config更新** (15分)
   - [ ] `model_name`を`TwoTierLightGBMClassifier`に設定

4. **テスト追加** (1時間)
   - [ ] Multi-targetラベル生成テスト
   - [ ] 予測カラム存在確認テスト

5. **バックテスト実行** (30分)
   - [ ] ML-on modeでバックテスト実行

**検証方法**:

```bash
# テスト実行
pytest tests/strategy/two_tier/test_two_tier_multimodel.py -v

# バックテスト実行
freqtrade backtesting --strategy TwoTierStrategy --config config_two_tier_ml_on.json
```

**詳細**: `PHASE4_IMPLEMENTATION_PLAN.md` を参照

---

### 🔄 Task 5.2: ML-on Backtest実行

**前提条件**:

- Task 5.1 (Phase 4 Multi-target) 完了
- または、Single modelでのML-on実装完了

**実行コマンド**:

```bash
freqtrade backtesting \
  --config config.json \
  --strategy TwoTierStrategy \
  --timerange 20240101-20240131
```

**所要時間**: 1-2時間 (実行 + 結果確認)

- [ ] Phase 4実装完了を確認
- [ ] FreqAI modelを学習
- [ ] ML-on backtestを実行
- [ ] 結果を確認

---

## 実装順序の推奨

### Week 1: Critical fixes

```
Day 1 (2時間):
├── Task 1.1: config.json修正 (15分)
├── Task 1.2: Config検証修正 (30分)
├── Task 3.1: ML-off backtest (30分)
└── Task 4.2: CHECKLIST更新 (15分)

Day 2-3 (4時間):
└── Task 2.1: MLLabelGenerator実装 (3時間)
    └── Task 3.3: Phase 5テスト実行 (30分)
```

### Week 2: Data leak fixes & verification 🆕

```
Day 4 (2.5時間):
├── Task 2.2: Return計算の実装修正 (30分) ✅ 決定済み
├── Task 2.3: 最後N行の動作修正 (1.5時間) ✅ 決定済み
├── Task 3.2: Phase 1-2テスト (30分)
└── Task 4.1: モジュールパス修正 (5分)

Day 5+ (2-9時間):
└── Task 5.1: Phase 4 Multi-target ✅ 方針決定済み
    ├── FreqAI実装調査 (1-2時間)
    ├── 実装 (2-9時間、方法による)
    └── Task 5.2: ML-on backtest (1-2時間)
```

---

## 進捗トラッキング

### チェックリストサマリー

- **Priority 1 (Critical)**: 3タスク
  - [ ] Task 1.1: config.json修正
  - [ ] Task 1.2: Config検証修正 (Phase 1)
  - [ ] Task 1.3: Config検証修正 (Phase 5)

- **Priority 2 (High)**: 3タスク 🆕
  - [ ] Task 2.1: MLLabelGenerator実装
  - [ ] Task 2.2: Return計算の実装修正 (.shift(-n))
  - [ ] Task 2.3: 最後N行の動作修正 (NaN/0統一)

- **Priority 3 (Medium)**: 3タスク
  - [ ] Task 3.1: ML-off backtest
  - [ ] Task 3.2: Phase 1-2統合テスト
  - [ ] Task 3.3: Phase 5統合テスト

- **Priority 4 (Low)**: 2タスク
  - [ ] Task 4.1: モジュールパス修正
  - [ ] Task 4.2: CHECKLIST更新

- **Priority 5 (Deferred)**: 2タスク
  - [ ] Task 5.1: Phase 4 Multi-target (判断待ち)
  - [ ] Task 5.2: ML-on backtest (Phase 4完了後)

**合計**: 13タスク (11 actionable + 2 deferred) 🆕

---

## Next Steps

1. **Immediate (今すぐ開始)**:
   - Task 1.1: config.json修正
   - Task 1.2: Config検証修正

2. **This Week (今週中)**:
   - Task 2.1: MLLabelGenerator実装
   - Task 2.2: Return計算の実装修正 ✅ 方針決定済み
   - Task 2.3: 最後N行の動作修正 ✅ 方針決定済み
   - Task 3.1: ML-off backtest実行

3. **Decisions Completed (判断完了)**: ✅ 🎉
   - ✅ Item 1: Return計算方法 → Option A (.shift(-n)使用)
   - ✅ Item 2: 最後N行の動作 → Option A (NaN/0統一)
   - ✅ Item 3: Phase 4 Multi-target → 実装可能な方法を選択

4. **Next Phase (次のフェーズ)**:
   - Phase 4 Multi-target実装 (FreqAI調査 → 実装)
   - ML-on backtest実行
