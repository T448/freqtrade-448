# design-challenges-strategy-factory.md 修正チェックリスト

## ドキュメント情報

- **対象ファイル**: `docs/memo/design-challenges-strategy-factory.md`
- **目的**: Phase 2設計提案の整合性・明確性向上
- **作成日**: 2025-10-11

---

## 🔴 最優先修正（実装不可能レベルの問題）

### ✅ 1. Phase 1/Phase 2スコープの明示

- [x] 冒頭（Line 1-20付近）に「ドキュメントの位置づけ」セクションを追加
  - 現状: Phase 1リファクタリング実施中
  - 背景: 既存実装の設計意図との乖離
  - 本ドキュメントの目的: Phase 1リファクタリング設計 + Phase 2拡張性考慮
  - Phase 1での対応: ファクトリーパターン基本実装、util層分離、Freqtrade/FreqAI統合
  - Phase 2以降の拡張: 複数戦略、Optuna最適化等
- [x] タイトルに"（Phase 1リファクタリング設計）"を追記

### ✅ 2. IStrategy統合レイヤーの設計変更

- [x] **設計変更**: TwoTierStrategyがIStrategyを直接継承する方式に変更
- [x] atr_ml_strategy.pyを削除し、two_tier_strategy.pyに統一
- [x] TwoTierStrategyクラスの完全書き換え（Line 476-584）
  - [x] IStrategy継承に変更
  - [x] `__init__`: StrategyFactory経由で1次戦略・2次モデルをロード
  - [x] `populate_indicators()`: Freqtradeメソッドシグネチャに修正
  - [x] `populate_entry_trend()`: エントリーシグナル生成（新規追加）
  - [x] `populate_exit_trend()`: エグジットシグナル生成（新規追加）
  - [x] `custom_entry_price()`: 指値価格設定（新規追加）
  - [x] `custom_exit_price()`: 決済価格設定（新規追加）
  - [x] `set_freqai_targets()`: FreqAI訓練用ラベル生成（新規追加）
- [x] ラッパー層を削除（TwoTierStrategyがFreqtradeエントリーポイント）

### ✅ 3. FreqAI統合の具体的なフロー明確化

- [x] 新セクション「FreqAI統合アーキテクチャ」を追加（SecondaryModelBaseの後）
- [x] 以下を明確化:
  - [x] `SecondaryModelBase`はFreqAIモデルのラッパーであることを明記
  - [x] FreqAI使用時の実装構成図を追加:

    ```
    TwoTierStrategy(IStrategy)
    ├── PrimaryStrategyBase (ATRBreakoutStrategy)
    └── SecondaryModelBase (LightGBMClassifier - wrapper)
        └── TwoTierLightGBMClassifier(BaseClassifierModel) ← FreqAIモデル
    ```

  - [x] FreqAIモデル名を`TwoTierLightGBMClassifier`に変更（1次戦略と独立）
  - [x] ラベル生成フローを追加:
    1. 訓練時: `set_freqai_targets()`内で`primary_strategy.calculate_returns()`を呼び出し
    2. 返り値をラベル化: `(returns > 0).astype(int)`
    3. FreqAIのdataframeに`&-target`カラムとして追加
  - [x] Config設定での組み合わせ例を追加

---

## 🟡 高優先修正（実装者の混乱を招く問題）

### ✅ 4. Config構造の統一

- [x] Line 43-68の旧config形式を削除
- [x] 新config形式のみに統一
- [x] 「設計上の課題」セクションから旧configへの言及を削除
- [x] TwoTierStrategy.__init__にバリデーション機能を追加
  - secondary指定時にfreqai.enabled=falseの場合はエラー
  - freqai.enabled=trueだがsecondary=nullの場合は警告

### ✅ 5. 用語の統一

- [ ] 全文を検索し、以下に統一:
  - "Primary Model" → "1次戦略"（または"Primary Strategy"）
  - "Secondary Model" → "2次モデル"
  - クラス名は維持: `PrimaryStrategyBase`, `SecondaryModelBase`
- [ ] 用語集セクションを追加（冒頭付近）:

  ```markdown
  ## 用語定義
  - **1次戦略**: ATRブレイクアウトなど、指値価格とリターンを計算する戦略ロジック
  - **2次モデル**: 1次戦略のリターン予測を行うMLモデル（LightGBMなど）
  - **Primary Strategy**: コード内では`PrimaryStrategyBase`サブクラス
  - **Secondary Model**: コード内では`SecondaryModelBase`サブクラス
  ```

### ✅ 6. fee/hold_periodsパラメータの用途明記

- [ ] Line 598-609のconfigにコメントを追加:

  ```json
  "primary_params": {
      "period": 14,           // ATR計算期間（実トレード用）
      "multiplier": 0.5,      // ATR乗数（実トレード用）
      "execution_mode": "one_candle",  // 約定シミュレーション方法（ラベル生成用）
      "fee": 0.00025,         // 手数料率（ラベル生成用、実トレードはFreqtrade設定）
      "hold_periods": 24      // 保有期間（ラベル生成用、実トレードは2次モデル判定）
  }
  ```

- [ ] Line 251-252の`execution_mode`説明を拡充（ラベル生成専用であることを明記）

### ✅ 7. execution_modeとML有効/無効の区別明確化

- [ ] Line 114-128の「設計原則 - richmanbtc概念との整合性」セクションを修正
- [ ] 以下の2つの概念を分離:

  ```markdown
  ### 約定シミュレーションモード（ラベル生成用）
  - `execution_mode`: "chase" | "one_candle"
  - 目的: 訓練データ生成時のリターン計算方法

  ### 2次モデル有効/無効（実トレード用）
  - `secondary`: "lightgbm_classifier" | null
  - 目的: MLフィルタリングの有無
  ```

### ✅ 8. Freqtrade必須メソッドとの接続

- [x] TwoTierStrategyに必須メソッドを追加（✅2で完了）:
  - [x] `populate_exit_trend()`: 出口シグナル
  - [x] `custom_exit_price()`: 決済指値価格
  - [x] `custom_entry_price()`: エントリー指値価格

---

## 🟢 中優先修正（品質向上）

### ✅ 9. 重複説明の整理

- [ ] 約定シミュレーション説明を統合:
  - Line 131-201: メインの説明として残す（詳細化）
  - Line 238-284: `calculate_returns()`のdocstringは簡潔化、Line 131-201への参照を追加
  - Line 312-341: 実装例として残すが、説明は削減
- [ ] 冒頭に「約定シミュレーションの詳細はLine 131-201参照」と明記

### ✅ 10. テスト要件の配置改善

- [ ] Line 877-930のテスト要件セクションを移動
  - 移動先: Line 816（Phase 1実装範囲の直後）
  - 理由: 実装前にテスト要件を認識すべき
- [ ] セクションタイトルを強調: "⚠️ 重要: テスト要件"
- [ ] Line 928の"全体での網羅性は重視しない"を修正 → "約定シミュレーションの正確性検証が最優先"

### ✅ 11. 設定組み合わせ表の追加

- [ ] Configセクション（Line 594-616の後）に追加:

  ```markdown
  ### 設定の組み合わせパターン

  | freqai.enabled | secondary | 動作モード |
  |---------------|-----------|---------|
  | true | "lightgbm_classifier" | ML予測でフィルタリング（推奨） |
  | true | null | FreqAI有効だがフィルタリングなし（非推奨） |
  | false | "lightgbm_classifier" | エラー（FreqAI必須） |
  | false | null | 1次戦略のみ（ML未使用） |
  ```

### ✅ 12. Optunaセクションの再構成

- [ ] Line 682-741のOptuna説明を簡略化
- [ ] "Phase 2機能プレビュー"サブセクションにまとめる
- [ ] Line 789-808の"実装しないもの"リストから重複削除

### ✅ 13. simple_close戦略の位置づけ明確化

- [ ] Line 344-367の説明を修正:

  ```markdown
  ### simple_close.py（参考実装）

  **用途**: 次足close価格予測用の最小実装例
  **注意**: ATRブレイクアウトとは異なり、指値注文ではなく成行相当の想定
  **目的**: PrimaryStrategyBaseの実装パターン例示
  ```

---

## 📝 修正作業メモ

### 進捗状況

- [x] 最優先修正（3項目） - **完了**
  - [x] ✅1: Phase 1/Phase 2スコープの明示
  - [x] ✅2: IStrategy統合レイヤーの設計変更（TwoTierStrategy直接継承に変更）
  - [x] ✅3: FreqAI統合の具体的なフロー明確化
- [ ] 高優先修正（5項目） - **進行中**
  - [x] ✅4: Config構造の統一とバリデーション追加
  - [ ] ✅5: 用語の統一
  - [ ] ✅6: fee/hold_periodsパラメータの用途明記（進行中）
  - [ ] ✅7: execution_modeとML有効/無効の区別明確化
  - [x] ✅8: Freqtrade必須メソッドとの接続
- [ ] 中優先修正（5項目） - **未着手**

### 作業ログ

**2025-10-11 - アーキテクチャ変更反映**

- TwoTierStrategyをIStrategy直接継承に変更
- StrategyFactory: `create_two_tier_strategy()` → `load_primary()` / `load_secondary()`
- メタ情報: atr_ml_strategy.py削除、two_tier_strategy.py追加
- Phase 1実装範囲: TwoTierStrategyの記述更新
- ドキュメント内のTwoTierStrategyクラス例を完全書き換え（Line 617-742）
- StrategyFactoryクラス例を更新（Line 747-810）

**2025-10-12 - FreqAI統合とConfig構造の明確化**

- FreqAI統合アーキテクチャセクションを追加（Line 453-645）
  - FreqAIモデル名を`TwoTierLightGBMClassifier`に変更（1次戦略と独立）
  - 全体構成図、ラベル生成フロー、Config設定例を追加
- 旧config形式を削除し、新形式のみに統一
- TwoTierStrategy.__init__にバリデーション機能を追加
  - `freqai.enabled`と`secondary`の連動チェック

---

## 確認事項

修正完了後、以下を確認:

- [x] FreqtradeのIStrategyとの統合が明確
- [ ] FreqAIのBaseClassifierModelとの統合が明確（部分完了）
- [x] Phase 1/Phase 2のスコープが冒頭で明示
- [ ] Config形式が統一されている
- [ ] 用語が統一されている
- [ ] 約定シミュレーションとML有効化が区別されている
- [ ] 重複説明が削減されている
- [x] 実装者が迷わない構成になっている（TwoTierStrategy部分）

### アーキテクチャ変更による影響

**旧設計（ラッパー層あり）**:

```
ATRMLStrategy(IStrategy) ← Freqtradeエントリーポイント
└── TwoTierStrategy ← ヘルパークラス
    ├── PrimaryStrategyBase
    └── SecondaryModelBase
```

**新設計（直接統合）**:

```
TwoTierStrategy(IStrategy) ← Freqtradeエントリーポイント
├── PrimaryStrategyBase (StrategyFactory.load_primary()でロード)
└── SecondaryModelBase (StrategyFactory.load_secondary()でロード)
```

**実行方法**:

```bash
freqtrade backtesting --strategy TwoTierStrategy --config config.json
```
