# 残タスク: 判断が必要な項目

## 概要

Phase 1-5の検証レポートから、**実装前に設計判断や調査が必要な3つの項目**を抽出しました。これらは単純な実装タスクではなく、アーキテクチャやテスト設計に関する意思決定が必要です。

---

## 1. Return計算の実装方法 (Phase 1)

### 背景

- テスト: `test_calculate_returns_uses_shift_negative`
- 期待: `.shift(-n)` を使用したreturn計算
- 現状: `.iloc` スライシングを使用した実装

### 現在の状況

```python
# 現在の実装 (動作は正しい)
future_prices = prices.iloc[n:].values
current_prices = prices.iloc[:-n].values
returns = (future_prices - current_prices) / current_prices
```

テストは `.shift(-n)` の使用を期待:

```python
assert ".shift(-" in source_code
```

### 選択肢

**Option A: 実装を変更 (.shift(-n) を使用)**

- メリット: テストが通る
- デメリット:
  - 現在の実装も正しく動作している
  - コード変更リスク
  - パフォーマンスへの影響不明

**Option B: テストを変更 (動作を検証)**

- メリット:
  - 実装変更不要
  - 実際の動作を検証するテストに改善
- デメリット: テスト設計の見直しが必要

### 推奨

**Option B: テストを変更**

理由:

1. 現在の実装は正しく動作している
2. `.shift(-n)` の使用を強制する必要性が不明確
3. 実装の内部詳細ではなく、動作を検証すべき

### 実装への影響

- 優先度: **Low**
- 影響範囲: テストコードのみ
- 所要時間: 30分

### 決定事項

- [x] 実装方法を決定 (Option A or B)
  - **決定: Option A (.shift(-n) を使用)** ✅
  - 理由: `.shift(-n)` はindexを維持するため、`.values` でindexを落とすより安全性が高い
- [ ] 決定に基づいて実装を修正

---

## 2. 最後のN行の動作検証 (Phase 1)

### 背景

- テスト: `test_return_calculation_uses_future_data_correctly`
- 期待: 最後のN行は「すべてNaN」または「すべて0」
- 現状: NaNと0が混在

### 現在の状況

実装は以下のような混在値を返している:

```
[0.0, 0.0, nan, nan, nan, ...]
```

テストの期待:

```python
assert all_nan or all_zero  # すべてNaNまたはすべて0
```

### 問題の本質

これは**実装の不具合かテスト設計の問題か**を判断する必要がある。

### 調査が必要な項目

1. **仕様の確認**
   - 元の設計書で最後N行の動作はどう定義されているか?
   - 「データが不足している場合」と「埋め合わせができない場合」で異なる値を返すべきか?

2. **実装の意図**
   - 現在の実装で0とNaNが混在する理由は何か?
   - これは意図的な設計か、バグか?

3. **ビジネスロジック**
   - データリーク防止の観点から、0とNaNのどちらが適切か?
   - 下流の処理(ML特徴量生成など)への影響は?

### 推奨アプローチ

1. 実装コードを読み、0とNaNが混在する理由を特定
2. 元の設計意図を確認
3. 以下のいずれかを選択:
   - **Option A**: 実装を修正してすべてNaN/0に統一
   - **Option B**: テストを修正して混在を許容

### 実装への影響

- 優先度: **Medium**
- 影響範囲: DataLeakDetector + テスト
- 所要時間: 1-2時間(調査込み)

### 決定事項

- [x] 実装コードを読んで動作理由を特定
- [x] 元の設計仕様を確認
- [x] 修正方針を決定 (実装 or テスト)
  - **決定: Option A (実装を修正してすべてNaN/0に統一)** ✅
- [ ] 決定に基づいて修正

---

## 3. Phase 4 Multi-target Architecture (Phase 4) 🔴 **Critical** ✅ **調査完了**

### 背景

Phase 4の検証レポートで**最大の問題**として指摘されている項目:

- 現状: Single FreqAI model (基本実装、40%完成)
- 仕様: Dual FreqAI instances (Buy/Sell独立モデル)

### 調査完了 ✅ 2025-10-13

**調査結果**: FreqTradeは既に**Multi-target機能を完全サポート**

詳細な調査結果と実装計画は **`PHASE4_IMPLEMENTATION_PLAN.md`** を参照。

### 決定した実装方針

**✅ Option B: Single Model Multi-Target**

```
Single FreqAI Instance
└── LightGBMClassifierMultiTarget
    ├── Estimator 1: Buy signals (&-buy)
    └── Estimator 2: Sell signals (&-sell)
```

### 主要な発見

1. **FreqTradeのMulti-target実装**:
   - `LightGBMClassifierMultiTarget`
   - `XGBoostRegressorMultiTarget`
   - `CatboostClassifierMultiTarget`
   など、複数のMulti-targetモデルが既に実装済み

2. **内部実装** (`FreqaiMultiOutputClassifier`):
   - 各ターゲットごとに独立したestimatorを訓練
   - 並列処理可能
   - sklearn互換のインターフェース

3. **Config構造**:
   - `freqai_buy` / `freqai_sell` セクションは**不要**
   - 単一の`freqai`セクションでOK

4. **ラベル命名規則**:
   - `&-buy`: Buy signal label
   - `&-sell`: Sell signal label
   - 予測カラムも同じ名前で自動生成

### 実装への影響

- 優先度: **High** (Phase 4のコア機能)
- 影響範囲:
  - FreqAIモデル: `LightGBMClassifierMultiTarget`を継承
  - TwoTierStrategy: Multi-targetラベル生成
  - Config: 単一の`freqai`セクション
  - Tests: Multi-target予測テスト
- **所要時間: 3.5時間** (Dual-instanceの6-9時間から大幅削減)

### PHASE4_VERIFICATION_REPORTとの差異

| 要素 | PHASE4_VERIFICATION_REPORT | 実装 |
|------|---------------------------|------|
| Config構造 | `freqai_buy` / `freqai_sell` | `freqai` (より簡潔) |
| Identifiers | `_buy` / `_sell` サフィックス | 単一identifier |
| FreqAIインスタンス | `self.freqai_buy` / `self.freqai_sell` | `self.freqai` |
| ラベル名 | `&-target` | `&-buy`, `&-sell` (明示的) |
| 予測カラム名 | `&-prediction_buy` / `&-prediction_sell` | `&-buy` / `&-sell` (一貫性) |

**結論**: 実装はより簡潔で、FreqTradeの標準パターンに準拠

### 決定事項

- [x] 実装方針を決定 ✅ **Option B - Single Model Multi-Target**
- [x] FreqTradeのFreqAI実装を調査 ✅
- [x] Multi-target supportを確認 ✅
- [x] 具体的な実装方法を決定 ✅
- [ ] 実装 (詳細は`PHASE4_IMPLEMENTATION_PLAN.md`参照)
- [ ] Phase 4テストを追加

### 次のステップ

実装の詳細は **`PHASE4_IMPLEMENTATION_PLAN.md`** を参照してください。

実装タスク (合計: 3.5時間):
1. FreqAIモデル修正 (15分)
2. TwoTierStrategy修正 (1.5時間)
3. Config更新 (15分)
4. テスト追加 (1時間)
5. バックテスト実行 (30分)

---

## 優先順位サマリー（更新済み）

| 項目 | 優先度 | 所要時間 | ブロッキング | 状態 |
|------|--------|----------|--------------|------|
| 3. Phase 4 Multi-target | 🔴 High | **3.5時間** ✅ | ML-on backtest | ✅ 調査完了 |
| 2. 最後N行の動作 | 🟡 Medium | 1-2時間 | 一部のテスト | ✅ 方針決定済み |
| 1. Return計算方法 | 🟢 Low | 30分 | 1つのテスト | ✅ 方針決定済み |

## 推奨実装順序（更新済み）

すべての判断が完了したため、実装の優先順位を更新:

1. **Item 1 (Return計算)** - 最初に実装 ✅
   - 理由: 所要時間が最短（30分）
   - 決定済み: `.shift(-n)` を使用
   - すぐに完了可能

2. **Item 2 (最後N行の動作)** - 次に実装 ✅
   - 理由: データリーク検出の正確性に関わる
   - 決定済み: NaN/0統一
   - 調査込みで1-2時間

3. **Item 3 (Phase 4 Multi-target)** - 最後に実装 ✅
   - 理由: 実装コスト3.5時間（当初見積の6-9時間から削減）
   - 決定済み: Single Model Multi-Target (Option B)
   - 詳細な実装計画あり (`PHASE4_IMPLEMENTATION_PLAN.md`)

**合計所要時間**: 5-6時間（当初見積の8.5-12時間から大幅削減）

---

## Next Steps ✅ 完了

1. ✅ このドキュメントをレビュー
2. ✅ 各項目の決定を行う
3. ✅ 決定事項を `REMAINING_TASKS_CHECKLIST.md` に反映
4. 🔄 実装を開始

**すべての判断が完了しました！** 🎉

実装の準備が整いました:
- Item 1: Return計算方法 → `.shift(-n)` 使用
- Item 2: 最後N行の動作 → NaN/0統一
- Item 3: Phase 4 Multi-target → Single Model Multi-Target

詳細な実装計画は以下を参照:
- `REMAINING_TASKS_CHECKLIST.md` - 実装タスク一覧
- `PHASE4_IMPLEMENTATION_PLAN.md` - Phase 4詳細実装計画
