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

## 3. Phase 4 Multi-target Architecture (Phase 4) 🔴 **Critical**

### 背景

Phase 4の検証レポートで**最大の問題**として指摘されている項目:

- 現状: Single FreqAI model (基本実装、40%完成)
- 仕様: Dual FreqAI instances (Buy/Sell独立モデル)

### 現在の状況

**実装されているもの (Single Model)**:

```python
class TwoTierStrategy:
    def populate_any_indicators(self, metadata):
        # Single FreqAI model
        dataframe = self.freqai.start(dataframe, metadata, self)
        # predictions列を使用
```

**Phase 4仕様 (Dual Models)**:

```json
{
  "freqai_buy": {
    "identifier": "buy_model_v1",
    "label_period_candles": 24,
    // Buy model config
  },
  "freqai_sell": {
    "identifier": "sell_model_v1",
    "label_period_candles": 12,
    // Sell model config
  }
}
```

### 必要な実装

1. **Config構造の変更**
   - `freqai` → `freqai_buy` / `freqai_sell`
   - 各モデルで異なるidentifierとlabel_period

2. **TwoTierStrategyの変更**
   - Dual FreqAI instancesの初期化
   - Buy用とSell用のpredict呼び出し
   - 異なるlabel生成ロジック

3. **テストの追加**
   - Dual model configuration tests
   - Independent prediction tests
   - Identifier-based label generation tests

### 重要な調査項目 🔍

**Question 1: FreqTradeはdual FreqAI instancesをサポートしているか?**

- FreqAIの内部実装を調査
- 複数のFreqAIインスタンスを同時に使用可能か?
- 既存の実装例はあるか?

**Question 2: Configの設計方針**

- `freqai_buy`/`freqai_sell` として分離すべきか?
- それとも `freqai.targets: [buy, sell]` のような構造か?
- FreqTradeの標準的なパターンは?

**Question 3: 代替アーキテクチャ**
Single modelで Buy/Sell を multi-target として学習する方法:

```python
# Alternative: Single model with multi-target
labels = {
    "buy": buy_labels,
    "sell": sell_labels
}
predictions = model.predict()  # → {buy: prob, sell: prob}
```

### 選択肢

**Option A: Full Dual-Instance Implementation**

- メリット:
  - Phase 4仕様に完全準拠
  - Buy/Sellの完全な独立性
  - 異なるパラメータ・特徴量が使用可能
- デメリット:
  - 実装コスト大 (6-9時間)
  - FreqTradeのサポート状況が不明
  - テスト・デバッグコスト増

**Option B: Single Model Multi-Target**

- メリット:
  - 実装コスト小 (2-3時間)
  - FreqTradeの標準的なパターン
  - デバッグが容易
- デメリット:
  - Phase 4仕様から逸脱
  - Buy/Sellで同じ特徴量を使用
  - 柔軟性が低い

**Option C: Deferred Implementation**

- メリット:
  - 他のタスクを優先できる
  - バックテストでML-off modeを先に検証
- デメリット:
  - Phase 4が不完全なまま

### 推奨アプローチ

**Step 1: 調査 (1-2時間)**

1. FreqTradeのFreqAI実装を読む
2. Dual instance supportを確認
3. 既存の実装例を探す

**Step 2: 設計判断**

- FreqTradeがdual instanceをサポート → Option A
- サポートしない → Option B または Option C

**Step 3: 実装**

- 決定したoptionに基づいて実装

### 実装への影響

- 優先度: **High** (Phase 4のコア機能)
- 影響範囲:
  - config.json
  - TwoTierStrategy
  - FreqAI integration
  - Tests
- 所要時間:
  - Option A: 6-9時間
  - Option B: 2-3時間
  - Option C: 0時間 (defer)

### 決定事項

- [x] 実装方針を決定 (Option A/B/C)
  - **決定: Phase 4仕様を満たせる実装であれば何でもOK** ✅
  - 方針: まずFreqTradeのFreqAI実装を調査し、実装可能な方法を選択
- [ ] FreqTradeのFreqAI実装を調査
- [ ] Dual instance supportを確認
- [ ] 具体的な実装方法を決定 (調査結果に基づく)
- [ ] 決定に基づいて実装
- [ ] Phase 4テストを追加

---

## 優先順位サマリー

| 項目 | 優先度 | 所要時間 | ブロッキング |
|------|--------|----------|--------------|
| 3. Phase 4 Multi-target | 🔴 High | 6-9時間 | ML-on backtest |
| 2. 最後N行の動作 | 🟡 Medium | 1-2時間 | 一部のテスト |
| 1. Return計算方法 | 🟢 Low | 30分 | 1つのテスト |

## 推奨実装順序

1. **Item 2 (最後N行の動作)** を先に調査・修正
   - 理由: データリーク検出の正確性に関わる
   - テスト失敗の原因を明確化

2. **Item 3 (Phase 4 Multi-target)** の調査を開始
   - 理由: 実装コストが大きいため早期に方針決定
   - ML-on backtestのブロッカー

3. **Item 1 (Return計算)** は最後
   - 理由: 機能的な影響が小さい
   - テスト設計の問題の可能性が高い

---

## Next Steps

1. このドキュメントをレビュー
2. 各項目の決定を行う
3. 決定事項を `REMAINING_TASKS_CHECKLIST.md` に反映
4. 実装を開始
