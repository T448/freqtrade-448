# 実装ガイド

[⬅️ README に戻る](./README.md)

このドキュメントでは、Phase 1の実装範囲、実装ステップ、完了条件について説明します。

## 目次

- [Phase 1実装範囲](#phase-1実装範囲)
- [実装タイミング](#実装タイミング)
- [具体的な実装ステップ](#具体的な実装ステップphase-2以降)
- [Phase 1完了判定の具体的な基準](#phase-1完了判定の具体的な基準)

## Phase 1実装範囲

### 実装するもの

#### 必須実装

- ✅ **PrimaryStrategyBase抽象クラス**
  - `calculate_prices()`: 指値価格計算
  - `calculate_returns()`: リターン計算（学習用ラベル生成）
  - `execution_mode`パラメータによる約定シミュレーション方法の切り替え

- ✅ **ATRBreakoutStrategy実装**
  - ATR指値価格計算
  - 2つの約定シミュレーション方法（chase / one_candle）
  - configからのパラメータ取得（fee, exit_periods, execution_mode等）

- ✅ **TwoTierStrategy統合クラス（IStrategy継承）**
  - `populate_indicators()`: 価格計算とML統合
  - `populate_entry_trend()` / `populate_exit_trend()`: エントリー/エグジットシグナル生成
  - `custom_entry_price()` / `custom_exit_price()`: 指値価格設定
  - `set_freqai_targets()`: FreqAI訓練用ラベル生成

- ✅ **StrategyFactory基本機能**
  - 名前ベースの戦略選択
  - configからのインスタンス生成

- ✅ **config.json設定**
  - ML有効/無効の切り替え（secondary=null）
  - 約定シミュレーション方法の選択（execution_mode）
  - パラメータのconfig管理（fee, exit_periods等）

#### テスト要件

- ✅ **約定シミュレーションロジックの包括的テスト**
  - `tests/primary/test_atr_breakout.py`
  - chase / one_candle両方の動作検証
  - エッジケースの検証（データ不足、価格異常等）
  - 手数料計算の正確性検証
  - リターン計算の精度検証

### 実装しないもの（Phase 2へ延期）

- ❌ **他の1次戦略実装**
  - atr_mean_reversion
  - bollinger_breakout
  - simple_close（ラベル生成用）

- ❌ **他の2次モデル実装**
  - xgboost_classifier
  - catboost_classifier

- ❌ **高度な機能**
  - Optuna最適化機能
  - 動的な戦略登録機能
  - パラメータ空間定義（PARAM_SPACE）

- ❌ **複雑な設定管理**
  - プリセット機能
  - バリデーション機能

### Phase 1完了条件

1. ATR戦略のみで、ML有効/無効両方の動作確認
2. 約定シミュレーション方法（chase / one_candle）の切り替え動作確認
3. 包括的なテストによる約定ロジックの正確性保証
4. richmanbtc概念（1次リターン計算 → 2次ML予測）の忠実な実装確認

## 実装タイミング

### Phase 2以降で実装予定

1. **Phase 1の目標達成を優先**
   - 現在のリファクタリング目標: ML統合ロジックの分離と品質向上
   - ATR戦略の完全な実装と検証

2. **要件の明確化が必要**
   - どのような戦略バリエーションが必要か、実運用での経験が必要
   - 過度な抽象化を避け、実際のニーズに基づいた設計が望ましい

3. **段階的な改善の方針**
   - Phase 1: 基本機能の安定化（ATR戦略とML統合）
   - Phase 2: ハイパーパラメータ最適化とアーキテクチャ改善（他戦略追加）
   - Phase 3: 性能検証と最終統合

## 具体的な実装ステップ（Phase 2以降）

### ステップ1: ディレクトリ構造の作成

```bash
mkdir -p user_data/strategies/primary
```

### ステップ2: 抽象基底クラスの実装

1. `primary/base.py`: PrimaryStrategyBase抽象クラス
2. TwoTierStrategyの簡略化（SecondaryModelBase削除）

### ステップ3: 既存戦略の移行

1. 既存の価格ブレイクアウト戦略を`primary/atr_breakout.py`として独立化
2. 既存のTwoTierStrategyを新しいアーキテクチャに対応

### ステップ4: ファクトリーの改修

1. StrategyFactoryに戦略ロード機能を追加
2. 名前解決とクラスロード機能の実装
3. load_secondary()の削除（FreqAI直接使用）

### ステップ5: config.jsonの拡張

1. 新しい設定形式のサポート追加
2. バリデーション機能の追加

### ステップ6: 新戦略の追加（Phase 2）

1. 平均回帰戦略（primary/atr_mean_reversion.py）
2. ボリンジャーバンド戦略（primary/bollinger_breakout.py）
3. シンプルclose予測戦略（primary/simple_close.py）

### ステップ7: テストとドキュメント

1. 各戦略の単体テスト
2. 統合テスト（2層戦略全体）
3. 新戦略追加ガイドのドキュメント化
4. マイグレーションガイドの作成

## Phase 1完了判定の具体的な基準

### 機能面の検証

- [ ] **ATRBreakoutStrategy が正しく動作**
  - [ ] buy_price = close - (atr × multiplier) の計算が正確
  - [ ] sell_price = close + (atr × multiplier) の計算が正確
  - [ ] 価格が DataFrame に正しく追加される

- [ ] **execution_mode=chase で FEP 計算が正確**
  - [ ] Force Entry Price の計算ロジックが richmanbtc 仕様に準拠
  - [ ] 追いかけ型約定シミュレーションが正確
  - [ ] 手数料計算が正確（2 × fee）

- [ ] **execution_mode=one_candle で約定判定が正確**
  - [ ] 次足での約定判定ロジックが正確
  - [ ] 約定しない場合 return=0 が設定される
  - [ ] 手数料計算が正確（2 × fee）

- [ ] **ML 無効時（secondary=null）で動作**
  - [ ] freqai.enabled=false + secondary=null で起動
  - [ ] 1次戦略のみでエントリーシグナル生成
  - [ ] バックテストが正常完了

- [ ] **ML 有効時に buy/sell 独立予測を取得**
  - [ ] 2つの FreqAI モデル（buy/sell）が訓練される
  - [ ] &-prediction_buy と &-prediction_sell が生成される
  - [ ] 各予測値に基づいて enter_long/enter_short が設定される

### テスト面の検証

- [ ] **約定シミュレーション（chase/one_candle）のユニットテスト**
  - [ ] test_chase_execution_mode_returns()
  - [ ] test_one_candle_execution_mode_returns()
  - [ ] test_force_entry_price_calculation()
  - [ ] エッジケース: データ不足、価格異常、NaN/Inf

- [ ] **ラベル生成の正確性テスト**
  - [ ] test_label_generation_from_returns()
  - [ ] return > 0 → label=1 の検証
  - [ ] return <= 0 → label=0 の検証
  - [ ] buy/sell 独立ラベルの検証

- [ ] **データリーク検出テスト**
  - [ ] test_no_future_data_in_features()
  - [ ] test_no_shift_negative_in_indicators()
  - [ ] test_label_future_data_isolation()
  - [ ] test_return_calculation_uses_future_data_correctly()

- [ ] **Config バリデーションテスト**
  - [ ] test_invalid_config_secondary_without_freqai()
  - [ ] test_invalid_config_freqai_without_secondary()
  - [ ] test_invalid_execution_mode()

### 統合面の検証

- [ ] **Freqtrade backtesting で正常実行**
  - [ ] ML無効モード: `--strategy TwoTierStrategy` で実行
  - [ ] ML有効モード: FreqAI訓練→バックテスト完了
  - [ ] 取引履歴、メトリクスが正常出力

- [ ] **FreqAI 訓練が正常完了**
  - [ ] Buy モデル訓練成功
  - [ ] Sell モデル訓練成功
  - [ ] モデルファイルが user_data/models/ に保存

- [ ] **ML 予測結果が entry シグナルに反映**
  - [ ] &-prediction_buy=1 → enter_long=1
  - [ ] &-prediction_sell=1 → enter_short=1
  - [ ] 予測=0 の場合はエントリーなし

### コード品質の検証

- [ ] **データリーク自動検出スクリプトでチェック**
  - [ ] CI/CD で自動実行
  - [ ] .shift(-n) の誤用がない
  - [ ] 時系列分割が正しい

- [ ] **ドキュメントとの整合性**
  - [ ] 実装が design-challenges-strategy-factory.md に準拠
  - [ ] すべての必須機能が実装済み
  - [ ] Phase 2 への拡張ポイントが明確

### 最終確認

- [ ] **すべてのテストケースがパス**
- [ ] **リターンとラベル計算の正確性が保証**
- [ ] **データリークが検出されない**
- [ ] **Freqtrade + FreqAI の統合が完了**

## Phase 2機能プレビュー: Optuna最適化

Phase 2で実装予定の機能：

- **パラメータ空間定義**: 各戦略クラスに`PARAM_SPACE`属性を追加
- **Optuna統合**: 戦略パラメータの自動最適化
- **利点**: 新戦略追加時も最適化ロジックの変更不要

詳細はPhase 2で設計・実装予定

## 関連ドキュメント

- [アーキテクチャ設計](./architecture.md) - クラス設計とディレクトリ構造
- [テスト戦略](./testing.md) - テスト要件とデータリーク検出
- [設定管理](./configuration.md) - config.json設計
- [CHECKLIST.md](./CHECKLIST.md) - 実装進捗チェックリスト

[⬅️ README に戻る](./README.md)
