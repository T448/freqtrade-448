# Strategy Factory アーキテクチャ設計書

## メタ情報

- **作成日**: 2025-10-11
- **最終更新**: 2025-10-11
- **コミットハッシュ**: 3ddd09b9dbb4929e14299fa2eab04c44229bd7f1
- **関連ファイル**:
  - `user_data/strategies/two_tier_strategy.py`
  - `user_data/strategies/utils/strategy_factory.py`
  - `config.json`

## ドキュメントの位置づけ

**現状**: Phase 1リファクタリング実施中（既存実装は動作しているが、設計意図と乖離）

**背景**: コードレビュー対応中に、`TwoTierStrategy`クラスにおける具体的な戦略ロジックのハードコーディングに関する設計上の課題が指摘された。既存実装はutil層に戦略ロジックが混在しており、拡張性と保守性に課題がある。

**本ドキュメントの目的**:

- **Phase 1**: 既存実装を設計意図に沿ってリファクタリングするための設計指針
- **Phase 2拡張性**: 将来の機能拡張（複数戦略の追加、Optuna最適化等）を見据えた設計

**Phase 1での対応**:

- 基本的なファクトリーパターンの実装（ATRBreakoutStrategy、LightGBMClassifier）
- util層からの具体的な戦略ロジックの分離
- FreqtradeおよびFreqAIとの正しい統合
- Phase 2で容易に拡張できるアーキテクチャの確立

**Phase 2以降の拡張**:

- 複数の1次戦略実装（平均回帰、ボリンジャーバンド等）
- 複数の2次モデル実装（XGBoost、CatBoost等）
- Optuna最適化機能の追加

## 用語定義

- **1次戦略（Primary Strategy）**: ATRブレイクアウトなど、指値価格とリターンを計算する戦略ロジック
- **2次モデル（Secondary Model）**: 1次戦略のリターン予測を行うMLモデル（LightGBMなど）
- **コード内の表記**: クラス名は `PrimaryStrategyBase`, `SecondaryModelBase` を使用

## ドキュメント構成

このディレクトリには、Strategy Factoryアーキテクチャの設計文書が含まれています。

### 📚 設計ドキュメント

1. **[architecture.md](./architecture.md)** - アーキテクチャ設計
   - ディレクトリ構造
   - クラス設計（PrimaryStrategyBase, ATRBreakoutStrategy, TwoTierStrategy等）
   - StrategyFactory実装
   - 責務の分離とコード配置

2. **[freqai-integration.md](./freqai-integration.md)** - FreqAI統合
   - FreqAI統合アーキテクチャ
   - ラベル生成フロー
   - マルチターゲット実装（Buy/Sellモデル）
   - 特徴量計算の責任分担

3. **[configuration.md](./configuration.md)** - 設定管理
   - config.json設計
   - 設定パターン（ML有効/無効）
   - バリデーションルール
   - 設定の組み合わせパターン

4. **[testing.md](./testing.md)** - テスト戦略
   - テスト要件
   - データリーク検出チェックリスト
   - テストコード例
   - 自動検出スクリプト

5. **[implementation.md](./implementation.md)** - 実装ガイド
   - Phase 1実装範囲
   - 実装ステップ
   - Phase 1完了条件
   - Phase 2以降の拡張計画

6. **[decisions.md](./decisions.md)** - 設計判断
   - 両建て決済メカニズム
   - SecondaryModelBase削除方針
   - 設計の利点と背景

### 📋 進捗管理

7. **[CHECKLIST.md](./CHECKLIST.md)** - 実装進捗チェックリスト
   - アーキテクチャ実装状況
   - FreqAI統合状況
   - テスト実装状況
   - ドキュメント整備状況

## クイックナビゲーション

### よくある質問

**Q: 新しい1次戦略を追加するには？**
→ [architecture.md - PrimaryStrategyBase](./architecture.md#1次戦略primary-strategy) を参照

**Q: ML有効/無効を切り替えるには？**
→ [configuration.md - 設定パターン](./configuration.md#基本形式2層戦略ml有効) を参照

**Q: FreqAIとの統合方法は？**
→ [freqai-integration.md - 統合アーキテクチャ](./freqai-integration.md#freqai統合アーキテクチャ) を参照

**Q: データリークを防ぐには？**
→ [testing.md - データリーク検出](./testing.md#データリーク検出チェックリスト) を参照

**Q: Phase 1で何を実装すべき？**
→ [implementation.md - Phase 1実装範囲](./implementation.md#phase-1実装範囲) を参照

**Q: 実装の進捗状況は？**
→ [CHECKLIST.md](./CHECKLIST.md) を参照

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

## richmanbtc概念との整合性

**richmanbtcチュートリアルの核心概念**:

- **1次モデル**: ATR指値戦略のリターン計算（約定シミュレーションによる理論リターン）
- **2次モデル**: ML分類による成功予測（1次モデルのリターンがプラスになるかを予測）

**本設計での実現方法**:

- **学習フェーズ**: 1次モデルが計算したリターンをラベル化（リターン > 0 で成功/失敗）→ 2次モデルの訓練
- **推論フェーズ（ML有効）**: 1次モデルの指値価格 + 2次モデルの予測 → 予測=1の場合のみ注文
- **推論フェーズ（ML無効）**: 1次モデルの指値価格のみで注文（2次モデルの出力が常に1と同義）

参考: [richmanbtcチュートリアル](https://note.com/btcml/n/n9f730e59848c)

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

## 注意事項

> **重要**: 約定シミュレーションロジックは取引の根幹となるため、ミスがあると大きな損失に繋がる可能性があります。Phase 1実装時に包括的なテストコードを必ず用意してください。詳細は [testing.md](./testing.md) を参照。

## 変更履歴

- 2025-10-11: 初版作成（元ファイル: `design-challenges-strategy-factory.md` から分割）
