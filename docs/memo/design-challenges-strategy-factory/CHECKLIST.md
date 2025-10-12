# 実装進捗チェックリスト

[⬅️ README に戻る](./README.md)

このチェックリストは、Strategy Factory アーキテクチャの実装進捗を追跡するためのものです。

## 📊 全体進捗

- **Phase**: Phase 1（基本実装）
- **最終更新**: 2025-10-11

## 🏗️ アーキテクチャ実装

### ディレクトリ構造

- [ ] `user_data/strategies/primary/` ディレクトリ作成
- [ ] `user_data/strategies/primary/__init__.py` 作成
- [ ] `user_data/freqaimodels/` 確認（FreqAIモデル配置用）

参照: [architecture.md - ディレクトリ構造](./architecture.md#ディレクトリ構造)

### PrimaryStrategyBase抽象クラス

- [ ] `user_data/strategies/primary/base.py` 作成
- [ ] `PrimaryStrategyBase.__init__()` 実装（params, execution_mode）
- [ ] `PrimaryStrategyBase.calculate_prices()` 抽象メソッド定義
- [ ] `PrimaryStrategyBase.calculate_returns()` 抽象メソッド定義
- [ ] ドキュメント文字列の記述

参照: [architecture.md - PrimaryStrategyBase](./architecture.md#primarystrategybase抽象クラス)

### ATRBreakoutStrategy実装

- [ ] `user_data/strategies/primary/atr_breakout.py` 作成
- [ ] `ATRBreakoutStrategy.__init__()` 実装（パラメータ読み込み）
  - [ ] period, multiplier（価格計算用）
  - [ ] fee, exit_periods, pips（ラベル生成用）
- [ ] `calculate_prices()` 実装（ATRベース指値価格）
- [ ] `calculate_returns()` 実装（execution_mode分岐）
- [ ] `_calculate_chase_returns()` 実装（FEP追いかけ型）
- [ ] `_calculate_one_candle_returns()` 実装（1足限定型）
- [ ] `_calculate_force_entry_price()` 実装（FEP計算）
- [ ] `_calculate_atr()` ヘルパーメソッド実装

参照: [architecture.md - ATRBreakoutStrategy](./architecture.md#atrbreakoutstrategy実装)

### TwoTierStrategy統合クラス

- [ ] `user_data/strategies/two_tier_strategy.py` リファクタリング
- [ ] `__init__()` 実装（config検証、primary_strategy読み込み）
  - [ ] freqai.enabled と secondary の整合性チェック
  - [ ] StrategyFactory.load_primary() 呼び出し
- [ ] `populate_indicators()` 実装
  - [ ] primary_strategy.calculate_prices() 呼び出し
  - [ ] FreqAI buy/sell モデル予測統合（ML有効時）
- [ ] `populate_entry_trend()` 実装（ML予測による判定）
- [ ] `populate_exit_trend()` 実装（明示的決済シグナル）
- [ ] `custom_entry_price()` 実装（指値価格取得）
- [ ] `custom_exit_price()` 実装（指値価格取得）
- [ ] `set_freqai_targets()` 実装（ラベル生成）

参照: [freqai-integration.md - TwoTierStrategy](./freqai-integration.md#twotierstrategy統合クラス)

### StrategyFactory

- [ ] `user_data/strategies/utils/strategy_factory.py` 作成
- [ ] `_primary_strategies` 辞書定義（戦略登録）
- [ ] `load_primary()` クラスメソッド実装
- [ ] `_load_class()` ヘルパーメソッド実装（動的ロード）
- [ ] エラーハンドリング（存在しない戦略名）

参照: [architecture.md - StrategyFactory](./architecture.md#strategyfactory)

## 🤖 FreqAI統合

### FreqAIモデル実装

- [ ] `user_data/freqaimodels/two_tier_lightgbm_classifier.py` 作成
- [ ] `TwoTierLightGBMClassifier(BaseClassifierModel)` クラス定義
- [ ] `populate_indicators()` 実装（特徴量生成）
  - [ ] テクニカル指標計算（RSI, MACD, 移動平均等）
  - [ ] %プレフィックス付きカラム追加
- [ ] `set_freqai_targets()` 実装（最小限 - TwoTierStrategyから呼ばれる）
- [ ] `fit()` / `predict()` デフォルト動作確認

参照: [freqai-integration.md - FreqAIモデル](./freqai-integration.md#freqaiモデルの実装場所)

### マルチターゲット設定（Buy/Sell独立モデル）

- [ ] config.json に `freqai_buy` セクション追加
- [ ] config.json に `freqai_sell` セクション追加
- [ ] 各セクションの `identifier` 設定（`_buy` / `_sell` サフィックス）
- [ ] `TwoTierStrategy.populate_indicators()` でBuy/Sell予測取得
- [ ] `&-prediction_buy` / `&-prediction_sell` カラム生成確認

参照: [freqai-integration.md - マルチターゲット](./freqai-integration.md#freqaiマルチターゲット実装)

### ラベル生成フロー

- [ ] `TwoTierStrategy.set_freqai_targets()` 実装
  - [ ] `primary_strategy.calculate_returns()` 呼び出し
  - [ ] リターン > 0 で成功ラベル（1）
  - [ ] `&-target_buy` / `&-target_sell` カラム生成
  - [ ] freqai.identifier による buy/sell 判定
- [ ] FreqAI訓練時のラベル使用確認

参照: [freqai-integration.md - ラベル生成フロー](./freqai-integration.md#ラベル生成フロー)

## ⚙️ 設定管理

### config.json設計

- [ ] `two_tier_strategy` セクション追加
  - [ ] `primary` 設定（戦略名）
  - [ ] `secondary` 設定（null許可）
  - [ ] `primary_params` 設定（戦略パラメータ）
- [ ] `freqai` セクション設定
  - [ ] `enabled` 設定
  - [ ] `identifier` 設定
  - [ ] `model_name` 設定
- [ ] ML有効モード設定例作成
- [ ] ML無効モード設定例作成

参照: [configuration.md - config設計](./configuration.md#configjson設計)

### バリデーション

- [ ] TwoTierStrategy.**init**() でconfig検証
  - [ ] freqai.enabled=true かつ secondary=null → エラー
  - [ ] freqai.enabled=false かつ secondary指定 → エラー
- [ ] エラーメッセージの明確化

参照: [configuration.md - バリデーションルール](./configuration.md#バリデーションルール)

## 🧪 テスト実装

### 約定シミュレーションテスト

- [ ] `tests/primary/test_atr_breakout.py` 作成
- [ ] `test_calculate_prices()` - 指値価格計算テスト
- [ ] `test_chase_execution_mode_returns()` - chase方式リターン計算テスト
  - [ ] FEP計算の正確性
  - [ ] 手数料計算の正確性
- [ ] `test_one_candle_execution_mode_returns()` - one_candle方式テスト
  - [ ] 約定判定の正確性
  - [ ] 約定しない場合のリターン=0確認
- [ ] `test_force_entry_price_calculation()` - FEP計算テスト
- [ ] エッジケーステスト
  - [ ] データ不足時の挙動
  - [ ] 価格異常時の挙動
  - [ ] NaN/Inf処理

参照: [testing.md - 約定シミュレーションテスト](./testing.md#1-約定シミュレーションロジック最重要)

### ラベル生成テスト

- [ ] `tests/utils/test_two_tier_strategy.py` 作成
- [ ] `test_label_generation_from_returns()` - ラベル生成テスト
  - [ ] リターン > 0 → ラベル=1 確認
  - [ ] リターン <= 0 → ラベル=0 確認
  - [ ] buy/sell独立ラベル確認
- [ ] `test_execution_mode_label_difference()` - execution_mode切り替え確認

参照: [testing.md - ラベル生成テスト](./testing.md#2-ラベル生成ロジック)

### データリーク検出テスト

- [ ] `tests/data_leak/test_feature_isolation.py` 作成
- [ ] `test_no_future_data_in_features()` - 特徴量未来データ依存性テスト
- [ ] `test_no_shift_negative_in_indicators()` - .shift(-n)検出テスト
- [ ] `test_label_future_data_isolation()` - ラベル隔離テスト
- [ ] `test_return_calculation_uses_future_data_correctly()` - リターン計算テスト
- [ ] `test_time_series_split()` - 時系列分割テスト

参照: [testing.md - データリーク検出](./testing.md#データリーク検出テストコード)

### Configバリデーションテスト

- [ ] `tests/utils/test_config_validation.py` 作成
- [ ] `test_invalid_config_secondary_without_freqai()` テスト
- [ ] `test_invalid_config_freqai_without_secondary()` テスト
- [ ] `test_valid_config_ml_enabled()` テスト
- [ ] `test_valid_config_ml_disabled()` テスト

参照: [configuration.md - バリデーション](./configuration.md#バリデーションルール)

### 自動検出スクリプト

- [ ] `scripts/detect_data_leak.py` 作成
- [ ] AST解析による .shift(-n) 検出実装
- [ ] CI/CD統合（GitHub Actions等）
- [ ] 検出レポート生成

参照: [testing.md - 自動検出スクリプト](./testing.md#自動検出スクリプトcicd統合用)

## 🔧 統合テスト

### Freqtrade バックテスト実行

- [ ] ML無効モードでバックテスト実行
  - [ ] `freqtrade backtesting --strategy TwoTierStrategy --config config_ml_off.json`
  - [ ] エラーなく完了
  - [ ] 取引履歴出力確認
- [ ] ML有効モードでバックテスト実行
  - [ ] FreqAI訓練実行
  - [ ] Buy/Sellモデル訓練成功確認
  - [ ] `freqtrade backtesting --strategy TwoTierStrategy --config config_ml_on.json`
  - [ ] エラーなく完了
  - [ ] 取引履歴出力確認

参照: [implementation.md - 統合面の検証](./implementation.md#統合面の検証)

### FreqAI訓練確認

- [ ] Buy モデル訓練成功
- [ ] Sell モデル訓練成功
- [ ] モデルファイル保存確認（`user_data/models/`）
- [ ] 訓練ログ確認（エラーなし）
- [ ] 予測結果カラム確認（`&-prediction_buy`, `&-prediction_sell`）

参照: [implementation.md - FreqAI訓練](./implementation.md#freqai-訓練が正常完了)

### エントリー/エグジットシグナル確認

- [ ] ML無効時: 常にenter_long=1, enter_short=1
- [ ] ML有効時: 予測=1の場合のみエントリー
  - [ ] `&-prediction_buy=1` → `enter_long=1`
  - [ ] `&-prediction_sell=1` → `enter_short=1`
  - [ ] 予測=0の場合エントリーなし
- [ ] エグジットシグナル確認
  - [ ] `&-prediction_sell=1` → `exit_long=1`
  - [ ] `&-prediction_buy=1` → `exit_short=1`

参照: [implementation.md - ML予測結果の反映](./implementation.md#ml-予測結果が-entry-シグナルに反映)

## 📚 ドキュメント整備

### 設計ドキュメント

- [x] README.md - 概要とナビゲーション
- [x] architecture.md - アーキテクチャ設計
- [x] freqai-integration.md - FreqAI統合
- [x] configuration.md - 設定管理
- [x] testing.md - テスト戦略
- [x] implementation.md - 実装ガイド
- [x] decisions.md - 設計判断
- [x] CHECKLIST.md - 進捗チェックリスト（本ファイル）

### コードドキュメント

- [ ] PrimaryStrategyBase のdocstring完備
- [ ] ATRBreakoutStrategy のdocstring完備
- [ ] TwoTierStrategy のdocstring完備
- [ ] StrategyFactory のdocstring完備
- [ ] 重要メソッドのWarning記述（calculate_returns等）

### 使用例・チュートリアル

- [ ] 新しい1次戦略の追加方法（Phase 2準備）
- [ ] config.json設定例の充実
- [ ] トラブルシューティングガイド

## 🎯 Phase 1完了条件

### 必須項目

- [ ] すべてのテストケースがパス
- [ ] リターンとラベル計算の正確性が保証
- [ ] データリークが検出されない
- [ ] Freqtrade + FreqAI の統合が完了
- [ ] ML有効/無効両モードで動作確認
- [ ] ドキュメントが完備

参照: [implementation.md - Phase 1完了条件](./implementation.md#phase-1完了条件)

### 推奨項目

- [ ] CI/CDパイプライン構築（データリーク自動検出）
- [ ] パフォーマンステスト（バックテスト実行時間）
- [ ] コードレビュー完了

## 📝 メモ・課題

### 既知の課題

（実装中に発見した課題をここに記録）

### 次のステップ（Phase 2準備）

- [ ] 他の1次戦略の要件定義（平均回帰、ボリンジャー等）
- [ ] Optuna統合の設計
- [ ] パラメータ空間定義（PARAM_SPACE）の設計

---

**更新履歴**:

- 2025-10-11: 初版作成

[⬅️ README に戻る](./README.md)
