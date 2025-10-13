# 設計判断

[⬅️ README に戻る](./README.md)

このドキュメントでは、主要な設計判断とその背景、設計の利点について説明します。

## 目次

- [両建て決済メカニズムの実装](#両建て決済メカニズムの実装)
- [SecondaryModelBase削除方針](#secondarymodelbase削除方針)
- [設計の利点](#設計の利点)

## 両建て決済メカニズムの実装

### 調査結果: Freqtrade 損益可視化の仕様

#### ❌ ポジション積み上げ方式は採用不可

**結論**: Freqtradeの損益可視化は**決済済み取引のみ**を対象としており、建玉の含み損益は可視化されません。

**Freqtrade公式ドキュメント調査結果**:

1. **損益プロット**: `strategy_stats["daily_profit"]` は決済済み取引の累積利益のみを表示
2. **未実現損益**: バックテスト中の建玉は「LEFT OPEN TRADES REPORT」として別途表示
3. **エクイティカーブ**: 決済時のみ更新され、建玉の含み損益は反映されない

```python
# Freqtradeの損益プロット実装
df["equity_daily"] = df["equity"].cumsum()  # 決済済み利益の累積のみ
```

#### ポジション積み上げ方式の問題点

- ❌ **可視化不可**: 全ポジション決済まで損益グラフは平坦
- ❌ **目的未達**: 「残高の時間変化を見たい」という要求を満たせない
- ❌ **評価困難**: シグナル単位のパフォーマンス評価ができない
- ❌ **非現実的**: 多数の建玉を保持し続けるのは実運用と乖離

### ✅ 推奨実装: 明示的な決済シグナル方式

**実装方針**: `populate_exit_trend()` で明示的に `exit_long` / `exit_short` シグナルを生成

```python
def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    """ML予測に基づく明示的な決済シグナル生成

    Freqtradeは反対売買による自動決済をサポートしていないため、
    exit_long/exit_short で明示的に決済指示が必要

    両建て状態（long + short 同時保有）では、両方のexitシグナルが
    同時に発生する可能性があり、その場合Freqtradeは両ポジションを決済する
    """
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
```

#### この方式のメリット

- ✅ **損益可視化**: 決済ごとに損益グラフが更新される
- ✅ **目的達成**: 残高の時間変化を追跡可能
- ✅ **シグナル評価**: 各エントリー/エグジットの有効性を検証可能
- ✅ **現実的**: 実運用に近いシミュレーション

## SecondaryModelBase削除方針

### ✅ 結論: Phase 1 では削除すべき

**理由**: FreqAIが既に十分な抽象化を提供しており、中間層は不要です。

#### 現在の設計の問題点

```python
# 不要な抽象化層
TwoTierStrategy
├── self.primary_strategy (ATRBreakoutStrategy)  # ✅ 使用される
├── self.secondary_model (LightGBMClassifier)     # ❌ 未使用
└── self.freqai_buy / self.freqai_sell            # ✅ 実際に使用
```

- `SecondaryModelBase` は Phase 1 で実質的に使われていない
- `TwoTierStrategy` が直接 `self.freqai.start()` を呼び出している
- FreqAIの `BaseClassifierModel` が既に抽象化層を提供

#### 削除対象ファイル

```
user_data/strategies/
├── secondary/
│   ├── base.py                     # ❌ 削除
│   └── lightgbm_classifier.py      # ❌ 削除
```

#### 簡略化後のアーキテクチャ

```python
TwoTierStrategy(IStrategy)
├── self.primary_strategy (ATRBreakoutStrategy)   # 1次戦略
└── self.freqai_buy / self.freqai_sell            # FreqAI直接呼び出し
    └── TwoTierLightGBMClassifier(BaseClassifierModel)  # FreqAI層
```

#### FreqAIによるモデル切り替え

```json
// config.json で直接切り替え可能
{
  "freqai_buy": {
    "model_name": "TwoTierLightGBMClassifier",  // LightGBM
    // "model_name": "TwoTierXGBoostClassifier",  // XGBoost
    // "model_name": "TwoTierCatBoostClassifier", // CatBoost
  }
}
```

#### Phase 2 以降で抽象化が必要になる条件

以下の場合**のみ**、その時点で `SecondaryModelBase` を追加:

1. **非FreqAIモデル対応**: sklearn、独自ニューラルネットなど
2. **共通ロジックの出現**: 複数の2次モデル間で重複コードが発生
3. **FreqAI外の統合**: 他のMLフレームワークとの統合が必要

現時点では該当しないため、**YAGNI原則**に従い削除すべきです。

### 簡略化後の TwoTierStrategy 実装例

```python
class TwoTierStrategy(IStrategy):
    """Config駆動の2層取引戦略（Freqtradeエントリーポイント）"""

    def __init__(self, config: dict):
        super().__init__(config)
        two_tier_config = config.get('two_tier_strategy', {})

        # 1次戦略のみロード（2次モデルは FreqAI 経由）
        from user_data.strategies.utils.strategy_factory import StrategyFactory
        self.primary_strategy = StrategyFactory.load_primary(two_tier_config)
        self.is_ml_enabled = config.get('freqai', {}).get('enabled', False)

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """指標計算（価格計算＋ML統合）"""
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
```

## 設計の利点

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
   - FreqAI層: 2次モデルの具体的実装

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
   - 各戦略クラスがパラメータ空間を定義（Phase 2）
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
- **SecondaryModelBase削除** - FreqAI が提供する抽象化を活用し、不要な中間層を排除

Phase 2以降で本設計を実装することで、保守性・拡張性・柔軟性を大幅に向上させることができる。

## 関連ドキュメント

- [アーキテクチャ設計](./architecture.md) - TwoTierStrategyとPrimaryStrategyBaseの設計
- [FreqAI統合](./freqai-integration.md) - FreqAI直接統合の詳細
- [実装ガイド](./implementation.md) - Phase 1/2の実装範囲

[⬅️ README に戻る](./README.md)
