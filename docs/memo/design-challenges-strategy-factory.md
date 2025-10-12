# 設計上の課題: Strategy Factory アーキテクチャ（Phase 1リファクタリング設計）

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

## 概要

本ドキュメントでは、Phase 2への拡張性を考慮しつつ、Phase 1で実装すべきより拡張性と柔軟性の高いアーキテクチャを提案する。

## 現状の実装

### 問題箇所

**ファイル**: `user_data/strategies/utils/strategy_factory.py:417-453`

```python
def _generate_basic_price_signals(self, dataframe, metadata: Dict[str, Any]):
    """価格ブレイクアウト信号生成

    ML予測なしの価格ブレイクアウト戦略
    前期間の指値価格に現在価格が到達した場合に信号を生成
    """
    # ... 価格ブレイクアウト戦略の具体的な実装 ...
```

### 問題点

1. **util層に具体的な戦略ロジックが存在**
   - `_generate_basic_price_signals`は「価格ブレイクアウト」という具体的なトレーディングロジックを実装
   - util層に特定の戦略がハードコードされている

2. **拡張性の課題**
   - ML無効時の戦略を変更する場合、`TwoTierStrategy`クラスを直接修正する必要がある
   - 新しい戦略を追加する際の柔軟性が低い

3. **設定の柔軟性不足**
   - 1次戦略と2次モデルを独立して選択できない
   - 戦略の組み合わせを変更する際にコード修正が必要

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

### richmanbtc概念との整合性

**richmanbtcチュートリアルの核心概念**:

- **1次モデル**: ATR指値戦略のリターン計算（約定シミュレーションによる理論リターン）
- **2次モデル**: ML分類による成功予測（1次モデルのリターンがプラスになるかを予測）

**本設計での実現方法**:

- **学習フェーズ**: 1次モデルが計算したリターンをラベル化（リターン > 0 で成功/失敗）→ 2次モデルの訓練
- **推論フェーズ（ML有効）**: 1次モデルの指値価格 + 2次モデルの予測 → 予測=1の場合のみ注文
- **推論フェーズ（ML無効）**: 1次モデルの指値価格のみで注文（2次モデルの出力が常に1と同義）

### 動作モードの整理

#### 実トレードの特徴: 両建て方式

- **買いと売りを独立して判定**: MLが有利と判断した方向のみエントリー
  - buy_model=1 → 買い指値発注
  - sell_model=1 → 売り指値発注
  - 両方=1 → 両建て（同時ポジション保有）
- **反対売買による決済**: 時間ベースではなく、反対注文の約定で決済
  - ロング保有中 → 売り注文約定で決済
  - ショート保有中 → 買い注文約定で決済
- **max_open_trades**: 最大同時ポジション数で制限

#### モード1: 2次モデルOFF（ML無効）

- **バックテスト/実取引**: 1次モデルの指値価格で両方向常に注文
- **学習**: なし（MLを使わないので学習不要）

#### モード2: 2次モデルON（ML有効）

- **学習フェーズ**:
  - 1次モデルが計算した指値価格で注文執行した場合のリターンをラベル化
  - buy/sell独立したラベル生成（両建て対応）
  - テクニカル指標等の特徴量 → リターン成功/失敗の予測モデルを訓練（2つのモデル）
- **バックテスト/実取引フェーズ**:
  - 1次モデル: 買い/売り指値価格計算
  - 2次モデル: buy/sellそれぞれML予測（0 or 1）
  - 最終判断: 各方向で予測=1の場合のみ注文

### 学習ラベルの定義

**2次モデル（ML）の学習ラベル**:

1次モデルが計算した指値価格で注文を執行した場合のリターンに基づく2値分類:

- **ラベル = 1**: 指値注文が執行され、利益がプラスになる取引
- **ラベル = 0**: 指値注文が執行されても損失になる、または執行されない取引

**約定シミュレーション方法（configで選択可能）**:

richmanbtcチュートリアルでは、以下の2通りの約定シミュレーション方法が提示されている:

#### アプローチ1: エントリー追いかけ型（約定シミュレーション例1）

**エントリールール**: 買うと決めたら約定するまで指値で追いかける
**エグジットルール**: t足後に売り始める。売ると決めたら約定するまで追いかける

```python
df['y_buy'] = df['sell_fep'].shift(-t) / df['buy_fep'] - 1 - 2 * fee
df['y_sell'] = -(df['buy_fep'].shift(-t) / df['sell_fep'] - 1) - 2 * fee
```

**特徴**:

- エントリーもエグジットも約定するまで追いかける
- Force Entry Price（FEP）を使用して実際の約定価格を計算
- 確実に取引が成立する前提
- より積極的な取引戦略

#### アプローチ2: エントリー1足限定型（約定シミュレーション例2、推奨）

**エントリールール**: 指値を出して次の足でキャンセル
**エグジットルール**: エントリーが約定した場合のみ、t足後に売り始める。売ると決めたら約定するまで追いかける

```python
df['y_buy'] = np.where(
    (df['buy_price'] / pips).round() > (df['lo'].shift(-1) / pips).round(),
    df['sell_fep'].shift(-t) / df['buy_price'] - 1 - 2 * fee,
    0
)
df['y_sell'] = np.where(
    (df['sell_price'] / pips).round() < (df['hi'].shift(-1) / pips).round(),
    -(df['buy_fep'].shift(-t) / df['sell_price'] - 1) - 2 * fee,
    0
)
```

**特徴**:

- エントリーは1足限定（約定しなければキャンセル）
- 約定した取引のみをラベル化（約定しない場合は0）
- より現実的な取引戦略
- リスク管理が容易

**Force Entry Price（FEP）の概念**:

買うと決めてから約定するまで指値で追いかけた場合に、実際に約定する価格。各足で指値を更新しながら、価格が指値に到達するまで待つ。

**戦略によるバリエーション**:

- 約定シミュレーション方法: 追いかけ型 / 1足限定型
- 利益計算期間（t）: 24足後、48足後など
- 手数料（fee）: maker手数料（例: -0.00025）、taker手数料など
- 指値の出し方: ATRベース、最良気配近似（close ± pips）など

**実装への反映**:

- 1次戦略は指値価格の計算とラベル生成ロジックを提供
- 2次モデルは学習済みMLで「この指値注文は利益になるか」を予測
- 予測結果に基づいて注文の執行を制御

参考: [richmanbtcチュートリアル](https://note.com/btcml/n/n9f730e59848c)

## 提案する設計

### ディレクトリ構造

```
user_data/strategies/
├── primary/                    # 1次戦略（ドメインロジック）
│   ├── __init__.py
│   ├── base.py                # PrimaryStrategyBase抽象クラス
│   ├── atr_breakout.py        # ATRBreakoutStrategy
│   ├── atr_mean_reversion.py  # ATRMeanReversionStrategy (Phase 2)
│   ├── bollinger_breakout.py  # BollingerBreakoutStrategy (Phase 2)
│   └── simple_close.py        # SimpleCloseStrategy (Phase 2)
├── secondary/                  # 2次モデル（ドメインロジック）
│   ├── __init__.py
│   ├── base.py                # SecondaryModelBase抽象クラス
│   ├── lightgbm_classifier.py # LightGBMClassifier
│   ├── xgboost_classifier.py  # XGBoostClassifier (Phase 2)
│   └── catboost_classifier.py # CatBoostClassifier (Phase 2)
├── utils/                      # ヘルパークラス
│   ├── strategy_factory.py    # StrategyFactory（戦略ロード用）
│   ├── price_calculator.py    # 既存の価格計算器（後方互換性用）
│   └── freqai_model_factory.py
└── two_tier_strategy.py       # TwoTierStrategy(IStrategy) - Freqtradeエントリーポイント
```

**ポイント**:

- `primary/`: 1次戦略のドメインロジック（IStrategyは**継承しない**）
- `secondary/`: 2次モデルのドメインロジック（IStrategyは**継承しない**）
- `utils/`: ファクトリーパターン実装（ヘルパークラス）
- `two_tier_strategy.py`: **FreqtradeのIStrategyを継承**し、config駆動で1次戦略・2次モデルを切り替え

**実行例**:
```bash
freqtrade backtesting --strategy TwoTierStrategy --config config.json
```

### クラス設計

#### 1次戦略（Primary Strategy）

```python
# primary/base.py
from abc import ABC, abstractmethod
import pandas as pd

class PrimaryStrategyBase(ABC):
    """1次戦略の抽象基底クラス

    指値価格計算とリターン計算を担当
    """

    def __init__(self, params: dict):
        self.params = params
        self.execution_mode = params.get("execution_mode", "one_candle")  # "chase" or "one_candle"

    @abstractmethod
    def calculate_prices(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """指値価格計算（buy_price, sell_priceカラムを追加）

        Args:
            dataframe: OHLCデータ

        Returns:
            buy_price, sell_priceが追加されたDataFrame

        Note:
            - ML有効/無効に関わらず常に実行される
            - 計算された指値価格は注文執行とML学習の両方で使用される
        """
        pass

    @abstractmethod
    def calculate_returns(self, dataframe: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        """指値戦略のリターン計算（ML学習用ラベル生成）

        Args:
            dataframe: 価格計算済みのDataFrame

        Returns:
            (buy_return, sell_return): 買い/売りそれぞれの理論リターン

        Note:
            - execution_mode設定に基づいて約定シミュレーション方法を切り替え
            - 買いと売りで独立したリターンを計算（両建て対応）
            - 計算されたリターンは、ML学習時にラベル化される（リターン > 0 で成功）
        """
        pass
```

```python
# primary/atr_breakout.py
class ATRBreakoutStrategy(PrimaryStrategyBase):
    """ATR指値戦略（richmanbtcチュートリアルの実装）"""

    def __init__(self, params: dict):
        super().__init__(params)
        self.period = params.get("period", 14)
        self.multiplier = params.get("multiplier", 0.5)
        # 以下のパラメータはラベル生成専用（configから取得）
        self.fee = params.get("fee", 0.00025)  # シミュレーション用手数料
        self.exit_periods = params.get("exit_periods", 24)  # N期間後のリターン計算
        self.pips = params.get("pips", 0.5)  # 価格丸め精度（オプション）

    def calculate_prices(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """ATRベースの指値価格計算"""
        # ATR計算
        atr = self._calculate_atr(dataframe, self.period)
        offset = atr * self.multiplier

        # 指値価格設定
        dataframe["buy_price"] = dataframe["close"] - offset
        dataframe["sell_price"] = dataframe["close"] + offset

        return dataframe

    def calculate_returns(self, dataframe: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        """指値戦略のリターン計算（ML学習用ラベル生成）

        richmanbtcの2つの約定シミュレーション方法を切り替え可能
        買いと売りで独立したリターンを計算（両建て対応）

        Warning:
            このロジックは取引の根幹となるため、ミスがあると大きな損失に繋がる
            必ず包括的なテストコードで検証すること（tests/primary/test_atr_breakout.py）

        Returns:
            (buy_return, sell_return): 買い/売りそれぞれの理論リターン
        """
        if self.execution_mode == "chase":
            return self._calculate_chase_returns(dataframe)
        else:  # one_candle
            return self._calculate_one_candle_returns(dataframe)

    def _calculate_chase_returns(self, dataframe: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        """アプローチ1: エントリー追いかけ型（richmanbtc例1）

        エントリー: 約定するまで指値で追いかける（Force Entry Price使用）
        エグジット: exit_periods足後に売り始め、約定するまで追いかける

        実装例:
            df['y_buy'] = df['sell_fep'].shift(-t) / df['buy_fep'] - 1 - 2 * fee
            df['y_sell'] = -(df['buy_fep'].shift(-t) / df['sell_fep'] - 1) - 2 * fee
        """
        # Force Entry Price (FEP) 計算
        buy_fep = self._calculate_force_entry_price(
            dataframe['buy_price'], dataframe['low'], self.pips
        )
        sell_fep = self._calculate_force_entry_price(
            dataframe['sell_price'], dataframe['high'], self.pips, direction='sell'
        )

        # exit_periods足後のFEPでリターン計算
        future_sell_fep = sell_fep.shift(-self.exit_periods)
        future_buy_fep = buy_fep.shift(-self.exit_periods)

        # 買いリターン: buy_fep で買い → exit_periods後に sell_fep で売り
        buy_return = (future_sell_fep / buy_fep) - 1 - 2 * self.fee

        # 売りリターン: sell_fep で売り → exit_periods後に buy_fep で買い戻し
        sell_return = -(future_buy_fep / sell_fep - 1) - 2 * self.fee

        return buy_return, sell_return

    def _calculate_one_candle_returns(self, dataframe: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        """アプローチ2: エントリー1足限定型（richmanbtc例2、推奨）

        エントリー: 指値を出して次の足でキャンセル（約定判定あり）
        エグジット: 約定した場合のみ、exit_periods足後に売り始める

        実装例:
            df['y_buy'] = np.where(
                (df['buy_price'] / pips).round() > (df['lo'].shift(-1) / pips).round(),
                df['sell_fep'].shift(-t) / df['buy_price'] - 1 - 2 * fee,
                0
            )
            df['y_sell'] = np.where(
                (df['sell_price'] / pips).round() < (df['hi'].shift(-1) / pips).round(),
                -(df['buy_fep'].shift(-t) / df['sell_price'] - 1) - 2 * fee,
                0
            )
        """
        # 次足での約定判定
        buy_filled = (dataframe['buy_price'] / self.pips).round() > \
                     (dataframe['low'].shift(-1) / self.pips).round()
        sell_filled = (dataframe['sell_price'] / self.pips).round() < \
                      (dataframe['high'].shift(-1) / self.pips).round()

        # エグジット用のFEP計算
        buy_fep = self._calculate_force_entry_price(
            dataframe['buy_price'], dataframe['low'], self.pips
        )
        sell_fep = self._calculate_force_entry_price(
            dataframe['sell_price'], dataframe['high'], self.pips, direction='sell'
        )

        # 約定した場合のみリターン計算、約定しない場合は0
        future_sell_fep = sell_fep.shift(-self.exit_periods)
        future_buy_fep = buy_fep.shift(-self.exit_periods)

        # 買いリターン
        buy_return = np.where(
            buy_filled,
            (future_sell_fep / dataframe['buy_price']) - 1 - 2 * self.fee,
            0
        )

        # 売りリターン
        sell_return = np.where(
            sell_filled,
            -(future_buy_fep / dataframe['sell_price'] - 1) - 2 * self.fee,
            0
        )

        return pd.Series(buy_return, index=dataframe.index), pd.Series(sell_return, index=dataframe.index)

    def _calculate_force_entry_price(self, entry_price, extreme_price, pips, direction='buy'):
        """Force Entry Price (FEP) 計算

        買うと決めてから約定するまで指値で追いかけた場合の実際の約定価格

        Args:
            entry_price: 指値価格
            extreme_price: low (買い) or high (売り)
            pips: 価格丸め精度
            direction: 'buy' or 'sell'

        Returns:
            各時点でのFEP
        """
        # richmanbtcのnumba実装を参考に実装
        # 詳細は実装時に記述（パフォーマンスのためnumba使用を検討）
        pass
```

```python
# primary/simple_close.py
class SimpleCloseStrategy(PrimaryStrategyBase):
    """次足close予測用シンプル戦略（ラベル生成用）

    ML学習時のラベル生成に使用
    次の足のclose上げ下げを予測するためのシンプルなロジック
    """

    def __init__(self, params: dict):
        super().__init__(params)
        self.threshold = params.get("threshold", 0.001)

    def calculate_prices(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """現在価格をそのまま使用"""
        dataframe["buy_price"] = dataframe["close"]
        dataframe["sell_price"] = dataframe["close"]
        return dataframe

    def calculate_returns(self, dataframe: pd.DataFrame) -> pd.Series:
        """次足のclose変化率を返す（シンプルな学習用）"""
        # 次足のclose変化率
        future_return = dataframe["close"].pct_change().shift(-1)
        return future_return
```

#### 2次モデル（Secondary Model）

```python
# secondary/base.py
from abc import ABC, abstractmethod
import pandas as pd

class SecondaryModelBase(ABC):
    """2次モデル（ML）の抽象基底クラス"""

    def __init__(self, params: dict):
        self.params = params

    @abstractmethod
    def integrate_with_freqai(self, dataframe: pd.DataFrame, metadata: dict, strategy) -> pd.DataFrame:
        """FreqAI予測データの統合

        Args:
            dataframe: DataFrame
            metadata: ペア情報
            strategy: Freqtradeストラテジーインスタンス（freqai.start()実行用）

        Returns:
            &-predictionカラムが追加されたDataFrame
        """
        pass

    @abstractmethod
    def filter_signals(self, dataframe: pd.DataFrame, signals: pd.DataFrame, config: dict) -> pd.DataFrame:
        """ML予測に基づいて信号をフィルタリング

        Args:
            dataframe: 予測データを含むDataFrame
            signals: 1次戦略からの信号
            config: フィルタリング設定

        Returns:
            フィルタリングされた信号
        """
        pass
```

```python
# secondary/lightgbm_classifier.py
class LightGBMClassifier(SecondaryModelBase):
    """LightGBM二値分類モデル（SecondaryModelBaseラッパー）

    FreqAIモデル（TwoTierLightGBMClassifier）のラッパーとして機能し、
    TwoTierStrategyからの呼び出しをFreqAIフレームワークに橋渡しする
    """

    def __init__(self, params: dict):
        super().__init__(params)
        self.confidence_threshold = params.get("confidence_threshold", 0.6)

    def integrate_with_freqai(self, dataframe: pd.DataFrame, metadata: dict, strategy) -> pd.DataFrame:
        """FreqAI統合でLightGBM予測を取得

        Note:
            - strategy.freqai.start()はFreqAIフレームワークのエントリーポイント
            - 内部でTwoTierLightGBMClassifier(BaseClassifierModel)が呼び出される
            - &-predictionカラムに予測結果（0 or 1）が追加される
        """
        # FreqAIのstart()を呼び出して予測データを取得
        dataframe = strategy.freqai.start(dataframe, metadata, strategy)
        return dataframe

    def filter_signals(self, dataframe: pd.DataFrame, signals: pd.DataFrame, config: dict) -> pd.DataFrame:
        """ML予測（0/1）で信号をフィルタリング"""
        if "&-prediction" not in dataframe.columns:
            logger.warning("ML prediction data not available")
            return signals

        # 予測が1の場合のみ信号を通す
        ml_filter = dataframe["&-prediction"] == 1

        # 信頼度フィルタリング
        if "&-probability" in dataframe.columns:
            confidence_filter = dataframe["&-probability"] >= self.confidence_threshold
            ml_filter = ml_filter & confidence_filter

        signals["enter_long"] = signals["enter_long"] & ml_filter
        signals["enter_short"] = signals["enter_short"] & ml_filter

        return signals
```

### FreqAI統合アーキテクチャ

#### 全体構成

```
TwoTierStrategy(IStrategy) ← Freqtradeエントリーポイント
├── PrimaryStrategyBase (例: ATRBreakoutStrategy)
│   ├── calculate_prices() → buy_price, sell_price
│   └── calculate_returns() → ラベル生成用リターン計算
│
└── SecondaryModelBase (例: LightGBMClassifier) ← ラッパー層
    └── FreqAIフレームワーク経由で呼び出し
        └── TwoTierLightGBMClassifier(BaseClassifierModel) ← 実際のMLモデル
            ├── populate_indicators() → 特徴量生成
            ├── set_freqai_targets() → ラベル生成（TwoTierStrategyから呼ばれる）
            ├── fit() → モデル訓練
            └── predict() → 予測実行
```

#### SecondaryModelBaseの役割

`SecondaryModelBase`（例: `LightGBMClassifier`）は、FreqAIモデルの**ラッパー**として機能します：

1. **FreqAI統合の橋渡し**
   - `TwoTierStrategy`からの呼び出しをFreqAIフレームワークに転送
   - `strategy.freqai.start()`を呼び出して予測データを取得

2. **予測結果のフィルタリング**
   - FreqAIから返された`&-prediction`カラムを使用
   - エントリーシグナルのフィルタリングロジックを提供

#### FreqAIモデルの実装場所

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

#### ラベル生成フロー

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

**実装例（TwoTierStrategy内）**:

```python
def set_freqai_targets(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    """FreqAI訓練用ラベル生成

    1次戦略のリターン計算結果をラベル化し、FreqAIに渡す
    どの1次戦略と組み合わせても動作する汎用的な実装
    """
    # 1次戦略でリターン計算（約定シミュレーション）
    # ATRBreakout、MeanReversion等、任意の1次戦略が使用可能
    returns = self.primary_strategy.calculate_returns(dataframe)

    # リターン > 0 で成功ラベル（1）、それ以外は失敗ラベル（0）
    dataframe['&-target'] = (returns > 0).astype(int)

    return dataframe
```

#### Config設定での組み合わせ例

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

#### TwoTierStrategy（統合クラス）

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
            logger.warning(
                "freqai.enabled=true but no secondary model specified. "
                "FreqAI will be enabled but predictions will not be used for filtering. "
                "Consider setting secondary to a model name (e.g., 'lightgbm_classifier') or disabling FreqAI."
            )

        # StrategyFactoryで1次戦略・2次モデルをロード
        from user_data.strategies.utils.strategy_factory import StrategyFactory
        self.primary_strategy = StrategyFactory.load_primary(two_tier_config)
        self.secondary_model = StrategyFactory.load_secondary(two_tier_config)
        self.is_ml_enabled = self.secondary_model is not None

        logger.info(
            f"TwoTierStrategy initialized: "
            f"primary={type(self.primary_strategy).__name__}, "
            f"secondary={type(self.secondary_model).__name__ if self.secondary_model else 'None'}, "
            f"freqai_enabled={freqai_enabled}"
        )

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """指標計算（価格計算＋ML統合）

        1次戦略で指値価格を計算し、ML有効時はFreqAI予測を統合
        """
        # 1次戦略: 指値価格計算
        dataframe = self.primary_strategy.calculate_prices(dataframe)

        # 2次モデル: FreqAI統合（ML有効時のみ）
        if self.is_ml_enabled:
            dataframe = self.freqai.start(dataframe, metadata, self)

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

        両建て方式: 反対売買による自動決済
        - ロングポジション保有中 → 売り注文約定で決済
        - ショートポジション保有中 → 買い注文約定で決済
        - 時間ベースの決済は行わない（ラベル生成とは異なる）

        Freqtradeのポジション管理:
        - ROI/stoploss: オプションで使用可能
        - max_open_trades: 最大同時ポジション数を制限
        """
        # 両建て方式では明示的なエグジットシグナルは不要
        # 反対注文の約定によってポジションが自動的にクローズされる
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

#### ファクトリー

```python
# utils/strategy_factory.py
class StrategyFactory:
    """戦略ファクトリー

    config.jsonの名前から具体的な戦略クラスをロード
    """

    # 1次戦略の登録
    _primary_strategies = {
        "atr_breakout": "user_data.strategies.primary.atr_breakout.ATRBreakoutStrategy",
        "atr_mean_reversion": "user_data.strategies.primary.atr_mean_reversion.ATRMeanReversionStrategy",
        "bollinger_breakout": "user_data.strategies.primary.bollinger_breakout.BollingerBreakoutStrategy",
        "simple_close": "user_data.strategies.primary.simple_close.SimpleCloseStrategy",
    }

    # 2次モデルの登録
    _secondary_models = {
        "lightgbm_classifier": "user_data.strategies.secondary.lightgbm_classifier.LightGBMClassifier",
        "xgboost_classifier": "user_data.strategies.secondary.xgboost_classifier.XGBoostClassifier",
        "catboost_classifier": "user_data.strategies.secondary.catboost_classifier.CatBoostClassifier",
    }

    @classmethod
    def load_primary(cls, config: dict):
        """1次戦略をロード

        Args:
            config: two_tier_strategy設定

        Returns:
            PrimaryStrategyBaseインスタンス

        Raises:
            ValueError: primary名が指定されていない場合
        """
        primary_name = config.get("primary")
        if not primary_name:
            raise ValueError("primary strategy name is required")

        primary_class = cls._load_class(cls._primary_strategies[primary_name])
        primary_params = config.get("primary_params", {})
        return primary_class(primary_params)

    @classmethod
    def load_secondary(cls, config: dict):
        """2次モデルをロード

        Args:
            config: two_tier_strategy設定

        Returns:
            SecondaryModelBaseインスタンス、またはNone（secondary=nullの場合）
        """
        secondary_name = config.get("secondary")
        if not secondary_name:
            return None

        secondary_class = cls._load_class(cls._secondary_models[secondary_name])
        secondary_params = config.get("secondary_params", {})
        return secondary_class(secondary_params)

    @classmethod
    def _load_class(cls, class_path: str):
        """モジュールパスからクラスをロード

        Args:
            class_path: "module.path.ClassName"形式のクラスパス

        Returns:
            ロードされたクラス
        """
        module_path, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
```

### config.jsonの設計

#### 基本形式（2層戦略：ML有効）

```json
{
  "two_tier_strategy": {
    "primary": "atr_breakout",
    "secondary": "lightgbm_classifier",
    "primary_params": {
      "period": 14,              // ATR計算期間
      "multiplier": 0.5,          // ATR乗数
      "execution_mode": "one_candle",  // ラベル生成用: "chase" or "one_candle"
      "fee": 0.00025,            // ラベル生成用: シミュレーション手数料
      "exit_periods": 24,        // ラベル生成用: N期間後のリターン計算
      "pips": 0.5                // ラベル生成用: FEP計算の価格精度
    },
    "secondary_params": {
      "confidence_threshold": 0.6
    }
  },
  "freqai": {
    "enabled": true,
    "identifier": "atr_lightgbm_v1"
    // ... FreqAI設定 ...
  }
}
```

#### 1次モデルのみ（ML無効：secondary=null）

```json
{
  "two_tier_strategy": {
    "primary": "atr_breakout",
    "secondary": null,
    "primary_params": {
      "period": 14,
      "multiplier": 0.5,
      "execution_mode": "one_candle",
      "fee": 0.00025,
      "exit_periods": 24,
      "pips": 0.5
    }
  },
  "freqai": {
    "enabled": false
  }
}
```

#### 平均回帰戦略の例

```json
{
  "two_tier_strategy": {
    "primary": "atr_mean_reversion",
    "secondary": "xgboost_classifier",
    "primary_params": {
      "period": 20,
      "multiplier": 1.0,
      "reversion_threshold": 0.02
    },
    "secondary_params": {
      "confidence_threshold": 0.7
    }
  }
}
```

#### ラベル生成用シンプル戦略

```json
{
  "two_tier_strategy": {
    "primary": "simple_close",
    "secondary": "lightgbm_classifier",
    "primary_params": {
      "threshold": 0.001
    },
    "secondary_params": {
      "confidence_threshold": 0.6
    }
  }
}
```

**設定の特徴**:

- `primary`: 1次戦略名（strategies/primary/配下のファイル/クラスに対応）
- `secondary`: 2次モデル名（strategies/secondary/配下のファイル/クラスに対応、nullで無効化）
- `primary_params`: 1次戦略のパラメータ
- `secondary_params`: 2次モデルのパラメータ

### Optuna最適化への対応（Phase 2）

Phase 2で実装予定のOptuna最適化を考慮した設計:

**最適化対象パラメータの定義**:

```python
# primary/atr_breakout.py
class ATRBreakoutStrategy(PrimaryStrategyBase):
    """ATR価格計算＋ブレイクアウト信号生成戦略"""

    # Optuna最適化用のパラメータ空間定義
    PARAM_SPACE = {
        "period": {"type": "int", "low": 7, "high": 28, "default": 14},
        "multiplier": {"type": "float", "low": 0.1, "high": 2.0, "default": 0.5},
        "lookback": {"type": "int", "low": 1, "high": 5, "default": 1}
    }

    def __init__(self, params: dict):
        super().__init__(params)
        self.period = params.get("period", self.PARAM_SPACE["period"]["default"])
        self.multiplier = params.get("multiplier", self.PARAM_SPACE["multiplier"]["default"])
        self.lookback = params.get("lookback", self.PARAM_SPACE["lookback"]["default"])
```

**Optuna統合イメージ**:

```python
# Phase 2で実装
def optimize_strategy_with_optuna(strategy_name: str, n_trials: int = 100):
    """Optunaで戦略パラメータを最適化"""

    def objective(trial):
        # 戦略クラスからパラメータ空間を取得
        strategy_class = StrategyFactory.get_strategy_class(strategy_name)
        param_space = strategy_class.PARAM_SPACE

        # Optunaでパラメータをサンプリング
        params = {}
        for param_name, space in param_space.items():
            if space["type"] == "int":
                params[param_name] = trial.suggest_int(param_name, space["low"], space["high"])
            elif space["type"] == "float":
                params[param_name] = trial.suggest_float(param_name, space["low"], space["high"])

        # バックテスト実行
        result = run_backtest(strategy_name, params)
        return result["profit"]

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params
```

**利点**:

- 各戦略クラスがパラメータ空間を定義
- config.jsonで戦略を指定すれば、その戦略のパラメータ空間が自動的に使用される
- 新しい戦略を追加しても、Optuna最適化ロジックの変更は不要

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
  - configからのパラメータ取得（fee, hold_periods, execution_mode等）

- ✅ **SecondaryModelBase抽象クラス**
  - `integrate_with_freqai()`: FreqAI統合
  - `filter_signals()`: ML予測によるフィルタリング

- ✅ **LightGBMClassifier実装**
  - FreqAI統合による予測データ取得
  - ML予測に基づく注文判定

- ✅ **TwoTierStrategy統合クラス（IStrategy継承）**
  - `populate_indicators()`: 価格計算とML統合
  - `populate_entry_trend()` / `populate_exit_trend()`: エントリー/エグジットシグナル生成
  - `custom_entry_price()` / `custom_exit_price()`: 指値価格設定
  - `set_freqai_targets()`: FreqAI訓練用ラベル生成

- ✅ **StrategyFactory基本機能**
  - 名前ベースの戦略/モデル選択
  - configからのインスタンス生成

- ✅ **config.json設定**
  - ML有効/無効の切り替え（secondary=null）
  - 約定シミュレーション方法の選択（execution_mode）
  - パラメータのconfig管理（fee, hold_periods等）

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
mkdir -p user_data/strategies/secondary
```

### ステップ2: 抽象基底クラスの実装

1. `primary/base.py`: PrimaryStrategyBase抽象クラス
2. `secondary/base.py`: SecondaryModelBase抽象クラス

### ステップ3: 既存戦略の移行

1. 既存の価格ブレイクアウト戦略を`primary/atr_breakout.py`として独立化
2. 既存のFreqAI統合を`secondary/lightgbm_classifier.py`として独立化

### ステップ4: ファクトリーの改修

1. StrategyFactoryに戦略ロード機能を追加
2. 名前解決とクラスロード機能の実装
3. 既存のTwoTierStrategyを新しいアーキテクチャに対応

### ステップ5: config.jsonの拡張

1. 新しい設定形式のサポート追加
2. バリデーション機能の追加
3. パラメータ空間定義の実装（Optuna対応）

### ステップ6: 新戦略の追加

1. 平均回帰戦略（primary/atr_mean_reversion.py）
2. ボリンジャーバンド戦略（primary/bollinger_breakout.py）
3. シンプルclose予測戦略（primary/simple_close.py）

### ステップ7: テストとドキュメント

1. 各戦略の単体テスト
2. 統合テスト（2層戦略全体）
3. 新戦略追加ガイドのドキュメント化
4. マイグレーションガイドの作成

## テスト要件

### 重要性

約定シミュレーションロジックは取引の根幹となるため、**ミスがあると大きな損失に繋がる**。Phase 1実装時に包括的なテストコードを必ず用意すること。

### テスト対象

#### 1. 約定シミュレーションロジック（最重要）

**ファイル**: `tests/primary/test_atr_breakout.py`

**テスト項目**:

- ✅ **chase方式の動作検証**
  - Force Entry Price（FEP）計算の正確性
  - 追いかけ型約定シミュレーションの正確性
  - 手数料計算の正確性

- ✅ **one_candle方式の動作検証**
  - 次足での約定判定の正確性
  - 約定しない場合のリターン=0の確認
  - 手数料計算の正確性

- ✅ **エッジケースの検証**
  - データ不足時の挙動（最初の数行）
  - 価格異常時の挙動（0以下の価格等）
  - NaN/Infの適切な処理

#### 2. ラベル生成ロジック

**ファイル**: `tests/utils/test_two_tier_strategy.py`

**テスト項目**:

- ✅ リターン > 0 で成功ラベル（1）
- ✅ リターン <= 0 で失敗ラベル（0）
- ✅ execution_mode切り替え時のラベル変化確認

### テストデータ

**サンプルデータ**:

- わかりやすい整数値で生成

### テスト実行基準

**Phase 1完了条件**:

- 全テストケースがパス
- リターンやラベルの計算が正しいかを重点的に見る
- それさえできていれば、全体での網羅性は重視しない
- 時系列データを扱うシステムであるため、データリーク(将来のデータを使っていないか)が最重要である

## 利点

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
   - secondary/: 2次モデルの具体的実装

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
   - 各戦略クラスがパラメータ空間を定義
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

Phase 2以降で本設計を実装することで、保守性・拡張性・柔軟性を大幅に向上させることができる。

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
