# アーキテクチャ設計

[⬅️ README に戻る](./README.md)

このドキュメントでは、Strategy Factoryアーキテクチャの全体構造、クラス設計、および責務の分離について説明します。

## 目次

- [現状の実装と問題点](#現状の実装と問題点)
- [設計原則](#設計原則)
- [ディレクトリ構造](#ディレクトリ構造)
- [クラス設計](#クラス設計)
- [StrategyFactory](#strategyfactory)
- [特徴量計算の責任明確化](#特徴量計算の責任明確化)

## 現状の実装と問題点

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

## ディレクトリ構造

```
user_data/strategies/
├── primary/                    # 1次戦略（ドメインロジック）
│   ├── __init__.py
│   ├── base.py                # PrimaryStrategyBase抽象クラス
│   ├── atr_breakout.py        # ATRBreakoutStrategy
│   ├── atr_mean_reversion.py  # ATRMeanReversionStrategy (Phase 2)
│   ├── bollinger_breakout.py  # BollingerBreakoutStrategy (Phase 2)
│   └── simple_close.py        # SimpleCloseStrategy (Phase 2)
├── utils/                      # ヘルパークラス
│   ├── strategy_factory.py    # StrategyFactory（戦略ロード用）
│   ├── price_calculator.py    # 既存の価格計算器（後方互換性用）
│   └── freqai_model_factory.py
└── two_tier_strategy.py       # TwoTierStrategy(IStrategy) - Freqtradeエントリーポイント
```

**ポイント**:

- `primary/`: 1次戦略のドメインロジック（IStrategyは**継承しない**）
- `utils/`: ファクトリーパターン実装（ヘルパークラス）
- `two_tier_strategy.py`: **FreqtradeのIStrategyを継承**し、config駆動で1次戦略・2次モデルを切り替え

**実行例**:

```bash
freqtrade backtesting --strategy TwoTierStrategy --config config.json
```

## クラス設計

### 1次戦略（Primary Strategy）

#### PrimaryStrategyBase抽象クラス

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

        約定シミュレーション方法の詳細は「学習ラベルの定義」セクションを参照

        Args:
            dataframe: 価格計算済みのDataFrame

        Returns:
            (buy_return, sell_return): 買い/売りそれぞれの理論リターン

        Note:
            - execution_mode設定に基づいて約定シミュレーション方法を切り替え（chase / one_candle）
            - 買いと売りで独立したリターンを計算（両建て対応）
            - 計算されたリターンは、ML学習時にラベル化される（リターン > 0 で成功）
        """
        pass
```

#### ATRBreakoutStrategy実装

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

        約定シミュレーション方法の詳細は「学習ラベルの定義」セクション参照

        Warning:
            このロジックは取引の根幹となるため、ミスがあると大きな損失に繋がる
            必ず包括的なテストコードで検証すること（tests/primary/test_atr_breakout.py）
        """
        if self.execution_mode == "chase":
            return self._calculate_chase_returns(dataframe)
        else:  # one_candle
            return self._calculate_one_candle_returns(dataframe)

    def _calculate_chase_returns(self, dataframe: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        """アプローチ1: エントリー追いかけ型（richmanbtc例1）

        エントリー/エグジット両方で約定するまで追いかける方式
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

        エントリーは1足限定、約定した場合のみリターン計算
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

#### SimpleCloseStrategy実装（Phase 2）

```python
# primary/simple_close.py
class SimpleCloseStrategy(PrimaryStrategyBase):
    """次足close予測用シンプル戦略（従来型ML取引の参考実装）

    用途: MLモデルのみの取引（約定シミュレーションなし）
    特徴: 多くのML取引で使われる単純なclose変化予測
    対比: richmanbtc型は指値注文の約定シミュレーションでラベル生成を工夫
    """

    def __init__(self, params: dict):
        super().__init__(params)
        self.threshold = params.get("threshold", 0.001)

    def calculate_prices(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """現在価格をそのまま使用（成行想定）"""
        dataframe["buy_price"] = dataframe["close"]
        dataframe["sell_price"] = dataframe["close"]
        return dataframe

    def calculate_returns(self, dataframe: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        """次足のclose変化率を返す（従来型ML学習用ラベル）

        Note:
            - richmanbtc型: 指値注文の約定シミュレーションでリターン計算
            - 従来型(本実装): 単純に次足のclose変化率を使用
            - 買い/売りで符号を反転（売りは価格下降で利益）
        """
        # 次足のclose変化率
        future_return = dataframe["close"].pct_change().shift(-1)
        buy_return = future_return      # 価格上昇で利益
        sell_return = -future_return    # 価格下降で利益（符号反転）
        return buy_return, sell_return
```

### TwoTierStrategy（統合クラス）

詳細は [freqai-integration.md](./freqai-integration.md#twotierstrategy統合クラス) を参照してください。

簡略版の概要:

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

        # Config validation
        freqai_enabled = freqai_config.get('enabled', False)
        has_secondary = two_tier_config.get('secondary') is not None

        if has_secondary and not freqai_enabled:
            raise ValueError(
                "Invalid configuration: secondary model is specified but freqai.enabled is False. "
                "Please set freqai.enabled=true when using a secondary model."
            )

        if freqai_enabled and not has_secondary:
            raise ValueError(
                "Invalid configuration: freqai.enabled is True but no secondary model specified. "
                "Please set secondary to a model name (e.g., 'lightgbm_classifier') or disable FreqAI."
            )

        # StrategyFactoryで1次戦略をロード
        from user_data.strategies.utils.strategy_factory import StrategyFactory
        self.primary_strategy = StrategyFactory.load_primary(two_tier_config)
        self.is_ml_enabled = freqai_enabled

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """指標計算（価格計算＋ML統合）"""
        # 1次戦略: 指値価格計算
        dataframe = self.primary_strategy.calculate_prices(dataframe)

        # FreqAI直接呼び出し（ML有効時のみ）
        if self.is_ml_enabled:
            # Buy/Sellモデルの予測取得
            # 詳細は freqai-integration.md 参照
            pass

        return dataframe

    # 他のメソッドは freqai-integration.md を参照
```

## StrategyFactory

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

## 特徴量計算の責任明確化

### 許容される重複

**1次モデルと2次モデルでの ATR 計算重複は許容**:

```python
# ✅ 許容: 1次モデルの ATR 計算（注文価格算出用）
class ATRBreakoutStrategy(PrimaryStrategyBase):
    def calculate_prices(self, dataframe):
        atr = ta.ATR(dataframe, timeperiod=self.period)  # 注文価格計算用
        dataframe["buy_price"] = dataframe["close"] - atr * self.multiplier
        # ...

# ✅ 許容: 2次モデルの ATR 計算（ML特徴量用）
class TwoTierLightGBMClassifier(BaseClassifierModel):
    def populate_indicators(self, dataframe, metadata):
        dataframe['%atr_14'] = ta.ATR(dataframe, timeperiod=14)  # ML特徴量用
        # ...
```

**理由**: 用途が異なるため（注文価格 vs ML入力）、重複しても問題なし

### 削除すべき重複

**TwoTierStrategy での重複計算は削除**:

```python
# ❌ 削除: 同じ目的での重複計算
def populate_indicators(self, dataframe, metadata):
    # PrimaryStrategy で既に計算済みなので不要
    # dataframe['atr'] = ta.ATR(...)  # 削除

    dataframe = self.primary_strategy.calculate_prices(dataframe)  # これだけで OK
    # ...
```

## 関連ドキュメント

- [FreqAI統合の詳細](./freqai-integration.md) - FreqAI統合アーキテクチャとラベル生成フロー
- [設定管理](./configuration.md) - config.json設計と設定パターン
- [実装ガイド](./implementation.md) - Phase 1実装範囲とステップ

[⬅️ README に戻る](./README.md)
