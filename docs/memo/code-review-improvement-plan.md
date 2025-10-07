# コードレビュー改善方針

## 全体的な実装ルール

### コメントに関するルール
1. **「○○方式」「○○式」などの抽象的な記述は使用しない**
   - 具体的な処理内容や目的を記述する
   - 例: ❌ "richmanbtc方式による価格調整" → ✅ "ML予測が0の場合、10%オフセットで取引を無効化"

2. **コードの動作を単に繰り返す説明は避ける**
   - 「なぜその実装にしたか」を説明する
   - 例: ❌ "ループで要素を処理する" → ✅ "高速化のため、リスト内包表記を使用"

3. **指示に従って実装する際、「Aの方法による○○処理」と書かない**
   - 処理の目的や背景を説明する

### エラーハンドリングに関するルール
1. **例外の握り潰しはしない**
   - 適切にエラーをハンドリングし、呼び出し元に伝播させる
   - 大部分をtry-exceptで囲むのではなく、必要な箇所で発生しうる例外を個別に扱う

2. **不正な値やデータ不足の場合はエラーを発生させる**
   - フォールバック処理で無理やり後続処理に進まない
   - 例: ATRを計算できない場合は処理を停止し、エラーを発生させる

3. **windowサイズの関係でNaN値が生じる場合は該当行を除外**
   - フォールバック値で埋めるのではなく、データから除外する

### コード品質に関するルール
1. **必要に応じて関数やクラスに分割し、可読性・拡張性を高める**

2. **条件を整理して見通しの良い構造にする**
   - フォールバック処理や局所的な分岐を増やすのではなく、要件を満たす構造を設計する

3. **ログは必要な箇所にだけ、問題調査に役立つ粒度で出力**
   - 不要に大量のログを追加しない

### テスタビリティに関するルール
1. **副作用の少ない関数を設計する**

2. **入出力が明確になるように、関数の引数と戻り値を意識する**

## レビュー実施日

2025-10-03

## 対象ブランチ

- 対象: `issue/3_1_refactoring`
- 比較元: `develop`

## 改善優先度

### 優先度1: コメントの改善（即座に対応可能）

#### 1.1 ML予測による価格調整ロジックのコメント改善

**ファイル**: `user_data/strategies/atr_ml_strategy.py:142-162`

**現状**:

```python
# 大きなオフセット（取引を実質的に無効化）
large_offset = dataframe["close"] * 0.1  # 10%のオフセット
```

**改善案**:

```python
# ML予測が0（損失予測）の場合、取引を避けるため
# 10%のオフセットで指値価格を市場価格から大きく外し、実質的に約定不可能にする
# 1次モデル(ATR)の指値は維持しつつ、2次モデル(ML)によるフィルタリングを実現
large_offset = dataframe["close"] * 0.1
```

#### 1.2 train()メソッドの使用状況確認と削除検討

**ファイル**: `user_data/strategies/utils/strategy_factory.py:51-56`

**現状**:
- `MLTrainerBase.train()`抽象メソッドと`FreqAIMLTrainer.train()`プレースホルダー実装が存在
- FreqAIトレーニングは`populate_indicators`内の`freqai.start()`で自動実行される

**改善方針**:
1. `train()`メソッドの使用箇所を確認
2. 使用されていない場合は、`MLTrainerBase`インターフェースから削除
3. 使用されている場合は、なぜこのメソッドが必要なのか理由をコメントに記載

#### 1.3 ATRフォールバック処理の削除

**ファイル**: `user_data/strategies/utils/price_calculator.py:147-153`

**問題点**:
- `calculate_price_single`メソッドで2%の固定ATR概算を使用
- このメソッド自体が使用されていない可能性が高い（参照が見つからない）
- ATRを計算できない状況は異常であり、フォールバックで処理を継続すべきではない

**改善方針**:
1. `calculate_price_single`メソッドの使用箇所を確認
2. 使用されていない場合は、メソッドごと削除
3. 使用されている場合は:
   - ATRが計算できない時点でエラーを発生させる
   - windowサイズの関係でNaN値が生じる場合は、該当行をデータから除外
   - フォールバック処理は削除

### 優先度2: 関数分割による可読性向上

#### 2.1 ML統合信号生成メソッドの分割

**ファイル**: `user_data/strategies/utils/strategy_factory.py:201-241`

**問題点**: メソッドが40行と長く、複数の責務（ML予測取得、信頼度フィルタリング、信号生成）が混在

**改善案**:

```python
def _generate_ml_integrated_signals(self, dataframe, metadata: Dict[str, Any]):
    """ML統合信号生成"""
    pair = metadata.get("pair", "unknown")
    result = dataframe.copy()
    result["enter_long"] = 0
    result["enter_short"] = 0

    # ML予測データの取得と検証
    ml_data = self._extract_ml_prediction_data(dataframe, pair)
    if ml_data is None:
        return result

    # ML統合条件の計算
    long_condition, short_condition = self._calculate_ml_entry_conditions(
        dataframe, ml_data, self.config
    )

    # 信号設定
    result.loc[long_condition, "enter_long"] = 1
    result.loc[short_condition, "enter_short"] = 1

    self._log_signal_summary(pair, long_condition, short_condition)
    return result

def _extract_ml_prediction_data(
    self, dataframe: pd.DataFrame, pair: str
) -> Optional[Dict[str, pd.Series]]:
    """ML予測データの抽出と検証

    Returns:
        ML予測データ辞書、または予測データが不足している場合はNone
    """
    has_prediction = "&-prediction" in dataframe.columns
    has_probability = "&-probability" in dataframe.columns

    if not has_prediction:
        logger.warning(f"ML prediction data missing for {pair}")
        return None

    return {
        "prediction": dataframe["&-prediction"] == 1,
        "probability": dataframe["&-probability"] if has_probability else None,
    }

def _calculate_ml_entry_conditions(
    self, dataframe: pd.DataFrame, ml_data: Dict[str, pd.Series], config: Dict[str, Any]
) -> Tuple[pd.Series, pd.Series]:
    """ML統合エントリー条件の計算

    Returns:
        (long_condition, short_condition)のタプル
    """
    ml_prediction = ml_data["prediction"]
    ml_probability = ml_data["probability"]

    # 信頼度フィルタリング
    confidence_filter = self._apply_confidence_threshold(
        ml_probability, config.get("entry", {}).get("confidence_threshold", 0.6)
    )

    # 価格データの有効性チェック
    price_valid = (dataframe["buy_price"] > 0) & (dataframe["sell_price"] > 0)

    # ML統合条件
    long_condition = ml_prediction & confidence_filter & price_valid
    short_condition = ~ml_prediction & confidence_filter & price_valid

    return long_condition, short_condition

def _apply_confidence_threshold(
    self, probability: Optional[pd.Series], threshold: float
) -> pd.Series:
    """信頼度閾値フィルタの適用

    Args:
        probability: ML予測確率（0-1の範囲）
        threshold: 信頼度閾値

    Raises:
        ValueError: probabilityがNoneまたはthresholdが不正な値の場合
    """
    if probability is None:
        raise ValueError("Probability data is required for confidence filtering")
    if not (0 <= threshold <= 1):
        raise ValueError(f"Threshold must be between 0 and 1, got {threshold}")

    return probability >= threshold

def _log_signal_summary(
    self, pair: str, long_condition: pd.Series, short_condition: pd.Series
) -> None:
    """信号生成サマリーのログ出力"""
    long_signals = long_condition.sum()
    short_signals = short_condition.sum()
    logger.info(f"ML integrated signals {pair}: long={long_signals}, short={short_signals}")
```

#### 2.2 populate_indicatorsメソッドの分割

**ファイル**: `user_data/strategies/atr_ml_strategy.py:125-177`

**問題点**: FreqAI呼び出し、価格調整、エラーハンドリングが混在

**改善案**:

```python
def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    """指標計算（統合アーキテクチャ）

    例外が発生した場合は適切にハンドリングし、呼び出し元に伝播させる
    """
    pair = metadata.get("pair", "unknown")

    # 1次モデルによる価格計算
    dataframe = self._calculate_primary_prices(dataframe)

    # FreqAI統合（有効時のみ）
    if self.freqai_enabled:
        dataframe = self._integrate_freqai_predictions(dataframe, metadata)

    logger.debug(f"Indicators calculation completed: {pair}, records={len(dataframe)}")
    return dataframe

def _calculate_primary_prices(self, dataframe: pd.DataFrame) -> pd.DataFrame:
    """1次モデルによる価格計算

    統合戦略の1次モデル（ATR等）でbuy_price/sell_priceを計算

    Raises:
        ValueError: ATR計算に必要なデータが不足している場合
    """
    return self.two_tier_strategy.primary_model.calculate_entry_prices(dataframe)

def _integrate_freqai_predictions(
    self, dataframe: pd.DataFrame, metadata: dict
) -> pd.DataFrame:
    """FreqAI予測の統合とML調整

    FreqAIによる予測を取得し、ML予測に基づく価格調整を適用

    Raises:
        Exception: FreqAI予測取得または価格調整に失敗した場合
    """
    # FreqAI予測取得
    dataframe = self.freqai.start(dataframe, metadata, self)

    # ML予測に基づく価格調整
    if "&-prediction" in dataframe.columns:
        dataframe = self._adjust_prices_by_ml_prediction(dataframe)
        logger.debug("ML-based price adjustment applied")

    logger.debug("FreqAI prediction data integrated successfully")
    return dataframe

def _adjust_prices_by_ml_prediction(self, dataframe: pd.DataFrame) -> pd.DataFrame:
    """ML予測に基づく指値価格調整

    予測が0（損失予測）の場合、10%オフセットで取引を実質無効化
    """
    # ML予測が0（損失予測）の場合、取引を避けるため
    # 10%のオフセットで指値価格を市場価格から大きく外し、実質的に約定不可能にする
    # 1次モデル(ATR)の指値は維持しつつ、2次モデル(ML)によるフィルタリングを実現
    large_offset = dataframe["close"] * 0.1

    buy_pred = dataframe["&-prediction"]
    sell_pred = dataframe["&-prediction"]

    # 予測が0の場合のみ価格を大きく外す
    dataframe["buy_price"] = dataframe["buy_price"].where(
        buy_pred != 0, dataframe["buy_price"] - large_offset
    )
    dataframe["sell_price"] = dataframe["sell_price"].where(
        sell_pred != 0, dataframe["sell_price"] + large_offset
    )

    return dataframe
```

### 優先度3: テスタビリティ向上

#### 3.1 FreqAI設定生成のデフォルト値外出し

**ファイル**: `user_data/strategies/utils/freqai_model_factory.py:55-104`

**改善案**:

```python
class FreqAIModelFactory:
    """FreqAI既存モデル統合ファクトリー"""

    # デフォルト設定定数
    DEFAULT_FEATURE_PARAMS = {
        "include_timeframes": ["5m", "15m", "1h"],
        "include_corr_pairlist": [],
        "label_period_candles": 24,
        "include_shifted_candles": 2,
        "DI_threshold": 0.9,
        "weight_factor": 0,
        "principal_component_analysis": False,
        "use_SVM_to_remove_outliers": True,
        "indicator_periods_candles": [10, 20, 50],
    }

    DEFAULT_DATA_SPLIT_PARAMS = {
        "test_size": 0.33,
        "shuffle": False,
    }

    DEFAULT_TRAINING_PARAMS = {
        "train_period_days": 30,
        "backtest_period_days": 7,
    }

    @classmethod
    def create_freqai_config(cls, strategy_config: Dict[str, Any]) -> Dict[str, Any]:
        """戦略設定からFreqAI設定生成"""
        secondary_config = strategy_config.get("secondary_model", {})

        if not secondary_config.get("enabled", False):
            return {}

        model_type = secondary_config.get("type", "lightgbm_classifier")
        model_params = secondary_config.get("params", {})

        freqai_model = cls.get_model_name(model_type)

        # 設定の統合（デフォルト値 + ユーザー設定）
        feature_params = {**cls.DEFAULT_FEATURE_PARAMS}
        feature_params.update(strategy_config.get("feature_parameters", {}))

        data_split_params = {**cls.DEFAULT_DATA_SPLIT_PARAMS}
        data_split_params.update(strategy_config.get("data_split_parameters", {}))

        training_params = {**cls.DEFAULT_TRAINING_PARAMS, **model_params}
        training_params["identifier"] = f"2tier_{model_type}"

        return {
            "enabled": True,
            "model_training_parameters": training_params,
            "feature_parameters": feature_params,
            "data_split_parameters": data_split_params,
        }
```

#### 3.2 不要なプレースホルダー実装の削除

**ファイル**: `user_data/strategies/utils/strategy_factory.py:317-334`

**方針**: `_generate_basic_features`と`_generate_basic_labels`は実際に使用されていないため、削除を検討。必要な場合は適切に実装する。

## 実装順序

### Phase 1: 構造改善（優先度1）

#### 1.1 使用されていないコードの調査と削除
1. `MLTrainerBase.train()`メソッドの使用箇所調査
2. `ATRPriceCalculator.calculate_price_single()`メソッドの使用箇所調査
3. 使用されていない場合は削除、使用されている場合は適切な実装・コメント追加

#### 1.2 エラーハンドリングの改善
1. 信頼度フィルタで不正な値の場合はエラーを発生させる
2. `populate_indicators`の過度なtry-exceptを削除
3. `_integrate_freqai_predictions`のtry-exceptを削除
4. 各メソッドで適切な例外を発生させ、呼び出し元に伝播させる

#### 1.3 コメントの改善
1. 「○○方式」などの不要な記述を削除
2. 実装の意図が明確になるようコメントを改善

### Phase 2: 関数分割（優先度2）

#### 2.1 ML統合信号生成メソッドの分割
- ML予測データ抽出
- ML統合条件計算
- 信頼度フィルタリング（エラーハンドリング込み）
- ログ出力

#### 2.2 populate_indicatorsメソッドの分割
- 1次モデル価格計算
- FreqAI統合処理
- ML価格調整

### Phase 3: テスタビリティ向上（優先度3）

#### 3.1 FreqAI設定生成のデフォルト値外出し
- クラス定数として定義
- テスト時のオーバーライドを容易にする

#### 3.2 不要なプレースホルダー実装の削除
- `_generate_basic_features`と`_generate_basic_labels`の使用状況確認
- 使用されていない場合は削除

## 備考

- Phase 1は既存機能の動作を改善し、コードの保守性を向上させる
- Phase 2以降は既存テストへの影響を確認しながら段階的に実施
- 各改善は個別にコミット可能な単位で実施
- エラーハンドリングの改善により、問題の早期発見と適切な対処が可能になる
