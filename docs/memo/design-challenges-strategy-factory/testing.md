# テスト戦略

[⬅️ README に戻る](./README.md)

このドキュメントでは、テスト要件、データリーク検出方法、テストコード例について説明します。

## 目次

- [テスト要件概要](#テスト要件概要)
- [データリーク検出チェックリスト](#データリーク検出チェックリスト)
- [データリーク検出テストコード](#データリーク検出テストコード)
- [自動検出スクリプト](#自動検出スクリプトcicd統合用)

## テスト要件概要

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
- OHLCデータの整合性を保つ

### テスト実行基準

**Phase 1完了条件**:

- 全テストケースがパス
- リターンやラベルの計算が正しいかを重点的に見る
- それさえできていれば、全体での網羅性は重視しない
- **時系列データを扱うシステムであるため、データリーク（将来のデータを使っていないか）が最重要である**

## データリーク検出チェックリスト

### 📋 必須チェック項目

#### 1. 未来データの使用箇所確認

- [ ] すべての `.shift(-n)` 使用箇所をリスト化
- [ ] 各使用箇所が訓練時専用（`calculate_returns` 内）か確認
- [ ] `populate_indicators` で `.shift(-n)` を使用していないか確認

#### 2. 特徴量生成の検証

- [ ] `populate_indicators` で生成される全カラムをリスト化
- [ ] 各カラムが時刻 t のデータのみを使用しているか確認
- [ ] rolling/expanding 計算で `min_periods` が適切に設定されているか確認

#### 3. ラベル生成の検証

- [ ] `calculate_returns()` で使用する未来データが特徴量に混入していないか
- [ ] ターゲット計算ロジックで lookahead bias がないか
- [ ] ラベル生成で使用するカラムが dataframe に追加されていないか

#### 4. 時系列分割の検証

- [ ] train/test 分割が時系列順に行われているか
- [ ] test 期間のデータが train に混入していないか
- [ ] K-fold CV を使用している場合、TimeSeriesSplit を使用しているか

## データリーク検出テストコード

### テスト1: 特徴量の未来データ依存性チェック

```python
def test_no_future_data_in_features():
    """populate_indicators で生成される特徴量に未来データが含まれないことを検証"""
    strategy = TwoTierStrategy(config)
    test_data = generate_test_ohlcv(1000)

    df = strategy.populate_indicators(test_data.copy(), {'pair': 'BTC/USDT'})

    # 時刻 t の特徴量が時刻 t+1 以降のデータに依存していないことを確認
    for i in range(100, len(df) - 1):
        # 時刻 i までのデータで特徴量計算
        df_partial = strategy.populate_indicators(
            test_data.iloc[:i+1].copy(),
            {'pair': 'BTC/USDT'}
        )

        # 時刻 i の特徴量が一致することを確認（未来データ不使用の証明）
        for col in df.columns:
            if col.startswith('%') or col.startswith('&'):  # FreqAI特徴量
                continue  # FreqAI予測は除外
            if col in ['open', 'high', 'low', 'close', 'volume']:
                continue  # 元データは除外

            assert np.isclose(
                df.iloc[i][col],
                df_partial.iloc[-1][col],
                equal_nan=True
            ), f"Feature {col} at index {i} depends on future data"
```

### テスト2: populate_indicators での .shift(-n) 検出

```python
def test_no_shift_negative_in_indicators():
    """populate_indicators で .shift(-n) が使われていないことを確認"""
    strategy = TwoTierStrategy(config)

    # ソースコード検査（静的チェック）
    import inspect
    source = inspect.getsource(strategy.populate_indicators)

    # .shift(-n) パターンを検出
    import re
    negative_shifts = re.findall(r'\.shift\s*\(\s*-\s*\d+', source)

    assert len(negative_shifts) == 0, (
        f"Found {len(negative_shifts)} .shift(-n) in populate_indicators: {negative_shifts}"
    )
```

### テスト3: ラベル生成での未来データ隔離確認

```python
def test_label_future_data_isolation():
    """ラベル生成で使用する未来データが特徴量に混入していないことを確認"""
    strategy = TwoTierStrategy(config)
    test_data = generate_test_ohlcv(1000)

    # 特徴量生成
    df_features = strategy.populate_indicators(test_data.copy(), {'pair': 'BTC/USDT'})

    # ラベル生成
    df_labels = strategy.set_freqai_targets(test_data.copy(), {'pair': 'BTC/USDT'})

    # ラベル生成で使用したカラム（&-target_buy, &-target_sell）が
    # 特徴量 DataFrame に存在しないことを確認
    label_cols = [col for col in df_labels.columns if col.startswith('&-target')]

    for col in label_cols:
        assert col not in df_features.columns, (
            f"Label column {col} leaked into features"
        )
```

### テスト4: リターン計算での未来データ使用確認

```python
def test_return_calculation_uses_future_data_correctly():
    """calculate_returns が正しく未来データを使用していることを確認"""
    primary_strategy = ATRBreakoutStrategy(params)
    test_data = generate_test_ohlcv(1000)

    df = primary_strategy.calculate_prices(test_data.copy())
    buy_return, sell_return = primary_strategy.calculate_returns(df)

    # リターン計算で未来データを使用している（期待される動作）
    # 時刻 i のリターンは時刻 i+exit_periods のデータを使用
    exit_periods = primary_strategy.exit_periods

    # 最後の exit_periods 行はリターン計算不可（未来データ不足）
    assert buy_return.iloc[-exit_periods:].isna().all(), (
        "Last rows should be NaN (no future data available)"
    )

    # それ以前の行はリターンが計算されている
    assert not buy_return.iloc[:-exit_periods].isna().all(), (
        "Returns should be calculated for earlier rows"
    )
```

### テスト5: 時系列分割の検証

```python
def test_time_series_split():
    """訓練/テスト分割が時系列順に行われていることを確認"""
    # FreqAI の data_split_parameters を検証
    config = load_config('config.json')

    # shuffle が False であることを確認
    assert config['freqai']['data_split_parameters']['shuffle'] is False, (
        "shuffle must be False for time-series data"
    )

    # test_size が適切な範囲であることを確認
    test_size = config['freqai']['data_split_parameters']['test_size']
    assert 0.1 <= test_size <= 0.3, (
        f"test_size {test_size} should be between 0.1 and 0.3"
    )
```

## 自動検出スクリプト（CI/CD統合用）

### Python AST ベースの検出スクリプト

```python
#!/usr/bin/env python3
"""データリーク自動検出スクリプト"""
import ast
import sys

def detect_shift_negative(file_path):
    """Python ファイル内の .shift(-n) を検出"""
    with open(file_path) as f:
        source = f.read()

    tree = ast.parse(source)
    violations = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if hasattr(node.func, 'attr') and node.func.attr == 'shift':
                if node.args and isinstance(node.args[0], ast.UnaryOp):
                    if isinstance(node.args[0].op, ast.USub):
                        violations.append({
                            'line': node.lineno,
                            'col': node.col_offset,
                            'function': get_function_name(node, tree)
                        })

    return violations

def get_function_name(node, tree):
    """ノードが属する関数名を取得"""
    for parent in ast.walk(tree):
        if isinstance(parent, ast.FunctionDef):
            if node.lineno >= parent.lineno:
                return parent.name
    return 'unknown'

if __name__ == '__main__':
    files = [
        'user_data/strategies/two_tier_strategy.py',
        'user_data/strategies/primary/atr_breakout.py',
        # ...
    ]

    has_violations = False
    for file in files:
        violations = detect_shift_negative(file)
        if violations:
            has_violations = True
            print(f"❌ {file}:")
            for v in violations:
                print(f"  Line {v['line']}: .shift(-n) in {v['function']}()")

    if has_violations:
        sys.exit(1)
    else:
        print("✅ No data leakage detected")
```

### GitHub Actions での統合例

```yaml
# .github/workflows/data-leak-check.yml
name: Data Leak Detection

on: [push, pull_request]

jobs:
  check-data-leak:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      - name: Run data leak detection
        run: python scripts/detect_data_leak.py
```

## テストコード配置

```
tests/
├── primary/
│   ├── test_atr_breakout.py          # ATRBreakoutStrategy単体テスト
│   ├── test_execution_simulation.py  # 約定シミュレーションテスト
│   └── test_fep_calculation.py       # FEP計算テスト
├── utils/
│   ├── test_two_tier_strategy.py     # TwoTierStrategy統合テスト
│   ├── test_strategy_factory.py      # StrategyFactoryテスト
│   └── test_config_validation.py     # Config バリデーションテスト
└── data_leak/
    ├── test_feature_isolation.py     # 特徴量データリークテスト
    ├── test_label_isolation.py       # ラベルデータリークテスト
    └── test_time_series_split.py     # 時系列分割テスト
```

## 関連ドキュメント

- [実装ガイド](./implementation.md) - Phase 1完了条件とテスト要件
- [アーキテクチャ設計](./architecture.md) - ATRBreakoutStrategyの実装詳細
- [設定管理](./configuration.md) - Configバリデーションルール

[⬅️ README に戻る](./README.md)
