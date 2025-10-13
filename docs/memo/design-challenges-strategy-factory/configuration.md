# 設定管理

[⬅️ README に戻る](./README.md)

このドキュメントでは、config.jsonの設計、設定パターン、バリデーションルールについて説明します。

## 目次

- [config.json設計](#configjson設計)
- [設定パターン](#設定パターン)
- [設定の組み合わせパターン](#設定の組み合わせパターン)

## config.json設計

### 基本形式（2層戦略：ML有効）

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

### 1次モデルのみ（ML無効：secondary=null）

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

### 平均回帰戦略の例（Phase 2）

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

### ラベル生成用シンプル戦略（Phase 2）

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

## 設定パターン

### パラメータ説明

#### two_tier_strategy設定

- **`primary`** (required): 1次戦略名
  - 値: `"atr_breakout"`, `"atr_mean_reversion"`, `"bollinger_breakout"`, `"simple_close"`
  - 対応: `strategies/primary/` 配下のファイル/クラス

- **`secondary`** (optional, nullable): 2次モデル名
  - 値: `"lightgbm_classifier"`, `"xgboost_classifier"`, `"catboost_classifier"`, `null`
  - 対応: FreqAIモデル（`user_data/freqaimodels/` 配下）
  - `null` の場合: ML無効モード

- **`primary_params`** (optional): 1次戦略のパラメータ
  - 戦略ごとに異なるパラメータ
  - ATRBreakoutStrategyの場合:
    - `period`: ATR計算期間（デフォルト: 14）
    - `multiplier`: ATR乗数（デフォルト: 0.5）
    - `execution_mode`: 約定シミュレーション方法（"chase" or "one_candle"）
    - `fee`: シミュレーション用手数料（デフォルト: 0.00025）
    - `exit_periods`: N期間後のリターン計算（デフォルト: 24）
    - `pips`: FEP計算の価格精度（オプション）

- **`secondary_params`** (optional): 2次モデルのパラメータ
  - モデルごとに異なるパラメータ
  - LightGBMClassifierの場合:
    - `confidence_threshold`: 予測信頼度の閾値（デフォルト: 0.6）

#### freqai設定

- **`enabled`**: FreqAI有効/無効
  - `true`: ML有効（`secondary`必須）
  - `false`: ML無効（`secondary`は`null`）

- **`identifier`**: モデル識別子
  - 例: `"atr_lightgbm_v1"`, `"two_tier_buy_v1"`

- **`model_name`**: FreqAIモデルクラス名
  - 例: `"TwoTierLightGBMClassifier"`

- **`model_training_parameters`**: モデル訓練パラメータ
  - モデルごとに異なる

### FreqAI マルチターゲット設定（Buy/Sell独立モデル）

```json
{
  "two_tier_strategy": {
    "primary": "atr_breakout",
    "primary_params": {
      "period": 14,
      "multiplier": 0.5,
      "execution_mode": "one_candle",
      "fee": 0.00025,
      "exit_periods": 24
    }
  },
  "freqai_buy": {
    "enabled": true,
    "identifier": "two_tier_buy_v1",
    "model_name": "TwoTierLightGBMClassifier",
    "model_training_parameters": {
      "n_estimators": 100,
      "learning_rate": 0.1
    }
  },
  "freqai_sell": {
    "enabled": true,
    "identifier": "two_tier_sell_v1",
    "model_name": "TwoTierLightGBMClassifier",
    "model_training_parameters": {
      "n_estimators": 100,
      "learning_rate": 0.1
    }
  }
}
```

**特徴**:

- `freqai_buy` / `freqai_sell`: 買い/売り独立したFreqAIモデル
- 各モデルは独立して訓練・予測を実行
- `identifier`で区別（`_buy` / `_sell` サフィックス）

## 設定の組み合わせパターン

### 有効な組み合わせ

| freqai.enabled | secondary | 動作モード | 説明 |
|---------------|-----------|---------|------|
| true | "lightgbm_classifier" | **ML予測でフィルタリング（推奨）** | buy/sell独立したML予測でエントリー判定 |
| false | null | **1次戦略のみ（ML未使用）** | 指値価格で常に両方向エントリー |

### エラーとなる組み合わせ

| freqai.enabled | secondary | エラー理由 |
|---------------|-----------|-----------|
| true | null | FreqAI有効ならsecondary必須 |
| false | "lightgbm_classifier" | secondaryにはFreqAI必須 |

### バリデーションルール

TwoTierStrategyの`__init__`で以下のバリデーションを実行：

```python
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
```

### 設定例とユースケース

#### ユースケース1: ATR戦略のみでバックテスト

**目的**: ML無しでATR戦略の基本性能を確認

```json
{
  "two_tier_strategy": {
    "primary": "atr_breakout",
    "secondary": null,
    "primary_params": {
      "period": 14,
      "multiplier": 0.5
    }
  },
  "freqai": {
    "enabled": false
  }
}
```

#### ユースケース2: ATR + LightGBMでML学習・バックテスト

**目的**: ML予測でエントリーをフィルタリング

```json
{
  "two_tier_strategy": {
    "primary": "atr_breakout",
    "secondary": "lightgbm_classifier",
    "primary_params": {
      "period": 14,
      "multiplier": 0.5,
      "execution_mode": "one_candle",
      "fee": 0.00025,
      "exit_periods": 24
    }
  },
  "freqai": {
    "enabled": true,
    "identifier": "atr_lgbm_v1",
    "model_name": "TwoTierLightGBMClassifier"
  }
}
```

#### ユースケース3: 約定シミュレーション方法の比較

**目的**: chase vs one_candle の性能比較

**Config 1** (chase方式):

```json
{
  "two_tier_strategy": {
    "primary": "atr_breakout",
    "primary_params": {
      "execution_mode": "chase",
      "fee": 0.00025,
      "exit_periods": 24
    }
  }
}
```

**Config 2** (one_candle方式):

```json
{
  "two_tier_strategy": {
    "primary": "atr_breakout",
    "primary_params": {
      "execution_mode": "one_candle",
      "fee": 0.00025,
      "exit_periods": 24
    }
  }
}
```

## 関連ドキュメント

- [アーキテクチャ設計](./architecture.md) - PrimaryStrategyBaseとパラメータ設計
- [FreqAI統合](./freqai-integration.md) - FreqAI設定の詳細
- [実装ガイド](./implementation.md) - Phase 1での設定実装範囲

[⬅️ README に戻る](./README.md)
