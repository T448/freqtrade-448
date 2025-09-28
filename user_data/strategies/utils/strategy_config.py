"""2層戦略設定

この設定を変更することで、コードを変更せずに手法を切り替え可能
"""

# 基本戦略設定
TWO_TIER_STRATEGY_CONFIG = {
    # 1次モデル設定（価格計算）
    "primary_model": {
        "type": "atr",  # 選択肢: "atr", "bollinger", "ma", "rsi"
        "params": {"period": 14, "multiplier": 0.5},
    },
    # 2次モデル設定（機械学習）
    "secondary_model": {
        "enabled": True,
        "type": "lightgbm_classifier",  # 選択肢: "lightgbm_classifier", "xgboost_classifier", "catboost_classifier"
        "params": {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "num_leaves": 31,
            "max_depth": -1,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
        },
    },
    # 前処理設定
    "preprocessor": {
        # テクニカル指標設定
        "indicators": {
            "sma": {"period": 20},
            "ema": {"period": 12},
            "rsi": {"period": 14},
            "macd": {"fast": 12, "slow": 26, "signal": 9},
            "bollinger": {"period": 20, "std": 2},
        },
        # ラベル生成設定
        "label": {
            "method": "return_based",  # "return_based" or "price_based"
            "look_ahead": 1,
            "threshold": 0.001,
        },
        # 品質管理設定
        "quality": {
            "max_missing_ratio": 0.1,
            "min_predictive_power": 0.01,
            "max_correlation": 0.95,
        },
    },
    # エントリー戦略設定
    "entry": {"confidence_threshold": 0.6, "min_data_length": 50},
    # FreqAI統合設定
    "train_period_days": 30,
    "backtest_period_days": 7,
    "include_timeframes": ["15m", "1h", "4h"],
    "include_corr_pairs": [],
    "label_period_candles": 24,
    "include_shifted_candles": 2,
    "DI_threshold": 0.9,
    "weight_factor": 0,
    "use_pca": False,
    "use_svm_outlier_removal": True,
    "indicator_periods": [10, 20, 50],
    "test_size": 0.33,
    "shuffle": False,
}

# 設定バリエーション例

# ボリンジャーバンド + XGBoost設定
BOLLINGER_XGBOOST_CONFIG = {
    "primary_model": {"type": "bollinger", "params": {"period": 20, "std_dev": 2}},
    "secondary_model": {
        "enabled": True,
        "type": "xgboost_classifier",
        "params": {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        },
    },
    # 他の設定は共通
    **{
        k: v
        for k, v in TWO_TIER_STRATEGY_CONFIG.items()
        if k not in ["primary_model", "secondary_model"]
    },
}

# 移動平均 + LightGBM設定
MA_LIGHTGBM_CONFIG = {
    "primary_model": {
        "type": "ma",
        "params": {"fast_period": 12, "slow_period": 26, "offset_ratio": 0.002},
    },
    "secondary_model": {
        "enabled": True,
        "type": "lightgbm_classifier",
        "params": {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "num_leaves": 31,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
        },
    },
    # 他の設定は共通
    **{
        k: v
        for k, v in TWO_TIER_STRATEGY_CONFIG.items()
        if k not in ["primary_model", "secondary_model"]
    },
}

# CatBoost設定（高精度モデル）
ATR_CATBOOST_CONFIG = {
    "primary_model": {"type": "atr", "params": {"period": 14, "multiplier": 0.5}},
    "secondary_model": {
        "enabled": True,
        "type": "catboost_classifier",
        "params": {
            "iterations": 100,
            "learning_rate": 0.1,
            "depth": 6,
            "l2_leaf_reg": 3.0,
            "bootstrap_type": "Bernoulli",
            "subsample": 0.8,
        },
    },
    # 他の設定は共通
    **{
        k: v
        for k, v in TWO_TIER_STRATEGY_CONFIG.items()
        if k not in ["primary_model", "secondary_model"]
    },
}

# ML無効設定（価格戦略のみ）
PRICE_ONLY_CONFIG = {**TWO_TIER_STRATEGY_CONFIG, "secondary_model": {"enabled": False}}

# 高頻度取引設定（短期間パラメータ）
HIGH_FREQ_CONFIG = {
    "primary_model": {
        "type": "atr",
        "params": {
            "period": 7,  # より短期間
            "multiplier": 0.3,  # より狭いスプレッド
        },
    },
    "secondary_model": {
        "enabled": True,
        "type": "lightgbm_classifier",
        "params": {
            "n_estimators": 50,  # 高速化
            "learning_rate": 0.2,  # 高速化
            "num_leaves": 15,  # 簡素化
            "feature_fraction": 0.9,
            "bagging_fraction": 0.9,
            "bagging_freq": 3,
        },
    },
    # FreqAI設定も高頻度向けに調整
    "train_period_days": 7,  # 短期間
    "backtest_period_days": 1,
    "include_timeframes": ["15m", "30m", "1h"],  # メインタイムフレーム対応
    "label_period_candles": 5,  # 短期予測
    **{
        k: v
        for k, v in TWO_TIER_STRATEGY_CONFIG.items()
        if k
        not in [
            "primary_model",
            "secondary_model",
            "train_period_days",
            "backtest_period_days",
            "include_timeframes",
            "label_period_candles",
        ]
    },
}

# 保守的設定（長期・安定重視）
CONSERVATIVE_CONFIG = {
    "primary_model": {
        "type": "atr",
        "params": {
            "period": 21,  # より長期間
            "multiplier": 0.8,  # より広いスプレッド
        },
    },
    "secondary_model": {
        "enabled": True,
        "type": "lightgbm_classifier",
        "params": {
            "n_estimators": 200,  # 高精度
            "learning_rate": 0.05,  # 保守的
            "num_leaves": 50,  # 複雑モデル
            "feature_fraction": 0.7,  # オーバーフィッティング防止
            "bagging_fraction": 0.7,
            "bagging_freq": 10,
            "min_data_in_leaf": 50,  # 安定性重視
        },
    },
    # 保守的なFreqAI設定
    "entry": {
        "confidence_threshold": 0.8,  # 高い信頼度要求
        "min_data_length": 100,
    },
    "train_period_days": 60,  # 長期間
    "backtest_period_days": 14,
    "include_timeframes": ["15m", "1h", "4h"],  # 長時間足
    "label_period_candles": 48,  # 長期予測
    **{
        k: v
        for k, v in TWO_TIER_STRATEGY_CONFIG.items()
        if k
        not in [
            "primary_model",
            "secondary_model",
            "entry",
            "train_period_days",
            "backtest_period_days",
            "include_timeframes",
            "label_period_candles",
        ]
    },
}


# 設定選択ヘルパー関数
def get_strategy_config(config_name: str = "default") -> dict:
    """設定名から戦略設定を取得

    Args:
        config_name: 設定名（"default", "bollinger_xgboost", "ma_lightgbm",
                    "catboost", "price_only", "high_freq", "conservative"）

    Returns:
        選択された戦略設定

    Raises:
        ValueError: 未知の設定名の場合
    """
    configs = {
        "default": TWO_TIER_STRATEGY_CONFIG,
        "bollinger_xgboost": BOLLINGER_XGBOOST_CONFIG,
        "ma_lightgbm": MA_LIGHTGBM_CONFIG,
        "catboost": ATR_CATBOOST_CONFIG,
        "price_only": PRICE_ONLY_CONFIG,
        "high_freq": HIGH_FREQ_CONFIG,
        "conservative": CONSERVATIVE_CONFIG,
    }

    if config_name not in configs:
        available = list(configs.keys())
        raise ValueError(f"Unknown config name: {config_name}. Available: {available}")

    return configs[config_name]


# 設定一覧取得
def list_available_configs() -> list:
    """利用可能な設定一覧を取得

    Returns:
        設定名のリスト
    """
    return [
        "default",
        "bollinger_xgboost",
        "ma_lightgbm",
        "catboost",
        "price_only",
        "high_freq",
        "conservative",
    ]
