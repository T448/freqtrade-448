"""Config validation tests - 設定検証テスト

このモジュールは、TwoTierStrategy.__init__()で実行される
Config検証ロジックが正しく動作することを検証するテストを提供します。

テスト方針:
- freqai.enabled と secondary の整合性チェック
- 無効な設定でValueErrorが発生すること
- 有効な設定でエラーが発生しないこと
"""

import pytest

from user_data.strategies.two_tier_strategy import TwoTierStrategy


class TestConfigValidation:
    """Config検証テストスイート"""

    def test_invalid_config_secondary_without_freqai(self):
        """無効な設定: secondaryが指定されているがfreqai.enabled=False

        この設定は不整合であり、ValueErrorが発生すべき
        """
        config = {
            "two_tier_strategy": {
                "primary": "atr_breakout",
                "secondary": "lightgbm_classifier",  # secondaryが指定されている
                "primary_params": {
                    "period": 14,
                    "multiplier": 0.5,
                    "execution_mode": "one_candle",
                    "fee": 0.00025,
                    "exit_periods": 24,
                    "pips": 0.5,
                },
            },
            "freqai": {
                "enabled": False,  # FreqAIが無効
            },
        }

        with pytest.raises(ValueError) as exc_info:
            TwoTierStrategy(config)

        assert "secondary model is specified but freqai.enabled is False" in str(exc_info.value)

    def test_invalid_config_freqai_without_secondary(self):
        """無効な設定: freqai.enabled=Trueだがsecondaryが指定されていない

        この設定は不整合であり、ValueErrorが発生すべき
        """
        config = {
            "two_tier_strategy": {
                "primary": "atr_breakout",
                "secondary": None,  # secondaryが未指定
                "primary_params": {
                    "period": 14,
                    "multiplier": 0.5,
                    "execution_mode": "one_candle",
                    "fee": 0.00025,
                    "exit_periods": 24,
                    "pips": 0.5,
                },
            },
            "freqai": {
                "enabled": True,  # FreqAIが有効
            },
        }

        with pytest.raises(ValueError) as exc_info:
            TwoTierStrategy(config)

        assert "freqai.enabled is True but no secondary model specified" in str(exc_info.value)

    def test_valid_config_ml_enabled(self):
        """有効な設定: ML有効モード（freqai.enabled=True + secondary指定）

        この設定は正常であり、エラーが発生しないこと
        """
        config = {
            "two_tier_strategy": {
                "primary": "atr_breakout",
                "secondary": "lightgbm_classifier",  # secondaryが指定されている
                "primary_params": {
                    "period": 14,
                    "multiplier": 0.5,
                    "execution_mode": "one_candle",
                    "fee": 0.00025,
                    "exit_periods": 24,
                    "pips": 0.5,
                },
            },
            "freqai": {
                "enabled": True,  # FreqAIが有効
                "identifier": "test_ml_on",
                "model_name": "TwoTierLightGBMClassifier",
            },
        }

        # エラーが発生しないことを確認
        strategy = TwoTierStrategy(config)

        assert strategy.is_ml_enabled is True
        assert strategy.primary_strategy is not None

    def test_valid_config_ml_disabled(self):
        """有効な設定: ML無効モード（freqai.enabled=False + secondary=None）

        この設定は正常であり、エラーが発生しないこと
        """
        config = {
            "two_tier_strategy": {
                "primary": "atr_breakout",
                "secondary": None,  # secondaryが未指定
                "primary_params": {
                    "period": 14,
                    "multiplier": 0.5,
                    "execution_mode": "one_candle",
                    "fee": 0.00025,
                    "exit_periods": 24,
                    "pips": 0.5,
                },
            },
            "freqai": {
                "enabled": False,  # FreqAIが無効
            },
        }

        # エラーが発生しないことを確認
        strategy = TwoTierStrategy(config)

        assert strategy.is_ml_enabled is False
        assert strategy.primary_strategy is not None

    def test_invalid_config_missing_two_tier_strategy(self):
        """無効な設定: two_tier_strategyセクションが存在しない

        KeyErrorまたは適切なエラーが発生すべき
        """
        config = {
            "freqai": {
                "enabled": False,
            },
        }

        with pytest.raises((KeyError, ValueError)):
            TwoTierStrategy(config)

    def test_invalid_config_missing_freqai(self):
        """無効な設定: freqaiセクションが存在しない

        KeyErrorまたは適切なエラーが発生すべき
        """
        config = {
            "two_tier_strategy": {
                "primary": "atr_breakout",
                "secondary": None,
                "primary_params": {
                    "period": 14,
                    "multiplier": 0.5,
                    "execution_mode": "one_candle",
                    "fee": 0.00025,
                    "exit_periods": 24,
                    "pips": 0.5,
                },
            },
        }

        with pytest.raises((KeyError, ValueError)):
            TwoTierStrategy(config)

    def test_primary_strategy_loaded_correctly(self):
        """1次戦略が正しくロードされることを確認"""
        config = {
            "two_tier_strategy": {
                "primary": "atr_breakout",
                "secondary": None,
                "primary_params": {
                    "period": 14,
                    "multiplier": 0.5,
                    "execution_mode": "one_candle",
                    "fee": 0.00025,
                    "exit_periods": 24,
                    "pips": 0.5,
                },
            },
            "freqai": {
                "enabled": False,
            },
        }

        strategy = TwoTierStrategy(config)

        # 1次戦略が正しくロードされている
        assert strategy.primary_strategy is not None
        assert strategy.primary_strategy.period == 14
        assert strategy.primary_strategy.multiplier == 0.5
        assert strategy.primary_strategy.execution_mode == "one_candle"

    def test_config_with_default_freqai_enabled(self):
        """freqai.enabledがデフォルト値（False）の場合の挙動確認

        freqai.enabledが明示的にFalseでなくても、デフォルトでFalseと扱われるべき
        """
        config = {
            "two_tier_strategy": {
                "primary": "atr_breakout",
                "secondary": None,
                "primary_params": {
                    "period": 14,
                    "multiplier": 0.5,
                    "execution_mode": "one_candle",
                    "fee": 0.00025,
                    "exit_periods": 24,
                    "pips": 0.5,
                },
            },
            "freqai": {
                # "enabled"が明示的に指定されていない
            },
        }

        strategy = TwoTierStrategy(config)

        # デフォルトでML無効として扱われる
        assert strategy.is_ml_enabled is False

    def test_config_with_secondary_as_empty_string(self):
        """secondaryが空文字列の場合の挙動確認

        空文字列はNoneと同様に扱われるべき（secondary未指定）
        """
        config = {
            "two_tier_strategy": {
                "primary": "atr_breakout",
                "secondary": "",  # 空文字列
                "primary_params": {
                    "period": 14,
                    "multiplier": 0.5,
                    "execution_mode": "one_candle",
                    "fee": 0.00025,
                    "exit_periods": 24,
                    "pips": 0.5,
                },
            },
            "freqai": {
                "enabled": False,
            },
        }

        # 空文字列はTruthyではないので、secondary未指定と扱われるべき
        # エラーが発生しないことを確認
        strategy = TwoTierStrategy(config)
        assert strategy.is_ml_enabled is False

    def test_real_config_files_validation(self):
        """実際の設定ファイルの検証

        config_two_tier_ml_off.jsonとconfig_two_tier_ml_on.jsonが
        正しく検証をパスすることを確認
        """
        import json
        import os

        config_files = [
            ("config_two_tier_ml_off.json", False),  # (ファイル名, 期待されるis_ml_enabled)
            ("config_two_tier_ml_on.json", True),
        ]

        for config_file, expected_ml_enabled in config_files:
            if not os.path.exists(config_file):
                pytest.skip(f"{config_file} not found, skipping real config test")

            with open(config_file) as f:
                config = json.load(f)

            # エラーが発生しないことを確認
            strategy = TwoTierStrategy(config)

            # is_ml_enabledが期待値と一致
            assert strategy.is_ml_enabled == expected_ml_enabled, (
                f"{config_file}: is_ml_enabled should be {expected_ml_enabled}"
            )
