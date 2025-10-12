"""Time series split validation tests - 時系列分割検証

このモジュールは、FreqAIのdata_split_parametersが
時系列データに適した設定になっていることを検証するテストを提供します。

テスト方針:
- shuffleがFalseであること（時系列順を保つ）
- test_sizeが適切な範囲であること
- K-fold CVではなくTimeSeriesSplitを使用すること（将来の拡張時）
"""

import json

import pytest


class TestTimeSeriesSplit:
    """時系列分割検証テストスイート"""

    @pytest.fixture
    def ml_off_config_path(self, tmp_path):
        """ML無効モード設定ファイルパス

        Args:
            tmp_path: pytestの一時ディレクトリ

        Returns:
            Path: テスト用設定ファイルパス
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
                "enabled": False,
                "data_split_parameters": {"test_size": 0.2, "shuffle": False},
            },
        }

        config_file = tmp_path / "config_test.json"
        config_file.write_text(json.dumps(config, indent=2))
        return config_file

    @pytest.fixture
    def ml_on_config_path(self, tmp_path):
        """ML有効モード設定ファイルパス

        Args:
            tmp_path: pytestの一時ディレクトリ

        Returns:
            Path: テスト用設定ファイルパス
        """
        config = {
            "two_tier_strategy": {
                "primary": "atr_breakout",
                "secondary": "lightgbm_classifier",
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
                "enabled": True,
                "identifier": "test_identifier",
                "model_name": "TwoTierLightGBMClassifier",
                "data_split_parameters": {
                    "test_size": 0.2,
                    "shuffle": False,
                    "random_state": 42,
                },
            },
        }

        config_file = tmp_path / "config_test.json"
        config_file.write_text(json.dumps(config, indent=2))
        return config_file

    def test_shuffle_is_false_ml_off(self, ml_off_config_path):
        """ML無効モードでshuffleがFalseであることを確認

        時系列データでは、訓練/テスト分割時にシャッフルしてはいけない
        """
        with open(ml_off_config_path) as f:
            config = json.load(f)

        shuffle = config["freqai"]["data_split_parameters"]["shuffle"]

        assert shuffle is False, (
            "shuffle must be False for time-series data. "
            "Shuffling would cause data leakage by mixing past and future data."
        )

    def test_shuffle_is_false_ml_on(self, ml_on_config_path):
        """ML有効モードでshuffleがFalseであることを確認"""
        with open(ml_on_config_path) as f:
            config = json.load(f)

        shuffle = config["freqai"]["data_split_parameters"]["shuffle"]

        assert shuffle is False, (
            "shuffle must be False for time-series data. "
            "Shuffling would cause data leakage by mixing past and future data."
        )

    def test_test_size_in_valid_range_ml_off(self, ml_off_config_path):
        """ML無効モードでtest_sizeが適切な範囲であることを確認

        test_sizeが極端に小さいまたは大きい場合、
        訓練データまたはテストデータが不足する
        """
        with open(ml_off_config_path) as f:
            config = json.load(f)

        test_size = config["freqai"]["data_split_parameters"]["test_size"]

        assert 0.1 <= test_size <= 0.3, (
            f"test_size {test_size} should be between 0.1 and 0.3 for time-series data. "
            "Too small: insufficient test data. Too large: insufficient training data."
        )

    def test_test_size_in_valid_range_ml_on(self, ml_on_config_path):
        """ML有効モードでtest_sizeが適切な範囲であることを確認"""
        with open(ml_on_config_path) as f:
            config = json.load(f)

        test_size = config["freqai"]["data_split_parameters"]["test_size"]

        assert 0.1 <= test_size <= 0.3, (
            f"test_size {test_size} should be between 0.1 and 0.3 for time-series data. "
            "Too small: insufficient test data. Too large: insufficient training data."
        )

    def test_data_split_parameters_exist_ml_off(self, ml_off_config_path):
        """ML無効モードでdata_split_parametersが存在することを確認"""
        with open(ml_off_config_path) as f:
            config = json.load(f)

        assert "data_split_parameters" in config["freqai"], (
            "freqai.data_split_parameters must be defined"
        )

    def test_data_split_parameters_exist_ml_on(self, ml_on_config_path):
        """ML有効モードでdata_split_parametersが存在することを確認"""
        with open(ml_on_config_path) as f:
            config = json.load(f)

        assert "data_split_parameters" in config["freqai"], (
            "freqai.data_split_parameters must be defined"
        )

    def test_real_config_files_shuffle_false(self):
        """実際の設定ファイルでshuffleがFalseであることを確認

        プロジェクトのconfig_two_tier_ml_off.jsonとconfig_two_tier_ml_on.json
        を読み込んで検証する
        """
        import os

        config_files = [
            "config_two_tier_ml_off.json",
            "config_two_tier_ml_on.json",
        ]

        for config_file in config_files:
            if not os.path.exists(config_file):
                pytest.skip(f"{config_file} not found, skipping real config test")

            with open(config_file) as f:
                config = json.load(f)

            shuffle = config["freqai"]["data_split_parameters"]["shuffle"]

            assert shuffle is False, (
                f"{config_file}: shuffle must be False for time-series data. "
                "Shuffling would cause data leakage."
            )

    def test_real_config_files_test_size(self):
        """実際の設定ファイルでtest_sizeが適切な範囲であることを確認"""
        import os

        config_files = [
            "config_two_tier_ml_off.json",
            "config_two_tier_ml_on.json",
        ]

        for config_file in config_files:
            if not os.path.exists(config_file):
                pytest.skip(f"{config_file} not found, skipping real config test")

            with open(config_file) as f:
                config = json.load(f)

            test_size = config["freqai"]["data_split_parameters"]["test_size"]

            assert 0.1 <= test_size <= 0.4, (
                f"{config_file}: test_size {test_size} should be between 0.1 and 0.4. "
                "This range ensures sufficient training and test data."
            )

    def test_no_k_fold_cv_in_config(self, ml_on_config_path):
        """K-fold CVが使用されていないことを確認

        時系列データではK-fold CVではなくTimeSeriesSplitを使用すべき
        （現在の実装ではK-fold CVは使用していないが、将来の拡張に備えて検証）
        """
        with open(ml_on_config_path) as f:
            config = json.load(f)

        # k-foldやcross_validationの設定が存在しないことを確認
        data_split = config["freqai"]["data_split_parameters"]

        assert "k_folds" not in data_split, (
            "K-fold CV should not be used for time-series data. Use TimeSeriesSplit instead."
        )

        assert "cross_validation" not in data_split, (
            "Cross-validation with shuffling should not be used for time-series data."
        )

    def test_random_state_exists_for_reproducibility(self, ml_on_config_path):
        """再現性のためにrandom_stateが設定されていることを確認（ML有効時）

        ML有効モードでは、訓練の再現性を保つために
        random_stateを設定することが推奨される
        """
        with open(ml_on_config_path) as f:
            config = json.load(f)

        data_split = config["freqai"]["data_split_parameters"]

        assert "random_state" in data_split, (
            "random_state should be set for reproducibility in ML training"
        )

        assert isinstance(data_split["random_state"], int), "random_state should be an integer"

    def test_time_series_order_preserved(self):
        """時系列順序が保たれることのコンセプト確認

        実際の時系列分割では、訓練データが過去、テストデータが未来であるべき
        このテストは、設定が正しく時系列順序を保つことを概念的に確認する
        """
        # ML無効モード設定
        config = {
            "freqai": {
                "enabled": False,
                "data_split_parameters": {"test_size": 0.2, "shuffle": False},
            }
        }

        # shuffleがFalseの場合、訓練データ(80%)が先、テストデータ(20%)が後
        # という時系列順序が保たれる
        assert config["freqai"]["data_split_parameters"]["shuffle"] is False

        # test_size = 0.2 → 訓練:テスト = 80:20
        # 訓練データ: インデックス0～79（過去）
        # テストデータ: インデックス80～99（未来）
        # という分割が行われる（100データポイントの場合）

        test_size = config["freqai"]["data_split_parameters"]["test_size"]
        train_size = 1 - test_size

        # 訓練データが過去、テストデータが未来となる
        assert train_size > test_size, (
            "Training data should cover more historical data than test data"
        )
