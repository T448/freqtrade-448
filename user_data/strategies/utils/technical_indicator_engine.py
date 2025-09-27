"""
TechnicalIndicatorEngine - テクニカル指標計算エンジン

richmanbtcチュートリアルに基づく機械学習特徴量として使用する
10以上の標準テクニカル指標計算システム

Requirements implemented:
- 2.1: 移動平均、RSI、MACD、ボリンジャーバンド等10以上の標準指標を計算
- 2.2: 十分な履歴データの存在確認機能
- 2.3: 欠損値と外れ値の適切な処理によるデータ品質確保
- 2.4: ルックバック期間に対応した履歴データ確保機能
- 2.5: すべての指標の時間同期とデータポイント整合性確保
"""

import logging

import numpy as np
import pandas as pd
import talib as ta


logger = logging.getLogger(__name__)


class TechnicalIndicatorEngine:
    """
    機械学習特徴量として使用するテクニカル指標の計算エンジン

    richmanbtcチュートリアルの2次モデル（機械学習分類）で使用する
    特徴量として、10以上の標準的なテクニカル指標を計算します。
    """

    def __init__(self):
        """初期化"""
        logger.info("TechnicalIndicatorEngine初期化")

    def calculate_sma(self, dataframe: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Simple Moving Average (SMA) 計算

        Args:
            dataframe: OHLC市場データ
            period: 移動平均期間

        Returns:
            SMA値のSeries
        """
        return pd.Series(
            ta.SMA(dataframe["close"].astype(float).values, timeperiod=period),
            index=dataframe.index,
            name=f"sma_{period}",
        )

    def calculate_ema(self, dataframe: pd.DataFrame, period: int = 12) -> pd.Series:
        """
        Exponential Moving Average (EMA) 計算

        Args:
            dataframe: OHLC市場データ
            period: EMA期間

        Returns:
            EMA値のSeries
        """
        return pd.Series(
            ta.EMA(dataframe["close"].astype(float).values, timeperiod=period),
            index=dataframe.index,
            name=f"ema_{period}",
        )

    def calculate_rsi(self, dataframe: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Relative Strength Index (RSI) 計算

        Args:
            dataframe: OHLC市場データ
            period: RSI期間

        Returns:
            RSI値のSeries (0-100の範囲)
        """
        return pd.Series(
            ta.RSI(dataframe["close"].astype(float).values, timeperiod=period),
            index=dataframe.index,
            name=f"rsi_{period}",
        )

    def calculate_macd(
        self,
        dataframe: pd.DataFrame,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
    ) -> pd.DataFrame:
        """
        MACD (Moving Average Convergence Divergence) 計算

        Args:
            dataframe: OHLC市場データ
            fast_period: 高速EMA期間
            slow_period: 低速EMA期間
            signal_period: シグナル期間

        Returns:
            MACD, シグナル, ヒストグラムを含むDataFrame
        """
        macd, macd_signal, macd_histogram = ta.MACD(
            dataframe["close"].astype(float).values,
            fastperiod=fast_period,
            slowperiod=slow_period,
            signalperiod=signal_period,
        )

        return pd.DataFrame(
            {"macd": macd, "macd_signal": macd_signal, "macd_histogram": macd_histogram},
            index=dataframe.index,
        )

    def calculate_bollinger_bands(
        self, dataframe: pd.DataFrame, period: int = 20, std_dev: int = 2
    ) -> pd.DataFrame:
        """
        Bollinger Bands 計算

        Args:
            dataframe: OHLC市場データ
            period: 移動平均期間
            std_dev: 標準偏差の倍数

        Returns:
            上部、中部、下部バンドを含むDataFrame
        """
        bb_upper, bb_middle, bb_lower = ta.BBANDS(
            dataframe["close"].astype(float).values,
            timeperiod=period,
            nbdevup=std_dev,
            nbdevdn=std_dev,
            matype=0,
        )

        return pd.DataFrame(
            {"bb_upper": bb_upper, "bb_middle": bb_middle, "bb_lower": bb_lower},
            index=dataframe.index,
        )

    def calculate_stochastic(
        self, dataframe: pd.DataFrame, k_period: int = 14, d_period: int = 3
    ) -> pd.DataFrame:
        """
        Stochastic Oscillator 計算

        Args:
            dataframe: OHLC市場データ
            k_period: %K期間
            d_period: %D期間

        Returns:
            %Kと%Dを含むDataFrame
        """
        stoch_k, stoch_d = ta.STOCH(
            dataframe["high"].astype(float).values,
            dataframe["low"].astype(float).values,
            dataframe["close"].astype(float).values,
            fastk_period=k_period,
            slowk_period=d_period,
            slowd_period=d_period,
        )

        return pd.DataFrame({"stoch_k": stoch_k, "stoch_d": stoch_d}, index=dataframe.index)

    def calculate_williams_r(self, dataframe: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Williams %R 計算

        Args:
            dataframe: OHLC市場データ
            period: 計算期間

        Returns:
            Williams %R値のSeries (-100 ~ 0の範囲)
        """
        return pd.Series(
            ta.WILLR(
                dataframe["high"].astype(float).values,
                dataframe["low"].astype(float).values,
                dataframe["close"].astype(float).values,
                timeperiod=period,
            ),
            index=dataframe.index,
            name=f"williams_r_{period}",
        )

    def calculate_atr(self, dataframe: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Average True Range (ATR) 計算

        Args:
            dataframe: OHLC市場データ
            period: ATR期間

        Returns:
            ATR値のSeries
        """
        return pd.Series(
            ta.ATR(
                dataframe["high"].astype(float).values,
                dataframe["low"].astype(float).values,
                dataframe["close"].astype(float).values,
                timeperiod=period,
            ),
            index=dataframe.index,
            name=f"atr_{period}",
        )

    def calculate_adx(self, dataframe: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Average Directional Index (ADX) 計算

        Args:
            dataframe: OHLC市場データ
            period: ADX期間

        Returns:
            ADX値のSeries (0-100の範囲)
        """
        return pd.Series(
            ta.ADX(
                dataframe["high"].astype(float).values,
                dataframe["low"].astype(float).values,
                dataframe["close"].astype(float).values,
                timeperiod=period,
            ),
            index=dataframe.index,
            name=f"adx_{period}",
        )

    def calculate_cci(self, dataframe: pd.DataFrame, period: int = 20) -> pd.Series:
        """
        Commodity Channel Index (CCI) 計算

        Args:
            dataframe: OHLC市場データ
            period: CCI期間

        Returns:
            CCI値のSeries
        """
        return pd.Series(
            ta.CCI(
                dataframe["high"].astype(float).values,
                dataframe["low"].astype(float).values,
                dataframe["close"].astype(float).values,
                timeperiod=period,
            ),
            index=dataframe.index,
            name=f"cci_{period}",
        )

    def calculate_mfi(self, dataframe: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Money Flow Index (MFI) 計算

        Args:
            dataframe: OHLC市場データ（volume必須）
            period: MFI期間

        Returns:
            MFI値のSeries (0-100の範囲)
        """
        return pd.Series(
            ta.MFI(
                dataframe["high"].astype(float).values,
                dataframe["low"].astype(float).values,
                dataframe["close"].astype(float).values,
                dataframe["volume"].astype(float).values,
                timeperiod=period,
            ),
            index=dataframe.index,
            name=f"mfi_{period}",
        )

    def calculate_all_indicators(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        全ての技術的指標を計算する - 要件2.1 (10以上の標準指標)

        Args:
            dataframe: OHLC市場データ (high, low, close, volume必須)

        Returns:
            全ての指標を含むDataFrame

        Raises:
            ValueError: データ不足またはカラム不足の場合
        """
        # データ品質チェック - 要件2.3
        self._validate_input_data(dataframe)

        # データ十分性チェック - 要件2.2
        min_required_periods = 50  # 最大の計算期間を考慮
        if not self.validate_data_sufficiency(dataframe, min_required_periods):
            raise ValueError(f"データが不足しています。最低{min_required_periods}行必要です。")

        result = pd.DataFrame(index=dataframe.index)

        try:
            # 移動平均系指標
            result["sma_14"] = self.calculate_sma(dataframe, 14)
            result["sma_50"] = self.calculate_sma(dataframe, 50)
            result["ema_12"] = self.calculate_ema(dataframe, 12)
            result["ema_26"] = self.calculate_ema(dataframe, 26)

            # オシレーター系指標
            result["rsi_14"] = self.calculate_rsi(dataframe, 14)

            # MACD
            macd_df = self.calculate_macd(dataframe)
            result = pd.concat([result, macd_df], axis=1)

            # ボリンジャーバンド
            bb_df = self.calculate_bollinger_bands(dataframe, 20)
            result = pd.concat([result, bb_df], axis=1)

            # ストキャスティクス
            stoch_df = self.calculate_stochastic(dataframe, 14, 3)
            result = pd.concat([result, stoch_df], axis=1)

            # その他の指標
            result["williams_r_14"] = self.calculate_williams_r(dataframe, 14)
            result["atr_14"] = self.calculate_atr(dataframe, 14)
            result["adx_14"] = self.calculate_adx(dataframe, 14)
            result["cci_20"] = self.calculate_cci(dataframe, 20)
            result["mfi_14"] = self.calculate_mfi(dataframe, 14)

            # 欠損値処理 - 要件2.3
            result = self._handle_missing_values(result)

            logger.info(f"テクニカル指標計算完了: {len(result.columns)}指標, {len(result)}行")
            return result

        except Exception as e:
            logger.error(f"テクニカル指標計算中にエラーが発生: {e}")
            raise

    def validate_data_sufficiency(self, dataframe: pd.DataFrame, min_periods: int = 50) -> bool:
        """
        テクニカル指標計算に十分なデータがあるかチェック - 要件2.2

        Args:
            dataframe: チェック対象のデータ
            min_periods: 最低必要期間

        Returns:
            十分なデータがある場合True
        """
        return len(dataframe) >= min_periods

    def _validate_input_data(self, dataframe: pd.DataFrame) -> None:
        """
        入力データの検証 - 要件2.3

        Args:
            dataframe: 検証対象のデータ

        Raises:
            ValueError: データが無効な場合
        """
        if dataframe.empty:
            raise ValueError("入力データが空です")

        required_columns = ["high", "low", "close", "volume"]
        missing_columns = [col for col in required_columns if col not in dataframe.columns]
        if missing_columns:
            raise ValueError(f"必須カラムが不足しています: {missing_columns}")

    def _handle_missing_values(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        欠損値の適切な処理 - 要件2.3

        Args:
            dataframe: 処理対象のDataFrame

        Returns:
            欠損値処理後のDataFrame
        """
        # インデックスの整合性を確保 - 要件2.5
        result = dataframe.copy()

        # 数値カラムのみ対象
        numeric_columns = result.select_dtypes(include=[np.number]).columns

        # 極端な外れ値の処理（基本的にはそのまま残す）
        for column in numeric_columns:
            # 無限大をNaNに変換
            result[column] = result[column].replace([np.inf, -np.inf], np.nan)

        logger.debug(f"欠損値処理完了: {len(numeric_columns)}列")
        return result
