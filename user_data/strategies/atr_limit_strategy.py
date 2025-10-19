from datetime import datetime, timedelta

import talib
from pandas import DataFrame

from freqtrade.persistence.trade_model import Order, Trade
from freqtrade.strategy import IStrategy
from freqtrade.strategy.parameters import DecimalParameter, IntParameter


class AtrLimitStrategy2(IStrategy):
    timeframe = "15m"
    can_short = True  # 両建て許可
    position_adjustment_enable = True  # ポジション積み増し許可
    use_custom_stoploss = False

    # リスク管理
    minimal_roi = {"0": 0.03}
    stoploss = -0.02
    trailing_stop = False

    # 両建て・積み増し許可設定
    allow_entry_in_open_trades = True

    # 指値設定
    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }

    # 最適化パラメータ
    # Long用パラメータ
    buy_atr_timeperiod = IntParameter(7, 20, default=14, space="buy")
    buy_atr_entry_point = DecimalParameter(0.01, 1, decimals=2, default=0.5, space="buy")

    # Short用パラメータ
    sell_atr_timeperiod = IntParameter(7, 20, default=14, space="sell")
    sell_atr_entry_point = DecimalParameter(0.01, 1, decimals=2, default=0.5, space="sell")

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        特徴量計算
        指値価格の算出に使用する

        Args:
            dataframe (DataFrame): ohlcvを含むDataFrame
            metadata (dict): _description_

        Returns:
            DataFrame: ohlcv、特徴量を含むDataFrame
        """
        dataframe["atr_buy"] = talib.ATR(
            dataframe["high"],
            dataframe["low"],
            dataframe["close"],
            timeperiod=self.buy_atr_timeperiod.value,
        )
        dataframe["atr_sell"] = talib.ATR(
            dataframe["high"],
            dataframe["low"],
            dataframe["close"],
            timeperiod=self.sell_atr_timeperiod.value,
        )

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        売買シグナルを生成する
        指値注文の場合も必要で、指値注文が刺さることを条件とする

        Args:
            dataframe (DataFrame): _description_
            metadata (dict): _description_

        Returns:
            DataFrame: _description_
        """
        atr_buy = dataframe["atr_buy"]
        atr_sell = dataframe["atr_sell"]

        dataframe.loc[
            (dataframe["low"] < dataframe["close"] - atr_buy * self.buy_atr_entry_point.value),
            ["enter_long", "enter_tag"],
        ] = (1, "atr_long")

        dataframe.loc[
            (dataframe["high"] > dataframe["close"] + atr_sell * self.sell_atr_entry_point.value),
            ["enter_short", "enter_tag"],
        ] = (1, "atr_short")

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        決済シグナルを生成する
        指値注文による決済の場合も必要で、指値注文が刺さることを条件とする

        Args:
            dataframe (DataFrame): _description_
            metadata (dict): _description_

        Returns:
            DataFrame: _description_
        """
        atr_buy = dataframe["atr_buy"]
        atr_sell = dataframe["atr_sell"]

        dataframe.loc[
            (dataframe["low"] < dataframe["close"] - atr_buy * self.buy_atr_entry_point.value),
            ["exit_long", "exit_tag"],
        ] = (1, "atr_exit_long")

        dataframe.loc[
            (dataframe["high"] > dataframe["close"] + atr_sell * self.sell_atr_entry_point.value),
            ["exit_short", "exit_tag"],
        ] = (1, "atr_exit_short")

        return dataframe

    def custom_entry_price(
        self,
        pair: str,
        trade: Trade | None,
        current_time: datetime,
        proposed_rate: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float:
        """
        指値価格を算出する

        Args:
            pair (str): _description_
            trade (Trade | None): _description_
            current_time (datetime): _description_
            proposed_rate (float): _description_
            entry_tag (str | None): _description_
            side (str): _description_

        Returns:
            float: _description_
        """

        return self.atr_limit_price(pair=pair, proposed_rate=proposed_rate, side=side)

    def custom_exit_price(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        proposed_rate: float,
        current_profit: float,
        exit_tag: str | None,
        **kwargs,
    ):
        """
        指値価格を算出する

        Args:
            pair (str): _description_
            trade (Trade): _description_
            current_time (datetime): _description_
            proposed_rate (float): _description_
            current_profit (float): _description_
            exit_tag (str | None): _description_

        Returns:
            _type_: _description_
        """
        side = "short" if trade.is_short else "long"

        return self.atr_limit_price(pair=pair, proposed_rate=proposed_rate, side=side)

    def check_entry_timeout(
        self, pair: str, trade: Trade, order: Order, current_time: datetime, **kwargs
    ) -> bool:
        """
        指値注文のタイムアウト

        Args:
            pair (str): _description_
            trade (Trade): _description_
            order (Order): _description_
            current_time (datetime): _description_

        Returns:
            bool: _description_
        """
        return self.check_timeout(order, current_time)

    def check_exit_timeout(
        self, pair: str, trade: Trade, order: Order, current_time: datetime, **kwargs
    ):
        """
        指値決済注文のタイムアウト

        Args:
            pair (str): _description_
            trade (Trade): _description_
            order (Order): _description_
            current_time (datetime): _description_

        Returns:
            _type_: _description_
        """
        return self.check_timeout(order, current_time)

    def atr_limit_price(
        self,
        pair: str,
        proposed_rate: float,
        side: str,
    ) -> float:
        dataframe, _last_updated = self.dp.get_analyzed_dataframe(
            pair=pair, timeframe=self.timeframe
        )
        """
        指値価格の算出
        long,shortの分岐は内部で行う

        Returns:
            _type_: _description_
        """

        atr_buy = dataframe["atr_buy"]
        atr_sell = dataframe["atr_sell"]

        if side == "long":
            new_entryprice = dataframe["close"] - atr_buy * self.buy_atr_entry_point.value
        elif side == "short":
            new_entryprice = dataframe["close"] + atr_sell * self.sell_atr_entry_point.value
        else:
            new_entryprice = proposed_rate

        return new_entryprice

    def check_timeout(self, order: Order, current_time: datetime) -> bool:
        """
        次のローソク足で注文が刺さっていない場合タイムアウトでキャンセルする

        Args:
            order (Order): _description_
            current_time (datetime): _description_

        Returns:
            bool: _description_
        """

        # timeframeを設定ファイルやstrategyから取得
        tf_str = self.timeframe  # 例: '5m', '1h', '1d'

        # timeframeをtimedeltaに変換
        unit = tf_str[-1]
        value = int(tf_str[:-1])
        if unit == "m":
            tf_delta = timedelta(minutes=value)
        elif unit == "h":
            tf_delta = timedelta(hours=value)
        elif unit == "d":
            tf_delta = timedelta(days=value)
        else:
            # 想定外のtimeframeの場合は何もしない
            return False

        # 現在時刻と発注時刻の差分がtimeframe以上ならキャンセル
        if current_time - order.order_date >= tf_delta:
            return True

        return False
