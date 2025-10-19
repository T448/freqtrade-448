from datetime import datetime, timedelta

import talib
from pandas import DataFrame

from freqtrade.persistence.trade_model import Order, Trade
from freqtrade.strategy import IStrategy


ATR_TIME_PERIOD = 14
ATR_ENTRY_POINT = 0.5


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

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["atr"] = talib.ATR(
            dataframe["high"], dataframe["low"], dataframe["close"], timeperiod=ATR_TIME_PERIOD
        )

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        atr = dataframe["atr"]

        dataframe.loc[
            (dataframe["low"] < dataframe["close"] - atr * ATR_ENTRY_POINT),
            ["enter_long", "enter_tag"],
        ] = (1, "atr_long")

        dataframe.loc[
            (dataframe["high"] > dataframe["close"] + atr * ATR_ENTRY_POINT),
            ["enter_short", "enter_tag"],
        ] = (1, "atr_short")

        print(dataframe)

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        atr = dataframe["atr"]

        dataframe.loc[
            (dataframe["low"] < dataframe["close"] - atr * ATR_ENTRY_POINT),
            ["exit_long", "exit_tag"],
        ] = (1, "atr_exit_long")

        dataframe.loc[
            (dataframe["high"] > dataframe["close"] + atr * ATR_ENTRY_POINT),
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
        side = "short" if trade.is_short else "long"

        return self.atr_limit_price(pair=pair, proposed_rate=proposed_rate, side=side)

    def check_entry_timeout(
        self, pair: str, trade: Trade, order: Order, current_time: datetime, **kwargs
    ) -> bool:
        return self.check_timeout(order, current_time)

    def check_exit_timeout(
        self, pair: str, trade: Trade, order: Order, current_time: datetime, **kwargs
    ):
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

        atr = dataframe["atr"]

        if side == "long":
            new_entryprice = dataframe["close"] - atr * ATR_ENTRY_POINT
        elif side == "short":
            new_entryprice = dataframe["close"] + atr * ATR_ENTRY_POINT
        else:
            new_entryprice = proposed_rate

        return new_entryprice

    def check_timeout(self, order: Order, current_time: datetime) -> bool:
        """
        指値注文を次の足でキャンセルする例。
        たとえば5分足なら、発注から5分後に未約定ならキャンセル。
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
