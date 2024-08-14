from pandas import DataFrame
from functools import reduce
import talib.abstract as ta
from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter, 
                                IStrategy, IntParameter)
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.persistence import Trade
from datetime import datetime, timedelta

class SolidStrategy(IStrategy):
    stoploss = -0.05  # 5% stoploss
    timeframe = '15m'  # 15-minute timeframe
    minimal_roi = {
        "0": 0.1,  # Default ROI, can be dynamically adjusted
    }

    # Hyperoptable parameters
    buy_rsi = IntParameter(20, 50, default=30, space='buy', optimize=True)
    sell_rsi = IntParameter(50, 80, default=70, space='sell', optimize=True)
    buy_ema_short = IntParameter(5, 20, default=10, space='buy', optimize=True)
    buy_ema_long = IntParameter(20, 50, default=30, space='buy', optimize=True)
    sell_ema_short = IntParameter(5, 20, default=10, space='sell', optimize=True)
    sell_ema_long = IntParameter(20, 50, default=30, space='sell', optimize=True)
    max_epa = CategoricalParameter([0, 1, 3, 5, 10], default=1, space="protection", optimize=True)
    use_stop_protection = BooleanParameter(default=True, space="protection", optimize=True)
    trailing_stop = DecimalParameter(0.02, 0.10, decimals=2, default=0.05, space='sell', optimize=True)

    # Time-based exit parameters
    max_trade_duration = IntParameter(30, 240, default=120, space="protection", optimize=True)

    @property
    def max_entry_position_adjustment(self):
        return self.max_epa.value

    @property
    def protections(self):
        """
        Define the protection mechanisms to prevent excessive losses or poor trading behavior.
        """
        prot = []

        # Cooldown period to prevent overtrading
        prot.append({
            "method": "CooldownPeriod",
            "stop_duration_candles": 2 * 24  # 2 days
        })

        # Stoploss guard to prevent further losses after a series of bad trades
        if self.use_stop_protection.value:
            prot.append({
                "method": "StoplossGuard",
                "lookback_period_candles": 24 * 3,  # Look back over the last 3 days
                "trade_limit": 3,  # Limit to 3 losing trades
                "stop_duration_candles": 24,  # Pause trading for 24 candles
                "only_per_pair": True  # Apply per pair
            })

        return prot

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Calculate and populate all necessary indicators used by the strategy.
        """
        dataframe['rsi'] = ta.RSI(dataframe)
        dataframe['ema_short'] = ta.EMA(dataframe, timeperiod=self.buy_ema_short.value)
        dataframe['ema_long'] = ta.EMA(dataframe, timeperiod=self.buy_ema_long.value)
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)  # Volatility measure

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Define the conditions under which the bot should enter a trade (buy signal).
        """
        conditions = []

        # Custom condition: RSI below threshold indicates oversold conditions
        conditions.append(dataframe['rsi'] < self.buy_rsi.value)

        # Trend condition: Short-term EMA above long-term EMA indicates upward trend
        conditions.append(dataframe['ema_short'] > dataframe['ema_long'])

        # Ensure there is trading volume
        conditions.append(dataframe['volume'] > 0)

        # Apply the conditions to set the 'enter_long' signal
        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'enter_long'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Define the conditions under which the bot should exit a trade (sell signal).
        """
        conditions = []

        # Custom condition: RSI above threshold indicates overbought conditions
        conditions.append(dataframe['rsi'] > self.sell_rsi.value)

        # Trend condition: Short-term EMA below long-term EMA indicates downward trend
        conditions.append(dataframe['ema_short'] < dataframe['ema_long'])

        # Ensure there is trading volume
        conditions.append(dataframe['volume'] > 0)

        # Apply the conditions to set the 'exit_long' signal
        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'exit_long'] = 1

        return dataframe

    def trailing_sell(self, pair: str, trade: Trade, current_rate: float, current_time: datetime, **kwargs) -> float:
        """
        Implements a trailing stop-loss mechanism.
        """
        if trade.open_rate is not None and self.trailing_stop.value is not None:
            # Calculate the trailing stop price
            stoploss_price = trade.open_rate * (1 - self.trailing_stop.value)
            if current_rate <= stoploss_price:
                return stoploss_price  # Suggest exit at stoploss price

        return None  # No sell signal if conditions are not met

    def check_time_based_exit(self, trade: Trade, current_time: datetime) -> bool:
        """
        Checks if the trade should be exited based on the maximum allowed duration.
        """
        max_duration = timedelta(minutes=self.max_trade_duration.value)
        trade_duration = current_time - trade.open_date_utc
        return trade_duration > max_duration

    def custom_sell(self, pair: str, trade: Trade, current_time: datetime, current_rate: float, current_profit: float, **kwargs) -> float:
        """
        Custom sell logic including time-based exits and partial profit-taking.
        Returns None if no sell signal, otherwise the sell price.
        """
        # Check for time-based exit
        if self.check_time_based_exit(trade, current_time):
            return current_rate

        # Implement partial profit-taking
        if current_profit > 0.05:  # Example condition for profit-taking
            return current_rate  # This will trigger a sell at the current rate

        return None  # No sell signal if conditions are not met

    def dynamic_roi(self, trade: Trade, current_profit: float) -> float:
        """
        Adjust ROI dynamically based on trade duration and current profit.
        Returns the adjusted ROI target.
        """
        trade_duration = (datetime.utcnow() - trade.open_date_utc).total_seconds() / 60
        if trade_duration < 60:
            return max(0.02, current_profit)  # Minimum ROI for short trades
        elif trade_duration < 120:
            return max(0.05, current_profit)  # Adjust ROI as trade ages
        else:
            return max(0.1, current_profit)  # Higher ROI for longer trades

        return self.minimal_roi["0"]
