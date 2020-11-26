import pandas as pd

def macd(stock_frame: pd.DataFrame, fast_period: int = 12, slow_period: int = 26) -> pd.DataFrame:
    """Calculates the Moving Average Convergence Divergence (MACD).

    ### Arguments:
    -------
    stock_frame: pd.DataFrame
        A multi-index Pandas DataFrame, that MUST INCLUDE the
        `open`, `high`, `low`, and `close` price. That way
        indicators may be calculated effectively.

    fast_period : int
        The number of periods to use when calculating 
        the fast moving MACD. (default is 12 periods.)

    slow_period : int
        The number of periods to use when calculating 
        the slow moving MACD. (default is 26 periods.)

    ### Returns:
    -------
    pd.DataFrame
        A Pandas data frame with the MACD indicator
        included.
    """

    # Calculate the Fast Moving MACD.
    stock_frame['macd_fast'] = stock_frame['close'].transform(
        lambda x: x.ewm(span = fast_period, min_periods = fast_period).mean()
    )

    # Calculate the Slow Moving MACD.
    stock_frame['macd_slow'] = stock_frame['close'].transform(
        lambda x: x.ewm(span = slow_period, min_periods = slow_period).mean()
    )

    # Calculate the difference between the fast and the slow.
    stock_frame['macd_diff'] = stock_frame['macd_fast'] - stock_frame['macd_slow']

    # Calculate the Exponential moving average of the fast.
    stock_frame['macd'] = stock_frame['macd_diff'].transform(
        lambda x: x.ewm(span = 9, min_periods = 8).mean()
    )

    return stock_frame 

def ema(stock_frame: pd.DataFrame, period: int, alpha: float = 0.0, column_name: str = 'ema') -> pd.DataFrame:
    """Calculates the Exponential Moving Average (EMA).

    ### Arguments:
    -------
    stock_frame: pd.DataFrame
        A multi-index Pandas DataFrame, that MUST INCLUDE the
        `open`, `high`, `low`, and `close` price. That way
        indicators may be calculated effectively.

    period : int
        The number of periods to use when calculating 
        the EMA.

    alpha : int
        The alpha weight used in the calculation. 
        (default is 0.0.)
    
    column_name: str
        If you plan to have multiple columns that contain
        the same indicator, then set the `column_name`
        argument so you can easily identify which column
        contains which indicator. For example, `ema`,
        `ema_20`, `ema_50`. (default is `ema`.)

    ### Returns:
    -------
    pd.DataFrame
        A Pandas data frame with the EMA indicator
        included.
    """

    # Group the dataframe by the symbols.
    stock_groups = stock_frame.groupby(
        by='symbol',
        as_index=False,
        sort=True
    )

    # Add the EMA
    stock_frame[column_name] = stock_groups['close'].transform(
        lambda x: x.ewm(span=period).mean()
    )

    return stock_frame