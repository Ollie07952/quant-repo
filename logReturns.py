def logReturns(tickers,period):
    """
    Calculates daily log returns for each given ticker over the given period.
    --
    :arg tickers: str; ticker list as a single str of the form "ABC DEFG XYZ ..."
    :arg period: str; the period of returns to analyse; one of: [1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max]
    --
    :returns logs: dataframe; daily log returns for each ticker over the given period
    """
    import yfinance as yf
    import numpy as np

    closes = yf.download(tickers, period = period, auto_adjust = True, progress = False)["Close"]
    closes.columns.name = "Close"
    logs = np.log(closes/closes.shift(1))
    logs.columns.name = "Log Returns"
    return logs
