def vol(tickers,period,k = 30):
    """
    Calculates rolling and stationary annualised volatility for each given ticker
    --
    :arg tickers: str; ticker list as a single str of the form "ABC DEFG XYZ ..."
    :arg period: str; the period of returns to analyse; one of: [1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max]
    :arg k: int; the history of the rolling volatility, in days. i.e. size of the rolling volatility window.
                 NOTE: k MUST NOT BE GREATER IN LENGTH THAN CHOSEN PERIOD
    --
    :returns roll: array; rolling annualised volatility with history k for each ticker
    :returns ann: array; annualised volatility of each ticker over the given period
    """
    import numpy as np
    from logReturns import logReturns

    logs = logReturns(tickers, period)
    roll = logs.rolling(window = k).std()*np.sqrt(252)
    ann = logs.std(ddof = 1, axis = 0)*np.sqrt(252)
    ann.name = "Annualised Vol"
    ann.index.name = "Ticker"
    return roll, ann
