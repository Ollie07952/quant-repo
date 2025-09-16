def riskMeasures(prices, q = 95):
    """
    Calculates expected final cumulative log return, and daily q% Value-at-Risk
    --
    :arg prices: array; array(s) of prices
    :arg q: int or float; Value-at-Risk percentage (VaR), 95% by default
    --
    :returns ev: float; mean of all final cumulative returns
    :returns var: float; Value-at-Risk based off q parameter

    """
    import numpy as np

    logs = np.atleast_2d([[np.log(x/price_sets[max(i-1,0)]) for i,x in enumerate(price_sets)] for price_sets in prices])
    ev = logs.sum(axis = 1).mean()
    var = np.percentile(logs, q = (100-q))
    return ev, var

