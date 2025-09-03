def riskMeasures(prices, q = 95):
    import numpy as np
    
    """
    
    Calculates expected final cumulative log return, and daily q% Value-at-Risk
    --
    
    Parameters:
    
    prices : array; array(s) of prices
    q : int or float; Value-at-Risk percentage (VaR), 95% by default
    --
    
    Returns: 
    
    ev : mean of all final cumulative returns
    var : Value-at-Risk based off q parameter

    """

    logs = np.atleast_2d([[np.log(x/price_sets[max(i-1,0)]) for i,x in enumerate(price_sets)] for price_sets in prices])
    ev = logs.sum(axis = 1).mean()
    var = np.percentile(logs, q = (100-q))

    return ev, var

