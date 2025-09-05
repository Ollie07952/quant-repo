def logReturns(prices):
    """
    Calculates daily and cumulative log returns for each set of given prices
    --
    :arg prices: array; array(s) of prices
    --
    :returns logs: array; daily log returns for each set of prices
    :returns cumulative_logs: array; cumulative sum of daily log returns for each set of prices
    """
    import numpy as np

    logs = np.atleast_2d([[np.log(x/price_sets[max(i-1,0)]) for i,x in enumerate(price_sets)] for price_sets in prices])
    cumulative_logs = np.array(np.cumsum(logs,axis = 1))
    return logs, cumulative_logs
