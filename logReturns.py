def logReturns(prices):
    import numpy as np
    
    """
    
    Calculates daily and cumulative log returns for each set of given prices
    --
    
    Parameters:
    
    prices : array; array(s) of prices
    --
    
    Returns: 
    
    logs : array of daily log returns for each set of prices
    cumulative_logs : cumulative sum of daily log returns for each set of prices

    """

    logs = np.atleast_2d([[np.log(x/price_sets[max(i-1,0)]) for i,x in enumerate(price_sets)] for price_sets in prices])
    cumulative_logs = np.array(np.cumsum(logs,axis = 1))

    return logs, cumulative_logs
