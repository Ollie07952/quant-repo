def maxDraw(prices):
    import numpy as np         
    """
             
    Calculates the maximum drawdown within an array(s) of prices, in both absolute and percentage terms
    --
    Parameters:
             
    prices : array(s); array(s) of prices
    --
    
    Returns:
    
    max_draws : array of tuples of floats; value of the maximum drawdown and corresponding percent return of this drawdown for each set of prices
    
    """
    
    highest = np.maximum.accumulate(prices, axis = 1)
    biggest_abs = np.subtract(highest, prices)
    biggest_perc = np.divide(biggest_abs,highest)
    abs_dds = biggest_abs.max(axis = 1)
    perc_dds = biggest_perc.max(axis = 1)
    max_draws = np.stack((abs_dds,perc_dds), axis = 1)
    return max_draws
