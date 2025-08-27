def maxDraw(prices):
    import numpy as np         
    """
             
    Calculates the maximum drawdown within an array(s) of prices
    --
    Parameters:
             
    prices : array(s); array(s) of prices
    --
    
    Returns:
    
    max_draws : array of tuples of floats; value of the maximum drawdown and corresponding percent return of this drawdown for each set of prices
    
    """
    
    abs_dds, percent_dds = [],[]
    
    for price_set in prices:
        highest, biggest_abs, biggest_perc = price_set[0], 0, 0
        for price in price_set[1:]:
            highest = max(highest, price)
            biggest_abs = max(biggest_abs, highest-price)
            biggest_perc = max(biggest_perc, (highest-price)/highest)
        abs_dds.append(biggest_abs)
        percent_dds.append(biggest_perc)
    max_draws = np.array([(abs_dd,perc_dd) for abs_dd,perc_dd in zip(abs_dds,percent_dds)])
    return max_draws
