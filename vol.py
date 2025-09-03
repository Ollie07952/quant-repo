def vol(prices,k = 30):
    import numpy as np
    
    """
    
    Calculates daily and annualised volatility for each set of prices
    --
    
    Parameters:
    
    prices : array; array(s) of prices
    k : int; the history of the rolling volatility, in days
    --
    
    Returns: 
    
    roll_vol : array of rolling volatility with history k for each set of prices
    ann_vol : annualised volatility of each set of prices

    """

    if k > np.shape(prices)[1]:
        raise ValueError("volatility history cannot be larger than number of days")
    logs = np.atleast_2d([[np.log(x/price_sets[max(i-1,0)]) for i,x in enumerate(price_sets)] for price_sets in prices])
    roll_vol = np.concatenate((np.full(shape = (np.shape(logs)[0],k-1),fill_value = np.nan),np.array([[np.std(log_sets[i-k:i], ddof = 1) for i in range(k, len(log_sets))] for log_sets in logs])),axis = 1)
    ann_vol = np.array([np.std(log_sets, ddof = 1)*np.sqrt(252) for log_sets in logs])

    return roll_vol, ann_vol
