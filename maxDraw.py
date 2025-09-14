def maxDraw(prices):
    """
    Calculates the maximum drawdown within an array(s) of prices
    --
    :arg prices: array(s); array(s) of prices
    --
    :returns max_draws: array of tuples of floats; value of the maximum drawdown and corresponding percent return of this drawdown for each set of prices
    """
    import numpy as np

    bigs_abs, bigs_draws = [],[]
    
    for price_set in prices:
        highest = price_set[0]
        big_abs, big_draw = 0,0
        for price in price_set[1:]:
            if price > highest:
                highest = price
            big_abs = max(big_abs, highest-price)
            big_draw = max(big_draw, (highest-price)/highest)
        bigs_abs.append(big_abs)
        bigs_draws.append(big_draw)
    max_draws = np.array([(abs_dd,percent_dd) for abs_dd,percent_dd in zip(bigs_abs,bigs_draws)])
    return max_draws