def bsCall(S, X, r, t, sigma):
    """
    Calculates theoretical call values using the Black-Scholes model.
    --
    :arg S: float; current underlying price
    :arg X: float; list of floats/ints; exercise price(s)
    :arg r: float; interest rate
    :arg t: float; time to expiration (in years)
    :arg sigma: float; annualised volatility
    --
    :return: float; Black-Scholes theoretical call value
    """
    import numpy as np
    from pandas import Series
    from scipy.stats import norm

    X = np.array(X)
    d1 = (np.log(S/X) + (r + 0.5*sigma**2)*t)/(sigma*np.sqrt(t))
    d2 = d1 - sigma*np.sqrt(t)
    call = S*norm.cdf(d1) - X*np.exp(-r*t)*norm.cdf(d2)
    call = Series(call, index = X)

    return call

def bsPut(S, X, r, t, sigma):
    """
    Calculates theoretical put values using the Black-Scholes model.
    --
    :arg S: float; current underlying price
    :arg X: float; list of floats/ints; exercise price(s)
    :arg r: float; interest rate
    :arg t: float; time to expiration (in years)
    :arg sigma: float; annualised volatility
    --
    :return: float; Black-Scholes theoretical put value
    """
    import numpy as np
    from pandas import Series
    from scipy.stats import norm

    X = np.array(X)
    d1 = (np.log(S/X) + (r + 0.5*sigma**2)*t)/(sigma*np.sqrt(t))
    d2 = d1 - sigma*np.sqrt(t)
    put = X*np.exp(-r*t)*norm.cdf(-d2) - S*norm.cdf(-d1)
    put = Series(put, index = X)

    return put

def optionChain(chain, S, r, t, sigma, strikes = 10):
    """
    Calculates and returns pandas dataframe for put and call option chains with Black-Scholes theoretical values.
    --
    :arg chain: str; relative or absolute directory of options chain, saved from yahoo finance as .html
    :arg S: float; current underlying price
    :arg r: float; interest rate
    :arg t: float; time to expiration (in years)
    :arg sigma: float; annualised volatility
    :kwarg strikes: int; number of strike prices to display
    --
    :return call: pandas dataframe; call option chain and Black-Scholes theoretical values
    :return put: pandas dataframe; put option chain and Black-Scholes theoretical values
    """
    import pandas as pd
    from pandas import Series, DataFrame

    c = DataFrame(pd.read_html(chain)[0])
    p = DataFrame(pd.read_html(chain)[1])
    c.set_index(["Strike"], inplace = True)
    p.set_index(["Strike"], inplace = True)
    c["Moneyness"] = Series(S - c.index, index = c.index).map(lambda x: max(0,x))
    p["Moneyness"] = Series(p.index - S, index = p.index).map(lambda x: max(0,x))
    c = c.iloc[(c.index.get_loc(c["Moneyness"].idxmin()) - strikes//2):(c.index.get_loc(c["Moneyness"].idxmin()) + strikes//2), 2:10] #idxmin returns first occurrence only
    p = p.iloc[(p.index.get_loc(p["Moneyness"][p["Moneyness"] != 0].idxmin()) - 1 - strikes//2):(p.index.get_loc(p["Moneyness"][p["Moneyness"] != 0].idxmin()) - 1 + strikes//2), 2:10]
    c.drop(columns = ["Change","% Change"]) #for some reason the % one is sometimes capital 'C', but sometimes lower case 'c'?
    p.drop(columns = ["Change","% Change"]) #Moneyness column is by design omitted, can easily be added in by removing right column indexes in above lines
    call = bsCall(S, c.index, r, t, sigma)
    put = bsPut(S, p.index, r, t, sigma)
    c["Black-Scholes"] = call
    p["Black-Scholes"] = put
    c.columns.name = "Calls"
    p.columns.name = "Puts"

    return c,p