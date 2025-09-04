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

def impliedCall(S, X, r, t, C):
    """
    Calculates call option implied volatilities given option chain data
    --
    :arg S: float; current underlying price
    :arg X: list of floats/ints; exercise price(s)
    :arg r: float; interest rate
    :arg t: float; time to expiration (in years)
    :arg C: list of floats; last market call prices
    --
    :return: pandas series of floats; call option implied volatilities
    """
    import numpy as np
    from pandas import Series
    from scipy.optimize import fsolve
    from scipy.stats import norm

    def fc(sigma, S, X, r, t, C):
        d1 = (np.log(S / np.array(X)) + (r + 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))
        d2 = d1 - sigma * np.sqrt(t)
        return S * norm.cdf(d1) - np.array(X) * np.exp(-r * t) * norm.cdf(d2) - np.array(C)

    x0 = np.where(np.subtract(X,S) == 0, 1, np.subtract(X,S)) #initial guesses = intrinsic value (!= 0)
    result = Series(fsolve(fc, x0, args = (S,X,r,t,C))*100, index = X).round(2)
    return result.astype(str) + "%"

def impliedPut(S, X, r, t, P):
    """
    Calculates put option implied volatilities given option chain data
    --
    :arg S: float; current underlying price
    :arg X: list of floats/ints; exercise price(s)
    :arg r: float; interest rate
    :arg t: float; time to expiration (in years)
    :arg P: list of floats; last market put prices
    --
    :return: pandas series of floats; put option implied volatilities
    """
    import numpy as np
    from pandas import Series
    from scipy.optimize import fsolve
    from scipy.stats import norm

    def fp(sigma, S, X, r, t, P):
        d1 = (np.log(S / np.array(X)) + (r + 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))
        d2 = d1 - sigma * np.sqrt(t)
        return np.array(X) * np.exp(-r * t) * norm.cdf(-d2) - S * norm.cdf(-d1) - np.array(P)

    x0 = np.where(np.subtract(S, X) == 0, 1, np.subtract(S, X))
    result = Series(fsolve(fp, x0, args = (S,X,r,t,P))*100, index = X).round(2)
    return result.astype(str) + "%"

def optionChain(chain, S, r, t, sigma, strikes = 10):
    """
    Calculates and returns pandas dataframe for put and call option chains with Black-Scholes theoretical values and implied volatilities.
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
    c["Intrinsic Value"] = Series(S - c.index, index = c.index).map(lambda x: max(0,x))
    p["Intrinsic Value"] = Series(p.index - S, index = p.index).map(lambda x: max(0,x))
    c = c.iloc[(c.index.get_loc(c["Intrinsic Value"].idxmin()) - strikes//2):(c.index.get_loc(c["Intrinsic Value"].idxmin()) + strikes//2), 2:9] #idxmin returns first occurrence only
    p = p.iloc[(p.index.get_loc(p["Intrinsic Value"][p["Intrinsic Value"] != 0].idxmin()) - 1 - strikes//2):(p.index.get_loc(p["Intrinsic Value"][p["Intrinsic Value"] != 0].idxmin()) - 1 + strikes//2), 2:9]
    c.drop(columns = ["Change","% Change"], inplace = True) #Intrinsic Value column is by design omitted, can easily be added in by removing right column indexes in above lines
    p.drop(columns = ["Change","% Change"], inplace = True)
    c["Black-Scholes"] = bsCall(S, c.index, r, t, sigma).round(3)
    p["Black-Scholes"] = bsPut(S, p.index, r, t, sigma).round(3)
    c["BS Implied Volatility"] = impliedCall(S, c.index, r, t, c["Last Price"])
    p["BS Implied Volatility"] = impliedPut(S, p.index, r, t, p["Last Price"])
    c.columns.name = "Calls"
    p.columns.name = "Puts"
    return c,p