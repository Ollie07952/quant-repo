def bsCall(S, X, r, t, sigma):
    """
    Calculates theoretical call values using the Black-Scholes model.
    --
    :arg S: float; current underlying price
    :arg X: float; array of floats/ints; exercise price(s)
    :arg r: float; interest rate
    :arg t: float; time to expiration (in years)
    :arg sigma: float; annualised volatility
    --
    :return: float; Black-Scholes theoretical call value
    """
    import numpy as np
    from pandas import Series, DataFrame
    from scipy.stats import norm

    X = np.array(X)
    d1 = (np.log(S/X) + (r + 0.5*sigma**2)*t)/(sigma*np.sqrt(t))
    d2 = d1 - sigma*np.sqrt(t)
    call = S*norm.cdf(d1) - X*np.exp(-r*t)*norm.cdf(d2)
    call = Series(call, index = X)
    delta = norm.cdf(d1)
    gamma = norm.pdf(d1)/(S*sigma*np.sqrt(t))
    theta = ((-S * norm.pdf(d1) * sigma) / (2 * np.sqrt(t)) - r*X*np.exp(-r*t)*norm.cdf(d2))/365
    greeks = DataFrame({"Delta":delta, "Gamma":gamma, "Theta":theta}, index=X)
    return call, greeks

def bsPut(S, X, r, t, sigma):
    """
    Calculates theoretical put values using the Black-Scholes model.
    --
    :arg S: float; current underlying price
    :arg X: float; array of floats/ints; exercise price(s)
    :arg r: float; interest rate
    :arg t: float; time to expiration (in years)
    :arg sigma: float; annualised volatility
    --
    :return: float; Black-Scholes theoretical put value
    """
    import numpy as np
    from pandas import Series, DataFrame
    from scipy.stats import norm

    X = np.array(X)
    d1 = (np.log(S/X) + (r + 0.5*sigma**2)*t)/(sigma*np.sqrt(t))
    d2 = d1 - sigma*np.sqrt(t)
    put = X*np.exp(-r*t)*norm.cdf(-d2) - S*norm.cdf(-d1)
    put = Series(put, index = X)
    delta = norm.cdf(d1)-1
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(t))
    theta = ((-S * norm.pdf(d1) * sigma) / (2 * np.sqrt(t)) + r*X*np.exp(-r*t)*norm.cdf(-d2))/365
    greeks = DataFrame({"Delta": delta, "Gamma": gamma, "Theta": theta}, index=X)
    return put, greeks

def impliedCall(S, X, r, t, C):
    """
    Calculates call option implied volatilities given option chain data
    --
    :arg S: float; current underlying price
    :arg X: array of floats/ints; exercise price(s)
    :arg r: float; interest rate
    :arg t: float; time to expiration (in years)
    :arg C: array of floats; last market call prices
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

    x0 = (np.sqrt(2*np.pi)/np.sqrt(t))*(np.array(C)-(S-np.array(X)*np.exp(-r*t))/2)/(S+np.array(X)*np.exp(-r*t)) #Corrado-Miller approximations as initial guesses
    result = Series(fsolve(fc, x0, args = (S,X,r,t,C))*100, index = X).round(2)
    return result.astype(str) + "%"

def impliedPut(S, X, r, t, P):
    """
    Calculates put option implied volatilities given option chain data
    --
    :arg S: float; current underlying price
    :arg X: array of floats/ints; exercise price(s)
    :arg r: float; interest rate
    :arg t: float; time to expiration (in years)
    :arg P: array of floats; last market put prices
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

    x0 = (np.sqrt(2*np.pi)/np.sqrt(t))*(np.array(P)-(S-np.array(X)*np.exp(-r*t))/2)/(S+np.array(X)*np.exp(-r*t)) #Corrado-Miller approximations as initial guesses
    result = Series(fsolve(fp, x0, args = (S,X,r,t,P))*100, index = X).round(2)
    return result.astype(str) + "%"

def optionChain(ticker, expiration, r, sigma, strikes = 10):
    """
    Calculates and returns pandas dataframe for put and call option chains with Black-Scholes theoretical values, implied volatilities, and greeks columns.
    Disclaimer - This tool is meant for purely educational purposes and no responsibility is accepted for the accuracy of the figures produced. Market quotes data is from Yahoo! Finance.
    --
    :arg ticker: str; ticker for which you want the option chain
    :arg expiration: str; expiration date in the form "YYYY-MM-DD"
    :arg r: float; interest rate over the life of the option
    :arg sigma: float; annualised volatility estimate
    :kwarg strikes: int; number of strike prices to display (default 10) (Note: fsolve begins to fail to converge for strikes > 14 due to inherent error of C-M initial guesses and limitations of scipy's fsolve method.)
    --
    :return call: pandas dataframe; call option chain and Black-Scholes theoretical values
    :return put: pandas dataframe; put option chain and Black-Scholes theoretical values
    """
    import yfinance as yf
    import datetime as dt
    from pandas import Series, DataFrame

    stock = yf.Ticker(ticker)
    S = stock.info["regularMarketPrice"]
    t = (dt.datetime(int(expiration[:4]),int(expiration[5:7]),int(expiration[8:]))-dt.datetime.now()).days/365

    c = DataFrame(stock.option_chain(date = expiration)[0])
    p = DataFrame(stock.option_chain(date = expiration)[1])
    c.set_index("strike", inplace = True)
    p.set_index("strike", inplace = True)
    c["Intrinsic Value"] = Series(S - c.index, index = c.index).map(lambda x: max(0,x))
    p["Intrinsic Value"] = Series(p.index - S, index = p.index).map(lambda x: max(0,x))
    c = c.iloc[(c.index.get_loc(c["Intrinsic Value"].idxmin()) - strikes//2):(c.index.get_loc(c["Intrinsic Value"].idxmin()) + strikes//2), 2:5] #idxmin returns first occurrence only
    p = p.iloc[(p.index.get_loc(p["Intrinsic Value"][p["Intrinsic Value"] != 0].idxmin()) - 1 - strikes//2):(p.index.get_loc(p["Intrinsic Value"][p["Intrinsic Value"] != 0].idxmin()) - 1 + strikes//2), 2:5]
    c["Black-Scholes"] = bsCall(S, c.index, r, t, sigma)[0].round(3)
    p["Black-Scholes"] = bsPut(S, p.index, r, t, sigma)[0].round(3)
    c["Implied Volatility"] = impliedCall(S, c.index, r, t, c["lastPrice"])
    p["Implied Volatility"] = impliedPut(S, p.index, r, t, p["lastPrice"])
    c = c.join(bsCall(S, c.index, r, t, sigma)[1].round(3))
    p = p.join(bsPut(S, p.index, r, t, sigma)[1].round(3))
    c.columns.name = "Calls"
    p.columns.name = "Puts"
    return c,p