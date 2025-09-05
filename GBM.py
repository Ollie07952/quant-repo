def GBM(s0,mu,sigma,paths,t_days):
    """
    Simulates Geometric Brownian Motion
    --
    :arg s0: float/int; starting stock price
    :arg mu: float; drift (annual return)
    :arg sigma: float; annualised volatility
    :arg t_days: int; the number of trading days to simulate
    :arg paths: int; the number of different paths to simulate simultaneously
    --
    :return walks: array; array(s) of stock price over the entire period
    """
    import numpy as np
    
    rng = np.random.default_rng()
    normals = rng.standard_normal(size = (paths,t_days))
    steps = np.exp((mu-0.5*np.square(sigma))*(1/252) + sigma*np.sqrt(1/252)*normals)
    steps = np.concatenate((np.full(shape = (paths,1),fill_value = s0),steps), axis = 1)
    walks = np.cumprod(steps, axis = 1)
    return walks
