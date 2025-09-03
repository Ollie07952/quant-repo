def GBM(s0,mu,sigma,paths,t_days):
    import numpy as np

    """
    
    Simulates Geometric Brownian Motion
    --
    
    Paramaters:
    
    s0 : float or int; starting stock price
    mu : float; drift (annual return)
    sigma : float; annualised volatility
    t_days : int; the number of trading days to simulate
    paths : int; the number of different paths to simulate simultaneously
    --
    
    Returns:
    
    walks : array(s); array(s) of stock price over the entire period

    """
    
    rng = np.random.default_rng()
    normals = rng.standard_normal(size = (paths,t_days))
    steps = np.exp((mu-0.5*np.square(sigma))*(1/252) + sigma*np.sqrt(1/252)*normals)
    steps = np.concatenate((np.full(shape = (paths,1),fill_value = s0),steps), axis = 1)
    walks = np.cumprod(steps, axis = 1)

    return walks

