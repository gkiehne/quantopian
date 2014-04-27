def l1_median(x):
    """
Computes L1 median (spatial median) using brute force grid search.

:param x: a numpy 1D ndarray (vector) of values
:returns: scalar estimate of L1 median of values
"""
    
    x_min = np.amin(x)
    x_max = np.amax(x)
    
    mu = np.linspace(x_min, x_max, num=50)
    
    sum_dist = np.zeros_like(mu)
    
    for k in range(len(mu)):
        mu_k = mu[k]*np.ones_like(x)
        sum_dist[k] = np.sum(np.absolute(x-mu_k))
        
    l_min = np.argmin(sum_dist)
    
    return mu[l_min]
