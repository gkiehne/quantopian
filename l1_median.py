def l1_median(x):
    """
Computes L1 median (spatial median) using scipy.optimize.minimize_scalar

:param x: a numpy 1D ndarray (vector) of values
:returns: scalar estimate of L1 median of values
"""
    a = float(np.amin(x))
    b = float(np.amax(x))
    
    res = minimize_scalar(dist_sum, bounds = (a,b), args = tuple(x), method='bounded')
    
    return res.x

def dist_sum(m,*args):
    """
1D sum of Euclidian distances

:param m: scalar position
:param *args: tuple of positions 
:returns: 1D sum of Euclidian distances
"""
    
    s = 0
    for x in args:
        s += abs(x-m)
    
    return s
