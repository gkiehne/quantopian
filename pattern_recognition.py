# References:
# Ming Li; Xin Chen; Xin Li; Bin Ma; Vitanyi, P.M.B.; , "The similarity metric,"
# Information Theory, IEEE Transactions on , vol.50, no.12, pp. 3250- 3264, Dec. 2004
# http://homepages.cwi.nl/~paulv/papers/similarity.pdf
# 
# Lin, Jessica, et al. "A symbolic representation of time series, with implications for 
# streaming algorithms." Proceedings of the 8th ACM SIGMOD workshop on Research issues 
# in data mining and knowledge discovery. ACM, 2003.
# www.cs.ucr.edu/~stelo/papers/DMKD03.pdf

import numpy as np
import zlib
from scipy import stats

# globals for get_avg batch transform decorator
R_P = 1  # refresh period in days
W_L = 30  # window length in days

def initialize(context):
    
    # context.stocks = [sid(26807),sid(32133)] # GLD & GDX
    context.stocks = [sid(8554),sid(21513)] # SPY & IVV
    
def handle_data(context, data):
    
    # get prices
    prices = get_prices(data, context.stocks)
    if prices is None: 
        return
   
    # normalize prices with z-score
    prices_z = stats.zscore(prices, axis=0, ddof=1)
    
    # code prices as text strings & compute NCD relative to first
    # security in context.stocks
    coded_prices = prices_z
    
    coded_prices[coded_prices >= 0.97] = 6
    coded_prices[(coded_prices >= 0.43) & (coded_prices < 0.97)] = 5
    coded_prices[(coded_prices >= 0) & (coded_prices < 0.43)] = 4
    coded_prices[(coded_prices >= -0.43) & (coded_prices < 0)] = 3
    coded_prices[(coded_prices >= -0.97) & (coded_prices < -0.43)] = 2
    coded_prices[coded_prices < -0.97] = 1
    
    m = len(context.stocks)-1
    N = np.zeros(m)    
    
    for j in range(m):
        X = ''
        Y = ''
        for i in range(W_L):
            X = X + str(int(coded_prices[i,0]))
            Y = Y + str(int(coded_prices[i,j+1]))
        N[j] = NCD(X,Y)    
    
    # plot NCD for securities 
    record(N_0 = N[0])
       
@batch_transform(refresh_period=R_P, window_length=W_L) # set globals R_P & W_L above
def get_prices(datapanel,sids):
    return datapanel['price'].as_matrix(sids)

def NCD( X, Y ):  
    CX = len(zlib.compress(X,9))  
    CY = len(zlib.compress(Y,9))  
    return (len(zlib.compress(X+Y,9)) - min(CX,CY)) / float(max(CX,CY))
