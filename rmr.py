"""
Robust Median Reversion Strategy for On-Line Portfolio Selection,
Dingjiang Huang, Junlong Zhou, Bin Li, Steven C.H. Hoi, and Shuigeng
Zhou. International Joint Conference on Artificial Intelligence, 2013.
http://ijcai.org/papers13/Papers/IJCAI13-296.pdf
"""

from itertools import repeat
import numpy as np
from pytz import timezone

def initialize(context):
    """
    Initialize context object.

    Context object is passed to the every call of handle_data.
    It uses to store data to use in the algo.

    :param context: context object
    :returns: None
    """
    context.stocks = [ sid(19662),  # XLY Consumer Discrectionary SPDR Fund
                       sid(19656),  # XLF Financial SPDR Fund
                       sid(19658),  # XLK Technology SPDR Fund
                       sid(19655),  # XLE Energy SPDR Fund
                       sid(19661),  # XLV Health Care SPRD Fund
                       sid(19657),  # XLI Industrial SPDR Fund
                       sid(19659),  # XLP Consumer Staples SPDR Fund
                       sid(19654),  # XLB Materials SPDR Fund
                       sid(19660) ] # XLU Utilities SPRD Fund

    context.eps = 5 # change epsilon here
    context.init = False

    set_slippage(slippage.FixedSlippage(spread=0.00))
    set_commission(commission.PerTrade(cost=0))

def handle_data(context, data):
    """
    The main proccessing function.

    This function is called by quantopian backedn whenever a market event
    occurs for any of algorithm's specified securities.

    :param context: context object
    :param data: A dictionary containing market data keyed by security id.
                 It represents a snapshot of your algorithm's universe as of
                 when this method was called.
    """
    record(cash=context.portfolio.cash)

    if not context.init:
        # initializisation. Buy the same amount of each security
        part = 1. / len(context.stocks)
        rebalance_portfolio(context, repeat(part))
        context.init = True
        return

    if not intradingwindow_check(context):
        return

    if get_open_orders():
        return

    # skip bar if any stocks did not trade
    for stock in context.stocks:
        if data[stock].datetime < get_datetime():
            return

    prices = history(6, '1d', 'price').as_matrix(context.stocks)[0:-1,:]

    parts = rmr_strategy(context.portfolio, context.stocks, data,
                         prices, context.eps)
    rebalance_portfolio(context, parts)

def rmr_strategy(portfolio, stocks, data, prices, eps):
    """
    Core of Robust Median Reviersion strategy implementation.

    :param portfolio: portfolio object
    :param stocks: list of sid objects used in the algo
    :param data: market event object
    :param prices: historical data in a form of numpy matrix
    :param eps: epsilon value
    :returns: new allocation for the portfolio securities
    """
    # update portfolio
    b_t = []
    for stock in stocks:
        b_t.append(portfolio.positions[stock].amount * data[stock].price)

    b_t = np.divide(b_t, np.sum(b_t))

    m = len(stocks)
    x_tilde = np.zeros(m)
    for i, stock in enumerate(stocks):
        # Use numpy median until L1 median (spatial median) implemented
        median_price = np.median(prices[:,i])
        x_tilde[i] = median_price/prices[-1,i]

    x_bar = x_tilde.mean()

    # Calculate terms for lambda (lam)
    dot_prod = np.dot(b_t, x_tilde)
    num = eps - dot_prod
    denom = (np.linalg.norm((x_tilde - x_bar))) ** 2

    b = b_t
    # test for divide-by-zero case
    if denom != 0.0:
        b =  b_t + max(0, num/denom) *  (x_tilde -  x_bar)

    return simplex_projection(b)

def rebalance_portfolio(context, desired_port):
    """
    Rebalance portfolio according to desired percentage.

    :param context: context object
    :param desired_port: list of desired percentages
    """
    for stock, portion in zip(context.stocks, desired_port):
        order_target_percent(stock, portion)

def intradingwindow_check(context):
    """
    Check if algo is in trading window.

    :param context: context object
    :returns: True if algo is in trading window, False otherwise.
    """
    # Converts all time-zones into US EST to avoid confusion
    loc_dt = get_datetime().astimezone(timezone('US/Eastern'))
    if loc_dt.hour == 10 and loc_dt.minute == 0:
        return True
    else:
        return False

def simplex_projection(v, b=1):
    """
    Projection vectors to the simplex domain.

    Implemented according to the paper: Efficient projections onto the
    l1-ball for learning in high dimensions, John Duchi, et al. ICML 2008.
    Implementation Time: 2011 June 17 by Bin@libin AT pmail.ntu.edu.sg
    Optimization Problem: min_{w}\| w - v \|_{2}^{2}
    s.t. sum_{i=1}^{m}=z, w_{i}\geq 0

    Input: A vector v \in R^{m}, and a scalar z > 0 (default=1)
    Output: Projection vector w

    :Example:
    >>> proj = simplex_projection([.4 ,.3, -.4, .5])
    >>> print proj
    array([ 0.33333333, 0.23333333, 0. , 0.43333333])
    >>> print proj.sum()
    1.0

    Original matlab implementation: John Duchi (jduchi@cs.berkeley.edu)
    Python-port: Copyright 2012 by Thomas Wiecki (thomas.wiecki@gmail.com).
    """
    v = np.asarray(v)
    p = len(v)

    # Sort v into u in descending order
    v = (v > 0) * v
    u = np.sort(v)[::-1]
    sv = np.cumsum(u)

    rho = np.where(u > (sv - b) / np.arange(1, p+1))[0][-1]
    theta = np.max([0, (sv[rho] - b) / (rho+1)])
    w = (v - theta)
    w[w < 0] = 0
    return w
