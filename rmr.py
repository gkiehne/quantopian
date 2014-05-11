"""
Robust Median Reversion Strategy for On-Line Portfolio Selection,
Dingjiang Huang, Junlong Zhou, Bin Li, Steven C.H. Hoi, and Shuigeng
Zhou. International Joint Conference on Artificial Intelligence, 2013.
http://ijcai.org/papers13/Papers/IJCAI13-296.pdf
"""

import numpy as np
from pytz import timezone
from scipy.optimize import minimize_scalar

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

    context.timeframe = 'Daily' # Algo time frame

    context.bar_count = 1
    context.trading_frequency = 1 # trading frequency in bars
    context.window = 15 # trailing window length in bars

    context.prices = np.zeros([context.window, len(context.stocks)])

    if context.timeframe == 'Daily':
        set_slippage(TradeAtTheOpenSlippageModel(0.1))
    else:
        set_slippage(slippage.FixedSlippage(spread=0.03))

    set_commission(commission.PerTrade(cost=5))

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
        for stock in context.stocks:
            order_target_percent(stock, part)

        accumulator(context, data)
        context.init = True
        return

    if context.timeframe == 'Daily':
        # accumulate price data
        accumulator(context, data)
        context.bar_count += 1

        if context.bar_count < context.window:
            return
    else:
        # Trade only once per day, at 10:00
        loc_dt = get_datetime().astimezone(timezone('US/Eastern'))
        if loc_dt.hour != 10 or loc_dt.minute != 0:
            return
        context.bar_count += 1

    if context.bar_count % context.trading_frequency:
        return

    if get_open_orders():
        return

    # skip bar if any stocks did not trade
    for stock in context.stocks:
        if data[stock].datetime < get_datetime():
            return

    if context.timeframe == 'Daily':
        prices = context.prices
    else:
        # uncomment below line for minute time frame to work
        # prices = history(15, '1d', 'price').as_matrix(context.stocks)[0:-1,:]
        pass

    parts = rmr_strategy(context.portfolio, context.stocks, data,
                         prices, context.eps)

    # rebalance portfolio accroding to new allocation
    for stock, portion in zip(context.stocks, parts):
        order_target_percent(stock, portion)

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

    x_tilde = np.zeros(len(stocks))
    for i, stock in enumerate(stocks):
        x_tilde[i] = l1_median(prices[:,i])/prices[-1,i]

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

def l1_median(x):
    """
    Computes L1 median (spatial median) using scipy.optimize.minimize_scalar

    :param x: a numpy 1D ndarray (vector) of values
    :returns: scalar estimate of L1 median of values
    """
    x_min = float(np.amin(x))
    x_max = float(np.amax(x))

    res = minimize_scalar(dist_sum, bounds = (x_min, x_max),
                          args = tuple(x), method='bounded')

    return res.x

def dist_sum(m, *args):
    """
    1D sum of Euclidian distances

    :param m: scalar position
    :param *args: tuple of positions
    :returns: 1D sum of Euclidian distances
    """
    return sum(abs(arg - m) for arg in args)

def accumulator(context, data):
    """
    Accumulate price data into context.prices

    :param context: context object
    :param data: A dictionary containing market data keyed by security id.
    :returns: None
    """
    if context.bar_count < context.window:
        for i, stock in enumerate(context.stocks):
            context.prices[context.bar_count, i] = data[stock].price
    else:
        context.prices = np.roll(context.prices, -1, axis=0)
        for i, stock in enumerate(context.stocks):
            context.prices[-1, i] = data[stock].price

class TradeAtTheOpenSlippageModel(slippage.SlippageModel):
    """
    Custom slippage model to allow trading at the open
    or at a fraction of the open to close range.
    """

    def __init__(self, fraction):
        """
        Constructor. Set fraction of the open to close range to
        add (subtract) from the open to model executions more optimistically

        :param fraction: fraction of open-close range to take as
                         the execution price
        """
        self.fraction = fraction

    def process_order(self, trade_bar, order):
        """
        Process order.
        https://www.quantopian.com/help#ide-slippage

        :param trade_bar: price data
        :param order: order object
        :returns: transaction object
        """
        open_price = trade_bar.open_price
        close_price = trade_bar.close_price
        ocrange = (close_price - open_price) * self.fraction
        exec_price = open_price + ocrange
        log.info('Order:{0} open:{1} close:{2} exec:{3} side:{4}'.format(
            trade_bar.sid.symbol, open_price, close_price,
            exec_price, order.direction))

        # Create the transaction using calculated execution price
        return slippage.create_transaction(trade_bar, order,
                                           exec_price, order.amount)
