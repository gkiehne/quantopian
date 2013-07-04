def rebalance_portfolio(context, data, desired_port):
    #rebalance portfolio
    current_amount = np.zeros_like(desired_port)
    desired_amount = np.zeros_like(desired_port)
    
    if not context.init:
        positions_value = context.portfolio.starting_cash
    else:
        positions_value = context.portfolio.positions_value + context.portfolio.cash  
    
    for i, stock in enumerate(context.stocks):
        current_amount[i] = context.portfolio.positions[stock].amount
        desired_amount[i] = desired_port[i]*positions_value/data[stock].price

    diff_amount = desired_amount - current_amount

    for i, stock in enumerate(context.stocks):
        order(stock, diff_amount[i]) #order_stock
