import numpy as np
import pandas as pd

from src.data_loader import get_market_data, calculate_returns


def create_random_weights(n_assets):
    weights = np.random.rand(n_assets)
    return weights / weights.sum()

def get_portfolio_stats(returns, weights):

    returns = pd.DataFrame(returns)

    R = returns.mean().values
    cov = returns.cov().values

    mu = weights.T @ R
    sigma = np.sqrt(weights.T @ cov @ weights)

    return mu, sigma

def create_random_portfolio(returns, n_portfolios=2000):

    n_assets = returns.shape[1]    

    mu_portfolios, sigma_portfolios = np.column_stack([get_portfolio_stats(returns, create_random_weights(n_assets)) for _ in range(n_portfolios)])
    
    return mu_portfolios, sigma_portfolios

if __name__ == '__main__':
        
    prices = get_market_data(['AAPL','MSFT'])
    returns = calculate_returns(prices)

    print(create_random_portfolio(returns, 10))




