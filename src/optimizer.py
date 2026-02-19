import numpy as np
from scipy.optimize import minimize

def get_portfolio_volatility(weights, cov_matrix):
    return np.sqrt(weights.T @ cov_matrix @ weights)

def optimize_portfolio(target_return, returns, cov_matrix):

    expected_returns = returns.mean().values

    N = len(expected_returns)

    init_guess = np.ones(N) / N

    bounds = [(0,1) for _ in range(N)]

    # Somme des poids = 1
    budget_constraint = {'type':'eq', 'fun': lambda w: np.sum(w) - 1}
    # Expected return = Target return
    return_constraint = {'type':'eq', 'fun': lambda w: w.T @ expected_returns - target_return}

    res = minimize(get_portfolio_volatility,
                            init_guess,
                            args=(cov_matrix,),
                            bounds=bounds,
                            constraints=(budget_constraint, return_constraint),
                            method='SLSQP')

    return res.x

def get_gmv_portfolio(returns, cov_matrix):

    N = len(cov_matrix)

    init_guess = np.ones(N) / N

    bounds = [(0,1) for _ in range(N)]

    # Somme des poids = 1
    budget_constraint = {'type':'eq', 'fun': lambda w: np.sum(w) - 1}

    res = minimize(get_portfolio_volatility,
                            init_guess,
                            args=(cov_matrix,),
                            bounds=bounds,
                            constraints=(budget_constraint,),
                            method='SLSQP')

    R = returns.mean().values
    weights = res.x

    mu = weights.T @ R
    sigma = get_portfolio_volatility(weights, cov_matrix)

    return mu, sigma 

def generate_efficient_frontier(returns, num_points=50):

    ptf_mu = []
    ptf_sigma = []

    expected_returns = returns.mean().values
    cov_matrix = returns.cov().values

    gmv_mu, gmv_sigma = get_gmv_portfolio(returns, cov_matrix)
    min_expected_return = gmv_mu
    max_expected_return = max(expected_returns)

    target_returns = np.linspace(min_expected_return, max_expected_return, num_points)

    for target_return in target_returns:
        res = optimize_portfolio(target_return, returns, cov_matrix)
        vol = get_portfolio_volatility(res, cov_matrix)
        
        ptf_mu.append(target_return)
        ptf_sigma.append(vol)
        
    return ptf_mu, ptf_sigma, gmv_mu, gmv_sigma

def get_max_sharpe_portfolio(returns, risk_free_rate=0.0):
    
    expected_returns = returns.mean().values
    cov_matrix = returns.cov().values
    N = len(expected_returns)
    
    # Ratio de Sharpe négatif
    def neg_sharpe(weights):
        p_ret = weights.T @ expected_returns
        p_vol = np.sqrt(weights.T @ cov_matrix @ weights)
        return -(p_ret - risk_free_rate) / p_vol
    
    init_guess = np.ones(N) / N
    bounds = [(0, 1) for _ in range(N)]
    budget_constraint = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    
    res = minimize(neg_sharpe, 
                   init_guess, 
                   method='SLSQP', 
                   bounds=bounds, 
                   constraints=(budget_constraint,))
    
    weights = res.x
    max_sharpe_ret = weights.T @ expected_returns
    max_sharpe_vol = np.sqrt(weights.T @ cov_matrix @ weights)
    
    return max_sharpe_ret, max_sharpe_vol










