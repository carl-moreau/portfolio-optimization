# Portfolio Optimization: Markowitz & Monte Carlo

## Description
This project is a Python implementation of Modern Portfolio Theory (Markowitz). It allows users to visualize the efficient frontier and identify optimal asset allocations using Monte Carlo simulations. 

The project analyzes a basket of stocks (e.g., AAPL, GOOG, AMZN, WMT, XOM, CVX) to find the best trade-off between risk (expected volatility) and expected return.

## Features
* **Data Fetching:** Downloads historical market prices and calculates returns.
* **Monte Carlo Simulations:** Generates thousands of random portfolios to visualize the universe of possible allocations.
* **Mathematical Optimization:** * Plots the **Efficient Frontier**.
  * Identifies the **Global Minimum Variance (GMV)** portfolio.
  * Identifies the **Maximum Sharpe Ratio** portfolio.
* **Data Visualization:** Comprehensive charting with `matplotlib` (Sharpe ratio scatter plots, frontier curves).

## Project Structure
* `example.ipynb`: The main Jupyter notebook containing the step-by-step analysis and plots.
* `src/`: Directory containing the source code modules:
  * `data_loader.py`: Handles fetching and preparing financial data (`get_market_data`, `calculate_returns`).
  * `simulations.py`: Logic for creating random portfolios (`create_random_portfolio`).
  * `optimizer.py`: Algorithms to compute the efficient frontier and optimal portfolios (`generate_efficient_frontier`, `get_gmv_portfolio`, `get_max_sharpe_portfolio`).