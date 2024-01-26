"""
Created on Mon Oct 16 11:52:09 2023

@author: mansourzarrin
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import yfinance as yf


class PortfolioOptimizer:
    def __init__(self, tickers, start_date, end_date):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.historical_returns = self.fetch_data()
        self.num_assets = len(tickers)
        self.log_ret = np.log(self.historical_returns / self.historical_returns.shift(1))
        
    def fetch_data(self):
        data = yf.download(self.tickers, start=self.start_date, end=self.end_date)['Adj Close']
        return data.pct_change(1).dropna()
        
    def monte_carlo_simulation(self, num_sims=10000):
        self.num_sims = num_sims
        self.portfolio_returns = np.zeros(num_sims)
        self.portfolio_volatility = np.zeros(num_sims)
        self.sharpe_ratio = np.zeros(num_sims)
        
        for i in range(num_sims):
            weights = np.random.random(self.num_assets)
            weights /= np.sum(weights)
            self.portfolio_returns[i] = np.sum(self.log_ret.mean() * weights) * len(self.log_ret)
            self.portfolio_volatility[i] = np.sqrt(np.dot(weights.T, np.dot(self.log_ret.cov() * len(self.log_ret), weights)))
            self.sharpe_ratio[i] = self.portfolio_returns[i] / self.portfolio_volatility[i]
            
        return self.portfolio_returns, self.portfolio_volatility, self.sharpe_ratio
    
    def optimize_portfolio(self):
        def neg_sharpe(weights):
            return -self.calculate_performance_metrics(weights)[2]
        
        def check_sum(weights):
            return np.sum(weights) - 1
        
        bounds = tuple((0, 1) for asset in range(self.num_assets))
        constraints = ({'type': 'eq', 'fun': check_sum})
        init_guess = [1./self.num_assets for asset in range(self.num_assets)]
        
        opt_results = minimize(neg_sharpe, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
        return opt_results
    
    def calculate_frontier_volatility(self, target_return):
        def objective(weights):
            return np.sqrt(np.dot(weights.T, np.dot(self.log_ret.cov() * len(self.log_ret), weights)))
    
        def check_sum(weights):
            return np.sum(weights) - 1
    
        def target_return_constraint(weights):
            return np.sum(self.log_ret.mean() * weights) * len(self.log_ret) - target_return
    
        bounds = tuple((0, 1) for asset in range(self.num_assets))
        constraints = ({'type': 'eq', 'fun': check_sum},
                       {'type': 'eq', 'fun': target_return_constraint})
        init_guess = [1./self.num_assets for asset in range(self.num_assets)]
    
        opt_results = minimize(objective, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
        return opt_results.fun
    
    def calculate_performance_metrics(self, weights):
        ret = np.sum(self.log_ret.mean() * weights) * len(self.log_ret)
        vol = np.sqrt(np.dot(weights.T, np.dot(self.log_ret.cov() * len(self.log_ret), weights)))
        sr = ret / vol
        return np.array([ret, vol, sr])

# Sample usage
if __name__ == "__main__":
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    start_date = '2023-06-01'
    end_date = '2023-10-01'
    
    optimizer = PortfolioOptimizer(tickers, start_date, end_date)
    optimizer.monte_carlo_simulation(num_sims=10000)
    opt_results = optimizer.optimize_portfolio()
    
    print("Optimal Portfolio Weights: ", opt_results.x)
    print("Optimal Portfolio Performance Metrics: ", optimizer.calculate_performance_metrics(opt_results.x))
    
    optimizer.plot_efficient_frontier()
    optimizer.display_results(opt_results)
