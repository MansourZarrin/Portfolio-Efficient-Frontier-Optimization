---
author:
- Mansour Zarrin
title: "Portfolio Efficient Frontier Optimization: A Pioneering Approach
  Using Monte Carlo Simulation and Non-linear Programming"
---

# Introduction

This report outlines a Portfolio Optimization model that employs Monte
Carlo Simulation and Quadratic Programming to determine the optimal
asset allocation for a given set of financial instruments. The model is
implemented in Python, leveraging libraries such as NumPy, pandas,
SciPy, Matplotlib, and finance. The model successfully identifies the
optimal portfolio by maximizing the Sharpe Ratio, providing valuable
insights for investment strategies. The efficient frontier is also
plotted to visualize the set of optimal portfolios.

# Parameters and Variables

-   **Tickers**: List of asset symbols.

-   **Start_Date, End_Date**: Time range for historical data.

-   **Historical_Returns**: Adjusted close prices.

-   **Log_Ret**: Logarithmic returns.

-   **Weights**: Asset allocation in the portfolio.

# Simulation Module

The Monte Carlo Simulation is employed to generate a wide range of
portfolios with varying asset allocations. The simulation runs for a
user-defined number of iterations (*num_sims*) and calculates the
following metrics for each portfolio:
$$\\begin{aligned}
    \\text{Portfolio Return} &= \\sum\_{i} (\\text{Log\\\_Ret}\_i \\times \\text{Weight}\_i) \\times N \\\\
    \\text{Portfolio Volatility} &= \\sqrt{\\text{Weight}^T \\times (\\text{Log\\\_Ret.cov()} \\times N) \\times \\text{Weight}} \\\\
    \\text{Sharpe Ratio} &= \\frac{\\text{Portfolio Return}}{\\text{Portfolio Volatility}}\\end{aligned}$$
where *N* is the number of data points in the historical returns,
serving as the annualization factor.

# Optimization Model

The optimization model employs Sequential Least Squares Quadratic
Programming (SLSQP) to maximize the Sharpe Ratio. The model is subject
to the constraint that the sum of the asset weights must be equal to
one. Mathematically, the optimization problem is formulated as:
$$\\begin{aligned}
    & \\underset{\\text{Weights}}{\\text{maximize}}
    & & \\frac{\\text{Portfolio Return}}{\\text{Portfolio Volatility}} \\\\
    & \\text{subject to}
    & & \\sum\_{i} \\text{Weight}\_i = 1
\\end{aligned}$$
