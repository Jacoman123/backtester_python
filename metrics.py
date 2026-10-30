"""
metrics.py - Performance statistics.

All functions take DAILY simple returns (or an equity curve) and annualise
assuming 252 trading days per year.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from engine import BacktestResult

TRADING_DAYS = 252


def annualised_volatility(returns: pd.Series, periods: int = TRADING_DAYS) -> float:
    """sigma_annual = sigma_daily * sqrt(252).

    Variance of independent daily returns adds up over days, so the standard
    deviation (its square root) scales with sqrt(time), not with time.
    """
    return float(returns.std(ddof=1) * np.sqrt(periods))


def sharpe_ratio(returns: pd.Series, rf_annual: float = 0.0, periods: int = TRADING_DAYS) -> float:
    """Annualised Sharpe ratio: return earned per unit of risk taken.

        SR = sqrt(252) * mean(r - rf/252) / std(r - rf/252)

    rf_annual is the risk-free rate (e.g. T-bill yield). Defaulting to 0 is a
    common simplification; with it, SR is slightly flattered when rates are high.
    """
    excess = returns - rf_annual / periods
    sd = excess.std(ddof=1)
    if not np.isfinite(sd) or sd < 1e-12:  # constant returns: Sharpe is undefined
        return float("nan")
    return float(np.sqrt(periods) * excess.mean() / sd)


def drawdown_series(equity: pd.Series) -> pd.Series:
    """Percentage below the previous peak, at each date (always <= 0).

        DD_t = E_t / max_{s <= t} E_s - 1
    """
    return equity / equity.cummax() - 1.0


def max_drawdown(equity: pd.Series) -> float:
    """Worst peak-to-trough loss: min_t DD_t (a negative number)."""
    return float(drawdown_series(equity).min())


def cagr(equity: pd.Series, periods: int = TRADING_DAYS) -> float:
    """Compound annual growth rate: CAGR = E_T^(252 / T) - 1, where T = number of days."""
    n_days = len(equity) - 1
    if n_days <= 0:
        return float("nan")
    return float(equity.iloc[-1] ** (periods / n_days) - 1.0)


def summarise(result: BacktestResult, rf_annual: float = 0.0) -> dict:
    """Collect the headline numbers for one backtest."""
    r = result.net_returns
    return {
        "Strategy": result.name,
        "Total return": result.equity.iloc[-1] - 1.0,
        "CAGR": cagr(result.equity),
        "Ann. vol": annualised_volatility(r),
        "Sharpe": sharpe_ratio(r, rf_annual),
        "Max drawdown": max_drawdown(result.equity),
        "Ann. turnover": result.turnover.mean() * TRADING_DAYS,
        "Ann. cost drag": result.costs.mean() * TRADING_DAYS,
    }
