"""
data.py - Load and prepare historical price data.

We use *adjusted close* prices. Adjusted means the history has been corrected
for stock splits and dividends, so that the return computed from two adjacent
prices is the return an investor would actually have earned (total return).
Using raw close prices would show a fake -50% "crash" on the day of a 2-for-1
split.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_prices(
    ticker: str,
    start: str,
    end: str,
    cache_dir: str = "data_cache",
) -> pd.Series:
    """Download adjusted close prices from Yahoo Finance (cached on disk).

    Returns a pd.Series indexed by date, sorted ascending, named after the ticker.
    """
    cache_path = Path(cache_dir) / f"{ticker}_{start}_{end}.csv"
    if cache_path.exists():
        prices = pd.read_csv(cache_path, index_col=0, parse_dates=True).iloc[:, 0]
        return _validate(prices.rename(ticker))

    import yfinance as yf  # imported here so the rest of the project works offline

    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for {ticker} between {start} and {end}")

    # Newer yfinance versions return MultiIndex columns like ("Close", "SPY").
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    prices = _validate(df["Close"].rename(ticker))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(cache_path)
    return prices


def synthetic_prices(
    n_days: int = 2500,
    annual_drift: float = 0.07,
    annual_vol: float = 0.18,
    seed: int = 42,
    start: str = "2015-01-01",
) -> pd.Series:
    """Generate a fake price series from geometric Brownian motion (GBM).

    Useful for offline testing. Because GBM has no exploitable pattern by
    construction, no strategy should have a real edge on it; if a strategy
    looks great on GBM, you have a bug (usually look-ahead).

    Daily log-return:  (mu - sigma^2 / 2) * dt + sigma * sqrt(dt) * Z,  Z ~ N(0, 1)
    """
    rng = np.random.default_rng(seed)
    dt = 1 / 252
    z = rng.standard_normal(n_days)
    log_returns = (annual_drift - 0.5 * annual_vol**2) * dt + annual_vol * np.sqrt(dt) * z
    prices = 100 * np.exp(np.cumsum(log_returns))
    index = pd.bdate_range(start=start, periods=n_days)
    return pd.Series(prices, index=index, name="SYNTH")


def _validate(prices: pd.Series) -> pd.Series:
    """Basic sanity checks: sorted, unique dates, no NaNs, strictly positive."""
    prices = prices.astype(float).dropna()
    prices = prices[~prices.index.duplicated(keep="first")].sort_index()
    if (prices <= 0).any():
        raise ValueError("Found non-positive prices; data looks corrupted.")
    return prices
