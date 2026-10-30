"""
strategy.py - Trading strategies.

THE ONE RULE OF THIS FILE
-------------------------
A strategy turns a price history into a *signal*: the position we WANT.

    signal[t] in {-1, 0, +1}     (-1 = short, 0 = flat, +1 = long)

signal[t] may only use prices up to and including day t. It must never peek at
t+1 or later. The engine then delays the signal before trading on it (see
engine.py), and tests/test_backtester.py checks this rule automatically.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class Strategy(ABC):
    """Base class: every strategy implements generate_signals()."""

    name: str = "strategy"

    @abstractmethod
    def generate_signals(self, prices: pd.Series) -> pd.Series:
        """Return the desired position at the close of each day, using only past data."""


class BuyAndHold(Strategy):
    """Always long. Used as the benchmark, run through the same engine (so it
    pays the same costs and is compared like-for-like)."""

    name = "Buy & Hold"

    def generate_signals(self, prices: pd.Series) -> pd.Series:
        return pd.Series(1.0, index=prices.index)


class MovingAverageCrossover(Strategy):
    """Trend following: be long when the fast average is above the slow average.

    Intuition: a fast moving average reacts quickly to recent prices, a slow one
    reacts slowly. When fast > slow, recent prices are higher than the longer-run
    average, i.e. the price is trending up.

        SMA_n(t) = (P_t + P_{t-1} + ... + P_{t-n+1}) / n
        signal   = +1 if SMA_fast(t) > SMA_slow(t), else 0 (or -1 if shorting)
    """

    def __init__(self, fast: int = 20, slow: int = 50, long_only: bool = True):
        if fast >= slow:
            raise ValueError("fast window must be smaller than slow window")
        self.fast, self.slow, self.long_only = fast, slow, long_only
        self.name = f"MA Crossover ({fast}/{slow})"

    def generate_signals(self, prices: pd.Series) -> pd.Series:
        # rolling() looks BACKWARDS by default, so the value at t uses P_t and earlier only.
        # (Never use center=True: that would use future prices.)
        fast_ma = prices.rolling(self.fast).mean()
        slow_ma = prices.rolling(self.slow).mean()

        off_value = 0.0 if self.long_only else -1.0
        signal = pd.Series(np.where(fast_ma > slow_ma, 1.0, off_value), index=prices.index)
        signal[slow_ma.isna()] = 0.0  # warm-up period: not enough history, stay flat
        return signal


class MeanReversion(Strategy):
    """Bet that prices which stray far from their recent average snap back.

    The z-score measures how many standard deviations the price is from its
    rolling mean:

        z_t = (P_t - mu_t) / sigma_t

    Enter long when z < -entry_z (unusually cheap), exit when z >= -exit_z
    (back near average). Mirror logic for shorts. This needs a *state machine*
    (the exit rule depends on whether we are already in a trade), so we loop.
    """

    def __init__(
        self,
        window: int = 20,
        entry_z: float = 1.5,
        exit_z: float = 0.0,
        long_only: bool = True,
    ):
        if exit_z >= entry_z:
            raise ValueError("exit_z must be smaller than entry_z")
        self.window, self.entry_z, self.exit_z, self.long_only = window, entry_z, exit_z, long_only
        self.name = f"Mean Reversion (w={window}, z={entry_z})"

    def generate_signals(self, prices: pd.Series) -> pd.Series:
        mean = prices.rolling(self.window).mean()
        std = prices.rolling(self.window).std().replace(0.0, np.nan)  # avoid divide-by-zero
        z = ((prices - mean) / std).to_numpy()

        signal = np.zeros(len(prices))
        position = 0.0
        for i, zi in enumerate(z):
            if np.isnan(zi):
                position = 0.0  # warm-up: no opinion yet
            elif position == 0.0:
                if zi < -self.entry_z:
                    position = 1.0  # unusually cheap -> buy
                elif zi > self.entry_z and not self.long_only:
                    position = -1.0  # unusually expensive -> short
            elif position == 1.0 and zi >= -self.exit_z:
                position = 0.0  # reverted to the mean -> take profit
            elif position == -1.0 and zi <= self.exit_z:
                position = 0.0
            signal[i] = position

        return pd.Series(signal, index=prices.index)


class Momentum(Strategy):
    """Time-series momentum: assets that went up over the past months tend to
    keep going up for a while (a well-documented, though modest, effect).

        mom_t = P_{t-skip} / P_{t-lookback} - 1
        signal = +1 if mom_t > 0, else 0 (or -1 if shorting)

    We skip the most recent `skip` days because very recent returns tend to
    partially reverse (short-term reversal), which would pollute the signal.
    The classic "12-1" setup is lookback = 252 days, skip = 21 days.
    """

    def __init__(self, lookback: int = 252, skip: int = 21, long_only: bool = True):
        if skip >= lookback:
            raise ValueError("skip must be smaller than lookback")
        self.lookback, self.skip, self.long_only = lookback, skip, long_only
        self.name = f"Momentum ({lookback}-{skip})"

    def generate_signals(self, prices: pd.Series) -> pd.Series:
        # shift(k) moves data FORWARD in time, so shift(k) at day t holds the price of day t-k. Past only.
        mom = prices.shift(self.skip) / prices.shift(self.lookback) - 1.0

        off_value = 0.0 if self.long_only else -1.0
        signal = pd.Series(np.where(mom > 0, 1.0, off_value), index=prices.index)
        signal[mom.isna()] = 0.0  # warm-up period
        return signal
