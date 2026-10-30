"""
engine.py - The backtest engine: signals -> positions -> returns -> equity curve.

Notation (all indexed by trading day t)
---------------------------------------
    r_t      asset return from close t-1 to close t:   P_t / P_{t-1} - 1
    s_t      signal decided at the close of day t, using data up to day t
    w_t      position actually HELD during day t (what earns r_t)
    kappa    cost per unit of traded notional (commission + slippage)

Core equations
--------------
    w_t        = s_{t - d}                      d = execution_delay >= 1
    gross_t    = w_t * r_t
    cost_t     = kappa * |w_t - w_{t-1}|
    net_t      = gross_t - cost_t
    E_t        = prod_{u <= t} (1 + net_u)      equity curve, E_0 = 1

Why the delay prevents look-ahead bias
--------------------------------------
r_t is earned between close t-1 and close t. To earn it we must already hold
the position at close t-1, so the decision can only use information available
at close t-1 or earlier. With d = 1:  w_t = s_{t-1}  -> decide at close t-1
(using P_{t-1} and earlier), hold through day t. Without the shift, s_t (which
uses P_t) would be used to earn r_t, the very return that produced P_t.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import pandas as pd

from strategy import Strategy


@dataclass
class BacktestResult:
    """Everything produced by one backtest, kept as time series for inspection."""

    name: str
    signals: pd.Series  # s_t: desired position, decided at close t
    positions: pd.Series  # w_t: position held during day t
    turnover: pd.Series  # |w_t - w_{t-1}|
    gross_returns: pd.Series  # w_t * r_t
    costs: pd.Series  # kappa * turnover
    net_returns: pd.Series  # gross - costs
    equity: pd.Series  # growth of 1 unit of capital, after costs


class Backtester:
    """Vectorised daily backtester for a single asset.

    Parameters
    ----------
    commission_bps : explicit broker/exchange fee, in basis points of traded notional.
                     (1 bp = 0.01% = 0.0001)
    slippage_bps   : implicit cost of trading: you buy at the ask and sell at the bid, and
                     large orders move the price. Also in bps of traded notional.
    execution_delay: days between computing a signal and holding the position.
                     1 = decide at close t, earn from day t+1 onward (the minimum valid value).
                     2 = more conservative (trade at the next day's close).
                     0 = LOOK-AHEAD BUG, only allowed for demonstration.
    """

    def __init__(
        self,
        commission_bps: float = 1.0,
        slippage_bps: float = 2.0,
        execution_delay: int = 1,
    ):
        if execution_delay < 0:
            raise ValueError("execution_delay cannot be negative")
        if execution_delay == 0:
            warnings.warn(
                "execution_delay=0 uses today's information to earn today's return: "
                "this is LOOK-AHEAD BIAS. Results are not achievable in practice.",
                stacklevel=2,
            )
        self.kappa = (commission_bps + slippage_bps) / 10_000
        self.execution_delay = execution_delay

    def run(self, prices: pd.Series, strategy: Strategy) -> BacktestResult:
        if not prices.index.is_monotonic_increasing:
            raise ValueError("prices must be sorted by date ascending")

        # r_t. The first day has no previous close, so its return is 0.
        asset_returns = prices.pct_change().fillna(0.0)

        # s_t, then w_t = s_{t-d}. Before the first delayed signal arrives we hold nothing.
        signals = strategy.generate_signals(prices)
        positions = signals.shift(self.execution_delay).fillna(0.0)

        # Turnover = how much we trade each day. Going long -> short is a change of 2 (sell the
        # long AND sell short), so it costs twice as much as going long -> flat.
        turnover = positions.diff().abs()
        turnover.iloc[0] = abs(positions.iloc[0])  # first trade starts from a flat book

        gross = positions * asset_returns
        costs = self.kappa * turnover
        net = gross - costs
        equity = (1.0 + net).cumprod()

        return BacktestResult(
            name=strategy.name,
            signals=signals,
            positions=positions,
            turnover=turnover,
            gross_returns=gross,
            costs=costs,
            net_returns=net,
            equity=equity,
        )
