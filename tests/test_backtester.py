"""
Tests. Run with:  python tests/test_backtester.py   (or `pytest` if installed)

The most important one is test_no_lookahead: it proves, for every strategy, that
the signal at day t does not change when we delete all data after day t.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from data import synthetic_prices
from engine import Backtester
from metrics import max_drawdown, sharpe_ratio
from strategy import BuyAndHold, MeanReversion, Momentum, MovingAverageCrossover


def test_no_lookahead():
    """Signal computed on prices[:t+1] must equal the full-history signal at t."""
    prices = synthetic_prices(n_days=600, seed=1)
    strategies = [
        MovingAverageCrossover(10, 30),
        MovingAverageCrossover(10, 30, long_only=False),
        MeanReversion(20, 1.0, long_only=False),
        Momentum(100, 10, long_only=False),
    ]
    for strat in strategies:
        full = strat.generate_signals(prices)
        for t in range(0, len(prices), 17):  # check a spread of cut points
            truncated = strat.generate_signals(prices.iloc[: t + 1])
            assert truncated.iloc[-1] == full.iloc[t], f"{strat.name} peeks at the future at t={t}"


def test_lookahead_test_catches_cheating():
    """Sanity-check the test itself: a strategy that peeks at tomorrow must fail it."""

    class Cheater(BuyAndHold):
        def generate_signals(self, p):
            return (p.shift(-1) > p).astype(float)  # uses tomorrow's price

    prices = synthetic_prices(n_days=200, seed=3)
    full = Cheater().generate_signals(prices)
    mismatches = sum(
        Cheater().generate_signals(prices.iloc[: t + 1]).iloc[-1] != full.iloc[t]
        for t in range(len(prices) - 1)
    )
    assert mismatches > 0


def test_costs_by_hand():
    """Hand-calculated example: prices 100, 110, 121; always long; 10 bp cost."""
    prices = pd.Series([100.0, 110.0, 121.0], index=pd.bdate_range("2020-01-01", periods=3))
    res = Backtester(commission_bps=10, slippage_bps=0, execution_delay=1).run(prices, BuyAndHold())
    # positions: [0, 1, 1]  (signal at day 0 is held from day 1)
    assert np.allclose(res.positions, [0, 1, 1])
    assert np.allclose(res.turnover, [0, 1, 0])
    assert np.allclose(res.gross_returns, [0, 0.10, 0.10])
    assert np.allclose(res.costs, [0, 0.001, 0])
    assert np.allclose(res.net_returns, [0, 0.099, 0.10])
    assert np.isclose(res.equity.iloc[-1], 1.099 * 1.10)


def test_zero_cost_net_equals_gross():
    prices = synthetic_prices(n_days=300)
    res = Backtester(0, 0).run(prices, MovingAverageCrossover(10, 30))
    assert np.allclose(res.net_returns, res.gross_returns)


def test_flip_costs_twice_flat():
    """Going long -> short trades 2 units of notional, so it costs 2x kappa."""
    prices = pd.Series([100.0] * 4, index=pd.bdate_range("2020-01-01", periods=4))

    class Flip(BuyAndHold):
        def generate_signals(self, p):
            return pd.Series([1.0, -1.0, -1.0, -1.0], index=p.index)

    res = Backtester(commission_bps=10, slippage_bps=0).run(prices, Flip())
    assert np.isclose(res.turnover.iloc[2], 2.0)  # position went 1 -> -1 on day 2


def test_max_drawdown():
    assert np.isclose(max_drawdown(pd.Series([1.0, 2.0, 1.0, 1.5])), -0.5)


def test_sharpe_constant_returns_is_nan():
    assert np.isnan(sharpe_ratio(pd.Series([0.01] * 50)))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS  {name}")
