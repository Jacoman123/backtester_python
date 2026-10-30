"""
run_demo.py - Run all strategies, print metrics, save the PnL chart, and run
the two experiments that go in the write-up (look-ahead and cost sensitivity).

Usage:
    python run_demo.py --ticker SPY --start 2010-01-01 --end 2024-12-31
    python run_demo.py --synthetic            # offline test on fake GBM data
"""
from __future__ import annotations

import argparse
import warnings

import pandas as pd

from data import load_prices, synthetic_prices
from engine import Backtester
from metrics import sharpe_ratio, summarise
from plots import plot_results
from strategy import BuyAndHold, MeanReversion, MovingAverageCrossover, Momentum

pd.options.display.float_format = "{:,.3f}".format
pd.options.display.width = 200
pd.options.display.max_columns = 20


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="SPY")
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--synthetic", action="store_true", help="use fake GBM prices (offline)")
    args = parser.parse_args()

    prices = synthetic_prices() if args.synthetic else load_prices(args.ticker, args.start, args.end)
    label = "SYNTHETIC" if args.synthetic else args.ticker
    print(f"{label}: {len(prices)} days, {prices.index[0].date()} -> {prices.index[-1].date()}\n")

    strategies = [
        BuyAndHold(),
        MovingAverageCrossover(20, 50),
        MeanReversion(window=20, entry_z=1.5),
        Momentum(lookback=252, skip=21),
    ]

    # ---- 1. Main results (1 bp commission + 2 bp slippage, delay = 1) ----
    bt = Backtester(commission_bps=1.0, slippage_bps=2.0, execution_delay=1)
    results = [bt.run(prices, s) for s in strategies]
    table = pd.DataFrame([summarise(r) for r in results]).set_index("Strategy")
    print("=== Main results (net of 3 bp costs, delay = 1) ===")
    print(table, "\n")
    out_path = "outputs/equity_synthetic.png" if args.synthetic else "outputs/equity.png"
    plot_results(results, title=f"{label}: strategies vs buy & hold (net of costs)", path=out_path)
    print(f"Saved {out_path}\n")

    # ---- 2. Look-ahead experiment: what does 'cheating' look like? ----
    print("=== Sharpe vs execution delay (0 = look-ahead bug) ===")
    rows = {}
    for s in strategies[1:]:
        row = {}
        for d in (0, 1, 2):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                row[f"delay={d}"] = sharpe_ratio(Backtester(execution_delay=d).run(prices, s).net_returns)
        rows[s.name] = row
    print(pd.DataFrame(rows).T, "\n")

    # ---- 3. Cost sensitivity: how fast do costs destroy the edge? ----
    print("=== Sharpe vs total cost per trade (bps of traded notional) ===")
    rows = {}
    for s in strategies[1:]:
        rows[s.name] = {
            f"{bps} bp": sharpe_ratio(
                Backtester(commission_bps=0, slippage_bps=bps).run(prices, s).net_returns
            )
            for bps in (0, 2, 5, 10, 25)
        }
    print(pd.DataFrame(rows).T)


if __name__ == "__main__":
    main()
