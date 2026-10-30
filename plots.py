"""
plots.py - Equity curve (PnL) and drawdown charts.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render to files; no display needed
import matplotlib.pyplot as plt

from engine import BacktestResult
from metrics import drawdown_series


def plot_results(
    results: list[BacktestResult],
    title: str = "Backtest results (net of costs)",
    path: str | None = None,
) -> None:
    """Top panel: growth of 1 unit of capital (PnL). Bottom panel: drawdowns."""
    fig, (ax_eq, ax_dd) = plt.subplots(
        2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1.5]}
    )

    for res in results:
        style = {"color": "black", "linestyle": "--", "linewidth": 1.2} if res.name == "Buy & Hold" else {}
        ax_eq.plot(res.equity.index, res.equity, label=res.name, **style)
        ax_dd.plot(res.equity.index, 100 * drawdown_series(res.equity), **style)

    ax_eq.set_title(title)
    ax_eq.set_ylabel("Equity (start = 1, log scale)")
    ax_eq.set_yscale("log")  # equal vertical distance = equal % change
    ax_eq.grid(alpha=0.3)
    ax_eq.legend()

    ax_dd.set_ylabel("Drawdown (%)")
    ax_dd.set_xlabel("Date")
    ax_dd.grid(alpha=0.3)

    fig.tight_layout()
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
    plt.close(fig)
