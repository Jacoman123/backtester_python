# Backtesting Framework: Look-Ahead Bias and Transaction Costs Done Properly

A small, readable Python framework (pandas + NumPy) that runs trading strategies on historical prices and reports Sharpe ratio, drawdown and a PnL chart. The code is quite simple. 

## Quickstart

```bash
pip install -r requirements.txt
python run_demo.py --ticker SPY --start 2010-01-01 --end 2024-12-31   # real data (Yahoo Finance)
python run_demo.py --synthetic                                        # offline, fake data
python tests/test_backtester.py                                       # correctness tests
```

## Layout

| File | Responsibility |
|---|---|
| `data.py` | Download adjusted prices (Yahoo Finance, cached), plus a synthetic GBM generator for testing |
| `strategy.py` | Strategies turn prices into a *signal* (desired position). They never touch returns or costs |
| `engine.py` | Signals to positions to returns to costs to equity curve. This is where look-ahead bias is prevented |
| `metrics.py` | Sharpe, volatility, CAGR, drawdown, turnover |
| `plots.py` | Equity curve and drawdown chart |
| `run_demo.py` | Runs everything, plus the look-ahead and cost experiments |
| `tests/` | Automated look-ahead test and hand-computed cost checks |

Separating *what we want to hold* (strategy) from *what that earns* (engine) means a strategy cannot accidentally access future returns. It only ever receives prices.

---

## Design decisions

### 1. Avoiding look-ahead bias

**Look-ahead bias** means using information in a decision that would not have been available at the time. It is the most common way a backtest produces impressive, worthless results.

Let $P_t$ be the adjusted close on day $t$, and let

$$
r_t = \frac{P_t}{P_{t-1}} - 1
$$

be the return earned between the close of day $t-1$ and the close of day $t$. A strategy computes a signal $s_t \in \{-1, 0, +1\}$ at the close of day $t$ using only $\{P_u : u \le t\}$. The position actually *held* during day $t$ is

$$
w_t = s_{t-d}, \qquad d \ge 1,
$$

and the strategy's gross return is

$$
r^{\text{gross}}_t = w_t \, r_t .
$$

**Why $d \ge 1$.** To earn $r_t$ we must already hold the position at the close of day $t-1$, so the decision can only use data up to $t-1$. If we used $w_t = s_t$ (that is, $d=0$), the signal, which contains $P_t$, would be used to earn $r_t$, the very return that produced $P_t$. In code this is one line, `signals.shift(1)`, and forgetting it is the classic bug.

**Other ways look-ahead sneaks in.** None of these are present here, and you should be able to name them:

- Rolling windows with `center=True`, which use future prices.
- Normalising with full-sample statistics, for example a z-score using the mean and standard deviation of the *whole* series.
- Choosing parameters by looking at the full-sample result, which is overfitting disguised as a backtest.
- Using today's index constituents to test the past, which is survivorship bias.

**A caveat on $d = 1$.** It assumes we can trade at the same closing price we used to compute the signal, which is roughly what a market-on-close order does. A more conservative assumption is $d = 2$, trading at the next close. The demo prints Sharpe for $d \in \{0,1,2\}$, and a real edge should survive $d=2$.

**Proof, not trust.** `test_no_lookahead` checks that for every strategy, the signal at day $t$ computed on `prices[:t+1]` is identical to the signal at day $t$ computed on the full history. If a strategy used any future data, the two would differ. A second test, `test_lookahead_test_catches_cheating`, feeds it a strategy that deliberately peeks at tomorrow and confirms the check fails, so the test itself is verified.

> Look-ahead bias does not always make results *better*. On the synthetic data, the $d=0$ bug made mean reversion's Sharpe much *worse* (about $-2.0$ against $-0.2$). Mean reversion buys after a fall, and with $d=0$ it is credited with the very fall that triggered the buy. The bias makes results *unachievable*, not uniformly inflated.

### 2. Transaction costs

Every change in position is a trade, and trading is not free. There are two components, both quoted in basis points ($1\ \text{bp} = 0.01\% = 10^{-4}$) of traded notional:

- **Commission** is the explicit fee paid to the broker or exchange.
- **Slippage** is the implicit cost. You buy at the ask and sell at the bid, so you pay roughly half the bid-ask spread each way, and larger orders move the price against you.

With total cost per unit traded

$$
\kappa = \frac{\text{commission}_{\text{bps}} + \text{slippage}_{\text{bps}}}{10^4},
$$

the daily cost and net return are

$$
c_t = \kappa \, \lvert w_t - w_{t-1} \rvert, \qquad
r^{\text{net}}_t = w_t \, r_t - c_t .
$$

The quantity $\lvert w_t - w_{t-1} \rvert$ is called **turnover**. Flipping from long to short is a change of $2$, so it costs twice as much as going flat (a change of $1$), because you must sell the long *and* open the short.

The approximate annual cost drag is

$$
\text{cost drag} \approx \kappa \times \text{annual turnover},
$$

which is why **high-turnover strategies are the most sensitive to costs**. The demo sweeps the cost from $0$ to $25$ bps and prints how the Sharpe ratio decays. On the synthetic data, mean reversion trades about twice as much as the other strategies (annual turnover of roughly $13$ against $6$ to $8$), and its Sharpe falls fastest as costs rise.

**Simplifications.** Costs are a fixed proportion of notional, whereas real slippage depends on spread, volatility and order size. Positions are expressed as a fraction of current equity.

### 3. Metrics

The equity curve, starting at $E_0 = 1$, is

$$
E_t = \prod_{u=1}^{t} \left(1 + r^{\text{net}}_u\right).
$$

**Annualised Sharpe ratio** measures return per unit of risk. With $r_f$ the annual risk-free rate (default $0$),

$$
\text{SR} = \sqrt{252}\;\frac{\overline{r^{\text{net}}} - r_f/252}{\sigma\!\left(r^{\text{net}}\right)} .
$$

The $\sqrt{252}$ appears because, for independent daily returns, the mean grows linearly with time while the standard deviation grows as $\sqrt{\text{time}}$.

**Maximum drawdown** is the worst peak-to-trough loss:

$$
\text{DD}_t = \frac{E_t}{\max_{s \le t} E_s} - 1, \qquad \text{MaxDD} = \min_t \text{DD}_t .
$$

**CAGR**, with $T$ trading days, is

$$
\text{CAGR} = E_T^{\,252/T} - 1 .
$$

**Annualised volatility** is

$$
\sigma_{\text{ann}} = \sqrt{252}\;\sigma_{\text{daily}} .
$$

**How much to trust a Sharpe ratio.** For roughly independent returns, the standard error of an annualised Sharpe estimated over $Y$ years is approximately

$$
\text{SE}\big(\widehat{\text{SR}}\big) \approx \sqrt{\frac{1 + \tfrac{1}{2}\,\text{SR}^2}{Y}} .
$$

For example, $\text{SR} = 0.5$ over $Y = 10$ years gives $\text{SE} \approx 0.34$, so the estimate is only about $1.5$ standard errors from zero, which is not statistically convincing. One backtest on one asset is weak evidence, and the write-up should discuss uncertainty rather than only reporting numbers.

### 4. Strategies

All three are long-only by default (`long_only=False` enables shorting). They use conventional, round-number parameters that I did **not** tune, because tuning on the same data I report on would be overfitting.

**Moving-average crossover (trend following).** With

$$
\text{SMA}_n(t) = \frac{1}{n}\sum_{i=0}^{n-1} P_{t-i},
$$

the signal is

$$
s_t = \begin{cases} +1 & \text{if } \text{SMA}_{20}(t) > \text{SMA}_{50}(t), \\ 0 & \text{otherwise.} \end{cases}
$$

**Mean reversion.** The z-score measures how stretched the price is relative to its recent range:

$$
z_t = \frac{P_t - \mu_t}{\sigma_t},
$$

where $\mu_t$ and $\sigma_t$ are the mean and standard deviation over the last $20$ days. The strategy enters long when $z_t < -1.5$ and exits when $z_t \ge 0$. This is a state machine, because the exit rule depends on whether we are already in a trade, so it is implemented as an explicit loop.

**Momentum (time-series).** The return from $L$ days ago to $k$ days ago is

$$
\text{mom}_t = \frac{P_{t-k}}{P_{t-L}} - 1, \qquad L = 252,\quad k = 21,
$$

and the signal is

$$
s_t = \mathbf{1}\{\text{mom}_t > 0\} .
$$

The most recent month is skipped ($k = 21$) because very recent returns tend to partially reverse, which would contaminate the signal.

---

## Verification on synthetic data

Before trusting results on real data, I ran everything on prices generated by geometric Brownian motion (GBM). GBM is a random walk with drift, so it has no exploitable pattern by construction, and no strategy should show a real edge on it. If one did, that would point to a bug, usually look-ahead.

The synthetic path ($10$ years, $7\%$ drift, $18\%$ volatility, seed $42$) happens to be an unlucky draw where the price fell by about half. All strategies lost money, none showed a convincing edge, and the costs scaled with turnover as expected. That is the correct behaviour.


## Limitations

- Single asset, daily bars and fully invested, with no portfolio construction or position sizing.
- Fills at the close with fixed-bps costs. There is no bid-ask model, market impact, borrowing cost for shorts or tax.
- Parameters are not optimised, but they are also not validated out of sample. A natural next step is a train/test split or walk-forward analysis.
- There is only one historical path. Results depend heavily on the period tested, and the bull-market sample of US equities flatters long-only strategies.
- The Sharpe ratio assumes roughly independent, roughly normal returns. Real returns have fat tails and autocorrelation.

## What I learned

A tiny fee per trade looked harmless, but for a rule that trades often it adds up to a real yearly loss. The mean-reversion rule traded about 13 times its account each year, so a $0.03\%$ fee became roughly a $0.4\%$ loss per year, and its results got worse fastest as costs rose. How often a rule trades matters as much as how clever it is.

The bug the design guards against is look-ahead bias: deciding to buy using today’s closing price and then being credited with today’s price move, which is impossible in real life. It’s hard to spot because the code runs without errors and the results don’t always look suspiciously good. In my test, the peeking version actually looked worse. The fix is a single line that makes every decision take effect the next day.

Right now every trade costs the same fixed amount and happens at exactly the closing price. In reality, costs are higher when prices are jumpy or when you trade a lot at once, and orders don’t always fill at the price you expected. I’d also test the rules on a period of years I didn’t use to choose their settings, so the results aren’t just fitted to the past.

When you can’t check the output by eye, check something that must always be true. A rule’s decision about a given day shouldn’t change if you delete all the prices after that day, so the test compares the two for every rule. I also wrote a rule that cheats on purpose to confirm the test catches it, so I know the test itself works.
## Next steps

- Profile the engine and rewrite the bottleneck (likely the mean-reversion loop) in C++ via pybind11, then benchmark the speedup.
- Walk-forward parameter selection with out-of-sample evaluation.
- Multi-asset portfolios, volatility-based position sizing and a bid-ask spread cost model.
