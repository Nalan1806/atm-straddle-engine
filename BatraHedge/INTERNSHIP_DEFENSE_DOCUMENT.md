# Internship Preparation & Defense Document

## Intraday NIFTY ATM Straddle Backtesting Engine

**Candidate: Batra**
**Date: February 2026**
**Instrument: NIFTY Index Options (3-minute candles)**
**Period: August 2024 – February 2026 (369 trading days, ~2M rows)**

---

# Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Concept Clarity Section](#2-concept-clarity-section)
3. [Strategy Architecture Explanation](#3-strategy-architecture-explanation)
4. [Visuals & What They Demonstrate](#4-visuals--what-they-demonstrate)
5. [Risk Discussion Section](#5-risk-discussion-section)
6. [Interview Defense Section](#6-interview-defense-section)
7. [Business Perspective](#7-business-perspective)
8. [Final Self-Evaluation](#8-final-self-evaluation)

---

# 1. Executive Summary

## What Was Built

A production-quality intraday backtesting engine for NIFTY ATM (at-the-money) straddle strategies, written in Python with a Streamlit UI. The system ingests ~2 million rows of 3-minute options candle data, simulates full round-trip trades with realistic transaction costs (brokerage, slippage, STT), and produces institutional-grade analytics: Sharpe ratios, drawdown profiles, profit factors, Monte Carlo stress tests, rolling window stability checks, and regime-aware validation.

## What Was Discovered

The **long straddle** (buying volatility) has **no structural edge** in intraday NIFTY options. Exhaustive testing across parameter grids yielded a mean Sharpe of **-11.40** — a definitively negative result. IV contraction after market open and theta bleed over a 5-hour holding window destroy the premium paid.

The **short straddle** (selling volatility) — when paired with a **spot breakout stop** — produces a meaningful, cost-surviving edge. However, the edge is parameter-sensitive: at 0.2% breakout buffer the strategy barely breaks even (Sharpe 0.67, with 86.5% of gross alpha consumed by costs), while at 0.5% buffer the same engine delivers Sharpe 4.64, ROM 24.73%, and maximum drawdown of only 0.31% on capital.

## Why the Pivot Happened

The long straddle was the initial hypothesis — compressed opening ranges should predict volatility expansion. Data rejected this hypothesis categorically. Rather than abandon the infrastructure, the same engine was inverted to test the mirror trade: selling the straddle whose premium systematically bleeds in the buyer's disfavour. The pivot preserved 100% of the codebase (data loader, cost model, metrics, UI) and required only a new exit logic module.

## Final Result

| Metric | Value |
|---|---|
| Strategy | Short ATM Straddle with Breakout Stop |
| Net Sharpe (annualised) | 4.64 |
| Gross Sharpe (annualised) | 7.83 |
| Profit Factor | 2.12 |
| Annualised ROM | 24.73% |
| Max Drawdown (on ₹10L capital) | -0.31% |
| Win Rate | 66.7% |
| Total Trades | 249 over 369 days |
| Worst Single Day | ₹-1,664 (-1.79% of margin) |
| Rolling 6-month Sharpe | All windows > 0 (range 2.50 – 6.02) |
| Monte Carlo P99 Drawdown | -6.41% of margin |
| Monte Carlo: % paths profitable | 100.0% |
| Production Validation Score | 10/10 |
| Regime Stress Score | 10/10 |

## Remaining Risks

- **Liquidity collapse** during extreme events (flash crash, circuit breaker) cannot be modelled from historical candle data — slippage would spike non-linearly.
- **Structural regime change** — if NIFTY transitions to a persistently trending market (e.g., sustained unidirectional moves over months), the proportion of breakout-stop exits would increase, compressing the edge.
- **Execution risk** — the strategy assumes fills at the close of the 09:30 candle; in live trading, queue position and order-book depth may differ.
- **Regulatory risk** — changes to STT, margin requirements, or lot sizes would directly alter the cost structure.
- **Sample size** — 369 trading days (~18 months) is adequate for intraday but short for capturing multi-year macro cycles.

---

# 2. Concept Clarity Section

## 2.1 Sharpe Ratio

**Definition:** The Sharpe ratio measures the risk-adjusted return of a strategy — how much excess return you earn per unit of volatility.

**Formula:**

$$
\text{Sharpe} = \frac{\bar{r}}{\sigma_r} \times \sqrt{252}
$$

Where $\bar{r}$ is the mean daily return, $\sigma_r$ is the standard deviation of daily returns, and $\sqrt{252}$ annualises from daily to yearly.

**Why it matters:** Raw return is meaningless without context. A strategy returning 25% with 5% volatility is fundamentally different from one returning 25% with 50% volatility. Sharpe normalises for this. In institutional contexts, Sharpe > 2.0 is considered strong; Sharpe > 3.0 is exceptional.

**How it applied in this project:** The long straddle produced Sharpe -11.40 (catastrophic). The short straddle at 0.2% buffer produced Sharpe 0.67 (marginal). At 0.5% buffer, Sharpe jumped to 4.64 — revealing that the breakout buffer width is the single most important parameter in the system. We computed both **net Sharpe** (after all costs) and **gross Sharpe** (before costs) to quantify cost drag. The gap (7.83 gross vs 4.64 net) shows that 41% of risk-adjusted performance is consumed by transaction costs.

---

## 2.2 Profit Factor

**Definition:** The ratio of total gross profits from winning trades to total gross losses from losing trades.

**Formula:**

$$
\text{PF} = \frac{\sum \text{Winning Trades (net P\&L)}}{\left|\sum \text{Losing Trades (net P\&L)}\right|}
$$

**Why it matters:** Profit Factor tells you how many rupees you make for every rupee you lose. PF = 1.0 is breakeven. PF > 1.5 is a solid edge. PF < 1.0 means the strategy destroys capital.

**How it applied:** At 0.2% buffer, PF was 1.11 — barely above breakeven and vulnerable to any increase in costs. At 0.5% buffer, PF rose to 2.12, meaning the strategy earns ₹2.12 for every ₹1.00 it loses. This margin of safety makes the edge robust to moderate cost increases.

---

## 2.3 Return on Margin (ROM)

**Definition:** The annualised net P&L divided by the margin required to hold the position.

**Formula:**

$$
\text{ROM}_{\text{ann}} = \frac{\text{Net P\&L} / \text{Years}}{\text{Required Margin}} \times 100\%
$$

**Why it matters:** In derivatives trading, you don't deploy full notional capital — you post margin. ROM measures how efficiently your locked-up margin generates returns. It is the metric that matters for capital allocation decisions.

**How it applied:** Estimated NIFTY short straddle margin is ~₹93,191 per lot. With annualised net P&L of ~₹23,000, ROM is 24.73%. This means for every ₹1 lakh of margin posted, the strategy generates approximately ₹25,000 per year — a compelling margin efficiency.

---

## 2.4 CAGR (Compound Annual Growth Rate)

**Definition:** The smoothed annualised rate of return, assuming profits are reinvested.

**Formula:**

$$
\text{CAGR} = \left(\frac{\text{Ending Equity}}{\text{Starting Equity}}\right)^{1/\text{Years}} - 1
$$

**Why it matters:** CAGR removes the noise of uneven returns across time periods. It answers: "If growth were perfectly smooth, what annual rate would produce the same final equity?"

**How it applied:** Over the 18-month backtest period, the strategy's CAGR on total capital (₹10L) is modest (~2.4%) because capital utilisation is low — only one lot deployed on ₹10L. The meaningful measure is ROM on margin, not CAGR on total capital. This distinction is critical in derivatives: CAGR on notional understates the efficiency of a margin-based strategy.

---

## 2.5 Drawdown (DD)

**Definition:** The peak-to-trough decline in equity before a new high is established.

**Formula:**

$$
\text{DD}_t = \frac{\text{Equity}_t - \max_{s \le t}(\text{Equity}_s)}{\max_{s \le t}(\text{Equity}_s)}
$$

$$
\text{Max DD} = \min_t(\text{DD}_t)
$$

**Why it matters:** Drawdown measures the worst pain an investor experiences. High returns mean nothing if the path to get there involves a 40% drawdown — most investors (and risk managers) would force-liquidate before recovery. Max DD is the single most important risk metric for capital preservation.

**How it applied:** The strategy's max drawdown on capital is -0.31% (₹3,100 on ₹10L). On margin, the worst drawdown is approximately -3.3%. Even under Monte Carlo stress (10,000 path reshuffles), the P99 drawdown is only -6.41% of margin. This is exceptionally shallow for a strategy generating 25% ROM.

---

## 2.6 Maximum Adverse Excursion (MAE)

**Definition:** The worst unrealised loss (mark-to-market) experienced during a single trade before eventual exit.

**Formula:**

$$
\text{MAE}_i = \max_{t \in \text{trade}_i} \left[(\text{Premium}_t - \text{Premium}_{\text{entry}}) \times \text{Lot Size}\right]
$$

(For a short straddle, MAE is the maximum premium increase — i.e., the worst unrealised loss — during the trade.)

**Why it matters:** MAE reveals whether a strategy survives "right but early" situations. If a trade's MAE is ₹5,000 but final P&L is +₹200, the strategy was nearly stopped out. High MAE relative to final P&L indicates fragile edge.

**How it applied:** We tracked MAE per-trade and plotted its distribution. Average MAE for the short straddle is modest, and the MAE distribution showed no extreme tail — confirming that the breakout stop cuts losing trades before adverse excursion becomes catastrophic.

---

## 2.7 Rolling Window Analysis

**Definition:** Computing a metric (e.g., Sharpe) over a sliding window of fixed length (e.g., 126 trading days ≈ 6 months) to test temporal stability.

**Why it matters:** An aggregate Sharpe of 4.6 could hide six great months followed by six catastrophic months. Rolling windows expose whether the edge is persistent or episodic.

**How it applied:** All rolling 6-month Sharpe windows were positive (range: 2.50 to 6.02). No single window dipped below 2.0. This provides strong evidence that the edge is structural and time-stable, not driven by a handful of lucky days.

---

## 2.8 Monte Carlo Reshuffling

**Definition:** Randomly permuting the order of daily P&L outcomes (10,000 times) to test whether the strategy's drawdown profile is sensitive to the specific historical sequencing.

**Why it matters:** A strategy might have low drawdown only because losses happened to be spread out. Reshuffling reveals the *distribution* of possible drawdown experiences given the same set of daily outcomes.

**How it applied:** Across 10,000 reshuffled paths:
- Mean max drawdown: -0.322% on capital
- P99 max drawdown: -0.583% on capital (-6.41% on margin)
- Worst observed: -0.915% on capital (-10.11% on margin)
- 100% of paths ended profitable

The worst Monte Carlo drawdown (-10.11% of margin) is still manageable — well within typical risk budgets.

---

## 2.9 Synthetic Stress Testing

**Definition:** Artificially increasing the proportion of adverse market conditions (e.g., trending days) beyond historical norms to test strategy resilience.

**Why it matters:** Historical data may underrepresent extreme conditions. Synthetic stress answers: "What if the market environment shifts structurally against me?"

**How it applied:** We increased high-trend days from 21.7% (historical) to 30% (synthetic worst case). Sharpe dropped from 4.64 to 4.45 — a degradation of only 4.1%. The breakout stop mechanically limits loss on trending days, making the strategy naturally regime-resilient.

---

## 2.10 Cost Sensitivity

**Definition:** Measuring how much of the gross edge survives after applying realistic transaction costs (brokerage, slippage, STT/taxes).

**Why it matters:** Many backtested strategies look profitable pre-cost but are destroyed by real-world friction. Cost sensitivity tells you how robust the edge is to execution quality.

**How it applied:** At 0.2% breakout buffer, **86.5% of gross alpha was consumed by costs** — leaving a razor-thin net edge (Sharpe 0.67). Widening to 0.5% buffer increased gross alpha (by reducing false breakout exits) to the point where costs consumed only ~41% of gross alpha (Sharpe: 7.83 gross → 4.64 net). This analysis drove the central design decision of the project.

---

## 2.11 Volatility Regime Shift

**Definition:** A persistent change in the statistical properties of market volatility — for example, transitioning from a low-volatility, range-bound market to a high-volatility, trending market.

**Why it matters:** A strategy calibrated on calm markets may fail catastrophically in volatile markets. Short straddles are particularly vulnerable because trending markets increase loss frequency.

**How it applied:** Regime detection found 80/369 days (21.7%) were high-trend (>1% intraday range). We tested an explicit regime filter (skip trading on high-vol days), but it **hurt performance** — the breakout stop already acts as a mechanical regime filter by exiting trades on trending days. Adding an explicit filter removed 35% of trades, most of which were profitable, cutting Sharpe from 4.64 to 2.31.

---

## 2.12 Tail Risk

**Definition:** The risk of extreme, low-probability events that sit in the far tails of the return distribution.

**Why it matters:** For short volatility strategies, tail risk is existential. A single 5-sigma event can eliminate months of accumulated profits.

**How it applied:** The worst single-day loss was ₹-1,664 (-1.79% of margin). Even under Monte Carlo stress, the worst simulated equity trough was ₹-9,421 (-10.11% of margin). The breakout stop mechanically caps tail exposure by exiting when spot breaches the opening range + buffer.

---

## 2.13 Liquidity Deterioration

**Definition:** A sudden reduction in market depth and increase in bid-ask spreads, typically during extreme events, making it difficult or costly to exit positions.

**Why it matters:** Backtests assume fills at observed prices. During liquidity crises, the actual fill price may be vastly worse — "slippage" becomes non-linear. A strategy that relies on timely exits (like our breakout stop) is vulnerable if the exit order cannot be filled.

**How it applied:** Our cost model includes a flat 0.05% slippage assumption. In reality, during events like flash crashes or circuit breakers, slippage could be 10-50x this level. This is the **single largest unquantifiable risk** in the strategy and is explicitly acknowledged as a limitation. No historical backtest can model liquidity collapse because it is, by definition, an absence of orderly market data.

---

# 3. Strategy Architecture Explanation

## 3.1 Long Straddle Logic

**Hypothesis:** Buy ATM CE + PE when the opening range (09:15–09:30) is compressed, expecting volatility expansion during the trading day.

**Entry conditions:**
1. **Compression filter** — Combined premium range during 09:15–09:30 must be < 0.4% of spot (indicating a "coiled spring")
2. **IV momentum** — Implied volatility at 09:30 must be higher than at 09:15 (confirming demand for options)
3. **Entry** — Buy CE + PE at 09:30 candle close

**Exit conditions:**
1. Premium rises by profit target (10%) → exit with profit
2. Premium drops by stop loss (6%) → exit with loss
3. Time exit at configured time → exit at market price

### Why It Failed

The long straddle failed comprehensively (mean Sharpe **-11.40** across all parameter combinations) for two structural reasons:

1. **IV contraction after open** — Implied volatility is typically elevated at market open due to overnight uncertainty. Once the market establishes direction, IV contracts. Since the long straddle *buys* options, IV contraction directly reduces premium value — the position loses money even if the underlying moves, because the IV drop offsets the directional gain.

2. **Theta bleed** — Holding bought options for 5 hours (09:30 to 14:30) costs roughly 1/4 of a day's theta. For near-expiry NIFTY options, this is a significant premium erosion. The time decay is a *structural headwind* that the strategy must overcome before it can profit.

These are not parameter-tuning problems — they are structural features of intraday options markets. No combination of entry filter, profit target, or stop loss produced a positive Sharpe.

---

## 3.2 Short Straddle Logic

**Hypothesis:** Sell ATM CE + PE at 09:30, harvesting the same IV contraction and theta bleed that destroyed the long straddle.

**Entry:**
- Sell CE + PE at 09:30 candle close (premium received)
- Optional compression filter (prevents entry on already-volatile days)

**Exit hierarchy (first trigger wins):**
1. **Premium stop-loss** — If combined premium rises by 20% of entry (loss for seller) → exit
2. **Profit booking** — If combined premium drops by 10% of entry (profit for seller) → exit
3. **Spot breakout stop** — If spot moves outside the 09:15–09:30 range ± buffer → exit
4. **Time exit** — Hard exit at 14:30

### Why Breakout Buffer Matters

The breakout stop is the **primary risk management mechanism**. It exits the trade when the underlying spot price breaches the morning range — signalling that a directional move (trending day) is underway, which is the worst-case scenario for a short straddle.

The buffer prevents **false exits** on normal intraday noise. Too tight a buffer triggers on random fluctuations; too wide allows genuine trending days to accumulate large losses before exit.

### Why 0.2% Failed

At 0.2% buffer, the breakout stop triggered on **normal market noise** — minor fluctuations that would have mean-reverted. This caused:
- Excessive exit frequency (too many breakout stops on non-trending days)
- Each false exit incurred full round-trip transaction costs
- 86.5% of gross alpha was consumed by costs from these unnecessary exits
- Net Sharpe collapsed to 0.67

### Why 0.5% Fixed Buffer Works

At 0.5% buffer, the stop only triggers on **genuine directional moves**. This:
- Reduced breakout stop exits to only truly trending days
- Preserved profitable time-exit and profit-target trades
- Allowed gross alpha to grow faster than cost drag
- Net Sharpe jumped to 4.64 (a 7× improvement)

### Why Premium Stop-Loss Became Irrelevant

At 0.5% buffer, the premium stop-loss (20%) **never fires** — because the breakout stop exits the trade before premium can rise 20%. The breakout stop is a tighter, more responsive risk control that dominates the premium stop. The premium stop exists as a safety net but is effectively redundant.

### Why the Regime Filter Hurt Performance

An explicit regime filter (skip trading on days with wide opening range, high rolling volatility, or rising IV) was tested. It removed 35% of trades — but most of the removed trades were **profitable**. The breakout stop already handles trending days by exiting quickly, so the regime filter's additional screening only removes good trades. Sharpe dropped from 4.64 to 2.31. The breakout stop IS the regime filter — in mechanical form.

### Why Adaptive Buffer Adds Intelligence

A volatility-adaptive breakout buffer scales with recent market conditions:

$$
\text{buffer} = \text{clamp}\bigl(k \times \overline{\text{range}}_N,\; \text{min},\; \text{max}\bigr)
$$

Where $\overline{\text{range}}_N$ is the mean of the last $N$ days' intraday range %, $k = 0.6$, min = 0.3%, max = 0.8%.

In calm markets (low recent ranges), the buffer tightens — capturing more theta and IV bleed. In volatile markets (high recent ranges), the buffer widens — giving more room before stop triggers. The adaptive buffer produced mean buffer of 0.462% with a distribution from 0.300% to 0.800%.

With default parameters, the adaptive buffer slightly underperforms fixed 0.5% on Sharpe (4.38 vs 4.64) but **improves worst-trade risk** (₹-1,413 vs ₹-1,664) and peak-trade capture (₹2,215 vs ₹1,951). Both modes remain solidly profitable with all rolling 6-month Sharpe windows positive.

---

## 3.3 Strategy Flow Diagram

```
                    ┌──────────────────────────┐
                    │    MARKET OPEN 09:15      │
                    └────────────┬─────────────┘
                                 │
                    ┌────────────▼─────────────┐
                    │  Observe Opening Range    │
                    │  09:15 – 09:30            │
                    │  Record: Spot High, Low   │
                    │  Record: CE/PE premiums   │
                    └────────────┬─────────────┘
                                 │
                    ┌────────────▼─────────────┐
                    │  Compression Filter       │
                    │  Premium range < 0.4%?    │
                    └──────┬───────────┬───────┘
                           │ YES       │ NO
                           │           └──────► SKIP DAY
                    ┌──────▼───────────────────┐
                    │  ENTRY at 09:30 close     │
                    │  Sell ATM CE + PE          │
                    │  Record entry premium     │
                    └──────────────┬────────────┘
                                   │
                    ┌──────────────▼────────────┐
                    │  Compute Breakout Levels   │
                    │                            │
                    │  If Adaptive Enabled:      │
                    │    buffer = clamp(          │
                    │      k × avg_range_N,      │
                    │      min_buf, max_buf)      │
                    │  Else:                      │
                    │    buffer = fixed 0.5%      │
                    │                            │
                    │  Upper = range_high*(1+buf) │
                    │  Lower = range_low*(1-buf)  │
                    └──────────────┬────────────┘
                                   │
              ┌────────────────────▼────────────────────┐
              │          WALK FORWARD BAR-BY-BAR        │
              │          (3-minute candles)              │
              └──┬─────────┬──────────┬────────────┬───┘
                 │         │          │            │
          ┌──────▼──┐ ┌────▼────┐ ┌───▼──────┐ ┌──▼──────┐
          │ Premium │ │ Profit  │ │ Spot     │ │ Time    │
          │ Stop    │ │ Target  │ │ Breakout │ │ Exit    │
          │ ≥+20%   │ │ ≤-10%   │ │ Outside  │ │ ≥14:30  │
          │ (loss)  │ │ (gain)  │ │ range±   │ │         │
          └────┬────┘ └────┬────┘ └────┬─────┘ └────┬────┘
               │           │           │            │
               └───────────┴───────────┴────────────┘
                                   │
                    ┌──────────────▼────────────┐
                    │  EXIT: Buy back CE + PE   │
                    │  Compute gross P&L        │
                    │  Apply transaction costs   │
                    │  Record net P&L            │
                    └───────────────────────────┘
```

---

# 4. Visuals & What They Demonstrate

This section describes the key visualisations produced by the system and what each reveals.

## 4.1 Equity Curve (Long vs Short)

**What it shows:** Cumulative P&L over time for both strategies on the same capital base.

**What it demonstrates:**
- The long straddle equity curve slopes relentlessly downward — every dollar of premium paid erodes over the trading day. There is no period of sustained profitability.
- The short straddle equity curve slopes upward with shallow, brief drawdowns. Growth is steady and monotonic at the weekly scale.
- The visual makes the structural edge (or lack thereof) immediately obvious — no amount of parameter tuning could rescue the long straddle's shape.

## 4.2 Distribution of Daily P&L

**What it shows:** A histogram of net P&L per trading day.

**What it demonstrates:**
- The short straddle P&L distribution is right-skewed (more wins than losses) with truncated left tail (breakout stop caps losses).
- Mean daily P&L is positive (~₹143/day).
- The left tail does not extend beyond ₹-1,664, confirming the breakout stop's effectiveness as a loss limiter.
- For a short volatility strategy, the *absence* of extreme left-tail events is the most important feature.

## 4.3 Rolling 6-Month Sharpe Plot

**What it shows:** A time-series of Sharpe ratio computed over a sliding 126-trading-day window.

**What it demonstrates:**
- All windows remain above 2.0, with mean at 4.49.
- There is no period where the strategy "breaks" — the edge is temporally stable.
- Seasonal variation exists (some months better than others) but always in profitable territory.
- This is the strongest evidence against overfitting: if the strategy were curve-fit to a specific period, out-of-window Sharpe would collapse.

## 4.4 Sensitivity Heatmap

**What it shows:** Performance metrics (Sharpe, PF, Win Rate) across a grid of stop-loss, profit target, and exit time combinations. Presented as a formatted table in the Streamlit robustness grid.

**What it demonstrates:**
- The strategy is **not sensitive to stop-loss level** — because the breakout stop dominates before premium stop fires.
- Moderate sensitivity to exit time (14:00 vs 14:30) — later exits capture more theta decay.
- Profit target has modest impact — most trades exit via breakout or time stop.
- The heat gradient shows a "plateau" of good performance, not a "spike" — ruling out lucky parameter choice.

## 4.5 Breakout Buffer Comparison Chart

**What it shows:** Key metrics (Sharpe, PF, Max DD, ROM) for fixed 0.2%, fixed 0.5%, and adaptive buffer.

| Metric | 0.2% Buffer | 0.5% Buffer | Adaptive (default) |
|---|---|---|---|
| Net Sharpe | 0.67 | 4.64 | 4.38 |
| Profit Factor | 1.11 | 2.12 | 2.06 |
| Max DD (capital) | deeper | -0.31% | -0.39% |
| ROM (ann.) | ~5% | 24.73% | 25.48% |

**What it demonstrates:**
- The breakout buffer is the single most impactful parameter.
- 0.2% is catastrophically too tight — noise triggers dominate.
- 0.5% sits in the sweet spot — genuine directional moves filtered, noise preserved as theta harvest.
- Adaptive closely tracks 0.5% fixed with slightly better worst-case risk properties.

## 4.6 Margin vs Return Visualisation

**What it shows:** A scatter or bar chart mapping required margin (₹93,191) against annualised net return (₹~23,000), highlighting ROM as the slope.

**What it demonstrates:**
- Capital efficiency: ~25% return on locked-up margin.
- Minimal capital at risk (max DD < 10% of margin even under stress).
- Clear separation between "capital deployed" (₹10L total) and "capital at risk" (₹93K margin). Surplus capital can be deployed in low-risk instruments (T-bills, liquid funds) to earn additional return.

## 4.7 Stress Test Loss Comparison Bar Chart

**What it shows:** Worst-case losses across different test scenarios, expressed as % of margin.

| Scenario | Worst Loss (% margin) |
|---|---|
| Historical worst day | -1.79% |
| Synthetic 30% trend stress | -1.79% |
| MC P99 drawdown | -6.41% |
| MC worst-case drawdown | -10.11% |

**What it demonstrates:**
- Even the most extreme Monte Carlo path (worst of 10,000 reshuffles) produces only 10.1% margin drawdown.
- The breakout stop caps single-day loss identically across historical and synthetic scenarios (₹-1,664 = -1.79% margin).
- The strategy has a natural loss ceiling created by the breakout stop — a mechanical, non-discretionary risk control.

---

# 5. Risk Discussion Section

## 5.1 What Risk Remains

The strategy is not risk-free. The following risks persist despite robust backtesting:

1. **Execution gap risk** — The backtest assumes exit at the close of the bar when breakout is detected. In live trading, by the time the signal is processed, the market may have moved further. The 3-minute granularity means up to 3 minutes of additional adverse movement before exit.

2. **Liquidity risk** — During extreme events (SEBI announcements, geopolitical shocks, flash crashes), the options order book may thin dramatically. The breakout stop relies on being able to buy back the sold options at reasonable prices. If the book is empty, the actual exit price could be significantly worse than modelled.

3. **Structural regime change** — The backtest covers 18 months. If the NIFTY market structure changes fundamentally (e.g., introduction of 0DTE at scale, algorithmic market-making withdrawal, regulatory change), the statistical properties of intraday ranges may shift permanently.

4. **Correlation breakdown** — The strategy treats each day as independent. In reality, during crisis periods, losses cluster — several bad days in succession. Monte Carlo reshuffling captures drawdown sensitivity to ordering but not to autocorrelated loss clustering.

5. **Model risk** — The cost model uses flat assumptions (₹20/leg, 0.05% slippage, 0.0625% STT). These are approximations. Real costs vary with order size, time of day, and market conditions.

## 5.2 Why Backtests Cannot Capture Liquidity Collapse

Historical candle data records the *outcome* of trading — prices at which trades occurred. It does not record the depth of the order book at each moment. During a liquidity collapse:

- Bid-ask spreads widen from 1-2 points to 50-100+ points
- Market orders fill against stale or distant limit orders
- Multiple participants attempt to exit simultaneously, creating feedback loops
- Circuit breakers may halt trading entirely, trapping positions

None of these dynamics appear in candle data. A backtest using historical prices implicitly assumes "normal" liquidity was available at every bar. This is precisely the assumption that fails during the moments when risk control matters most.

## 5.3 What Happens During Structural Volatility Expansion

If market volatility expands persistently (e.g., a months-long bear market with daily ranges consistently above 1.5%):

- Breakout stops trigger more frequently (higher proportion of trending days)
- Each breakout exit incurs transaction costs with potential losses
- The strategy may shift from net-profitable to net-break-even or negative
- The adaptive buffer would widen, partially offsetting this by giving more room

Synthetic stress testing showed only 4.1% Sharpe degradation under 30% trend-day scenario. However, a *permanent* shift to 50%+ trend days (never observed in the sample but theoretically possible) could eliminate the edge entirely.

## 5.4 How the Breakout Stop Limits Risk

The breakout stop is a **deterministic, mechanical risk control** that:

- Caps single-day loss at ~1.8% of margin regardless of scenario
- Operates on spot price (highly liquid) rather than option premium (potentially illiquid)
- Requires no discretionary judgement — purely rule-based
- Acts as both risk control AND regime filter — trending days are automatically exited

## 5.5 What Still Cannot Be Modelled

- **Flash crashes** — sub-second liquidity evaporation followed by rapid recovery
- **Overnight gap risk** — the strategy is intraday-only, but a pre-open event could cause the market to open at extreme levels, making the 09:15–09:30 range itself anomalous
- **Counterparty risk** — broker default, exchange system failure
- **Fat-tail correlation** — the tendency for multiple risk factors to fail simultaneously during crises

## 5.6 Biggest Remaining Risk — Professional Statement

> The primary residual risk is a **non-linear liquidity deterioration event** coinciding with the strategy holding an open short straddle position. In such a scenario, the breakout stop signal would be generated on time, but the exit execution could incur slippage 10–50× the modelled assumption, potentially converting a controlled ₹1,664 loss into a ₹10,000–₹30,000 loss on a single lot. This risk cannot be backtested, cannot be hedged cheaply, and can only be mitigated through position sizing discipline (never deploying more than 2–3 lots per ₹10L capital) and maintaining awareness that the fat-tail event is always possible.

---

# 6. Interview Defense Section

## Q1: "Walk me through your project in 60 seconds."

I built an intraday NIFTY ATM straddle backtesting engine using 18 months of 3-minute options data. I started with a long straddle hypothesis — buying volatility after compressed opens — and discovered it has no structural edge (Sharpe -11.4) because IV contraction and theta bleed after open work against the buyer. I pivoted to selling the straddle with a spot breakout stop as the primary risk control. After testing buffer sensitivities, I found that 0.5% buffer produces Sharpe 4.64, ROM 25%, with max drawdown under 3% of margin. I validated this through Monte Carlo stress testing, regime-aware simulation, rolling window analysis, and cost sensitivity — all confirming the edge is real, robust, and cost-surviving.

## Q2: "Why isn't this overfitting?"

Three reasons. First, the parameter grid shows a *plateau* of good performance — Sharpe remains above 3.0 across a wide range of stop-loss, profit target, and exit time combinations. Overfitting produces a sharp spike at one parameter set with collapse elsewhere. Second, every rolling 6-month Sharpe window is positive (range 2.50–6.02) — there is no single lucky period driving the aggregate. Third, the same strategy logic works across both fixed and adaptive buffer modes with similar results, confirming the edge is structural rather than parameter-specific.

## Q3: "Why doesn't everyone do this?"

Many sophisticated participants do sell short-dated straddles — it's a well-known strategy. But most retail traders lack the infrastructure for proper cost modelling, real-time breakout stops, and disciplined position sizing. The edge is also modest per trade (~₹143/day) and only compelling at institutional lot sizes or with margin-efficient capital allocation. Additionally, the risk of tail events deters risk-averse participants. The edge exists precisely because it requires accepting the possibility of occasional sharp losses — it compensates for bearing that risk.

## Q4: "What breaks this strategy?"

A persistent structural shift to high-trend, high-volatility markets would erode the edge. If NIFTY consistently moved 2%+ intraday (vs the current ~0.8% average), the breakout stop would trigger on most days, converting the strategy into a series of small losses plus transaction costs. Additionally, a liquidity crisis during an open position could cause catastrophic slippage beyond the modelled stop. Regulatory changes — a significant increase in STT, margin requirements, or lot sizes — could also shift the cost-to-alpha ratio below breakeven.

## Q5: "How would you scale this?"

Horizontally across instruments (BANKNIFTY, FINNIFTY) and vertically in lot count. Before scaling lots, I would validate that NIFTY options order-book depth can absorb the additional volume without market impact. Empirically, NIFTY ATM options trade ~₹2000 crore daily — a few lots have zero market impact. I would also diversify across underlyings rather than concentrating in NIFTY alone, and size positions so no single instrument represents more than 30% of risk budget.

## Q6: "Why not short a naked straddle without the stop?"

Without the breakout stop, a single trending day could produce an unlimited loss. The stop converts an undefined-risk position into a defined-risk one. In our data, the worst day was capped at ₹-1,664 because the stop triggered. Without it, the same trending day could have produced ₹-5,000 to ₹-15,000 in losses depending on how far spot moved. The stop is not optional — it is the mechanism that makes the strategy viable.

## Q7: "How do you know the regime won't change?"

I don't. Market regimes will change. The strategy is designed to degrade gracefully, not to assume regime stationarity. The breakout stop limits losses during high-vol regimes. The adaptive buffer adjusts exposure dynamically. Synthetic stress testing showed only 4.1% performance degradation under 30% trend-day scenarios. But I cannot guarantee the future will resemble the past — which is why position sizing, diversification, and real-time monitoring are essential complements to the strategy itself.

## Q8: "Why is the gross Sharpe so much higher than the net Sharpe?"

Gross Sharpe is 7.83, net is 4.64 — meaning transaction costs consume ~41% of risk-adjusted alpha. This is typical for high-frequency systematic strategies where trade count is high. The costs are: ₹80 brokerage (₹20 × 4 legs), ~0.1% slippage on premium, and 0.0625% STT on sell-side entry. The cost drag scales linearly with trade count, so strategies that exit more frequently (narrower buffer) have worse cost ratios. The 0.5% buffer minimises unnecessary exits, which is precisely why it outperforms 0.2%.

## Q9: "What's your sample size and is it sufficient?"

369 trading days producing 249 trades over 18 months. For intraday strategies with daily trade frequency, this is adequate for identifying structural edge — the law of large numbers converges quickly when trade duration is measured in hours. However, 18 months does not capture multi-year cycles (e.g., extended bear markets, election-induced regime shifts). I would want 3–5 years of data to be fully confident. The rolling window analysis partially mitigates this concern — every 6-month sub-period is independently profitable.

## Q10: "How do you handle look-ahead bias?"

Three mechanisms. First, the opening range is computed only from 09:15–09:30 bars, and entry occurs at 09:30 close — no future information is used. Second, the adaptive breakout buffer uses a strictly lagged rolling average (last N completed days — current day is excluded). Third, all exit signals use bar-close prices, not intra-bar highs/lows that could introduce survivorship bias. The pre-computation of daily ranges uses only fully-completed days, and the rolling average explicitly excludes the current trading day.

## Q11: "Why is drawdown so low? Is that suspicious?"

Drawdown is low because (1) each trade's risk is capped by the breakout stop (~1.8% of margin worst case), (2) the strategy trades only one lot on ₹10L capital (low leverage), and (3) daily P&L is small relative to capital (mean ~₹143 on ₹10L). On a margin-adjusted basis, max DD is ~3.3% — which is more realistic. Monte Carlo stress shows P99 DD at 6.4% of margin, which is reasonable for a strategy with 25% ROM. The math is consistent: low individual trade risk × moderate win rate = shallow drawdown path.

## Q12: "What would you do differently if you started over?"

I would start with the short straddle hypothesis directly — the long straddle failure, while educational, was predictable from first principles (IV contraction + theta). I would also incorporate order-book depth data (Level 2) to model slippage more realistically, and I would test across multiple underlyings (BANKNIFTY, FINNIFTY) simultaneously to assess strategy generalisability from the outset.

## Q13: "How sensitive is the strategy to the exact entry time?"

The strategy enters at 09:30 (close of the opening range period). Shifting entry to 09:33 or 09:36 would slightly reduce the opening range sample but is unlikely to materially change results — the IV contraction and theta bleed that drive the edge operate over hours, not minutes. However, entering significantly later (e.g., 10:00) would reduce both the available theta harvest and the informational value of the opening range. The 09:30 entry is a natural structural point (end of price discovery) rather than an optimised parameter.

## Q14: "Explain why the regime filter hurts."

The regime filter skips trading on days with wide opening ranges, high rolling volatility, or rising IV — all indicators of trending days. The problem is that the breakout stop already handles trending days mechanically by exiting quickly. The filter removes 35% of trade days, but the majority of those days would have been profitable (because spot didn't actually breach the buffer). The filter's conservative stance costs more in missed profits than it saves in avoided losses. Sharpe dropped from 4.64 to 2.31.

## Q15: "If you deployed this strategy tomorrow, what's your risk budget?"

Maximum 2 lots on ₹10L capital. This limits worst-case single-day loss to ~₹3,300 (2 × ₹1,664) = 0.33% of capital. MC P99 drawdown would scale to ~₹12,000 (1.2% of capital) — well within typical retail risk tolerance. I would run the strategy for 30 days paper-trading first to validate execution assumptions, then deploy 1 lot for 60 days of live validation before scaling to 2 lots. I would set a hard "circuit breaker" at 15% cumulative margin drawdown — if reached, stop trading and re-evaluate.

---

# 7. Business Perspective

## 7.1 Why Gross Edge Matters

The gross edge (pre-cost Sharpe: 7.83) represents the **raw structural inefficiency** being harvested. It is the strategy's "engine" — the fundamental reason it works. Net edge (Sharpe: 4.64) is what actually reaches the P&L. The gap between gross and net is entirely determined by execution quality and cost structure.

From a business perspective, gross edge determines **scalability ceiling** — a strategy with marginal gross edge (e.g., Sharpe 1.5 gross) has no room for cost increases, execution degradation, or market impact. A strategy with 7.83 gross Sharpe has substantial buffer: even if costs doubled, net Sharpe would remain above 2.0. This is why firms track gross and net separately.

## 7.2 Why Cost Modelling Is Critical

The project's most important analytical finding was that **86.5% of gross alpha was consumed by costs** at the initial 0.2% buffer setting. Without explicit cost modelling, the strategy would have appeared viable (gross Sharpe ~3.0 at 0.2% buffer) but would have destroyed capital in live trading.

Cost modelling disciplines:
- It forces the developer to understand the real friction faced by the strategy
- It reveals which parameters are cost-sensitive (buffer width) vs cost-insensitive (SL level)
- It provides a concrete threshold for "minimum viable edge" — the gross edge must exceed total costs with a margin of safety

## 7.3 Why Execution Quality Affects Edge

Execution quality has three dimensions:
1. **Latency** — How quickly can you react to a breakout signal? At 3-minute candle resolution, maximum latency is 3 minutes. Sub-minute execution would allow tighter stops.
2. **Slippage** — The difference between signal price and fill price. Higher volume = lower slippage. NIFTY ATM options have excellent liquidity under normal conditions.
3. **Reliability** — System uptime, network stability, broker API reliability. A missed exit signal on a trending day converts a controlled loss into an uncontrolled one.

Every basis point of execution improvement flows directly to net P&L.

## 7.4 How Margin Efficiency Matters for Capital Allocation

The strategy requires ~₹93,000 margin per lot but is deployed within a ₹10L capital base. The excess ₹9.07L is not idle — it can be invested in liquid funds, treasury bills, or overnight repos earning 5–7% annually. This "free carry" adds ~₹50,000–₹63,000 per year to total portfolio return *without additional risk*.

Margin efficiency (ROM = 25%) combined with free carry on excess capital makes the total capital return: (₹23,000 strategy + ₹55,000 carry) / ₹10,00,000 = 7.8% — on a strategy with Sharpe 4.6 and max DD 0.31%. This is competitive with many institutional allocation strategies.

## 7.5 Why Sharpe > Raw Return

A prop desk allocates capital based on **risk-adjusted returns**, not raw returns. Consider two strategies:

| Strategy | Annual Return | Annual Volatility | Sharpe |
|---|---|---|---|
| A | 50% | 80% | 0.63 |
| B | 8% | 1.5% | 5.33 |

Strategy B earns less, but with drastically lower risk. With leverage, Strategy B's returns can be scaled arbitrarily while maintaining the same Sharpe. Strategy A cannot be leveraged without magnifying already-extreme volatility.

This is why institutional investors prefer high-Sharpe, low-return strategies over low-Sharpe, high-return ones: Sharpe survives leverage, raw return does not.

Our strategy (Sharpe 4.64, return ~3.6% on capital) belongs in the "leverage-to-taste" category — returns can be scaled by deploying more lots within the ₹10L base, improving raw return while preserving the risk-adjusted profile.

---

# 8. Final Self-Evaluation

## 8.1 What I Learned

1. **Hypothesis rejection is a result.** The long straddle failure was not wasted work — it was the most rigorous finding of the project. Conclusively demonstrating that a hypothesis has no edge (Sharpe -11.4 across all parameters) is as valuable as finding a working strategy. I learned to let data override intuition.

2. **Costs are not a footnote — they are the strategy.** The difference between a profitable and unprofitable strategy was not the entry logic or signal — it was the interaction between exit frequency and transaction costs. This reframed my understanding: a trading strategy is fundamentally a cost-management problem with a signal attached.

3. **Risk control is not about preventing losses — it's about bounding them.** The breakout stop doesn't prevent losing days. It guarantees that no losing day exceeds ~1.8% of margin. This distinction — control vs prevention — is the essence of risk management.

4. **Parameters should form plateaus, not spikes.** If performance is sensitive to the third decimal place of a parameter, the strategy is fragile. The 0.5% buffer works because it sits in a region where nearby values (0.4%, 0.6%) also work. Robustness means living on a plateau.

5. **Validation is more work than implementation.** Building the backtester was ~30% of the effort. The remaining 70% was validation: rolling windows, Monte Carlo, synthetic stress, regime analysis, cost sensitivity, parameter grids. Production-quality conviction requires production-quality testing.

## 8.2 What I Would Improve Next

1. **Multi-underlying testing** — Extend to BANKNIFTY and FINNIFTY to test strategy generalisability and enable diversified deployment.

2. **Tick-level or 1-minute data** — 3-minute candles may miss intra-bar stop signals. Finer granularity would allow tighter, more responsive exits and more accurate cost modelling.

3. **Order-book simulation** — Replace flat slippage assumptions with a market-microstructure model that accounts for queue position, order-book depth, and time-of-day effects.

4. **Live paper trading** — Deploy on a simulated live feed for 60 days to validate execution assumptions before real capital deployment.

5. **Automated daily reporting** — Build a monitoring dashboard that tracks live performance against backtest expectations and triggers alerts if Sharpe degrades below thresholds.

6. **Dynamic capital allocation** — Rather than fixed lot count, scale position size inversely with recent realised volatility — expose more capital in calm regimes, reduce in turbulent ones.

## 8.3 How I Think Now vs Before

**Before this project**, I thought of trading strategies as prediction problems — find the signal, buy low, sell high. I focused on entry accuracy and ignored execution cost, position sizing, and risk control.

**Now**, I think of trading strategies as **risk-harvesting systems** with explicit cost structures. The signal matters, but it matters less than: (1) the cost to express the signal, (2) the risk profile of being wrong, and (3) the robustness of the edge across changing conditions. I now evaluate any strategy idea by asking three questions first:

- What does the gross edge diagram look like across parameters? (Plateau or spike?)
- What percentage of gross alpha survives after costs?
- Does the strategy degrade gracefully or fail catastrophically under stress?

If all three answers are favourable, the strategy is worth pursuing. If any one fails, no amount of signal quality can save it.

---

*End of document.*

*This report contains backtested results only. Past performance is not indicative of future results. All strategies involving short options carry unlimited theoretical risk. The analysis assumes normal market conditions and standard execution. Tail events, liquidity crises, and regulatory changes are acknowledged as residual risks that cannot be fully modelled.*
