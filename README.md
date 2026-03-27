# Intraday NIFTY ATM Straddle Backtesting Engine

A research-grade intraday backtesting system for **short ATM straddle strategies** on NIFTY index options, with spot breakout stop as primary risk control, full transaction cost modelling, and multi-layered robustness validation.

---

## Research Evolution

The project began by testing a **long (buy-side) ATM straddle** hypothesis: that compressed opening ranges predict subsequent volatility expansion. This hypothesis was rejected across all parameter combinations (mean Sharpe: -11.40). The structural headwinds — IV contraction after open and intraday theta decay — make buying intraday straddles systematically unprofitable on NIFTY.

The infrastructure was then inverted to test the **short ATM straddle**, which harvests the same IV contraction and theta bleed that destroyed the long side. With a 0.5% spot breakout buffer as the primary risk control, the short straddle produces a net Sharpe of 4.64 after full transaction costs.

---

## Strategy Architecture

### Entry

Sell ATM CE and PE at the close of the 09:30 candle. An optional compression filter skips days where the opening range (09:15–09:30) exceeds a configurable threshold, avoiding already-volatile sessions.

### Exit Hierarchy (First Trigger Wins)

1. **Premium stop-loss** — Combined premium rises by 20% of entry value (a loss for the seller).
2. **Profit booking** — Combined premium falls by 10% of entry value (a gain for the seller).
3. **Spot breakout stop** — Spot price moves outside the opening range ± configured buffer.
4. **Time exit** — Hard exit at 14:30.

### Breakout Buffer Modes

- **Fixed buffer** — A static percentage (default 0.5%) applied symmetrically to the opening range.
- **Adaptive buffer** — Volatility-scaled: `clamp(k × mean_range_N, min_buffer, max_buffer)`. Tightens in calm markets, widens in volatile markets.

### Transaction Cost Model

Every trade incurs: brokerage (₹20 per leg × 4 legs), slippage (0.05% on premium per leg), and STT (0.0625% on sell-side turnover). Cost modelling is critical — at a 0.2% buffer, 86.5% of gross alpha is consumed by costs; at 0.5%, cost drag drops to ~41%.

---

## Validation Suite

| Method | Description |
|---|---|
| **Rolling Sharpe** | 126-day sliding windows; all windows positive (range: 2.50–6.02) |
| **Monte Carlo** | 10,000 daily P&L reshuffles; 100% of paths end profitable; P99 drawdown -6.41% on margin |
| **Synthetic Stress** | Trend-day proportion inflated from 21.7% to 30%; Sharpe degrades only 4.1% |
| **Parameter Grid** | Sensitivity across stop-loss, target, exit time, buffer width; broad plateau of Sharpe > 3.0 |
| **Cost Sensitivity** | Explicit quantification of cost drag across buffer widths |
| **Regime Analysis** | Breakout stop acts as mechanical regime filter; explicit regime filter reduces Sharpe from 4.64 to 2.31 |

---

## Project Structure

```
BatraHedge/
├── Data/
│   └── Options_3minute/           ← 3-minute NIFTY options candles (not included)
│
├── straddle_scalper/
│   ├── config.py                  ← Strategy parameters (dataclass)
│   ├── data_loader.py             ← CSV ingestion, CE/PE alignment
│   ├── backtester.py              ← Bar-by-bar trade simulation, exit logic, cost computation
│   ├── metrics.py                 ← Sharpe, PF, drawdown, MAE, equity curve, exit breakdown
│   ├── robustness_validator.py    ← Parameter sensitivity grid
│   ├── validation_runner.py       ← Rolling windows, Monte Carlo, stress, cost analysis
│   ├── app.py                     ← Streamlit interactive dashboard
│   ├── main.py                    ← CLI runner
│   ├── __init__.py
│   └── __main__.py
│
├── requirements.txt
├── TECHNICAL_REPORT.md
└── README.md
```

---

## Data

**Data files are not included in this repository due to size (~2M rows).**

The system expects 3-minute OHLCV candle data for NIFTY options in CSV format with columns including: `date`, `time`, `symbol`, `option_type`, `type` (CE/PE), `strike_offset`, `open`, `high`, `low`, `close`, `volume`, `oi`, `iv`, `spot`.

Place partitioned CSV files in `Data/Options_3minute/` as `NIFTY_part_1.csv`, `NIFTY_part_2.csv`, `NIFTY_part_3.csv`.

---

## Setup

### Requirements

- Python 3.10+
- Dependencies listed in `requirements.txt`

### Installation

```bash
cd BatraHedge
pip install -r requirements.txt
```

---

## Usage

### Streamlit Dashboard (Recommended)

```bash
cd BatraHedge
streamlit run straddle_scalper/app.py
```

The dashboard provides:
- Sidebar parameter controls (sliders for strategy knobs, inputs for costs/capital)
- One-click backtest execution
- Summary metrics, equity curve, daily P&L distribution
- Exit reason breakdown, MAE histogram, adaptive buffer analysis
- Full trade log with export capability
- Robustness grid runner

### CLI Backtest

```bash
cd BatraHedge
python -m straddle_scalper
```

Runs the backtest with default parameters and outputs results to `straddle_scalper/output/`.

---

## Default Parameters

| Parameter | Default |
|---|---|
| Compression threshold | 0.4% |
| Profit target (premium drop) | 10% |
| Stop loss (premium rise) | 20% |
| Breakout buffer | 0.5% (fixed) |
| Adaptive buffer k | 0.6 |
| Adaptive buffer range | 0.3%–0.8% |
| Brokerage per leg | ₹20 |
| Slippage | 0.05% |
| STT on sell side | 0.0625% |
| Initial capital | ₹10,00,000 |
| Lot size | 25 |
| Exit time | 14:30 |

All parameters are configurable via `straddle_scalper/config.py` or the Streamlit sidebar.

---

## Performance Summary

| Metric | Value |
|---|---|
| Net Sharpe (annualised) | 4.64 |
| Gross Sharpe (annualised) | 7.83 |
| Profit Factor | 2.12 |
| Win Rate | 66.7% |
| Max Drawdown (capital) | -0.31% |
| Worst Single Day (margin) | -1.79% |
| Rolling 6-Month Sharpe | 2.50–6.02 (all positive) |
| Monte Carlo: Paths Profitable | 100% (10,000 paths) |

Results are from backtesting over 369 trading days (Aug 2024–Feb 2026) with full transaction costs applied.

---

## Disclaimer

All results presented are backtested and subject to the limitations inherent in historical simulation. Past performance is not indicative of future results. Short option strategies carry theoretically unlimited risk. The analysis assumes normal market conditions and standard execution quality. Liquidity deterioration during extreme events, structural regime changes, and execution gaps between signal and fill are residual risks that cannot be fully modelled.

---

## License

Internal use only — BatraHedge.
