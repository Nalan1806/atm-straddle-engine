# ATM Straddle Scalper – NIFTY Options Backtester

A production-quality intraday **buy-side ATM straddle scalping** system for NIFTY options, with a Streamlit UI for interactive parameter tuning.

---

## Project Structure

```
BatraHedge/
├── Data/
│   └── Options_3minute/          ← 3-minute NIFTY options data
│       ├── NIFTY_part_1.csv
│       ├── NIFTY_part_2.csv
│       └── NIFTY_part_3.csv
│
├── straddle_scalper/             ← Core package
│   ├── __init__.py
│   ├── __main__.py               ← Allows `python -m straddle_scalper`
│   ├── config.py                 ← All parameters & paths
│   ├── data_loader.py            ← CSV ingestion & cleaning
│   ├── backtester.py             ← Strategy engine
│   ├── metrics.py                ← Performance analytics
│   ├── main.py                   ← CLI runner
│   └── app.py                    ← Streamlit UI
│
├── requirements.txt
└── README.md
```

---

## Data Requirements

The system expects **3-minute candle data** for NIFTY options in CSV format with the following columns:

| Column         | Description                                      |
|----------------|--------------------------------------------------|
| `date`         | Trading date (Excel-escaped `="DD-MM-YY"`)       |
| `time`         | Candle time `HH:MM:SS`                           |
| `symbol`       | `NIFTY`                                          |
| `option_type`  | `CALL` or `PUT`                                  |
| `type`         | `CE` or `PE`                                     |
| `strike_offset`| `ATM`, `ATM+1`, `ATM-1`, etc.                   |
| `open`         | Open price                                       |
| `high`         | High price                                       |
| `low`          | Low price                                        |
| `close`        | Close price                                      |
| `volume`       | Volume                                           |
| `oi`           | Open interest                                    |
| `iv`           | Implied volatility                               |
| `spot`         | Underlying NIFTY spot price                      |

**Both CE and PE data reside in the same CSV files.** The loader handles this automatically.

Place the part files in `Data/Options_3minute/` as:
- `NIFTY_part_1.csv`
- `NIFTY_part_2.csv`
- `NIFTY_part_3.csv`

---

## Setup Instructions

### 1. Install Python 3.10+

### 2. Install dependencies

```bash
cd BatraHedge
pip install -r requirements.txt
```

### 3. Verify data files exist

```
Data/Options_3minute/NIFTY_part_1.csv
Data/Options_3minute/NIFTY_part_2.csv
Data/Options_3minute/NIFTY_part_3.csv
```

---

## How to Run

### Option A: CLI Backtest

```bash
cd BatraHedge
python -m straddle_scalper
```

This will:
- Load all 3-minute data (~2M rows)
- Run the backtest with default parameters
- Print summary metrics to console
- Save outputs to `straddle_scalper/output/`:
  - `trade_log.csv`
  - `equity_curve.csv`
  - `equity_curve.png`
  - `daily_return_dist.png`
  - `summary_metrics.csv`

### Option B: Streamlit Interactive UI

```bash
cd BatraHedge
streamlit run straddle_scalper/app.py
```

The UI allows you to:
- Adjust compression threshold, profit target, stop loss
- Modify transaction cost parameters
- Set capital and lot size
- Run backtest with a single click
- View equity curve, daily PnL, return distribution
- Browse the full trade log
- Export results as CSV

---

## Strategy Logic

### Entry
1. For each trading day, compute the **opening range** (09:15–09:30) of the ATM straddle.
2. Calculate `opening_range_pct = (combined_high - combined_low) / spot`.
3. If `opening_range_pct < compression_threshold` → **Buy ATM straddle** at 09:30 close.

### Exit (first condition met)
- **Profit Target**: combined premium ≥ entry × (1 + profit_target)
- **Stop Loss**: combined premium ≤ entry × (1 - stop_loss)
- **Time Stop**: time ≥ 14:45

### Transaction Costs
- Brokerage: flat per leg (4 legs total per round-trip)
- Slippage: % of premium on each leg
- STT: charged on sell-side turnover

---

## Default Parameters

| Parameter              | Default  |
|------------------------|----------|
| `compression_threshold`| 0.4%     |
| `profit_target`        | 10%      |
| `stop_loss`            | 6%       |
| `brokerage_per_leg`    | ₹20      |
| `slippage_pct`         | 0.05%    |
| `stt_on_sell_pct`      | 0.0625%  |
| `initial_capital`      | ₹10,00,000 |
| `lot_size`             | 25       |
| `exit_time`            | 14:45    |

All parameters are configurable in `straddle_scalper/config.py` or via the Streamlit sidebar.

---

## Performance Metrics

| Metric               | Description                              |
|----------------------|------------------------------------------|
| Total Return         | Net PnL / Initial Capital                |
| CAGR                 | Compound Annual Growth Rate              |
| Max Drawdown         | Largest peak-to-trough equity decline    |
| Win Rate             | % of profitable trades                   |
| Average Win / Loss   | Mean PnL for winning / losing trades     |
| Profit Factor        | Gross profits / Gross losses             |
| Sharpe Ratio         | Annualised (√252 × mean / std)           |
| Avg Hold Time        | Mean holding period in minutes           |

---

## License

Internal use only – BatraHedge.
