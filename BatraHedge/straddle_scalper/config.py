"""
Configuration for ATM Straddle Scalping Backtester
===================================================
All tunable parameters in one place.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Root of the raw data directory (relative to project root)
DATA_DIR: Path = Path(__file__).resolve().parent.parent / "Data" / "Options_3minute"

# Glob pattern for the CSV part files inside DATA_DIR
DATA_GLOB: str = "NIFTY_part_*.csv"

# Output directory for results
OUTPUT_DIR: Path = Path(__file__).resolve().parent / "output"


# ---------------------------------------------------------------------------
# Strategy Parameters (defaults)
# ---------------------------------------------------------------------------

@dataclass
class StrategyParams:
    """All tuneable strategy knobs."""

    # --- Strategy type ---
    strategy_type: str = "Short Straddle"   # "Long Straddle" | "Short Straddle"

    # --- Entry filters ---
    compression_threshold: float = 0.004   # 0.4 % opening-range / spot
    compression_filter_enabled: bool = True # toggle compression filter on/off
    iv_momentum_threshold: float = 0.005   # 0.5% IV expansion required (9:30 > 9:15)
    entry_time: str = "09:30:00"           # enter on close of this candle
    opening_range_start: str = "09:15:00"
    opening_range_end: str = "09:30:00"    # inclusive

    # --- Exit targets (Long Straddle) ---
    profit_target: float = 0.10            # 10 % of entry premium
    stop_loss: float = 0.06                # 6 % of entry premium

    # --- Exit targets (Short Straddle) ---
    short_profit_target_pct: float = 0.10  # exit when premium drops 10% from entry (profit)
    short_stop_loss_pct: float = 0.20      # exit when premium rises 20% from entry (loss)
    breakout_buffer_pct: float = 0.002     # 0.2% spot breakout beyond opening 15-min range

    # --- Time stop ---
    exit_time: str = "14:30:00"            # hard time-based exit

    # --- Transaction cost model ---
    brokerage_per_leg: float = 20.0        # INR flat per leg per trade
    slippage_pct: float = 0.05             # 0.05 % of premium
    stt_on_sell_pct: float = 0.0625        # 0.0625 % of sell-side turnover (options STT on exercise/sell)

    # --- Capital ---
    initial_capital: float = 1_000_000.0   # INR 10 lakh
    lot_size: int = 25                     # NIFTY lot size

    # --- Adaptive breakout buffer (Short Straddle) ---
    adaptive_breakout_enabled: bool = True    # use vol-scaled buffer instead of fixed
    adaptive_k: float = 0.6                   # multiplier on rolling avg range
    adaptive_lookback_days: int = 5            # rolling window N
    adaptive_min_buffer_pct: float = 0.003     # floor 0.3 %
    adaptive_max_buffer_pct: float = 0.008     # cap  0.8 %

    # --- Regime filter (Short Straddle) ---
    regime_filter_enabled: bool = False    # toggle volatility regime filter
    regime_or_threshold: float = 0.006     # skip if opening-range % > 0.6%
    regime_rolling_range_days: int = 5     # rolling window for avg daily range
    regime_rolling_range_threshold: float = 0.01  # skip if 5-day avg range > 1.0%
    regime_iv_intraday_threshold: float = 0.01    # skip if IV rises > 1% intraday (entry bar vs OR start)

    # --- Data shape ---
    strike_filter: str = "ATM"             # only ATM straddle


# ---------------------------------------------------------------------------
# Convenience singleton
# ---------------------------------------------------------------------------
DEFAULT_PARAMS = StrategyParams()
