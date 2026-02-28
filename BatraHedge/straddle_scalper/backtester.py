"""
Backtester – ATM Straddle Engine (Long + Short)
================================================
Vectorised day-by-day simulation with per-bar exit checks.

Supports two modes controlled by ``params.strategy_type``:

**Long Straddle** (buy-side)
  Entry: buy CE + PE at 09:30 close.
  Exits: premium rises by profit_target, drops by stop_loss, or time.

**Short Straddle** (sell-side)
  Entry: sell CE + PE at 09:30 close.
  Exits:
    1. Premium stop-loss – premium >= entry*(1+short_stop_loss_pct)
    2. Profit booking   – premium <= entry*(1-short_profit_target_pct)
    3. Spot breakout     – spot moves outside 15-min range ± buffer
    4. Time exit         – hard exit (default 14:30)

Transaction Costs
-----------------
Applied on both entry and exit:
  - Brokerage: flat per leg  (× 2 legs × 2 sides = 4 legs total)
  - Slippage: % of premium on each leg
  - STT: charged on sell-side turnover only
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

from .config import StrategyParams

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Trade record
# --------------------------------------------------------------------------- #

@dataclass
class TradeRecord:
    date: object                     # datetime.date
    strategy_type: str               # "Long Straddle" | "Short Straddle"
    entry_time: str
    exit_time: str
    exit_reason: str                 # varies by strategy type

    spot_at_entry: float
    spot_at_exit: float

    ce_entry: float
    pe_entry: float
    ce_exit: float
    pe_exit: float

    entry_premium: float             # ce_entry + pe_entry
    exit_premium: float              # ce_exit  + pe_exit

    gross_pnl: float                 # direction-aware
    txn_cost: float                  # total transaction costs
    net_pnl: float                   # gross_pnl - txn_cost

    # ---- diagnostics ----
    opening_range_pct: float         # high-low / spot during 09:15-09:30
    intraday_range_pct: float        # high-low / spot after entry till exit

    iv_ce_entry: float
    iv_pe_entry: float
    iv_ce_exit: float
    iv_pe_exit: float

    holding_minutes: float

    # ---- short-straddle specific ----
    max_adverse_excursion: float = 0.0     # worst mark-to-market loss (INR)
    spot_range_high: float = 0.0           # opening 15-min spot high
    spot_range_low: float = 0.0            # opening 15-min spot low
    adaptive_breakout_buffer: float = 0.0  # actual buffer used (0 = fixed mode)


# --------------------------------------------------------------------------- #
# Cost calculator
# --------------------------------------------------------------------------- #

def compute_txn_cost(
    ce_entry: float, pe_entry: float,
    ce_exit: float, pe_exit: float,
    lot_size: int,
    params: StrategyParams,
) -> float:
    """Return total transaction cost for a round-trip straddle trade (INR)."""

    # Brokerage: per leg, 2 legs × 2 sides
    brokerage = params.brokerage_per_leg * 4

    # Slippage on entry
    entry_slippage = (
        (ce_entry * params.slippage_pct / 100)
        + (pe_entry * params.slippage_pct / 100)
    ) * lot_size

    # Slippage on exit
    exit_slippage = (
        (ce_exit * params.slippage_pct / 100)
        + (pe_exit * params.slippage_pct / 100)
    ) * lot_size

    # STT on sell-side turnover (options)
    # For long straddle, sell side = exit.  For short straddle, sell side = entry.
    if params.strategy_type == "Short Straddle":
        sell_turnover = (ce_entry + pe_entry) * lot_size
    else:
        sell_turnover = (ce_exit + pe_exit) * lot_size
    stt = sell_turnover * params.stt_on_sell_pct / 100

    return brokerage + entry_slippage + exit_slippage + stt


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _time_to_min(t: str) -> float:
    """Convert HH:MM:SS string to minutes since midnight."""
    parts = str(t).split(":")
    if len(parts) >= 3:
        return int(parts[0]) * 60 + int(parts[1]) + int(parts[2]) / 60
    elif len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    return 0.0


# --------------------------------------------------------------------------- #
# Core engine
# --------------------------------------------------------------------------- #

def run_backtest(
    panel: pd.DataFrame,
    params: StrategyParams | None = None,
) -> List[TradeRecord]:
    """
    Execute the straddle strategy on the merged intraday panel.

    Dispatches to long or short logic based on ``params.strategy_type``.
    """
    if params is None:
        params = StrategyParams()

    if params.strategy_type == "Short Straddle":
        return _run_short_straddle(panel, params)
    else:
        return _run_long_straddle(panel, params)


# --------------------------------------------------------------------------- #
# LONG STRADDLE engine (original logic preserved)
# --------------------------------------------------------------------------- #

def _run_long_straddle(
    panel: pd.DataFrame,
    params: StrategyParams,
) -> List[TradeRecord]:
    """Buy-side ATM straddle with compression + IV momentum filters."""
    or_start = params.opening_range_start
    or_end   = params.opening_range_end
    entry_t  = params.entry_time
    exit_t   = params.exit_time

    grouped = panel.groupby("date")
    trading_days = sorted(grouped.groups.keys())
    logger.info("Trading days in panel: %d", len(trading_days))

    trades: List[TradeRecord] = []

    for day in trading_days:
        day_df = grouped.get_group(day).sort_values("datetime").reset_index(drop=True)

        # --- Opening range ---
        or_mask = (day_df["time"] >= or_start) & (day_df["time"] <= or_end)
        or_bars = day_df[or_mask]
        if or_bars.empty:
            continue

        or_high_ce = or_bars["ce_high"].max()
        or_low_ce  = or_bars["ce_low"].min()
        or_high_pe = or_bars["pe_high"].max()
        or_low_pe  = or_bars["pe_low"].min()

        combined_high = or_high_ce + or_high_pe
        combined_low  = or_low_ce  + or_low_pe
        spot_ref = or_bars["spot"].iloc[-1]
        opening_range_pct = (combined_high - combined_low) / spot_ref if spot_ref else 0.0

        # --- Compression filter ---
        if params.compression_filter_enabled and opening_range_pct >= params.compression_threshold:
            continue

        # --- Entry bar ---
        entry_bar = day_df[day_df["time"] == entry_t]
        if entry_bar.empty:
            continue
        entry_bar = entry_bar.iloc[0]

        # --- IV momentum filter ---
        or_first = or_bars.iloc[0]
        iv_open = (or_first["ce_iv"] + or_first["pe_iv"]) / 2
        iv_current = (entry_bar["ce_iv"] + entry_bar["pe_iv"]) / 2
        if iv_current <= iv_open * (1 + params.iv_momentum_threshold):
            continue

        ce_entry = entry_bar["ce_close"]
        pe_entry = entry_bar["pe_close"]
        entry_premium = ce_entry + pe_entry
        if entry_premium <= 0:
            continue

        iv_ce_entry = entry_bar["ce_iv"]
        iv_pe_entry = entry_bar["pe_iv"]

        # --- Walk forward ---
        post_entry = day_df[day_df["time"] > entry_t].sort_values("datetime")

        exit_reason: Optional[str] = None
        ce_exit = ce_entry
        pe_exit = pe_entry
        exit_time_str = entry_t
        iv_ce_exit = iv_ce_entry
        iv_pe_exit = iv_pe_entry

        post_entry_highs_ce = [ce_entry]
        post_entry_lows_ce  = [ce_entry]
        post_entry_highs_pe = [pe_entry]
        post_entry_lows_pe  = [pe_entry]

        for _, bar in post_entry.iterrows():
            post_entry_highs_ce.append(bar["ce_high"])
            post_entry_lows_ce.append(bar["ce_low"])
            post_entry_highs_pe.append(bar["pe_high"])
            post_entry_lows_pe.append(bar["pe_low"])
            current_premium = bar["ce_close"] + bar["pe_close"]

            if current_premium >= entry_premium * (1 + params.profit_target):
                exit_reason = "profit_target"
                ce_exit = bar["ce_close"]; pe_exit = bar["pe_close"]
                iv_ce_exit = bar["ce_iv"]; iv_pe_exit = bar["pe_iv"]
                exit_time_str = bar["time"]; break

            if current_premium <= entry_premium * (1 - params.stop_loss):
                exit_reason = "stop_loss"
                ce_exit = bar["ce_close"]; pe_exit = bar["pe_close"]
                iv_ce_exit = bar["ce_iv"]; iv_pe_exit = bar["pe_iv"]
                exit_time_str = bar["time"]; break

            if bar["time"] >= exit_t:
                exit_reason = "time_stop"
                ce_exit = bar["ce_close"]; pe_exit = bar["pe_close"]
                iv_ce_exit = bar["ce_iv"]; iv_pe_exit = bar["pe_iv"]
                exit_time_str = bar["time"]; break

        if exit_reason is None:
            exit_reason = "time_stop"
            exit_time_str = entry_t

        exit_premium = ce_exit + pe_exit

        intraday_high = max(post_entry_highs_ce) + max(post_entry_highs_pe)
        intraday_low  = min(post_entry_lows_ce)  + min(post_entry_lows_pe)
        intraday_range_pct = (intraday_high - intraday_low) / spot_ref if spot_ref else 0.0

        holding_minutes = _time_to_min(exit_time_str) - _time_to_min(entry_t)

        exit_bar_data = day_df[day_df["time"] == exit_time_str]
        spot_at_exit = exit_bar_data["spot"].iloc[0] if not exit_bar_data.empty else spot_ref

        gross_pnl = (exit_premium - entry_premium) * params.lot_size
        txn_cost = compute_txn_cost(ce_entry, pe_entry, ce_exit, pe_exit, params.lot_size, params)
        net_pnl = gross_pnl - txn_cost

        trades.append(TradeRecord(
            date=day, strategy_type="Long Straddle",
            entry_time=entry_t, exit_time=exit_time_str, exit_reason=exit_reason,
            spot_at_entry=spot_ref, spot_at_exit=spot_at_exit,
            ce_entry=ce_entry, pe_entry=pe_entry, ce_exit=ce_exit, pe_exit=pe_exit,
            entry_premium=entry_premium, exit_premium=exit_premium,
            gross_pnl=round(gross_pnl, 2), txn_cost=round(txn_cost, 2), net_pnl=round(net_pnl, 2),
            opening_range_pct=round(opening_range_pct, 6),
            intraday_range_pct=round(intraday_range_pct, 6),
            iv_ce_entry=round(iv_ce_entry, 4), iv_pe_entry=round(iv_pe_entry, 4),
            iv_ce_exit=round(iv_ce_exit, 4), iv_pe_exit=round(iv_pe_exit, 4),
            holding_minutes=round(holding_minutes, 1),
        ))

    logger.info("Total trades generated: %d", len(trades))
    return trades


# --------------------------------------------------------------------------- #
# SHORT STRADDLE engine
# --------------------------------------------------------------------------- #

def _run_short_straddle(
    panel: pd.DataFrame,
    params: StrategyParams,
) -> List[TradeRecord]:
    """
    Sell-side ATM straddle with strict intraday risk control.

    Entry: Sell ATM CE + PE at 09:30 close.
    Exits (any triggers):
      1. Premium stop-loss  : combined >= entry * (1 + short_stop_loss_pct)
      2. Profit booking     : combined <= entry * (1 - short_profit_target_pct)
      3. Spot breakout stop : spot exits opening 15-min range ± breakout_buffer_pct
      4. Time exit          : hard time exit (default 14:30)
    """
    or_start = params.opening_range_start
    or_end   = params.opening_range_end
    entry_t  = params.entry_time
    exit_t   = params.exit_time

    grouped = panel.groupby("date")
    trading_days = sorted(grouped.groups.keys())
    logger.info("Trading days in panel: %d", len(trading_days))

    # --- Pre-compute daily intraday range % (no look-ahead) ---
    # Used for both regime filter AND adaptive breakout buffer
    _daily_range_pct: dict = {}
    for _d in trading_days:
        _ddf = grouped.get_group(_d)
        _spot_hi = _ddf["spot"].max()
        _spot_lo = _ddf["spot"].min()
        _spot_open = _ddf["spot"].iloc[0]
        _daily_range_pct[_d] = (_spot_hi - _spot_lo) / _spot_open if _spot_open else 0.0

    trades: List[TradeRecord] = []
    regime_skipped = 0

    for day in trading_days:
        day_df = grouped.get_group(day).sort_values("datetime").reset_index(drop=True)

        # --- Opening range (15-min: 09:15 to 09:30) ---
        or_mask = (day_df["time"] >= or_start) & (day_df["time"] <= or_end)
        or_bars = day_df[or_mask]
        if or_bars.empty:
            continue

        # Premium opening range
        or_high_ce = or_bars["ce_high"].max()
        or_low_ce  = or_bars["ce_low"].min()
        or_high_pe = or_bars["pe_high"].max()
        or_low_pe  = or_bars["pe_low"].min()
        combined_high = or_high_ce + or_high_pe
        combined_low  = or_low_ce  + or_low_pe

        spot_ref = or_bars["spot"].iloc[-1]
        opening_range_pct = (combined_high - combined_low) / spot_ref if spot_ref else 0.0

        # --- Spot opening range for breakout stop ---
        spot_range_high = or_bars["spot"].max()
        spot_range_low  = or_bars["spot"].min()

        # --- Determine breakout buffer (adaptive vs fixed) ---
        day_idx = trading_days.index(day)
        if params.adaptive_breakout_enabled and day_idx >= params.adaptive_lookback_days:
            lookback = trading_days[day_idx - params.adaptive_lookback_days : day_idx]
            rolling_avg = np.mean([_daily_range_pct[d] for d in lookback])
            effective_buffer = max(
                params.adaptive_min_buffer_pct,
                min(params.adaptive_max_buffer_pct, params.adaptive_k * rolling_avg),
            )
        else:
            # Fallback: fixed buffer (also used for first N days)
            effective_buffer = params.breakout_buffer_pct

        breakout_upper = spot_range_high * (1 + effective_buffer)
        breakout_lower = spot_range_low  * (1 - effective_buffer)

        # --- Compression filter (optional) ---
        if params.compression_filter_enabled and opening_range_pct >= params.compression_threshold:
            continue

        # --- Regime filter (optional) ---
        if params.regime_filter_enabled:
            # Rule 1: opening range too wide
            spot_or_range = (spot_range_high - spot_range_low) / spot_ref if spot_ref else 0.0
            if spot_or_range > params.regime_or_threshold:
                regime_skipped += 1
                continue
            # Rule 2: rolling N-day avg range too high
            day_idx = trading_days.index(day)
            if day_idx >= params.regime_rolling_range_days:
                lookback_days = trading_days[day_idx - params.regime_rolling_range_days : day_idx]
                avg_range = np.mean([_daily_range_pct.get(d, 0.0) for d in lookback_days])
                if avg_range > params.regime_rolling_range_threshold:
                    regime_skipped += 1
                    continue
            # Rule 3: IV rising > threshold intraday (compare OR start vs entry)
            first_or = or_bars.iloc[0]
            last_or = or_bars.iloc[-1]
            iv_start = (first_or["ce_iv"] + first_or["pe_iv"]) / 2
            iv_entry = (last_or["ce_iv"] + last_or["pe_iv"]) / 2
            if iv_start > 0 and (iv_entry - iv_start) / iv_start > params.regime_iv_intraday_threshold:
                regime_skipped += 1
                continue

        # --- Entry at 09:30 close ---
        entry_bar = day_df[day_df["time"] == entry_t]
        if entry_bar.empty:
            continue
        entry_bar = entry_bar.iloc[0]

        ce_entry = entry_bar["ce_close"]
        pe_entry = entry_bar["pe_close"]
        entry_premium = ce_entry + pe_entry
        if entry_premium <= 0:
            continue

        iv_ce_entry = entry_bar["ce_iv"]
        iv_pe_entry = entry_bar["pe_iv"]

        # Pre-compute exit thresholds
        premium_stop_level  = entry_premium * (1 + params.short_stop_loss_pct)
        premium_profit_level = entry_premium * (1 - params.short_profit_target_pct)

        # --- Walk forward bar-by-bar ---
        post_entry = day_df[day_df["time"] > entry_t].sort_values("datetime")

        exit_reason: Optional[str] = None
        ce_exit = ce_entry
        pe_exit = pe_entry
        exit_time_str = entry_t
        iv_ce_exit = iv_ce_entry
        iv_pe_exit = iv_pe_entry

        # Track max adverse excursion (worst unrealised loss for the short)
        max_adverse_excursion = 0.0   # in INR

        # Track ranges
        post_entry_highs_ce = [ce_entry]
        post_entry_lows_ce  = [ce_entry]
        post_entry_highs_pe = [pe_entry]
        post_entry_lows_pe  = [pe_entry]

        for _, bar in post_entry.iterrows():
            post_entry_highs_ce.append(bar["ce_high"])
            post_entry_lows_ce.append(bar["ce_low"])
            post_entry_highs_pe.append(bar["pe_high"])
            post_entry_lows_pe.append(bar["pe_low"])

            current_premium = bar["ce_close"] + bar["pe_close"]
            current_spot = bar["spot"]

            # Track MAE: for a short, loss = (current_premium - entry_premium) * lot
            unrealised_loss = (current_premium - entry_premium) * params.lot_size
            if unrealised_loss > max_adverse_excursion:
                max_adverse_excursion = unrealised_loss

            # EXIT 1: Premium stop-loss (premium rising = loss for short)
            if current_premium >= premium_stop_level:
                exit_reason = "premium_stop"
                ce_exit = bar["ce_close"]; pe_exit = bar["pe_close"]
                iv_ce_exit = bar["ce_iv"]; iv_pe_exit = bar["pe_iv"]
                exit_time_str = bar["time"]; break

            # EXIT 2: Profit booking (premium falling = profit for short)
            if current_premium <= premium_profit_level:
                exit_reason = "profit_target"
                ce_exit = bar["ce_close"]; pe_exit = bar["pe_close"]
                iv_ce_exit = bar["ce_iv"]; iv_pe_exit = bar["pe_iv"]
                exit_time_str = bar["time"]; break

            # EXIT 3: Spot breakout stop
            if current_spot > breakout_upper or current_spot < breakout_lower:
                exit_reason = "breakout_stop"
                ce_exit = bar["ce_close"]; pe_exit = bar["pe_close"]
                iv_ce_exit = bar["ce_iv"]; iv_pe_exit = bar["pe_iv"]
                exit_time_str = bar["time"]; break

            # EXIT 4: Time exit
            if bar["time"] >= exit_t:
                exit_reason = "time_stop"
                ce_exit = bar["ce_close"]; pe_exit = bar["pe_close"]
                iv_ce_exit = bar["ce_iv"]; iv_pe_exit = bar["pe_iv"]
                exit_time_str = bar["time"]; break

        if exit_reason is None:
            exit_reason = "time_stop"
            exit_time_str = entry_t

        exit_premium = ce_exit + pe_exit

        # Intraday range
        intraday_high = max(post_entry_highs_ce) + max(post_entry_highs_pe)
        intraday_low  = min(post_entry_lows_ce)  + min(post_entry_lows_pe)
        intraday_range_pct = (intraday_high - intraday_low) / spot_ref if spot_ref else 0.0

        holding_minutes = _time_to_min(exit_time_str) - _time_to_min(entry_t)

        exit_bar_data = day_df[day_df["time"] == exit_time_str]
        spot_at_exit = exit_bar_data["spot"].iloc[0] if not exit_bar_data.empty else spot_ref

        # SHORT P&L: we sold at entry_premium, buy back at exit_premium
        gross_pnl = (entry_premium - exit_premium) * params.lot_size
        txn_cost = compute_txn_cost(ce_entry, pe_entry, ce_exit, pe_exit, params.lot_size, params)
        net_pnl = gross_pnl - txn_cost

        trades.append(TradeRecord(
            date=day, strategy_type="Short Straddle",
            entry_time=entry_t, exit_time=exit_time_str, exit_reason=exit_reason,
            spot_at_entry=spot_ref, spot_at_exit=spot_at_exit,
            ce_entry=ce_entry, pe_entry=pe_entry, ce_exit=ce_exit, pe_exit=pe_exit,
            entry_premium=entry_premium, exit_premium=exit_premium,
            gross_pnl=round(gross_pnl, 2), txn_cost=round(txn_cost, 2), net_pnl=round(net_pnl, 2),
            opening_range_pct=round(opening_range_pct, 6),
            intraday_range_pct=round(intraday_range_pct, 6),
            iv_ce_entry=round(iv_ce_entry, 4), iv_pe_entry=round(iv_pe_entry, 4),
            iv_ce_exit=round(iv_ce_exit, 4), iv_pe_exit=round(iv_pe_exit, 4),
            holding_minutes=round(holding_minutes, 1),
            max_adverse_excursion=round(max_adverse_excursion, 2),
            spot_range_high=round(spot_range_high, 2),
            spot_range_low=round(spot_range_low, 2),
            adaptive_breakout_buffer=round(effective_buffer, 6),
        ))

    if params.regime_filter_enabled and regime_skipped:
        logger.info("Regime filter skipped %d days", regime_skipped)
    logger.info("Total trades generated: %d", len(trades))
    return trades


# --------------------------------------------------------------------------- #
# Helper: trades -> DataFrame
# --------------------------------------------------------------------------- #

def trades_to_dataframe(trades: List[TradeRecord]) -> pd.DataFrame:
    """Convert list of TradeRecord to a clean DataFrame."""
    if not trades:
        return pd.DataFrame()
    data = [t.__dict__ for t in trades]
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    return df
