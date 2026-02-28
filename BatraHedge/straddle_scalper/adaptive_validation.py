"""
Adaptive Breakout Buffer — Validation Comparison
==================================================
Compares:
  A) Fixed breakout_buffer_pct = 0.5%
  B) Adaptive breakout (k=0.6, N=5, min=0.3%, max=0.8%)

Outputs: Net Sharpe, Gross Sharpe, ROM, Max DD, Worst trade,
         Rolling 6-month Sharpe, and adaptive buffer distribution stats.

Run:  python -m straddle_scalper.adaptive_validation
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_pkg_root = Path(__file__).resolve().parent.parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

from straddle_scalper.config import StrategyParams, OUTPUT_DIR
from straddle_scalper.data_loader import load_data, build_intraday_panel
from straddle_scalper.backtester import run_backtest, trades_to_dataframe
from straddle_scalper.metrics import compute_metrics, build_equity_curve, daily_return_series

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

OUT = OUTPUT_DIR / "adaptive_validation"
OUT.mkdir(parents=True, exist_ok=True)

# ─── helpers ────────────────────────────────────────────────────────────── #

def _gross_sharpe(tdf: pd.DataFrame, capital: float) -> float:
    """Annualised Sharpe on gross (pre-cost) daily PnL."""
    daily = tdf.groupby("date")["gross_pnl"].sum() / capital
    return daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else 0.0


def _rolling_6m_sharpe(tdf: pd.DataFrame, capital: float) -> pd.Series:
    """Rolling 6-month (126-day) Sharpe on net daily returns."""
    daily = tdf.groupby("date")["net_pnl"].sum() / capital
    daily = daily.sort_index()
    rolling_mean = daily.rolling(126, min_periods=60).mean()
    rolling_std  = daily.rolling(126, min_periods=60).std()
    return (rolling_mean / rolling_std * np.sqrt(252)).dropna()


def _rom(tdf: pd.DataFrame, capital: float, margin: float) -> float:
    """Return on Margin annualised %."""
    total_pnl = tdf["net_pnl"].sum()
    first  = tdf["date"].min()
    last   = tdf["date"].max()
    years  = max((last - first).days / 365.25, 1 / 365.25)
    ann_pnl = total_pnl / years
    return ann_pnl / margin * 100


# ─── main ───────────────────────────────────────────────────────────────── #

def main():
    log.info("Loading data...")
    ce, pe = load_data(strike_filter="ATM")
    panel  = build_intraday_panel(ce, pe)

    CAPITAL = 1_000_000.0
    LOT     = 25
    MARGIN  = 93_191.0   # estimated NIFTY short straddle margin

    # ── A: Fixed buffer 0.5% ─────────────────────────────────────────── #
    params_fixed = StrategyParams(
        strategy_type="Short Straddle",
        breakout_buffer_pct=0.005,
        adaptive_breakout_enabled=False,
        short_stop_loss_pct=0.20,
        short_profit_target_pct=0.10,
        exit_time="14:30:00",
        compression_filter_enabled=True,
        compression_threshold=0.004,
        initial_capital=CAPITAL,
        lot_size=LOT,
    )

    log.info("Running FIXED 0.5%% buffer...")
    trades_fixed = run_backtest(panel, params_fixed)
    tdf_fixed    = trades_to_dataframe(trades_fixed)
    met_fixed    = compute_metrics(tdf_fixed, params_fixed)

    # ── B: Adaptive defaults ─────────────────────────────────────────── #
    params_adapt = StrategyParams(
        strategy_type="Short Straddle",
        breakout_buffer_pct=0.005,          # fallback for first N days
        adaptive_breakout_enabled=True,
        adaptive_k=0.6,
        adaptive_lookback_days=5,
        adaptive_min_buffer_pct=0.003,
        adaptive_max_buffer_pct=0.008,
        short_stop_loss_pct=0.20,
        short_profit_target_pct=0.10,
        exit_time="14:30:00",
        compression_filter_enabled=True,
        compression_threshold=0.004,
        initial_capital=CAPITAL,
        lot_size=LOT,
    )

    log.info("Running ADAPTIVE buffer (k=0.6, N=5, min=0.3%%, max=0.8%%)...")
    trades_adapt = run_backtest(panel, params_adapt)
    tdf_adapt    = trades_to_dataframe(trades_adapt)
    met_adapt    = compute_metrics(tdf_adapt, params_adapt)

    # ── comparison table ─────────────────────────────────────────────── #
    log.info("")
    log.info("=" * 76)
    log.info("  FIXED vs ADAPTIVE BREAKOUT BUFFER — COMPARISON")
    log.info("=" * 76)
    log.info("")

    rows = [
        ("Trades",               met_fixed["total_trades"],
                                 met_adapt["total_trades"]),
        ("Win Rate (%)",         f'{met_fixed["win_rate"]*100:.1f}',
                                 f'{met_adapt["win_rate"]*100:.1f}'),
        ("Net PnL (₹)",         f'{met_fixed["total_pnl"]:,.0f}',
                                 f'{met_adapt["total_pnl"]:,.0f}'),
        ("Net Sharpe",           f'{met_fixed["sharpe_ratio"]:.2f}',
                                 f'{met_adapt["sharpe_ratio"]:.2f}'),
        ("Gross Sharpe",         f'{_gross_sharpe(tdf_fixed, CAPITAL):.2f}',
                                 f'{_gross_sharpe(tdf_adapt, CAPITAL):.2f}'),
        ("Profit Factor",       f'{met_fixed["profit_factor"]:.3f}',
                                 f'{met_adapt["profit_factor"]:.3f}'),
        ("Max DD (%)",           f'{met_fixed["max_drawdown"]*100:.3f}',
                                 f'{met_adapt["max_drawdown"]*100:.3f}'),
        ("ROM (ann. %)",         f'{_rom(tdf_fixed, CAPITAL, MARGIN):.2f}',
                                 f'{_rom(tdf_adapt, CAPITAL, MARGIN):.2f}'),
        ("Worst Trade (₹)",     f'{met_fixed["max_loss"]:,.0f}',
                                 f'{met_adapt["max_loss"]:,.0f}'),
        ("Best Trade (₹)",      f'{met_fixed["max_win"]:,.0f}',
                                 f'{met_adapt["max_win"]:,.0f}'),
        ("Avg Hold (min)",       f'{met_fixed["avg_hold_minutes"]:.0f}',
                                 f'{met_adapt["avg_hold_minutes"]:.0f}'),
    ]

    log.info("  %-25s  %15s  %15s", "Metric", "FIXED 0.5%", "ADAPTIVE")
    log.info("  %s", "-" * 57)
    for label, v_fixed, v_adapt in rows:
        log.info("  %-25s  %15s  %15s", label, v_fixed, v_adapt)

    # ── Rolling 6-month Sharpe ────────────────────────────────────────── #
    log.info("")
    roll_fixed = _rolling_6m_sharpe(tdf_fixed, CAPITAL)
    roll_adapt = _rolling_6m_sharpe(tdf_adapt, CAPITAL)

    if not roll_fixed.empty and not roll_adapt.empty:
        log.info("  ROLLING 6-MONTH SHARPE:")
        log.info("    Fixed  — mean: %.2f  min: %.2f  max: %.2f  all>0: %s",
                 roll_fixed.mean(), roll_fixed.min(), roll_fixed.max(),
                 "YES" if (roll_fixed > 0).all() else "NO")
        log.info("    Adapt  — mean: %.2f  min: %.2f  max: %.2f  all>0: %s",
                 roll_adapt.mean(), roll_adapt.min(), roll_adapt.max(),
                 "YES" if (roll_adapt > 0).all() else "NO")
    log.info("")

    # ── Exit reason breakdown ─────────────────────────────────────────── #
    log.info("  EXIT REASON BREAKDOWN:")
    for label, tdf in [("Fixed", tdf_fixed), ("Adaptive", tdf_adapt)]:
        counts = tdf["exit_reason"].value_counts()
        parts = [f"{r}: {c}" for r, c in counts.items()]
        log.info("    %-8s — %s", label, "  |  ".join(parts))
    log.info("")

    # ── Adaptive buffer distribution stats ────────────────────────────── #
    if "adaptive_breakout_buffer" in tdf_adapt.columns:
        buf = tdf_adapt["adaptive_breakout_buffer"] * 100  # to %
        log.info("  ADAPTIVE BUFFER DISTRIBUTION:")
        log.info("    Mean:    %.3f%%", buf.mean())
        log.info("    Median:  %.3f%%", buf.median())
        log.info("    Min:     %.3f%%", buf.min())
        log.info("    Max:     %.3f%%", buf.max())
        log.info("    Std:     %.3f%%", buf.std())
        log.info("    P10:     %.3f%%", np.percentile(buf, 10))
        log.info("    P90:     %.3f%%", np.percentile(buf, 90))
        log.info("")

    # ── Δ summary ─────────────────────────────────────────────────────── #
    sh_fixed = met_fixed["sharpe_ratio"]
    sh_adapt = met_adapt["sharpe_ratio"]
    delta_sh = sh_adapt - sh_fixed
    pct_chg  = delta_sh / abs(sh_fixed) * 100 if sh_fixed != 0 else 0.0

    log.info("  ╔═══════════════════════════════════════════════════════════╗")
    log.info("  ║  ADAPTIVE vs FIXED — DELTA SUMMARY                      ║")
    log.info("  ╠═══════════════════════════════════════════════════════════╣")
    log.info("  ║  Sharpe Δ: %+.2f  (%+.1f%%)                              ║",
             delta_sh, pct_chg)

    dd_delta = (met_adapt["max_drawdown"] - met_fixed["max_drawdown"]) * 100
    log.info("  ║  Max DD Δ: %+.3f pp                                     ║", dd_delta)

    rom_f = _rom(tdf_fixed, CAPITAL, MARGIN)
    rom_a = _rom(tdf_adapt, CAPITAL, MARGIN)
    log.info("  ║  ROM Δ:    %+.2f pp                                     ║", rom_a - rom_f)

    if sh_adapt > sh_fixed:
        log.info("  ║  ✓ ADAPTIVE IMPROVES performance                        ║")
    elif sh_adapt < sh_fixed:
        log.info("  ║  ✗ ADAPTIVE UNDERPERFORMS fixed buffer                  ║")
    else:
        log.info("  ║  ≈ No meaningful difference                             ║")

    log.info("  ╚═══════════════════════════════════════════════════════════╝")
    log.info("")
    log.info("All artefacts saved to: %s", OUT)


if __name__ == "__main__":
    main()
