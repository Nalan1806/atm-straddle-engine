"""
Main Runner – CLI entry-point
==============================
``python -m straddle_scalper.main``

Loads data, runs the backtest with default parameters,
prints summary metrics, saves trade log CSV & equity curve plot.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd

from .config import StrategyParams, OUTPUT_DIR
from .data_loader import load_data, build_intraday_panel
from .backtester import run_backtest, trades_to_dataframe
from .metrics import compute_metrics, build_equity_curve, daily_return_series

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    params = StrategyParams()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ load
    logger.info("=== Loading data ===")
    ce_df, pe_df = load_data(strike_filter=params.strike_filter)
    panel = build_intraday_panel(ce_df, pe_df)

    # ------------------------------------------------------------------ run
    logger.info("=== Running backtest ===")
    trades = run_backtest(panel, params)
    trade_df = trades_to_dataframe(trades)

    if trade_df.empty:
        logger.warning("No trades generated. Check compression_threshold or data range.")
        sys.exit(0)

    # ------------------------------------------------------------------ metrics
    metrics = compute_metrics(trade_df, params)
    logger.info("=== Summary Metrics ===")
    for k, v in metrics.items():
        logger.info("  %-25s : %s", k, v)

    # ------------------------------------------------------------------ exports
    # Trade log
    trade_csv_path = OUTPUT_DIR / "trade_log.csv"
    trade_df.to_csv(trade_csv_path, index=False)
    logger.info("Trade log saved → %s", trade_csv_path)

    # Equity curve data
    eq = build_equity_curve(trade_df, params.initial_capital)
    eq_csv_path = OUTPUT_DIR / "equity_curve.csv"
    eq.to_csv(eq_csv_path, index=False)
    logger.info("Equity curve data saved → %s", eq_csv_path)

    # Equity curve plot
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]})

    ax1 = axes[0]
    ax1.plot(eq["date"], eq["equity"], linewidth=1.2, color="#1f77b4")
    ax1.axhline(params.initial_capital, color="grey", linestyle="--", linewidth=0.8)
    ax1.set_title("Equity Curve – ATM Straddle Scalper", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Equity (INR)")
    ax1.grid(True, alpha=0.3)
    ax1.ticklabel_format(style='plain', axis='y')

    # Daily PnL bar chart
    ax2 = axes[1]
    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in eq["daily_pnl"]]
    ax2.bar(eq["date"], eq["daily_pnl"], color=colors, width=1.0, alpha=0.7)
    ax2.set_ylabel("Daily PnL (INR)")
    ax2.set_xlabel("Date")
    ax2.axhline(0, color="grey", linewidth=0.5)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    plot_path = OUTPUT_DIR / "equity_curve.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    logger.info("Equity curve plot saved → %s", plot_path)

    # Daily return histogram
    dr = daily_return_series(trade_df, params.initial_capital)
    fig2, ax3 = plt.subplots(figsize=(10, 5))
    ax3.hist(dr.values * 100, bins=50, edgecolor="black", alpha=0.7, color="#1f77b4")
    ax3.set_title("Daily Return Distribution (%)", fontsize=13, fontweight="bold")
    ax3.set_xlabel("Daily Return (%)")
    ax3.set_ylabel("Frequency")
    ax3.axvline(0, color="red", linestyle="--", linewidth=0.8)
    ax3.grid(True, alpha=0.3)
    fig2.tight_layout()
    hist_path = OUTPUT_DIR / "daily_return_dist.png"
    fig2.savefig(hist_path, dpi=150)
    plt.close(fig2)
    logger.info("Daily return distribution saved → %s", hist_path)

    # Metrics summary to CSV
    metrics_flat = {k: v for k, v in metrics.items() if k != "exit_reason_breakdown"}
    exit_bd = metrics.get("exit_reason_breakdown", {})
    for reason, cnt in exit_bd.items():
        metrics_flat[f"exits_{reason}"] = cnt
    metrics_df = pd.DataFrame([metrics_flat])
    metrics_csv_path = OUTPUT_DIR / "summary_metrics.csv"
    metrics_df.to_csv(metrics_csv_path, index=False)
    logger.info("Summary metrics saved → %s", metrics_csv_path)

    logger.info("=== Done ===")


if __name__ == "__main__":
    main()
