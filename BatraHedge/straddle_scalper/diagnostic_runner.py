"""
Diagnostic Runner
=================
Full diagnostic analysis of strategy performance.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

from .config import StrategyParams, OUTPUT_DIR
from .data_loader import load_data, build_intraday_panel
from .backtester import run_backtest, trades_to_dataframe
from .metrics import compute_metrics
from .diagnostics import run_diagnostics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_full_diagnostic() -> None:
    """Execute full diagnostic analysis with plots and summaries."""
    params = StrategyParams()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ================================================================= LOAD
    logger.info("=" * 70)
    logger.info("STEP 1: LOADING DATA")
    logger.info("=" * 70)
    ce_df, pe_df = load_data(strike_filter=params.strike_filter)
    panel = build_intraday_panel(ce_df, pe_df)

    # ================================================================= RUN
    logger.info("\n" + "=" * 70)
    logger.info("STEP 2: RUNNING BACKTEST")
    logger.info("=" * 70)
    trades = run_backtest(panel, params)
    trade_df = trades_to_dataframe(trades)

    if trade_df.empty:
        logger.error("No trades generated. Exiting.")
        sys.exit(1)

    # ================================================================= STANDARD METRICS
    logger.info("\n" + "=" * 70)
    logger.info("STEP 3: COMPUTING STANDARD METRICS")
    logger.info("=" * 70)
    metrics = compute_metrics(trade_df, params)
    for k, v in metrics.items():
        if k != "exit_reason_breakdown":
            logger.info("  %-30s : %s", k, v)

    # ================================================================= DIAGNOSTICS
    logger.info("\n" + "=" * 70)
    logger.info("STEP 4: RUNNING FULL DIAGNOSTICS")
    logger.info("=" * 70)
    diag_results, pnl_dist, holding_dist = run_diagnostics(panel, trade_df, OUTPUT_DIR)

    # ================================================================= EXPORTS
    logger.info("\n" + "=" * 70)
    logger.info("STEP 5: EXPORTING RESULTS")
    logger.info("=" * 70)

    # Trade log
    trade_csv = OUTPUT_DIR / "trade_log.csv"
    trade_df.to_csv(trade_csv, index=False)
    logger.info("Trade log → %s", trade_csv)

    # Diagnostic metrics
    diag_export = OUTPUT_DIR / "diagnostic_metrics.csv"
    diag_df = pd.DataFrame([diag_results])
    diag_df.to_csv(diag_export, index=False)
    logger.info("Diagnostic metrics → %s", diag_export)

    # PnL distribution
    pnl_export = OUTPUT_DIR / "pnl_distribution.csv"
    pnl_dist.to_csv(pnl_export, index=False)
    logger.info("PnL distribution → %s", pnl_export)

    # Holding time distribution
    holding_export = OUTPUT_DIR / "holding_time_distribution.csv"
    holding_dist.to_csv(holding_export, index=False)
    logger.info("Holding time distribution → %s", holding_export)

    # ================================================================= PRINT SUMMARY
    logger.info("\n" + "=" * 70)
    logger.info("KEY FINDINGS – WHY THE STRATEGY LOSES MONEY")
    logger.info("=" * 70)
    
    # Core performance metrics
    total_return = metrics.get("total_return", 0)
    sharpe = metrics.get("sharpe_ratio", 0)
    win_rate = metrics.get("win_rate", 0)
    
    logger.info("\n[PERFORMANCE SUMMARY]")
    logger.info("  Total Return:              %.2f%%", total_return * 100)
    logger.info("  Sharpe Ratio:              %.2f", sharpe)
    logger.info("  Win Rate:                  %.1f%%", win_rate * 100)
    
    # Exit breakdown
    logger.info("\n[EXIT PATTERN]")
    logger.info("  Profit Target Hits:        %d (%.1f%%)",
               diag_results.get("exits_profit_target_count", 0),
               diag_results.get("exits_profit_target_pct", 0))
    logger.info("  Stop Loss Hits:            %d (%.1f%%)",
               diag_results.get("exits_stop_loss_count", 0),
               diag_results.get("exits_stop_loss_pct", 0))
    logger.info("  Time Stops:                %d (%.1f%%)",
               diag_results.get("exits_time_stop_count", 0),
               diag_results.get("exits_time_stop_pct", 0))
    
    # Range analysis
    logger.info("\n[VOLATILITY ANALYSIS]")
    logger.info("  Avg Opening Range %%:     %.4f%%",
               diag_results.get("avg_opening_range_pct", 0))
    logger.info("  Avg Intraday Range %%:    %.4f%%",
               diag_results.get("avg_intraday_range_pct", 0))
    logger.info("  Intraday/Opening Ratio:    %.2f×",
               diag_results.get("ratio_intraday_to_opening", 0))
    
    # IV decay
    logger.info("\n[VOLATILITY DECAY]")
    logger.info("  Avg IV Change (Entry→Exit): %.4f",
               diag_results.get("avg_iv_change_blended", 0))
    logger.info("  IV Change – Wins:         %.4f",
               diag_results.get("avg_iv_change_blended_wins", 0))
    logger.info("  IV Change – Losses:       %.4f",
               diag_results.get("avg_iv_change_blended_losses", 0))
    
    # Cost analysis
    logger.info("\n[COST BURDEN]")
    logger.info("  Total Gross PnL:           ₹%.2f",
               diag_results.get("total_gross_pnl", 0))
    logger.info("  Total Transaction Costs:   ₹%.2f",
               diag_results.get("total_transaction_costs", 0))
    logger.info("  Total Net PnL:             ₹%.2f",
               diag_results.get("total_net_pnl", 0))
    logger.info("  Avg Cost per Trade:        ₹%.2f",
               diag_results.get("avg_txn_cost_per_trade", 0))
    
    # Correlation insights
    logger.info("\n[CORRELATION WITH PnL]")
    logger.info("  Opening Range %% vs PnL:   %.4f (weak/neutral)",
               diag_results.get("correlation_opening_range_pct_vs_pnl", 0))
    logger.info("  Intraday Range %% vs PnL:  %.4f",
               diag_results.get("correlation_intraday_range_pct_vs_pnl", 0))
    logger.info("  IV Change vs PnL:          %.4f",
               diag_results.get("correlation_iv_change_vs_pnl", 0))
    
    logger.info("\n" + "=" * 70)
    logger.info("Output files saved to: %s", OUTPUT_DIR)
    logger.info("=" * 70)
    logger.info("\nDiagnostic files:")
    logger.info("  • trade_log.csv                – Full trade-by-trade details")
    logger.info("  • diagnostic_metrics.csv       – All diagnostic metrics")
    logger.info("  • pnl_distribution.csv         – PnL histogram buckets")
    logger.info("  • holding_time_distribution.csv – Holding time buckets")
    logger.info("  • diagnostic_scatter.png       – 4-panel scatter analysis")
    logger.info("  • exit_distribution.png        – Exit reason breakdown")
    logger.info("  • cost_analysis.png            – Cost impact visualization")
    logger.info("  • range_comparison.png         – Win vs loss range comparison")


if __name__ == "__main__":
    run_full_diagnostic()
