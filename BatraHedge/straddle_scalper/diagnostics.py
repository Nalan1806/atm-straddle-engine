"""
Diagnostic Analysis Module
===========================
Comprehensive post-backtest analysis to identify why the strategy loses money.
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# 1. DATASET OVERVIEW
# --------------------------------------------------------------------------- #

def dataset_overview(
    panel: pd.DataFrame,
    trade_df: pd.DataFrame,
) -> Dict[str, float | int]:
    """Compute dataset statistics."""
    results = {}
    
    total_days = panel["date"].nunique()
    traded_days = trade_df["date"].nunique()
    pct_traded = (traded_days / total_days * 100) if total_days > 0 else 0
    
    # Average trades per month
    date_range = panel["date"].max() - panel["date"].min()
    months = date_range.days / 30.44
    trades_per_month = len(trade_df) / months if months > 0 else 0
    
    results["total_unique_days"] = int(total_days)
    results["days_meeting_compression_filter"] = int(traded_days)
    results["pct_days_traded"] = round(pct_traded, 2)
    results["avg_trades_per_month"] = round(trades_per_month, 2)
    results["total_trades"] = len(trade_df)
    results["backtest_days"] = int(date_range.days)
    
    return results


# --------------------------------------------------------------------------- #
# 2. ENTRY CONDITION DIAGNOSTICS
# --------------------------------------------------------------------------- #

def entry_diagnostics(trade_df: pd.DataFrame) -> Dict:
    """Analyze entry conditions and their correlation with outcomes."""
    results = {}
    
    # Range statistics
    results["avg_opening_range_pct"] = round(
        trade_df["opening_range_pct"].mean() * 100, 4
    )
    results["avg_intraday_range_pct"] = round(
        trade_df["intraday_range_pct"].mean() * 100, 4
    )
    results["ratio_intraday_to_opening"] = round(
        (trade_df["intraday_range_pct"].mean() / 
         trade_df["opening_range_pct"].mean())
        if trade_df["opening_range_pct"].mean() > 0 else 0,
        2
    )
    
    # Split by outcome
    wins = trade_df[trade_df["net_pnl"] > 0]
    losses = trade_df[trade_df["net_pnl"] <= 0]
    
    results["avg_opening_range_pct_wins"] = round(
        wins["opening_range_pct"].mean() * 100, 4
    ) if len(wins) > 0 else 0
    
    results["avg_opening_range_pct_losses"] = round(
        losses["opening_range_pct"].mean() * 100, 4
    ) if len(losses) > 0 else 0
    
    results["avg_intraday_range_pct_wins"] = round(
        wins["intraday_range_pct"].mean() * 100, 4
    ) if len(wins) > 0 else 0
    
    results["avg_intraday_range_pct_losses"] = round(
        losses["intraday_range_pct"].mean() * 100, 4
    ) if len(losses) > 0 else 0
    
    # Correlation analysis
    corr_or = trade_df["opening_range_pct"].corr(trade_df["net_pnl"])
    corr_ir = trade_df["intraday_range_pct"].corr(trade_df["net_pnl"])
    
    results["correlation_opening_range_pct_vs_pnl"] = round(corr_or, 4)
    results["correlation_intraday_range_pct_vs_pnl"] = round(corr_ir, 4)
    
    return results


# --------------------------------------------------------------------------- #
# 3. EXIT BREAKDOWN
# --------------------------------------------------------------------------- #

def exit_breakdown(trade_df: pd.DataFrame) -> Dict:
    """Analyze exit conditions and their profitability."""
    results = {}
    
    total = len(trade_df)
    exit_counts = trade_df["exit_reason"].value_counts()
    
    for reason in ["profit_target", "stop_loss", "time_stop"]:
        subset = trade_df[trade_df["exit_reason"] == reason]
        count = len(subset)
        pct = (count / total * 100) if total > 0 else 0
        
        results[f"exits_{reason}_count"] = int(count)
        results[f"exits_{reason}_pct"] = round(pct, 2)
        results[f"exits_{reason}_avg_pnl"] = round(subset["net_pnl"].mean(), 2)
        results[f"exits_{reason}_median_pnl"] = round(subset["net_pnl"].median(), 2)
        results[f"exits_{reason}_std_pnl"] = round(subset["net_pnl"].std(), 2)
    
    return results


# --------------------------------------------------------------------------- #
# 4. VOLATILITY & IV ANALYSIS
# --------------------------------------------------------------------------- #

def volatility_analysis(trade_df: pd.DataFrame) -> Dict:
    """Analyze IV changes and their impact."""
    results = {}
    
    # Average IVs
    results["avg_iv_ce_entry"] = round(trade_df["iv_ce_entry"].mean(), 4)
    results["avg_iv_pe_entry"] = round(trade_df["iv_pe_entry"].mean(), 4)
    results["avg_iv_ce_exit"] = round(trade_df["iv_ce_exit"].mean(), 4)
    results["avg_iv_pe_exit"] = round(trade_df["iv_pe_exit"].mean(), 4)
    
    # IV changes
    trade_df["iv_ce_change"] = trade_df["iv_ce_exit"] - trade_df["iv_ce_entry"]
    trade_df["iv_pe_change"] = trade_df["iv_pe_exit"] - trade_df["iv_pe_entry"]
    trade_df["iv_avg_entry"] = (trade_df["iv_ce_entry"] + trade_df["iv_pe_entry"]) / 2
    trade_df["iv_avg_exit"] = (trade_df["iv_ce_exit"] + trade_df["iv_pe_exit"]) / 2
    trade_df["iv_blended_change"] = trade_df["iv_avg_exit"] - trade_df["iv_avg_entry"]
    
    results["avg_iv_change_ce"] = round(trade_df["iv_ce_change"].mean(), 4)
    results["avg_iv_change_pe"] = round(trade_df["iv_pe_change"].mean(), 4)
    results["avg_iv_change_blended"] = round(trade_df["iv_blended_change"].mean(), 4)
    
    # Impact on winning vs losing trades
    wins = trade_df[trade_df["net_pnl"] > 0]
    losses = trade_df[trade_df["net_pnl"] <= 0]
    
    results["avg_iv_change_blended_wins"] = round(
        wins["iv_blended_change"].mean(), 4
    ) if len(wins) > 0 else 0
    
    results["avg_iv_change_blended_losses"] = round(
        losses["iv_blended_change"].mean(), 4
    ) if len(losses) > 0 else 0
    
    # Correlation: IV decay vs PnL
    corr_iv = trade_df["iv_blended_change"].corr(trade_df["net_pnl"])
    results["correlation_iv_change_vs_pnl"] = round(corr_iv, 4)
    
    return results


# --------------------------------------------------------------------------- #
# 5. COST IMPACT ANALYSIS
# --------------------------------------------------------------------------- #

def cost_impact_analysis(trade_df: pd.DataFrame) -> Dict:
    """Measure the cost burden."""
    results = {}
    
    total_pnl_net = trade_df["net_pnl"].sum()
    total_pnl_gross = trade_df["gross_pnl"].sum()
    total_costs = trade_df["txn_cost"].sum()
    
    avg_cost_per_trade = trade_df["txn_cost"].mean()
    avg_cost_pct_of_entry_premium = (
        (trade_df["txn_cost"] / (trade_df["entry_premium"] * trade_df.index.size / len(trade_df))).mean()
        if len(trade_df) > 0 else 0
    )
    
    # More accurate: cost as % of entry premium
    cost_pct = []
    for _, row in trade_df.iterrows():
        if row["entry_premium"] > 0:
            cost_pct.append(row["txn_cost"] / row["entry_premium"])
    
    avg_cost_pct = np.mean(cost_pct) * 100 if cost_pct else 0
    
    results["total_gross_pnl"] = round(total_pnl_gross, 2)
    results["total_transaction_costs"] = round(total_costs, 2)
    results["total_net_pnl"] = round(total_pnl_net, 2)
    results["cost_as_pct_of_gross_pnl"] = round(abs(total_costs / total_pnl_gross * 100) if total_pnl_gross != 0 else 0, 2)
    results["avg_txn_cost_per_trade"] = round(avg_cost_per_trade, 2)
    results["avg_txn_cost_pct_of_entry_premium"] = round(avg_cost_pct, 4)
    
    return results


# --------------------------------------------------------------------------- #
# 6. DISTRIBUTION ANALYSIS
# --------------------------------------------------------------------------- #

def distribution_analysis(trade_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return distribution summary stats."""
    pnl_dist = pd.DataFrame({
        "bin": ["< -1000", "-1000 to -500", "-500 to -100", "-100 to 0", 
                "0 to 100", "100 to 500", "500 to 1000", "> 1000"],
        "count": [
            len(trade_df[trade_df["net_pnl"] < -1000]),
            len(trade_df[(trade_df["net_pnl"] >= -1000) & (trade_df["net_pnl"] < -500)]),
            len(trade_df[(trade_df["net_pnl"] >= -500) & (trade_df["net_pnl"] < -100)]),
            len(trade_df[(trade_df["net_pnl"] >= -100) & (trade_df["net_pnl"] < 0)]),
            len(trade_df[(trade_df["net_pnl"] >= 0) & (trade_df["net_pnl"] < 100)]),
            len(trade_df[(trade_df["net_pnl"] >= 100) & (trade_df["net_pnl"] < 500)]),
            len(trade_df[(trade_df["net_pnl"] >= 500) & (trade_df["net_pnl"] < 1000)]),
            len(trade_df[trade_df["net_pnl"] >= 1000]),
        ]
    })
    
    holding_dist = pd.DataFrame({
        "bucket": ["0-60 min", "60-120 min", "120-180 min", "180-240 min", "> 240 min"],
        "count": [
            len(trade_df[trade_df["holding_minutes"] <= 60]),
            len(trade_df[(trade_df["holding_minutes"] > 60) & (trade_df["holding_minutes"] <= 120)]),
            len(trade_df[(trade_df["holding_minutes"] > 120) & (trade_df["holding_minutes"] <= 180)]),
            len(trade_df[(trade_df["holding_minutes"] > 180) & (trade_df["holding_minutes"] <= 240)]),
            len(trade_df[trade_df["holding_minutes"] > 240]),
        ]
    })
    
    return pnl_dist, holding_dist


# --------------------------------------------------------------------------- #
# 7. VISUALIZATION
# --------------------------------------------------------------------------- #

def create_diagnostic_plots(
    trade_df: pd.DataFrame,
    output_dir,
) -> None:
    """Generate comprehensive diagnostic plots."""
    
    # Plot 1: Trade PnL Distribution
    fig1, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    ax = axes[0, 0]
    ax.hist(trade_df["net_pnl"], bins=40, edgecolor="black", alpha=0.7, color="#1f77b4")
    ax.axvline(0, color="red", linestyle="--", linewidth=1)
    ax.set_title("Trade PnL Distribution", fontweight="bold")
    ax.set_xlabel("Net PnL (INR)")
    ax.set_ylabel("Frequency")
    ax.grid(alpha=0.3)
    
    # Plot 2: Holding Time Distribution
    ax = axes[0, 1]
    ax.hist(trade_df["holding_minutes"], bins=30, edgecolor="black", alpha=0.7, color="#2ca02c")
    ax.set_title("Holding Time Distribution", fontweight="bold")
    ax.set_xlabel("Minutes")
    ax.set_ylabel("Frequency")
    ax.grid(alpha=0.3)
    
    # Plot 3: Opening Range vs PnL
    ax = axes[1, 0]
    scatter_colors = ["#2ca02c" if p > 0 else "#d62728" for p in trade_df["net_pnl"]]
    ax.scatter(trade_df["opening_range_pct"] * 100, trade_df["net_pnl"], 
              c=scatter_colors, alpha=0.6, s=50)
    ax.axhline(0, color="black", linestyle="-", linewidth=0.5)
    ax.set_title("Opening Range % vs Trade PnL", fontweight="bold")
    ax.set_xlabel("Opening Range (%)")
    ax.set_ylabel("Net PnL (INR)")
    ax.grid(alpha=0.3)
    
    # Plot 4: IV Change vs PnL
    ax = axes[1, 1]
    trade_df["iv_blended_change"] = (trade_df["iv_ce_exit"] - trade_df["iv_ce_entry"] + 
                                     trade_df["iv_pe_exit"] - trade_df["iv_pe_entry"]) / 2
    ax.scatter(trade_df["iv_blended_change"], trade_df["net_pnl"],
              c=scatter_colors, alpha=0.6, s=50)
    ax.axhline(0, color="black", linestyle="-", linewidth=0.5)
    ax.axvline(0, color="black", linestyle="--", linewidth=0.5)
    ax.set_title("IV Change (avg) vs Trade PnL", fontweight="bold")
    ax.set_xlabel("IV Change (blended)")
    ax.set_ylabel("Net PnL (INR)")
    ax.grid(alpha=0.3)
    
    fig1.tight_layout()
    fig1.savefig(output_dir / "diagnostic_scatter.png", dpi=150)
    plt.close(fig1)
    
    # Plot 5: Exit Distribution
    fig2, ax = plt.subplots(figsize=(10, 6))
    exit_counts = trade_df["exit_reason"].value_counts()
    colors_exits = ["#2ca02c", "#d62728", "#ff7f0e"]
    ax.bar(exit_counts.index, exit_counts.values, color=colors_exits, alpha=0.7, edgecolor="black")
    ax.set_title("Exit Reason Distribution", fontweight="bold", fontsize=12)
    ax.set_ylabel("Count")
    for i, v in enumerate(exit_counts.values):
        ax.text(i, v + 1, str(v), ha="center", fontweight="bold")
    fig2.tight_layout()
    fig2.savefig(output_dir / "exit_distribution.png", dpi=150)
    plt.close(fig2)
    
    # Plot 6: Cost Impact
    fig3, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    ax = axes[0]
    data = [
        trade_df["gross_pnl"].sum(),
        -trade_df["txn_cost"].sum(),
    ]
    labels = ["Gross PnL", "Transaction Costs"]
    colors = ["#2ca02c", "#d62728"]
    ax.bar(labels, data, color=colors, alpha=0.7, edgecolor="black")
    ax.set_title("Gross PnL vs Transaction Costs", fontweight="bold")
    ax.set_ylabel("INR")
    ax.axhline(0, color="black", linewidth=0.5)
    for i, v in enumerate(data):
        ax.text(i, v + (abs(data[i] * 0.02) if data[i] > 0 else -abs(data[i] * 0.05)), 
               f"₹{v:,.0f}", ha="center", fontweight="bold")
    ax.grid(alpha=0.3, axis="y")
    
    ax = axes[1]
    cost_pct_per_trade = (trade_df["txn_cost"] / trade_df["entry_premium"] * 100)
    ax.hist(cost_pct_per_trade, bins=30, edgecolor="black", alpha=0.7, color="#ff7f0e")
    ax.set_title("Transaction Cost % of Entry Premium", fontweight="bold")
    ax.set_xlabel("Cost %")
    ax.set_ylabel("Frequency")
    ax.grid(alpha=0.3)
    
    fig3.tight_layout()
    fig3.savefig(output_dir / "cost_analysis.png", dpi=150)
    plt.close(fig3)
    
    # Plot 7: Win vs Loss Box Plot
    fig4, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    wins = trade_df[trade_df["net_pnl"] > 0]
    losses = trade_df[trade_df["net_pnl"] <= 0]
    
    ax = axes[0]
    bp = ax.boxplot([wins["opening_range_pct"] * 100, losses["opening_range_pct"] * 100],
                     labels=["Winning Trades", "Losing Trades"],
                     patch_artist=True)
    for patch, color in zip(bp["boxes"], ["#2ca02c", "#d62728"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_title("Opening Range % – Win vs Loss", fontweight="bold")
    ax.set_ylabel("Opening Range (%)")
    ax.grid(alpha=0.3, axis="y")
    
    ax = axes[1]
    bp = ax.boxplot([wins["intraday_range_pct"] * 100, losses["intraday_range_pct"] * 100],
                     labels=["Winning Trades", "Losing Trades"],
                     patch_artist=True)
    for patch, color in zip(bp["boxes"], ["#2ca02c", "#d62728"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_title("Intraday Range % – Win vs Loss", fontweight="bold")
    ax.set_ylabel("Intraday Range (%)")
    ax.grid(alpha=0.3, axis="y")
    
    fig4.tight_layout()
    fig4.savefig(output_dir / "range_comparison.png", dpi=150)
    plt.close(fig4)
    
    logger.info("Diagnostic plots saved.")


# --------------------------------------------------------------------------- #
# 8. MAIN DIAGNOSTIC RUNNER
# --------------------------------------------------------------------------- #

def run_diagnostics(
    panel: pd.DataFrame,
    trade_df: pd.DataFrame,
    output_dir,
) -> Dict:
    """Execute full diagnostic analysis."""
    
    all_results = {}
    
    logger.info("=== 1. Dataset Overview ===")
    do = dataset_overview(panel, trade_df)
    all_results.update(do)
    for k, v in do.items():
        logger.info("  %-40s : %s", k, v)
    
    logger.info("\n=== 2. Entry Condition Diagnostics ===")
    ed = entry_diagnostics(trade_df)
    all_results.update(ed)
    for k, v in ed.items():
        logger.info("  %-40s : %s", k, v)
    
    logger.info("\n=== 3. Exit Breakdown ===")
    eb = exit_breakdown(trade_df)
    all_results.update(eb)
    for k, v in eb.items():
        logger.info("  %-40s : %s", k, v)
    
    logger.info("\n=== 4. Volatility & IV Analysis ===")
    va = volatility_analysis(trade_df)
    all_results.update(va)
    for k, v in va.items():
        logger.info("  %-40s : %s", k, v)
    
    logger.info("\n=== 5. Cost Impact ===")
    ci = cost_impact_analysis(trade_df)
    all_results.update(ci)
    for k, v in ci.items():
        logger.info("  %-40s : %s", k, v)
    
    logger.info("\n=== 6. Distribution Analysis ===")
    pnl_dist, holding_dist = distribution_analysis(trade_df)
    logger.info("PnL Distribution:")
    for _, row in pnl_dist.iterrows():
        logger.info("  %-30s : %d", row["bin"], row["count"])
    logger.info("Holding Time Distribution:")
    for _, row in holding_dist.iterrows():
        logger.info("  %-30s : %d", row["bucket"], row["count"])
    
    logger.info("\n=== 7. Generating Plots ===")
    create_diagnostic_plots(trade_df, output_dir)
    
    return all_results, pnl_dist, holding_dist
