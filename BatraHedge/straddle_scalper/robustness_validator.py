"""
Robustness & Edge Validation Module
====================================
Comprehensive parameter sensitivity, rolling windows, regime testing, and edge analysis.
"""

from __future__ import annotations

import logging
from itertools import product
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from .config import StrategyParams
from .data_loader import load_data, build_intraday_panel
from .backtester import run_backtest, trades_to_dataframe
from .metrics import compute_metrics

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# 1. PARAMETER SENSITIVITY GRID
# --------------------------------------------------------------------------- #

def parameter_sensitivity_grid(
    panel: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Test all combinations of parameters across reasonable ranges.
    Returns top configs ranked by Sharpe and Profit Factor.
    """
    # Define parameter ranges
    profit_targets = [0.03, 0.05, 0.07, 0.10]
    stop_losses = [0.04, 0.06, 0.08]
    exit_times = ["11:30:00", "12:00:00", "13:00:00", "14:00:00", "14:45:00"]
    iv_momentum_thresholds = [0.0, 0.003, 0.005, 0.010]
    
    results = []
    total_combos = (
        len(profit_targets) * len(stop_losses) * 
        len(exit_times) * len(iv_momentum_thresholds)
    )
    
    logger.info("Testing %d parameter combinations...", total_combos)
    combo_count = 0
    
    for pt, sl, et, iv_mom in product(
        profit_targets, stop_losses, exit_times, iv_momentum_thresholds
    ):
        combo_count += 1
        if combo_count % 20 == 0:
            logger.info("  %d/%d combinations tested", combo_count, total_combos)
        
        params = StrategyParams(
            profit_target=pt,
            stop_loss=sl,
            exit_time=et,
            iv_momentum_threshold=iv_mom,
        )
        
        trades = run_backtest(panel, params)
        trade_df = trades_to_dataframe(trades)
        
        if trade_df.empty:
            continue
        
        metrics = compute_metrics(trade_df, params)
        
        results.append({
            "profit_target": pt,
            "stop_loss": sl,
            "exit_time": et,
            "iv_momentum_threshold": iv_mom,
            "trades": metrics["total_trades"],
            "total_return": metrics["total_return"],
            "sharpe": metrics["sharpe_ratio"],
            "profit_factor": metrics["profit_factor"],
            "max_drawdown": metrics["max_drawdown"],
            "win_rate": metrics["win_rate"],
            "avg_win": metrics["avg_win"],
            "avg_loss": metrics["avg_loss"],
            "cagr": metrics["cagr"],
        })
    
    results_df = pd.DataFrame(results)
    
    # Rank by Sharpe (handle NaN/inf)
    results_df["sharpe_valid"] = results_df["sharpe"].replace([np.inf, -np.inf], np.nan)
    top_sharpe = results_df.nlargest(10, "sharpe_valid")[
        ["profit_target", "stop_loss", "exit_time", "iv_momentum_threshold",
         "trades", "total_return", "sharpe", "profit_factor", "max_drawdown", "win_rate"]
    ].copy()
    
    # Rank by Profit Factor
    results_df["pf_valid"] = results_df["profit_factor"].replace([np.inf, -np.inf], np.nan)
    top_pf = results_df.nlargest(10, "pf_valid")[
        ["profit_target", "stop_loss", "exit_time", "iv_momentum_threshold",
         "trades", "total_return", "sharpe", "profit_factor", "max_drawdown", "win_rate"]
    ].copy()
    
    logger.info("Top 10 by Sharpe found. Total valid combos: %d", len(results_df))
    return results_df, top_sharpe, top_pf


# --------------------------------------------------------------------------- #
# 2. ROLLING WINDOW VALIDATION
# --------------------------------------------------------------------------- #

def rolling_window_validation(
    panel: pd.DataFrame,
    params: StrategyParams,
    window_months: int = 6,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform 50/50 split validation and rolling window tests.
    """
    # 50/50 split
    mid_idx = len(panel) // 2
    panel_first_half = panel.iloc[:mid_idx].copy()
    panel_second_half = panel.iloc[mid_idx:].copy()
    
    trades_1 = run_backtest(panel_first_half, params)
    trades_2 = run_backtest(panel_second_half, params)
    
    df_trades_1 = trades_to_dataframe(trades_1)
    df_trades_2 = trades_to_dataframe(trades_2)
    
    metrics_1 = compute_metrics(df_trades_1, params) if not df_trades_1.empty else {}
    metrics_2 = compute_metrics(df_trades_2, params) if not df_trades_2.empty else {}
    
    split_results = pd.DataFrame([
        {
            "period": "First 50%",
            "start_date": panel_first_half["date"].min(),
            "end_date": panel_first_half["date"].max(),
            "trades": metrics_1.get("total_trades", 0),
            "total_return": metrics_1.get("total_return", 0),
            "sharpe": metrics_1.get("sharpe_ratio", np.nan),
            "profit_factor": metrics_1.get("profit_factor", np.nan),
        },
        {
            "period": "Second 50%",
            "start_date": panel_second_half["date"].min(),
            "end_date": panel_second_half["date"].max(),
            "trades": metrics_2.get("total_trades", 0),
            "total_return": metrics_2.get("total_return", 0),
            "sharpe": metrics_2.get("sharpe_ratio", np.nan),
            "profit_factor": metrics_2.get("profit_factor", np.nan),
        }
    ])
    
    # Rolling windows (6-month windows)
    rolling_results = []
    panel_sorted = panel.sort_values("date").reset_index(drop=True)
    dates = pd.to_datetime(panel_sorted["date"])
    
    # Group by month
    year_month = dates.dt.to_period("M")
    unique_months = year_month.unique()
    
    for i in range(len(unique_months) - window_months + 1):
        window_start_month = unique_months[i]
        window_end_month = unique_months[i + window_months - 1]
        
        mask = (year_month >= window_start_month) & (year_month <= window_end_month)
        window_panel = panel_sorted[mask].copy()
        
        if len(window_panel) < 10:
            continue
        
        trades_w = run_backtest(window_panel, params)
        df_trades_w = trades_to_dataframe(trades_w)
        
        if df_trades_w.empty:
            continue
        
        metrics_w = compute_metrics(df_trades_w, params)
        
        rolling_results.append({
            "window_start": window_start_month,
            "window_end": window_end_month,
            "trades": metrics_w.get("total_trades", 0),
            "total_return": metrics_w.get("total_return", 0),
            "sharpe": metrics_w.get("sharpe_ratio", np.nan),
            "profit_factor": metrics_w.get("profit_factor", np.nan),
        })
    
    rolling_df = pd.DataFrame(rolling_results) if rolling_results else pd.DataFrame()
    
    logger.info("Rolling window validation complete. Windows: %d", len(rolling_df))
    return split_results, rolling_df


# --------------------------------------------------------------------------- #
# 3. REGIME SEGMENTATION
# --------------------------------------------------------------------------- #

def regime_segmentation(
    panel: pd.DataFrame,
    params: StrategyParams,
) -> Dict[str, Dict]:
    """
    Segment trades by IV regime and range regime.
    """
    trades = run_backtest(panel, params)
    trade_df = trades_to_dataframe(trades)
    
    if trade_df.empty:
        return {}
    
    regime_results = {}
    
    # === IV Regime ===
    # Calculate daily IV change
    daily_iv = panel.groupby("date").agg({
        "ce_iv": "mean",
        "pe_iv": "mean",
    }).reset_index()
    daily_iv["iv_avg"] = (daily_iv["ce_iv"] + daily_iv["pe_iv"]) / 2
    daily_iv["iv_change"] = daily_iv["iv_avg"].diff()
    
    iv_rising_dates = set(daily_iv[daily_iv["iv_change"] > 0]["date"].values)
    iv_falling_dates = set(daily_iv[daily_iv["iv_change"] < 0]["date"].values)
    
    trades_iv_rising = trade_df[trade_df["date"].isin(iv_rising_dates)]
    trades_iv_falling = trade_df[trade_df["date"].isin(iv_falling_dates)]
    
    regime_results["iv_rising"] = {
        "trades": len(trades_iv_rising),
        "return": trades_iv_rising["net_pnl"].sum() / params.initial_capital if len(trades_iv_rising) > 0 else 0,
        "sharpe": compute_metrics(trades_iv_rising, params).get("sharpe_ratio", np.nan) if len(trades_iv_rising) > 0 else np.nan,
        "avg_pnl": trades_iv_rising["net_pnl"].mean() if len(trades_iv_rising) > 0 else 0,
    }
    
    regime_results["iv_falling"] = {
        "trades": len(trades_iv_falling),
        "return": trades_iv_falling["net_pnl"].sum() / params.initial_capital if len(trades_iv_falling) > 0 else 0,
        "sharpe": compute_metrics(trades_iv_falling, params).get("sharpe_ratio", np.nan) if len(trades_iv_falling) > 0 else np.nan,
        "avg_pnl": trades_iv_falling["net_pnl"].mean() if len(trades_iv_falling) > 0 else 0,
    }
    
    # === Range Regime ===
    daily_range = panel.groupby("date").agg({
        "ce_high": "max",
        "ce_low": "min",
        "pe_high": "max",
        "pe_low": "min",
        "spot": "mean",
    }).reset_index()
    daily_range["range_pct"] = (
        (daily_range["ce_high"] - daily_range["ce_low"] + 
         daily_range["pe_high"] - daily_range["pe_low"]) / 2 / daily_range["spot"]
    )
    
    range_high_threshold = daily_range["range_pct"].quantile(0.70)
    range_low_threshold = daily_range["range_pct"].quantile(0.30)
    
    high_range_dates = set(daily_range[daily_range["range_pct"] >= range_high_threshold]["date"].values)
    low_range_dates = set(daily_range[daily_range["range_pct"] <= range_low_threshold]["date"].values)
    
    trades_high_range = trade_df[trade_df["date"].isin(high_range_dates)]
    trades_low_range = trade_df[trade_df["date"].isin(low_range_dates)]
    
    regime_results["high_range"] = {
        "trades": len(trades_high_range),
        "return": trades_high_range["net_pnl"].sum() / params.initial_capital if len(trades_high_range) > 0 else 0,
        "sharpe": compute_metrics(trades_high_range, params).get("sharpe_ratio", np.nan) if len(trades_high_range) > 0 else np.nan,
        "avg_pnl": trades_high_range["net_pnl"].mean() if len(trades_high_range) > 0 else 0,
    }
    
    regime_results["low_range"] = {
        "trades": len(trades_low_range),
        "return": trades_low_range["net_pnl"].sum() / params.initial_capital if len(trades_low_range) > 0 else 0,
        "sharpe": compute_metrics(trades_low_range, params).get("sharpe_ratio", np.nan) if len(trades_low_range) > 0 else np.nan,
        "avg_pnl": trades_low_range["net_pnl"].mean() if len(trades_low_range) > 0 else 0,
    }
    
    logger.info("Regime segmentation complete.")
    return regime_results


# --------------------------------------------------------------------------- #
# 4. COST SENSITIVITY TEST
# --------------------------------------------------------------------------- #

def cost_sensitivity_test(
    panel: pd.DataFrame,
    top_configs: List[Dict],
    cost_multipliers: List[float] = [1.0, 0.5, 0.0],
) -> pd.DataFrame:
    """
    Test top configs with different cost levels.
    """
    results = []
    
    for config in top_configs[:5]:  # Top 5
        for cost_mult in cost_multipliers:
            params = StrategyParams(
                profit_target=config["profit_target"],
                stop_loss=config["stop_loss"],
                exit_time=config["exit_time"],
                iv_momentum_threshold=config["iv_momentum_threshold"],
                brokerage_per_leg=20.0 * cost_mult,
                slippage_pct=0.05 * cost_mult,
                stt_on_sell_pct=0.0625 * cost_mult,
            )
            
            trades = run_backtest(panel, params)
            trade_df = trades_to_dataframe(trades)
            
            if trade_df.empty:
                continue
            
            metrics = compute_metrics(trade_df, params)
            
            results.append({
                "config_id": f"PT{config['profit_target']*100:.0f}_SL{config['stop_loss']*100:.0f}",
                "cost_multiplier": cost_mult,
                "trades": metrics["total_trades"],
                "gross_pnl": trade_df["gross_pnl"].sum(),
                "net_pnl": metrics["total_pnl"],
                "total_return": metrics["total_return"],
                "sharpe": metrics["sharpe_ratio"],
            })
    
    results_df = pd.DataFrame(results)
    logger.info("Cost sensitivity test complete.")
    return results_df


# --------------------------------------------------------------------------- #
# 5. CONVEXITY TEST
# --------------------------------------------------------------------------- #

def convexity_test(trade_df: pd.DataFrame) -> Dict:
    """
    Analyze if strategy edge depends on rare large trades.
    """
    if trade_df.empty:
        return {}
    
    trade_df_sorted = trade_df.sort_values("net_pnl", ascending=False)
    
    top_10_pnl = trade_df_sorted.head(10)["net_pnl"].sum()
    total_pnl = trade_df["net_pnl"].sum()
    
    top_10_contribution = (top_10_pnl / total_pnl * 100) if total_pnl != 0 else 0
    
    # Pareto analysis
    sorted_pnl = np.sort(trade_df["net_pnl"].values)[::-1]
    cumsum_pnl = np.cumsum(sorted_pnl)
    cumsum_pnl_pct = cumsum_pnl / cumsum_pnl[-1] * 100 if cumsum_pnl[-1] != 0 else cumsum_pnl
    
    # Find how many trades needed for 80% of PnL
    trades_for_80pct = np.argmax(cumsum_pnl_pct >= 80) + 1 if any(cumsum_pnl_pct >= 80) else len(trade_df)
    
    results = {
        "top_10_contribution_pct": round(abs(top_10_contribution), 2),
        "total_trades": len(trade_df),
        "trades_for_80pct_of_pnl": trades_for_80pct,
        "top_10_pnl": round(top_10_pnl, 2),
        "total_pnl": round(total_pnl, 2),
        "is_edge_concentrated": abs(top_10_contribution) > 50,  # >50% from top 10 = fragile
    }
    
    logger.info("Convexity analysis complete. Top 10 trades contribute: %.1f%%", abs(top_10_contribution))
    return results


# --------------------------------------------------------------------------- #
# 6. VISUALIZATION FUNCTIONS
# --------------------------------------------------------------------------- #

def plot_sharpe_heatmap(
    results_df: pd.DataFrame,
    output_dir,
) -> None:
    """Create 2D heatmaps of Sharpe across parameter pairs."""
    # PT vs SL
    pivot_pt_sl = results_df.pivot_table(
        values="sharpe",
        index="stop_loss",
        columns="profit_target",
        aggfunc="mean"
    )
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot_pt_sl, annot=True, fmt=".2f", cmap="RdYlGn", center=0, ax=ax)
    ax.set_title("Sharpe Ratio: Profit Target vs Stop Loss", fontweight="bold", fontsize=13)
    ax.set_xlabel("Profit Target")
    ax.set_ylabel("Stop Loss")
    fig.tight_layout()
    fig.savefig(output_dir / "heatmap_pt_sl.png", dpi=150)
    plt.close(fig)
    
    # Exit Time vs IV Momentum
    pivot_et_iv = results_df.pivot_table(
        values="sharpe",
        index="iv_momentum_threshold",
        columns="exit_time",
        aggfunc="mean"
    )
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(pivot_et_iv, annot=True, fmt=".2f", cmap="RdYlGn", center=0, ax=ax)
    ax.set_title("Sharpe Ratio: Exit Time vs IV Momentum Threshold", fontweight="bold", fontsize=13)
    ax.set_xlabel("Exit Time")
    ax.set_ylabel("IV Momentum Threshold")
    fig.tight_layout()
    fig.savefig(output_dir / "heatmap_et_iv.png", dpi=150)
    plt.close(fig)


def plot_rolling_performance(
    split_results: pd.DataFrame,
    rolling_df: pd.DataFrame,
    output_dir,
) -> None:
    """Plot rolling window returns and Sharpe."""
    if rolling_df.empty:
        return
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # Rolling return
    ax = axes[0]
    ax.plot(rolling_df.index, rolling_df["total_return"] * 100, marker="o", linewidth=2, color="#1f77b4")
    ax.axhline(0, color="red", linestyle="--", linewidth=1)
    ax.fill_between(rolling_df.index, rolling_df["total_return"] * 100, 0, alpha=0.3)
    ax.set_title("Rolling 6-Month Returns", fontweight="bold", fontsize=12)
    ax.set_ylabel("Return (%)")
    ax.grid(alpha=0.3)
    
    # Rolling Sharpe
    ax = axes[1]
    ax.plot(rolling_df.index, rolling_df["sharpe"], marker="s", linewidth=2, color="#ff7f0e")
    ax.axhline(0, color="red", linestyle="--", linewidth=1)
    ax.set_title("Rolling 6-Month Sharpe Ratio", fontweight="bold", fontsize=12)
    ax.set_ylabel("Sharpe")
    ax.set_xlabel("Window")
    ax.grid(alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(output_dir / "rolling_performance.png", dpi=150)
    plt.close(fig)


def plot_regime_performance(
    regime_results: Dict,
    output_dir,
) -> None:
    """Plot regime performance comparison."""
    if not regime_results:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    
    # IV Regime
    ax = axes[0, 0]
    iv_regimes = ["iv_rising", "iv_falling"]
    iv_returns = [regime_results[r]["return"] * 100 for r in iv_regimes if r in regime_results]
    colors_iv = ["#2ca02c", "#d62728"]
    ax.bar(iv_regimes, iv_returns, color=colors_iv, alpha=0.7, edgecolor="black")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title("Return by IV Regime", fontweight="bold")
    ax.set_ylabel("Return (%)")
    ax.grid(alpha=0.3, axis="y")
    
    # Range Regime
    ax = axes[0, 1]
    range_regimes = ["high_range", "low_range"]
    range_returns = [regime_results[r]["return"] * 100 for r in range_regimes if r in regime_results]
    colors_range = ["#1f77b4", "#ff7f0e"]
    ax.bar(range_regimes, range_returns, color=colors_range, alpha=0.7, edgecolor="black")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title("Return by Range Regime", fontweight="bold")
    ax.set_ylabel("Return (%)")
    ax.grid(alpha=0.3, axis="y")
    
    # IV Regime Sharpe
    ax = axes[1, 0]
    iv_sharpes = [regime_results.get(r, {}).get("sharpe", 0) for r in iv_regimes]
    ax.bar(iv_regimes, iv_sharpes, color=colors_iv, alpha=0.7, edgecolor="black")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title("Sharpe by IV Regime", fontweight="bold")
    ax.set_ylabel("Sharpe")
    ax.grid(alpha=0.3, axis="y")
    
    # Range Regime Sharpe
    ax = axes[1, 1]
    range_sharpes = [regime_results.get(r, {}).get("sharpe", 0) for r in range_regimes]
    ax.bar(range_regimes, range_sharpes, color=colors_range, alpha=0.7, edgecolor="black")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title("Sharpe by Range Regime", fontweight="bold")
    ax.set_ylabel("Sharpe")
    ax.grid(alpha=0.3, axis="y")
    
    fig.tight_layout()
    fig.savefig(output_dir / "regime_performance.png", dpi=150)
    plt.close(fig)


def plot_cost_sensitivity(
    cost_sensitivity_df: pd.DataFrame,
    output_dir,
) -> None:
    """Plot cost sensitivity for top configs."""
    if cost_sensitivity_df.empty:
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    config_ids = cost_sensitivity_df["config_id"].unique()
    cost_mults = sorted(cost_sensitivity_df["cost_multiplier"].unique())
    
    x = np.arange(len(config_ids))
    width = 0.25
    
    for i, cm in enumerate(cost_mults):
        subset = cost_sensitivity_df[cost_sensitivity_df["cost_multiplier"] == cm]
        returns = []
        for cid in config_ids:
            val = subset[subset["config_id"] == cid]["total_return"].values
            returns.append(val[0] * 100 if len(val) > 0 else 0)
        
        label = f"{cm*100:.0f}% Costs" if cm > 0 else "Zero Costs"
        ax.bar(x + i*width, returns, width, label=label, alpha=0.8)
    
    ax.axhline(0, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Configuration")
    ax.set_ylabel("Return (%)")
    ax.set_title("Cost Sensitivity: Top 5 Configs", fontweight="bold", fontsize=13)
    ax.set_xticks(x + width)
    ax.set_xticklabels(config_ids, rotation=45, ha="right")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(output_dir / "cost_sensitivity.png", dpi=150)
    plt.close(fig)


def plot_convexity(
    trade_df: pd.DataFrame,
    output_dir,
) -> None:
    """Pareto plot of trade contributions."""
    if trade_df.empty:
        return
    
    sorted_pnl = np.sort(trade_df["net_pnl"].values)[::-1]
    cumsum_pnl = np.cumsum(sorted_pnl)
    cumsum_pnl_pct = cumsum_pnl / cumsum_pnl[-1] * 100
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(range(1, len(cumsum_pnl_pct) + 1), cumsum_pnl_pct, linewidth=2, marker="o", markersize=3, color="#1f77b4")
    ax.axhline(80, color="red", linestyle="--", linewidth=1.5, label="80% threshold")
    ax.fill_between(range(1, len(cumsum_pnl_pct) + 1), cumsum_pnl_pct, alpha=0.3)
    ax.set_xlabel("# Trades (sorted by PnL)")
    ax.set_ylabel("Cumulative % of Total PnL")
    ax.set_title("Pareto Chart: Trade Contribution to Total PnL", fontweight="bold", fontsize=13)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "pareto_convexity.png", dpi=150)
    plt.close(fig)
