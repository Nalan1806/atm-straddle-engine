"""
Short Straddle — Cutthroat Robustness Validation
=================================================
Capital/Margin analysis, Gross vs Net, Tail Risk,
Rolling Stability, Sensitivity Heatmaps.
"""

from __future__ import annotations

import logging
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from .config import StrategyParams, OUTPUT_DIR
from .data_loader import load_data, build_intraday_panel
from .backtester import run_backtest, trades_to_dataframe, compute_txn_cost
from .metrics import compute_metrics, build_equity_curve

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

OUT = OUTPUT_DIR / "validation"


# =========================================================================== #
#  HELPERS
# =========================================================================== #

def _sharpe(daily_pnl: pd.Series, capital: float) -> float:
    dr = daily_pnl / capital
    return dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0.0


def _profit_factor(pnl_series: pd.Series) -> float:
    wins = pnl_series[pnl_series > 0].sum()
    losses = abs(pnl_series[pnl_series <= 0].sum())
    return wins / losses if losses > 0 else float("inf")


def _run_with_params(panel, **overrides):
    """Run short straddle with specific parameter overrides, return trade_df."""
    kw = dict(
        strategy_type="Short Straddle",
        compression_filter_enabled=True,
        compression_threshold=0.004,
        short_profit_target_pct=0.10,
        short_stop_loss_pct=0.20,
        breakout_buffer_pct=0.002,
        exit_time="14:30:00",
        brokerage_per_leg=20.0,
        slippage_pct=0.05,
        stt_on_sell_pct=0.0625,
        initial_capital=1_000_000.0,
        lot_size=25,
    )
    kw.update(overrides)
    params = StrategyParams(**kw)
    trades = run_backtest(panel, params)
    return trades_to_dataframe(trades), params


# =========================================================================== #
#  SECTION 1 — CAPITAL & MARGIN
# =========================================================================== #

def section_1_capital_margin(trade_df: pd.DataFrame, params: StrategyParams):
    log.info("=" * 80)
    log.info("SECTION 1: CAPITAL & MARGIN CLARIFICATION")
    log.info("=" * 80)

    capital = params.initial_capital
    lot = params.lot_size

    # Short straddle margin ≈ max(CE, PE) premium + OTM premium + SPAN margin
    # NIFTY short option SPAN ≈ ~₹1.0-1.5L per lot (we use a realistic estimate)
    # Approximate: margin ≈ spot * lot * margin_pct (typically ~12-15% for index options)
    avg_spot = trade_df["spot_at_entry"].mean()
    avg_entry_premium = trade_df["entry_premium"].mean()

    # SEBI margin for short options: ~12-15% of notional + premium received
    # Conservative estimate: margin per lot = spot * lot * 0.12 + premium * lot
    # But since we sell straddle (2 legs), margin is roughly for the riskier side
    # Practical NIFTY short straddle margin: ~₹1.2-1.5L per lot
    span_margin_per_lot = avg_spot * lot * 0.12  # ~12% of notional
    exposure_margin = avg_spot * lot * 0.03      # ~3% additional
    premium_margin = avg_entry_premium * lot     # premium received (reduces margin slightly)
    estimated_margin_per_trade = span_margin_per_lot + exposure_margin

    total_pnl = trade_df["net_pnl"].sum()
    gross_pnl = trade_df["gross_pnl"].sum()
    n_trades = len(trade_df)
    days_elapsed = (trade_df["date"].max() - trade_df["date"].min()).days
    years = days_elapsed / 365.25

    rom = total_pnl / estimated_margin_per_trade  # return on margin (total, over full period)
    rom_annualised = (1 + rom) ** (1 / years) - 1 if (1 + rom) > 0 else -1.0

    # Capital utilisation
    capital_utilised_pct = estimated_margin_per_trade / capital * 100

    log.info("")
    log.info("  Capital base assumed:          ₹%s", f"{capital:,.0f}")
    log.info("  Lot size:                      %d", lot)
    log.info("  Avg spot at entry:             ₹%s", f"{avg_spot:,.0f}")
    log.info("  Avg entry premium (CE+PE):     ₹%.2f", avg_entry_premium)
    log.info("  Avg premium received (lot):    ₹%s", f"{avg_entry_premium * lot:,.0f}")
    log.info("")
    log.info("  ESTIMATED MARGIN PER TRADE:")
    log.info("    SPAN margin (~12%%):          ₹%s", f"{span_margin_per_lot:,.0f}")
    log.info("    Exposure margin (~3%%):       ₹%s", f"{exposure_margin:,.0f}")
    log.info("    Total margin required:       ₹%s", f"{estimated_margin_per_trade:,.0f}")
    log.info("    Capital utilisation:          %.1f%%", capital_utilised_pct)
    log.info("")
    log.info("  RETURN ON CAPITAL:")
    log.info("    Net PnL:                     ₹%s", f"{total_pnl:,.2f}")
    log.info("    Return on ₹10L capital:      %.2f%%", total_pnl / capital * 100)
    log.info("    CAGR on capital:             %.2f%%", ((capital + total_pnl) / capital) ** (1 / years) * 100 - 100)
    log.info("")
    log.info("  RETURN ON MARGIN (the real number):")
    log.info("    Total ROM:                   %.2f%%", rom * 100)
    log.info("    Annualised ROM:              %.2f%%", rom_annualised * 100)
    log.info("    This is the actual return on deployed capital.")
    log.info("")

    return {
        "capital": capital,
        "estimated_margin": round(estimated_margin_per_trade, 0),
        "capital_utilisation_pct": round(capital_utilised_pct, 1),
        "total_rom": round(rom * 100, 2),
        "annualised_rom": round(rom_annualised * 100, 2),
        "avg_premium_received_per_lot": round(avg_entry_premium * lot, 2),
    }


# =========================================================================== #
#  SECTION 2 — GROSS vs NET EDGE
# =========================================================================== #

def section_2_gross_vs_net(trade_df: pd.DataFrame, params: StrategyParams):
    log.info("=" * 80)
    log.info("SECTION 2: GROSS vs NET EDGE")
    log.info("=" * 80)

    capital = params.initial_capital
    n = len(trade_df)

    gross_pnl = trade_df["gross_pnl"].sum()
    total_costs = trade_df["txn_cost"].sum()
    net_pnl = trade_df["net_pnl"].sum()

    # Gross metrics
    daily_gross = trade_df.groupby("date")["gross_pnl"].sum()
    sharpe_gross = _sharpe(daily_gross, capital)
    pf_gross = _profit_factor(trade_df["gross_pnl"])

    # Net metrics
    daily_net = trade_df.groupby("date")["net_pnl"].sum()
    sharpe_net = _sharpe(daily_net, capital)
    pf_net = _profit_factor(trade_df["net_pnl"])

    avg_cost_per_trade = total_costs / n
    cost_pct_of_gross = total_costs / gross_pnl * 100 if gross_pnl != 0 else float("inf")
    avg_entry = trade_df["entry_premium"].mean()
    cost_as_pct_premium = avg_cost_per_trade / (avg_entry * params.lot_size) * 100

    log.info("")
    log.info("  %-35s ₹%s", "Gross PnL (no costs):", f"{gross_pnl:,.2f}")
    log.info("  %-35s ₹%s", "Total transaction costs:", f"{total_costs:,.2f}")
    log.info("  %-35s ₹%s", "Net PnL (after costs):", f"{net_pnl:,.2f}")
    log.info("")
    log.info("  %-35s %.2f", "Sharpe BEFORE costs:", sharpe_gross)
    log.info("  %-35s %.2f", "Sharpe AFTER costs:", sharpe_net)
    log.info("  %-35s %.4f", "Profit Factor BEFORE costs:", pf_gross)
    log.info("  %-35s %.4f", "Profit Factor AFTER costs:", pf_net)
    log.info("")
    log.info("  %-35s ₹%.2f", "Avg cost per trade:", avg_cost_per_trade)
    log.info("  %-35s %.1f%%", "Costs as %% of gross PnL:", cost_pct_of_gross)
    log.info("  %-35s %.2f%%", "Avg cost as %% of premium:", cost_as_pct_premium)
    log.info("")

    if gross_pnl > 0:
        log.info("  ✓ GROSS EDGE EXISTS: Strategy generates positive gross PnL.")
        if net_pnl > 0:
            log.info("  ✓ NET EDGE EXISTS: Survives full cost model.")
        else:
            log.info("  ✗ NET EDGE DESTROYED BY COSTS.")
    else:
        log.info("  ✗ NO GROSS EDGE: Strategy is structurally unprofitable.")

    log.info("")
    return {
        "gross_pnl": round(gross_pnl, 2),
        "net_pnl": round(net_pnl, 2),
        "total_costs": round(total_costs, 2),
        "sharpe_gross": round(sharpe_gross, 2),
        "sharpe_net": round(sharpe_net, 2),
        "pf_gross": round(pf_gross, 4),
        "pf_net": round(pf_net, 4),
    }


# =========================================================================== #
#  SECTION 3 — TAIL RISK
# =========================================================================== #

def section_3_tail_risk(trade_df: pd.DataFrame, params: StrategyParams, margin: float):
    log.info("=" * 80)
    log.info("SECTION 3: TAIL RISK EXAMINATION")
    log.info("=" * 80)

    capital = params.initial_capital

    # --- Worst 5 trading days ---
    daily_pnl = trade_df.groupby("date")["net_pnl"].sum().reset_index()
    daily_pnl.columns = ["date", "daily_pnl"]
    worst_5 = daily_pnl.nsmallest(5, "daily_pnl")

    log.info("")
    log.info("  WORST 5 TRADING DAYS:")
    log.info("  %-15s  %12s  %12s  %12s", "Date", "PnL (₹)", "% Capital", "% Margin")
    for _, row in worst_5.iterrows():
        d = row["date"]
        pnl = row["daily_pnl"]
        log.info("  %-15s  %12s  %11.2f%%  %11.2f%%",
                 d.strftime("%Y-%m-%d") if hasattr(d, 'strftime') else str(d),
                 f"₹{pnl:,.0f}", pnl / capital * 100, pnl / margin * 100)

    worst_day_pnl = worst_5["daily_pnl"].iloc[0]
    worst_day_pct_margin = worst_day_pnl / margin * 100

    log.info("")
    log.info("  Worst single-day loss:          ₹%s", f"{worst_day_pnl:,.0f}")
    log.info("  As %% of margin:                 %.2f%%", worst_day_pct_margin)
    log.info("  As %% of capital:                %.2f%%", worst_day_pnl / capital * 100)

    # --- Max adverse excursion ---
    if "max_adverse_excursion" in trade_df.columns:
        mae = trade_df["max_adverse_excursion"]
        log.info("")
        log.info("  MAX ADVERSE EXCURSION (per trade):")
        log.info("    Mean MAE:                    ₹%s", f"{mae.mean():,.0f}")
        log.info("    Median MAE:                  ₹%s", f"{mae.median():,.0f}")
        log.info("    P95 MAE:                     ₹%s", f"{mae.quantile(0.95):,.0f}")
        log.info("    P99 MAE:                     ₹%s", f"{mae.quantile(0.99):,.0f}")
        log.info("    Max MAE:                     ₹%s", f"{mae.max():,.0f}")
        log.info("    Mean MAE as %% margin:       %.2f%%", mae.mean() / margin * 100)
        log.info("    Max MAE as %% margin:        %.2f%%", mae.max() / margin * 100)

    # --- Plot daily PnL distribution ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Histogram
    ax = axes[0]
    ax.hist(daily_pnl["daily_pnl"], bins=50, edgecolor="black", alpha=0.7, color="#4c72b0")
    ax.axvline(0, color="red", linestyle="--", linewidth=1)
    ax.axvline(daily_pnl["daily_pnl"].mean(), color="green", linestyle="-", linewidth=1.5,
               label=f"Mean: ₹{daily_pnl['daily_pnl'].mean():,.0f}")
    ax.set_title("Daily PnL Distribution (Net)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Daily PnL (₹)")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(alpha=0.3)

    # MAE distribution
    if "max_adverse_excursion" in trade_df.columns:
        ax2 = axes[1]
        ax2.hist(mae, bins=50, edgecolor="black", alpha=0.7, color="#c44e52")
        ax2.axvline(mae.mean(), color="green", linestyle="-", linewidth=1.5,
                    label=f"Mean: ₹{mae.mean():,.0f}")
        ax2.axvline(mae.quantile(0.95), color="orange", linestyle="--", linewidth=1.5,
                    label=f"P95: ₹{mae.quantile(0.95):,.0f}")
        ax2.set_title("Max Adverse Excursion Distribution", fontsize=12, fontweight="bold")
        ax2.set_xlabel("MAE per Trade (₹)")
        ax2.set_ylabel("Frequency")
        ax2.legend()
        ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT / "tail_risk_distributions.png", dpi=150)
    plt.close(fig)
    log.info("  → tail_risk_distributions.png saved")
    log.info("")

    return {
        "worst_day_pnl": round(worst_day_pnl, 2),
        "worst_day_pct_margin": round(worst_day_pct_margin, 2),
        "mean_mae": round(mae.mean(), 2) if "max_adverse_excursion" in trade_df.columns else None,
        "p95_mae": round(mae.quantile(0.95), 2) if "max_adverse_excursion" in trade_df.columns else None,
        "max_mae": round(mae.max(), 2) if "max_adverse_excursion" in trade_df.columns else None,
    }


# =========================================================================== #
#  SECTION 4 — ROLLING STABILITY
# =========================================================================== #

def section_4_rolling(trade_df: pd.DataFrame, params: StrategyParams):
    log.info("=" * 80)
    log.info("SECTION 4: ROLLING STABILITY")
    log.info("=" * 80)

    capital = params.initial_capital
    daily = trade_df.groupby("date")["net_pnl"].sum().reset_index()
    daily.columns = ["date", "pnl"]
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date").reset_index(drop=True)

    # Also compute gross daily
    daily_gross = trade_df.groupby("date")["gross_pnl"].sum().reset_index()
    daily_gross.columns = ["date", "gross_pnl"]
    daily_gross["date"] = pd.to_datetime(daily_gross["date"])
    daily = daily.merge(daily_gross, on="date", how="left")

    daily["month"] = daily["date"].dt.to_period("M")
    unique_months = daily["month"].unique()

    # 6-month rolling windows
    window = 6
    rolling_results = []

    for i in range(len(unique_months) - window + 1):
        start_m = unique_months[i]
        end_m = unique_months[i + window - 1]
        mask = (daily["month"] >= start_m) & (daily["month"] <= end_m)
        w = daily[mask]
        if len(w) < 10:
            continue

        w_return = w["pnl"].sum()
        w_sharpe = _sharpe(w["pnl"], capital)
        w_gross_return = w["gross_pnl"].sum()
        w_sharpe_gross = _sharpe(w["gross_pnl"], capital)

        rolling_results.append({
            "window_start": str(start_m),
            "window_end": str(end_m),
            "net_return": w_return,
            "net_return_pct": w_return / capital * 100,
            "sharpe_net": round(w_sharpe, 2),
            "gross_return": w_gross_return,
            "sharpe_gross": round(w_sharpe_gross, 2),
            "trades": len(trade_df[(pd.to_datetime(trade_df["date"]).dt.to_period("M") >= start_m) &
                                    (pd.to_datetime(trade_df["date"]).dt.to_period("M") <= end_m)]),
        })

    roll_df = pd.DataFrame(rolling_results)

    log.info("")
    log.info("  6-MONTH ROLLING WINDOWS:")
    log.info("  %-12s %-12s %8s %10s %10s %10s", "Start", "End", "Trades",
             "Net Ret%", "Sharpe(N)", "Sharpe(G)")
    for _, r in roll_df.iterrows():
        log.info("  %-12s %-12s %8d %9.2f%% %10.2f %10.2f",
                 r["window_start"], r["window_end"], r["trades"],
                 r["net_return_pct"], r["sharpe_net"], r["sharpe_gross"])

    n_positive_net = (roll_df["net_return"] > 0).sum()
    n_positive_gross = (roll_df["gross_return"] > 0).sum()
    n_windows = len(roll_df)

    log.info("")
    log.info("  Total rolling windows:          %d", n_windows)
    log.info("  Windows with positive NET:      %d (%.0f%%)", n_positive_net,
             n_positive_net / n_windows * 100 if n_windows else 0)
    log.info("  Windows with positive GROSS:    %d (%.0f%%)", n_positive_gross,
             n_positive_gross / n_windows * 100 if n_windows else 0)
    log.info("  Avg rolling Sharpe (Net):       %.2f", roll_df["sharpe_net"].mean())
    log.info("  Avg rolling Sharpe (Gross):     %.2f", roll_df["sharpe_gross"].mean())

    if n_positive_net / n_windows > 0.6 if n_windows else False:
        log.info("  ASSESSMENT: Edge is PERSISTENT across windows.")
    else:
        log.info("  ASSESSMENT: Edge is CLUSTERED / inconsistent.")

    # --- Plot ---
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    ax1 = axes[0]
    x = range(len(roll_df))
    ax1.bar(x, roll_df["sharpe_net"], alpha=0.7, color=["#2ca02c" if s > 0 else "#d62728"
            for s in roll_df["sharpe_net"]], label="Net Sharpe")
    ax1.axhline(0, color="black", linewidth=0.5)
    ax1.axhline(roll_df["sharpe_net"].mean(), color="blue", linestyle="--",
                label=f'Avg: {roll_df["sharpe_net"].mean():.2f}')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{r["window_start"]}\n{r["window_end"]}' for _, r in roll_df.iterrows()],
                        rotation=45, fontsize=7)
    ax1.set_title("6-Month Rolling Sharpe Ratio (Net)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Sharpe Ratio")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2 = axes[1]
    ax2.bar(x, roll_df["net_return_pct"], alpha=0.7,
            color=["#2ca02c" if r > 0 else "#d62728" for r in roll_df["net_return_pct"]])
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{r["window_start"]}\n{r["window_end"]}' for _, r in roll_df.iterrows()],
                        rotation=45, fontsize=7)
    ax2.set_title("6-Month Rolling Return %", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Return (%)")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT / "rolling_stability.png", dpi=150)
    plt.close(fig)
    log.info("  → rolling_stability.png saved")
    log.info("")

    roll_df.to_csv(OUT / "rolling_windows.csv", index=False)
    return {
        "n_windows": n_windows,
        "pct_positive_net": round(n_positive_net / n_windows * 100, 1) if n_windows else 0,
        "pct_positive_gross": round(n_positive_gross / n_windows * 100, 1) if n_windows else 0,
        "avg_rolling_sharpe_net": round(roll_df["sharpe_net"].mean(), 2),
        "avg_rolling_sharpe_gross": round(roll_df["sharpe_gross"].mean(), 2),
    }


# =========================================================================== #
#  SECTION 5 — SENSITIVITY HEATMAP
# =========================================================================== #

def section_5_sensitivity(panel: pd.DataFrame):
    log.info("=" * 80)
    log.info("SECTION 5: SENSITIVITY TEST")
    log.info("=" * 80)

    breakout_levels = [0.002, 0.003, 0.004, 0.005]
    sl_levels = [0.15, 0.20, 0.25, 0.30]
    et_levels = ["14:00:00", "14:30:00"]

    rows = []
    total = len(breakout_levels) * len(sl_levels) * len(et_levels)
    count = 0

    for bo, sl, et in product(breakout_levels, sl_levels, et_levels):
        count += 1
        if count % 8 == 0:
            log.info("  %d/%d combinations tested", count, total)

        tdf, p = _run_with_params(
            panel,
            breakout_buffer_pct=bo,
            short_stop_loss_pct=sl,
            exit_time=et,
        )
        if tdf.empty:
            continue

        m = compute_metrics(tdf, p)
        daily_gross = tdf.groupby("date")["gross_pnl"].sum()

        rows.append({
            "breakout_buffer": bo,
            "stop_loss": sl,
            "exit_time": et,
            "trades": m["total_trades"],
            "net_return_pct": round(m["total_return"] * 100, 3),
            "sharpe_net": round(m["sharpe_ratio"], 2),
            "sharpe_gross": round(_sharpe(daily_gross, p.initial_capital), 2),
            "pf_net": round(m["profit_factor"], 3),
            "pf_gross": round(_profit_factor(tdf["gross_pnl"]), 3),
            "win_rate": round(m["win_rate"] * 100, 1),
            "max_dd": round(m["max_drawdown"] * 100, 2),
        })

    sens_df = pd.DataFrame(rows)
    sens_df.to_csv(OUT / "sensitivity_grid.csv", index=False)

    log.info("")
    log.info("  FULL SENSITIVITY GRID:")
    log.info("  %-8s %-6s %-6s %6s %8s %8s %8s %8s %8s",
             "Breakout", "SL%", "Exit", "Trades", "Ret%", "Sh(N)", "Sh(G)", "PF(N)", "PF(G)")
    for _, r in sens_df.iterrows():
        log.info("  %-8.1f%% %-6.0f%% %-6s %6d %7.2f%% %8.2f %8.2f %8.3f %8.3f",
                 r["breakout_buffer"] * 100, r["stop_loss"] * 100, r["exit_time"][:5],
                 r["trades"], r["net_return_pct"], r["sharpe_net"], r["sharpe_gross"],
                 r["pf_net"], r["pf_gross"])

    # --- Stability analysis ---
    sharpe_values = sens_df["sharpe_net"].values
    pf_values = sens_df["pf_net"].values
    n_positive_sharpe = (sharpe_values > 0).sum()
    n_pf_above_1 = (pf_values > 1.0).sum()

    log.info("")
    log.info("  STABILITY ANALYSIS:")
    log.info("    Configs with Sharpe > 0:     %d / %d (%.0f%%)",
             n_positive_sharpe, len(sens_df), n_positive_sharpe / len(sens_df) * 100)
    log.info("    Configs with PF > 1.0:       %d / %d (%.0f%%)",
             n_pf_above_1, len(sens_df), n_pf_above_1 / len(sens_df) * 100)
    log.info("    Mean Sharpe (net):           %.2f", sharpe_values.mean())
    log.info("    Std Sharpe (net):            %.2f", sharpe_values.std())
    log.info("    Min/Max Sharpe:              %.2f / %.2f", sharpe_values.min(), sharpe_values.max())

    if n_positive_sharpe / len(sens_df) > 0.7:
        log.info("    ✓ EDGE IS BROAD: Positive across >70%% of parameter space.")
    elif n_positive_sharpe / len(sens_df) > 0.4:
        log.info("    ~ EDGE IS MODERATE: Positive in 40-70%% of parameter space.")
    else:
        log.info("    ✗ EDGE IS NARROW: Positive in <40%% — fragile.")

    # --- Heatmaps ---
    for et_val in et_levels:
        sub = sens_df[sens_df["exit_time"] == et_val]
        if sub.empty:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Sharpe heatmap
        pivot_sharpe = sub.pivot_table(
            values="sharpe_net", index="stop_loss", columns="breakout_buffer"
        )
        sns.heatmap(pivot_sharpe, annot=True, fmt=".2f", cmap="RdYlGn", center=0,
                    ax=axes[0], linewidths=0.5,
                    xticklabels=[f"{x*100:.1f}%" for x in pivot_sharpe.columns],
                    yticklabels=[f"{y*100:.0f}%" for y in pivot_sharpe.index])
        axes[0].set_title(f"Sharpe Ratio (Net) — Exit {et_val[:5]}", fontsize=12, fontweight="bold")
        axes[0].set_xlabel("Breakout Buffer %")
        axes[0].set_ylabel("Stop Loss %")

        # PF heatmap
        pivot_pf = sub.pivot_table(
            values="pf_net", index="stop_loss", columns="breakout_buffer"
        )
        sns.heatmap(pivot_pf, annot=True, fmt=".2f", cmap="RdYlGn", center=1.0,
                    ax=axes[1], linewidths=0.5,
                    xticklabels=[f"{x*100:.1f}%" for x in pivot_pf.columns],
                    yticklabels=[f"{y*100:.0f}%" for y in pivot_pf.index])
        axes[1].set_title(f"Profit Factor (Net) — Exit {et_val[:5]}", fontsize=12, fontweight="bold")
        axes[1].set_xlabel("Breakout Buffer %")
        axes[1].set_ylabel("Stop Loss %")

        fig.tight_layout()
        fig.savefig(OUT / f"heatmap_{et_val[:5].replace(':', '')}.png", dpi=150)
        plt.close(fig)
        log.info("  → heatmap_%s.png saved", et_val[:5].replace(":", ""))

    log.info("")
    return {
        "n_configs": len(sens_df),
        "pct_positive_sharpe": round(n_positive_sharpe / len(sens_df) * 100, 1),
        "pct_pf_above_1": round(n_pf_above_1 / len(sens_df) * 100, 1),
        "mean_sharpe": round(sharpe_values.mean(), 2),
        "sharpe_range": f"{sharpe_values.min():.2f} to {sharpe_values.max():.2f}",
    }


# =========================================================================== #
#  FINAL VERDICT
# =========================================================================== #

def final_verdict(s1, s2, s3, s4, s5):
    log.info("=" * 80)
    log.info("FINAL VERDICT — CUTTHROAT ASSESSMENT")
    log.info("=" * 80)

    log.info("")
    log.info("  ┌─────────────────────────────────────────────────────────┐")
    log.info("  │  QUESTION 1: Is this structurally tradable?            │")
    log.info("  │                                                         │")

    has_gross = s2["gross_pnl"] > 0
    has_net = s2["net_pnl"] > 0
    broad_edge = s5["pct_positive_sharpe"] > 50

    if has_gross and has_net and broad_edge:
        log.info("  │  YES — Gross edge exists, survives costs, and is      │")
        log.info("  │  broad across parameter space.                         │")
    elif has_gross and has_net:
        log.info("  │  MARGINAL — Edge exists but may be narrow or fragile. │")
    elif has_gross and not has_net:
        log.info("  │  NO — Gross edge exists but costs destroy it.         │")
    else:
        log.info("  │  NO — No structural edge found.                       │")

    log.info("  │                                                         │")
    log.info("  │  QUESTION 2: Is this margin-efficient?                 │")
    log.info("  │                                                         │")

    rom = s1["annualised_rom"]
    if rom > 10:
        log.info("  │  YES — %.1f%% annualised ROM is attractive.           │", rom)
    elif rom > 3:
        log.info("  │  MARGINAL — %.1f%% ROM barely beats FD rates.         │", rom)
    else:
        log.info("  │  NO — %.1f%% ROM is not worth the risk.               │", rom)

    log.info("  │                                                         │")
    log.info("  │  QUESTION 3: Is risk-adjusted return acceptable?       │")
    log.info("  │                                                         │")

    rolling_positive = s4["pct_positive_net"]
    if s2["sharpe_net"] > 1.0 and rolling_positive > 60:
        log.info("  │  YES — Sharpe %.2f with %d%% rolling windows positive.│", s2["sharpe_net"], int(rolling_positive))
    elif s2["sharpe_net"] > 0.5 and rolling_positive > 40:
        log.info("  │  MARGINAL — Sharpe %.2f, %d%% rolling windows +ve.    │", s2["sharpe_net"], int(rolling_positive))
    else:
        log.info("  │  NO — Sharpe %.2f is too low for operational risk.    │", s2["sharpe_net"])

    log.info("  │                                                         │")
    log.info("  └─────────────────────────────────────────────────────────┘")

    # Composite score
    score = 0
    if has_gross: score += 1
    if has_net: score += 1
    if s2["sharpe_net"] > 0.5: score += 1
    if s2["sharpe_net"] > 1.0: score += 1
    if rolling_positive > 50: score += 1
    if rolling_positive > 70: score += 1
    if rom > 5: score += 1
    if rom > 10: score += 1
    if s5["pct_positive_sharpe"] > 60: score += 1
    if s3["worst_day_pct_margin"] > -10: score += 1

    log.info("")
    log.info("  COMPOSITE SCORE: %d / 10", score)
    if score >= 7:
        log.info("  RECOMMENDATION: PROCEED WITH PAPER TRADING")
    elif score >= 5:
        log.info("  RECOMMENDATION: CONDITIONAL — needs tighter risk or cost reduction")
    elif score >= 3:
        log.info("  RECOMMENDATION: MARGINAL — not worth operational overhead")
    else:
        log.info("  RECOMMENDATION: REJECT — no tradable edge")
    log.info("")


# =========================================================================== #
#  MAIN
# =========================================================================== #

def run_validation():
    OUT.mkdir(parents=True, exist_ok=True)

    log.info("Loading data...")
    ce, pe = load_data(strike_filter="ATM")
    panel = build_intraday_panel(ce, pe)

    # Base run
    log.info("Running base short straddle backtest...")
    trade_df, params = _run_with_params(panel)

    log.info("Total trades: %d | Days: %d\n", len(trade_df),
             (trade_df["date"].max() - trade_df["date"].min()).days)

    # Run all sections
    s1 = section_1_capital_margin(trade_df, params)
    s2 = section_2_gross_vs_net(trade_df, params)
    s3 = section_3_tail_risk(trade_df, params, s1["estimated_margin"])
    s4 = section_4_rolling(trade_df, params)
    s5 = section_5_sensitivity(panel)

    final_verdict(s1, s2, s3, s4, s5)

    log.info("All validation artefacts saved to: %s", OUT)


if __name__ == "__main__":
    run_validation()
