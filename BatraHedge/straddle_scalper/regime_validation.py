"""
Regime-Aware Validation — Short ATM Straddle
=============================================
breakout_buffer = 0.5%, SL = 20%, exit = 14:30

Sections:
  1. Detect Trending Regime (historical classification)
  2. Simulate High-Trend Regime (synthetic stress)
  3. Re-test with Regime Filter ON
  4. Monte Carlo Trade-Path Stress (10,000 reshuffles)
  5. Final Decision
"""

from __future__ import annotations
import logging, sys, copy
from pathlib import Path
from itertools import product

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

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)
OUT = OUTPUT_DIR / "regime_validation"


# ======================================================================= #
#  HELPERS                                                                  #
# ======================================================================= #

BASE_KW = dict(
    strategy_type="Short Straddle",
    compression_filter_enabled=True,
    compression_threshold=0.004,
    short_profit_target_pct=0.10,
    short_stop_loss_pct=0.20,
    breakout_buffer_pct=0.005,
    exit_time="14:30:00",
    brokerage_per_leg=20.0,
    slippage_pct=0.05,
    stt_on_sell_pct=0.0625,
    initial_capital=1_000_000.0,
    lot_size=25,
)

def _p(**kw):
    d = {**BASE_KW, **kw}
    return StrategyParams(**d)

def _run(panel, **kw):
    p = _p(**kw)
    trades = run_backtest(panel, p)
    return trades_to_dataframe(trades), p

def _sharpe(daily_pnl, capital):
    dr = daily_pnl / capital
    return dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0.0

def _pf(pnl):
    w = pnl[pnl > 0].sum(); l = abs(pnl[pnl <= 0].sum())
    return w / l if l > 0 else float("inf")

def _margin(tdf):
    return tdf["spot_at_entry"].mean() * 25 * 0.15


def _full_metrics(tdf, p):
    """Compute the standard summary dict for a trade_df."""
    m = compute_metrics(tdf, p)
    margin = _margin(tdf)
    daily_net = tdf.groupby("date")["net_pnl"].sum()
    daily_gross = tdf.groupby("date")["gross_pnl"].sum()
    net_pnl = tdf["net_pnl"].sum()
    days = (tdf["date"].max() - tdf["date"].min()).days
    years = days / 365.25
    rom_ann = ((1 + net_pnl / margin) ** (1 / years) - 1) * 100 if (1 + net_pnl / margin) > 0 else -100
    worst_day = daily_net.min()
    return {
        "trades": m["total_trades"],
        "win_rate": round(m["win_rate"] * 100, 1),
        "net_pnl": round(net_pnl, 0),
        "gross_pnl": round(tdf["gross_pnl"].sum(), 0),
        "total_ret_pct": round(m["total_return"] * 100, 3),
        "sharpe_gross": round(_sharpe(daily_gross, p.initial_capital), 2),
        "sharpe_net": round(_sharpe(daily_net, p.initial_capital), 2),
        "pf_net": round(m["profit_factor"], 3),
        "max_dd_pct": round(m["max_drawdown"] * 100, 3),
        "rom_ann_pct": round(rom_ann, 2),
        "margin": round(margin, 0),
        "worst_day": round(worst_day, 0),
        "worst_day_pct_margin": round(worst_day / margin * 100, 2),
    }


def _print_metrics(label, r):
    log.info("  %-35s %s", "Config:", label)
    log.info("  %-35s %d", "Trades:", r["trades"])
    log.info("  %-35s %.1f%%", "Win Rate:", r["win_rate"])
    log.info("  %-35s ₹%s", "Net PnL:", f'{r["net_pnl"]:,.0f}')
    log.info("  %-35s %.2f", "Sharpe (gross):", r["sharpe_gross"])
    log.info("  %-35s %.2f", "Sharpe (net):", r["sharpe_net"])
    log.info("  %-35s %.3f", "Profit Factor (net):", r["pf_net"])
    log.info("  %-35s %.3f%%", "Max Drawdown:", r["max_dd_pct"])
    log.info("  %-35s %.2f%%", "ROM (annualised):", r["rom_ann_pct"])
    log.info("  %-35s ₹%s (%.2f%% margin)", "Worst day PnL:", f'{r["worst_day"]:,.0f}', r["worst_day_pct_margin"])


# ======================================================================= #
#  SECTION 1  –  DETECT TRENDING REGIME                                    #
# ======================================================================= #

def section_1(panel):
    log.info("=" * 80)
    log.info("SECTION 1: DETECT TRENDING REGIME")
    log.info("=" * 80)

    grouped = panel.groupby("date")
    days = sorted(grouped.groups.keys())

    records = []
    for d in days:
        ddf = grouped.get_group(d).sort_values("datetime")
        spot_open = ddf["spot"].iloc[0]
        spot_close = ddf["spot"].iloc[-1]
        spot_high = ddf["spot"].max()
        spot_low = ddf["spot"].min()
        spot_mid = (spot_high + spot_low) / 2

        full_range_pct = (spot_high - spot_low) / spot_mid * 100 if spot_mid else 0
        directional_pct = abs(spot_close - spot_open) / spot_open * 100 if spot_open else 0

        # IV data
        iv_open = (ddf.iloc[0]["ce_iv"] + ddf.iloc[0]["pe_iv"]) / 2
        iv_close = (ddf.iloc[-1]["ce_iv"] + ddf.iloc[-1]["pe_iv"]) / 2

        records.append({
            "date": d,
            "spot_open": spot_open,
            "spot_close": spot_close,
            "spot_high": spot_high,
            "spot_low": spot_low,
            "full_range_pct": round(full_range_pct, 4),
            "directional_pct": round(directional_pct, 4),
            "iv_open": round(iv_open, 4),
            "iv_close": round(iv_close, 4),
        })

    day_df = pd.DataFrame(records)
    day_df["date"] = pd.to_datetime(day_df["date"])
    day_df = day_df.sort_values("date").reset_index(drop=True)

    # Rolling 10-day average daily range
    day_df["rolling_10d_range"] = day_df["full_range_pct"].rolling(10, min_periods=5).mean()

    # Define high-trend day
    day_df["high_trend"] = (day_df["full_range_pct"] > 1.2) | (day_df["directional_pct"] > 0.8)

    n_total = len(day_df)
    n_trend = day_df["high_trend"].sum()
    pct_trend = n_trend / n_total * 100

    log.info("")
    log.info("  Total trading days:             %d", n_total)
    log.info("  High-trend days:                %d (%.1f%%)", n_trend, pct_trend)
    log.info("  Non-trend (calm) days:          %d (%.1f%%)", n_total - n_trend, 100 - pct_trend)
    log.info("")
    log.info("  DAILY SPOT RANGE DISTRIBUTION:")
    log.info("    Mean range:                   %.3f%%", day_df["full_range_pct"].mean())
    log.info("    Median range:                 %.3f%%", day_df["full_range_pct"].median())
    log.info("    P75 range:                    %.3f%%", day_df["full_range_pct"].quantile(0.75))
    log.info("    P90 range:                    %.3f%%", day_df["full_range_pct"].quantile(0.90))
    log.info("    P95 range:                    %.3f%%", day_df["full_range_pct"].quantile(0.95))
    log.info("    Max range:                    %.3f%%", day_df["full_range_pct"].max())
    log.info("")
    log.info("  DIRECTIONAL MOVE DISTRIBUTION:")
    log.info("    Mean directional:             %.3f%%", day_df["directional_pct"].mean())
    log.info("    Median directional:           %.3f%%", day_df["directional_pct"].median())
    log.info("    P90 directional:              %.3f%%", day_df["directional_pct"].quantile(0.90))
    log.info("    P95 directional:              %.3f%%", day_df["directional_pct"].quantile(0.95))
    log.info("")
    log.info("  ROLLING 10d AVG RANGE:")
    log.info("    Mean:                         %.3f%%", day_df["rolling_10d_range"].mean())
    log.info("    Max:                          %.3f%%", day_df["rolling_10d_range"].max())
    log.info("    %% days >1.0%%:                  %.1f%%",
             (day_df["rolling_10d_range"] > 1.0).mean() * 100)
    log.info("")

    # Save
    day_df.to_csv(OUT / "daily_regime_classification.csv", index=False)

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    ax = axes[0, 0]
    ax.hist(day_df["full_range_pct"], bins=50, edgecolor="black", alpha=0.7, color="#4c72b0")
    ax.axvline(1.2, color="red", linestyle="--", label="High-trend threshold (1.2%)")
    ax.set_title("Daily Spot Range %", fontweight="bold")
    ax.set_xlabel("Range %"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.hist(day_df["directional_pct"], bins=50, edgecolor="black", alpha=0.7, color="#55a868")
    ax.axvline(0.8, color="red", linestyle="--", label="Directional threshold (0.8%)")
    ax.set_title("Daily Directional Move %", fontweight="bold")
    ax.set_xlabel("|Close - Open| / Open %"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(day_df["date"], day_df["rolling_10d_range"], color="#4c72b0", linewidth=0.8)
    ax.axhline(1.0, color="red", linestyle="--", label="Regime threshold (1.0%)")
    ax.set_title("Rolling 10-Day Avg Range %", fontweight="bold")
    ax.set_xlabel("Date"); ax.set_ylabel("Range %"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    colors = ["#d62728" if ht else "#2ca02c" for ht in day_df["high_trend"]]
    ax.scatter(day_df["full_range_pct"], day_df["directional_pct"], c=colors, alpha=0.5, s=15)
    ax.axvline(1.2, color="red", linestyle="--", alpha=0.5)
    ax.axhline(0.8, color="red", linestyle="--", alpha=0.5)
    ax.set_xlabel("Full Range %"); ax.set_ylabel("Directional %")
    ax.set_title("Regime Classification (Red = High-Trend)", fontweight="bold")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT / "regime_detection.png", dpi=150)
    plt.close(fig)
    log.info("  → regime_detection.png saved")
    log.info("")

    return day_df


# ======================================================================= #
#  SECTION 2  –  SIMULATE HIGH-TREND REGIME                                #
# ======================================================================= #

def section_2(panel, day_regime_df):
    log.info("=" * 80)
    log.info("SECTION 2: SIMULATE HIGH-TREND REGIME")
    log.info("=" * 80)

    # --- First: run base case for comparison ---
    log.info("  Running BASE case (historical)...")
    tdf_base, p = _run(panel)
    base = _full_metrics(tdf_base, p)

    # --- Identify calm days to inject trend into ---
    calm_days = day_regime_df[~day_regime_df["high_trend"]]["date"].values
    trend_days = day_regime_df[day_regime_df["high_trend"]]["date"].values
    current_pct = len(trend_days) / len(day_regime_df) * 100

    # We want 30% of days to be high-trend. Figure out how many calm days to corrupt.
    target_trend_count = int(0.30 * len(day_regime_df))
    additional_needed = max(0, target_trend_count - len(trend_days))

    log.info("")
    log.info("  Current high-trend days: %d (%.1f%%)", len(trend_days), current_pct)
    log.info("  Target high-trend days:  %d (30%%)", target_trend_count)
    log.info("  Additional days to inject: %d", additional_needed)

    # Select random calm days to corrupt
    np.random.seed(42)
    if additional_needed > 0 and len(calm_days) > additional_needed:
        inject_days = set(np.random.choice(calm_days, size=additional_needed, replace=False))
    else:
        inject_days = set()

    # --- Build synthetic panel ---
    # For injected days: widen the spot range to simulate 1-1.5% intraday range
    # and bump IV by +1 point
    log.info("  Building synthetic high-trend panel...")
    synthetic_panel = panel.copy()
    synthetic_panel["date_dt"] = pd.to_datetime(synthetic_panel["date"])

    for d in inject_days:
        mask = synthetic_panel["date_dt"] == d
        day_data = synthetic_panel.loc[mask]
        if day_data.empty:
            continue

        spot_mid = day_data["spot"].mean()
        current_range = day_data["spot"].max() - day_data["spot"].min()
        target_range = spot_mid * np.random.uniform(0.01, 0.015)  # 1-1.5% range

        if current_range < target_range:
            scale_factor = target_range / current_range if current_range > 0 else 1.0
            # Stretch spot deviations from mean
            synthetic_panel.loc[mask, "spot"] = spot_mid + (day_data["spot"] - spot_mid) * scale_factor

            # Bump option premiums proportionally (delta ~0.5 effect)
            premium_bump = (target_range - current_range) * 0.5 / 2  # split between CE and PE
            synthetic_panel.loc[mask, "ce_high"] = day_data["ce_high"] + premium_bump
            synthetic_panel.loc[mask, "ce_close"] = day_data["ce_close"] + premium_bump * 0.7
            synthetic_panel.loc[mask, "pe_high"] = day_data["pe_high"] + premium_bump
            synthetic_panel.loc[mask, "pe_close"] = day_data["pe_close"] + premium_bump * 0.7

        # Bump IV by +1 point on injected days
        synthetic_panel.loc[mask, "ce_iv"] = day_data["ce_iv"] + 1.0
        synthetic_panel.loc[mask, "pe_iv"] = day_data["pe_iv"] + 1.0

    synthetic_panel.drop(columns=["date_dt"], inplace=True)

    # --- Run backtest on synthetic panel ---
    log.info("  Running SYNTHETIC HIGH-TREND backtest...")
    tdf_syn, p_syn = _run(synthetic_panel)
    syn = _full_metrics(tdf_syn, p_syn)

    log.info("")
    log.info("  %-35s %-15s %-15s", "Metric", "BASE", "HIGH-TREND (30%)")
    for key in ["trades", "win_rate", "net_pnl", "sharpe_gross", "sharpe_net",
                "pf_net", "max_dd_pct", "rom_ann_pct", "worst_day", "worst_day_pct_margin"]:
        fmt_b = f'{base[key]:,.0f}' if isinstance(base[key], (int, float)) and key in ["net_pnl", "worst_day"] else \
                f'{base[key]:.2f}' if isinstance(base[key], float) else str(base[key])
        fmt_s = f'{syn[key]:,.0f}' if isinstance(syn[key], (int, float)) and key in ["net_pnl", "worst_day"] else \
                f'{syn[key]:.2f}' if isinstance(syn[key], float) else str(syn[key])
        log.info("  %-35s %-15s %-15s", key, fmt_b, fmt_s)

    # Degradation
    sharpe_delta = syn["sharpe_net"] - base["sharpe_net"]
    rom_delta = syn["rom_ann_pct"] - base["rom_ann_pct"]
    log.info("")
    log.info("  DEGRADATION UNDER HIGH-TREND:")
    log.info("    Sharpe Δ: %.2f  (%.1f%%)", sharpe_delta,
             sharpe_delta / base["sharpe_net"] * 100 if base["sharpe_net"] else 0)
    log.info("    ROM Δ:    %.2f pp", rom_delta)
    log.info("    Max DD:   %.3f%% → %.3f%%", base["max_dd_pct"], syn["max_dd_pct"])

    if syn["sharpe_net"] > 2.0:
        log.info("    ✓ Strategy SURVIVES high-trend regime (Sharpe %.2f)", syn["sharpe_net"])
    elif syn["sharpe_net"] > 1.0:
        log.info("    ~ Strategy WEAKENED but still profitable (Sharpe %.2f)", syn["sharpe_net"])
    elif syn["sharpe_net"] > 0:
        log.info("    ⚠ Strategy MARGINALLY positive (Sharpe %.2f)", syn["sharpe_net"])
    else:
        log.info("    ✗ Strategy FAILS under high-trend regime (Sharpe %.2f)", syn["sharpe_net"])
    log.info("")

    return base, syn, synthetic_panel


# ======================================================================= #
#  SECTION 3  –  RE-TEST WITH REGIME FILTER                                #
# ======================================================================= #

def section_3(panel, synthetic_panel):
    log.info("=" * 80)
    log.info("SECTION 3: RE-TEST WITH REGIME FILTER ON")
    log.info("=" * 80)

    configs = [
        ("BASE — no filter", panel, {}),
        ("BASE — filter ON", panel, {"regime_filter_enabled": True}),
        ("HIGH-TREND — no filter", synthetic_panel, {}),
        ("HIGH-TREND — filter ON", synthetic_panel, {"regime_filter_enabled": True}),
    ]

    results = []
    for label, pnl, kw in configs:
        log.info("  Running: %s", label)
        tdf, p = _run(pnl, **kw)
        if tdf.empty:
            log.info("    → No trades generated!")
            results.append({"label": label, "trades": 0})
            continue
        r = _full_metrics(tdf, p)
        r["label"] = label
        results.append(r)

    log.info("")
    log.info("  %-35s %6s %6s %8s %8s %8s %8s %8s",
             "Config", "Trades", "WR%", "Sh(N)", "PF", "MaxDD%", "ROM%", "Worst₹")
    for r in results:
        if r["trades"] == 0:
            log.info("  %-35s %6d %6s %8s %8s %8s %8s %8s",
                     r["label"], 0, "—", "—", "—", "—", "—", "—")
            continue
        log.info("  %-35s %6d %5.1f%% %8.2f %8.3f %7.3f%% %7.2f%% ₹%6s",
                 r["label"], r["trades"], r["win_rate"],
                 r["sharpe_net"], r["pf_net"], r["max_dd_pct"],
                 r["rom_ann_pct"], f'{r["worst_day"]:,.0f}')

    # Assessment
    log.info("")
    base_no = [r for r in results if r["label"] == "BASE — no filter"][0]
    base_yes = [r for r in results if r["label"] == "BASE — filter ON"][0]
    syn_no = [r for r in results if r["label"] == "HIGH-TREND — no filter"][0]
    syn_yes = [r for r in results if r["label"] == "HIGH-TREND — filter ON"][0]

    if base_yes.get("trades", 0) > 0:
        filter_trade_reduction = (1 - base_yes["trades"] / base_no["trades"]) * 100
        sharpe_change = base_yes["sharpe_net"] - base_no["sharpe_net"]
        log.info("  REGIME FILTER IMPACT (on BASE):")
        log.info("    Trade reduction:         %.0f%%  (%d → %d)",
                 filter_trade_reduction, base_no["trades"], base_yes["trades"])
        log.info("    Sharpe change:           %.2f → %.2f (Δ %.2f)",
                 base_no["sharpe_net"], base_yes["sharpe_net"], sharpe_change)
        log.info("    ROM change:              %.2f%% → %.2f%%",
                 base_no["rom_ann_pct"], base_yes["rom_ann_pct"])
        if sharpe_change > 0.5:
            log.info("    ✓ Filter IMPROVES Sharpe significantly")
        elif sharpe_change > -0.5:
            log.info("    ~ Filter has MINIMAL impact on Sharpe")
        else:
            log.info("    ✗ Filter HURTS Sharpe (reduces trade count too much)")
    else:
        log.info("  ⚠ Filter ON produced zero trades: thresholds too tight")

    if syn_yes.get("trades", 0) > 0:
        log.info("  REGIME FILTER IMPACT (on HIGH-TREND SCENARIO):")
        log.info("    Trade reduction:         %.0f%%  (%d → %d)",
                 (1 - syn_yes["trades"] / syn_no["trades"]) * 100,
                 syn_no["trades"], syn_yes["trades"])
        log.info("    Sharpe:                  %.2f → %.2f",
                 syn_no["sharpe_net"], syn_yes["sharpe_net"])
        log.info("    Worst day:               ₹%s → ₹%s",
                 f'{syn_no["worst_day"]:,.0f}', f'{syn_yes["worst_day"]:,.0f}')

    log.info("")
    return results


# ======================================================================= #
#  SECTION 4  –  MONTE CARLO TRADE-PATH STRESS                            #
# ======================================================================= #

def section_4(panel):
    log.info("=" * 80)
    log.info("SECTION 4: MONTE CARLO TRADE-PATH STRESS  (10,000 reshuffles)")
    log.info("=" * 80)

    tdf, p = _run(panel)
    capital = p.initial_capital
    daily_pnl = tdf.groupby("date")["net_pnl"].sum().values
    n_days = len(daily_pnl)

    log.info("")
    log.info("  Input: %d trading days with daily PnL", n_days)
    log.info("  Actual total PnL:  ₹%s", f'{daily_pnl.sum():,.0f}')
    log.info("  Running 10,000 Monte Carlo reshuffles...")

    N_SIM = 10_000
    np.random.seed(2026)

    max_dds = np.zeros(N_SIM)
    final_pnls = np.zeros(N_SIM)
    worst_peaks_to_troughs = np.zeros(N_SIM)

    for i in range(N_SIM):
        shuffled = np.random.permutation(daily_pnl)
        cum = np.cumsum(shuffled)
        equity = capital + cum
        running_peak = np.maximum.accumulate(equity)
        drawdowns = (equity - running_peak) / running_peak
        max_dds[i] = drawdowns.min()
        final_pnls[i] = cum[-1]
        # Peak-to-trough in absolute INR
        worst_peaks_to_troughs[i] = (equity - running_peak).min()

    # Statistics
    dd_pct = max_dds * 100  # already fractional
    margin = _margin(tdf)

    log.info("  Done.")
    log.info("")
    log.info("  MONTE CARLO MAX DRAWDOWN (on capital):")
    log.info("    Mean DD:                     %.3f%%", dd_pct.mean())
    log.info("    Median DD:                   %.3f%%", np.median(dd_pct))
    log.info("    P95 DD (5%% worst):            %.3f%%", np.percentile(dd_pct, 5))
    log.info("    P99 DD (1%% worst):            %.3f%%", np.percentile(dd_pct, 1))
    log.info("    Worst DD observed:           %.3f%%", dd_pct.min())
    log.info("")

    # Convert to margin-based
    dd_margin_pct = worst_peaks_to_troughs / margin * 100
    log.info("  MONTE CARLO MAX DRAWDOWN (on margin, INR):")
    log.info("    Mean trough:                 ₹%s", f'{worst_peaks_to_troughs.mean():,.0f}')
    log.info("    P99 trough (1%% worst):       ₹%s  (%.2f%% margin)",
             f'{np.percentile(worst_peaks_to_troughs, 1):,.0f}',
             np.percentile(dd_margin_pct, 1))
    log.info("    Worst trough:                ₹%s  (%.2f%% margin)",
             f'{worst_peaks_to_troughs.min():,.0f}',
             dd_margin_pct.min())
    log.info("")

    # Expected shortfall (CVaR 1%) — on final PnL
    threshold_1pct = np.percentile(final_pnls, 1)
    es_1pct = final_pnls[final_pnls <= threshold_1pct].mean()
    pnl_has_variance = (final_pnls.max() - final_pnls.min()) > 1
    log.info("  EXPECTED SHORTFALL (CVaR):")
    if pnl_has_variance:
        log.info("    1%% VaR (final PnL):          ₹%s", f'{threshold_1pct:,.0f}')
        log.info("    1%% Expected Shortfall:        ₹%s", f'{es_1pct:,.0f}')
        log.info("    As %% of margin:               %.2f%%", es_1pct / margin * 100)
    else:
        log.info("    (Final PnL identical across all paths — sum is order-invariant)")
        log.info("    Total PnL (all paths):        ₹%s", f'{final_pnls.mean():,.0f}')
        log.info("    Drawdown-based CVaR used instead (see DD metrics above)")
    log.info("")

    # 99th percentile equity drop
    p99_dd = np.percentile(dd_pct, 1)  # 1st percentile = worst 1%
    log.info("  99th PERCENTILE EQUITY DROP:   %.3f%% (on capital)", p99_dd)
    log.info("  99th PERCENTILE EQUITY DROP:   %.2f%% (on margin)", np.percentile(dd_margin_pct, 1))
    log.info("")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    dd_range = dd_pct.max() - dd_pct.min()
    dd_bins = max(1, min(80, int(dd_range / 0.005))) if dd_range > 0 else 1
    ax.hist(dd_pct, bins=dd_bins, edgecolor="black", alpha=0.7, color="#c44e52")
    ax.axvline(np.percentile(dd_pct, 1), color="orange", linestyle="--",
               label=f'P99: {np.percentile(dd_pct, 1):.3f}%')
    ax.axvline(dd_pct.min(), color="red", linestyle="-",
               label=f'Worst: {dd_pct.min():.3f}%')
    ax.set_title("MC Max Drawdown Distribution (% capital)", fontweight="bold")
    ax.set_xlabel("Max DD %"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1]
    if final_pnls.max() - final_pnls.min() > 1:
        ax.hist(final_pnls, bins=80, edgecolor="black", alpha=0.7, color="#4c72b0")
    else:
        ax.bar([final_pnls.mean()], [N_SIM], width=max(abs(final_pnls.mean()) * 0.02, 100),
               alpha=0.7, color="#4c72b0")
    ax.axvline(threshold_1pct, color="orange", linestyle="--",
               label=f'1% VaR: ₹{threshold_1pct:,.0f}')
    ax.axvline(0, color="red", linestyle="-", linewidth=1)
    ax.set_title("MC Final PnL Distribution", fontweight="bold")
    ax.set_xlabel("Total PnL (₹)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[2]
    # Plot 100 sample equity paths
    for i in range(min(200, N_SIM)):
        shuffled = np.random.permutation(daily_pnl)
        cum = capital + np.cumsum(shuffled)
        ax.plot(cum, alpha=0.03, color="#4c72b0", linewidth=0.5)
    ax.axhline(capital, color="red", linestyle="--", linewidth=1)
    ax.set_title("MC Equity Paths (200 samples)", fontweight="bold")
    ax.set_xlabel("Trading Day"); ax.set_ylabel("Equity (₹)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'₹{x:,.0f}'))
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT / "monte_carlo.png", dpi=150)
    plt.close(fig)
    log.info("  → monte_carlo.png saved")
    log.info("")

    return {
        "p99_dd_capital": round(p99_dd, 3),
        "worst_dd_capital": round(dd_pct.min(), 3),
        "p99_dd_margin": round(np.percentile(dd_margin_pct, 1), 2),
        "worst_dd_margin": round(dd_margin_pct.min(), 2),
        "expected_shortfall_1pct": round(es_1pct, 0),
        "es_pct_margin": round(es_1pct / margin * 100, 2),
        "pct_paths_profitable": round((final_pnls > 0).mean() * 100, 1),
    }


# ======================================================================= #
#  SECTION 5  –  FINAL DECISION                                            #
# ======================================================================= #

def section_5(base, syn, filter_results, mc):
    log.info("=" * 80)
    log.info("SECTION 5: FINAL DECISION")
    log.info("=" * 80)

    # Extract filter results
    base_nofilt = [r for r in filter_results if r["label"] == "BASE — no filter"][0]
    base_filt   = [r for r in filter_results if r["label"] == "BASE — filter ON"][0]
    syn_nofilt  = [r for r in filter_results if r["label"] == "HIGH-TREND — no filter"][0]

    log.info("")
    log.info("  ┌──────────────────────────────────────────────────────────────────┐")
    log.info("  │                     FINAL DECISION TABLE                         │")
    log.info("  ├──────────────────────────────────────────────────────────────────┤")

    # Q1: Does edge survive high-trend regime?
    log.info("  │                                                                  │")
    log.info("  │  Q1: Does edge SURVIVE high-trend regime?                        │")
    surviving = syn["sharpe_net"] > 1.0
    if syn["sharpe_net"] > 2.0:
        log.info("  │  ✓ YES — Sharpe %.2f under 30%% high-trend scenario           │", syn["sharpe_net"])
    elif surviving:
        log.info("  │  ~ CONDITIONAL — Sharpe %.2f (weakened but positive)           │", syn["sharpe_net"])
    else:
        log.info("  │  ✗ NO  — Sharpe %.2f under stress                              │", syn["sharpe_net"])
    log.info("  │    Base: Sharpe %.2f → Synthetic: Sharpe %.2f                    │", base["sharpe_net"], syn["sharpe_net"])
    log.info("  │    Base: ROM %.1f%% → Synthetic: ROM %.1f%%                        │", base["rom_ann_pct"], syn["rom_ann_pct"])

    # Q2: Does breakout stop protect capital?
    log.info("  │                                                                  │")
    log.info("  │  Q2: Does breakout stop PROTECT capital?                         │")
    worst_hist = base["worst_day_pct_margin"]
    worst_syn = syn["worst_day_pct_margin"]
    worst_mc = mc["worst_dd_margin"]
    protected = abs(worst_hist) < 5 and abs(worst_syn) < 10
    if protected:
        log.info("  │  ✓ YES                                                          │")
    else:
        log.info("  │  ✗ NO                                                            │")
    log.info("  │    Worst day (base):       %.2f%% margin                          │", worst_hist)
    log.info("  │    Worst day (synthetic):  %.2f%% margin                          │", worst_syn)
    log.info("  │    MC P99 drawdown:        %.2f%% margin                          │", mc["p99_dd_margin"])
    log.info("  │    MC worst drawdown:      %.2f%% margin                          │", worst_mc)

    # Q3: Is regime filter necessary?
    log.info("  │                                                                  │")
    log.info("  │  Q3: Is regime filter NECESSARY?                                 │")
    if base_filt.get("trades", 0) > 0:
        filter_helps = base_filt["sharpe_net"] > base_nofilt["sharpe_net"] + 0.5
        filter_ok = base_filt["sharpe_net"] >= base_nofilt["sharpe_net"] - 0.5
        if filter_helps:
            log.info("  │  YES — Filter improves Sharpe: %.2f → %.2f                   │",
                     base_nofilt["sharpe_net"], base_filt["sharpe_net"])
        elif filter_ok:
            log.info("  │  NO  — Filter OPTIONAL. Sharpe: %.2f → %.2f                  │",
                     base_nofilt["sharpe_net"], base_filt["sharpe_net"])
            log.info("  │    Edge already robust without filter.                        │")
        else:
            log.info("  │  NO  — Filter HURTS. Sharpe: %.2f → %.2f                     │",
                     base_nofilt["sharpe_net"], base_filt["sharpe_net"])
    else:
        log.info("  │  UNTESTABLE — Filter produced zero trades (thresholds tight)  │")

    # Q4: Is strategy robust across volatility shifts?
    log.info("  │                                                                  │")
    log.info("  │  Q4: Robust across volatility shifts?                            │")
    mc_profitable = mc["pct_paths_profitable"]
    robust = mc_profitable > 95 and surviving
    if robust:
        log.info("  │  ✓ YES                                                          │")
    elif mc_profitable > 80:
        log.info("  │  ~ MOSTLY — %.1f%% MC paths profitable                          │", mc_profitable)
    else:
        log.info("  │  ✗ NO  — Only %.1f%% MC paths profitable                        │", mc_profitable)
    log.info("  │    MC paths profitable:       %.1f%%                               │", mc_profitable)
    log.info("  │    MC expected shortfall:      ₹%s (%.2f%% margin)              │",
             f'{mc["expected_shortfall_1pct"]:,.0f}', mc["es_pct_margin"])

    log.info("  │                                                                  │")
    log.info("  └──────────────────────────────────────────────────────────────────┘")

    # Summary Table
    log.info("")
    log.info("  ╔══════════════════════════════════════════════════════════════════╗")
    log.info("  ║                     SUMMARY SCORECARD                           ║")
    log.info("  ╠══════════════════════════════╤═══════════════════════════════════╣")
    log.info("  ║ Metric                       │ Base    │ Syn 30%% │ Filter ON   ║")
    log.info("  ╠══════════════════════════════╪═══════════════════════════════════╣")
    log.info("  ║ Trades                       │ %5d   │ %5d    │ %5s        ║",
             base["trades"], syn["trades"],
             str(base_filt["trades"]) if base_filt.get("trades", 0) > 0 else "N/A")

    def _sv(r, k, fmt="{:.2f}"):
        return fmt.format(r[k]) if r.get("trades", 0) > 0 else "N/A"

    log.info("  ║ Sharpe (net)                 │ %7s │ %7s  │ %7s      ║",
             _sv(base, "sharpe_net"), _sv(syn, "sharpe_net"), _sv(base_filt, "sharpe_net"))
    log.info("  ║ Profit Factor                │ %7s │ %7s  │ %7s      ║",
             _sv(base, "pf_net", "{:.3f}"), _sv(syn, "pf_net", "{:.3f}"), _sv(base_filt, "pf_net", "{:.3f}"))
    log.info("  ║ ROM (ann.)                   │ %6s%% │ %6s%%  │ %6s%%     ║",
             _sv(base, "rom_ann_pct"), _sv(syn, "rom_ann_pct"), _sv(base_filt, "rom_ann_pct"))
    log.info("  ║ Max DD (capital)             │ %6s%% │ %6s%%  │ %6s%%     ║",
             _sv(base, "max_dd_pct", "{:.3f}"), _sv(syn, "max_dd_pct", "{:.3f}"), _sv(base_filt, "max_dd_pct", "{:.3f}"))
    log.info("  ║ Worst day (%% margin)         │ %6s%% │ %6s%%  │ %6s%%     ║",
             _sv(base, "worst_day_pct_margin"), _sv(syn, "worst_day_pct_margin"),
             _sv(base_filt, "worst_day_pct_margin"))
    log.info("  ║ MC P99 DD (margin)           │         %.2f%%                    ║", mc["p99_dd_margin"])
    log.info("  ║ MC paths profitable          │         %.1f%%                     ║", mc["pct_paths_profitable"])
    log.info("  ║ MC Exp Shortfall (1%%)        │         ₹%s                    ║",
             f'{mc["expected_shortfall_1pct"]:,.0f}')
    log.info("  ╠══════════════════════════════╧═══════════════════════════════════╣")

    # Composite
    score = 0
    if base["sharpe_net"] > 3: score += 2
    if syn["sharpe_net"] > 1: score += 2
    if syn["sharpe_net"] > 2: score += 1
    if protected: score += 1
    if mc_profitable > 95: score += 2
    if abs(mc["p99_dd_margin"]) < 10: score += 1
    if base["rom_ann_pct"] > 15: score += 1

    log.info("  ║                                                                  ║")
    log.info("  ║  COMPOSITE SCORE: %d / 10                                        ║", score)

    if score >= 8:
        log.info("  ║  ✓✓ VERDICT: REGIME-RESILIENT — proceed to live deployment      ║")
    elif score >= 6:
        log.info("  ║  ✓  VERDICT: CONDITIONALLY ROBUST — monitor vol regime live     ║")
    elif score >= 4:
        log.info("  ║  ~  VERDICT: FRAGILE — needs regime overlay for safety           ║")
    else:
        log.info("  ║  ✗  VERDICT: NOT REGIME-ROBUST                                   ║")

    log.info("  ║                                                                  ║")
    log.info("  ╚══════════════════════════════════════════════════════════════════╝")
    log.info("")


# ======================================================================= #
#  MAIN                                                                     #
# ======================================================================= #

def run_regime_validation():
    OUT.mkdir(parents=True, exist_ok=True)

    log.info("Loading data...")
    ce, pe = load_data(strike_filter="ATM")
    panel = build_intraday_panel(ce, pe)
    log.info("")

    day_df = section_1(panel)
    base, syn, synthetic_panel = section_2(panel, day_df)
    filter_results = section_3(panel, synthetic_panel)
    mc = section_4(panel)
    section_5(base, syn, filter_results, mc)

    log.info("All artefacts saved to: %s", OUT)


if __name__ == "__main__":
    run_regime_validation()
