"""
Production-Level Validation — Short ATM Straddle
=================================================
breakout_buffer_pct = 0.5%
stop_loss_pct = 20% and 25%
exit_time = 14:30

5 Sections:
  1. Full Robustness Re-Validation
  2. Synthetic Tail Stress Test
  3. Cost Sensitivity
  4. Capital Efficiency
  5. Final Decision Table
"""

from __future__ import annotations
import logging, sys
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
OUT = OUTPUT_DIR / "production_validation"


# ======================================================================= #
#  HELPERS                                                                  #
# ======================================================================= #

def _sharpe(daily_pnl: pd.Series, capital: float) -> float:
    dr = daily_pnl / capital
    return dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0.0

def _profit_factor(pnl: pd.Series) -> float:
    w = pnl[pnl > 0].sum(); l = abs(pnl[pnl <= 0].sum())
    return w / l if l > 0 else float("inf")

def _make_params(**kw) -> StrategyParams:
    base = dict(
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
    base.update(kw)
    return StrategyParams(**base)

def _run(panel, **kw):
    p = _make_params(**kw)
    trades = run_backtest(panel, p)
    return trades_to_dataframe(trades), p

def _margin(trade_df):
    """Estimated NIFTY short straddle margin per lot."""
    avg_spot = trade_df["spot_at_entry"].mean()
    return avg_spot * 25 * 0.15  # SPAN ~12% + exposure ~3%


# ======================================================================= #
#  SECTION 1  –  FULL ROBUSTNESS RE-VALIDATION                             #
# ======================================================================= #

def section_1(panel):
    log.info("=" * 80)
    log.info("SECTION 1: FULL ROBUSTNESS RE-VALIDATION  (buffer=0.5%%)")
    log.info("=" * 80)

    configs = [
        {"label": "SL=20%", "short_stop_loss_pct": 0.20},
        {"label": "SL=25%", "short_stop_loss_pct": 0.25},
    ]

    results = []

    for cfg in configs:
        label = cfg.pop("label")
        tdf, p = _run(panel, **cfg)
        m = compute_metrics(tdf, p)
        margin = _margin(tdf)

        daily_net = tdf.groupby("date")["net_pnl"].sum()
        daily_gross = tdf.groupby("date")["gross_pnl"].sum()
        sharpe_gross = _sharpe(daily_gross, p.initial_capital)
        sharpe_net = _sharpe(daily_net, p.initial_capital)
        pf_gross = _profit_factor(tdf["gross_pnl"])
        pf_net = _profit_factor(tdf["net_pnl"])
        total_pnl = tdf["net_pnl"].sum()
        gross_pnl = tdf["gross_pnl"].sum()
        rom = total_pnl / margin * 100
        days = (tdf["date"].max() - tdf["date"].min()).days
        years = days / 365.25
        rom_ann = ((1 + total_pnl / margin) ** (1 / years) - 1) * 100

        # Worst 5 trades
        worst5 = tdf.nsmallest(5, "net_pnl")[["date", "net_pnl", "exit_reason", "max_adverse_excursion"]].reset_index(drop=True)

        # Rolling 6-month Sharpe
        daily_all = tdf.groupby("date").agg(net=("net_pnl", "sum"), gross=("gross_pnl", "sum")).reset_index()
        daily_all["date"] = pd.to_datetime(daily_all["date"])
        daily_all["month"] = daily_all["date"].dt.to_period("M")
        months = daily_all["month"].unique()
        roll = []
        for i in range(len(months) - 5):
            s, e = months[i], months[i + 5]
            w = daily_all[(daily_all["month"] >= s) & (daily_all["month"] <= e)]
            if len(w) < 10:
                continue
            roll.append({
                "window": f"{s}→{e}",
                "sharpe_net": round(_sharpe(w["net"], p.initial_capital), 2),
                "sharpe_gross": round(_sharpe(w["gross"], p.initial_capital), 2),
                "ret_pct": round(w["net"].sum() / p.initial_capital * 100, 3),
            })
        roll_df = pd.DataFrame(roll)

        r = {
            "config": label,
            "trades": m["total_trades"],
            "win_rate": round(m["win_rate"] * 100, 1),
            "gross_pnl": round(gross_pnl, 0),
            "net_pnl": round(total_pnl, 0),
            "total_ret_pct": round(m["total_return"] * 100, 3),
            "rom_pct": round(rom, 2),
            "rom_ann_pct": round(rom_ann, 2),
            "sharpe_gross": round(sharpe_gross, 2),
            "sharpe_net": round(sharpe_net, 2),
            "pf_gross": round(pf_gross, 3),
            "pf_net": round(pf_net, 3),
            "max_dd_pct": round(m["max_drawdown"] * 100, 3),
            "margin": round(margin, 0),
            "rolling_sharpe_net_avg": round(roll_df["sharpe_net"].mean(), 2) if len(roll_df) else 0,
            "rolling_positive_pct": round((roll_df["sharpe_net"] > 0).mean() * 100, 0) if len(roll_df) else 0,
            "rolling_sharpe_gross_avg": round(roll_df["sharpe_gross"].mean(), 2) if len(roll_df) else 0,
        }
        results.append(r)

        log.info("")
        log.info("  CONFIG: %s  |  breakout_buffer=0.5%%  |  exit=14:30", label)
        log.info("  Trades:          %d  |  Win Rate: %.1f%%", r["trades"], r["win_rate"])
        log.info("  Gross PnL:       ₹%s  |  Net PnL: ₹%s", f'{r["gross_pnl"]:,.0f}', f'{r["net_pnl"]:,.0f}')
        log.info("  Total Return:    %.3f%% on ₹10L capital", r["total_ret_pct"])
        log.info("  Return on Margin: %.2f%% total  |  %.2f%% annualised", r["rom_pct"], r["rom_ann_pct"])
        log.info("  Sharpe (gross):  %.2f  |  Sharpe (net): %.2f", r["sharpe_gross"], r["sharpe_net"])
        log.info("  PF (gross):      %.3f  |  PF (net): %.3f", r["pf_gross"], r["pf_net"])
        log.info("  Max Drawdown:    %.3f%%", r["max_dd_pct"])
        log.info("")
        log.info("  ROLLING 6-MONTH SHARPE:")
        for _, rr in roll_df.iterrows():
            log.info("    %-18s  Sh(N)=%6.2f  Sh(G)=%6.2f  Ret=%7.3f%%",
                     rr["window"], rr["sharpe_net"], rr["sharpe_gross"], rr["ret_pct"])
        log.info("  Avg rolling Sharpe (Net): %.2f  |  %% positive: %.0f%%",
                 r["rolling_sharpe_net_avg"], r["rolling_positive_pct"])
        log.info("")
        log.info("  WORST 5 TRADES:")
        for _, w in worst5.iterrows():
            d = w["date"]
            log.info("    %s  PnL=₹%7.0f  MAE=₹%7.0f  Exit: %s",
                     d.strftime("%Y-%m-%d") if hasattr(d, 'strftime') else d,
                     w["net_pnl"], w["max_adverse_excursion"], w["exit_reason"])

    log.info("")
    log.info("  ──── CONFIRMATION CHECKS ────")
    for r in results:
        log.info("  %s: Gross Sharpe >3? %s (%.2f)  |  Net Sharpe >1? %s (%.2f)  |  Rolling stable? %s (%.0f%% +ve)",
                 r["config"],
                 "YES" if r["sharpe_gross"] > 3 else "NO", r["sharpe_gross"],
                 "YES" if r["sharpe_net"] > 1 else "NO", r["sharpe_net"],
                 "YES" if r["rolling_positive_pct"] > 60 else "NO", r["rolling_positive_pct"])
    log.info("")

    return results


# ======================================================================= #
#  SECTION 2  –  SYNTHETIC TAIL STRESS TEST                                #
# ======================================================================= #

def section_2(panel):
    log.info("=" * 80)
    log.info("SECTION 2: SYNTHETIC TAIL STRESS TEST")
    log.info("=" * 80)

    # Run base to get representative trade parameters
    tdf, p = _run(panel, short_stop_loss_pct=0.20)
    margin = _margin(tdf)
    lot = p.lot_size
    avg_spot = tdf["spot_at_entry"].mean()
    avg_ce = tdf["ce_entry"].mean()
    avg_pe = tdf["pe_entry"].mean()
    avg_premium = avg_ce + avg_pe

    # NIFTY ATM option sensitivities (representative)
    # ATM delta ~ 0.50 for CE, -0.50 for PE
    # ATM gamma ~ 0.0008 per point
    # ATM vega  ~ 5 per 1% IV
    delta_ce = 0.50
    delta_pe = -0.50
    gamma_per_pt = 0.0008     # gamma per NIFTY point move
    vega_per_1pct = 5.0       # vega per 1% IV change (per option)

    normal_slippage_pct = 0.05 / 100
    stress_slippage_pct = 5 * normal_slippage_pct  # 5× slippage

    log.info("")
    log.info("  Reference values:")
    log.info("    Avg Spot:      ₹%s", f"{avg_spot:,.0f}")
    log.info("    Avg CE entry:  ₹%.2f  |  Avg PE entry: ₹%.2f", avg_ce, avg_pe)
    log.info("    Entry premium: ₹%.2f  |  Margin: ₹%s", avg_premium, f"{margin:,.0f}")
    log.info("    Lot size:      %d", lot)
    log.info("")

    scenarios = []

    # ------ Scenario A: +2% trend day from 10:00 AM ------
    spot_move_a = avg_spot * 0.02   # +2% = ~497 pts
    # CE premium rises: delta * move + 0.5 * gamma * move^2
    # PE premium falls: delta * move + 0.5 * gamma * move^2
    # IV typically rises on trend days: assume +3% IV spike
    iv_spike = 3.0
    ce_change_a = delta_ce * spot_move_a + 0.5 * gamma_per_pt * spot_move_a**2 + vega_per_1pct * iv_spike
    pe_change_a = delta_pe * spot_move_a + 0.5 * gamma_per_pt * spot_move_a**2 + vega_per_1pct * iv_spike
    new_ce_a = avg_ce + ce_change_a
    new_pe_a = max(avg_pe + pe_change_a, 5.0)  # PE floor at 5
    exit_premium_a = new_ce_a + new_pe_a
    gross_loss_a = (avg_premium - exit_premium_a) * lot
    # Also: breakout stop would fire well before +2%, so actual loss clipped
    # But we model the scenario WITHOUT breakout stop (worst case)
    slippage_stress_a = exit_premium_a * stress_slippage_pct * lot * 2
    total_loss_a = gross_loss_a - slippage_stress_a  # loss is negative, slippage makes it worse
    txn_cost_a = compute_txn_cost(avg_ce, avg_pe, new_ce_a, new_pe_a, lot, p)

    scenarios.append({
        "scenario": "A: +2% trend day",
        "spot_move": f"+{spot_move_a:.0f} pts (+2%)",
        "iv_spike": f"+{iv_spike}%",
        "entry_premium": round(avg_premium, 2),
        "exit_premium": round(exit_premium_a, 2),
        "gross_pnl": round(gross_loss_a, 0),
        "txn_cost": round(txn_cost_a, 0),
        "stress_slippage": round(slippage_stress_a, 0),
        "net_pnl_normal": round(gross_loss_a - txn_cost_a, 0),
        "net_pnl_stress": round(gross_loss_a - txn_cost_a - slippage_stress_a, 0),
        "pct_margin_normal": round((gross_loss_a - txn_cost_a) / margin * 100, 2),
        "pct_margin_stress": round((gross_loss_a - txn_cost_a - slippage_stress_a) / margin * 100, 2),
    })

    # ------ Scenario B: -2% trend day from 10:00 AM ------
    spot_move_b = avg_spot * -0.02
    ce_change_b = delta_ce * spot_move_b + 0.5 * gamma_per_pt * spot_move_b**2 + vega_per_1pct * iv_spike
    pe_change_b = delta_pe * spot_move_b + 0.5 * gamma_per_pt * spot_move_b**2 + vega_per_1pct * iv_spike
    new_ce_b = max(avg_ce + ce_change_b, 5.0)
    new_pe_b = avg_pe + pe_change_b
    exit_premium_b = new_ce_b + new_pe_b
    gross_loss_b = (avg_premium - exit_premium_b) * lot
    slippage_stress_b = exit_premium_b * stress_slippage_pct * lot * 2
    total_loss_b = gross_loss_b - slippage_stress_b
    txn_cost_b = compute_txn_cost(avg_ce, avg_pe, new_ce_b, new_pe_b, lot, p)

    scenarios.append({
        "scenario": "B: -2% trend day",
        "spot_move": f"{spot_move_b:.0f} pts (-2%)",
        "iv_spike": f"+{iv_spike}%",
        "entry_premium": round(avg_premium, 2),
        "exit_premium": round(exit_premium_b, 2),
        "gross_pnl": round(gross_loss_b, 0),
        "txn_cost": round(txn_cost_b, 0),
        "stress_slippage": round(slippage_stress_b, 0),
        "net_pnl_normal": round(gross_loss_b - txn_cost_b, 0),
        "net_pnl_stress": round(gross_loss_b - txn_cost_b - slippage_stress_b, 0),
        "pct_margin_normal": round((gross_loss_b - txn_cost_b) / margin * 100, 2),
        "pct_margin_stress": round((gross_loss_b - txn_cost_b - slippage_stress_b) / margin * 100, 2),
    })

    # ------ Scenario C: +1.5% sudden spike in 3 candles (9 min) ------
    spot_move_c = avg_spot * 0.015
    # Liquidity shock: assume IV spikes 5% (higher for sudden)
    iv_spike_c = 5.0
    ce_change_c = delta_ce * spot_move_c + 0.5 * gamma_per_pt * spot_move_c**2 + vega_per_1pct * iv_spike_c
    pe_change_c = delta_pe * spot_move_c + 0.5 * gamma_per_pt * spot_move_c**2 + vega_per_1pct * iv_spike_c
    new_ce_c = avg_ce + ce_change_c
    new_pe_c = max(avg_pe + pe_change_c, 5.0)
    exit_premium_c = new_ce_c + new_pe_c
    gross_loss_c = (avg_premium - exit_premium_c) * lot
    # Sudden move → much worse slippage. 5x on both legs.
    slippage_stress_c = exit_premium_c * stress_slippage_pct * lot * 2
    txn_cost_c = compute_txn_cost(avg_ce, avg_pe, new_ce_c, new_pe_c, lot, p)

    scenarios.append({
        "scenario": "C: +1.5% spike (3 candles)",
        "spot_move": f"+{spot_move_c:.0f} pts (+1.5%)",
        "iv_spike": f"+{iv_spike_c}%",
        "entry_premium": round(avg_premium, 2),
        "exit_premium": round(exit_premium_c, 2),
        "gross_pnl": round(gross_loss_c, 0),
        "txn_cost": round(txn_cost_c, 0),
        "stress_slippage": round(slippage_stress_c, 0),
        "net_pnl_normal": round(gross_loss_c - txn_cost_c, 0),
        "net_pnl_stress": round(gross_loss_c - txn_cost_c - slippage_stress_c, 0),
        "pct_margin_normal": round((gross_loss_c - txn_cost_c) / margin * 100, 2),
        "pct_margin_stress": round((gross_loss_c - txn_cost_c - slippage_stress_c) / margin * 100, 2),
    })

    log.info("")
    log.info("  %-30s %12s %12s %12s %12s %10s %10s",
             "Scenario", "Gross PnL", "Txn Cost", "Net(normal)", "Net(5×slip)", "%Margin(N)", "%Margin(S)")
    for s in scenarios:
        log.info("  %-30s ₹%10s ₹%10s ₹%10s ₹%10s %9.2f%% %9.2f%%",
                 s["scenario"],
                 f'{s["gross_pnl"]:,.0f}', f'{s["txn_cost"]:,.0f}',
                 f'{s["net_pnl_normal"]:,.0f}', f'{s["net_pnl_stress"]:,.0f}',
                 s["pct_margin_normal"], s["pct_margin_stress"])

    worst = min(scenarios, key=lambda x: x["pct_margin_stress"])
    log.info("")
    log.info("  WORST-CASE STRESS LOSS: ₹%s  =  %.2f%% of margin",
             f'{worst["net_pnl_stress"]:,.0f}', worst["pct_margin_stress"])
    log.info("  (Scenario: %s with 5× slippage)", worst["scenario"])
    log.info("")

    # ---- Realistic note: breakout stop would fire much earlier ----
    breakout_fire_pct = 0.005  # 0.5% buffer
    breakout_fire_move = avg_spot * breakout_fire_pct
    # But opening range width adds to this. Approximate total spot move before breakout:
    avg_or_width = tdf["spot_range_high"].mean() - tdf["spot_range_low"].mean()
    total_buffer = avg_or_width / 2 + breakout_fire_move
    total_buffer_pct = total_buffer / avg_spot * 100

    ce_change_bo = delta_ce * total_buffer + vega_per_1pct * 1.0
    pe_change_bo = abs(delta_pe) * total_buffer * -1 + vega_per_1pct * 1.0
    exit_at_breakout = avg_premium + ce_change_bo + pe_change_bo
    breakout_loss = (avg_premium - exit_at_breakout) * lot
    breakout_pct_margin = breakout_loss / margin * 100

    log.info("  BREAKOUT STOP REALITY CHECK:")
    log.info("    Avg opening range width:    ₹%.0f pts", avg_or_width)
    log.info("    Breakout fires at:          ~%.2f%% spot move from mid-range", total_buffer_pct)
    log.info("    Estimated loss at breakout: ₹%.0f  =  %.2f%% margin",
             breakout_loss, breakout_pct_margin)
    log.info("    ➜ In practice, scenarios A/B/C are largely clipped by breakout stop.")
    log.info("    ➜ Unclipped tail only occurs if breakout stop fails (gap/illiquidity).")
    log.info("")

    return scenarios, worst


# ======================================================================= #
#  SECTION 3  –  COST SENSITIVITY                                          #
# ======================================================================= #

def section_3(panel):
    log.info("=" * 80)
    log.info("SECTION 3: COST SENSITIVITY  (breakout=0.5%%, SL=20%%, exit=14:30)")
    log.info("=" * 80)

    cost_configs = [
        {"label": "Current costs", "brokerage_per_leg": 20.0, "slippage_pct": 0.05, "stt_on_sell_pct": 0.0625},
        {"label": "50% brokerage", "brokerage_per_leg": 10.0, "slippage_pct": 0.05, "stt_on_sell_pct": 0.0625},
        {"label": "Zero brokerage", "brokerage_per_leg": 0.0, "slippage_pct": 0.05, "stt_on_sell_pct": 0.0625},
        {"label": "2× slippage", "brokerage_per_leg": 20.0, "slippage_pct": 0.10, "stt_on_sell_pct": 0.0625},
        {"label": "Zero all costs", "brokerage_per_leg": 0.0, "slippage_pct": 0.0, "stt_on_sell_pct": 0.0},
    ]

    results = []
    for cfg in cost_configs:
        label = cfg.pop("label")
        tdf, p = _run(panel, short_stop_loss_pct=0.20, **cfg)
        m = compute_metrics(tdf, p)
        margin = _margin(tdf)
        daily = tdf.groupby("date")["net_pnl"].sum()
        net_pnl = tdf["net_pnl"].sum()
        total_cost = tdf["txn_cost"].sum()
        days = (tdf["date"].max() - tdf["date"].min()).days
        years = days / 365.25
        rom_ann = ((1 + net_pnl / margin) ** (1 / years) - 1) * 100

        r = {
            "label": label,
            "net_pnl": round(net_pnl, 0),
            "total_costs": round(total_cost, 0),
            "sharpe_net": round(m["sharpe_ratio"], 2),
            "cagr_pct": round(m["cagr"] * 100, 3),
            "rom_ann_pct": round(rom_ann, 2),
            "pf_net": round(m["profit_factor"], 3),
            "max_dd_pct": round(m["max_drawdown"] * 100, 3),
        }
        results.append(r)

    log.info("")
    log.info("  %-20s %10s %10s %8s %8s %8s %8s %8s",
             "Cost Scenario", "Net PnL", "Costs", "Sharpe", "CAGR%", "ROM%ann", "PF", "MaxDD%")
    for r in results:
        log.info("  %-20s ₹%8s ₹%8s %8.2f %7.3f%% %7.2f%% %8.3f %7.3f%%",
                 r["label"],
                 f'{r["net_pnl"]:,.0f}', f'{r["total_costs"]:,.0f}',
                 r["sharpe_net"], r["cagr_pct"], r["rom_ann_pct"],
                 r["pf_net"], r["max_dd_pct"])

    # Execution sensitivity
    current_sharpe = results[0]["sharpe_net"]
    double_slip_sharpe = results[3]["sharpe_net"]
    zero_cost_sharpe = results[4]["sharpe_net"]

    log.info("")
    log.info("  EXECUTION SENSITIVITY ANALYSIS:")
    log.info("    Sharpe at current costs:  %.2f", current_sharpe)
    log.info("    Sharpe at 2× slippage:    %.2f  (Δ = %.2f)", double_slip_sharpe, double_slip_sharpe - current_sharpe)
    log.info("    Sharpe at zero costs:     %.2f  (ceiling)", zero_cost_sharpe)
    log.info("    Sharpe degradation from 2× slip: %.1f%%", (current_sharpe - double_slip_sharpe) / current_sharpe * 100 if current_sharpe else 0)

    if double_slip_sharpe > 1.0:
        log.info("    ✓ Strategy SURVIVES 2× slippage (Sharpe still >1)")
    elif double_slip_sharpe > 0.5:
        log.info("    ~ Strategy WEAKENED by 2× slippage but still positive")
    else:
        log.info("    ✗ Strategy DESTROYED by 2× slippage")

    log.info("")
    return results


# ======================================================================= #
#  SECTION 4  –  CAPITAL EFFICIENCY                                        #
# ======================================================================= #

def section_4(panel):
    log.info("=" * 80)
    log.info("SECTION 4: CAPITAL EFFICIENCY ANALYSIS")
    log.info("=" * 80)

    configs = [
        {"label": "SL=20%", "short_stop_loss_pct": 0.20},
        {"label": "SL=25%", "short_stop_loss_pct": 0.25},
    ]

    results = []
    for cfg in configs:
        label = cfg.pop("label")
        tdf, p = _run(panel, **cfg)
        m = compute_metrics(tdf, p)
        margin = _margin(tdf)
        capital = p.initial_capital
        total_pnl = tdf["net_pnl"].sum()
        days = (tdf["date"].max() - tdf["date"].min()).days
        years = days / 365.25

        rom = total_pnl / margin
        rom_ann = (1 + rom) ** (1 / years) - 1

        daily = tdf.groupby("date")["net_pnl"].sum()
        worst_day = daily.min()
        worst_day_pct_margin = worst_day / margin * 100

        max_dd_pct = m["max_drawdown"]  # on capital
        max_dd_on_margin = abs(max_dd_pct) * capital / margin * 100

        risk_adj_rom = rom_ann / abs(max_dd_pct) if max_dd_pct != 0 else float("inf")

        # Alternative benchmark: NIFTY index return
        first_spot = tdf.iloc[0]["spot_at_entry"]
        last_spot = tdf.iloc[-1]["spot_at_entry"]
        nifty_return = (last_spot / first_spot - 1) * 100
        nifty_ann = ((last_spot / first_spot) ** (1 / years) - 1) * 100

        # FD rate benchmark
        fd_ann = 7.0  # 7% FD rate

        r = {
            "label": label,
            "margin": round(margin, 0),
            "rom_total_pct": round(rom * 100, 2),
            "rom_ann_pct": round(rom_ann * 100, 2),
            "worst_day_pnl": round(worst_day, 0),
            "worst_day_pct_margin": round(worst_day_pct_margin, 2),
            "max_dd_capital_pct": round(max_dd_pct * 100, 3),
            "max_dd_margin_pct": round(max_dd_on_margin, 2),
            "risk_adj_rom": round(risk_adj_rom, 2),
            "nifty_ann_pct": round(nifty_ann, 2),
            "fd_ann_pct": fd_ann,
            "rom_minus_fd": round(rom_ann * 100 - fd_ann, 2),
        }
        results.append(r)

    log.info("")
    for r in results:
        log.info("  CONFIG: %s  |  breakout_buffer=0.5%%", r["label"])
        log.info("    Estimated margin per trade:   ₹%s", f'{r["margin"]:,.0f}')
        log.info("    Return on margin (total):     %.2f%%", r["rom_total_pct"])
        log.info("    Return on margin (annualised):%.2f%%", r["rom_ann_pct"])
        log.info("    Worst daily loss:             ₹%s  (%.2f%% of margin)",
                 f'{r["worst_day_pnl"]:,.0f}', r["worst_day_pct_margin"])
        log.info("    Max DD (on capital):          %.3f%%", r["max_dd_capital_pct"])
        log.info("    Max DD (on margin):           %.2f%%", r["max_dd_margin_pct"])
        log.info("    Risk-Adjusted ROM (ROM/DD):   %.2f", r["risk_adj_rom"])
        log.info("")
        log.info("    BENCHMARKS:")
        log.info("      NIFTY buy-hold ann. return: %.2f%%", r["nifty_ann_pct"])
        log.info("      FD rate (ann.):             %.2f%%", r["fd_ann_pct"])
        log.info("      ROM excess over FD:         %.2f%%", r["rom_minus_fd"])
        log.info("")

        if r["rom_ann_pct"] > r["fd_ann_pct"]:
            log.info("    ✓ Margin-efficient: ROM (%.2f%%) exceeds FD rate (%.2f%%)", r["rom_ann_pct"], r["fd_ann_pct"])
        else:
            log.info("    ✗ Not margin-efficient: ROM (%.2f%%) below FD rate (%.2f%%)", r["rom_ann_pct"], r["fd_ann_pct"])
        log.info("")

    return results


# ======================================================================= #
#  SECTION 5  –  FINAL DECISION TABLE                                      #
# ======================================================================= #

def section_5(s1_results, s2_scenarios, s2_worst, s3_results, s4_results):
    log.info("=" * 80)
    log.info("SECTION 5: FINAL DECISION OUTPUT")
    log.info("=" * 80)

    # Use SL=20% as primary config for final assessment
    r1 = s1_results[0]  # SL=20%
    r1_25 = s1_results[1]  # SL=25%
    r3_current = s3_results[0]
    r3_double_slip = s3_results[3]
    r3_zero = s3_results[4]
    r4 = s4_results[0]  # SL=20%

    log.info("")
    log.info("  ┌──────────────────────────────────────────────────────────────────┐")
    log.info("  │                    FINAL DECISION TABLE                          │")
    log.info("  ├──────────────────────────────────────────────────────────────────┤")
    log.info("  │                                                                  │")

    # Q1: Structurally tradable?
    gross_ok = r1["sharpe_gross"] > 3.0
    net_ok = r1["sharpe_net"] > 1.0
    broad = r1["rolling_positive_pct"] > 50
    pf_ok = r1["pf_net"] > 1.1

    q1 = gross_ok and net_ok and broad
    log.info("  │  Q1: STRUCTURALLY TRADABLE?                                     │")
    if q1:
        log.info("  │  ✓ YES                                                          │")
    elif gross_ok and (net_ok or broad):
        log.info("  │  ~ CONDITIONAL                                                   │")
    else:
        log.info("  │  ✗ NO                                                            │")
    log.info("  │    Gross Sharpe: %.2f (need >3)  %s                            │",
             r1["sharpe_gross"], "✓" if gross_ok else "✗")
    log.info("  │    Net Sharpe:   %.2f (need >1)  %s                            │",
             r1["sharpe_net"], "✓" if net_ok else "✗")
    log.info("  │    Rolling %%+ve: %.0f%% (need >50)  %s                          │",
             r1["rolling_positive_pct"], "✓" if broad else "✗")
    log.info("  │    PF net:       %.3f (need >1.1)  %s                          │",
             r1["pf_net"], "✓" if pf_ok else "✗")

    # Q2: Execution-sensitive?
    sharpe_drop = (r3_current["sharpe_net"] - r3_double_slip["sharpe_net"]) / r3_current["sharpe_net"] * 100 if r3_current["sharpe_net"] else 0
    execution_sensitive = sharpe_drop > 30
    log.info("  │                                                                  │")
    log.info("  │  Q2: EXECUTION-SENSITIVE?                                        │")
    if execution_sensitive:
        log.info("  │  ⚠ YES — Sharpe drops %.0f%% at 2× slippage                     │", sharpe_drop)
    else:
        log.info("  │  ✓ NO  — Sharpe drops only %.0f%% at 2× slippage                 │", sharpe_drop)
    log.info("  │    Current Sharpe: %.2f  |  2× Slippage Sharpe: %.2f             │",
             r3_current["sharpe_net"], r3_double_slip["sharpe_net"])
    log.info("  │    Zero-cost Sharpe ceiling: %.2f                                │",
             r3_zero["sharpe_net"])

    # Q3: Tail risk manageable?
    worst_stress = s2_worst["pct_margin_stress"]
    tail_ok = abs(worst_stress) < 15
    log.info("  │                                                                  │")
    log.info("  │  Q3: TAIL RISK MANAGEABLE?                                       │")
    if tail_ok:
        log.info("  │  ✓ YES                                                          │")
    else:
        log.info("  │  ✗ NO                                                            │")
    log.info("  │    Worst stress loss:   %.2f%% of margin (5× slippage)            │", worst_stress)
    log.info("  │    Worst historical:    %.2f%% of margin                          │", r4["worst_day_pct_margin"])
    log.info("  │    Max DD on margin:    %.2f%%                                    │", r4["max_dd_margin_pct"])

    # Q4: Edge scalable?
    # Scalability limited by: (a) single lot, (b) NIFTY options liquidity (massive)
    # (c) cost amortisation improves with lots
    cost_pct_of_gross = s3_results[0]["total_costs"] / (s3_results[0]["net_pnl"] + s3_results[0]["total_costs"]) * 100 if (s3_results[0]["net_pnl"] + s3_results[0]["total_costs"]) > 0 else 100
    # Fixed cost per trade = brokerage = 80. At 2 lots, brokerage/turnover halves.
    scalable = cost_pct_of_gross < 90 and r4["rom_ann_pct"] > 5
    log.info("  │                                                                  │")
    log.info("  │  Q4: EDGE SCALABLE?                                              │")
    if scalable:
        log.info("  │  ✓ YES — costs amortise with more lots                          │")
    else:
        log.info("  │  ~ CONDITIONAL — costs eat %.0f%% of gross; ROM %.1f%%             │",
                 cost_pct_of_gross, r4["rom_ann_pct"])
    log.info("  │    Fixed cost (brokerage): ₹80/trade — amortises at >1 lot      │")
    log.info("  │    NIFTY liquidity: no fill risk even at 10+ lots                │")
    log.info("  │    At 2 lots: brokerage cost halves per-lot                      │")

    log.info("  │                                                                  │")
    log.info("  └──────────────────────────────────────────────────────────────────┘")

    # ---- SUMMARY TABLE ----
    log.info("")
    log.info("  ╔══════════════════════════════════════════════════════════════════╗")
    log.info("  ║                     SUMMARY SCORECARD                           ║")
    log.info("  ╠═══════════════════════════════╤════════════════════════════════╣")
    log.info("  ║ Metric                        │ SL=20%%      │ SL=25%%         ║")
    log.info("  ╠═══════════════════════════════╪════════════════════════════════╣")
    log.info("  ║ Trades                        │ %4d         │ %4d            ║", r1["trades"], r1_25["trades"])
    log.info("  ║ Win Rate                      │ %5.1f%%       │ %5.1f%%          ║", r1["win_rate"], r1_25["win_rate"])
    log.info("  ║ Net Return (%%cap)             │ %6.3f%%      │ %6.3f%%         ║", r1["total_ret_pct"], r1_25["total_ret_pct"])
    log.info("  ║ ROM (annualised)              │ %6.2f%%      │ %6.2f%%         ║", r1["rom_ann_pct"], r1_25["rom_ann_pct"])
    log.info("  ║ Sharpe GROSS                  │ %6.2f       │ %6.2f          ║", r1["sharpe_gross"], r1_25["sharpe_gross"])
    log.info("  ║ Sharpe NET                    │ %6.2f       │ %6.2f          ║", r1["sharpe_net"], r1_25["sharpe_net"])
    log.info("  ║ Profit Factor (net)           │ %6.3f       │ %6.3f          ║", r1["pf_net"], r1_25["pf_net"])
    log.info("  ║ Max Drawdown                  │ %6.3f%%      │ %6.3f%%         ║", r1["max_dd_pct"], r1_25["max_dd_pct"])
    log.info("  ║ Rolling 6M positive           │ %5.0f%%       │ %5.0f%%          ║", r1["rolling_positive_pct"], r1_25["rolling_positive_pct"])
    log.info("  ║ Avg rolling Sharpe (net)      │ %6.2f       │ %6.2f          ║", r1["rolling_sharpe_net_avg"], r1_25["rolling_sharpe_net_avg"])
    log.info("  ║ Worst stress loss (%%margin)   │ %6.2f%%      │                ║", s2_worst["pct_margin_stress"])
    log.info("  ║ Execution sensitivity (2×slp) │ %.0f%% Sharpe drop              ║", sharpe_drop)
    log.info("  ╠═══════════════════════════════╧════════════════════════════════╣")

    # Final call
    score = 0
    if gross_ok: score += 2
    if net_ok: score += 2
    if broad: score += 1
    if pf_ok: score += 1
    if not execution_sensitive: score += 1
    if tail_ok: score += 1
    if r4["rom_ann_pct"] > 7: score += 1
    if r4["rom_ann_pct"] > 15: score += 1

    log.info("  ║                                                                  ║")
    log.info("  ║  COMPOSITE SCORE: %d / 10                                        ║", score)
    log.info("  ║                                                                  ║")

    if score >= 8:
        verdict = "PROCEED TO PAPER TRADING → LIVE with tight position limits"
        log.info("  ║  ✓✓ VERDICT: TRADABLE                                           ║")
    elif score >= 6:
        verdict = "CONDITIONAL: Needs cost reduction or wider buffer"
        log.info("  ║  ~  VERDICT: CONDITIONAL                                        ║")
    elif score >= 4:
        verdict = "MARGINAL: Not worth operational overhead at current params"
        log.info("  ║  ~  VERDICT: MARGINAL                                            ║")
    else:
        verdict = "REJECT: No deployable edge"
        log.info("  ║  ✗  VERDICT: REJECT                                              ║")

    log.info("  ║  %s", verdict.ljust(65) + "║")
    log.info("  ║                                                                  ║")
    log.info("  ╚══════════════════════════════════════════════════════════════════╝")
    log.info("")

    return {"score": score, "verdict": verdict}


# ======================================================================= #
#  MAIN                                                                     #
# ======================================================================= #

def run_production_validation():
    OUT.mkdir(parents=True, exist_ok=True)

    log.info("Loading data...")
    ce, pe = load_data(strike_filter="ATM")
    panel = build_intraday_panel(ce, pe)
    log.info("")

    s1 = section_1(panel)
    s2_scenarios, s2_worst = section_2(panel)
    s3 = section_3(panel)
    s4 = section_4(panel)
    section_5(s1, s2_scenarios, s2_worst, s3, s4)

    # Save all result tables
    pd.DataFrame(s1).to_csv(OUT / "section1_robustness.csv", index=False)
    pd.DataFrame(s2_scenarios).to_csv(OUT / "section2_stress.csv", index=False)
    pd.DataFrame(s3).to_csv(OUT / "section3_cost_sensitivity.csv", index=False)
    pd.DataFrame(s4).to_csv(OUT / "section4_capital_efficiency.csv", index=False)

    log.info("All artefacts saved to: %s", OUT)


if __name__ == "__main__":
    run_production_validation()
