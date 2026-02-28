"""
Streamlit UI – ATM Straddle Engine
===================================
Interactive front-end for the NIFTY short-straddle backtester.

Launch:
    streamlit run -m straddle_scalper.app
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Ensure the project root is on sys.path so absolute imports work
# when Streamlit runs this file as a standalone script.
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from straddle_scalper.config import StrategyParams
from straddle_scalper.data_loader import load_data, build_intraday_panel
from straddle_scalper.backtester import run_backtest, trades_to_dataframe
from straddle_scalper.metrics import compute_metrics, build_equity_curve, daily_return_series
from straddle_scalper.robustness_validator import parameter_sensitivity_grid

logger = logging.getLogger(__name__)

# ───────────────────────────────────────────────────────────────────────────── #
# 1. PAGE CONFIG & GLOBAL STYLES
# ───────────────────────────────────────────────────────────────────────────── #

st.set_page_config(
    page_title="ATM Straddle Engine – NIFTY Backtest",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Dense dark-theme CSS
st.markdown("""
<style>
    .block-container {
        padding-top: 1.2rem !important;
        padding-bottom: 0.5rem !important;
        max-width: 100% !important;
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 0.6rem !important;
    }
    /* Use standard sans-serif in sidebar to avoid slashed-zero glyphs */
    section[data-testid="stSidebar"] * {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
        font-variant-numeric: normal !important;
        font-feature-settings: normal !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.75rem !important;
        color: #8a8d93 !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.25rem !important;
        font-weight: 500 !important;
    }
    /* Hide anchor/link icons next to headers */
    .stMainBlockContainer h1 a,
    .stMainBlockContainer h2 a,
    .stMainBlockContainer h3 a,
    [data-testid="stHeaderActionElements"] { display: none !important; visibility: hidden !important; }
    a.header-link { display: none !important; }
    .stPlotlyChart { margin-top: -6px !important; margin-bottom: -2px !important; }
    .subcaption-block {
        font-size: 0.78rem;
        color: #9ca3af;
        line-height: 1.6;
        margin-top: -0.4rem;
        margin-bottom: 0.8rem;
    }
    .subcaption-block strong { color: #d1d5db; }
</style>
""", unsafe_allow_html=True)

# ───────────────────────────────────────────────────────────────────────────── #
# 2. HEADER
# ───────────────────────────────────────────────────────────────────────────── #

st.header("ATM Straddle Engine – NIFTY Backtest")

st.markdown("""
<div class="subcaption-block">
    <strong>Strategy:</strong> Intraday ATM Short Straddle with Breakout Stop &nbsp;·&nbsp;
    <strong>Dataset:</strong> Aug 2024 – Feb 2026 (~369 trading days) &nbsp;·&nbsp;
    <strong>Data:</strong> 3-minute NIFTY options candles<br>
    <strong>Risk Model:</strong> Spot Breakout Exit + Premium Stop + Hard Time Exit &nbsp;·&nbsp;
    <strong>Cost Model:</strong> Brokerage + Slippage + STT<br>
    <strong>Margin Assumption:</strong> SPAN + Exposure (~₹93k per lot) &nbsp;·&nbsp;
    <strong>Validation:</strong> Rolling Sharpe, Monte Carlo, Regime Stress &nbsp;·&nbsp;
    <strong>Version:</strong> Production Candidate<br>
    <em style="color:#6b7280;">A project by Nalan Baburajan</em>
</div>
""", unsafe_allow_html=True)

# ───────────────────────────────────────────────────────────────────────────── #
# 3. SIDEBAR CONTROLS
# ───────────────────────────────────────────────────────────────────────────── #

with st.sidebar:
    # ── 🎯 Entry Filters ─────────────────────────────────────────────────
    st.markdown("**🎯 Entry Filters**")
    compression_filter_enabled = st.checkbox(
        "Enable Compression Filter",
        value=StrategyParams.compression_filter_enabled,
    )
    compression_threshold = st.slider(
        "Compression Threshold (%)",
        min_value=0.0,
        max_value=2.0,
        value=StrategyParams.compression_threshold * 100,
        step=0.05,
        format="%.2f",
    ) / 100.0

    # ── 📋 Exit Parameters ───────────────────────────────────────────────
    st.markdown("**📋 Exit Parameters**")
    st.caption("Short straddle exits")
    short_stop_loss_pct = st.slider(
        "Premium Stop-Loss (%)",
        min_value=0.0,
        max_value=100.0,
        value=StrategyParams.short_stop_loss_pct * 100,
        step=1.0,
        format="%.2f",
    ) / 100.0
    short_profit_target_pct = st.slider(
        "Profit Booking (%)",
        min_value=0.0,
        max_value=100.0,
        value=StrategyParams.short_profit_target_pct * 100,
        step=1.0,
        format="%.2f",
    ) / 100.0

    # ── 📊 Adaptive Breakout Buffer ────────────────────────────────────
    st.markdown("**📊 Adaptive Breakout Buffer**")
    adaptive_breakout_enabled = st.checkbox(
        "Use Adaptive Breakout Buffer",
        value=StrategyParams.adaptive_breakout_enabled,
    )

    if adaptive_breakout_enabled:
        adaptive_k = st.slider(
            "k (volatility multiplier)",
            min_value=0.0,
            max_value=3.0,
            value=StrategyParams.adaptive_k,
            step=0.05,
            format="%.2f",
        )
        adaptive_lookback_days = st.slider(
            "Lookback Days (N)",
            min_value=1,
            max_value=30,
            value=StrategyParams.adaptive_lookback_days,
            step=1,
        )
        adaptive_min_buffer_pct = st.slider(
            "Min Buffer (%)",
            min_value=0.0,
            max_value=2.0,
            value=StrategyParams.adaptive_min_buffer_pct * 100,
            step=0.05,
            format="%.2f",
        ) / 100.0
        adaptive_max_buffer_pct = st.slider(
            "Max Buffer (%)",
            min_value=0.0,
            max_value=3.0,
            value=StrategyParams.adaptive_max_buffer_pct * 100,
            step=0.05,
            format="%.2f",
        ) / 100.0
    else:
        adaptive_k = StrategyParams.adaptive_k
        adaptive_lookback_days = StrategyParams.adaptive_lookback_days
        adaptive_min_buffer_pct = StrategyParams.adaptive_min_buffer_pct
        adaptive_max_buffer_pct = StrategyParams.adaptive_max_buffer_pct

    breakout_buffer_pct = st.slider(
        "Spot Breakout Buffer – Fixed (%)",
        min_value=0.0,
        max_value=3.0,
        value=StrategyParams.breakout_buffer_pct * 100,
        step=0.05,
        format="%.2f",
    ) / 100.0

    # ── ⏱ Hard Exit Time ────────────────────────────────────────────────
    st.markdown("**⏱ Hard Exit Time**")
    exit_time_options = [
        "13:00:00", "13:30:00", "14:00:00", "14:30:00", "14:45:00", "15:00:00",
    ]
    exit_time = st.selectbox(
        "Hard Exit Time",
        options=exit_time_options,
        index=exit_time_options.index("14:30:00"),
    )

    # ── 💰 Transaction Costs ────────────────────────────────────────────
    st.markdown("**💰 Transaction Costs**")
    brokerage_per_leg = st.number_input(
        "Brokerage per leg (INR)",
        min_value=0.0,
        max_value=500.0,
        value=StrategyParams.brokerage_per_leg,
        step=5.0,
        format="%.2f",
    )
    slippage_pct = st.number_input(
        "Slippage (%)",
        min_value=0.0,
        max_value=5.0,
        value=StrategyParams.slippage_pct,
        step=0.01,
        format="%.3f",
    )
    stt_on_sell_pct = st.number_input(
        "STT on sell (%)",
        min_value=0.0,
        max_value=1.0,
        value=StrategyParams.stt_on_sell_pct,
        step=0.005,
        format="%.4f",
    )

    # ── 📈 Capital ──────────────────────────────────────────────────────
    st.markdown("**📈 Capital**")
    initial_capital = st.number_input(
        "Initial Capital (INR)",
        min_value=10_000.0,
        max_value=100_000_000.0,
        value=StrategyParams.initial_capital,
        step=100_000.0,
        format="%.0f",
    )
    lot_size = st.number_input(
        "Lot Size",
        min_value=1,
        max_value=5000,
        value=StrategyParams.lot_size,
        step=25,
    )

    st.markdown("---")
    run_backtest_btn = st.button("Run Backtest", type="primary", use_container_width=True)
    run_robustness_btn = st.button("Run Robustness Grid", use_container_width=True)


# ───────────────────────────────────────────────────────────────────────────── #
# 4. BUILD PARAMS FROM SIDEBAR
# ───────────────────────────────────────────────────────────────────────────── #

def _build_params() -> StrategyParams:
    """Assemble a StrategyParams instance from sidebar inputs."""
    return StrategyParams(
        strategy_type="Short Straddle",
        compression_filter_enabled=compression_filter_enabled,
        compression_threshold=compression_threshold,
        short_stop_loss_pct=short_stop_loss_pct,
        short_profit_target_pct=short_profit_target_pct,
        adaptive_breakout_enabled=adaptive_breakout_enabled,
        adaptive_k=adaptive_k,
        adaptive_lookback_days=adaptive_lookback_days,
        adaptive_min_buffer_pct=adaptive_min_buffer_pct,
        adaptive_max_buffer_pct=adaptive_max_buffer_pct,
        breakout_buffer_pct=breakout_buffer_pct,
        exit_time=exit_time,
        brokerage_per_leg=brokerage_per_leg,
        slippage_pct=slippage_pct,
        stt_on_sell_pct=stt_on_sell_pct,
        initial_capital=initial_capital,
        lot_size=lot_size,
    )


# ───────────────────────────────────────────────────────────────────────────── #
# 5. DATA LOADING (cached)
# ───────────────────────────────────────────────────────────────────────────── #

@st.cache_data(show_spinner="Loading NIFTY options data …")
def _load_panel() -> pd.DataFrame:
    ce_df, pe_df = load_data()
    return build_intraday_panel(ce_df, pe_df)


# ───────────────────────────────────────────────────────────────────────────── #
# 6. MAIN OUTPUT – BACKTEST
# ───────────────────────────────────────────────────────────────────────────── #

if run_backtest_btn:
    params = _build_params()
    panel = _load_panel()

    with st.spinner("Running backtest …"):
        trades = run_backtest(panel, params)
        trade_df = trades_to_dataframe(trades)

    if trade_df.empty:
        st.warning("No trades generated with these parameters.")
        st.stop()

    metrics = compute_metrics(trade_df, params)
    equity_df = build_equity_curve(trade_df, params.initial_capital)
    daily_rets = daily_return_series(trade_df, params.initial_capital)

    # ================================================================== #
    # SECTION 1 — Summary Metrics  (st.metric, 5-col × 2 rows)
    # ================================================================== #
    st.subheader("📋 Short Straddle – Summary Metrics")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Trades", f"{metrics['total_trades']}")
    m2.metric("Win Rate", f"{metrics['win_rate']:.1%}")
    m3.metric("Total Return", f"{metrics['total_return']:.2%}")
    m4.metric("CAGR", f"{metrics['cagr']:.2%}")
    m5.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")

    m6, m7, m8, m9, m10 = st.columns(5)
    m6.metric("Max Drawdown", f"{metrics['max_drawdown']:.2%}")
    m7.metric("Profit Factor", f"{metrics['profit_factor']:.2f}")
    m8.metric("Avg Win (INR)", f"₹{metrics['avg_win']:,.0f}")
    m9.metric("Avg Loss (INR)", f"₹{metrics['avg_loss']:,.0f}")
    m10.metric("Avg Hold (min)", f"{metrics['avg_hold_minutes']:.0f}")

    # ================================================================== #
    # SECTION 2 — Exit Reason Breakdown  (expander, 4-col metrics)
    # ================================================================== #
    exit_counts = metrics.get("exit_reason_breakdown", {})
    with st.expander("Exit Reason Breakdown", expanded=True):
        if exit_counts:
            total_t = metrics["total_trades"]
            # Normalise exit reason names for display
            label_map = {
                "profit_target": "Profit Target",
                "premium_stop": "Premium Stop",
                "breakout_stop": "Breakout Stop",
                "time_stop": "Time Exit",
            }
            ordered_keys = ["profit_target", "premium_stop", "breakout_stop", "time_stop"]
            cols = st.columns(len(ordered_keys))
            for col, key in zip(cols, ordered_keys):
                cnt = exit_counts.get(key, 0)
                pct = cnt / total_t * 100 if total_t else 0
                display_label = label_map.get(key, key)
                col.metric(display_label, f"{cnt} ({pct:.1f}%)")
        else:
            st.info("No exit reason data available.")

    # ================================================================== #
    # SECTION 3 — Risk Diagnostics  (expander, metrics + MAE histogram)
    # ================================================================== #
    with st.expander("📋 Short Straddle Risk Diagnostics", expanded=True):
        if "max_adverse_excursion" in trade_df.columns:
            avg_mae = trade_df["max_adverse_excursion"].mean()
            max_mae = trade_df["max_adverse_excursion"].max()
            tail_threshold = 2 * avg_mae if avg_mae > 0 else 0
            tail_events = int((trade_df["max_adverse_excursion"] > tail_threshold).sum())

            r1, r2, r3 = st.columns(3)
            r1.metric("Avg MAE (INR)", f"₹{avg_mae:,.0f}")
            r2.metric("Max MAE (INR)", f"₹{max_mae:,.0f}")
            r3.metric("Tail Loss Events", f"{tail_events}")

            st.metric("Intraday Drawdown (trade-level)", f"{metrics['max_drawdown']:.2%}")

            st.markdown("**Distribution of Max Adverse Excursion per Trade**")
            fig_mae = px.histogram(
                trade_df,
                x="max_adverse_excursion",
                nbins=40,
                labels={"max_adverse_excursion": "Max Adverse Excursion (INR)", "count": "count"},
                color_discrete_sequence=["#6366f1"],
            )
            fig_mae.update_layout(
                template="plotly_dark",
                height=280,
                margin=dict(t=5, b=40, l=50, r=20),
                showlegend=False,
            )
            st.plotly_chart(fig_mae, use_container_width=True)
        else:
            st.info("MAE data not available in trade log.")

    # ================================================================== #
    # SECTION 4 — Adaptive Breakout Buffer Diagnostics  (expander)
    # ================================================================== #
    with st.expander("📊 Adaptive Breakout Buffer Diagnostics", expanded=True):
        if adaptive_breakout_enabled and "adaptive_breakout_buffer" in trade_df.columns:
            buf_series = trade_df["adaptive_breakout_buffer"]
            buf_nonzero = buf_series[buf_series > 0]

            if not buf_nonzero.empty:
                avg_buf = buf_nonzero.mean()
                min_buf = buf_nonzero.min()
                max_buf = buf_nonzero.max()

                b1, b2, b3 = st.columns(3)
                b1.metric("Avg Buffer Used", f"{avg_buf * 100:.3f}%")
                b2.metric("Min Buffer Used", f"{min_buf * 100:.3f}%")
                b3.metric("Max Buffer Used", f"{max_buf * 100:.3f}%")

                # Histogram – full width
                st.markdown("**Distribution of Adaptive Breakout Buffer per Trade**")
                fig_buf_hist = px.histogram(
                    buf_nonzero * 100,
                    nbins=30,
                    labels={"value": "Adaptive Buffer (%)", "count": "count"},
                    color_discrete_sequence=["#6366f1"],
                )
                fig_buf_hist.update_layout(
                    template="plotly_dark",
                    height=280,
                    margin=dict(t=5, b=40, l=50, r=20),
                    showlegend=False,
                    xaxis_title="Adaptive Buffer (%)",
                    yaxis_title="count",
                )
                st.plotly_chart(fig_buf_hist, use_container_width=True)

                # Time-series – full width with min/max dashed lines
                st.markdown("**Adaptive Buffer Over Time**")
                fig_buf_ts = go.Figure()
                fig_buf_ts.add_trace(go.Scatter(
                    x=trade_df["date"],
                    y=buf_series * 100,
                    mode="lines",
                    line=dict(color="#f59e0b", width=1.5),
                    name="Buffer %",
                ))
                # Max buffer dashed line
                fig_buf_ts.add_hline(
                    y=adaptive_max_buffer_pct * 100,
                    line_dash="dash",
                    line_color="#ef4444",
                    annotation_text="Max",
                    annotation_position="right",
                    annotation_font_color="#ef4444",
                )
                # Min buffer dashed line
                fig_buf_ts.add_hline(
                    y=adaptive_min_buffer_pct * 100,
                    line_dash="dash",
                    line_color="#22c55e",
                    annotation_text="Min",
                    annotation_position="right",
                    annotation_font_color="#22c55e",
                )
                fig_buf_ts.update_layout(
                    template="plotly_dark",
                    height=300,
                    margin=dict(t=5, b=40, l=50, r=40),
                    xaxis_title="Date",
                    yaxis_title="Buffer (%)",
                    showlegend=False,
                )
                st.plotly_chart(fig_buf_ts, use_container_width=True)
            else:
                st.info("Adaptive buffer was enabled but no buffer values recorded.")
        else:
            st.caption("Adaptive breakout buffer is disabled or data not available.")

    # ================================================================== #
    # SECTION 4b — Full Metrics Table  (expander)
    # ================================================================== #
    with st.expander("Full Metrics Table", expanded=False):
        metrics_display = {k: v for k, v in metrics.items() if k != "exit_reason_breakdown"}
        metrics_table = pd.DataFrame(
            list(metrics_display.items()), columns=["Metric", "Value"]
        )
        st.dataframe(metrics_table, use_container_width=True, hide_index=True)

    # ================================================================== #
    # SECTION 5 — Equity Curve  (with dashed baseline)
    # ================================================================== #
    st.subheader("📈 Equity Curve")
    if not equity_df.empty:
        fig_eq = go.Figure()
        fig_eq.add_trace(
            go.Scatter(
                x=equity_df["date"],
                y=equity_df["equity"],
                mode="lines",
                name="Equity",
                line=dict(color="#00b4d8", width=1.5),
            )
        )
        # Dashed baseline at initial capital
        fig_eq.add_hline(
            y=params.initial_capital,
            line_dash="dash",
            line_color="rgba(180,180,180,0.4)",
        )
        fig_eq.update_layout(
            xaxis_title="Date",
            yaxis_title="Equity (INR)",
            template="plotly_dark",
            height=380,
            margin=dict(t=10, b=40, l=60, r=20),
            showlegend=False,
        )
        st.plotly_chart(fig_eq, use_container_width=True)

    # ================================================================== #
    # SECTION 6 — Daily PnL
    # ================================================================== #
    st.subheader("📊 Daily PnL")
    if not equity_df.empty:
        colors = [
            "#2ecc71" if v >= 0 else "#e8a0a0" for v in equity_df["daily_pnl"]
        ]
        fig_pnl = go.Figure(
            go.Bar(
                x=equity_df["date"],
                y=equity_df["daily_pnl"],
                marker_color=colors,
            )
        )
        fig_pnl.update_layout(
            xaxis_title="Date",
            yaxis_title="Daily PnL (INR)",
            template="plotly_dark",
            height=320,
            margin=dict(t=10, b=40, l=60, r=20),
        )
        st.plotly_chart(fig_pnl, use_container_width=True)

    # ================================================================== #
    # SECTION 7 — Daily Return Distribution
    # ================================================================== #
    st.subheader("📉 Daily Return Distribution")
    if not daily_rets.empty:
        fig_hist = px.histogram(
            daily_rets * 100,
            nbins=50,
            labels={"value": "Daily Return (%)", "count": "count"},
            color_discrete_sequence=["#6366f1"],
        )
        fig_hist.update_layout(
            template="plotly_dark",
            height=340,
            margin=dict(t=10, b=40, l=60, r=20),
            showlegend=False,
            xaxis_title="Daily Return (%)",
            yaxis_title="count",
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    # ================================================================== #
    # SECTION 8 — Trade Log  (full dataframe, all columns)
    # ================================================================== #
    st.subheader("📒 Trade Log")
    st.dataframe(
        trade_df.reset_index(drop=True),
        use_container_width=True,
        height=480,
    )


# ───────────────────────────────────────────────────────────────────────────── #
# 7. ROBUSTNESS GRID
# ───────────────────────────────────────────────────────────────────────────── #

if run_robustness_btn:
    panel = _load_panel()

    with st.spinner("Running robustness parameter grid – this may take a few minutes …"):
        results_df, top_sharpe, top_pf = parameter_sensitivity_grid(panel)

    st.subheader("🧪 Robustness – Parameter Sensitivity Grid")

    st.markdown("**Top 10 by Sharpe Ratio**")
    st.dataframe(top_sharpe, use_container_width=True, hide_index=True)

    st.markdown("**Top 10 by Profit Factor**")
    st.dataframe(top_pf, use_container_width=True, hide_index=True)

    with st.expander("Full grid results", expanded=False):
        st.dataframe(results_df, use_container_width=True, hide_index=True)
