"""
Performance Metrics
===================
Computes strategy-level analytics from the trade log.

All monetary values in INR.  Returns are computed on starting capital.
"""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import pandas as pd

from .config import StrategyParams

logger = logging.getLogger(__name__)


def compute_metrics(
    trade_df: pd.DataFrame,
    params: StrategyParams | None = None,
) -> Dict[str, float | str]:
    """
    Compute a full suite of performance metrics.

    Parameters
    ----------
    trade_df : pd.DataFrame
        Output of ``backtester.trades_to_dataframe``.
    params : StrategyParams, optional
        Used for ``initial_capital``.

    Returns
    -------
    dict  – metric name -> value
    """
    if params is None:
        params = StrategyParams()

    capital = params.initial_capital
    n_trades = len(trade_df)

    if n_trades == 0:
        return {"error": "No trades to evaluate."}

    # ----- equity curve -----
    trade_df = trade_df.sort_values("date").copy()
    trade_df["cum_pnl"] = trade_df["net_pnl"].cumsum()
    trade_df["equity"] = capital + trade_df["cum_pnl"]

    total_pnl = trade_df["net_pnl"].sum()
    total_return = total_pnl / capital

    # ----- CAGR -----
    first_date = trade_df["date"].min()
    last_date = trade_df["date"].max()
    days_elapsed = (last_date - first_date).days
    years = max(days_elapsed / 365.25, 1 / 365.25)
    ending_equity = capital + total_pnl
    cagr = (ending_equity / capital) ** (1 / years) - 1 if ending_equity > 0 else -1.0

    # ----- drawdown -----
    equity_series = trade_df["equity"]
    running_max = equity_series.cummax()
    drawdown = (equity_series - running_max) / running_max
    max_drawdown = drawdown.min()  # most negative

    # ----- win / loss stats -----
    wins = trade_df[trade_df["net_pnl"] > 0]
    losses = trade_df[trade_df["net_pnl"] <= 0]
    n_wins = len(wins)
    n_losses = len(losses)
    win_rate = n_wins / n_trades if n_trades else 0.0

    avg_win = wins["net_pnl"].mean() if n_wins else 0.0
    avg_loss = losses["net_pnl"].mean() if n_losses else 0.0
    max_win = trade_df["net_pnl"].max()
    max_loss = trade_df["net_pnl"].min()

    # ----- profit factor -----
    gross_profits = wins["net_pnl"].sum() if n_wins else 0.0
    gross_losses = abs(losses["net_pnl"].sum()) if n_losses else 0.0
    profit_factor = gross_profits / gross_losses if gross_losses > 0 else float("inf")

    # ----- Sharpe ratio (daily returns, annualised) -----
    daily_returns = trade_df.groupby("date")["net_pnl"].sum() / capital
    sharpe = (
        daily_returns.mean() / daily_returns.std() * np.sqrt(252)
        if daily_returns.std() > 0
        else 0.0
    )

    # ----- exit reason breakdown -----
    exit_counts = trade_df["exit_reason"].value_counts().to_dict()

    # ----- average holding time -----
    # rough: difference in time strings -> minutes
    def _time_to_min(t: str) -> float:
        parts = t.split(":")
        return int(parts[0]) * 60 + int(parts[1]) + int(parts[2]) / 60

    trade_df["hold_min"] = trade_df.apply(
        lambda r: _time_to_min(str(r["exit_time"])) - _time_to_min(str(r["entry_time"])), axis=1
    )
    avg_hold_min = trade_df["hold_min"].mean()

    # ----- average transaction cost -----
    avg_txn = trade_df["txn_cost"].mean()

    metrics = {
        "total_trades": n_trades,
        "winning_trades": n_wins,
        "losing_trades": n_losses,
        "win_rate": round(win_rate, 4),
        "total_pnl": round(total_pnl, 2),
        "total_return": round(total_return, 4),
        "cagr": round(cagr, 4),
        "max_drawdown": round(max_drawdown, 4),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "max_win": round(max_win, 2),
        "max_loss": round(max_loss, 2),
        "profit_factor": round(profit_factor, 4),
        "sharpe_ratio": round(sharpe, 4),
        "avg_hold_minutes": round(avg_hold_min, 1),
        "avg_txn_cost": round(avg_txn, 2),
        "initial_capital": capital,
        "ending_equity": round(ending_equity, 2),
        "exit_reason_breakdown": exit_counts,
        "days_in_backtest": days_elapsed,
    }

    logger.info("Metrics computed – Total return: %.2f%%  Sharpe: %.2f", total_return * 100, sharpe)
    return metrics


def build_equity_curve(
    trade_df: pd.DataFrame,
    initial_capital: float = 1_000_000.0,
) -> pd.DataFrame:
    """
    Return a DataFrame with columns ``[date, daily_pnl, cum_pnl, equity]``
    aggregated to daily granularity.
    """
    if trade_df.empty:
        return pd.DataFrame(columns=["date", "daily_pnl", "cum_pnl", "equity"])

    daily = (
        trade_df
        .groupby("date")["net_pnl"]
        .sum()
        .reset_index()
        .rename(columns={"net_pnl": "daily_pnl"})
        .sort_values("date")
    )
    daily["cum_pnl"] = daily["daily_pnl"].cumsum()
    daily["equity"] = initial_capital + daily["cum_pnl"]
    return daily


def daily_return_series(
    trade_df: pd.DataFrame,
    initial_capital: float = 1_000_000.0,
) -> pd.Series:
    """Return a ``pd.Series`` of daily percentage returns."""
    eq = build_equity_curve(trade_df, initial_capital)
    if eq.empty:
        return pd.Series(dtype=float)
    eq["daily_ret"] = eq["daily_pnl"] / initial_capital
    return eq.set_index("date")["daily_ret"]
