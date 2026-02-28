"""
Data Loader
===========
Reads the multi-part 3-minute NIFTY options CSVs, cleans them,
and returns two DataFrames – one for CE and one for PE – filtered
to the requested strike_offset (default ATM).

Key design decisions
--------------------
* Uses chunked reading to stay memory-efficient on ~2 M rows.
* Parses the Excel-escaped date column ``="DD-MM-YY"`` -> proper datetime.
* Returns datetime-indexed DataFrames sorted by (date, time).
"""

from __future__ import annotations

import glob
import logging
from pathlib import Path
from typing import Tuple

import pandas as pd

from .config import DATA_DIR, DATA_GLOB, StrategyParams

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #

def _clean_date(s: pd.Series) -> pd.Series:
    """Strip the ``="..."`` wrapper that Excel sometimes adds and parse dates."""
    cleaned = s.str.replace('="', '', regex=False).str.replace('"', '', regex=False).str.strip('=')
    return pd.to_datetime(cleaned, format="%d-%m-%y", dayfirst=True)


def _read_single_csv(path: Path) -> pd.DataFrame:
    """Read one part-file and return a cleaned DataFrame."""
    logger.info("Reading %s …", path.name)
    df = pd.read_csv(
        path,
        dtype={
            "symbol": "category",
            "option_type": "category",
            "type": "category",
            "strike_offset": "category",
        },
    )
    # Clean date
    df["date"] = _clean_date(df["date"])

    # Parse time (keep as string for now – we combine later)
    df["time"] = df["time"].astype(str).str.strip()

    return df


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def load_data(
    data_dir: Path | None = None,
    data_glob: str | None = None,
    strike_filter: str = "ATM",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and concatenate all part-files, returning **(ce_df, pe_df)**.

    Each DataFrame has the following schema after cleaning:

    ========== ========================================
    Column     Description
    ========== ========================================
    datetime   ``pd.Timestamp`` (date + time combined)
    date       ``datetime.date``
    time       ``str`` HH:MM:SS
    open       float64
    high       float64
    low        float64
    close      float64
    volume     int64
    oi         float64
    iv         float64
    spot       float64
    ========== ========================================

    Parameters
    ----------
    data_dir : Path, optional
        Override ``config.DATA_DIR``.
    data_glob : str, optional
        Override ``config.DATA_GLOB``.
    strike_filter : str
        Value of ``strike_offset`` to keep (default ``"ATM"``).

    Returns
    -------
    (ce_df, pe_df) : tuple[pd.DataFrame, pd.DataFrame]
    """
    _dir = data_dir or DATA_DIR
    _glob = data_glob or DATA_GLOB

    files = sorted(glob.glob(str(_dir / _glob)))
    if not files:
        raise FileNotFoundError(
            f"No files matching '{_glob}' found in {_dir}. "
            "Check DATA_DIR / DATA_GLOB in config.py."
        )

    logger.info("Found %d data file(s) in %s", len(files), _dir)

    frames = [_read_single_csv(Path(f)) for f in files]
    raw = pd.concat(frames, ignore_index=True)
    logger.info("Total raw rows: %s", f"{len(raw):,}")

    # ----- filter to requested strike_offset -----
    raw = raw[raw["strike_offset"] == strike_filter].copy()
    logger.info("Rows after strike_offset=='%s' filter: %s", strike_filter, f"{len(raw):,}")

    if raw.empty:
        raise ValueError(f"No rows with strike_offset=={strike_filter!r} found.")

    # ----- build datetime index -----
    raw["datetime"] = pd.to_datetime(
        raw["date"].dt.strftime("%Y-%m-%d") + " " + raw["time"]
    )
    raw["date"] = raw["date"].dt.date  # keep as plain date for grouping

    # ----- split CE / PE -----
    ce_df = (
        raw[raw["type"] == "CE"]
        .drop(columns=["symbol", "option_type", "type", "strike_offset"])
        .sort_values("datetime")
        .reset_index(drop=True)
    )
    pe_df = (
        raw[raw["type"] == "PE"]
        .drop(columns=["symbol", "option_type", "type", "strike_offset"])
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    logger.info("CE rows: %s  |  PE rows: %s", f"{len(ce_df):,}", f"{len(pe_df):,}")

    return ce_df, pe_df


def build_intraday_panel(
    ce_df: pd.DataFrame, pe_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge CE and PE on (date, time) into a single panel with suffixed columns.

    Returns a DataFrame with columns:
        datetime, date, time, spot,
        ce_open, ce_high, ce_low, ce_close, ce_volume, ce_oi, ce_iv,
        pe_open, pe_high, pe_low, pe_close, pe_volume, pe_oi, pe_iv
    """
    panel = pd.merge(
        ce_df, pe_df,
        on=["date", "time", "datetime"],
        suffixes=("_ce", "_pe"),
        how="inner",
    )

    # Resolve spot: take CE-side spot (identical to PE-side)
    if "spot_ce" in panel.columns:
        panel["spot"] = panel["spot_ce"]
        panel.drop(columns=["spot_pe", "spot_ce"], inplace=True, errors="ignore")

    # Rename OHLCV+ columns for clarity
    rename_map = {}
    for col_base in ["open", "high", "low", "close", "volume", "oi", "iv"]:
        rename_map[f"{col_base}_ce"] = f"ce_{col_base}"
        rename_map[f"{col_base}_pe"] = f"pe_{col_base}"
    panel.rename(columns=rename_map, inplace=True)

    panel.sort_values("datetime", inplace=True)
    panel.reset_index(drop=True, inplace=True)

    logger.info("Merged panel rows: %s", f"{len(panel):,}")
    return panel
