"""
Macro Data Fetcher — Yahoo Finance only (no API key needed).

Fetches via yfinance:
  - VIX   (^VIX)   — CBOE Volatility Index
  - VIX3M (^VIX3M) — CBOE 3-month VIX
  - OAS proxy       — Synthetic credit spread from HYG/TLT ratio
  - T10Y2Y proxy    — From ^TNX (10Y yield)

OAS Proxy Method:
  The HYG/TLT ratio is a well-known proxy for credit spreads.
  When HYG drops relative to TLT, high-yield spreads are widening.
  We invert and normalise to get a spread-like series.

Local SQLite cache for incremental updates.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DB_PATH = PROJECT_ROOT / "data" / "fred_cache.db"


def _init_db(db_path: Path | None = None) -> sqlite3.Connection:
    """Initialise SQLite cache database."""
    db_path = db_path or DB_PATH
    os.makedirs(db_path.parent, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS macro_data (
            series TEXT NOT NULL,
            date TEXT NOT NULL,
            value REAL,
            fetched_at TEXT NOT NULL,
            PRIMARY KEY (series, date)
        )
    """)
    conn.commit()
    return conn


def _store_series(conn: sqlite3.Connection, series: str, data: pd.Series):
    """Store a pandas Series into SQLite cache."""
    now = datetime.utcnow().isoformat()
    rows = []
    for date, value in data.items():
        if pd.notna(value):
            date_str = date.strftime("%Y-%m-%d") if hasattr(date, "strftime") else str(date)
            rows.append((series, date_str, float(value), now))
    if rows:
        conn.executemany(
            "INSERT OR REPLACE INTO macro_data (series, date, value, fetched_at) VALUES (?, ?, ?, ?)",
            rows,
        )
        conn.commit()
        logger.info("Cached %d rows for %s", len(rows), series)


def _load_cached(conn: sqlite3.Connection, series: str,
                 start: str, end: str) -> pd.Series:
    """Load cached data as pandas Series."""
    rows = conn.execute(
        "SELECT date, value FROM macro_data WHERE series = ? AND date >= ? AND date <= ? ORDER BY date",
        (series, start, end),
    ).fetchall()
    if not rows:
        return pd.Series(dtype=float)
    dates = [pd.Timestamp(r[0]) for r in rows]
    values = [r[1] for r in rows]
    return pd.Series(values, index=pd.DatetimeIndex(dates), name=series)


def _get_last_cached_date(conn: sqlite3.Connection, series: str) -> str | None:
    row = conn.execute(
        "SELECT MAX(date) FROM macro_data WHERE series = ?", (series,)
    ).fetchone()
    return row[0] if row and row[0] else None


def fetch_from_yfinance(ticker: str, start: str, end: str) -> pd.Series:
    """Fetch closing prices from Yahoo Finance."""
    try:
        import yfinance as yf
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if df.empty:
            return pd.Series(dtype=float)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df["Close"]
    except Exception as e:
        logger.error("yfinance fetch failed for %s: %s", ticker, e)
        return pd.Series(dtype=float)


def _compute_synthetic_oas(hyg: pd.Series, tlt: pd.Series) -> pd.Series:
    """
    Compute synthetic OAS from HYG/TLT ratio.

    HYG = iShares High Yield Corporate Bond ETF
    TLT = iShares 20+ Year Treasury Bond ETF

    When credit spreads widen: HYG drops, TLT rises → ratio drops
    We invert: synthetic_oas = -log(HYG/TLT) × 100, then shift to positive range

    This gives a spread-like series that increases when stress increases.
    """
    # Align
    aligned = pd.DataFrame({"hyg": hyg, "tlt": tlt}).dropna()
    if aligned.empty:
        return pd.Series(dtype=float)

    ratio = aligned["hyg"] / aligned["tlt"]
    # Invert and scale: higher = more stress
    # Use z-score relative to expanding window to get a "spread level"
    log_ratio = np.log(ratio)
    # Normalise: mean of ~0, so we shift to have typical values around 3-5
    # (mimicking OAS levels in percentage points)
    expanding_mean = log_ratio.expanding(min_periods=60).mean()
    expanding_std = log_ratio.expanding(min_periods=60).std()

    # Synthetic OAS: deviation from historical mean, scaled
    # Negative deviation (HYG underperforming) = high spread
    synthetic = -(log_ratio - expanding_mean) / expanding_std.replace(0, np.nan)
    # Scale to typical OAS range (3-8 for normal, 8+ for stress)
    synthetic = synthetic * 2 + 4  # centre around 4, 1 std = 2 points

    return synthetic


def get_macro_data(
    start: str = "2010-01-01",
    end: str | None = None,
    db_path: Path | None = None,
    force_refresh: bool = False,
) -> dict[str, pd.Series]:
    """
    Get all macro data. Uses yfinance (no API key needed).

    Returns dict with keys: 'vix', 'vix3m', 'oas'
    """
    end = end or datetime.now().strftime("%Y-%m-%d")
    conn = _init_db(db_path)
    result = {}

    # Buffer start for expanding window calculations
    buffer_start = (pd.Timestamp(start) - pd.DateOffset(years=1)).strftime("%Y-%m-%d")

    yf_series = {
        "vix": "^VIX",
        "vix3m": "^VIX3M",
    }

    for name, ticker in yf_series.items():
        cached = _load_cached(conn, name, start, end)
        last_date = _get_last_cached_date(conn, name)

        fetch_start = buffer_start
        if last_date and not force_refresh:
            next_day = (pd.Timestamp(last_date) + timedelta(days=1)).strftime("%Y-%m-%d")
            if next_day > fetch_start:
                fetch_start = next_day

        if fetch_start <= end:
            fresh = fetch_from_yfinance(ticker, fetch_start, end)
            if not fresh.empty:
                _store_series(conn, name, fresh)
                cached = _load_cached(conn, name, start, end)

        result[name] = cached

    # Synthetic OAS from HYG/TLT
    oas_cached = _load_cached(conn, "oas_synthetic", start, end)
    if oas_cached.empty or force_refresh:
        hyg = fetch_from_yfinance("HYG", buffer_start, end)
        tlt = fetch_from_yfinance("TLT", buffer_start, end)

        if not hyg.empty and not tlt.empty:
            synthetic_oas = _compute_synthetic_oas(hyg, tlt)
            if not synthetic_oas.empty:
                _store_series(conn, "oas_synthetic", synthetic_oas)
                oas_cached = _load_cached(conn, "oas_synthetic", start, end)

    result["oas"] = oas_cached

    conn.close()

    # Log data availability
    for name, series in result.items():
        if len(series) > 0:
            logger.info("%s: %d points (%s to %s)", name, len(series),
                        series.index[0].strftime("%Y-%m-%d"),
                        series.index[-1].strftime("%Y-%m-%d"))
        else:
            logger.warning("%s: NO DATA", name)

    return result


def get_macro_dataframe(
    start: str = "2010-01-01",
    end: str | None = None,
    db_path: Path | None = None,
) -> pd.DataFrame:
    """Get all macro data as a single aligned DataFrame (forward-filled)."""
    data = get_macro_data(start, end, db_path)
    df = pd.DataFrame(data)
    df = df.sort_index()
    df = df.ffill()
    return df
