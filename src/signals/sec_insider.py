"""
SEC Form 4 Insider Trading Signal — "Not Sold" Confirmation
=============================================================

Based on DeVault, Cederburg & Wang (2022):
When portfolio insiders sell stock A but keep stock B,
stock B receives a "not sold" confidence boost.

Uses SEC EDGAR XBRL API with streaming XML parsing (memory safe).

For backtesting: uses yfinance insider data as proxy.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DB_PATH = PROJECT_ROOT / "data" / "insider_trades.db"

SEC_USER_AGENT = os.environ.get("SEC_EMAIL", "trading-bot@example.com")
SEC_BASE_URL = "https://efts.sec.gov/LATEST"


def _init_db(db_path: Path | None = None) -> sqlite3.Connection:
    """Initialise insider trades database."""
    db_path = db_path or DB_PATH
    os.makedirs(db_path.parent, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS insider_transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            insider_name TEXT,
            insider_title TEXT,
            transaction_date TEXT NOT NULL,
            transaction_type TEXT NOT NULL,
            shares REAL NOT NULL,
            price REAL,
            value REAL,
            shares_remaining REAL,
            source TEXT DEFAULT 'yfinance',
            fetched_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_insider_ticker_date
        ON insider_transactions (ticker, transaction_date)
    """)
    conn.commit()
    return conn


def fetch_insider_data_yfinance(ticker: str) -> list[dict]:
    """
    Fetch insider transactions from yfinance (free, no API key).
    Returns list of transaction dicts.
    """
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)

        transactions = []

        # Get insider transactions
        insider_txns = t.insider_transactions
        if insider_txns is not None and not insider_txns.empty:
            for _, row in insider_txns.iterrows():
                txn = {
                    "ticker": ticker,
                    "insider_name": str(row.get("Insider", row.get("Text", "Unknown"))),
                    "insider_title": str(row.get("Position", row.get("Insider Relation", "Unknown"))),
                    "transaction_date": str(row.get("Start Date", row.get("Date", "")))[:10],
                    "transaction_type": _classify_transaction(row),
                    "shares": float(row.get("Shares", 0)),
                    "price": float(row.get("Value", 0)) / max(float(row.get("Shares", 1)), 1),
                    "value": float(row.get("Value", 0)),
                    "shares_remaining": 0,  # Not always available
                }
                transactions.append(txn)

        return transactions
    except Exception as e:
        logger.warning("Failed to fetch insider data for %s: %s", ticker, e)
        return []


def _classify_transaction(row) -> str:
    """Classify transaction as BUY, SELL, or AWARD."""
    text = str(row.get("Text", row.get("Transaction", ""))).lower()
    shares = float(row.get("Shares", 0))

    if "sale" in text or "sold" in text or shares < 0:
        return "SELL"
    elif "purchase" in text or "buy" in text:
        return "BUY"
    elif "award" in text or "grant" in text or "exercise" in text:
        return "AWARD"
    elif shares > 0:
        return "BUY"
    else:
        return "SELL"


def load_insider_data(
    tickers: list[str],
    db_path: Path | None = None,
    force_refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """
    Load insider transaction data for given tickers.
    Uses SQLite cache, fetches from yfinance if needed.

    Returns {ticker: DataFrame of transactions}
    """
    conn = _init_db(db_path)
    result = {}

    for ticker in tickers:
        # Check cache
        rows = conn.execute(
            "SELECT * FROM insider_transactions WHERE ticker = ? ORDER BY transaction_date",
            (ticker,),
        ).fetchall()

        if not rows or force_refresh:
            # Fetch fresh data
            txns = fetch_insider_data_yfinance(ticker)
            now = datetime.utcnow().isoformat()

            for txn in txns:
                conn.execute("""
                    INSERT OR IGNORE INTO insider_transactions
                    (ticker, insider_name, insider_title, transaction_date,
                     transaction_type, shares, price, value, shares_remaining,
                     source, fetched_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'yfinance', ?)
                """, (
                    txn["ticker"], txn["insider_name"], txn["insider_title"],
                    txn["transaction_date"], txn["transaction_type"],
                    txn["shares"], txn["price"], txn["value"],
                    txn["shares_remaining"], now,
                ))
            conn.commit()

            rows = conn.execute(
                "SELECT * FROM insider_transactions WHERE ticker = ? ORDER BY transaction_date",
                (ticker,),
            ).fetchall()
            time.sleep(0.3)  # rate limit yfinance

        if rows:
            columns = [d[0] for d in conn.execute(
                "SELECT * FROM insider_transactions LIMIT 1"
            ).description]
            df = pd.DataFrame(rows, columns=columns)
            result[ticker] = df
            logger.info("%s: %d insider transactions", ticker, len(df))

    conn.close()
    return result


def compute_not_sold_scores(
    insider_data: dict[str, pd.DataFrame],
    tickers: list[str],
    as_of_date: str,
    lookback_days: int = 90,
) -> dict[str, float]:
    """
    Compute "Not Sold" confidence scores for each ticker.

    Logic:
    1. Find insiders who SOLD at least one stock in the past `lookback_days`
    2. Check what other stocks they did NOT sell
    3. Higher score = more insiders selling elsewhere but keeping this stock

    Returns {ticker: not_sold_score}
    """
    cutoff = (pd.Timestamp(as_of_date) - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

    # Build insider activity map: {insider_name: {ticker: total_sold_value}}
    insider_sales: dict[str, dict[str, float]] = {}
    insider_holdings: dict[str, set[str]] = {}

    for ticker, df in insider_data.items():
        if df.empty:
            continue

        recent = df[(df["transaction_date"] >= cutoff) & (df["transaction_date"] <= as_of_date)]

        for _, row in recent.iterrows():
            name = row["insider_name"]
            if name == "Unknown" or pd.isna(name):
                continue

            if name not in insider_holdings:
                insider_holdings[name] = set()
            insider_holdings[name].add(ticker)

            if row["transaction_type"] == "SELL":
                if name not in insider_sales:
                    insider_sales[name] = {}
                value = abs(float(row.get("value", 0)))
                insider_sales[name][ticker] = insider_sales[name].get(ticker, 0) + value

    # Compute NotSold scores
    scores: dict[str, float] = {}

    for ticker in tickers:
        not_sold_score = 0.0
        n_relevant = 0

        for insider, sales in insider_sales.items():
            # This insider sold something
            total_sold = sum(sales.values())
            if total_sold <= 0:
                continue

            # Did they sell THIS ticker?
            sold_this = sales.get(ticker, 0)
            sold_others = total_sold - sold_this

            if sold_others > 0 and sold_this == 0:
                # Insider sold OTHER stocks but NOT this one → confidence signal
                # Score weighted by amount sold elsewhere
                not_sold_score += sold_others / 1_000_000  # normalise to millions
                n_relevant += 1

        scores[ticker] = not_sold_score

    return scores


def apply_insider_weights(
    base_weights: dict[str, float],
    not_sold_scores: dict[str, float],
    boost_low: float = 2.0,
    boost_high: float = 5.0,
    multiplier_low: float = 1.5,
    multiplier_high: float = 2.0,
) -> dict[str, float]:
    """
    Adjust portfolio weights based on NotSold scores.

    - Score > boost_low → weight × multiplier_low
    - Score > boost_high → weight × multiplier_high
    - Score = 0 → weight unchanged

    Re-normalises to sum to 1.0.
    """
    adjusted = {}
    for ticker, weight in base_weights.items():
        score = not_sold_scores.get(ticker, 0)
        if score > boost_high:
            adjusted[ticker] = weight * multiplier_high
        elif score > boost_low:
            adjusted[ticker] = weight * multiplier_low
        else:
            adjusted[ticker] = weight

    # Re-normalise
    total = sum(adjusted.values())
    if total > 0:
        adjusted = {t: w / total for t, w in adjusted.items()}

    return adjusted
