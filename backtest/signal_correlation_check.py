"""
Step 2 — Signal Correlation Check
===================================

Checks whether Signal 1 (macro switch) and Signal 2 (liquidity filter)
are redundant. If overlap > 80%, only keep the better one.

Usage:
    python -m backtest.signal_correlation_check
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.signals.macro_data import get_macro_data, fetch_from_yfinance
from src.signals.vix_term_structure import MacroRiskSwitch
from src.signals.liquidity_filter import compute_crowding_risk, filter_stocks
from src.universe.stock_pool import TICKERS

logger = logging.getLogger(__name__)

TEST_START = "2022-01-01"
TEST_END = "2024-12-31"


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    output_dir = str(PROJECT_ROOT / "results")
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("  SIGNAL CORRELATION CHECK")
    print("=" * 70)

    # Load data
    print("  Loading macro data...")
    macro_data = get_macro_data("2014-01-01", TEST_END)
    macro_df = pd.DataFrame(macro_data).sort_index().ffill()

    switch = MacroRiskSwitch(
        rv_trigger=0.98, rv_warning=0.92, rv_recover=0.90,
        oas_trigger=0.15, oas_recover=0.08, oas_lookback=21, min_hold_days=5,
    )
    macro_signals = switch.compute_signals(macro_df)

    print("  Loading stock data...")
    import yfinance as yf
    all_data = {}
    buffer_start = (pd.Timestamp(TEST_START) - pd.DateOffset(months=6)).strftime("%Y-%m-%d")
    for ticker in TICKERS:
        df = yf.download(ticker, start=buffer_start, end=TEST_END, progress=False, auto_adjust=True)
        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            all_data[ticker] = df

    spy = fetch_from_yfinance("SPY", TEST_START, TEST_END)
    trading_days = spy.loc[(spy.index >= TEST_START) & (spy.index <= TEST_END)].index

    # 1. Time-series correlation
    print("\n  Computing time-series correlations...")
    vix_ratio_series = []
    avg_crowding_series = []
    oas_series = []
    avg_amihud_series = []

    for day in trading_days[::5]:  # sample every 5 days
        # VIX ratio
        if day in macro_signals.index:
            vix_ratio_series.append(float(macro_signals.loc[day, "ratio"]))
        else:
            prev = macro_signals.loc[macro_signals.index <= day]
            vix_ratio_series.append(float(prev["ratio"].iloc[-1]) if not prev.empty else np.nan)

        # OAS
        if day in macro_df.index:
            oas_series.append(float(macro_df.loc[day, "oas"]) if pd.notna(macro_df.loc[day, "oas"]) else np.nan)
        else:
            oas_series.append(np.nan)

        # Crowding risk
        scores = compute_crowding_risk(all_data, day)
        valid_scores = [v for v in scores.values() if v != 0.0]
        avg_crowding_series.append(np.mean(valid_scores) if valid_scores else 0)

        # Average Amihud (using crowding as proxy)
        avg_amihud_series.append(np.max(list(scores.values())) if scores else 0)

    corr_df = pd.DataFrame({
        "vix_ratio": vix_ratio_series,
        "avg_crowding": avg_crowding_series,
        "oas": oas_series,
        "max_crowding": avg_amihud_series,
    }).dropna()

    if not corr_df.empty:
        corr_vix_crowd = float(corr_df["vix_ratio"].corr(corr_df["avg_crowding"]))
        corr_oas_crowd = float(corr_df["oas"].corr(corr_df["avg_crowding"]))
    else:
        corr_vix_crowd = 0
        corr_oas_crowd = 0

    print(f"    VIX/VIX3M ratio vs avg Crowding Risk: r = {corr_vix_crowd:.3f}")
    print(f"    OAS delta vs avg Crowding Risk:        r = {corr_oas_crowd:.3f}")

    # 2. Signal overlap
    print("\n  Computing signal overlap...")
    risk_off_days = 0
    high_exclusion_days = 0
    overlap_days = 0

    for day in trading_days:
        # Macro: RISK_OFF or WARNING?
        if day in macro_signals.index:
            regime = macro_signals.loc[day, "regime"]
        else:
            prev = macro_signals.loc[macro_signals.index <= day]
            regime = prev["regime"].iloc[-1] if not prev.empty else "NORMAL"

        is_macro_caution = regime in ("RISK_OFF", "WARNING")

        # Liquidity: > 30% excluded?
        scores = compute_crowding_risk(all_data, day)
        _, excluded = filter_stocks(scores, threshold=1.5)
        high_exclusion = len(excluded) / max(len(scores), 1) > 0.30

        if is_macro_caution:
            risk_off_days += 1
        if high_exclusion:
            high_exclusion_days += 1
        if is_macro_caution and high_exclusion:
            overlap_days += 1

    overlap_pct = overlap_days / max(risk_off_days, 1) * 100

    print(f"    Macro caution days: {risk_off_days}/{len(trading_days)}")
    print(f"    High exclusion days: {high_exclusion_days}/{len(trading_days)}")
    print(f"    Overlap days: {overlap_days}")
    print(f"    Overlap rate: {overlap_pct:.1f}% (of macro caution days)")

    # 3. Verdict
    print(f"\n{'=' * 70}")
    print("  CORRELATION VERDICT")
    print(f"{'=' * 70}")

    is_redundant = overlap_pct > 80
    low_correlation = abs(corr_vix_crowd) < 0.5

    print(f"\n  VIX-Crowding correlation: {corr_vix_crowd:.3f} ({'LOW' if low_correlation else 'HIGH'})")
    print(f"  Signal overlap: {overlap_pct:.1f}% ({'REDUNDANT' if is_redundant else 'INDEPENDENT'})")

    if is_redundant:
        print(f"\n  SIGNALS ARE REDUNDANT (overlap > 80%)")
        print(f"  -> Keep only the better-performing signal")
    else:
        print(f"\n  SIGNALS ARE COMPLEMENTARY (overlap <= 80%)")
        print(f"  -> Both signals add independent value")
        print(f"  -> Proceed to Step 3")

    # Save
    output = {
        "correlations": {
            "vix_ratio_vs_crowding": corr_vix_crowd,
            "oas_vs_crowding": corr_oas_crowd,
        },
        "overlap": {
            "macro_caution_days": risk_off_days,
            "high_exclusion_days": high_exclusion_days,
            "overlap_days": overlap_days,
            "overlap_pct": overlap_pct,
        },
        "is_redundant": is_redundant,
        "recommendation": "KEEP_BOTH" if not is_redundant else "KEEP_BEST",
    }
    filepath = os.path.join(output_dir, "signal_correlation.json")
    with open(filepath, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {filepath}")


if __name__ == "__main__":
    main()
