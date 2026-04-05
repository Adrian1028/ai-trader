"""
Step 2 — Multi-Index Backtest
==============================

Tests the same VIX/VIX3M + OAS macro signal on:
  SPY (S&P 500), QQQ (Nasdaq 100), IWM (Russell 2000), EFA (MSCI EAFE)

All use T+1 Open execution.

Usage:
    python -m backtest.multi_index_backtest
"""

from __future__ import annotations

import json, logging, os, sys
from pathlib import Path

import numpy as np, pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.signals.macro_data import get_macro_data
from src.signals.vix_term_structure import MacroRiskSwitch
from backtest.t1_execution_backtest import run_version, _download_ohlcv, SWITCH_PARAMS

logger = logging.getLogger(__name__)

TRAIN_START = "2010-01-01"
TEST_START = "2022-01-01"
TEST_END = "2024-12-31"
ETFS = ["SPY", "QQQ", "IWM", "EFA"]


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    output_dir = str(PROJECT_ROOT / "results")
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print("  STEP 2: Multi-Index Backtest (T+1 Open execution)")
    print("=" * 80)

    # Load macro signals (same for all)
    print("\n  Loading macro data...")
    macro_data = get_macro_data(TRAIN_START, TEST_END)
    macro_df = pd.DataFrame(macro_data).sort_index().ffill()
    switch = MacroRiskSwitch(**SWITCH_PARAMS)
    macro_signals = switch.compute_signals(macro_df)

    results = {}
    for etf in ETFS:
        print(f"\n  Testing {etf}...")
        ohlcv = _download_ohlcv(etf, TRAIN_START, TEST_END)
        if ohlcv.empty:
            print(f"    SKIPPED: no data")
            continue

        r = run_version(ohlcv, macro_signals, TEST_START, TEST_END, "B")
        results[etf] = r
        print(f"    Sharpe: {r.sharpe:.3f}, Return: {r.total_return_pct:.2f}%, "
              f"DD: {r.max_dd_pct:.2f}%, 2022: {r.year_2022_return_pct:.1f}%")

    # Summary
    print(f"\n{'=' * 80}")
    print(f"  MULTI-INDEX RESULTS (T+1 Open, {TEST_START} to {TEST_END})")
    print(f"{'=' * 80}")
    print(f"\n  {'ETF':<6} {'Sharpe':>8} {'Return':>10} {'Max DD':>9} {'2022':>9} {'SPY Sharpe':>11} {'Signal OK?':>10}")
    print(f"  {'-' * 73}")

    effective = []
    for etf, r in results.items():
        y2022 = f"{r.year_2022_return_pct:.1f}%" if r.year_2022_return_pct else "N/A"
        beats_dd = r.max_dd_pct < r.spy_max_dd_pct * 0.80
        beats_2022 = (r.year_2022_return_pct is not None and r.spy_2022_return_pct is not None
                      and r.year_2022_return_pct > r.spy_2022_return_pct)
        signal_ok = beats_dd and beats_2022
        status = "YES" if signal_ok else "NO"
        if signal_ok:
            effective.append(etf)
        print(f"  {etf:<6} {r.sharpe:>8.3f} {r.total_return_pct:>9.2f}% {r.max_dd_pct:>8.2f}% "
              f"{y2022:>9} {r.spy_sharpe:>11.3f} {status:>10}")

    print(f"\n  Signal effective on: {', '.join(effective) if effective else 'NONE'}")

    # Correlation analysis between effective ETFs
    if len(effective) > 1:
        print(f"\n  Daily return correlations among effective ETFs:")
        etf_returns = {}
        for etf in effective:
            ohlcv = _download_ohlcv(etf, TEST_START, TEST_END)
            ret = ohlcv["Close"].pct_change().dropna()
            etf_returns[etf] = ret
        corr_df = pd.DataFrame(etf_returns).corr()
        print(corr_df.to_string(float_format=lambda x: f"{x:.3f}"))

    # Simple diversified portfolio test
    if len(effective) > 1:
        print(f"\n  Testing equal-weight diversified portfolio of: {effective}")
        # Build a synthetic OHLCV that's the average of effective ETFs
        ohlcv_list = []
        for etf in effective:
            ohlcv_list.append(_download_ohlcv(etf, TRAIN_START, TEST_END))

        # Align and average
        aligned = pd.DataFrame()
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            series_list = []
            for df in ohlcv_list:
                if col in df.columns:
                    series_list.append(df[col])
            if series_list:
                combined = pd.concat(series_list, axis=1)
                aligned[col] = combined.mean(axis=1)
        aligned = aligned.dropna()

        r_div = run_version(aligned, macro_signals, TEST_START, TEST_END, "B")
        print(f"    Diversified Sharpe: {r_div.sharpe:.3f}, Return: {r_div.total_return_pct:.2f}%, "
              f"DD: {r_div.max_dd_pct:.2f}%")

        spy_result = results.get("SPY")
        if spy_result:
            if r_div.sharpe > spy_result.sharpe:
                print(f"    -> Diversification IMPROVES Sharpe ({r_div.sharpe:.3f} > {spy_result.sharpe:.3f})")
            else:
                print(f"    -> Diversification does NOT improve, stay with SPY")

    # Save
    output = {
        "results": {etf: {
            "sharpe": r.sharpe,
            "total_return_pct": r.total_return_pct,
            "max_dd_pct": r.max_dd_pct,
            "year_2022_return": r.year_2022_return_pct,
            "spy_sharpe": r.spy_sharpe,
        } for etf, r in results.items()},
        "effective_etfs": effective,
    }
    filepath = os.path.join(output_dir, "multi_index_backtest.json")
    with open(filepath, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to {filepath}")


if __name__ == "__main__":
    main()
