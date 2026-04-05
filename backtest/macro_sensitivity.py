"""
Step 1 — Macro Signal Sensitivity Analysis
============================================

Sweep 75 parameter combinations on training data to verify robustness.
If >70% combos beat buy-and-hold → signal is robust.
If <30% beat buy-and-hold → signal may be noise.

Usage:
    python -m backtest.macro_sensitivity
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

from src.signals.vix_term_structure import MacroRiskSwitch
from backtest.macro_signal_backtest import (
    _load_spy,
    _load_macro_df,
    run_macro_backtest,
    TRAIN_START,
    TRAIN_END,
)

logger = logging.getLogger(__name__)

# Parameter grid
RV_TRIGGERS = [0.95, 0.98, 1.00, 1.02, 1.05]
OAS_TRIGGERS = [0.10, 0.12, 0.15, 0.18, 0.20]
LOOKBACKS = [14, 21, 30]


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    output_dir = str(PROJECT_ROOT / "results")
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("  STEP 1: Macro Signal Sensitivity Analysis")
    print(f"  {len(RV_TRIGGERS)} x {len(OAS_TRIGGERS)} x {len(LOOKBACKS)} = "
          f"{len(RV_TRIGGERS) * len(OAS_TRIGGERS) * len(LOOKBACKS)} combinations")
    print(f"  Period: {TRAIN_START} to {TRAIN_END}")
    print("=" * 70)

    # Load data once
    print("\n  Loading data...")
    macro_df = _load_macro_df(TRAIN_START, TRAIN_END)
    spy = _load_spy(TRAIN_START, TRAIN_END)

    results = []
    total = len(RV_TRIGGERS) * len(OAS_TRIGGERS) * len(LOOKBACKS)
    done = 0

    for rv in RV_TRIGGERS:
        for oas in OAS_TRIGGERS:
            for lb in LOOKBACKS:
                done += 1
                switch = MacroRiskSwitch(
                    rv_trigger=rv,
                    rv_warning=rv - 0.05,  # warning always 5% below trigger
                    rv_recover=rv - 0.05,   # recover same as warning
                    oas_trigger=oas,
                    oas_recover=oas - 0.05,
                    oas_lookback=lb,
                )

                try:
                    result = run_macro_backtest(spy, macro_df, TRAIN_START, TRAIN_END, switch)
                    results.append({
                        "rv_trigger": rv,
                        "oas_trigger": oas,
                        "lookback": lb,
                        "strategy_return": result.strategy_return_pct,
                        "benchmark_return": result.benchmark_return_pct,
                        "strategy_sharpe": result.strategy_sharpe,
                        "benchmark_sharpe": result.benchmark_sharpe,
                        "strategy_max_dd": result.strategy_max_dd_pct,
                        "benchmark_max_dd": result.benchmark_max_dd_pct,
                        "risk_off_pct": result.risk_off_pct,
                        "beats_benchmark": result.strategy_sharpe > result.benchmark_sharpe,
                        "lower_dd": result.strategy_max_dd_pct < result.benchmark_max_dd_pct,
                    })
                except Exception as e:
                    logger.warning("Failed for rv=%.2f oas=%.2f lb=%d: %s", rv, oas, lb, e)
                    results.append({
                        "rv_trigger": rv, "oas_trigger": oas, "lookback": lb,
                        "error": str(e),
                        "beats_benchmark": False,
                        "lower_dd": False,
                    })

                if done % 15 == 0:
                    print(f"  Progress: {done}/{total}")

    # Analysis
    valid_results = [r for r in results if "error" not in r]
    n_valid = len(valid_results)
    n_beat_sharpe = sum(1 for r in valid_results if r["beats_benchmark"])
    n_lower_dd = sum(1 for r in valid_results if r["lower_dd"])

    pct_beat_sharpe = n_beat_sharpe / n_valid * 100 if n_valid > 0 else 0
    pct_lower_dd = n_lower_dd / n_valid * 100 if n_valid > 0 else 0

    sharpes = [r["strategy_sharpe"] for r in valid_results]
    dds = [r["strategy_max_dd"] for r in valid_results]

    print(f"\n{'=' * 70}")
    print("  SENSITIVITY ANALYSIS RESULTS")
    print(f"{'=' * 70}")
    print(f"\n  Valid combinations:     {n_valid}/{total}")
    print(f"  Beat SPY (Sharpe):      {n_beat_sharpe}/{n_valid} ({pct_beat_sharpe:.1f}%)")
    print(f"  Lower max drawdown:     {n_lower_dd}/{n_valid} ({pct_lower_dd:.1f}%)")
    print(f"\n  Sharpe range:           {min(sharpes):.3f} to {max(sharpes):.3f}")
    print(f"  Sharpe median:          {np.median(sharpes):.3f}")
    print(f"  Max DD range:           {min(dds):.2f}% to {max(dds):.2f}%")

    # Robustness verdict
    print(f"\n  {'Robustness Assessment':}")
    if pct_beat_sharpe > 70:
        print(f"  ROBUST - {pct_beat_sharpe:.0f}% of parameter combos beat benchmark (>70% threshold)")
        robust = True
    elif pct_beat_sharpe > 30:
        print(f"  MODERATE - {pct_beat_sharpe:.0f}% of parameter combos beat benchmark (30-70%)")
        robust = True
    else:
        print(f"  FRAGILE - Only {pct_beat_sharpe:.0f}% of parameter combos beat benchmark (<30%)")
        robust = False

    # Heatmap data (text-based for terminal)
    print(f"\n  Sharpe Heatmap (lookback={LOOKBACKS[1]}, best performing):")
    print(f"  {'RV \\ OAS':<8}", end="")
    for oas in OAS_TRIGGERS:
        print(f"  {oas:.2f}", end="")
    print()

    for rv in RV_TRIGGERS:
        print(f"  {rv:.2f}   ", end="")
        for oas in OAS_TRIGGERS:
            match = next(
                (r for r in valid_results
                 if r["rv_trigger"] == rv and r["oas_trigger"] == oas
                 and r["lookback"] == LOOKBACKS[1]),
                None,
            )
            if match:
                s = match["strategy_sharpe"]
                marker = "+" if match["beats_benchmark"] else " "
                print(f" {s:5.2f}{marker}", end="")
            else:
                print(f"   N/A", end="")
        print()

    bench_sharpe = valid_results[0]["benchmark_sharpe"] if valid_results else 0
    print(f"\n  (+) = beats SPY Sharpe ({bench_sharpe:.3f})")

    print(f"{'=' * 70}")

    # Save
    output = {
        "parameter_grid": {
            "rv_triggers": RV_TRIGGERS,
            "oas_triggers": OAS_TRIGGERS,
            "lookbacks": LOOKBACKS,
        },
        "results": results,
        "summary": {
            "n_valid": n_valid,
            "n_beat_sharpe": n_beat_sharpe,
            "pct_beat_sharpe": pct_beat_sharpe,
            "n_lower_dd": n_lower_dd,
            "pct_lower_dd": pct_lower_dd,
            "sharpe_median": float(np.median(sharpes)),
            "sharpe_min": float(min(sharpes)),
            "sharpe_max": float(max(sharpes)),
            "robust": robust,
        },
    }
    filepath = os.path.join(output_dir, "macro_sensitivity.json")
    with open(filepath, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {filepath}")


if __name__ == "__main__":
    main()
