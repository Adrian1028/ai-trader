"""
Step 1 — Macro Risk Switch Independent Backtest
=================================================

Tests whether the VIX term structure + HY OAS signal adds value
as a market timing overlay on SPY.

Strategy:
  NORMAL   → 100% SPY
  WARNING  → 50% SPY + 50% cash
  RISK_OFF → 100% cash

Benchmark: 100% buy-and-hold SPY

Usage:
    python -m backtest.macro_signal_backtest
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.signals.macro_data import get_macro_data, fetch_from_yfinance
from src.signals.vix_term_structure import MacroRiskSwitch, Regime

logger = logging.getLogger(__name__)

# Data splits
TRAIN_START, TRAIN_END = "2010-01-01", "2019-12-31"
VALID_START, VALID_END = "2020-01-01", "2021-12-31"
TEST_START, TEST_END = "2022-01-01", "2024-12-31"

SLIPPAGE_BPS = 3  # minimal for SPY


@dataclass
class MacroBacktestResult:
    period: str
    strategy_return_pct: float
    benchmark_return_pct: float
    strategy_sharpe: float
    benchmark_sharpe: float
    strategy_max_dd_pct: float
    benchmark_max_dd_pct: float
    dd_reduction_pct: float
    risk_off_days: int
    warning_days: int
    normal_days: int
    total_trading_days: int
    risk_off_pct: float
    n_regime_changes: int
    year_2022_strategy: float | None
    year_2022_benchmark: float | None
    first_risk_off_2022: str | None
    first_risk_on_after_2022: str | None


def _load_spy(start: str, end: str) -> pd.Series:
    """Load SPY daily close prices."""
    spy = fetch_from_yfinance("SPY", start, end)
    if spy.empty:
        raise ValueError("Failed to download SPY data")
    return spy


def _load_macro_df(start: str, end: str) -> pd.DataFrame:
    """Load macro data as aligned DataFrame."""
    data = get_macro_data(start, end)

    # Build DataFrame
    df = pd.DataFrame(data)
    df = df.sort_index()
    df = df.ffill()
    return df


def run_macro_backtest(
    spy: pd.Series,
    macro_df: pd.DataFrame,
    start: str,
    end: str,
    switch: MacroRiskSwitch | None = None,
) -> MacroBacktestResult:
    """Run the macro timing backtest on a specific period."""
    switch = switch or MacroRiskSwitch()

    # Compute signals for entire history up to end (for state machine context)
    full_signals = switch.compute_signals(macro_df.loc[macro_df.index <= pd.Timestamp(end)])

    if full_signals.empty:
        raise ValueError("No signals computed — check macro data")

    # Slice to test period
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    spy_period = spy.loc[(spy.index >= start_ts) & (spy.index <= end_ts)].copy()
    if spy_period.empty:
        raise ValueError(f"No SPY data for {start} to {end}")

    # Align signals to SPY trading days
    signals_aligned = full_signals["signal"].reindex(spy_period.index, method="ffill")
    regimes_aligned = full_signals["regime"].reindex(spy_period.index, method="ffill")

    # Fill NaN signals with 1.0 (NORMAL — fully invested)
    signals_aligned = signals_aligned.fillna(1.0)
    regimes_aligned = regimes_aligned.fillna("NORMAL")

    # Daily returns
    spy_returns = spy_period.pct_change().fillna(0)

    # Strategy returns: signal * spy_return (with slippage on regime changes)
    prev_signal = signals_aligned.shift(1).fillna(1.0)
    signal_changed = (signals_aligned != prev_signal)

    strategy_returns = signals_aligned * spy_returns
    # Apply slippage cost on days when allocation changes
    strategy_returns = strategy_returns - signal_changed * (SLIPPAGE_BPS / 10_000)

    # NAV curves
    strategy_nav = (1 + strategy_returns).cumprod()
    benchmark_nav = (1 + spy_returns).cumprod()

    # Metrics
    strat_total = float(strategy_nav.iloc[-1]) - 1
    bench_total = float(benchmark_nav.iloc[-1]) - 1

    n_years = len(spy_returns) / 252

    strat_sharpe = 0.0
    if strategy_returns.std() > 0:
        strat_sharpe = float((strategy_returns.mean() / strategy_returns.std()) * np.sqrt(252))

    bench_sharpe = 0.0
    if spy_returns.std() > 0:
        bench_sharpe = float((spy_returns.mean() / spy_returns.std()) * np.sqrt(252))

    strat_dd = float(((strategy_nav / strategy_nav.cummax()) - 1).min()) * 100
    bench_dd = float(((benchmark_nav / benchmark_nav.cummax()) - 1).min()) * 100

    dd_reduction = 0.0
    if bench_dd != 0:
        dd_reduction = (1 - abs(strat_dd) / abs(bench_dd)) * 100

    # Regime statistics
    risk_off_days = int((regimes_aligned == "RISK_OFF").sum())
    warning_days = int((regimes_aligned == "WARNING").sum())
    normal_days = int((regimes_aligned == "NORMAL").sum())
    total_days = len(regimes_aligned)
    risk_off_pct = risk_off_days / total_days * 100 if total_days > 0 else 0

    # Regime change count
    regime_changes = int((regimes_aligned != regimes_aligned.shift(1)).sum()) - 1
    regime_changes = max(0, regime_changes)

    # 2022 specific analysis
    y2022_strategy = None
    y2022_benchmark = None
    first_ro_2022 = None
    first_on_2022 = None

    mask_2022 = (spy_returns.index >= "2022-01-01") & (spy_returns.index <= "2022-12-31")
    if mask_2022.any():
        sr_2022 = strategy_returns[mask_2022]
        br_2022 = spy_returns[mask_2022]
        y2022_strategy = float((1 + sr_2022).prod() - 1) * 100
        y2022_benchmark = float((1 + br_2022).prod() - 1) * 100

        # First RISK_OFF in 2022
        ro_2022 = regimes_aligned[mask_2022]
        risk_off_mask = ro_2022 == "RISK_OFF"
        if risk_off_mask.any():
            first_ro_2022 = risk_off_mask.idxmax().strftime("%Y-%m-%d")
            # First NORMAL after that
            after_ro = ro_2022.loc[ro_2022.index > first_ro_2022]
            normal_after = after_ro[after_ro == "NORMAL"]
            if not normal_after.empty:
                first_on_2022 = normal_after.index[0].strftime("%Y-%m-%d")

    return MacroBacktestResult(
        period=f"{start} to {end}",
        strategy_return_pct=strat_total * 100,
        benchmark_return_pct=bench_total * 100,
        strategy_sharpe=strat_sharpe,
        benchmark_sharpe=bench_sharpe,
        strategy_max_dd_pct=abs(strat_dd),
        benchmark_max_dd_pct=abs(bench_dd),
        dd_reduction_pct=dd_reduction,
        risk_off_days=risk_off_days,
        warning_days=warning_days,
        normal_days=normal_days,
        total_trading_days=total_days,
        risk_off_pct=risk_off_pct,
        n_regime_changes=regime_changes,
        year_2022_strategy=y2022_strategy,
        year_2022_benchmark=y2022_benchmark,
        first_risk_off_2022=first_ro_2022,
        first_risk_on_after_2022=first_on_2022,
    )


def print_macro_report(result: MacroBacktestResult):
    """Print formatted macro backtest report."""
    print(f"\n{'=' * 70}")
    print(f"  MACRO SIGNAL BACKTEST — {result.period}")
    print(f"{'=' * 70}")

    print(f"\n  {'Metric':<35} {'Strategy':>12} {'SPY B&H':>12}")
    print(f"  {'-' * 59}")
    print(f"  {'Total Return':<35} {result.strategy_return_pct:>11.2f}% {result.benchmark_return_pct:>11.2f}%")
    print(f"  {'Sharpe Ratio':<35} {result.strategy_sharpe:>12.3f} {result.benchmark_sharpe:>12.3f}")
    print(f"  {'Max Drawdown':<35} {result.strategy_max_dd_pct:>11.2f}% {result.benchmark_max_dd_pct:>11.2f}%")
    print(f"  {'DD Reduction':<35} {result.dd_reduction_pct:>11.1f}%")

    print(f"\n  {'Regime Statistics':<35}")
    print(f"  {'-' * 59}")
    print(f"  {'NORMAL days':<35} {result.normal_days:>12} ({result.normal_days/result.total_trading_days*100:.1f}%)")
    print(f"  {'WARNING days':<35} {result.warning_days:>12} ({result.warning_days/result.total_trading_days*100:.1f}%)")
    print(f"  {'RISK_OFF days':<35} {result.risk_off_days:>12} ({result.risk_off_pct:.1f}%)")
    print(f"  {'Regime changes':<35} {result.n_regime_changes:>12}")
    n_years = result.total_trading_days / 252
    print(f"  {'Changes per year':<35} {result.n_regime_changes / max(n_years, 0.1):>12.1f}")

    if result.year_2022_strategy is not None:
        print(f"\n  {'2022 Analysis':<35}")
        print(f"  {'-' * 59}")
        print(f"  {'2022 Return':<35} {result.year_2022_strategy:>11.2f}% {result.year_2022_benchmark:>11.2f}%")
        if result.first_risk_off_2022:
            print(f"  {'First RISK_OFF':<35} {result.first_risk_off_2022:>12}")
        if result.first_risk_on_after_2022:
            print(f"  {'First recovery':<35} {result.first_risk_on_after_2022:>12}")


def check_gates(test_result: MacroBacktestResult) -> dict[str, bool]:
    """Check Phase 1 Step 1 gate conditions."""
    gates = {
        "Max DD < 60% of SPY DD": (
            test_result.strategy_max_dd_pct < test_result.benchmark_max_dd_pct * 0.60
        ),
        "Sharpe > SPY Sharpe": (
            test_result.strategy_sharpe > test_result.benchmark_sharpe
        ),
        "2022 Return > SPY 2022": (
            test_result.year_2022_strategy is not None
            and test_result.year_2022_benchmark is not None
            and test_result.year_2022_strategy > test_result.year_2022_benchmark
        ),
        "RISK_OFF < 30% of days": (
            test_result.risk_off_pct < 30
        ),
        "Regime changes < 20/year": (
            test_result.n_regime_changes / max(test_result.total_trading_days / 252, 0.1) < 20
        ),
    }
    return gates


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    output_dir = str(PROJECT_ROOT / "results")
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("  STEP 1: Macro Risk Switch Backtest")
    print("=" * 70)

    # Load data
    print("\n  Loading macro data...")
    macro_df = _load_macro_df(TRAIN_START, TEST_END)
    print(f"  Macro data: {len(macro_df)} rows")
    print(f"  Columns: {list(macro_df.columns)}")
    print(f"  Date range: {macro_df.index[0].strftime('%Y-%m-%d')} to {macro_df.index[-1].strftime('%Y-%m-%d')}")

    # Check data availability
    for col in ["vix", "vix3m", "oas"]:
        n_valid = macro_df[col].notna().sum()
        print(f"  {col}: {n_valid} valid observations")

    print("\n  Loading SPY data...")
    spy = _load_spy(TRAIN_START, TEST_END)
    print(f"  SPY: {len(spy)} rows")

    # Optimised from sensitivity analysis on training data:
    # - Lower rv thresholds catch risk earlier → better DD reduction
    # - All 75 param combos beat SPY on training, so wide robustness
    switch = MacroRiskSwitch(
        rv_trigger=0.98,     # backwardation trigger (from sensitivity: 0.95-0.98 range best)
        rv_warning=0.92,     # early warning
        rv_recover=0.90,     # must drop well below warning to recover
        oas_trigger=0.15,
        oas_recover=0.08,
        oas_lookback=21,
        min_hold_days=5,
    )

    # --- Training period (observation only) ---
    print(f"\n{'~' * 50}")
    print(f"  TRAINING PERIOD: {TRAIN_START} to {TRAIN_END}")
    print(f"{'~' * 50}")
    train_result = run_macro_backtest(spy, macro_df, TRAIN_START, TRAIN_END, switch)
    print_macro_report(train_result)

    # --- Validation period (COVID crash) ---
    print(f"\n{'~' * 50}")
    print(f"  VALIDATION PERIOD: {VALID_START} to {VALID_END}")
    print(f"{'~' * 50}")
    valid_result = run_macro_backtest(spy, macro_df, VALID_START, VALID_END, switch)
    print_macro_report(valid_result)

    # --- Test period ---
    print(f"\n{'~' * 50}")
    print(f"  TEST PERIOD: {TEST_START} to {TEST_END}")
    print(f"{'~' * 50}")
    test_result = run_macro_backtest(spy, macro_df, TEST_START, TEST_END, switch)
    print_macro_report(test_result)

    # Gate check
    gates = check_gates(test_result)
    print(f"\n{'=' * 70}")
    print("  STEP 1 GATE CHECK (Test Period)")
    print(f"{'=' * 70}")

    all_pass = True
    for gate, passed in gates.items():
        status = "PASS" if passed else "FAIL"
        symbol = "v" if passed else "x"
        if not passed:
            all_pass = False
        print(f"  {symbol} {status}  {gate}")

    if all_pass:
        print(f"\n  ALL GATES PASSED -> Proceed to Step 2")
    else:
        failed = [g for g, p in gates.items() if not p]
        print(f"\n  GATE CHECK FAILED ({len(failed)} gates)")
        print("  -> Adjust thresholds using TRAINING data only")
        print("  -> Do NOT use test period data for tuning")

    print(f"{'=' * 70}")

    # Save results
    all_results = {
        "train": asdict(train_result),
        "validation": asdict(valid_result),
        "test": asdict(test_result),
        "gates": {g: p for g, p in gates.items()},
        "all_gates_passed": all_pass,
        "parameters": {
            "rv_trigger": switch.rv_trigger,
            "rv_warning": switch.rv_warning,
            "rv_recover": switch.rv_recover,
            "oas_trigger": switch.oas_trigger,
            "oas_recover": switch.oas_recover,
            "oas_lookback": switch.oas_lookback,
        },
    }
    filepath = os.path.join(output_dir, "macro_signal_backtest.json")
    with open(filepath, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved to {filepath}")

    return all_results


if __name__ == "__main__":
    main()
