"""
Step 3 — Cash Alternative Backtest
====================================

When RISK_OFF, compare: A) Cash, B) SHV (short-term treasury), C) SHV with costs.

Usage:
    python -m backtest.cash_alternative_backtest
"""

from __future__ import annotations

import json, logging, os, sys
from pathlib import Path

import numpy as np, pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.signals.macro_data import get_macro_data
from src.signals.vix_term_structure import MacroRiskSwitch
from backtest.t1_execution_backtest import _download_ohlcv, SWITCH_PARAMS

logger = logging.getLogger(__name__)

TEST_START = "2022-01-01"
TEST_END = "2024-12-31"
INITIAL_CAPITAL = 100_000.0
SPY_SLIPPAGE_BPS = 8
SHV_SLIPPAGE_BPS = 2


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    output_dir = str(PROJECT_ROOT / "results")
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print("  STEP 3: Cash Alternative Backtest (RISK_OFF periods)")
    print("=" * 80)

    # Load data
    macro_data = get_macro_data("2010-01-01", TEST_END)
    macro_df = pd.DataFrame(macro_data).sort_index().ffill()
    switch = MacroRiskSwitch(**SWITCH_PARAMS)
    macro_signals = switch.compute_signals(macro_df)

    spy = _download_ohlcv("SPY", "2021-01-01", TEST_END)
    shv = _download_ohlcv("SHV", "2021-01-01", TEST_END)

    start_ts = pd.Timestamp(TEST_START)
    end_ts = pd.Timestamp(TEST_END)
    trading_days = spy.loc[(spy.index >= start_ts) & (spy.index <= end_ts)].index

    # Run three variants
    for variant, name in [("cash", "A) 100% Cash"), ("shv", "B) SHV"), ("shv_costs", "C) SHV + costs")]:
        cash = INITIAL_CAPITAL
        spy_shares = 0.0
        shv_shares = 0.0
        nav_list = []
        prev_regime = "NORMAL"

        signal_series = pd.Series(index=trading_days, dtype=float)
        regime_series = pd.Series(index=trading_days, dtype=str)
        for day in trading_days:
            if day in macro_signals.index:
                signal_series[day] = float(macro_signals.loc[day, "signal"])
                regime_series[day] = str(macro_signals.loc[day, "regime"])
            else:
                prev = macro_signals.loc[macro_signals.index <= day]
                signal_series[day] = float(prev["signal"].iloc[-1]) if not prev.empty else 1.0
                regime_series[day] = str(prev["regime"].iloc[-1]) if not prev.empty else "NORMAL"

        # Shift for T+1
        signal_series = signal_series.shift(1).fillna(1.0)
        regime_series = regime_series.shift(1).fillna("NORMAL")

        for day in trading_days:
            signal = signal_series[day]
            regime = regime_series[day]

            spy_price = float(spy.loc[day, "Open"]) if day in spy.index else 0
            shv_price = float(shv.loc[day, "Close"]) if day in shv.index else 0

            if regime != prev_regime:
                # Sell everything first
                if spy_shares > 0 and spy_price > 0:
                    cash += spy_shares * spy_price * (1 - SPY_SLIPPAGE_BPS / 10_000)
                    spy_shares = 0
                if shv_shares > 0 and shv_price > 0:
                    cost_mult = 1 - SHV_SLIPPAGE_BPS / 10_000 if variant == "shv_costs" else 1
                    cash += shv_shares * shv_price * cost_mult
                    shv_shares = 0

                # Reallocate
                if signal > 0 and spy_price > 0:
                    invest = cash * signal
                    spy_shares = invest / (spy_price * (1 + SPY_SLIPPAGE_BPS / 10_000))
                    cash -= spy_shares * spy_price * (1 + SPY_SLIPPAGE_BPS / 10_000)

                if signal < 1.0 and variant in ("shv", "shv_costs") and shv_price > 0:
                    # Put non-SPY portion into SHV
                    shv_invest = cash * 0.95  # keep 5% as buffer
                    cost_mult = 1 + SHV_SLIPPAGE_BPS / 10_000 if variant == "shv_costs" else 1
                    shv_shares = shv_invest / (shv_price * cost_mult)
                    cash -= shv_shares * shv_price * cost_mult

                prev_regime = regime

            # NAV at close
            spy_close = float(spy.loc[day, "Close"]) if day in spy.index else spy_price
            shv_close = float(shv.loc[day, "Close"]) if day in shv.index else shv_price
            nav = cash + spy_shares * spy_close + shv_shares * shv_close
            nav_list.append(nav)

        nav_s = pd.Series(nav_list, index=trading_days[:len(nav_list)])
        dr = nav_s.pct_change().dropna()
        total_ret = (nav_s.iloc[-1] / INITIAL_CAPITAL - 1) * 100
        sharpe = float((dr.mean() / dr.std()) * np.sqrt(252)) if dr.std() > 0 else 0
        dd = abs(float((nav_s / nav_s.cummax() - 1).min())) * 100

        print(f"\n  {name}: Return {total_ret:.2f}%, Sharpe {sharpe:.3f}, DD {dd:.2f}%")

    # Save
    filepath = os.path.join(output_dir, "cash_alternative.json")
    with open(filepath, "w") as f:
        json.dump({"status": "completed"}, f)
    print(f"\n  Results saved to {filepath}")


if __name__ == "__main__":
    main()
