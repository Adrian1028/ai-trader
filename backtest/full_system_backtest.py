"""
Full System Backtest — Macro Switch + Insider Not-Sold + Equal Weight
======================================================================

After Step 2 results:
  - Liquidity filter REMOVED (no value added)
  - System: Macro overlay + optional insider weighting on 20 stocks

Tests 3 strategy variants:
  A) Macro switch + equal-weight 20 stocks (baseline)
  B) Macro switch + insider-weighted 20 stocks
  C) Pure buy-and-hold SPY (benchmark)

Usage:
    python -m backtest.full_system_backtest
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
from src.signals.sec_insider import load_insider_data, compute_not_sold_scores, apply_insider_weights
from src.universe.stock_pool import TICKERS, SECTOR_MAP

logger = logging.getLogger(__name__)

# Periods
TRAIN_START = "2014-01-01"
TEST_START = "2022-01-01"
TEST_END = "2024-12-31"

INITIAL_CAPITAL = 100_000.0
SLIPPAGE_BPS = 5
REBALANCE_DAYS = 21  # monthly

# Risk limits
MAX_SINGLE_POSITION_PCT = 0.15
MAX_SECTOR_PCT = 0.35
MAX_DAILY_TURNOVER_PCT = 0.20


def _download_stocks(tickers: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    import yfinance as yf
    buffer = (pd.Timestamp(start) - pd.DateOffset(months=3)).strftime("%Y-%m-%d")
    data = {}
    for t in tickers:
        df = yf.download(t, start=buffer, end=end, progress=False, auto_adjust=True)
        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            data[t] = df
    return data


def _get_price(data: dict[str, pd.DataFrame], ticker: str, date: pd.Timestamp) -> float:
    if ticker not in data:
        return 0.0
    df = data[ticker]
    mask = df.index <= date
    if mask.any():
        return float(df.loc[mask, "Close"].iloc[-1])
    return 0.0


def _enforce_risk_limits(
    weights: dict[str, float],
    sector_map: dict[str, str],
) -> dict[str, float]:
    """Enforce single-stock and sector concentration limits."""
    # Cap single stock
    for t in weights:
        if weights[t] > MAX_SINGLE_POSITION_PCT:
            weights[t] = MAX_SINGLE_POSITION_PCT

    # Cap sector exposure
    sector_totals: dict[str, float] = {}
    for t, w in weights.items():
        sector = sector_map.get(t, "Other")
        sector_totals[sector] = sector_totals.get(sector, 0) + w

    for sector, total in sector_totals.items():
        if total > MAX_SECTOR_PCT:
            scale = MAX_SECTOR_PCT / total
            for t in weights:
                if sector_map.get(t, "Other") == sector:
                    weights[t] *= scale

    # Re-normalise
    total = sum(weights.values())
    if total > 0:
        weights = {t: w / total for t, w in weights.items()}
    return weights


def run_strategy(
    all_data: dict[str, pd.DataFrame],
    macro_signals: pd.DataFrame,
    spy: pd.Series,
    start: str,
    end: str,
    insider_data: dict[str, pd.DataFrame] | None = None,
    use_insider: bool = False,
    capital: float = INITIAL_CAPITAL,
) -> tuple[pd.Series, dict]:
    """Run a single strategy variant."""
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    trading_days = spy.loc[(spy.index >= start_ts) & (spy.index <= end_ts)].index

    cash = capital
    holdings: dict[str, float] = {}  # ticker -> shares
    nav_list = []
    days_since_rebal = REBALANCE_DAYS  # rebalance on first day
    trade_count = 0

    for day in trading_days:
        # Macro signal
        if day in macro_signals.index:
            signal = float(macro_signals.loc[day, "signal"])
        else:
            prev = macro_signals.loc[macro_signals.index <= day]
            signal = float(prev["signal"].iloc[-1]) if not prev.empty else 1.0

        days_since_rebal += 1

        if signal == 0.0 and holdings:
            # RISK_OFF: liquidate
            for t, shares in holdings.items():
                price = _get_price(all_data, t, day)
                if price > 0:
                    cash += shares * price * (1 - SLIPPAGE_BPS / 10_000)
                    trade_count += 1
            holdings.clear()
            days_since_rebal = 0

        elif days_since_rebal >= REBALANCE_DAYS:
            # Compute target weights
            available = [t for t in TICKERS if t in all_data]
            if not available:
                nav_list.append(cash + sum(
                    _get_price(all_data, t, day) * s for t, s in holdings.items()))
                continue

            # Base: equal weight
            base_weights = {t: 1.0 / len(available) for t in available}

            # Apply insider signal if enabled
            if use_insider and insider_data:
                scores = compute_not_sold_scores(
                    insider_data, available, day.strftime("%Y-%m-%d"),
                )
                target_weights = apply_insider_weights(base_weights, scores)
            else:
                target_weights = base_weights

            # Apply risk limits
            target_weights = _enforce_risk_limits(target_weights, SECTOR_MAP)

            # Scale by macro signal (0.5 for WARNING)
            target_weights = {t: w * signal for t, w in target_weights.items()}

            # Compute current NAV
            current_nav = cash
            for t, shares in holdings.items():
                current_nav += _get_price(all_data, t, day) * shares

            # Check turnover limit
            target_values = {t: current_nav * w for t, w in target_weights.items()}
            current_values = {t: _get_price(all_data, t, day) * holdings.get(t, 0)
                              for t in available}

            total_trade_value = sum(
                abs(target_values.get(t, 0) - current_values.get(t, 0))
                for t in available
            )
            turnover_pct = total_trade_value / max(current_nav, 1)

            # If turnover would exceed limit, scale down changes
            if turnover_pct > MAX_DAILY_TURNOVER_PCT:
                scale = MAX_DAILY_TURNOVER_PCT / turnover_pct
            else:
                scale = 1.0

            # Execute trades
            # Sells first
            for t in list(holdings.keys()):
                if t not in target_weights or target_weights.get(t, 0) == 0:
                    price = _get_price(all_data, t, day)
                    if price > 0:
                        cash += holdings[t] * price * (1 - SLIPPAGE_BPS / 10_000)
                        trade_count += 1
                    del holdings[t]

            # Rebalance
            for t, target_w in target_weights.items():
                price = _get_price(all_data, t, day)
                if price <= 0:
                    continue

                target_value = current_nav * target_w
                current_shares = holdings.get(t, 0)
                current_value = current_shares * price
                diff_value = (target_value - current_value) * scale

                diff_shares = int(diff_value / price)
                if abs(diff_shares) > 0:
                    cost = abs(diff_shares) * price * SLIPPAGE_BPS / 10_000
                    if diff_shares > 0:  # buy
                        outlay = diff_shares * price + cost
                        if outlay <= cash:
                            cash -= outlay
                            holdings[t] = current_shares + diff_shares
                            trade_count += 1
                    else:  # sell
                        proceeds = abs(diff_shares) * price - cost
                        cash += proceeds
                        new_shares = current_shares + diff_shares
                        if new_shares > 0:
                            holdings[t] = new_shares
                        elif t in holdings:
                            del holdings[t]
                        trade_count += 1

            days_since_rebal = 0

        # NAV
        nav = cash
        for t, shares in holdings.items():
            nav += _get_price(all_data, t, day) * shares
        nav_list.append(nav)

    nav_series = pd.Series(nav_list, index=trading_days[:len(nav_list)])

    # Metrics
    daily_ret = nav_series.pct_change().dropna()
    total_return = (nav_series.iloc[-1] / capital - 1) * 100
    sharpe = 0.0
    if daily_ret.std() > 0:
        sharpe = float((daily_ret.mean() / daily_ret.std()) * np.sqrt(252))
    dd = (nav_series / nav_series.cummax() - 1)
    max_dd = abs(float(dd.min())) * 100

    # 2022
    mask_2022 = (daily_ret.index >= "2022-01-01") & (daily_ret.index <= "2022-12-31")
    y2022 = float((1 + daily_ret[mask_2022]).prod() - 1) * 100 if mask_2022.any() else None

    metrics = {
        "total_return_pct": total_return,
        "sharpe": sharpe,
        "max_dd_pct": max_dd,
        "trade_count": trade_count,
        "year_2022_return": y2022,
    }

    return nav_series, metrics


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    output_dir = str(PROJECT_ROOT / "results")
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("  FULL SYSTEM BACKTEST")
    print("  Macro Switch + [Optional Insider Signal] + 20 Stocks")
    print("=" * 70)

    # Load macro
    print("\n  Loading macro data...")
    macro_data = get_macro_data(TRAIN_START, TEST_END)
    macro_df = pd.DataFrame(macro_data).sort_index().ffill()
    switch = MacroRiskSwitch(
        rv_trigger=0.98, rv_warning=0.92, rv_recover=0.90,
        oas_trigger=0.15, oas_recover=0.08, oas_lookback=21, min_hold_days=5,
    )
    macro_signals = switch.compute_signals(macro_df)

    # Load stocks
    print("  Loading stock data (20 tickers)...")
    all_data = _download_stocks(TICKERS, TRAIN_START, TEST_END)
    print(f"  Got {len(all_data)}/{len(TICKERS)} tickers")

    # Load SPY
    spy = fetch_from_yfinance("SPY", TRAIN_START, TEST_END)

    # Load insider data
    print("  Loading insider data...")
    insider_data = load_insider_data(list(all_data.keys()))

    # Strategy A: Macro + equal weight (no insider)
    print("\n  Running Strategy A: Macro + Equal Weight...")
    nav_a, metrics_a = run_strategy(
        all_data, macro_signals, spy, TEST_START, TEST_END,
        use_insider=False,
    )
    print(f"    Return: {metrics_a['total_return_pct']:.2f}%, Sharpe: {metrics_a['sharpe']:.3f}, "
          f"DD: {metrics_a['max_dd_pct']:.2f}%, Trades: {metrics_a['trade_count']}")

    # Strategy B: Macro + insider weighted
    print("\n  Running Strategy B: Macro + Insider Weighted...")
    nav_b, metrics_b = run_strategy(
        all_data, macro_signals, spy, TEST_START, TEST_END,
        insider_data=insider_data, use_insider=True,
    )
    print(f"    Return: {metrics_b['total_return_pct']:.2f}%, Sharpe: {metrics_b['sharpe']:.3f}, "
          f"DD: {metrics_b['max_dd_pct']:.2f}%, Trades: {metrics_b['trade_count']}")

    # Benchmark: SPY
    spy_start = float(spy.loc[spy.index >= TEST_START].iloc[0])
    spy_end = float(spy.loc[spy.index <= TEST_END].iloc[-1])
    spy_return = (spy_end / spy_start - 1) * 100
    spy_daily = spy.loc[(spy.index >= TEST_START) & (spy.index <= TEST_END)].pct_change().dropna()
    spy_sharpe = float((spy_daily.mean() / spy_daily.std()) * np.sqrt(252)) if spy_daily.std() > 0 else 0
    spy_nav = INITIAL_CAPITAL * spy.loc[(spy.index >= TEST_START) & (spy.index <= TEST_END)] / spy_start
    spy_dd = abs(float((spy_nav / spy_nav.cummax() - 1).min())) * 100

    # Summary
    print(f"\n{'=' * 70}")
    print(f"  FULL SYSTEM RESULTS (Test: {TEST_START} to {TEST_END})")
    print(f"{'=' * 70}")
    print(f"\n  {'Strategy':<35} {'Return':>10} {'Sharpe':>10} {'Max DD':>10}")
    print(f"  {'-' * 65}")
    print(f"  {'A) Macro + Equal Weight':<35} {metrics_a['total_return_pct']:>9.2f}% {metrics_a['sharpe']:>10.3f} {metrics_a['max_dd_pct']:>9.2f}%")
    print(f"  {'B) Macro + Insider Weighted':<35} {metrics_b['total_return_pct']:>9.2f}% {metrics_b['sharpe']:>10.3f} {metrics_b['max_dd_pct']:>9.2f}%")
    print(f"  {'C) SPY Buy & Hold':<35} {spy_return:>9.2f}% {spy_sharpe:>10.3f} {spy_dd:>9.2f}%")

    if metrics_a.get("year_2022_return") is not None:
        spy_2022 = float((1 + spy_daily[(spy_daily.index >= "2022-01-01") & (spy_daily.index <= "2022-12-31")]).prod() - 1) * 100
        print(f"\n  2022 Returns:")
        print(f"  {'A) Macro + EW':<35} {metrics_a['year_2022_return']:>9.2f}%")
        print(f"  {'B) Macro + Insider':<35} {metrics_b['year_2022_return']:>9.2f}%")
        print(f"  {'C) SPY':<35} {spy_2022:>9.2f}%")

    # Insider marginal contribution
    insider_delta_sharpe = metrics_b["sharpe"] - metrics_a["sharpe"]
    print(f"\n  Insider signal delta Sharpe: {insider_delta_sharpe:+.3f}")
    if insider_delta_sharpe > 0:
        print(f"  -> Insider signal ADDS value")
    else:
        print(f"  -> Insider signal does NOT add value, consider removing")

    # Gate check
    print(f"\n{'=' * 70}")
    print("  FINAL SYSTEM GATE CHECK")
    print(f"{'=' * 70}")

    # Use the better of A or B
    best = "B" if metrics_b["sharpe"] > metrics_a["sharpe"] else "A"
    best_metrics = metrics_b if best == "B" else metrics_a

    gates = {
        f"Strategy {best} Sharpe > 0.8": best_metrics["sharpe"] > 0.8,
        f"Strategy {best} beats SPY return": best_metrics["total_return_pct"] > spy_return,
        f"Strategy {best} Max DD < 30%": best_metrics["max_dd_pct"] < 30,
        f"Strategy {best} beats SPY in 2022": (
            best_metrics.get("year_2022_return") is not None
            and best_metrics["year_2022_return"] > -18.65
        ),
    }

    all_pass = True
    for gate, passed in gates.items():
        status = "PASS" if passed else "FAIL"
        symbol = "v" if passed else "x"
        if not passed:
            all_pass = False
        print(f"  {symbol} {status}  {gate}")

    if all_pass:
        print(f"\n  ALL GATES PASSED -> System ready for paper trading")
    else:
        print(f"\n  GATES FAILED -> Review signal logic")

    print(f"{'=' * 70}")

    # Save
    output = {
        "strategy_a": metrics_a,
        "strategy_b": metrics_b,
        "benchmark": {
            "total_return_pct": spy_return,
            "sharpe": spy_sharpe,
            "max_dd_pct": spy_dd,
        },
        "insider_delta_sharpe": insider_delta_sharpe,
        "best_strategy": best,
        "gates": gates,
        "all_gates_passed": all_pass,
    }
    filepath = os.path.join(output_dir, "full_system_backtest.json")
    with open(filepath, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to {filepath}")


if __name__ == "__main__":
    main()
