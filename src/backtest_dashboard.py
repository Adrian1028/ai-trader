"""
Backtest Dashboard Page — Streamlit UI
=======================================
Interactive configuration, execution, and visualization of backtests.
Integrated as a tab in the main AI Trading War Room dashboard.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DATA_DIR = os.getenv("DATA_DIR", "data")
BACKTEST_DIR = os.path.join(DATA_DIR, "backtest")
RESULTS_FILE = os.path.join(BACKTEST_DIR, "backtest_results.json")
PROGRESS_FILE = os.path.join(BACKTEST_DIR, "backtest_progress.json")

# Available tickers for backtesting
AVAILABLE_TICKERS = [
    "AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "GOOG", "META",
    "JPM", "BRK-B", "SPY", "QQQ", "NFLX", "DIS", "V", "MA",
    "BARC.L", "HSBA.L", "SHEL.L", "BP.L", "AZN.L",
]

DEFAULT_TICKERS = ["AAPL", "MSFT", "NVDA", "SPY"]


@st.cache_data(ttl=5)
def _load_results() -> dict | None:
    if not os.path.exists(RESULTS_FILE):
        return None
    try:
        with open(RESULTS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


@st.cache_data(ttl=3)
def _load_progress() -> dict | None:
    if not os.path.exists(PROGRESS_FILE):
        return None
    try:
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _is_running() -> bool:
    progress = _load_progress()
    return progress is not None and progress.get("status") == "running"


def render_backtest_page() -> None:
    """Main entry point for the backtest dashboard tab."""

    # ── Config Sidebar ────────────────────────────────────────────
    st.subheader("Backtest Configuration")

    col_cfg1, col_cfg2 = st.columns(2)
    with col_cfg1:
        selected_tickers = st.multiselect(
            "Tickers",
            options=AVAILABLE_TICKERS,
            default=DEFAULT_TICKERS,
            help="Select stocks to backtest",
        )
        start_date = st.date_input(
            "Start Date",
            value=pd.Timestamp("2020-01-01"),
            min_value=pd.Timestamp("2010-01-01"),
            max_value=pd.Timestamp("2025-12-31"),
        )

    with col_cfg2:
        initial_capital = st.number_input(
            "Initial Capital (£)",
            value=10000.0,
            min_value=1000.0,
            max_value=1000000.0,
            step=1000.0,
        )
        end_date = st.date_input(
            "End Date",
            value=pd.Timestamp("2025-01-01"),
            min_value=pd.Timestamp("2010-01-01"),
            max_value=pd.Timestamp("2026-03-31"),
        )

    with st.expander("Advanced Settings", expanded=False):
        adv1, adv2 = st.columns(2)
        with adv1:
            scan_interval = st.slider("Scan Interval (days)", 1, 20, 5)
            min_score = st.slider("Min Entry Score", 0.01, 0.50, 0.05, 0.01)
        with adv2:
            reflection_interval = st.slider("OPRO Reflection (days)", 5, 60, 20)
            confidence_floor = st.slider("Confidence Floor", 0.30, 0.90, 0.60, 0.05)

    st.divider()

    # ── Run Button ────────────────────────────────────────────────
    if _is_running():
        progress = _load_progress()
        if progress:
            pct = progress.get("pct", 0)
            st.warning(f"Backtest is running... Day {progress.get('day', '?')}/{progress.get('total_days', '?')} ({pct}%)")
            st.progress(pct / 100)
            st.caption(f"Date: {progress.get('current_date', '?')} | NAV: £{progress.get('nav', 0):,.2f} | Trades: {progress.get('trades', 0)}")
            st.markdown('<meta http-equiv="refresh" content="5">', unsafe_allow_html=True)
        return

    if st.button("Run Backtest", type="primary", use_container_width=True):
        if not selected_tickers:
            st.error("Please select at least one ticker.")
            return

        # Write initial progress file
        os.makedirs(BACKTEST_DIR, exist_ok=True)
        with open(PROGRESS_FILE, "w") as f:
            json.dump({"status": "running", "day": 0, "total_days": 0, "pct": 0}, f)

        # Launch backtest as subprocess
        cmd = [
            sys.executable, "src/backtest.py",
            "--tickers", *selected_tickers,
            "--start", str(start_date),
            "--end", str(end_date),
            "--capital", str(initial_capital),
            "--scan-interval", str(scan_interval),
            "--reflection-interval", str(reflection_interval),
            "--min-score", str(min_score),
            "--confidence-floor", str(confidence_floor),
        ]
        subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        st.success("Backtest started! This page will auto-refresh to show progress.")
        st.markdown('<meta http-equiv="refresh" content="5">', unsafe_allow_html=True)
        return

    st.divider()

    # ── Results Display ───────────────────────────────────────────
    results = _load_results()

    if results is None:
        st.info("No backtest results yet. Configure parameters above and click 'Run Backtest'.")
        return

    st.header("Backtest Results")

    # Meta info
    st.caption(
        f"Tickers: {', '.join(results.get('tickers', []))} | "
        f"Period: {results.get('start_date', '?')} to {results.get('end_date', '?')}"
    )

    # ── KPI Cards ─────────────────────────────────────────────────
    k1, k2, k3, k4, k5, k6 = st.columns(6)

    total_ret = results.get("total_return_pct", 0)
    bm_ret = results.get("benchmark_return_pct", 0)
    alpha = total_ret - bm_ret

    k1.metric("Final NAV", f"£{results.get('final_nav', 0):,.2f}")
    k2.metric("Total Return", f"{total_ret:+.2f}%")
    k3.metric("vs S&P 500", f"{alpha:+.2f}%",
              delta=f"{'Outperform' if alpha > 0 else 'Underperform'}")
    k4.metric("Max Drawdown", f"{results.get('max_drawdown_pct', 0):.2f}%")
    k5.metric("Sharpe Ratio", f"{results.get('sharpe_ratio', 0):.2f}")
    k6.metric("Win Rate", f"{results.get('win_rate', 0) * 100:.1f}%")

    m1, m2, m3 = st.columns(3)
    m1.metric("Total Trades", f"{results.get('total_trades', 0)}")
    m2.metric("OPRO Generations", f"{results.get('opro_generations', 0)}")
    m3.metric("S&P 500 Return", f"{bm_ret:+.2f}%")

    st.divider()

    # ── Equity Curve (Strategy vs Benchmark) ──────────────────────
    daily_nav = results.get("daily_nav", [])
    benchmark_nav = results.get("benchmark_nav", [])

    if daily_nav:
        st.subheader("Equity Curve: Strategy vs S&P 500 Benchmark")

        fig = go.Figure()

        # Strategy NAV
        dates = [d["date"] for d in daily_nav]
        navs = [d["nav"] for d in daily_nav]
        fig.add_trace(go.Scatter(
            x=dates, y=navs,
            name="AI Strategy",
            line=dict(color="#00CC96", width=2),
        ))

        # Benchmark NAV
        if benchmark_nav:
            bm_dates = [d["date"] for d in benchmark_nav]
            bm_navs = [d["nav"] for d in benchmark_nav]
            fig.add_trace(go.Scatter(
                x=bm_dates, y=bm_navs,
                name="S&P 500 (Buy & Hold)",
                line=dict(color="#636EFA", width=2, dash="dash"),
            ))

        # Initial capital line
        fig.add_hline(
            y=results.get("initial_capital", 10000),
            line_dash="dot", line_color="gray",
            annotation_text=f"Initial: £{results.get('initial_capital', 10000):,.0f}",
        )

        fig.update_layout(
            height=450,
            template="plotly_dark",
            yaxis_title="NAV (£)",
            xaxis_title="Date",
            legend=dict(x=0.01, y=0.99),
            margin=dict(t=10, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Drawdown Chart ────────────────────────────────────────────
    if daily_nav:
        st.subheader("Drawdown")
        nav_series = pd.Series([d["nav"] for d in daily_nav])
        peak = nav_series.cummax()
        drawdown_pct = (nav_series - peak) / peak * 100

        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=[d["date"] for d in daily_nav],
            y=drawdown_pct,
            name="Drawdown",
            fill="tozeroy",
            line=dict(color="#EF553B", width=1),
            fillcolor="rgba(239,85,59,0.3)",
        ))
        fig_dd.update_layout(
            height=250,
            template="plotly_dark",
            yaxis_title="Drawdown (%)",
            margin=dict(t=10, b=10),
        )
        st.plotly_chart(fig_dd, use_container_width=True)

    # ── OPRO Weight Evolution ─────────────────────────────────────
    weight_evo = results.get("weight_evolution", [])
    if weight_evo:
        st.subheader("OPRO Weight Evolution (Brain Neural Adaptation)")
        wdf = pd.DataFrame(weight_evo)
        if "date" in wdf.columns:
            wdf = wdf.set_index("date")
        weight_cols = [c for c in wdf.columns if c != "regime"]

        colors = {
            "trend_following": "#636EFA",
            "mean_reversion": "#00CC96",
            "volatility": "#FFA15A",
        }

        fig_w = go.Figure()
        for col in weight_cols:
            fig_w.add_trace(go.Scatter(
                x=wdf.index, y=wdf[col],
                name=col.replace("_", " ").title(),
                stackgroup="weights",
                line=dict(width=0.5),
                fillcolor=colors.get(col, "#AB63FA"),
            ))
        fig_w.update_layout(
            height=300,
            template="plotly_dark",
            yaxis_title="Weight",
            margin=dict(t=10, b=10),
        )
        st.plotly_chart(fig_w, use_container_width=True)
