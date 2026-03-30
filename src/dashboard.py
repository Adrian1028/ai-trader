"""
AI Trading War Room — Streamlit Dashboard
==========================================
Real-time monitoring of the Multi-Agent Trading System.

Reads directly from the persistent JSON/JSONL files in data/ and logs/
without requiring a database. Designed to run alongside the trading bot
via docker-compose.

Panels:
  1. Virtual Account KPIs (NAV, cash, positions, ROI)
  2. Episodic Memory analytics (win rate, ROI scatter, regime distribution)
  3. Failure Attribution pie chart (execution / temporal / semantic)
  4. OPRO Evolution tracker (generation, candidate scores, parameter drift)
  5. Audit Trail table (recent trades with outcomes)

Usage:
  streamlit run src/dashboard.py
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Page config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Trading War Room",
    page_icon="\U0001f916",
    layout="wide",
)

# ── File paths (match persistence layer exactly) ──────────────────
DATA_DIR = os.getenv("DATA_DIR", "data")
LOGS_DIR = os.getenv("LOGS_DIR", "logs")

VIRTUAL_ACCOUNTS_FILE = os.path.join(DATA_DIR, "virtual_accounts.json")
LEARNING_REPORT_FILE = os.path.join(DATA_DIR, "learning_report.json")
EPISODES_FILE = os.path.join(LOGS_DIR, "memory", "episodes.jsonl")
OPRO_STATE_FILE = os.path.join(LOGS_DIR, "opro", "opro_state.json")
AUDIT_TRAIL_FILE = os.path.join(LOGS_DIR, "audit", "audit_trail.jsonl")


# ── Data loaders (cached with 10s TTL for live refresh) ───────────

@st.cache_data(ttl=10)
def load_virtual_accounts() -> dict:
    if not os.path.exists(VIRTUAL_ACCOUNTS_FILE):
        return {}
    try:
        with open(VIRTUAL_ACCOUNTS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


@st.cache_data(ttl=10)
def load_episodes() -> list[dict]:
    """Load episodic memory from JSONL (one JSON object per line)."""
    if not os.path.exists(EPISODES_FILE):
        return []
    episodes = []
    try:
        with open(EPISODES_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    episodes.append(json.loads(line))
    except (json.JSONDecodeError, OSError):
        pass
    return episodes


@st.cache_data(ttl=10)
def load_opro_state() -> dict:
    if not os.path.exists(OPRO_STATE_FILE):
        return {}
    try:
        with open(OPRO_STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


@st.cache_data(ttl=10)
def load_audit_trail() -> list[dict]:
    """Load audit trail from JSONL."""
    if not os.path.exists(AUDIT_TRAIL_FILE):
        return []
    records = []
    try:
        with open(AUDIT_TRAIL_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    except (json.JSONDecodeError, OSError):
        pass
    return records


@st.cache_data(ttl=10)
def load_learning_report() -> dict:
    if not os.path.exists(LEARNING_REPORT_FILE):
        return {}
    try:
        with open(LEARNING_REPORT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


# ── Attribution parser (matches AdaptiveOPRO._parse_attribution) ──

_RE_CHINESE = re.compile(
    r"\[執行:([\d.]+)%\|時序:([\d.]+)%\|語義:([\d.]+)%\]"
)
_RE_ENGLISH = re.compile(
    r"Primary failure:\s*(semantic|temporal|execution)\s*\((\d+(?:\.\d+)?)%\)"
)


def parse_attribution(
    failure_detail: str,
    failure_layer: str = "",
) -> tuple[float, float, float] | None:
    """
    Parse failure attribution from failure_detail string.
    Returns (execution%, temporal%, semantic%) or None.
    """
    if not failure_detail and not failure_layer:
        return None

    # Chinese format: [執行:10.0%|時序:80.0%|語義:10.0%]
    m_cn = _RE_CHINESE.search(failure_detail or "")
    if m_cn:
        return float(m_cn.group(1)), float(m_cn.group(2)), float(m_cn.group(3))

    # English format: Primary failure: semantic (65%) — ...
    m_en = _RE_ENGLISH.search(failure_detail or "")
    if m_en:
        primary = m_en.group(1)
        pct = float(m_en.group(2))
        remaining = (100.0 - pct) / 2
        result = {"execution": remaining, "temporal": remaining, "semantic": remaining}
        result[primary] = pct
        return result["execution"], result["temporal"], result["semantic"]

    # Fallback: only failure_layer → 100% to that layer
    if failure_layer in ("semantic", "temporal", "execution"):
        result = {"execution": 0.0, "temporal": 0.0, "semantic": 0.0}
        result[failure_layer] = 100.0
        return result["execution"], result["temporal"], result["semantic"]

    return None


# ══════════════════════════════════════════════════════════════════
#  DASHBOARD UI
# ══════════════════════════════════════════════════════════════════

st.title("\U0001f916 AI Quantitative Trading War Room")
st.caption(
    "Real-time monitoring of Multi-Agent System: "
    "Virtual accounts, episodic memory, failure attribution & OPRO evolution."
)

# Load all data
accounts = load_virtual_accounts()
episodes = load_episodes()
opro_state = load_opro_state()
audit_records = load_audit_trail()
learning_report = load_learning_report()


# ── Panel 1: Virtual Account KPIs ─────────────────────────────────

st.header("Virtual Account Status")

if accounts:
    bot_ids = list(accounts.keys())
    tabs = st.tabs(bot_ids) if len(bot_ids) > 1 else [st.container()]

    for idx, bot_id in enumerate(bot_ids):
        acc = accounts[bot_id]
        container = tabs[idx] if len(bot_ids) > 1 else tabs[0]

        with container:
            if len(bot_ids) > 1:
                pass  # tab already has the label
            else:
                st.subheader(f"Bot: {bot_id}")

            allocated = acc.get("allocated_capital", 0)
            cash = acc.get("available_cash", 0)
            positions = acc.get("positions", {})
            realised_pnl = acc.get("realised_pnl", 0)
            trade_count = acc.get("trade_count", 0)

            # Estimate NAV = cash + sum(qty * avg_price)
            total_invested = sum(
                p.get("quantity", 0) * p.get("average_price", 0)
                for p in positions.values()
            )
            est_nav = cash + total_invested
            roi_pct = ((est_nav - allocated) / allocated * 100) if allocated else 0

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Est. NAV", f"\u00a3{est_nav:,.2f}", f"{roi_pct:+.2f}%")
            c2.metric("Available Cash", f"\u00a3{cash:,.2f}")
            c3.metric("Invested Cost", f"\u00a3{total_invested:,.2f}")
            c4.metric("Realised PnL", f"\u00a3{realised_pnl:,.2f}")
            c5.metric("Trades / Positions", f"{trade_count} / {len(positions)}")

            if positions:
                st.subheader("Open Positions")
                pos_data = []
                for ticker, p in positions.items():
                    qty = p.get("quantity", 0)
                    avg_px = p.get("average_price", 0)
                    pos_data.append({
                        "Ticker": ticker,
                        "Quantity": qty,
                        "Avg Price": f"\u00a3{avg_px:.4f}",
                        "Cost Basis": f"\u00a3{qty * avg_px:.2f}",
                    })
                st.dataframe(
                    pd.DataFrame(pos_data),
                    use_container_width=True,
                    hide_index=True,
                )
else:
    st.info("No virtual account data yet. Waiting for system initialisation...")

st.divider()

# ── Panel 2: Episodic Memory Analytics ─────────────────────────────

st.header("\U0001f9e0 Episodic Memory & Performance")

if episodes:
    df = pd.DataFrame(episodes)

    # Derived columns
    df["is_win"] = df["roi"].apply(lambda r: r > 0 if pd.notna(r) else False)
    df["roi_pct"] = df["roi"] * 100

    total_trades = len(df)
    wins = df["is_win"].sum()
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    avg_roi = df["roi"].mean() * 100 if total_trades > 0 else 0

    # KPI row
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Episodes", f"{total_trades}")
    k2.metric("Win Rate", f"{win_rate:.1f}%")
    k3.metric("Avg ROI", f"{avg_roi:.2f}%")
    k4.metric(
        "Regime (latest)",
        df["regime_tag"].iloc[-1] if not df.empty and "regime_tag" in df else "N/A",
    )

    # Charts row
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Trade ROI Distribution")
        if "timestamp" in df.columns and total_trades > 0:
            df_plot = df.copy()
            df_plot["time"] = pd.to_datetime(df_plot["timestamp"], unit="s", errors="coerce")
            df_plot["result"] = df_plot["is_win"].map({True: "Win", False: "Loss"})
            fig_scatter = px.scatter(
                df_plot,
                x="time",
                y="roi_pct",
                color="result",
                hover_data=["ticker", "action", "regime_tag"],
                labels={"roi_pct": "ROI (%)", "time": "Time"},
                color_discrete_map={"Win": "#00CC96", "Loss": "#EF553B"},
            )
            fig_scatter.update_layout(height=350, margin=dict(t=10, b=10))
            st.plotly_chart(fig_scatter, use_container_width=True)

    with col_right:
        st.subheader("Failure Attribution Breakdown")

        # Parse attributions from losing trades
        failed = df[~df["is_win"]].copy()

        if not failed.empty:
            exec_tot, temp_tot, sem_tot = 0.0, 0.0, 0.0
            parsed_count = 0

            for _, row in failed.iterrows():
                detail = row.get("failure_detail", "") or ""
                layer = row.get("failure_layer", "") or ""
                result = parse_attribution(detail, layer)
                if result is not None:
                    exec_tot += result[0]
                    temp_tot += result[1]
                    sem_tot += result[2]
                    parsed_count += 1

            if parsed_count > 0:
                labels = [
                    "Execution (slippage/fees)",
                    "Temporal (timing/regime)",
                    "Semantic (signal error)",
                ]
                values = [
                    exec_tot / parsed_count,
                    temp_tot / parsed_count,
                    sem_tot / parsed_count,
                ]
                fig_pie = px.pie(
                    names=labels,
                    values=values,
                    title=f"Avg attribution across {parsed_count} losses",
                    color_discrete_sequence=["#FFA15A", "#636EFA", "#EF553B"],
                )
                fig_pie.update_layout(height=350, margin=dict(t=40, b=10))
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("No parseable attribution reports yet.")
        else:
            st.success("No losing trades recorded!")

    # Regime distribution
    if "regime_tag" in df.columns:
        col_regime, col_roi_regime = st.columns(2)
        with col_regime:
            st.subheader("Regime Distribution")
            regime_counts = df["regime_tag"].value_counts()
            fig_regime = px.bar(
                x=regime_counts.index,
                y=regime_counts.values,
                labels={"x": "Regime", "y": "Count"},
                color=regime_counts.index,
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig_regime.update_layout(
                height=280, margin=dict(t=10, b=10), showlegend=False,
            )
            st.plotly_chart(fig_regime, use_container_width=True)

        with col_roi_regime:
            st.subheader("Avg ROI by Regime")
            roi_by_regime = df.groupby("regime_tag")["roi_pct"].mean()
            fig_roi_regime = px.bar(
                x=roi_by_regime.index,
                y=roi_by_regime.values,
                labels={"x": "Regime", "y": "Avg ROI (%)"},
                color=roi_by_regime.values,
                color_continuous_scale=["#EF553B", "#FECB52", "#00CC96"],
            )
            fig_roi_regime.update_layout(
                height=280, margin=dict(t=10, b=10), showlegend=False,
            )
            st.plotly_chart(fig_roi_regime, use_container_width=True)

    # Recent episodes table
    st.subheader("Recent Episodes (last 15)")
    display_cols = [
        c for c in [
            "episode_id", "ticker", "action", "fused_confidence",
            "roi", "regime_tag", "failure_layer", "failure_detail",
        ]
        if c in df.columns
    ]
    df_display = df[display_cols].tail(15).copy()
    if "roi" in df_display.columns:
        df_display["roi"] = df_display["roi"].apply(lambda x: f"{x * 100:.2f}%")
    if "fused_confidence" in df_display.columns:
        df_display["fused_confidence"] = df_display["fused_confidence"].apply(
            lambda x: f"{x:.2f}" if pd.notna(x) and x != 0 else ""
        )
    st.dataframe(
        df_display.iloc[::-1],  # newest first
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("No episodic memory yet. The bot is accumulating experience...")

st.divider()

# ── Panel 3: OPRO Evolution ────────────────────────────────────────

st.header("\U0001f9ec OPRO Evolution Tracker")

if opro_state:
    generation = opro_state.get("generation", 0)
    active_idx = opro_state.get("active_idx", 0)
    candidates = opro_state.get("candidates", [])
    history = opro_state.get("history", [])

    o1, o2, o3 = st.columns(3)
    o1.metric("Generation", f"{generation}")
    active_name = candidates[active_idx]["id"] if candidates else "N/A"
    o2.metric("Active Candidate", active_name)
    best_score = max((c.get("score", 0) for c in candidates), default=0)
    o3.metric("Best Score", f"{best_score:.1f}")

    if candidates:
        col_cand, col_params = st.columns(2)

        with col_cand:
            st.subheader("Population Scores")
            cand_data = []
            for c in candidates:
                cand_data.append({
                    "ID": c.get("id", ""),
                    "Score": round(c.get("score", 0), 1),
                    "Trades": c.get("trade_count", 0),
                    "Avg ROI": f"{c.get('cumulative_roi', 0) / max(c.get('trade_count', 1), 1) * 100:.2f}%",
                    "Active": "\u2705" if c.get("active", False) else "",
                })
            st.dataframe(
                pd.DataFrame(cand_data),
                use_container_width=True,
                hide_index=True,
            )

        with col_params:
            st.subheader("Active Parameters")
            if candidates:
                active_params = candidates[active_idx].get("parameters", {})
                # Group parameters
                groups = {
                    "Intelligence Weights": [
                        "weight_fundamental", "weight_technical", "weight_sentiment",
                    ],
                    "Decision Thresholds": [
                        "min_confidence_to_trade", "min_buy_score", "max_sell_score",
                    ],
                    "Risk Tuning": [
                        "atr_stop_multiplier", "atr_tp_multiplier", "half_kelly_scaling",
                    ],
                }
                for group_name, keys in groups.items():
                    st.markdown(f"**{group_name}**")
                    for k in keys:
                        if k in active_params:
                            st.text(f"  {k}: {active_params[k]:.4f}")

    # Evolution history chart
    if history and len(history) > 1:
        st.subheader("Score Evolution Over Generations")
        gen_scores = []
        for gen_record in history:
            gen_num = gen_record.get("generation", 0)
            scores = gen_record.get("scores", [])
            if scores:
                best = max(s.get("score", 0) for s in scores)
                worst = min(s.get("score", 0) for s in scores)
                avg = sum(s.get("score", 0) for s in scores) / len(scores)
                gen_scores.append({
                    "Generation": gen_num,
                    "Best": best,
                    "Average": avg,
                    "Worst": worst,
                })

        if gen_scores:
            df_gen = pd.DataFrame(gen_scores)
            fig_evo = go.Figure()
            fig_evo.add_trace(go.Scatter(
                x=df_gen["Generation"], y=df_gen["Best"],
                mode="lines+markers", name="Best", line=dict(color="#00CC96"),
            ))
            fig_evo.add_trace(go.Scatter(
                x=df_gen["Generation"], y=df_gen["Average"],
                mode="lines+markers", name="Average", line=dict(color="#636EFA"),
            ))
            fig_evo.add_trace(go.Scatter(
                x=df_gen["Generation"], y=df_gen["Worst"],
                mode="lines", name="Worst", line=dict(color="#EF553B", dash="dot"),
            ))
            fig_evo.update_layout(
                yaxis_title="OPRO Score",
                height=300,
                margin=dict(t=10, b=10),
            )
            st.plotly_chart(fig_evo, use_container_width=True)
else:
    st.info("No OPRO state data yet. Waiting for first evolution cycle...")

st.divider()

# ── Panel 4: Audit Trail ───────────────────────────────────────────

st.header("\U0001f4dc Audit Trail (Recent Trades)")

if audit_records:
    df_audit = pd.DataFrame(audit_records)

    # Show recent 20
    display_audit_cols = [
        c for c in [
            "record_id", "ticker", "action", "fused_score", "fused_confidence",
            "proposed_quantity", "proposed_value", "order_status",
            "fill_price", "slippage", "realised_pnl", "realised_roi",
            "failure_layer",
        ]
        if c in df_audit.columns
    ]
    df_audit_display = df_audit[display_audit_cols].tail(20).copy()

    # Format columns
    for col in ["fused_score", "fused_confidence", "slippage"]:
        if col in df_audit_display.columns:
            df_audit_display[col] = df_audit_display[col].apply(
                lambda x: f"{x:.3f}" if pd.notna(x) and x != 0 else ""
            )
    for col in ["proposed_value", "fill_price", "realised_pnl"]:
        if col in df_audit_display.columns:
            df_audit_display[col] = df_audit_display[col].apply(
                lambda x: f"\u00a3{x:.2f}" if pd.notna(x) and x != 0 else ""
            )
    if "realised_roi" in df_audit_display.columns:
        df_audit_display["realised_roi"] = df_audit_display["realised_roi"].apply(
            lambda x: f"{x * 100:.2f}%" if pd.notna(x) and x != 0 else ""
        )

    st.dataframe(
        df_audit_display.iloc[::-1],  # newest first
        use_container_width=True,
        hide_index=True,
    )

    # Audit summary KPIs
    closed = df_audit[df_audit["realised_pnl"].notna() & (df_audit["realised_pnl"] != 0)]
    if not closed.empty:
        ac1, ac2, ac3, ac4 = st.columns(4)
        total_pnl = closed["realised_pnl"].sum()
        audit_wins = (closed["realised_pnl"] > 0).sum()
        audit_win_rate = audit_wins / len(closed) * 100

        ac1.metric("Closed Trades", f"{len(closed)}")
        ac2.metric("Total PnL", f"\u00a3{total_pnl:.2f}")
        ac3.metric("Audit Win Rate", f"{audit_win_rate:.1f}%")
        ac4.metric(
            "Avg Slippage",
            f"{closed['slippage'].abs().mean():.4f}"
            if "slippage" in closed.columns else "N/A",
        )
else:
    st.info("No audit records yet. Waiting for first trading cycle...")

st.divider()

# ── Panel 5: Learning Report ──────────────────────────────────────

if learning_report:
    with st.expander("Daily Learning Report (latest)", expanded=False):
        st.json(learning_report)

# ── Footer ─────────────────────────────────────────────────────────
st.caption("Auto-refreshes every 10 seconds via Streamlit cache. Click below to force refresh.")
if st.button("Refresh Now"):
    st.cache_data.clear()
    st.rerun()
