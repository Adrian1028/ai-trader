"""
Hierarchical Failure Attribution Engine
=======================================
Overcomes LLM self-attribution bias by decomposing trade failures
into three orthogonal diagnostic layers:

Layer 1 — SEMANTIC (推理錯誤)
  Signal: high confidence + wrong direction
  Root cause: fundamentals misread, sentiment misinterpreted, indicator miscalculated
  Evidence: compare agent signals vs actual price movement

Layer 2 — TEMPORAL (時效不匹配)
  Signal: correct direction eventually, but wrong entry timing or regime mismatch
  Root cause: trend strategy in ranging market, mean-reversion in breakout
  Evidence: compare regime tag at entry vs regime at exit; check if delayed
  entry would have been profitable

Layer 3 — EXECUTION (微觀結構摩擦)
  Signal: correct direction + timing, but slippage/fees ate the profit
  Root cause: wide spread, API latency, partial fills, stamp duty
  Evidence: compare theoretical PnL (no friction) vs realised PnL

Output: FailureReport with layer attribution, evidence vectors, and
corrective action recommendations fed back to Adaptive-OPRO.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.agents.audit.audit_trail import AuditRecord
from src.memory.episodic_memory import Episode, EpisodicMemory

logger = logging.getLogger(__name__)


@dataclass
class LayerDiagnosis:
    """Diagnosis for a single attribution layer."""
    layer: str                      # "semantic", "temporal", "execution"
    contribution_pct: float = 0.0   # 0-100%, how much this layer explains the loss
    evidence: list[str] = field(default_factory=list)
    corrective_actions: list[str] = field(default_factory=list)


@dataclass
class FailureReport:
    """Full hierarchical failure attribution for one losing trade."""
    report_id: str = ""
    audit_record_id: str = ""
    ticker: str = ""
    timestamp: float = field(default_factory=time.time)

    # Outcome
    roi: float = 0.0
    pnl: float = 0.0

    # Layer diagnoses
    semantic: LayerDiagnosis = field(
        default_factory=lambda: LayerDiagnosis(layer="semantic")
    )
    temporal: LayerDiagnosis = field(
        default_factory=lambda: LayerDiagnosis(layer="temporal")
    )
    execution: LayerDiagnosis = field(
        default_factory=lambda: LayerDiagnosis(layer="execution")
    )

    # Primary attribution (layer with highest contribution)
    primary_layer: str = ""
    summary: str = ""

    # Contrastive evidence
    similar_winning_episodes: list[str] = field(default_factory=list)
    counterfactual_roi: float | None = None


class FailureAttributionEngine:
    """
    Stateless analysis engine that takes an AuditRecord (with outcome)
    and produces a FailureReport.
    """

    def __init__(self, episodic_memory: EpisodicMemory) -> None:
        self._memory = episodic_memory
        self._counter = 0

    def analyse(
        self,
        record: AuditRecord,
        price_series: np.ndarray | None = None,
        actual_regime_at_exit: str = "",
    ) -> FailureReport:
        """
        Produce a hierarchical failure report for a losing trade.

        Parameters
        ----------
        record : AuditRecord with outcome fields populated
        price_series : price array from entry to exit (for counterfactual)
        actual_regime_at_exit : market regime at trade close
        """
        self._counter += 1
        report = FailureReport(
            report_id=f"FR-{self._counter:06d}",
            audit_record_id=record.record_id,
            ticker=record.ticker,
            roi=record.realised_roi or 0.0,
            pnl=record.realised_pnl or 0.0,
        )

        # Only analyse losses
        if report.roi >= 0:
            report.summary = "Trade was profitable — no failure attribution needed."
            return report

        total_loss = abs(report.pnl)
        if total_loss == 0:
            report.summary = "Zero PnL — no failure to attribute."
            return report

        # ── Layer 1: Semantic diagnosis ───────────────────────────────
        self._diagnose_semantic(report, record, total_loss)

        # ── Layer 2: Temporal diagnosis ───────────────────────────────
        self._diagnose_temporal(
            report, record, price_series, actual_regime_at_exit, total_loss,
        )

        # ── Layer 3: Execution diagnosis ──────────────────────────────
        self._diagnose_execution(report, record, total_loss)

        # ── Normalise contributions to sum to 100% ────────────────────
        total_contrib = (
            report.semantic.contribution_pct
            + report.temporal.contribution_pct
            + report.execution.contribution_pct
        )
        if total_contrib > 0:
            report.semantic.contribution_pct *= 100 / total_contrib
            report.temporal.contribution_pct *= 100 / total_contrib
            report.execution.contribution_pct *= 100 / total_contrib

        # ── Primary attribution ───────────────────────────────────────
        layers = [report.semantic, report.temporal, report.execution]
        primary = max(layers, key=lambda l: l.contribution_pct)
        report.primary_layer = primary.layer

        # ── Contrastive evidence from episodic memory ─────────────────
        self._attach_contrastive_evidence(report, record)

        # ── Summary ───────────────────────────────────────────────────
        report.summary = (
            f"Primary failure: {primary.layer} ({primary.contribution_pct:.0f}%) — "
            f"{'; '.join(primary.evidence[:2])}"
        )

        logger.info(
            "FailureReport %s for %s: %s",
            report.report_id, record.ticker, report.summary,
        )
        return report

    # ── layer 1: semantic ─────────────────────────────────────────────

    def _diagnose_semantic(
        self,
        report: FailureReport,
        record: AuditRecord,
        total_loss: float,
    ) -> None:
        diag = report.semantic
        contribution = 0.0

        # High confidence + wrong direction = semantic failure
        if record.fused_confidence >= 0.6:
            contribution += 40
            diag.evidence.append(
                f"High confidence ({record.fused_confidence:.2f}) but trade lost "
                f"{abs(record.realised_roi or 0):.1%}"
            )

            # Which agent(s) were most wrong?
            for sig in record.agent_signals:
                if record.action == "BUY" and sig.get("direction") in ("SELL", "STRONG_SELL"):
                    diag.evidence.append(
                        f"Agent '{sig['source']}' correctly signalled against — "
                        f"was overridden by fusion weights"
                    )
                    diag.corrective_actions.append(
                        f"Consider increasing weight for '{sig['source']}'"
                    )
                elif record.action == "BUY" and sig.get("direction") in ("BUY", "STRONG_BUY"):
                    if sig.get("confidence", 0) > 0.7:
                        contribution += 10
                        diag.evidence.append(
                            f"Agent '{sig['source']}' was highly confident "
                            f"({sig.get('confidence', 0):.2f}) but wrong — "
                            f"reasoning: {sig.get('reasoning', '')[:100]}"
                        )
                        diag.corrective_actions.append(
                            f"Review scoring logic in '{sig['source']}' agent"
                        )

        elif record.fused_confidence < 0.4:
            # Low confidence + loss = less semantic, more temporal
            contribution += 10
            diag.evidence.append(
                f"Low confidence ({record.fused_confidence:.2f}) — "
                f"system was uncertain, suggesting data was ambiguous"
            )

        diag.contribution_pct = min(contribution, 80)

    # ── layer 2: temporal ─────────────────────────────────────────────

    def _diagnose_temporal(
        self,
        report: FailureReport,
        record: AuditRecord,
        price_series: np.ndarray | None,
        regime_at_exit: str,
        total_loss: float,
    ) -> None:
        diag = report.temporal
        contribution = 0.0

        # Regime mismatch detection
        if regime_at_exit and record.failure_layer:
            # We can infer entry regime from the record's agent signals
            had_trend_signal = any(
                "SMA" in sig.get("reasoning", "") or "trend" in sig.get("reasoning", "").lower()
                for sig in record.agent_signals
            )
            if had_trend_signal and regime_at_exit in ("mean_reverting", "quiet"):
                contribution += 35
                diag.evidence.append(
                    f"Trend strategy used in {regime_at_exit} regime — regime mismatch"
                )
                diag.corrective_actions.append(
                    "Add regime detection gate before trend-following signals"
                )

        # Delayed profitability check (counterfactual timing)
        if price_series is not None and len(price_series) > 10:
            entry_price = record.entry_price
            if entry_price > 0:
                # Would a later entry have been profitable?
                future_min = np.min(price_series[5:])  # skip first 5 bars
                future_max = np.max(price_series[5:])

                if record.action == "BUY":
                    delayed_roi = (future_max - entry_price) / entry_price
                    if delayed_roi > 0.01:
                        contribution += 25
                        diag.evidence.append(
                            f"Price reached {future_max:.2f} after initial drawdown — "
                            f"timing was premature (delayed ROI would be {delayed_roi:.1%})"
                        )
                        diag.corrective_actions.append(
                            "Tighten entry timing using shorter-timeframe confirmation"
                        )
                        report.counterfactual_roi = delayed_roi

        # Medium confidence (unsure zone) often indicates temporal issues
        if 0.35 <= record.fused_confidence <= 0.55:
            contribution += 15
            diag.evidence.append(
                f"Moderate confidence ({record.fused_confidence:.2f}) — "
                f"signals were mixed, suggesting transitional market state"
            )
            diag.corrective_actions.append(
                "Require higher confidence threshold in ambiguous regimes"
            )

        # Low confidence → temporal uncertainty is a likely contributor
        if record.fused_confidence < 0.4:
            contribution += 25
            diag.evidence.append(
                f"Low confidence ({record.fused_confidence:.2f}) — "
                f"timing and market state were likely ambiguous"
            )

        diag.contribution_pct = min(contribution, 80)

    # ── layer 3: execution ────────────────────────────────────────────

    def _diagnose_execution(
        self,
        report: FailureReport,
        record: AuditRecord,
        total_loss: float,
    ) -> None:
        diag = report.execution
        contribution = 0.0

        # Slippage analysis
        if record.slippage != 0:
            slippage_cost = abs(record.slippage) * (record.fill_quantity or record.proposed_quantity)
            slippage_as_pct_of_loss = slippage_cost / total_loss if total_loss > 0 else 0

            if slippage_as_pct_of_loss > 0.30:
                contribution += 40
                diag.evidence.append(
                    f"Slippage ({record.slippage:+.4f}) contributed "
                    f"{slippage_as_pct_of_loss:.0%} of total loss"
                )
                diag.corrective_actions.append(
                    "Use limit orders instead of market orders for this instrument"
                )
            elif slippage_as_pct_of_loss > 0.10:
                contribution += 15
                diag.evidence.append(
                    f"Moderate slippage impact ({slippage_as_pct_of_loss:.0%} of loss)"
                )

        # UK transaction costs (stamp duty 0.5%, PTM levy)
        stamp_duty_cost = record.proposed_value * 0.005
        if stamp_duty_cost > total_loss * 0.15:
            contribution += 15
            diag.evidence.append(
                f"Stamp duty (est. {stamp_duty_cost:.2f}) was significant "
                f"relative to total loss ({total_loss:.2f})"
            )
            diag.corrective_actions.append(
                "Increase minimum expected ROI to cover UK transaction costs"
            )

        # API-related issues
        if record.execution_errors:
            contribution += 20
            diag.evidence.append(
                f"Execution errors encountered: {record.execution_errors[:3]}"
            )
            diag.corrective_actions.append(
                "Investigate API reliability; add retry logic or fallback broker"
            )

        # Partial fill
        if (record.fill_quantity and record.proposed_quantity
                and record.fill_quantity < record.proposed_quantity * 0.9):
            contribution += 10
            diag.evidence.append(
                f"Partial fill: {record.fill_quantity:.2f} / {record.proposed_quantity:.2f} "
                f"({record.fill_quantity / record.proposed_quantity:.0%})"
            )

        diag.contribution_pct = min(contribution, 80)

    # ── contrastive evidence ──────────────────────────────────────────

    def _attach_contrastive_evidence(
        self,
        report: FailureReport,
        record: AuditRecord,
    ) -> None:
        """
        Find similar episodes in memory that were profitable, to highlight
        what could have been done differently.
        """
        features = {
            "fused_score": record.fused_score,
            "fused_confidence": record.fused_confidence,
            "var_95_pct": record.var_95,
            "kelly_fraction": record.kelly_fraction,
        }
        # Add agent-level features
        for sig in record.agent_signals:
            src = sig.get("source", "")
            features[f"{src}_confidence"] = sig.get("confidence", 0)

        similar = self._memory.query_by_features(features, k=5, min_similarity=0.3)
        winning = [(ep, sim) for ep, sim in similar if ep.roi > 0.005]

        for ep, sim in winning[:3]:
            report.similar_winning_episodes.append(
                f"{ep.episode_id} (ticker={ep.ticker}, ROI={ep.roi:.1%}, "
                f"regime={ep.regime_tag}, sim={sim:.2f})"
            )

        if winning:
            best_win = winning[0][0]
            # Suggest parameter differences
            for key in best_win.features:
                if key in features:
                    diff = best_win.features[key] - features.get(key, 0)
                    if abs(diff) > 0.2:
                        report.temporal.corrective_actions.append(
                            f"Winning episode had {key}={best_win.features[key]:.2f} "
                            f"vs this trade's {features.get(key, 0):.2f}"
                        )

    # ── convenience: diagnose from raw params + writeback ───────────────

    def diagnose_and_update(
        self,
        episode_id: str,
        expected_price: float,
        fill_price: float,
        close_price: float,
        roi: float,
        agent_confidences: dict[str, float] | None = None,
    ) -> FailureReport:
        """
        快捷介面：接受原始交易參數而非 AuditRecord，自動診斷並回寫記憶。

        此方法整合了用戶 Canvas 的三大特色：
        1. 滑價率隔離 — 以 ``slippage_pct / |roi|`` 切分執行層比例
        2. 信心比例分配 — 以各代理人信心比例分配語義/時效層
        3. 直接記憶回寫 — 自動呼叫 ``update_episode_failure()``

        Parameters
        ----------
        episode_id : 對應的 Episode ID
        expected_price : 預期成交價
        fill_price : 實際成交價
        close_price : 最終平倉價
        roi : 已實現 ROI (負數 = 虧損)
        agent_confidences : 各代理人信心 dict，例如
            {"technical": 0.8, "fundamental": 0.3, "sentiment": 0.5}

        Returns
        -------
        FailureReport with layer attributions normalised to 100%.
        """
        self._counter += 1
        report = FailureReport(
            report_id=f"FR-{self._counter:06d}",
            ticker="",
            roi=roi,
        )

        # Retrieve episode info from memory
        episode = self._memory.get_episode(episode_id)
        if episode is not None:
            report.ticker = episode.ticker
            report.audit_record_id = episode.audit_record_id

        # Only attribute losses
        if roi >= 0:
            report.summary = "Trade was profitable — no failure attribution needed."
            return report

        abs_roi = abs(roi)
        if abs_roi == 0:
            report.summary = "Zero ROI — no failure to attribute."
            return report

        agent_confidences = agent_confidences or {}

        # ── Execution layer: slippage isolation ──────────────────────
        slippage_pct = abs(fill_price - expected_price) / expected_price if expected_price > 0 else 0.0
        exec_share = min(slippage_pct / abs_roi, 1.0) if abs_roi > 0 else 0.0
        report.execution.contribution_pct = exec_share * 100

        if exec_share > 0.30:
            report.execution.evidence.append(
                f"Slippage ({slippage_pct:.2%} of price) consumed "
                f"{exec_share:.0%} of the loss"
            )
            report.execution.corrective_actions.append(
                "Use limit orders or reduce order size to mitigate slippage"
            )
        elif exec_share > 0.05:
            report.execution.evidence.append(
                f"Moderate slippage impact ({exec_share:.0%} of loss)"
            )

        # ── Semantic + Temporal: confidence-proportional split ────────
        remaining = 1.0 - exec_share
        total_conf = sum(agent_confidences.values()) if agent_confidences else 0.0

        if total_conf > 0:
            # High overall confidence → more semantic blame (推理錯了卻很有信心)
            avg_conf = total_conf / len(agent_confidences)
            semantic_ratio = avg_conf  # 信心越高，語義層佔比越大
            temporal_ratio = 1.0 - semantic_ratio
        else:
            # No confidence data → default 50/50
            semantic_ratio = 0.5
            temporal_ratio = 0.5

        report.semantic.contribution_pct = remaining * semantic_ratio * 100
        report.temporal.contribution_pct = remaining * temporal_ratio * 100

        # Semantic evidence
        if semantic_ratio > 0.5:
            high_conf_agents = [
                name for name, c in agent_confidences.items() if c > 0.6
            ]
            if high_conf_agents:
                report.semantic.evidence.append(
                    f"High-confidence agents {high_conf_agents} were wrong — "
                    f"signals misread the market direction"
                )
                report.semantic.corrective_actions.append(
                    "Review signal logic for high-confidence-yet-wrong agents"
                )
            else:
                report.semantic.evidence.append(
                    f"Average confidence = {avg_conf:.2f} — "
                    f"moderate semantic attribution"
                )
        else:
            report.temporal.evidence.append(
                "Low overall confidence suggests timing/regime ambiguity"
            )
            report.temporal.corrective_actions.append(
                "Add regime confirmation gate before acting on uncertain signals"
            )

        # ── Normalise to 100% ────────────────────────────────────────
        total_contrib = (
            report.semantic.contribution_pct
            + report.temporal.contribution_pct
            + report.execution.contribution_pct
        )
        if total_contrib > 0:
            report.semantic.contribution_pct *= 100 / total_contrib
            report.temporal.contribution_pct *= 100 / total_contrib
            report.execution.contribution_pct *= 100 / total_contrib

        # ── Primary layer ────────────────────────────────────────────
        layers = [report.semantic, report.temporal, report.execution]
        primary = max(layers, key=lambda l: l.contribution_pct)
        report.primary_layer = primary.layer
        report.summary = (
            f"Primary failure: {primary.layer} "
            f"({primary.contribution_pct:.0f}%) — "
            f"{'; '.join(primary.evidence[:2]) or 'see corrective actions'}"
        )

        # ── Memory writeback ─────────────────────────────────────────
        self._memory.update_episode_failure(
            episode_id=episode_id,
            failure_layer=primary.layer,
            failure_detail=report.summary,
        )

        logger.info(
            "diagnose_and_update %s → %s (%s %.0f%%)",
            episode_id, report.report_id, primary.layer,
            primary.contribution_pct,
        )
        return report
