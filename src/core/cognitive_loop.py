"""
Cognitive Feedback Loop
=======================
Post-trade reflection engine that connects:
  AuditRecord → Failure Attribution → Episodic Memory → OPRO → System Tuning

This runs asynchronously after each trade cycle to:
  1. Attribute failures to the correct diagnostic layer
  2. Store the trade episode (win or lose) in episodic memory
  3. Feed outcomes into Adaptive-OPRO for parameter evolution
  4. Apply any parameter changes back to the live system
  5. Run counterfactual replays when a failure pattern recurs
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from src.agents.audit.audit_trail import AuditRecord, AuditTrailAgent
from src.agents.audit.failure_attribution import FailureAttributionEngine, FailureReport
from src.agents.intelligence.orchestrator import IntelligenceOrchestrator
from src.memory.counterfactual_replay import CounterfactualReplayEngine
from src.memory.episodic_memory import EpisodicMemory
from src.prompts.adaptive_opro import AdaptiveOPRO

logger = logging.getLogger(__name__)


class CognitiveLoop:
    """
    Orchestrates the self-reflection cycle after each trade or batch.
    """

    # 每 N 次反思觸發一次失敗歸因驅動優化
    _FAILURE_OPTIMIZATION_INTERVAL = 10

    def __init__(
        self,
        audit: AuditTrailAgent,
        memory: EpisodicMemory,
        opro: AdaptiveOPRO,
        replay: CounterfactualReplayEngine,
        intelligence: IntelligenceOrchestrator,
    ) -> None:
        self._audit = audit
        self._memory = memory
        self._opro = opro
        self._replay = replay
        self._intelligence = intelligence
        self._attribution = FailureAttributionEngine(memory)
        self._reflection_count = 0

    async def reflect_on_trade(
        self,
        record: AuditRecord,
        market_features: dict[str, float],
        price_series: np.ndarray | None = None,
        regime_at_exit: str = "",
    ) -> dict[str, Any]:
        """
        Full cognitive reflection on a completed trade.

        Parameters
        ----------
        record : AuditRecord with outcome fields (exit_price, realised_pnl, etc.)
        market_features : dict of normalised market features at trade time
        price_series : price array from entry to exit (for counterfactual)
        regime_at_exit : market regime tag at trade close
        """
        self._reflection_count += 1
        result: dict[str, Any] = {
            "record_id": record.record_id,
            "ticker": record.ticker,
            "roi": record.realised_roi,
        }

        # ── 1. Store episode in memory ────────────────────────────────
        from dataclasses import asdict
        episode_id = self._memory.store_from_audit(
            audit_record=asdict(record) if hasattr(record, '__dataclass_fields__') else record.__dict__,
            market_features=market_features,
        )
        result["episode_id"] = episode_id

        # ── 2. Feed ROI into OPRO ─────────────────────────────────────
        roi = record.realised_roi or 0.0
        opro_score = self._opro.record_trade_outcome(roi)
        result["opro_score"] = opro_score

        # ── 3. Failure attribution (if losing trade) ──────────────────
        failure_report: FailureReport | None = None
        if roi < -0.005:
            failure_report = self._attribution.analyse(
                record=record,
                price_series=price_series,
                actual_regime_at_exit=regime_at_exit,
            )
            result["failure_report"] = {
                "primary_layer": failure_report.primary_layer,
                "semantic_pct": failure_report.semantic.contribution_pct,
                "temporal_pct": failure_report.temporal.contribution_pct,
                "execution_pct": failure_report.execution.contribution_pct,
                "summary": failure_report.summary,
                "corrective_actions": (
                    failure_report.semantic.corrective_actions
                    + failure_report.temporal.corrective_actions
                    + failure_report.execution.corrective_actions
                ),
                "similar_winners": failure_report.similar_winning_episodes,
                "counterfactual_roi": failure_report.counterfactual_roi,
            }

            # Update the audit record with refined failure layer
            record.failure_layer = failure_report.primary_layer
            record.failure_detail = failure_report.summary

        # ── 4. Maybe evolve OPRO population ───────────────────────────
        evolved = self._opro.maybe_evolve()
        if evolved:
            result["opro_evolved"] = True
            self._apply_opro_to_system()

        # ── 5. Periodic failure-driven agent-weight optimisation ───────
        if self._reflection_count % self._FAILURE_OPTIMIZATION_INTERVAL == 0:
            fdo_result = self._apply_failure_driven_tuning()
            if fdo_result.get("status") == "OPTIMIZED":
                result["failure_driven_updates"] = fdo_result["updates"]

        # ── 6. Periodic counterfactual replay ─────────────────────────
        if self._reflection_count % 10 == 0 and price_series is not None:
            result["counterfactual"] = self._run_counterfactual_comparison(
                price_series, market_features,
            )

        logger.info(
            "Cognitive reflection #%d for %s: ROI=%.4f, OPRO=%.1f%s",
            self._reflection_count,
            record.ticker,
            roi,
            opro_score,
            f", failure={failure_report.primary_layer}" if failure_report else "",
        )
        return result

    async def reflect_on_batch(
        self,
        records: list[AuditRecord],
        features_map: dict[str, dict[str, float]],
        price_map: dict[str, np.ndarray] | None = None,
    ) -> list[dict[str, Any]]:
        """Reflect on multiple completed trades."""
        results = []
        for record in records:
            features = features_map.get(record.ticker, {})
            prices = price_map.get(record.ticker) if price_map else None
            result = await self.reflect_on_trade(
                record=record,
                market_features=features,
                price_series=prices,
            )
            results.append(result)
        return results

    # ── system tuning ─────────────────────────────────────────────────

    def _apply_opro_to_system(self) -> None:
        """Push OPRO's active parameters into the live intelligence layer."""
        weights = self._opro.get_intelligence_weights()
        self._intelligence.update_weights(weights)
        logger.info("Applied OPRO weights to intelligence layer: %s", weights)

    def force_apply_parameters(self) -> dict[str, float]:
        """Manually trigger parameter application (for testing)."""
        self._apply_opro_to_system()
        return self._opro.active_parameters

    # ── failure-driven agent-weight tuning ────────────────────────────

    def _apply_failure_driven_tuning(self) -> dict[str, Any]:
        """
        委派 OPRO 執行失敗歸因驅動的代理權重突變。

        將 IntelligenceOrchestrator 持有的三個代理實例傳遞給 OPRO，
        讓它根據 EpisodicMemory 中的虧損歸因自動突變內部策略權重。
        """
        return self._opro.run_failure_driven_optimization(
            memory=self._memory,
            technical_agent=getattr(self._intelligence, "_technical", None),
            fundamental_agent=getattr(self._intelligence, "_fundamental", None),
            sentiment_agent=getattr(self._intelligence, "_sentiment", None),
        )

    def run_failure_optimization(self) -> dict[str, Any]:
        """Public API for on-demand failure-driven optimization."""
        return self._apply_failure_driven_tuning()

    # ── counterfactual comparison ─────────────────────────────────────

    def _run_counterfactual_comparison(
        self,
        price_series: np.ndarray,
        features: dict[str, float],
    ) -> dict[str, Any]:
        """
        Compare the active OPRO parameters against the current best
        alternative candidate.
        """
        active_params = self._opro.active_parameters

        # Generate a simple signal array from features (placeholder)
        signals = np.zeros(len(price_series))
        fused = features.get("fused_score", 0)
        signals[:] = fused  # constant signal for simplicity

        results = self._replay.compare_parameters(
            price_series=price_series,
            signals=signals,
            param_sets=[active_params],
            labels=["active"],
        )

        if results:
            return {
                "active_net_pnl": results[0].total_net_pnl,
                "active_sharpe": results[0].sharpe_ratio,
                "active_win_rate": results[0].win_rate,
            }
        return {}

    # ── analytics ─────────────────────────────────────────────────────

    def get_learning_summary(self) -> dict[str, Any]:
        """Summary of what the system has learned."""
        return {
            "total_reflections": self._reflection_count,
            "episodes_stored": self._memory.count,
            "opro_generation": self._opro._generation,
            "opro_active_candidate": self._opro.active_candidate.candidate_id,
            "opro_active_score": self._opro.active_candidate.score,
            "opro_active_params": self._opro.active_parameters,
            "regime_distribution": self._memory.regime_distribution(),
            "avg_roi_by_regime": self._memory.avg_roi_by_regime(),
            "audit_stats": self._audit.summary_stats(),
        }
