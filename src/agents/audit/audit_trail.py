"""
Audit Trail Agent
=================
Immutable, append-only record of every decision and execution in the system.
Provides post-trade analysis, slippage tracking, and data for the
Hierarchical Failure Attribution system (Phase 4).

Every trade produces an AuditRecord that captures the full causal chain:
  Intelligence signals → Decision → Risk envelope → Execution → Outcome
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.agents.decision.decision_fusion import TradeAction, TradeProposal
from src.agents.execution.executor import OrderStatus, OrderTicket
from src.agents.intelligence.orchestrator import MarketView
from src.core.base_agent import SignalDirection

logger = logging.getLogger(__name__)


@dataclass
class AuditRecord:
    """Immutable record of a single trade lifecycle."""
    record_id: str = ""
    timestamp: float = field(default_factory=time.time)

    # ── Intelligence layer snapshot ───────────────────────────────────
    ticker: str = ""
    isin: str = ""
    fused_score: float = 0.0
    fused_confidence: float = 0.0
    fused_direction: str = ""
    agent_signals: list[dict[str, Any]] = field(default_factory=list)

    # ── Decision layer snapshot ───────────────────────────────────────
    action: str = ""
    decision_reasoning: str = ""
    proposed_quantity: float = 0.0
    proposed_value: float = 0.0

    # ── Risk layer snapshot ───────────────────────────────────────────
    stop_loss: float = 0.0
    take_profit: float = 0.0
    var_95: float = 0.0
    kelly_fraction: float = 0.0
    risk_reward_ratio: float = 0.0

    # ── Execution layer snapshot ──────────────────────────────────────
    order_status: str = ""
    api_order_id: int | None = None
    fill_price: float | None = None
    fill_quantity: float | None = None
    slippage: float = 0.0
    execution_errors: list[str] = field(default_factory=list)
    compliance_veto: str = ""

    # ── Outcome (filled in post-trade) ────────────────────────────────
    entry_price: float = 0.0
    exit_price: float | None = None
    realised_pnl: float | None = None
    realised_roi: float | None = None
    holding_period_seconds: float | None = None

    # ── Failure attribution tags (Phase 4 will consume these) ─────────
    failure_layer: str = ""       # "semantic", "temporal", "execution", ""
    failure_detail: str = ""


class AuditTrailAgent:
    """
    Append-only audit logger.
    Persists records to JSONL files for post-analysis.
    """

    def __init__(self, log_dir: str = "logs/audit") -> None:
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._records: list[AuditRecord] = []
        self._counter = 0

    # ── record creation ───────────────────────────────────────────────

    def record_trade(
        self,
        view: MarketView,
        proposal: TradeProposal,
        ticket: OrderTicket,
    ) -> AuditRecord:
        """Create an audit record from the full decision-execution chain."""
        self._counter += 1
        record = AuditRecord(
            record_id=f"AUD-{self._counter:06d}",
            ticker=view.ticker,
            isin=view.isin,
            # Intelligence
            fused_score=view.fused_score,
            fused_confidence=view.fused_confidence,
            fused_direction=view.fused_direction.name,
            agent_signals=[
                {
                    "source": s.source,
                    "direction": s.direction.name,
                    "confidence": s.confidence,
                    "reasoning": s.reasoning,
                    "score": s.weighted_score,
                }
                for s in view.signals
            ],
            # Decision
            action=proposal.action.name,
            decision_reasoning=proposal.reasoning,
            proposed_quantity=proposal.quantity,
            proposed_value=proposal.estimated_value,
            # Risk
            stop_loss=proposal.risk.stop_loss_price,
            take_profit=proposal.risk.take_profit_price,
            var_95=proposal.risk.var_95,
            kelly_fraction=proposal.risk.kelly_fraction,
            risk_reward_ratio=proposal.risk.risk_reward_ratio,
            # Execution
            order_status=ticket.status.name,
            api_order_id=ticket.api_order_id,
            fill_price=ticket.fill_price,
            fill_quantity=ticket.fill_quantity,
            slippage=ticket.slippage,
            execution_errors=ticket.validation_errors,
            compliance_veto=ticket.veto.detail if ticket.veto else "",
            entry_price=ticket.fill_price or proposal.current_price,
        )

        self._records.append(record)
        self._persist(record)

        logger.info("Audit record %s created for %s", record.record_id, record.ticker)
        return record

    def record_outcome(
        self,
        record_id: str,
        exit_price: float,
        holding_period_seconds: float,
    ) -> AuditRecord | None:
        """Update an audit record with the trade outcome (at close)."""
        record = self._find(record_id)
        if record is None:
            logger.warning("Audit record %s not found", record_id)
            return None

        record.exit_price = exit_price
        record.holding_period_seconds = holding_period_seconds

        if record.entry_price > 0:
            if record.action == "BUY":
                record.realised_pnl = (exit_price - record.entry_price) * (record.fill_quantity or record.proposed_quantity)
                record.realised_roi = (exit_price - record.entry_price) / record.entry_price
            elif record.action == "SELL":
                record.realised_pnl = (record.entry_price - exit_price) * (record.fill_quantity or record.proposed_quantity)
                record.realised_roi = (record.entry_price - exit_price) / record.entry_price

        # Auto-tag failure layer (Phase 4 refines this)
        if record.realised_roi is not None and record.realised_roi < -0.02:
            record.failure_layer = self._classify_failure(record)

        self._persist(record, append=False)
        return record

    # ── analytics ─────────────────────────────────────────────────────

    def summary_stats(self) -> dict[str, Any]:
        """Aggregate performance statistics across all closed trades."""
        closed = [r for r in self._records if r.realised_pnl is not None]
        if not closed:
            return {"total_trades": 0}

        pnls = [r.realised_pnl for r in closed]
        rois = [r.realised_roi for r in closed if r.realised_roi is not None]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        slippages = [r.slippage for r in closed]

        return {
            "total_trades": len(closed),
            "total_pnl": sum(pnls),
            "win_count": len(wins),
            "loss_count": len(losses),
            "win_rate": len(wins) / len(closed) if closed else 0,
            "avg_roi": sum(rois) / len(rois) if rois else 0,
            "avg_slippage": sum(slippages) / len(slippages) if slippages else 0,
            "max_drawdown_single": min(pnls) if pnls else 0,
            "best_trade": max(pnls) if pnls else 0,
            "failure_breakdown": self._failure_breakdown(closed),
        }

    def get_records_for_instrument(self, isin: str) -> list[AuditRecord]:
        return [r for r in self._records if r.isin == isin]

    # ── failure classification (preliminary — Phase 4 deepens this) ───

    @staticmethod
    def _classify_failure(record: AuditRecord) -> str:
        """
        Three-layer failure attribution:
          1. semantic   — wrong analysis / reasoning
          2. temporal   — right idea, wrong timing / regime
          3. execution  — microstructure friction (slippage, delays)
        """
        # Execution layer: high slippage relative to loss
        if record.realised_pnl is not None and record.slippage != 0:
            slippage_contribution = abs(record.slippage) * (record.fill_quantity or 1)
            if record.realised_pnl < 0 and slippage_contribution > abs(record.realised_pnl) * 0.3:
                return "execution"

        # Temporal layer: low confidence suggests uncertain regime
        if record.fused_confidence < 0.4:
            return "temporal"

        # Semantic layer: high confidence but wrong direction
        if record.fused_confidence >= 0.6:
            return "semantic"

        return "temporal"

    # ── persistence ───────────────────────────────────────────────────

    def _persist(self, record: AuditRecord, append: bool = True) -> None:
        filepath = self._log_dir / "audit_trail.jsonl"
        mode = "a" if append else "r+"

        try:
            data = self._record_to_dict(record)
            line = json.dumps(data, default=str, ensure_ascii=False)

            if append:
                with open(filepath, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
            else:
                # Update existing line (re-write full file for simplicity)
                self._rewrite_record(filepath, record)
        except Exception:
            logger.exception("Failed to persist audit record %s", record.record_id)

    def _rewrite_record(self, filepath: Path, updated: AuditRecord) -> None:
        if not filepath.exists():
            return
        lines = filepath.read_text(encoding="utf-8").splitlines()
        new_lines = []
        for line in lines:
            try:
                obj = json.loads(line)
                if obj.get("record_id") == updated.record_id:
                    obj = self._record_to_dict(updated)
                new_lines.append(json.dumps(obj, default=str, ensure_ascii=False))
            except json.JSONDecodeError:
                new_lines.append(line)
        filepath.write_text("\n".join(new_lines) + "\n", encoding="utf-8")

    def _find(self, record_id: str) -> AuditRecord | None:
        for r in self._records:
            if r.record_id == record_id:
                return r
        return None

    @staticmethod
    def _record_to_dict(record: AuditRecord) -> dict[str, Any]:
        return asdict(record)

    @staticmethod
    def _failure_breakdown(records: list[AuditRecord]) -> dict[str, int]:
        counts: dict[str, int] = {"semantic": 0, "temporal": 0, "execution": 0}
        for r in records:
            if r.failure_layer in counts:
                counts[r.failure_layer] += 1
        return counts
