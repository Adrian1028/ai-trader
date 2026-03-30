"""Unit tests for Hierarchical Failure Attribution."""
from __future__ import annotations

import numpy as np
import pytest

from src.agents.audit.audit_trail import AuditRecord
from src.agents.audit.failure_attribution import FailureAttributionEngine
from src.memory.episodic_memory import Episode, EpisodicMemory


@pytest.fixture
def engine(tmp_path):
    mem = EpisodicMemory(store_dir=str(tmp_path / "test_fa_mem"))
    return FailureAttributionEngine(mem)


def _make_losing_record(**overrides) -> AuditRecord:
    defaults = dict(
        record_id="AUD-TEST",
        ticker="AAPL",
        isin="US0378331005",
        fused_score=0.8,
        fused_confidence=0.75,
        fused_direction="BUY",
        action="BUY",
        agent_signals=[
            {"source": "fundamental", "direction": "BUY", "confidence": 0.8,
             "reasoning": "Strong P/E ratio", "score": 0.8},
            {"source": "technical", "direction": "BUY", "confidence": 0.7,
             "reasoning": "SMA50 above SMA200 trend", "score": 0.7},
            {"source": "sentiment", "direction": "NEUTRAL", "confidence": 0.3,
             "reasoning": "Mixed news", "score": 0.0},
        ],
        entry_price=150.0,
        exit_price=140.0,
        realised_pnl=-50.0,
        realised_roi=-0.0667,
        proposed_quantity=5.0,
        proposed_value=750.0,
        fill_quantity=5.0,
        slippage=0.0,
    )
    defaults.update(overrides)
    return AuditRecord(**defaults)


class TestSemanticAttribution:
    def test_high_confidence_wrong_direction(self, engine):
        record = _make_losing_record(fused_confidence=0.85)
        report = engine.analyse(record)
        # High confidence + loss → strong semantic component
        assert report.semantic.contribution_pct > 0
        assert len(report.semantic.evidence) > 0

    def test_low_confidence_less_semantic(self, engine):
        record = _make_losing_record(fused_confidence=0.25)
        report = engine.analyse(record)
        assert report.semantic.contribution_pct < 30


class TestTemporalAttribution:
    def test_regime_mismatch(self, engine):
        record = _make_losing_record(fused_confidence=0.5)
        # Simulate price that eventually recovers
        prices = np.concatenate([
            np.linspace(150, 140, 10),  # initial drop
            np.linspace(140, 160, 20),  # recovery
        ])
        report = engine.analyse(record, price_series=prices)
        # Should detect that delayed entry would have been profitable
        if report.temporal.contribution_pct > 0:
            assert report.counterfactual_roi is not None or len(report.temporal.evidence) > 0


class TestExecutionAttribution:
    def test_high_slippage_attribution(self, engine):
        record = _make_losing_record(
            slippage=-2.0,       # severe slippage
            realised_pnl=-15.0,  # small loss
            fill_quantity=5.0,
        )
        report = engine.analyse(record)
        # Slippage contributed heavily to the loss
        assert report.execution.contribution_pct > 0
        assert any("slippage" in e.lower() for e in report.execution.evidence)

    def test_stamp_duty_impact(self, engine):
        record = _make_losing_record(
            proposed_value=50_000,
            realised_pnl=-300.0,  # stamp duty alone is £250
        )
        report = engine.analyse(record)
        assert report.execution.contribution_pct > 0


class TestNormalisation:
    def test_contributions_sum_to_100(self, engine):
        record = _make_losing_record()
        report = engine.analyse(record)
        total = (
            report.semantic.contribution_pct
            + report.temporal.contribution_pct
            + report.execution.contribution_pct
        )
        assert total == pytest.approx(100.0, abs=1.0)

    def test_primary_layer_set(self, engine):
        record = _make_losing_record()
        report = engine.analyse(record)
        assert report.primary_layer in ("semantic", "temporal", "execution")


class TestProfitableTrade:
    def test_no_attribution_for_winners(self, engine):
        record = _make_losing_record(realised_pnl=100.0, realised_roi=0.05)
        report = engine.analyse(record)
        assert "profitable" in report.summary.lower() or "no failure" in report.summary.lower()


# ── diagnose_and_update() tests ────────────────────────────────────

class TestDiagnoseAndUpdate:
    """測試 diagnose_and_update() 快捷介面"""

    @pytest.fixture
    def mem(self, tmp_path):
        return EpisodicMemory(store_dir=str(tmp_path / "dau_mem"))

    @pytest.fixture
    def eng(self, mem):
        return FailureAttributionEngine(mem)

    def test_basic_loss_attribution(self, mem, eng):
        ep_id = mem.store(Episode(
            ticker="AAPL", embedding=np.random.randn(32), roi=-0.05,
        ))
        report = eng.diagnose_and_update(
            episode_id=ep_id,
            expected_price=150.0,
            fill_price=151.0,       # 0.67% slippage
            close_price=140.0,
            roi=-0.05,
            agent_confidences={"technical": 0.8, "fundamental": 0.7},
        )
        assert report.primary_layer in ("semantic", "temporal", "execution")
        total = (
            report.semantic.contribution_pct
            + report.temporal.contribution_pct
            + report.execution.contribution_pct
        )
        assert total == pytest.approx(100.0, abs=1.0)

    def test_high_slippage_favours_execution(self, mem, eng):
        """大滑價 → 執行層佔比高"""
        ep_id = mem.store(Episode(
            ticker="TSLA", embedding=np.random.randn(32), roi=-0.02,
        ))
        report = eng.diagnose_and_update(
            episode_id=ep_id,
            expected_price=200.0,
            fill_price=204.0,       # 2% slippage on a 2% loss → ~100% execution
            close_price=196.0,
            roi=-0.02,
        )
        assert report.execution.contribution_pct > 50

    def test_high_confidence_favours_semantic(self, mem, eng):
        """高信心 + 虧損 → 語義層佔比高"""
        ep_id = mem.store(Episode(
            ticker="MSFT", embedding=np.random.randn(32), roi=-0.08,
        ))
        report = eng.diagnose_and_update(
            episode_id=ep_id,
            expected_price=300.0,
            fill_price=300.0,       # 零滑價
            close_price=276.0,
            roi=-0.08,
            agent_confidences={"technical": 0.9, "fundamental": 0.85, "sentiment": 0.8},
        )
        # No execution blame (zero slippage) → all semantic+temporal
        assert report.execution.contribution_pct < 5
        assert report.semantic.contribution_pct > report.temporal.contribution_pct

    def test_low_confidence_favours_temporal(self, mem, eng):
        """低信心 → 時效層佔比高"""
        ep_id = mem.store(Episode(
            ticker="GOOG", embedding=np.random.randn(32), roi=-0.04,
        ))
        report = eng.diagnose_and_update(
            episode_id=ep_id,
            expected_price=100.0,
            fill_price=100.0,
            close_price=96.0,
            roi=-0.04,
            agent_confidences={"technical": 0.2, "fundamental": 0.1},
        )
        assert report.temporal.contribution_pct > report.semantic.contribution_pct

    def test_memory_writeback(self, mem, eng):
        """診斷結果應自動回寫到 Episode"""
        ep_id = mem.store(Episode(
            ticker="AMZN", embedding=np.random.randn(32), roi=-0.06,
        ))
        eng.diagnose_and_update(
            episode_id=ep_id,
            expected_price=180.0,
            fill_price=180.0,
            close_price=169.2,
            roi=-0.06,
            agent_confidences={"technical": 0.7},
        )
        ep = mem.get_episode(ep_id)
        assert ep.failure_layer != ""
        assert ep.failure_detail != ""
        assert "Primary failure" in ep.failure_detail

    def test_profitable_trade_no_writeback(self, mem, eng):
        """獲利交易不觸發歸因也不回寫"""
        ep_id = mem.store(Episode(
            ticker="NVDA", embedding=np.random.randn(32), roi=0.10,
        ))
        report = eng.diagnose_and_update(
            episode_id=ep_id,
            expected_price=500.0,
            fill_price=500.0,
            close_price=550.0,
            roi=0.10,
        )
        assert "profitable" in report.summary.lower()
        ep = mem.get_episode(ep_id)
        # Should NOT have been overwritten
        assert ep.failure_layer == ""

    def test_no_confidence_data_defaults_5050(self, mem, eng):
        """沒有 agent_confidences → 語義/時效各半"""
        ep_id = mem.store(Episode(
            ticker="META", embedding=np.random.randn(32), roi=-0.03,
        ))
        report = eng.diagnose_and_update(
            episode_id=ep_id,
            expected_price=400.0,
            fill_price=400.0,
            close_price=388.0,
            roi=-0.03,
        )
        # With zero slippage and no confidence → 50/50 split
        assert report.semantic.contribution_pct == pytest.approx(50.0, abs=1.0)
        assert report.temporal.contribution_pct == pytest.approx(50.0, abs=1.0)

    def test_report_id_increments(self, mem, eng):
        for i in range(3):
            ep_id = mem.store(Episode(
                ticker="X", embedding=np.random.randn(32), roi=-0.01,
            ))
            report = eng.diagnose_and_update(
                ep_id, 100.0, 100.0, 99.0, roi=-0.01,
            )
        assert report.report_id == "FR-000003"
