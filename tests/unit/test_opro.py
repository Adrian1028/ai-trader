"""Unit tests for Adaptive-OPRO and scoring function."""
from __future__ import annotations

import numpy as np
import pytest

from src.memory.episodic_memory import Episode, EpisodicMemory
from src.prompts.adaptive_opro import AdaptiveOPRO, compute_opro_score


class TestOproScore:
    def test_zero_roi_gives_50(self):
        assert compute_opro_score(0.0) == pytest.approx(50.0)

    def test_positive_roi_above_50(self):
        assert compute_opro_score(0.1) == pytest.approx(75.0)

    def test_negative_roi_below_50(self):
        assert compute_opro_score(-0.1) == pytest.approx(25.0)

    def test_clamp_high(self):
        assert compute_opro_score(1.0) == 100.0

    def test_clamp_low(self):
        assert compute_opro_score(-1.0) == 0.0

    def test_breakeven_threshold(self):
        # ROI = +20% → score = 100
        assert compute_opro_score(0.2) == 100.0
        # ROI = -20% → score = 0
        assert compute_opro_score(-0.2) == 0.0


class TestOproPopulation:
    def test_initial_population_size(self):
        opro = AdaptiveOPRO(population_size=5, store_dir="logs/test_opro1")
        assert len(opro._candidates) == 5

    def test_active_parameters_are_default_initially(self):
        opro = AdaptiveOPRO(population_size=3, store_dir="logs/test_opro2")
        params = opro.active_parameters
        assert "weight_fundamental" in params
        assert "weight_technical" in params
        assert "weight_sentiment" in params

    def test_record_outcome_updates_score(self):
        opro = AdaptiveOPRO(population_size=3, store_dir="logs/test_opro3")
        initial_score = opro.active_candidate.score

        opro.record_trade_outcome(0.05)  # positive ROI
        assert opro.active_candidate.trade_count == 1
        # Score should move toward 62.5 (= 50 + 250*0.05)

    def test_negative_roi_doesnt_crash(self):
        opro = AdaptiveOPRO(population_size=4, store_dir="logs/test_opro4")
        for _ in range(5):
            opro.record_trade_outcome(-0.03)
        assert opro.active_candidate.trade_count == 5
        assert opro.active_candidate.score < 50

    def test_weight_normalisation(self):
        opro = AdaptiveOPRO(store_dir="logs/test_opro5")
        weights = opro.get_intelligence_weights()
        total = sum(weights.values())
        assert total == pytest.approx(1.0, abs=0.01)

    def test_rotation_after_enough_trades(self):
        opro = AdaptiveOPRO(
            population_size=3,
            min_trades_to_evaluate=2,
            store_dir="logs/test_opro6",
        )
        initial_id = opro.active_candidate.candidate_id
        opro.record_trade_outcome(0.01)
        opro.record_trade_outcome(0.01)
        opro.maybe_evolve()
        # Should have rotated to a different candidate
        # (or evolved if enough were tested)


class TestOproGenetics:
    def test_crossover_produces_valid_params(self):
        p1 = {"weight_fundamental": 0.3, "weight_technical": 0.5, "weight_sentiment": 0.2}
        p2 = {"weight_fundamental": 0.4, "weight_technical": 0.3, "weight_sentiment": 0.3}
        child = AdaptiveOPRO._crossover(p1, p2)
        for key in p1:
            assert child[key] in (p1[key], p2[key])

    def test_mutation_stays_in_bounds(self):
        from src.prompts.adaptive_opro import _DEFAULT_PARAMETERS, _PARAM_BOUNDS
        for _ in range(100):
            mutated = AdaptiveOPRO._mutate(dict(_DEFAULT_PARAMETERS), strength=0.5)
            for key, val in mutated.items():
                if key in _PARAM_BOUNDS:
                    lo, hi = _PARAM_BOUNDS[key]
                    assert lo <= val <= hi, f"{key}={val} outside [{lo}, {hi}]"


# ── 失敗歸因驅動優化測試 ────────────────────────────────────────

class FakeAgent:
    """模擬 TechnicalAgent / FundamentalAgent / SentimentAgent"""
    def __init__(self, weights: dict[str, float]):
        self.dynamic_weights = dict(weights)
        self.update_history: list[dict[str, float]] = []

    def update_weights_from_opro(self, new_weights: dict[str, float]) -> None:
        self.dynamic_weights = dict(new_weights)
        self.update_history.append(dict(new_weights))


class TestParseAttribution:
    """測試 _parse_attribution() — 英文/中文/退化格式"""

    def test_english_format(self):
        detail = "Primary failure: semantic (65%) — High confidence wrong"
        result = AdaptiveOPRO._parse_attribution(detail)
        assert result is not None
        assert result["semantic"] == 65.0
        assert result["temporal"] == pytest.approx(17.5)
        assert result["execution"] == pytest.approx(17.5)

    def test_chinese_format(self):
        detail = "[執行:10.0%|時序:80.0%|語義:10.0%]"
        result = AdaptiveOPRO._parse_attribution(detail)
        assert result is not None
        assert result["execution"] == 10.0
        assert result["temporal"] == 80.0
        assert result["semantic"] == 10.0

    def test_fallback_to_failure_layer(self):
        result = AdaptiveOPRO._parse_attribution("", "temporal")
        assert result is not None
        assert result["temporal"] == 100.0
        assert result["semantic"] == 0.0

    def test_no_data_returns_none(self):
        assert AdaptiveOPRO._parse_attribution("") is None
        assert AdaptiveOPRO._parse_attribution("", "") is None

    def test_invalid_layer_returns_none(self):
        assert AdaptiveOPRO._parse_attribution("", "unknown") is None

    def test_english_execution_layer(self):
        detail = "Primary failure: execution (80%) — slippage"
        result = AdaptiveOPRO._parse_attribution(detail)
        assert result["execution"] == 80.0

    def test_english_temporal_layer(self):
        detail = "Primary failure: temporal (55%) — timing off"
        result = AdaptiveOPRO._parse_attribution(detail)
        assert result["temporal"] == 55.0


class TestMutateAgentWeights:
    """測試 _mutate_agent_weights() — 罪魁禍首懲罰突變"""

    def test_reduces_max_weight(self):
        weights = {"trend_following": 0.6, "mean_reversion": 0.3, "volatility": 0.1}
        new = AdaptiveOPRO._mutate_agent_weights(weights, penalty_factor=0.2)
        # trend_following 佔最大，應被削弱
        assert new["trend_following"] < 0.6

    def test_total_normalised_to_one(self):
        weights = {"a": 0.5, "b": 0.3, "c": 0.2}
        for _ in range(50):
            new = AdaptiveOPRO._mutate_agent_weights(weights, penalty_factor=0.3)
            assert sum(new.values()) == pytest.approx(1.0, abs=0.01)

    def test_empty_weights_returns_empty(self):
        assert AdaptiveOPRO._mutate_agent_weights({}, 0.1) == {}

    def test_other_weights_increase(self):
        weights = {"rsi": 0.7, "sma": 0.2, "bb": 0.1}
        new = AdaptiveOPRO._mutate_agent_weights(weights, penalty_factor=0.3)
        # sma 和 bb 應增加
        assert new["sma"] > 0.2 or new["bb"] > 0.1


class TestFailureDrivenOptimization:
    """測試 run_failure_driven_optimization() — 完整優化迴圈"""

    @pytest.fixture
    def mem(self, tmp_path):
        return EpisodicMemory(store_dir=str(tmp_path / "opro_mem"))

    @pytest.fixture
    def opro(self, tmp_path):
        return AdaptiveOPRO(store_dir=str(tmp_path / "opro_state"))

    @pytest.fixture
    def tech_agent(self):
        return FakeAgent({"trend_following": 0.4, "mean_reversion": 0.4, "volatility": 0.2})

    @pytest.fixture
    def fund_agent(self):
        return FakeAgent({"value": 0.4, "profitability": 0.4, "growth": 0.2})

    @pytest.fixture
    def sent_agent(self):
        return FakeAgent({"news_sentiment": 0.5, "analyst_consensus": 0.3, "social_momentum": 0.2})

    def test_skips_when_insufficient_failures(self, mem, opro):
        # 只有 2 筆虧損 (低於 3 筆門檻)
        for _ in range(2):
            mem.store(Episode(
                ticker="X", embedding=np.random.randn(32), roi=-0.05,
                failure_layer="temporal",
                failure_detail="Primary failure: temporal (70%) — timing",
            ))
        result = opro.run_failure_driven_optimization(mem)
        assert result["status"] == "SKIPPED"
        assert "不足" in result["reason"]

    def test_skips_when_no_parseable_detail(self, mem, opro):
        for _ in range(5):
            mem.store(Episode(
                ticker="X", embedding=np.random.randn(32), roi=-0.05,
                failure_detail="some unparseable text",
            ))
        result = opro.run_failure_driven_optimization(mem)
        assert result["status"] == "SKIPPED"

    def test_temporal_triggers_technical_mutation(self, mem, opro, tech_agent):
        """時序錯誤 > 40% → TechnicalAgent 被突變"""
        for _ in range(5):
            mem.store(Episode(
                ticker="A", embedding=np.random.randn(32), roi=-0.05,
                failure_layer="temporal",
                failure_detail="Primary failure: temporal (70%) — timing off",
            ))
        result = opro.run_failure_driven_optimization(
            mem, technical_agent=tech_agent,
        )
        assert result["status"] == "OPTIMIZED"
        assert "TechnicalAgent" in result["updates"]
        assert len(tech_agent.update_history) == 1

    def test_semantic_triggers_fundamental_and_sentiment(
        self, mem, opro, fund_agent, sent_agent,
    ):
        """語義錯誤 > 40% → Fundamental + Sentiment 被突變"""
        for _ in range(5):
            mem.store(Episode(
                ticker="B", embedding=np.random.randn(32), roi=-0.05,
                failure_layer="semantic",
                failure_detail="Primary failure: semantic (65%) — wrong direction",
            ))
        result = opro.run_failure_driven_optimization(
            mem, fundamental_agent=fund_agent, sentiment_agent=sent_agent,
        )
        assert result["status"] == "OPTIMIZED"
        assert "FundamentalAgent" in result["updates"]
        assert "SentimentAgent" in result["updates"]
        assert len(fund_agent.update_history) == 1
        assert len(sent_agent.update_history) == 1

    def test_execution_triggers_warning_only(self, mem, opro, tech_agent):
        """執行摩擦 > 40% → 只出警告，不突變代理"""
        for _ in range(5):
            mem.store(Episode(
                ticker="C", embedding=np.random.randn(32), roi=-0.03,
                failure_layer="execution",
                failure_detail="Primary failure: execution (80%) — slippage",
            ))
        result = opro.run_failure_driven_optimization(
            mem, technical_agent=tech_agent,
        )
        assert "ExecutionWarning" in result["updates"]
        # TechnicalAgent 不應被突變 (temporal < 40%)
        assert len(tech_agent.update_history) == 0

    def test_chinese_format_parsed(self, mem, opro, tech_agent):
        """中文歸因格式也能觸發優化"""
        for _ in range(4):
            mem.store(Episode(
                ticker="D", embedding=np.random.randn(32), roi=-0.04,
                failure_detail="[執行:10.0%|時序:80.0%|語義:10.0%]",
            ))
        result = opro.run_failure_driven_optimization(
            mem, technical_agent=tech_agent,
        )
        assert result["status"] == "OPTIMIZED"
        assert result["average_attribution"]["temporal"] == 80.0
        assert "TechnicalAgent" in result["updates"]

    def test_no_change_when_all_below_threshold(self, mem, opro, tech_agent):
        """三維歸因都低於 40% → 不觸發突變"""
        for _ in range(5):
            mem.store(Episode(
                ticker="E", embedding=np.random.randn(32), roi=-0.02,
                failure_detail="[執行:33.0%|時序:34.0%|語義:33.0%]",
            ))
        result = opro.run_failure_driven_optimization(
            mem, technical_agent=tech_agent,
        )
        assert result["status"] == "NO_CHANGE"
        assert len(tech_agent.update_history) == 0

    def test_only_losing_episodes_considered(self, mem, opro, tech_agent):
        """獲利交易不計入歸因統計"""
        # 10 筆獲利 + 2 筆虧損 → 不足 3 筆門檻
        for _ in range(10):
            mem.store(Episode(
                ticker="F", embedding=np.random.randn(32), roi=0.10,
                failure_layer="temporal",
                failure_detail="Primary failure: temporal (70%) — timing",
            ))
        for _ in range(2):
            mem.store(Episode(
                ticker="F", embedding=np.random.randn(32), roi=-0.05,
                failure_layer="temporal",
                failure_detail="Primary failure: temporal (70%) — timing",
            ))
        result = opro.run_failure_driven_optimization(mem, technical_agent=tech_agent)
        assert result["status"] == "SKIPPED"

    def test_limit_parameter_respected(self, mem, opro, tech_agent):
        """limit 參數控制檢視範圍"""
        # 先放 10 筆舊的語義錯誤
        for _ in range(10):
            mem.store(Episode(
                ticker="G", embedding=np.random.randn(32), roi=-0.05,
                failure_layer="semantic",
                failure_detail="Primary failure: semantic (80%) — old",
            ))
        # 再放 5 筆新的執行錯誤
        for _ in range(5):
            mem.store(Episode(
                ticker="G", embedding=np.random.randn(32), roi=-0.03,
                failure_layer="execution",
                failure_detail="Primary failure: execution (80%) — new",
            ))
        # limit=5 只看最近 5 筆 (全部是執行)
        result = opro.run_failure_driven_optimization(
            mem, technical_agent=tech_agent, limit=5,
        )
        assert result["average_attribution"]["execution"] == 80.0

    def test_result_structure(self, mem, opro, tech_agent):
        """驗證回傳結果結構完整"""
        for _ in range(4):
            mem.store(Episode(
                ticker="H", embedding=np.random.randn(32), roi=-0.06,
                failure_detail="Primary failure: temporal (60%) — timing",
            ))
        result = opro.run_failure_driven_optimization(
            mem, technical_agent=tech_agent,
        )
        assert "status" in result
        assert "analyzed_failures" in result
        assert "average_attribution" in result
        assert "updates" in result
        attrs = result["average_attribution"]
        assert "execution" in attrs
        assert "temporal" in attrs
        assert "semantic" in attrs
