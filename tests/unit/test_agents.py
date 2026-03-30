"""Unit tests for Market Intelligence agents and orchestrator."""
from __future__ import annotations

import asyncio

import pytest

from src.core.base_agent import AnalysisSignal, BaseAgent, SignalDirection


class MockBullishAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__("mock_bull")

    async def analyse(self, context):
        return AnalysisSignal(
            source=self.name,
            direction=SignalDirection.BUY,
            confidence=0.8,
            reasoning="mock bullish",
        )


class MockBearishAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__("mock_bear")

    async def analyse(self, context):
        return AnalysisSignal(
            source=self.name,
            direction=SignalDirection.SELL,
            confidence=0.7,
            reasoning="mock bearish",
        )


class MockNeutralAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__("mock_neutral")

    async def analyse(self, context):
        return AnalysisSignal(
            source=self.name,
            direction=SignalDirection.NEUTRAL,
            confidence=0.3,
            reasoning="mock neutral",
        )


class MockStrongBullAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__("mock_strong_bull")

    async def analyse(self, context):
        return AnalysisSignal(
            source=self.name,
            direction=SignalDirection.STRONG_BUY,
            confidence=0.9,
            reasoning="mock strong bullish",
        )


class MockFailingAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__("mock_fail")

    async def analyse(self, context):
        raise RuntimeError("Simulated failure")


class TestAnalysisSignal:
    def test_weighted_score_buy(self):
        sig = AnalysisSignal(source="test", direction=SignalDirection.BUY, confidence=0.8)
        assert sig.weighted_score == pytest.approx(0.8)

    def test_weighted_score_sell(self):
        sig = AnalysisSignal(source="test", direction=SignalDirection.STRONG_SELL, confidence=0.5)
        assert sig.weighted_score == pytest.approx(-1.0)


class TestSafeAnalyse:
    @pytest.mark.asyncio
    async def test_returns_neutral_on_failure(self):
        agent = MockFailingAgent()
        result = await agent.safe_analyse({"ticker": "TEST"})
        assert result.direction is SignalDirection.NEUTRAL
        assert result.confidence == 0.0


class TestOrchestrator:
    @pytest.mark.asyncio
    async def test_fuses_signals(self):
        from src.agents.intelligence.orchestrator import IntelligenceOrchestrator

        agents = [MockBullishAgent(), MockBearishAgent()]
        weights = {"mock_bull": 0.6, "mock_bear": 0.4}
        orch = IntelligenceOrchestrator(agents, weights=weights)

        view = await orch.evaluate({"isin": "TEST", "ticker": "TEST"})
        assert len(view.signals) == 2
        # Bull weight 0.6 * 0.8 (BUY=1*0.8) + Bear weight 0.4 * -0.7 (SELL=-1*0.7)
        # = (0.48 - 0.28) / 1.0 = 0.20 → likely NEUTRAL or weak BUY
        assert view.fused_direction in (SignalDirection.NEUTRAL, SignalDirection.BUY)

    @pytest.mark.asyncio
    async def test_handles_agent_failure(self):
        from src.agents.intelligence.orchestrator import IntelligenceOrchestrator

        agents = [MockBullishAgent(), MockFailingAgent()]
        orch = IntelligenceOrchestrator(agents)

        view = await orch.evaluate({"isin": "TEST", "ticker": "TEST"})
        assert len(view.signals) == 2
        # Failed agent returns NEUTRAL with confidence 0
        assert any(s.direction is SignalDirection.NEUTRAL for s in view.signals)

    @pytest.mark.asyncio
    async def test_weight_update(self):
        from src.agents.intelligence.orchestrator import IntelligenceOrchestrator

        orch = IntelligenceOrchestrator([MockBullishAgent()])
        orch.update_weights({"mock_bull": 0.99})
        assert orch._weights["mock_bull"] == 0.99


class TestOrchestratorDivergence:
    """測試分歧懲罰邏輯"""

    @pytest.mark.asyncio
    async def test_divergence_detected_when_agents_disagree(self):
        """Bull vs Bear → divergence_detected = True"""
        from src.agents.intelligence.orchestrator import IntelligenceOrchestrator

        agents = [MockBullishAgent(), MockBearishAgent()]
        weights = {"mock_bull": 0.5, "mock_bear": 0.5}
        orch = IntelligenceOrchestrator(agents, weights=weights)

        view = await orch.evaluate({"isin": "TEST", "ticker": "TEST"})
        assert view.divergence_detected is True

    @pytest.mark.asyncio
    async def test_no_divergence_when_agents_agree(self):
        """兩個 Bull → divergence_detected = False"""
        from src.agents.intelligence.orchestrator import IntelligenceOrchestrator

        agents = [MockBullishAgent(), MockStrongBullAgent()]
        weights = {"mock_bull": 0.5, "mock_strong_bull": 0.5}
        orch = IntelligenceOrchestrator(agents, weights=weights)

        view = await orch.evaluate({"isin": "TEST", "ticker": "TEST"})
        assert view.divergence_detected is False

    @pytest.mark.asyncio
    async def test_divergence_reduces_confidence(self):
        """分歧時信心度應低於無分歧時"""
        from src.agents.intelligence.orchestrator import IntelligenceOrchestrator

        # 無分歧場景
        agents_agree = [MockBullishAgent(), MockStrongBullAgent()]
        weights = {"mock_bull": 0.5, "mock_strong_bull": 0.5}
        orch_agree = IntelligenceOrchestrator(agents_agree, weights=weights)
        view_agree = await orch_agree.evaluate({"isin": "T", "ticker": "T"})

        # 有分歧場景
        agents_disagree = [MockBullishAgent(), MockBearishAgent()]
        weights_d = {"mock_bull": 0.5, "mock_bear": 0.5}
        orch_disagree = IntelligenceOrchestrator(agents_disagree, weights=weights_d)
        view_disagree = await orch_disagree.evaluate({"isin": "T", "ticker": "T"})

        assert view_disagree.fused_confidence < view_agree.fused_confidence

    @pytest.mark.asyncio
    async def test_all_agents_fail_returns_neutral(self):
        """所有代理崩潰 → 安全降級 NEUTRAL"""
        from src.agents.intelligence.orchestrator import IntelligenceOrchestrator

        agents = [MockFailingAgent(), MockFailingAgent()]
        orch = IntelligenceOrchestrator(agents)

        view = await orch.evaluate({"isin": "TEST", "ticker": "TEST"})
        assert view.fused_direction == SignalDirection.NEUTRAL
        assert view.fused_confidence == 0.0


class TestMarketView:
    def test_summary_format(self):
        from src.agents.intelligence.orchestrator import MarketView

        view = MarketView(
            isin="US0378331005", ticker="AAPL",
            signals=[
                AnalysisSignal(source="fundamental", direction=SignalDirection.BUY, confidence=0.7),
                AnalysisSignal(source="technical", direction=SignalDirection.SELL, confidence=0.5),
            ],
            fused_score=0.15,
            fused_direction=SignalDirection.NEUTRAL,
            fused_confidence=0.45,
            divergence_detected=True,
        )
        s = view.summary
        assert "AAPL" in s
        assert "NEUTRAL" in s
        assert "DIVERGENCE" in s

    def test_reasons_property(self):
        from src.agents.intelligence.orchestrator import MarketView

        view = MarketView(
            isin="", ticker="TEST",
            signals=[
                AnalysisSignal(source="tech", reasoning="RSI oversold"),
                AnalysisSignal(source="sent", reasoning="Bullish news"),
            ],
        )
        reasons = view.reasons
        assert len(reasons) == 2
        assert any("RSI" in r for r in reasons)

    @pytest.mark.asyncio
    async def test_evaluate_batch(self):
        from src.agents.intelligence.orchestrator import IntelligenceOrchestrator

        agents = [MockBullishAgent()]
        orch = IntelligenceOrchestrator(agents, weights={"mock_bull": 1.0})

        contexts = [
            {"isin": "A", "ticker": "AAPL"},
            {"isin": "B", "ticker": "MSFT"},
            {"isin": "C", "ticker": "GOOG"},
        ]
        views = await orch.evaluate_batch(contexts, max_concurrency=2)
        assert len(views) == 3
        assert all(v.ticker in ("AAPL", "MSFT", "GOOG") for v in views)


class TestTechnicalIndicators:
    def test_rsi_oversold(self):
        import numpy as np
        from src.agents.intelligence.technical import TechnicalAgent

        # Simulate a declining price series
        closes = np.array([100.0 - i * 0.5 for i in range(30)])
        rsi = TechnicalAgent._compute_rsi(closes, 14)
        assert rsi is not None
        assert rsi < 50  # should be low after sustained decline

    def test_rsi_insufficient_data(self):
        import numpy as np
        from src.agents.intelligence.technical import TechnicalAgent

        closes = np.array([100.0, 101.0, 99.0])
        rsi = TechnicalAgent._compute_rsi(closes, 14)
        assert rsi is None

    def test_atr(self):
        import numpy as np
        from src.agents.intelligence.technical import TechnicalAgent

        highs = np.array([102.0 + i * 0.1 for i in range(20)])
        lows = np.array([98.0 + i * 0.1 for i in range(20)])
        closes = np.array([100.0 + i * 0.1 for i in range(20)])
        atr = TechnicalAgent._compute_atr(highs, lows, closes, 14)
        assert atr is not None
        assert atr > 0


class TestTechnicalDynamicWeights:
    """測試 OPRO 動態權重系統"""

    def test_default_weights(self):
        from src.agents.intelligence.technical import TechnicalAgent
        from src.data.providers.alpha_vantage import AlphaVantageProvider
        from src.data.providers.polygon import PolygonProvider

        av = AlphaVantageProvider(api_key="test")
        pg = PolygonProvider(api_key="test")
        agent = TechnicalAgent(av, pg)

        assert agent.dynamic_weights["trend_following"] == pytest.approx(0.4)
        assert agent.dynamic_weights["mean_reversion"] == pytest.approx(0.4)
        assert agent.dynamic_weights["volatility"] == pytest.approx(0.2)

    def test_update_weights_from_opro(self):
        from src.agents.intelligence.technical import TechnicalAgent
        from src.data.providers.alpha_vantage import AlphaVantageProvider
        from src.data.providers.polygon import PolygonProvider

        av = AlphaVantageProvider(api_key="test")
        pg = PolygonProvider(api_key="test")
        agent = TechnicalAgent(av, pg)

        agent.update_weights_from_opro({
            "trend_following": 0.6,
            "mean_reversion": 0.1,
            "volatility": 0.3,
        })
        assert agent.dynamic_weights["trend_following"] == pytest.approx(0.6)
        assert agent.dynamic_weights["mean_reversion"] == pytest.approx(0.1)
        assert agent.dynamic_weights["volatility"] == pytest.approx(0.3)

    def test_partial_weight_update(self):
        """只更新部分權重，其餘保持不變"""
        from src.agents.intelligence.technical import TechnicalAgent
        from src.data.providers.alpha_vantage import AlphaVantageProvider
        from src.data.providers.polygon import PolygonProvider

        av = AlphaVantageProvider(api_key="test")
        pg = PolygonProvider(api_key="test")
        agent = TechnicalAgent(av, pg)

        agent.update_weights_from_opro({"mean_reversion": 0.0})
        assert agent.dynamic_weights["mean_reversion"] == 0.0
        assert agent.dynamic_weights["trend_following"] == pytest.approx(0.4)  # 未變

    def test_weighted_score_rsi_oversold(self):
        """RSI 超賣 → mean_reversion 產生正分數"""
        from src.agents.intelligence.technical import TechnicalAgent
        from src.data.providers.alpha_vantage import AlphaVantageProvider
        from src.data.providers.polygon import PolygonProvider

        av = AlphaVantageProvider(api_key="test")
        pg = PolygonProvider(api_key="test")
        agent = TechnicalAgent(av, pg)

        # 模擬 RSI < 30 + 布林通道底部
        ind = {
            "rsi_14": 25.0,
            "bb_position": 0.05,
            "price_to_sma20_pct": -0.02,
        }
        score = agent._compute_weighted_score(ind)
        assert score > 0, "RSI 超賣 + BB 底部應產生正分數"

    def test_weighted_score_rsi_overbought(self):
        """RSI 超買 → mean_reversion 產生負分數"""
        from src.agents.intelligence.technical import TechnicalAgent
        from src.data.providers.alpha_vantage import AlphaVantageProvider
        from src.data.providers.polygon import PolygonProvider

        av = AlphaVantageProvider(api_key="test")
        pg = PolygonProvider(api_key="test")
        agent = TechnicalAgent(av, pg)

        ind = {
            "rsi_14": 80.0,
            "bb_position": 0.95,
            "price_to_sma20_pct": 0.05,
        }
        score = agent._compute_weighted_score(ind)
        assert score < 0, "RSI 超買 + BB 頂部應產生負分數"

    def test_zero_weight_disables_strategy(self):
        """將某策略權重設為 0，該策略應完全失效"""
        from src.agents.intelligence.technical import TechnicalAgent
        from src.data.providers.alpha_vantage import AlphaVantageProvider
        from src.data.providers.polygon import PolygonProvider

        av = AlphaVantageProvider(api_key="test")
        pg = PolygonProvider(api_key="test")
        agent = TechnicalAgent(av, pg)

        # 停用均值回歸
        agent.update_weights_from_opro({
            "mean_reversion": 0.0,
            "trend_following": 0.0,
            "volatility": 0.0,
        })
        ind = {"rsi_14": 10.0, "bb_position": 0.01}  # 極度超賣
        score = agent._compute_weighted_score(ind)
        assert score == pytest.approx(0.0), "所有權重為 0 時分數應為 0"

    def test_calculate_indicators_minimal_data(self):
        """資料不足 20 根時，部分指標為 None"""
        import numpy as np
        from src.agents.intelligence.technical import TechnicalAgent
        from src.data.providers.alpha_vantage import AlphaVantageProvider
        from src.data.providers.polygon import PolygonProvider

        av = AlphaVantageProvider(api_key="test")
        pg = PolygonProvider(api_key="test")
        agent = TechnicalAgent(av, pg)

        closes = np.array([100.0 + i * 0.1 for i in range(10)])
        highs = closes + 1
        lows = closes - 1
        volumes = np.full(10, 1000.0)

        ind = agent._calculate_indicators(closes, highs, lows, volumes)
        assert ind["rsi_14"] is None  # 只有 10 根，不足 RSI(14)
        assert "bb_width" not in ind  # 不足 20 根

    def test_calculate_indicators_full_data(self):
        """充足資料時，所有指標都應被計算"""
        import numpy as np
        from src.agents.intelligence.technical import TechnicalAgent
        from src.data.providers.alpha_vantage import AlphaVantageProvider
        from src.data.providers.polygon import PolygonProvider

        av = AlphaVantageProvider(api_key="test")
        pg = PolygonProvider(api_key="test")
        agent = TechnicalAgent(av, pg)

        np.random.seed(42)
        n = 250
        closes = 100 + np.cumsum(np.random.randn(n) * 0.5)
        highs = closes + np.abs(np.random.randn(n)) * 0.5
        lows = closes - np.abs(np.random.randn(n)) * 0.5
        volumes = np.random.randint(100_000, 500_000, size=n).astype(float)

        ind = agent._calculate_indicators(closes, highs, lows, volumes)
        assert ind["rsi_14"] is not None
        assert "bb_width" in ind
        assert "bb_position" in ind
        assert ind["atr"] is not None
        assert "sma_50" in ind
        assert "sma_200" in ind
        assert len(ind["_reasons"]) > 0


class TestSentimentKeywords:
    def test_keyword_counting(self):
        from src.agents.intelligence.sentiment import SentimentAgent

        articles = [
            {"headline": "Company beats earnings, stock surges", "summary": "Strong growth reported"},
            {"headline": "Lawsuit filed, shares drop", "summary": "Negative outlook"},
        ]
        pos, neg = SentimentAgent._keyword_sentiment(articles)
        assert pos >= 3   # beat, surge, strong, growth
        assert neg >= 2   # lawsuit, drop, negative
