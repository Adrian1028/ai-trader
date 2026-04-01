"""
System Orchestrator
===================
Top-level async pipeline that wires together:
  ISIN Mapper → Intelligence Agents → Decision Fusion → Risk →
  Compliance Guard → Execution → Audit Trail

This is the single entry point for one full trading cycle.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

import numpy as np

from config import T212Config, DataSourceConfig
from config.settings import config
from src.agents.audit.audit_trail import AuditTrailAgent
from src.agents.decision.decision_fusion import DecisionFusionAgent, TradeAction
from src.agents.decision.portfolio_optimizer import (
    BlackLittermanOptimizer,
    DynamicRebalancer,
    RiskParityAllocator,
)
from src.agents.decision.risk import PortfolioRiskState, RiskAgent
from src.agents.decision.stress_tester import StressTester
from src.agents.execution.executor import ExecutionAgent
from src.agents.intelligence.fundamental import FundamentalAgent
from src.agents.intelligence.orchestrator import IntelligenceOrchestrator, MarketView
from src.agents.intelligence.sentiment import SentimentAgent
from src.agents.intelligence.gemini_strategist import GeminiStrategist
from src.agents.intelligence.technical import TechnicalAgent
from src.agents.intelligence.macro_agent import MacroAgent
from src.agents.intelligence.insider_agent import InsiderTradingAgent
from src.agents.intelligence.options_flow_agent import OptionsFlowAgent
from src.agents.intelligence.social_sentiment_agent import SocialSentimentAgent
from src.core.regime_detector import RegimeDetector, RegimeSnapshot
from src.compliance.guard import ComplianceGuard
from src.compliance.watchdog import ComplianceWatchdog
from src.core.client import Trading212Client
from src.core.cognitive_loop import CognitiveLoop
from src.core.virtual_account import VirtualAccountManager, VirtualSubAccount
from src.data.pipelines.isin_mapper import ISINMapper, MARKET_US, MARKET_UK
from src.agents.execution.slippage_model import SlippagePredictor
from src.agents.execution.timing import SmartTiming
from src.agents.execution.order_splitter import OrderSplitter
from src.data.providers.alpha_vantage import AlphaVantageProvider
from src.data.providers.finnhub import FinnhubProvider
from src.data.providers.intrinio import IntrinioProvider
from src.data.providers.macro import MacroDataProvider
from src.data.providers.options_flow import OptionsFlowProvider
from src.data.providers.polygon import PolygonProvider
from src.data.providers.sec_edgar import SECEdgarProvider
from src.data.providers.social_sentiment import SocialSentimentProvider
from src.data.providers.yfinance_provider import YFinanceProvider
from src.memory.counterfactual_replay import CounterfactualReplayEngine
from src.memory.episodic_memory import EpisodicMemory
from src.prompts.adaptive_opro import AdaptiveOPRO

logger = logging.getLogger(__name__)


class TradingSystem:
    """
    Full MAS trading pipeline.

    Usage
    -----
    ```python
    system = TradingSystem.from_env()
    async with system:
        await system.initialise()
        results = await system.run_cycle(["US0378331005"])  # Apple ISIN
    ```
    """

    def __init__(
        self,
        t212_config: T212Config,
        data_config: DataSourceConfig,
        bot_id: str = "default_bot",
        initial_capital: float = 10_000.0,
        account_manager: VirtualAccountManager | None = None,
    ) -> None:
        self._t212_cfg = t212_config
        self._data_cfg = data_config
        self._bot_id = bot_id

        # Core infrastructure
        self._client = Trading212Client(t212_config)
        self._mapper = ISINMapper()
        # Virtual sub-account (資金隔離) — 必須在 ComplianceGuard 之前初始化
        self._account_manager = account_manager or VirtualAccountManager()
        self._virtual_account: VirtualSubAccount = self._account_manager.allocate_account(
            bot_id=bot_id,
            initial_capital=initial_capital,
        )

        # Per-bot compliance guard (與虛擬子帳戶整合)
        self._compliance = ComplianceGuard(
            bot_id=bot_id,
            max_order_pct_of_nav=t212_config.max_order_pct_of_nav,
            kill_switch_max_ops=t212_config.kill_switch_orders_per_second,
            max_pending_per_instrument=t212_config.max_pending_orders_per_instrument,
        )

        # Data providers (original)
        self._av = AlphaVantageProvider(data_config.alpha_vantage_key)
        self._polygon = PolygonProvider(data_config.polygon_key)
        self._finnhub = FinnhubProvider(data_config.finnhub_key)
        self._intrinio = IntrinioProvider(data_config.intrinio_key)
        self._yfinance = YFinanceProvider()  # free, no API key needed

        # Data providers (Phase 6: alternative data)
        self._macro_provider = MacroDataProvider(data_config.fred_api_key)
        self._sec_edgar = SECEdgarProvider()
        self._options_flow_provider = OptionsFlowProvider(data_config.unusual_whales_key)
        self._social_provider = SocialSentimentProvider(
            client_id=data_config.reddit_client_id,
            client_secret=data_config.reddit_client_secret,
        )

        # Intelligence layer (original 4 + Phase 6 new 4)
        self._fundamental = FundamentalAgent(self._av, self._intrinio)
        self._technical = TechnicalAgent(self._av, self._polygon)
        self._sentiment = SentimentAgent(self._finnhub)
        self._gemini = GeminiStrategist()  # reads GEMINI_API_KEY from env
        self._macro_agent = MacroAgent(self._macro_provider)
        self._insider_agent = InsiderTradingAgent(self._sec_edgar)
        self._options_flow_agent = OptionsFlowAgent(self._options_flow_provider)
        self._social_agent = SocialSentimentAgent(self._social_provider)

        self._intelligence = IntelligenceOrchestrator(
            agents=[
                self._fundamental, self._technical, self._sentiment, self._gemini,
                self._macro_agent, self._insider_agent,
                self._options_flow_agent, self._social_agent,
            ],
        )

        # Decision & Risk layer (Phase 7: portfolio optimization + stress testing)
        self._risk = RiskAgent()
        self._bl_optimizer = BlackLittermanOptimizer()
        self._rp_allocator = RiskParityAllocator()
        self._stress_tester = StressTester()
        self._rebalancer = DynamicRebalancer()
        self._decision = DecisionFusionAgent(
            self._risk,
            bl_optimizer=self._bl_optimizer,
            rp_allocator=self._rp_allocator,
            stress_tester=self._stress_tester,
        )

        # Execution layer (Phase 6: with smart timing, slippage model, order splitter)
        self._slippage_model = SlippagePredictor(
            max_acceptable_slippage_bps=config.MAX_SLIPPAGE_BPS,
        )
        self._smart_timing = SmartTiming(
            avoid_open_minutes=config.AVOID_OPEN_MINUTES,
            avoid_close_minutes=config.AVOID_CLOSE_MINUTES,
        )
        self._order_splitter = OrderSplitter(
            split_threshold=config.ORDER_SPLIT_THRESHOLD,
            max_slices=config.ORDER_SPLIT_MAX_SLICES,
            twap_interval_seconds=config.TWAP_INTERVAL_SECONDS,
        )
        self._executor = ExecutionAgent(
            self._client,
            self._compliance,
            virtual_account=self._virtual_account,
            account_manager=self._account_manager,
            slippage_model=self._slippage_model,
            smart_timing=self._smart_timing,
            order_splitter=self._order_splitter,
        )

        # Audit
        self._audit = AuditTrailAgent()

        # Phase 5: Independent compliance watchdog
        self._watchdog = ComplianceWatchdog(self._client, self._compliance)

        # Phase 5b: Market Regime Detector
        self._regime_detector = RegimeDetector()
        self._current_regime: RegimeSnapshot = RegimeSnapshot()  # default SIDEWAYS

        # Phase 4: Cognitive layer
        self._memory = EpisodicMemory()
        self._opro = AdaptiveOPRO()
        self._opro.load()  # restore state from disk if available
        self._replay = CounterfactualReplayEngine()
        self._cognitive = CognitiveLoop(
            audit=self._audit,
            memory=self._memory,
            opro=self._opro,
            replay=self._replay,
            intelligence=self._intelligence,
        )

    @classmethod
    def from_env(
        cls,
        bot_id: str = "default_bot",
        initial_capital: float = 10_000.0,
        account_manager: VirtualAccountManager | None = None,
    ) -> "TradingSystem":
        return cls(
            t212_config=T212Config.from_env(),
            data_config=DataSourceConfig(),
            bot_id=bot_id,
            initial_capital=initial_capital,
            account_manager=account_manager,
        )

    # ── lifecycle ─────────────────────────────────────────────────────

    async def __aenter__(self) -> "TradingSystem":
        await self._client.__aenter__()
        await self._av.__aenter__()
        await self._polygon.__aenter__()
        await self._finnhub.__aenter__()
        await self._intrinio.__aenter__()
        await self._yfinance.__aenter__()
        # Phase 6: alternative data providers
        await self._macro_provider.__aenter__()
        await self._sec_edgar.__aenter__()
        await self._options_flow_provider.__aenter__()
        await self._social_provider.__aenter__()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await asyncio.gather(
            self._client.close(),
            self._av.close(),
            self._polygon.close(),
            self._finnhub.close(),
            self._intrinio.close(),
            self._yfinance.close(),
            self._macro_provider.close(),
            self._sec_edgar.close(),
            self._options_flow_provider.close(),
            self._social_provider.close(),
            return_exceptions=True,
        )

    async def initialise(self) -> None:
        """Load ISIN map and verify connectivity."""
        logger.info(
            "Initialising TradingSystem [env=%s, url=%s]",
            self._t212_cfg.environment.value,
            self._t212_cfg.base_url,
        )

        # Start independent compliance watchdog
        asyncio.create_task(self._watchdog.start())

        # Load instrument map
        instruments = await self._client.exchange_instruments()
        self._mapper.load_instruments(instruments)
        logger.info("Loaded %d instruments into ISIN mapper", self._mapper.count)

        # Verify data provider health (non-blocking)
        health_checks = await asyncio.gather(
            self._av.health_check(),
            self._polygon.health_check(),
            self._finnhub.health_check(),
            self._intrinio.health_check(),
            self._yfinance.health_check(),
            self._macro_provider.health_check(),
            self._sec_edgar.health_check(),
            self._options_flow_provider.health_check(),
            self._social_provider.health_check(),
            return_exceptions=True,
        )
        providers = [
            "AlphaVantage", "Polygon", "Finnhub", "Intrinio", "YFinance",
            "FRED (Macro)", "SEC EDGAR", "Options Flow", "Reddit (Social)",
        ]
        for name, ok in zip(providers, health_checks):
            status = "OK" if ok is True else f"DEGRADED ({ok})"
            logger.info("Provider %s: %s", name, status)

    # ── main trading cycle ────────────────────────────────────────────

    async def run_cycle(self, isins: list[str]) -> list[dict[str, Any]]:
        """
        Execute one full trading cycle for the given ISINs.
        Returns a list of cycle result dicts.
        """
        # Signal liveness to independent watchdog
        self._watchdog.heartbeat()

        if self._compliance.is_killed:
            logger.critical("Kill switch active — cycle aborted")
            return [{"error": "Kill switch active"}]

        # ── 0. Rebalance check (Phase 7) ─────────────────────────────
        # Check if current portfolio drifts from target allocation
        # Target weights will be set by the optimizer in decide_batch()
        if hasattr(self, '_target_weights') and self._target_weights:
            current_weights: dict[str, float] = {}
            v_nav = self._virtual_account.available_cash
            for t, vpos in self._virtual_account.positions.items():
                val = vpos.quantity * vpos.average_price
                v_nav += val
                current_weights[t] = val / v_nav if v_nav > 0 else 0.0
            rebal = self._rebalancer.check_drift(current_weights, self._target_weights)
            if rebal.needs_rebalance:
                logger.info(
                    "[%s] Rebalance triggered: %s", self._bot_id, rebal.summary,
                )

        # ── 1. Fetch real portfolio state & build virtual view ────────
        account_info, portfolio_positions = await asyncio.gather(
            self._client.account_info(),
            self._client.portfolio(),
        )
        real_portfolio = RiskAgent.build_portfolio_state(account_info, portfolio_positions)

        # Cross-validate virtual vs real accounts
        validation = self._account_manager.validate_against_real_account(real_portfolio.total_nav)
        if validation["over_allocated"]:
            logger.warning(
                "[%s] 虛擬帳戶加總超過實體淨值！差距: %.2f",
                self._bot_id, validation["gap"],
            )

        # Use VIRTUAL portfolio state for risk decisions (資金隔離)
        portfolio_state = RiskAgent.build_virtual_portfolio_state(
            self._virtual_account,
        )

        logger.info(
            "[%s] Virtual Portfolio: NAV=%.2f Cash=%.2f Exposure=%.1f%% | "
            "Real NAV=%.2f",
            self._bot_id,
            portfolio_state.total_nav,
            portfolio_state.cash,
            portfolio_state.exposure_pct * 100,
            real_portfolio.total_nav,
        )

        # ── 2. Resolve ISINs to tickers ──────────────────────────────
        contexts = []
        for isin in isins:
            t212_ticker = self._mapper.ticker_for_isin(isin)
            if not t212_ticker:
                logger.warning("ISIN %s not found in mapper — skipping", isin)
                continue
            # market_ticker = standard ticker for external data APIs
            #   US: 'AAPL', 'AMD' (for Polygon/AlphaVantage)
            #   UK: 'BARC.L', 'BP.L' (for YFinance)
            market_ticker = self._mapper.standard_ticker_for_isin(isin)
            market = self._mapper.market_for_isin(isin)
            if not market_ticker:
                logger.warning(
                    "ISIN %s → T212 ticker %s but no standard ticker found — skipping",
                    isin, t212_ticker,
                )
                continue
            contexts.append({
                "isin": isin,
                "ticker": market_ticker,
                "t212_ticker": t212_ticker,
                "market": market,
            })

        if not contexts:
            logger.warning("No valid instruments to evaluate")
            return []

        # ── 3. Intelligence evaluation (parallel) ─────────────────────
        views = await self._intelligence.evaluate_batch(contexts)

        # ── 4. Fetch price data for decision/risk layer ───────────────
        market_map = {c["ticker"]: c.get("market", MARKET_US) for c in contexts}
        price_map, returns_map, atr_map, closes_map = await self._fetch_price_data(
            [c["ticker"] for c in contexts],
            market_map=market_map,
        )

        # ── 4b. Detect market regime from aggregate price data ────────
        self._current_regime = self._detect_regime(closes_map)
        logger.info(
            "[%s] Market Regime: %s", self._bot_id, self._current_regime.summary,
        )

        # Inject regime into each MarketView for downstream agents
        for view in views:
            view.regime = self._current_regime

        # Build position map from current portfolio
        position_map: dict[str, float] = {}
        for pos in portfolio_positions:
            pos_ticker = pos.get("ticker", "")
            isin = self._mapper.isin_for_ticker(pos_ticker) or ""
            if isin:
                position_map[isin] = (
                    float(pos.get("currentPrice", 0)) * float(pos.get("quantity", 0))
                )

        # ── 5. Decision fusion ────────────────────────────────────────
        proposals = self._decision.decide_batch(
            views=views,
            portfolio=portfolio_state,
            price_map=price_map,
            returns_map=returns_map,
            atr_map=atr_map,
            position_map=position_map,
        )

        # ── 6. Execute actionable proposals ───────────────────────────
        results: list[dict[str, Any]] = []
        for proposal, view in zip(proposals, views):
            if not proposal.is_actionable:
                results.append({
                    "ticker": proposal.ticker,
                    "action": proposal.action.name,
                    "reasoning": proposal.reasoning,
                })
                continue

            # Count pending orders for this instrument
            pending_count = 0  # TODO: track from order cache

            ticket = await self._executor.execute(
                proposal=proposal,
                portfolio=portfolio_state,
                pending_count_for_instrument=pending_count,
            )

            # ── 7. Audit trail ────────────────────────────────────────
            record = self._audit.record_trade(view, proposal, ticket)

            results.append({
                "ticker": proposal.ticker,
                "action": proposal.action.name,
                "ticket_id": ticket.ticket_id,
                "order_status": ticket.status.name,
                "quantity": proposal.quantity,
                "value": proposal.estimated_value,
                "audit_id": record.record_id,
                "reasoning": proposal.reasoning,
            })

        return results

    # ── internal helpers ──────────────────────────────────────────────

    async def _fetch_price_data(
        self, tickers: list[str], market_map: dict[str, str] | None = None,
    ) -> tuple[dict[str, float], dict[str, np.ndarray], dict[str, float | None], dict[str, np.ndarray]]:
        """Fetch current prices, historical returns, ATR, and raw closes for each ticker.

        Parameters
        ----------
        tickers : list of ticker strings
        market_map : optional dict mapping ticker → market ('US' or 'UK').
            If provided, UK tickers use YFinance; US tickers use Polygon.
            If not provided, all tickers use Polygon (backward compatible).
        """
        price_map: dict[str, float] = {}
        returns_map: dict[str, np.ndarray] = {}
        atr_map: dict[str, float | None] = {}
        closes_map: dict[str, np.ndarray] = {}
        _market_map = market_map or {}

        async def _fetch_one(ticker: str) -> None:
            try:
                # YFinance is the primary data source for ALL markets.
                # It's free, unlimited, and supports US + UK + global stocks.
                # Polygon (5 req/min free tier) is too slow for 20+ stocks.
                bars = await self._yfinance.daily_bars(ticker, days=365)

                if not bars:
                    return

                closes = np.array([b.close for b in bars], dtype=np.float64)
                highs = np.array([b.high for b in bars], dtype=np.float64)
                lows = np.array([b.low for b in bars], dtype=np.float64)

                price_map[ticker] = closes[-1]
                returns_map[ticker] = np.diff(np.log(closes))
                closes_map[ticker] = closes

                # ATR(14)
                if len(closes) >= 15:
                    atr = TechnicalAgent._compute_atr(highs, lows, closes, 14)
                    atr_map[ticker] = atr

            except Exception:
                logger.warning("Failed to fetch price data for %s", ticker)

        await asyncio.gather(*[_fetch_one(t) for t in tickers])
        return price_map, returns_map, atr_map, closes_map

    # ── regime detection ────────────────────────────────────────────────

    def _detect_regime(self, closes_map: dict[str, np.ndarray]) -> RegimeSnapshot:
        """
        Detect market regime from aggregate price data.

        Strategy: use the ticker with the most data as the representative
        series. If multiple tickers available, average their regime scores
        for a more robust estimate.
        """
        if not closes_map:
            return RegimeSnapshot()

        # Use all tickers and average regime scores for robustness
        snapshots = []
        for ticker, closes in closes_map.items():
            if len(closes) >= 60:
                snap = self._regime_detector.detect(closes)
                snapshots.append(snap)

        if not snapshots:
            return RegimeSnapshot()

        if len(snapshots) == 1:
            return snapshots[0]

        # Average across tickers for a market-wide regime view
        avg_bull = sum(s.bull_score for s in snapshots) / len(snapshots)
        avg_bear = sum(s.bear_score for s in snapshots) / len(snapshots)
        avg_side = sum(s.sideways_score for s in snapshots) / len(snapshots)
        avg_momentum = sum(s.return_momentum for s in snapshots) / len(snapshots)
        avg_trend = sum(s.trend_strength for s in snapshots) / len(snapshots)
        avg_vol = sum(s.volatility_ratio for s in snapshots) / len(snapshots)
        avg_breadth = sum(s.positive_day_pct for s in snapshots) / len(snapshots)

        scores = {"BULL": avg_bull, "BEAR": avg_bear, "SIDEWAYS": avg_side}
        total = sum(scores.values())
        best = max(scores, key=scores.get)  # type: ignore[arg-type]

        from src.core.regime_detector import MarketRegime
        return RegimeSnapshot(
            regime=MarketRegime(best),
            confidence=scores[best] / total if total > 0 else 0.33,
            bull_score=avg_bull,
            bear_score=avg_bear,
            sideways_score=avg_side,
            return_momentum=avg_momentum,
            trend_strength=avg_trend,
            volatility_ratio=avg_vol,
            positive_day_pct=avg_breadth,
        )

    # ── accessors ─────────────────────────────────────────────────────

    # ── post-cycle reflection ────────────────────────────────────────

    async def reflect(
        self,
        record_ids: list[str],
        price_map: dict[str, np.ndarray] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Run cognitive reflection on completed trades.
        Call this after trades have been filled and outcomes recorded.
        """
        results = []
        for rid in record_ids:
            record = self._audit._find(rid)
            if record is None or record.realised_pnl is None:
                continue

            features = {
                "fused_score": record.fused_score,
                "fused_confidence": record.fused_confidence,
            }
            prices = price_map.get(record.ticker) if price_map else None

            result = await self._cognitive.reflect_on_trade(
                record=record,
                market_features=features,
                price_series=prices,
            )
            results.append(result)

        return results

    def learning_summary(self) -> dict[str, Any]:
        """Get a summary of what the system has learned so far."""
        return self._cognitive.get_learning_summary()

    # ── accessors ─────────────────────────────────────────────────────

    @property
    def audit(self) -> AuditTrailAgent:
        return self._audit

    @property
    def compliance(self) -> ComplianceGuard:
        return self._compliance

    @property
    def intelligence(self) -> IntelligenceOrchestrator:
        return self._intelligence

    @property
    def mapper(self) -> ISINMapper:
        return self._mapper

    @property
    def cognitive(self) -> CognitiveLoop:
        return self._cognitive

    @property
    def opro(self) -> AdaptiveOPRO:
        return self._opro

    @property
    def watchdog(self) -> ComplianceWatchdog:
        return self._watchdog

    @property
    def virtual_account(self) -> VirtualSubAccount:
        return self._virtual_account

    @property
    def account_manager(self) -> VirtualAccountManager:
        return self._account_manager

    @property
    def bot_id(self) -> str:
        return self._bot_id

    @property
    def regime(self) -> RegimeSnapshot:
        return self._current_regime

    @property
    def regime_detector(self) -> RegimeDetector:
        return self._regime_detector

    @property
    def stress_tester(self) -> StressTester:
        return self._stress_tester

    @property
    def rebalancer(self) -> DynamicRebalancer:
        return self._rebalancer
