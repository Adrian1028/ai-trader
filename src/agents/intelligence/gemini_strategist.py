"""
Gemini Strategist Agent — LLM semantic reasoning agent
=======================================================
The fourth intelligence agent in the MAS pipeline.  Unlike the other
three rule-based agents, this one calls the Gemini API to perform
semantic reasoning over raw market data, news, and fundamentals.

Architecture
------------
  Other agents produce signals based on fixed rules (RSI thresholds,
  P/E ratios, sentiment scores).  GeminiStrategist receives the same
  raw data and applies *reasoning*:

    "Revenue growth is 8% but the P/E is 35 — overvalued relative to
     growth.  Meanwhile RSI is 72 and approaching the upper Bollinger
     Band.  The recent news about supply chain disruption adds
     downside risk.  Direction: SELL, confidence: 0.65."

  This kind of multi-factor semantic judgement is impossible with
  pure rule-based logic.

Cost Control
------------
  - Uses gemini-2.5-flash by default (FREE tier: 10 RPM)
  - ~2K input + ~300 output per call = $0 within free tier
  - 20 stocks x 28 scans/day = 560 calls/day (within free limits)
  - Gracefully degrades to NEUTRAL if API key is missing or call fails
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

from src.core.base_agent import AnalysisSignal, BaseAgent, SignalDirection

logger = logging.getLogger(__name__)

# System instruction for Gemini
_SYSTEM_INSTRUCTION = """\
You are a senior quantitative analyst at a hedge fund.
You receive market data for a single stock and must decide: BUY, SELL, or NEUTRAL.

Rules:
1. Respond ONLY with valid JSON — no markdown, no explanation outside the JSON.
2. Schema:
   {"direction": "BUY"|"SELL"|"NEUTRAL", "confidence": 0.0-1.0, "reasoning": "one sentence"}
3. confidence reflects how certain you are (0.5 = coin flip, 0.9 = very confident).
4. Consider ALL data holistically: technicals, fundamentals, news sentiment, price action.
5. Be contrarian when data conflicts — if technicals say BUY but fundamentals scream overvalued, lower confidence or go NEUTRAL.
6. Never hallucinate data you were not given.\
"""


class GeminiStrategist(BaseAgent):
    """
    LLM-powered intelligence agent using Google Gemini API.

    Analyses raw market data through semantic reasoning rather than
    fixed rules.  Designed as a drop-in fourth agent for the
    IntelligenceOrchestrator.

    Parameters
    ----------
    model : str
        Gemini model ID (default: gemini-2.5-flash for free tier).
    api_key : str | None
        Google AI API key.  Falls back to GEMINI_API_KEY env var.
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
    ) -> None:
        super().__init__("gemini_strategist")
        self._model = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self._api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self._client: Any = None  # lazy init

    def _get_client(self) -> Any:
        """Lazy-initialise the Gemini client."""
        if self._client is None:
            try:
                from google import genai
                self._client = genai.Client(api_key=self._api_key)
                logger.info(
                    "GeminiStrategist initialised (model=%s)", self._model,
                )
            except Exception as exc:
                logger.warning("Failed to init Gemini client: %s", exc)
                raise
        return self._client

    # -- core interface ------------------------------------------------

    async def analyse(self, context: dict[str, Any]) -> AnalysisSignal:
        """
        Call Gemini to analyse a single stock.

        The context dict is the same one passed to all agents by the
        Orchestrator.  We extract whatever data is available and build
        a concise prompt for Gemini.
        """
        if not self._api_key:
            return self._neutral("No GEMINI_API_KEY configured")

        ticker = context.get("ticker", "UNKNOWN")

        # Build a data summary from whatever the context contains
        prompt = self._build_prompt(context)

        try:
            client = self._get_client()

            # Call Gemini API
            response = client.models.generate_content(
                model=self._model,
                contents=prompt,
                config={
                    "system_instruction": _SYSTEM_INSTRUCTION,
                    "max_output_tokens": 256,
                    "temperature": 0.3,
                    "response_mime_type": "application/json",
                },
            )

            # Parse the JSON response
            text = response.text or ""

            # Extract token usage
            input_tokens = 0
            output_tokens = 0
            if response.usage_metadata:
                input_tokens = getattr(response.usage_metadata, "prompt_token_count", 0) or 0
                output_tokens = getattr(response.usage_metadata, "candidates_token_count", 0) or 0

            result = json.loads(text)

            direction_str = result.get("direction", "NEUTRAL").upper()
            direction_map = {
                "STRONG_BUY": SignalDirection.STRONG_BUY,
                "BUY": SignalDirection.BUY,
                "NEUTRAL": SignalDirection.NEUTRAL,
                "SELL": SignalDirection.SELL,
                "STRONG_SELL": SignalDirection.STRONG_SELL,
            }
            direction = direction_map.get(direction_str, SignalDirection.NEUTRAL)
            confidence = max(0.0, min(1.0, float(result.get("confidence", 0.5))))
            reasoning = result.get("reasoning", "No reasoning provided")

            logger.info(
                "[Gemini] %s -> %s (conf=%.2f) | %s",
                ticker, direction.name, confidence, reasoning,
            )

            return AnalysisSignal(
                source=self.name,
                direction=direction,
                confidence=confidence,
                reasoning=reasoning,
                data={
                    "model": self._model,
                    "raw_response": text,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                },
            )

        except json.JSONDecodeError as exc:
            logger.warning("[Gemini] %s: invalid JSON response: %s", ticker, exc)
            return self._neutral(f"Invalid JSON from Gemini: {exc}")

        except Exception as exc:
            logger.warning("[Gemini] %s: API call failed: %s", ticker, exc)
            return self._neutral(f"Gemini API error: {exc}")

    # -- prompt construction -------------------------------------------

    def _build_prompt(self, context: dict[str, Any]) -> str:
        """Build a concise data summary prompt for Gemini."""
        ticker = context.get("ticker", "?")
        sections = [f"## {ticker} — Market Analysis Request\n"]

        # Price data (if available from prior agents or feeder)
        price = context.get("current_price")
        if price:
            sections.append(f"Current Price: ${price:.2f}")

        # Technical data (injected by orchestrator or prior analysis)
        tech = context.get("technical_data", {})
        if tech:
            parts = []
            for key in ("rsi", "atr_pct", "bb_position", "bb_width",
                        "sma_20", "sma_50", "price"):
                if key in tech:
                    parts.append(f"{key}={tech[key]:.2f}" if isinstance(tech[key], float) else f"{key}={tech[key]}")
            if parts:
                sections.append(f"Technicals: {', '.join(parts)}")

        # Fundamental data
        fund = context.get("fundamental_data", {})
        if fund:
            parts = []
            for key in ("PERatio", "ForwardPE", "ReturnOnEquityTTM",
                        "ProfitMargin", "QuarterlyRevenueGrowthYOY",
                        "QuarterlyEarningsGrowthYOY", "Beta"):
                if key in fund:
                    parts.append(f"{key}={fund[key]}")
            if parts:
                sections.append(f"Fundamentals: {', '.join(parts)}")

        # News/sentiment data
        news = context.get("news_data", [])
        if news:
            headlines = [n.get("headline", "") for n in news[:3]]
            if any(headlines):
                sections.append(f"Recent Headlines: {' | '.join(h for h in headlines if h)}")

        sentiment = context.get("sentiment_data", {})
        if sentiment:
            sections.append(f"Sentiment: {sentiment}")

        # Other agent signals (if orchestrator provides them)
        prior_signals = context.get("prior_signals", [])
        if prior_signals:
            sig_summary = []
            for sig in prior_signals:
                sig_summary.append(
                    f"{sig.get('source', '?')}: {sig.get('direction', '?')} "
                    f"(conf={sig.get('confidence', 0):.2f})"
                )
            sections.append(f"Other Agents: {' | '.join(sig_summary)}")

        sections.append("\nGive your analysis as JSON.")
        return "\n".join(sections)

    # -- helpers -------------------------------------------------------

    def _neutral(self, reason: str) -> AnalysisSignal:
        """Return a NEUTRAL signal with zero confidence."""
        return AnalysisSignal(
            source=self.name,
            direction=SignalDirection.NEUTRAL,
            confidence=0.0,
            reasoning=reason,
            data={},
        )
