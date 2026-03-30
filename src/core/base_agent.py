"""
Base Agent Protocol
===================
All MAS agents implement this interface. The orchestrator treats every
agent as an async callable that consumes a context dict and produces
an AnalysisSignal.
"""
from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SignalDirection(Enum):
    STRONG_BUY = 2
    BUY = 1
    NEUTRAL = 0
    SELL = -1
    STRONG_SELL = -2


@dataclass
class AnalysisSignal:
    """Standardised output vector from any intelligence agent."""
    source: str                          # agent name
    direction: SignalDirection = SignalDirection.NEUTRAL
    confidence: float = 0.0              # 0.0 – 1.0
    reasoning: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    @property
    def weighted_score(self) -> float:
        return self.direction.value * self.confidence


class BaseAgent(ABC):
    """Abstract base for every agent in the MAS hierarchy."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.logger = logging.getLogger(f"agent.{name}")

    @abstractmethod
    async def analyse(self, context: dict[str, Any]) -> AnalysisSignal:
        """
        Run the agent's analysis pipeline.

        Parameters
        ----------
        context : dict
            Must contain at minimum:
              - "isin": str
              - "ticker": str  (external symbol for data providers)
              - "current_price": float (optional, best-effort)

        Returns
        -------
        AnalysisSignal with the agent's verdict.
        """

    async def safe_analyse(self, context: dict[str, Any]) -> AnalysisSignal:
        """Wrapper that catches exceptions and returns a NEUTRAL signal."""
        try:
            return await self.analyse(context)
        except Exception:
            self.logger.exception("Agent %s failed — returning NEUTRAL", self.name)
            return AnalysisSignal(
                source=self.name,
                direction=SignalDirection.NEUTRAL,
                confidence=0.0,
                reasoning=f"Agent {self.name} encountered an error.",
            )
