"""
Market Regime Detector
======================
Statistical market regime classification using feature clustering.

Detects 3 primary regimes:
  - BULL:     Positive trend momentum, rising prices, low-moderate volatility
  - BEAR:     Negative trend momentum, falling prices, rising volatility
  - SIDEWAYS: No clear direction, range-bound, mixed signals

Method:
  Uses a multi-feature scoring approach with exponential smoothing.
  No external ML dependencies required (pure NumPy).

Features used:
  1. Return momentum (20d rolling mean of daily returns)
  2. Trend strength (price vs SMA-50 deviation)
  3. Volatility regime (20d rolling std vs long-term std)
  4. Breadth signal (% of recent days positive)

Integration points:
  - IntelligenceOrchestrator: injected into context for all agents
  - Adaptive Kelly: regime affects position sizing
  - OPRO: regime-aware weight optimization
  - EpisodicMemory: tagged on trade records
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    BULL = "BULL"
    BEAR = "BEAR"
    SIDEWAYS = "SIDEWAYS"


@dataclass
class RegimeSnapshot:
    """Complete regime analysis output."""
    regime: MarketRegime = MarketRegime.SIDEWAYS
    confidence: float = 0.0       # 0-1, how strongly the regime is detected
    bull_score: float = 0.0       # raw bull evidence score
    bear_score: float = 0.0       # raw bear evidence score
    sideways_score: float = 0.0   # raw sideways evidence score

    # Underlying feature values
    return_momentum: float = 0.0   # 20d avg daily return
    trend_strength: float = 0.0    # price deviation from SMA50 (%)
    volatility_ratio: float = 0.0  # short-term vol / long-term vol
    positive_day_pct: float = 0.0  # % of recent days with positive returns

    @property
    def summary(self) -> str:
        return (
            f"{self.regime.value} (conf={self.confidence:.2f}) | "
            f"bull={self.bull_score:.2f} bear={self.bear_score:.2f} "
            f"side={self.sideways_score:.2f}"
        )


class RegimeDetector:
    """
    Stateless market regime classifier.

    Call `detect(closes)` with a price array (at least 60 bars)
    to get a RegimeSnapshot.

    Designed for daily bars but works on any timeframe.
    """

    # ── Feature thresholds ────────────────────────────────────────
    # Return momentum: annualized basis
    _BULL_RETURN_THRESHOLD = 0.0003    # ~7.5% annualized
    _BEAR_RETURN_THRESHOLD = -0.0003   # ~-7.5% annualized

    # Trend: price vs SMA-50 deviation (%)
    _STRONG_TREND_PCT = 0.03           # 3% above/below SMA50

    # Volatility ratio: short/long-term std
    _HIGH_VOL_RATIO = 1.3              # 30% above long-term vol = elevated
    _LOW_VOL_RATIO = 0.7               # 30% below = compressed

    # Breadth: % of positive days in lookback
    _BULL_BREADTH = 0.55               # >55% positive days
    _BEAR_BREADTH = 0.45               # <45% positive days

    def detect(self, closes: np.ndarray) -> RegimeSnapshot:
        """
        Classify market regime from closing prices.

        Parameters
        ----------
        closes : np.ndarray
            Array of closing prices (oldest → newest), at least 60 bars.

        Returns
        -------
        RegimeSnapshot with regime classification and confidence.
        """
        snap = RegimeSnapshot()

        if len(closes) < 60:
            logger.debug("RegimeDetector: insufficient data (%d bars)", len(closes))
            return snap  # default SIDEWAYS

        returns = np.diff(np.log(closes))

        # ── Feature extraction ────────────────────────────────────
        # 1. Return momentum (20d exponential weighted)
        short_returns = returns[-20:]
        weights = np.exp(np.linspace(-1, 0, len(short_returns)))
        weights /= weights.sum()
        return_momentum = float(np.average(short_returns, weights=weights))
        snap.return_momentum = return_momentum

        # 2. Trend strength (price vs SMA-50)
        sma_50 = float(np.mean(closes[-50:]))
        current_price = float(closes[-1])
        trend_strength = (current_price - sma_50) / sma_50 if sma_50 > 0 else 0.0
        snap.trend_strength = trend_strength

        # 3. Volatility ratio (20d vol / 60d vol)
        short_vol = float(np.std(returns[-20:], ddof=1)) if len(returns) >= 20 else 0.0
        long_vol = float(np.std(returns[-60:], ddof=1)) if len(returns) >= 60 else short_vol
        vol_ratio = short_vol / long_vol if long_vol > 0 else 1.0
        snap.volatility_ratio = vol_ratio

        # 4. Breadth (% of positive days in last 20)
        positive_pct = float(np.mean(short_returns > 0))
        snap.positive_day_pct = positive_pct

        # ── Scoring ───────────────────────────────────────────────
        bull_score = 0.0
        bear_score = 0.0
        sideways_score = 0.0

        # Return momentum scoring
        if return_momentum > self._BULL_RETURN_THRESHOLD:
            bull_score += 0.30
        elif return_momentum < self._BEAR_RETURN_THRESHOLD:
            bear_score += 0.30
        else:
            sideways_score += 0.20

        # Trend strength scoring
        if trend_strength > self._STRONG_TREND_PCT:
            bull_score += 0.30
        elif trend_strength < -self._STRONG_TREND_PCT:
            bear_score += 0.30
        else:
            sideways_score += 0.25

        # Volatility scoring (bears tend to have high vol)
        if vol_ratio > self._HIGH_VOL_RATIO:
            bear_score += 0.15
            sideways_score += 0.05
        elif vol_ratio < self._LOW_VOL_RATIO:
            bull_score += 0.10  # compressed vol → potential breakout
            sideways_score += 0.15
        else:
            sideways_score += 0.10

        # Breadth scoring
        if positive_pct > self._BULL_BREADTH:
            bull_score += 0.25
        elif positive_pct < self._BEAR_BREADTH:
            bear_score += 0.25
        else:
            sideways_score += 0.20

        snap.bull_score = bull_score
        snap.bear_score = bear_score
        snap.sideways_score = sideways_score

        # ── Classification ────────────────────────────────────────
        scores = {"BULL": bull_score, "BEAR": bear_score, "SIDEWAYS": sideways_score}
        total = sum(scores.values())

        if total > 0:
            # Normalize to get confidence
            best_regime = max(scores, key=scores.get)  # type: ignore[arg-type]
            snap.regime = MarketRegime(best_regime)
            snap.confidence = scores[best_regime] / total
        else:
            snap.regime = MarketRegime.SIDEWAYS
            snap.confidence = 0.33

        logger.info(
            "[RegimeDetector] %s | momentum=%.5f trend=%.3f "
            "vol_ratio=%.2f breadth=%.2f",
            snap.summary, return_momentum, trend_strength,
            vol_ratio, positive_pct,
        )

        return snap

    def detect_from_returns(self, returns: np.ndarray) -> RegimeSnapshot:
        """
        Alternative entry point using log returns directly.
        Constructs synthetic closes from returns for feature calculation.
        """
        if len(returns) < 60:
            return RegimeSnapshot()
        # Reconstruct prices from cumulative returns
        closes = 100.0 * np.exp(np.concatenate([[0.0], np.cumsum(returns)]))
        return self.detect(closes)
