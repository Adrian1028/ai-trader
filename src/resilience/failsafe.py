"""
FailsafeManager — Data Source Fallback & Degradation Logic
==========================================================

Handles:
  1. Data source fallback chain (primary → backup → stale cache)
  2. Signal staleness detection
  3. Automatic position reduction when data degrades
  4. Emergency all-cash mode on total data failure

Design principle: When in doubt, go to cash. False safety is cheap;
false confidence is catastrophic.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Callable

import pandas as pd

logger = logging.getLogger(__name__)


class DataHealth(str, Enum):
    LIVE = "LIVE"           # Fresh data, <30 min old
    STALE = "STALE"         # Data 30min-24h old (use cache, reduce position)
    DEGRADED = "DEGRADED"   # Data 1-3 days old (half position max)
    DEAD = "DEAD"           # No data >3 days (go to cash)


class SystemMode(str, Enum):
    NORMAL = "NORMAL"           # All systems operational
    REDUCED = "REDUCED"         # Some data stale, reduced exposure
    EMERGENCY_CASH = "EMERGENCY_CASH"  # Total failure, all cash


@dataclass
class DataSourceStatus:
    name: str
    last_success: datetime | None = None
    last_error: str | None = None
    consecutive_failures: int = 0
    health: DataHealth = DataHealth.DEAD


@dataclass
class FailsafeState:
    mode: SystemMode = SystemMode.NORMAL
    max_exposure_pct: float = 1.0  # 1.0 = normal, 0.5 = half, 0.0 = cash
    sources: dict[str, DataSourceStatus] = field(default_factory=dict)
    last_check: datetime | None = None
    reason: str = ""


class FailsafeManager:
    """Manages data source health and automatic degradation."""

    # Staleness thresholds
    STALE_MINUTES = 30
    DEGRADED_HOURS = 24
    DEAD_HOURS = 72

    # Max consecutive failures before marking source dead
    MAX_FAILURES = 5

    def __init__(self, cache_dir: Path | None = None):
        self.state = FailsafeState()
        self.cache_dir = cache_dir or Path("data/failsafe_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._fetchers: dict[str, list[Callable]] = {}

    def register_source(self, name: str, fetchers: list[Callable]):
        """
        Register a data source with its fallback chain.

        fetchers: ordered list [primary, backup1, backup2, ...]
        Each fetcher is a callable() -> pd.Series | pd.DataFrame
        """
        self._fetchers[name] = fetchers
        self.state.sources[name] = DataSourceStatus(name=name)
        logger.info("Registered data source '%s' with %d fetchers", name, len(fetchers))

    def fetch_with_fallback(self, name: str) -> pd.Series | pd.DataFrame | None:
        """
        Try each fetcher in the chain. Return first successful result.
        On total failure, try stale cache. On cache miss, return None.
        """
        if name not in self._fetchers:
            logger.error("Unknown data source: %s", name)
            return None

        status = self.state.sources[name]

        for i, fetcher in enumerate(self._fetchers[name]):
            try:
                result = fetcher()
                if result is not None and (isinstance(result, (pd.Series, pd.DataFrame)) and not result.empty):
                    status.last_success = datetime.utcnow()
                    status.consecutive_failures = 0
                    status.last_error = None
                    status.health = DataHealth.LIVE

                    # Cache successful result
                    self._cache_result(name, result)

                    if i > 0:
                        logger.warning("Source '%s': primary failed, using fallback #%d", name, i)
                    return result
            except Exception as e:
                logger.warning("Source '%s' fetcher #%d failed: %s", name, i, e)
                status.last_error = str(e)

        # All fetchers failed — try cache
        status.consecutive_failures += 1
        logger.error("Source '%s': all %d fetchers failed (consecutive: %d)",
                     name, len(self._fetchers[name]), status.consecutive_failures)

        cached = self._load_cache(name)
        if cached is not None:
            age = self._get_cache_age(name)
            if age and age < timedelta(hours=self.DEGRADED_HOURS):
                status.health = DataHealth.STALE
                logger.warning("Source '%s': using stale cache (%.1f hours old)", name, age.total_seconds() / 3600)
            elif age and age < timedelta(hours=self.DEAD_HOURS):
                status.health = DataHealth.DEGRADED
                logger.warning("Source '%s': using degraded cache (%.1f hours old)", name, age.total_seconds() / 3600)
            else:
                status.health = DataHealth.DEAD
                logger.error("Source '%s': cache too old (%.1f hours), marking DEAD",
                             name, age.total_seconds() / 3600 if age else float("inf"))
                return None
            return cached

        status.health = DataHealth.DEAD
        return None

    def evaluate_system_health(self) -> FailsafeState:
        """
        Evaluate overall system health based on all data sources.
        Returns updated FailsafeState with mode and max exposure.
        """
        self.state.last_check = datetime.utcnow()

        if not self.state.sources:
            self.state.mode = SystemMode.EMERGENCY_CASH
            self.state.max_exposure_pct = 0.0
            self.state.reason = "No data sources registered"
            return self.state

        healths = {name: s.health for name, s in self.state.sources.items()}

        # Critical sources that must be live for trading
        critical = {"vix", "vix3m", "oas"}
        critical_healths = {k: v for k, v in healths.items() if k in critical}

        dead_count = sum(1 for h in critical_healths.values() if h == DataHealth.DEAD)
        degraded_count = sum(1 for h in critical_healths.values() if h == DataHealth.DEGRADED)
        stale_count = sum(1 for h in critical_healths.values() if h == DataHealth.STALE)

        if dead_count > 0:
            # Any critical source dead → emergency cash
            self.state.mode = SystemMode.EMERGENCY_CASH
            self.state.max_exposure_pct = 0.0
            dead_sources = [k for k, v in critical_healths.items() if v == DataHealth.DEAD]
            self.state.reason = f"Critical source(s) DEAD: {', '.join(dead_sources)}"
            logger.critical("EMERGENCY CASH: %s", self.state.reason)

        elif degraded_count > 0:
            # Degraded → half exposure max
            self.state.mode = SystemMode.REDUCED
            self.state.max_exposure_pct = 0.5
            self.state.reason = f"{degraded_count} critical source(s) degraded"
            logger.warning("REDUCED MODE: %s", self.state.reason)

        elif stale_count > 0:
            # Stale → 75% exposure max
            self.state.mode = SystemMode.REDUCED
            self.state.max_exposure_pct = 0.75
            self.state.reason = f"{stale_count} critical source(s) stale"
            logger.warning("REDUCED MODE: %s", self.state.reason)

        else:
            self.state.mode = SystemMode.NORMAL
            self.state.max_exposure_pct = 1.0
            self.state.reason = "All sources healthy"

        return self.state

    def clamp_signal(self, raw_signal: float) -> float:
        """Clamp a trading signal by the current max exposure."""
        return min(raw_signal, self.state.max_exposure_pct)

    def should_force_exit(self) -> bool:
        """True if system should immediately liquidate all positions."""
        return self.state.mode == SystemMode.EMERGENCY_CASH

    def get_status_report(self) -> dict:
        """Human-readable status for logging/Discord."""
        return {
            "mode": self.state.mode.value,
            "max_exposure_pct": self.state.max_exposure_pct,
            "reason": self.state.reason,
            "sources": {
                name: {
                    "health": s.health.value,
                    "last_success": s.last_success.isoformat() if s.last_success else None,
                    "consecutive_failures": s.consecutive_failures,
                    "last_error": s.last_error,
                }
                for name, s in self.state.sources.items()
            },
        }

    # ── Cache helpers ──────────────────────────────────────────────

    def _cache_result(self, name: str, data: pd.Series | pd.DataFrame):
        path = self.cache_dir / f"{name}.parquet"
        try:
            if isinstance(data, pd.Series):
                data = data.to_frame(name=name)
            data.to_parquet(path)
        except Exception as e:
            logger.warning("Failed to cache '%s': %s", name, e)

    def _load_cache(self, name: str) -> pd.DataFrame | None:
        path = self.cache_dir / f"{name}.parquet"
        if not path.exists():
            return None
        try:
            return pd.read_parquet(path)
        except Exception as e:
            logger.warning("Failed to load cache '%s': %s", name, e)
            return None

    def _get_cache_age(self, name: str) -> timedelta | None:
        path = self.cache_dir / f"{name}.parquet"
        if not path.exists():
            return None
        mtime = datetime.fromtimestamp(path.stat().st_mtime)
        return datetime.utcnow() - mtime
