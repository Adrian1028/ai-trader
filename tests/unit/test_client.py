"""Unit tests for Trading212Client rate limiting and environment switching."""
from __future__ import annotations

import pytest

from config import T212Config, Environment
from src.core.rate_limiter import ExponentialBackoff, RateLimitGuard, RateLimitState


class TestExponentialBackoff:
    def test_initial_delay_equals_base(self):
        bo = ExponentialBackoff(base=1.0, multiplier=2.0, ceiling=60.0)
        assert bo.delay == 1.0

    def test_escalation_doubles(self):
        bo = ExponentialBackoff(base=1.0, multiplier=2.0, ceiling=60.0)
        assert bo.escalate() == 1.0   # attempt 0
        assert bo.escalate() == 2.0   # attempt 1
        assert bo.escalate() == 4.0   # attempt 2

    def test_ceiling_respected(self):
        bo = ExponentialBackoff(base=1.0, multiplier=2.0, ceiling=5.0)
        for _ in range(20):
            bo.escalate()
        assert bo.delay <= 5.0

    def test_reset_returns_to_base(self):
        bo = ExponentialBackoff(base=1.0, multiplier=2.0, ceiling=60.0)
        bo.escalate()
        bo.escalate()
        bo.reset()
        assert bo.delay == 1.0


class TestRateLimitState:
    def test_update_from_headers(self):
        state = RateLimitState()
        state.update_from_headers({
            "x-ratelimit-remaining": "3",
            "x-ratelimit-reset": "1700000000.0",
        })
        assert state.remaining == 3
        assert state.reset_timestamp == 1700000000.0


class TestEnvironmentSwitching:
    def test_demo_url(self):
        cfg = T212Config(environment=Environment.DEMO, api_key="test")
        assert "demo.trading212.com" in cfg.base_url

    def test_live_url(self):
        cfg = T212Config(environment=Environment.LIVE, api_key="test")
        assert "live.trading212.com" in cfg.base_url
