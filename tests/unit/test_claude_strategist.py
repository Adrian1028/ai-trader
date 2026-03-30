"""
Unit tests for ClaudeStrategist agent.
"""
from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock, patch

import pytest

from src.agents.intelligence.claude_strategist import ClaudeStrategist
from src.core.base_agent import SignalDirection


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class TestClaudeStrategistInit:
    def test_default_model(self):
        agent = ClaudeStrategist()
        assert agent._model == "claude-haiku-4-5"

    def test_custom_model(self):
        agent = ClaudeStrategist(model="claude-sonnet-4-6")
        assert agent._model == "claude-sonnet-4-6"

    def test_name(self):
        agent = ClaudeStrategist()
        assert agent.name == "claude_strategist"


class TestNoApiKey:
    def test_returns_neutral_without_key(self):
        agent = ClaudeStrategist(api_key="")
        result = run(agent.analyse({"ticker": "AAPL", "isin": "US0378331005"}))
        assert result.direction == SignalDirection.NEUTRAL
        assert result.confidence == 0.0
        assert "No ANTHROPIC_API_KEY" in result.reasoning


class TestBuildPrompt:
    def test_minimal_context(self):
        agent = ClaudeStrategist()
        prompt = agent._build_prompt({"ticker": "AAPL"})
        assert "AAPL" in prompt
        assert "JSON" in prompt

    def test_with_technical_data(self):
        agent = ClaudeStrategist()
        ctx = {
            "ticker": "AAPL",
            "technical_data": {"rsi": 72.5, "atr_pct": 2.1},
        }
        prompt = agent._build_prompt(ctx)
        assert "rsi=72.50" in prompt
        assert "atr_pct=2.10" in prompt

    def test_with_fundamental_data(self):
        agent = ClaudeStrategist()
        ctx = {
            "ticker": "MSFT",
            "fundamental_data": {"PERatio": "35.2", "Beta": "1.1"},
        }
        prompt = agent._build_prompt(ctx)
        assert "PERatio=35.2" in prompt

    def test_with_news(self):
        agent = ClaudeStrategist()
        ctx = {
            "ticker": "TSLA",
            "news_data": [
                {"headline": "Tesla beats Q4 estimates"},
                {"headline": "EV demand slowing"},
            ],
        }
        prompt = agent._build_prompt(ctx)
        assert "Tesla beats Q4" in prompt
        assert "EV demand" in prompt

    def test_with_prior_signals(self):
        agent = ClaudeStrategist()
        ctx = {
            "ticker": "NVDA",
            "prior_signals": [
                {"source": "technical", "direction": "BUY", "confidence": 0.8},
                {"source": "fundamental", "direction": "SELL", "confidence": 0.6},
            ],
        }
        prompt = agent._build_prompt(ctx)
        assert "technical: BUY" in prompt
        assert "fundamental: SELL" in prompt


class TestResponseParsing:
    def _make_mock_response(self, text: str, input_tokens=100, output_tokens=50):
        block = MagicMock()
        block.type = "text"
        block.text = text
        response = MagicMock()
        response.content = [block]
        response.usage = MagicMock()
        response.usage.input_tokens = input_tokens
        response.usage.output_tokens = output_tokens
        return response

    def test_buy_signal(self):
        agent = ClaudeStrategist(api_key="test-key")
        resp = self._make_mock_response(
            json.dumps({"direction": "BUY", "confidence": 0.75, "reasoning": "Strong momentum"})
        )
        with patch.object(agent, "_get_client") as mock_client:
            mock_client.return_value.messages.create.return_value = resp
            result = run(agent.analyse({"ticker": "AAPL"}))

        assert result.direction == SignalDirection.BUY
        assert result.confidence == 0.75
        assert "Strong momentum" in result.reasoning

    def test_sell_signal(self):
        agent = ClaudeStrategist(api_key="test-key")
        resp = self._make_mock_response(
            json.dumps({"direction": "SELL", "confidence": 0.6, "reasoning": "Overvalued"})
        )
        with patch.object(agent, "_get_client") as mock_client:
            mock_client.return_value.messages.create.return_value = resp
            result = run(agent.analyse({"ticker": "TSLA"}))

        assert result.direction == SignalDirection.SELL
        assert result.confidence == 0.6

    def test_invalid_json_returns_neutral(self):
        agent = ClaudeStrategist(api_key="test-key")
        resp = self._make_mock_response("This is not JSON")
        with patch.object(agent, "_get_client") as mock_client:
            mock_client.return_value.messages.create.return_value = resp
            result = run(agent.analyse({"ticker": "AAPL"}))

        assert result.direction == SignalDirection.NEUTRAL
        assert result.confidence == 0.0

    def test_api_error_returns_neutral(self):
        agent = ClaudeStrategist(api_key="test-key")
        with patch.object(agent, "_get_client") as mock_client:
            mock_client.return_value.messages.create.side_effect = RuntimeError("timeout")
            result = run(agent.analyse({"ticker": "AAPL"}))

        assert result.direction == SignalDirection.NEUTRAL
        assert "Claude API error" in result.reasoning

    def test_confidence_clamped(self):
        agent = ClaudeStrategist(api_key="test-key")
        resp = self._make_mock_response(
            json.dumps({"direction": "BUY", "confidence": 1.5, "reasoning": "test"})
        )
        with patch.object(agent, "_get_client") as mock_client:
            mock_client.return_value.messages.create.return_value = resp
            result = run(agent.analyse({"ticker": "AAPL"}))

        assert result.confidence == 1.0

    def test_token_usage_in_data(self):
        agent = ClaudeStrategist(api_key="test-key")
        resp = self._make_mock_response(
            json.dumps({"direction": "NEUTRAL", "confidence": 0.5, "reasoning": "flat"}),
            input_tokens=200,
            output_tokens=80,
        )
        with patch.object(agent, "_get_client") as mock_client:
            mock_client.return_value.messages.create.return_value = resp
            result = run(agent.analyse({"ticker": "AAPL"}))

        assert result.data["input_tokens"] == 200
        assert result.data["output_tokens"] == 80
        assert result.data["model"] == "claude-haiku-4-5"
