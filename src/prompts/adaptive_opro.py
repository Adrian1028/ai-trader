"""
Adaptive-OPRO: Dynamic Prompt & Weight Optimiser
=================================================
Implements a self-optimising feedback loop that tunes:
  1. Intelligence agent weights (fundamental / technical / sentiment)
  2. Decision thresholds (min confidence, buy/sell score gates)
  3. Risk parameters (ATR multipliers, Kelly scaling)

Scoring function:
  s = clip(0, 100, 50 + 250 * ROI)

When ROI is negative, the system:
  - Flags the current reasoning path as failing
  - Generates alternative parameter candidates
  - Evaluates candidates via historical replay
  - Promotes the best candidate to active configuration

This is a BANDIT-style optimiser, not an LLM prompt rewriter.
The "prompts" are the numeric hyperparameters that configure agent behaviour.
"""
from __future__ import annotations

import copy
import json
import logging
import math
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ParameterCandidate:
    """A candidate configuration being evaluated."""
    candidate_id: str = ""
    parameters: dict[str, float] = field(default_factory=dict)
    score: float = 50.0           # initial neutral score
    trade_count: int = 0
    cumulative_roi: float = 0.0
    created_at: float = field(default_factory=time.time)
    active: bool = False

    @property
    def avg_roi(self) -> float:
        return self.cumulative_roi / self.trade_count if self.trade_count > 0 else 0.0


# The parameter space that OPRO optimises
_DEFAULT_PARAMETERS = {
    # Intelligence weights
    "weight_fundamental": 0.35,
    "weight_technical": 0.40,
    "weight_sentiment": 0.25,

    # Decision thresholds
    "min_confidence_to_trade": 0.30,
    "min_buy_score": 0.30,
    "max_sell_score": -0.30,

    # Risk tuning
    "atr_stop_multiplier": 2.0,
    "atr_tp_multiplier": 3.0,
    "half_kelly_scaling": 0.50,

    # Scoring sensitivities
    "roi_score_multiplier": 250.0,
    "roi_score_offset": 50.0,
}

# Bounds for each parameter
_PARAM_BOUNDS: dict[str, tuple[float, float]] = {
    "weight_fundamental": (0.05, 0.60),
    "weight_technical": (0.10, 0.70),
    "weight_sentiment": (0.05, 0.50),
    "min_confidence_to_trade": (0.10, 0.60),
    "min_buy_score": (0.10, 0.80),
    "max_sell_score": (-0.80, -0.10),
    "atr_stop_multiplier": (1.0, 4.0),
    "atr_tp_multiplier": (1.5, 6.0),
    "half_kelly_scaling": (0.20, 0.80),
    "roi_score_multiplier": (100.0, 500.0),
    "roi_score_offset": (30.0, 70.0),
}


def compute_opro_score(roi: float) -> float:
    """
    s = clip(0, 100, 50 + 250 * ROI)
    Maps ROI to a 0-100 score. ROI=0 → 50, ROI=+20% → 100, ROI=-20% → 0.
    """
    raw = 50.0 + 250.0 * roi
    return max(0.0, min(100.0, raw))


class AdaptiveOPRO:
    """
    Bandit-style hyperparameter optimiser.

    Maintains a population of parameter candidates, scores them
    based on live trading performance, and promotes the best
    to active configuration.
    """

    def __init__(
        self,
        population_size: int = 8,
        mutation_rate: float = 0.15,
        elite_count: int = 2,
        min_trades_to_evaluate: int = 5,
        store_dir: str = "logs/opro",
    ) -> None:
        self._pop_size = population_size
        self._mutation_rate = mutation_rate
        self._elite_count = elite_count
        self._min_trades = min_trades_to_evaluate
        self._store_dir = Path(store_dir)
        self._store_dir.mkdir(parents=True, exist_ok=True)

        self._candidates: list[ParameterCandidate] = []
        self._active_idx: int = 0
        self._generation: int = 0
        self._history: list[dict[str, Any]] = []

        self._initialise_population()

    # ── core interface ────────────────────────────────────────────────

    @property
    def active_parameters(self) -> dict[str, float]:
        """Return the currently active parameter configuration."""
        return self._candidates[self._active_idx].parameters

    @property
    def active_candidate(self) -> ParameterCandidate:
        return self._candidates[self._active_idx]

    def record_trade_outcome(self, roi: float, mar_penalty: float = 0.0) -> float:
        """
        Feed a trade outcome into the active candidate.
        Returns the OPRO score for this trade.

        Parameters
        ----------
        roi : realized return on investment
        mar_penalty : MAR compliance penalty from ComplianceGuard.compute_mar_penalty()
                      Range: 0 to -100. Heavily penalises high cancel rates,
                      excessive amendments, and kill switch triggers.

        When ROI is negative, triggers alternative reasoning path generation.
        """
        base_score = compute_opro_score(roi)
        # Inject MAR penalty: high cancel rate / amendments / kill switch
        # are punished SEVERELY in the reward function
        score = max(0.0, base_score + mar_penalty)
        candidate = self._candidates[self._active_idx]
        candidate.trade_count += 1
        candidate.cumulative_roi += roi

        # Exponential moving average of score
        alpha = 2.0 / (candidate.trade_count + 1)
        candidate.score = alpha * score + (1 - alpha) * candidate.score

        logger.info(
            "OPRO: candidate %s trade #%d ROI=%.4f score=%.1f (avg=%.1f)",
            candidate.candidate_id, candidate.trade_count, roi,
            score, candidate.score,
        )

        # Negative ROI → flag for alternative path generation
        if roi < 0:
            self._on_negative_roi(roi, candidate)

        return score

    def maybe_evolve(self) -> bool:
        """
        Check if it's time to evolve the population.
        Returns True if evolution occurred.
        """
        # Only evolve if active candidate has enough trades
        active = self._candidates[self._active_idx]
        if active.trade_count < self._min_trades:
            return False

        # Check if all candidates have been tested enough
        tested = [c for c in self._candidates if c.trade_count >= self._min_trades]
        if len(tested) < min(3, self._pop_size):
            # Rotate to next untested candidate
            self._rotate_active()
            return False

        # Evolve
        self._evolve()
        return True

    def get_intelligence_weights(self) -> dict[str, float]:
        """Extract intelligence agent weights from active parameters."""
        p = self.active_parameters
        raw = {
            "fundamental": p["weight_fundamental"],
            "technical": p["weight_technical"],
            "sentiment": p["weight_sentiment"],
        }
        # Normalise to sum to 1.0
        total = sum(raw.values())
        if total > 0:
            return {k: v / total for k, v in raw.items()}
        return raw

    def get_decision_thresholds(self) -> dict[str, float]:
        p = self.active_parameters
        return {
            "min_confidence": p["min_confidence_to_trade"],
            "min_buy_score": p["min_buy_score"],
            "max_sell_score": p["max_sell_score"],
        }

    def get_risk_parameters(self) -> dict[str, float]:
        p = self.active_parameters
        return {
            "atr_stop_multiplier": p["atr_stop_multiplier"],
            "atr_tp_multiplier": p["atr_tp_multiplier"],
            "half_kelly_scaling": p["half_kelly_scaling"],
        }

    # ── population management ─────────────────────────────────────────

    def _initialise_population(self) -> None:
        """Seed the population: one default + (N-1) random perturbations."""
        # Candidate 0: default parameters
        default = ParameterCandidate(
            candidate_id="C-000-default",
            parameters=dict(_DEFAULT_PARAMETERS),
            active=True,
        )
        self._candidates = [default]

        # Fill remaining with random perturbations
        for i in range(1, self._pop_size):
            params = self._mutate(dict(_DEFAULT_PARAMETERS), strength=0.3)
            self._candidates.append(ParameterCandidate(
                candidate_id=f"C-000-rand{i}",
                parameters=params,
            ))

        self._active_idx = 0
        self._persist()

    def _evolve(self) -> None:
        """
        One generation of evolution:
          1. Sort by score
          2. Keep elite
          3. Generate offspring via crossover + mutation
          4. Replace worst candidates
        """
        self._generation += 1
        logger.info("OPRO evolution — generation %d", self._generation)

        # Sort by score (descending)
        ranked = sorted(
            range(len(self._candidates)),
            key=lambda i: self._candidates[i].score,
            reverse=True,
        )

        # Record history
        self._history.append({
            "generation": self._generation,
            "timestamp": time.time(),
            "scores": [
                {
                    "id": self._candidates[i].candidate_id,
                    "score": self._candidates[i].score,
                    "trades": self._candidates[i].trade_count,
                    "avg_roi": self._candidates[i].avg_roi,
                }
                for i in ranked
            ],
        })

        # Elite preservation
        elites = [self._candidates[ranked[i]] for i in range(self._elite_count)]

        # Generate offspring
        new_pop: list[ParameterCandidate] = []
        for elite in elites:
            new_pop.append(ParameterCandidate(
                candidate_id=f"C-{self._generation:03d}-elite{len(new_pop)}",
                parameters=dict(elite.parameters),
                score=elite.score,
            ))

        while len(new_pop) < self._pop_size:
            # Tournament selection
            p1 = self._tournament_select(ranked)
            p2 = self._tournament_select(ranked)
            child_params = self._crossover(
                self._candidates[p1].parameters,
                self._candidates[p2].parameters,
            )
            child_params = self._mutate(child_params)
            new_pop.append(ParameterCandidate(
                candidate_id=f"C-{self._generation:03d}-child{len(new_pop)}",
                parameters=child_params,
            ))

        self._candidates = new_pop
        self._active_idx = 0
        self._candidates[0].active = True
        self._persist()

        logger.info(
            "OPRO evolved: best score=%.1f (gen %d), promoted %s",
            self._candidates[0].score,
            self._generation,
            self._candidates[0].candidate_id,
        )

    def _rotate_active(self) -> None:
        """Rotate to the next candidate that needs testing."""
        self._candidates[self._active_idx].active = False
        for i in range(len(self._candidates)):
            idx = (self._active_idx + 1 + i) % len(self._candidates)
            if self._candidates[idx].trade_count < self._min_trades:
                self._active_idx = idx
                self._candidates[idx].active = True
                logger.info(
                    "OPRO rotated to candidate %s for evaluation",
                    self._candidates[idx].candidate_id,
                )
                return
        # All tested — stay on current best
        best_idx = max(range(len(self._candidates)), key=lambda i: self._candidates[i].score)
        self._active_idx = best_idx
        self._candidates[best_idx].active = True

    def _on_negative_roi(self, roi: float, candidate: ParameterCandidate) -> None:
        """
        When ROI is negative: generate an alternative reasoning path
        by creating a new mutation. This forces the system to explore
        different parameter regions rather than staying stuck.
        """
        if candidate.score < 30 and candidate.trade_count >= 3:
            logger.warning(
                "OPRO: candidate %s score dropped to %.1f — "
                "generating alternative reasoning path",
                candidate.candidate_id, candidate.score,
            )
            # Strong mutation to escape local minimum
            alt_params = self._mutate(candidate.parameters, strength=0.4)

            # Replace the worst candidate in the population
            worst_idx = min(
                range(len(self._candidates)),
                key=lambda i: self._candidates[i].score,
            )
            if worst_idx != self._active_idx:
                self._candidates[worst_idx] = ParameterCandidate(
                    candidate_id=f"C-{self._generation:03d}-alt{worst_idx}",
                    parameters=alt_params,
                )

    # ── genetic operators ─────────────────────────────────────────────

    @staticmethod
    def _crossover(p1: dict[str, float], p2: dict[str, float]) -> dict[str, float]:
        """Uniform crossover between two parameter sets."""
        child: dict[str, float] = {}
        for key in p1:
            child[key] = p1[key] if random.random() < 0.5 else p2[key]
        return child

    @staticmethod
    def _mutate(
        params: dict[str, float],
        strength: float = 0.15,
    ) -> dict[str, float]:
        """Gaussian mutation within parameter bounds."""
        mutated = dict(params)
        for key, val in mutated.items():
            if random.random() < 0.5:  # mutate 50% of parameters
                lo, hi = _PARAM_BOUNDS.get(key, (val * 0.5, val * 2.0))
                span = hi - lo
                noise = random.gauss(0, strength * span)
                mutated[key] = max(lo, min(hi, val + noise))

        # Re-normalise weights so they sum to ~1, then clamp to bounds
        weight_keys = ["weight_fundamental", "weight_technical", "weight_sentiment"]
        weight_sum = sum(mutated[k] for k in weight_keys)
        if weight_sum > 0:
            for k in weight_keys:
                mutated[k] /= weight_sum
                lo, hi = _PARAM_BOUNDS.get(k, (0.0, 1.0))
                mutated[k] = max(lo, min(hi, mutated[k]))

        return mutated

    def _tournament_select(self, ranked: list[int], k: int = 3) -> int:
        """Select the best of k random candidates."""
        sample = random.sample(ranked, min(k, len(ranked)))
        return min(sample, key=lambda i: ranked.index(i))

    # ── failure-driven agent-weight optimisation ─────────────────────

    # 用於解析 FailureAttributionEngine 寫入的歸因字串
    # 英文格式: "Primary failure: semantic (65%) — ..."
    _RE_ENGLISH = re.compile(
        r"Primary failure:\s*(semantic|temporal|execution)\s*\((\d+(?:\.\d+)?)%\)"
    )
    # 中文格式: "[執行:10.0%|時序:80.0%|語義:10.0%]"
    _RE_CHINESE = re.compile(
        r"\[執行:([\d.]+)%\|時序:([\d.]+)%\|語義:([\d.]+)%\]"
    )

    # 歸因閾值：某維度平均超過此值才觸發突變
    _ATTRIBUTION_THRESHOLD = 40.0
    # 失敗樣本下限
    _MIN_FAILURE_SAMPLES = 3

    def run_failure_driven_optimization(
        self,
        memory: "EpisodicMemory",
        technical_agent: Any = None,
        fundamental_agent: Any = None,
        sentiment_agent: Any = None,
        limit: int = 30,
    ) -> dict[str, Any]:
        """
        失敗歸因驅動的代理權重突變優化。

        掃描 EpisodicMemory 中近期虧損記憶的 ``failure_detail``，
        解析三維歸因比例，並據此突變各情報代理的內部策略權重。

        此方法與既有的 bandit 族群演化互補：
        - bandit 演化：調整 *Orchestrator 層級* 融合權重（基本面/技術/情緒佔比）
        - 歸因突變：調整 *代理內部* 策略權重（如 RSI vs SMA vs Bollinger）

        建議在每日收盤後或累積足夠失敗樣本後執行。

        Parameters
        ----------
        memory : EpisodicMemory 記憶庫
        technical_agent : TechnicalAgent (有 dynamic_weights 與 update_weights_from_opro)
        fundamental_agent : FundamentalAgent
        sentiment_agent : SentimentAgent
        limit : 檢視近 N 筆記憶

        Returns
        -------
        dict with status, attribution averages, and applied updates.
        """
        logger.info(
            "[OPRO] 啟動失敗歸因驅動權重優化迴圈 (檢視近 %d 筆記憶)...", limit,
        )

        # 1. 撈取近期虧損記憶
        episodes = memory._episodes[-limit:]  # noqa: SLF001 — 同包模組存取
        failed_episodes = [
            ep for ep in episodes
            if ep.roi < -0.005 and (ep.failure_layer or ep.failure_detail)
        ]

        if len(failed_episodes) < self._MIN_FAILURE_SAMPLES:
            logger.info(
                "[OPRO] 近期失敗樣本數不足 (%d < %d)，暫不進行權重突變。",
                len(failed_episodes), self._MIN_FAILURE_SAMPLES,
            )
            return {"status": "SKIPPED", "reason": "失敗樣本不足",
                    "failed_count": len(failed_episodes)}

        # 2. 統計平均歸因比例
        total_exec, total_temp, total_sem = 0.0, 0.0, 0.0
        parsed_count = 0

        for ep in failed_episodes:
            pcts = self._parse_attribution(ep.failure_detail, ep.failure_layer)
            if pcts is not None:
                total_exec += pcts["execution"]
                total_temp += pcts["temporal"]
                total_sem += pcts["semantic"]
                parsed_count += 1

        if parsed_count == 0:
            return {"status": "SKIPPED", "reason": "無法解析歸因格式",
                    "failed_count": len(failed_episodes)}

        avg_exec = total_exec / parsed_count
        avg_temp = total_temp / parsed_count
        avg_sem = total_sem / parsed_count

        logger.info(
            "[OPRO] 近期 %d 筆虧損平均歸因 — "
            "執行:%.1f%% | 時序:%.1f%% | 語義:%.1f%%",
            parsed_count, avg_exec, avg_temp, avg_sem,
        )

        updates_applied: dict[str, Any] = {}
        penalty = self._mutation_rate  # 基礎學習率

        # 3. 時序錯誤過高 → TechnicalAgent 需要被教訓
        if avg_temp > self._ATTRIBUTION_THRESHOLD and technical_agent is not None:
            logger.warning(
                "[OPRO] 時序錯誤偏高 (%.1f%%)，觸發 TechnicalAgent 權重突變！",
                avg_temp,
            )
            scaled_penalty = penalty * (avg_temp / 100.0)
            new_weights = self._mutate_agent_weights(
                technical_agent.dynamic_weights, scaled_penalty,
            )
            technical_agent.update_weights_from_opro(new_weights)
            updates_applied["TechnicalAgent"] = new_weights

        # 4. 語義錯誤過高 → Fundamental / Sentiment 需要被教訓
        if avg_sem > self._ATTRIBUTION_THRESHOLD:
            logger.warning(
                "[OPRO] 語義錯誤偏高 (%.1f%%)，觸發 Fundamental/Sentiment 權重突變！",
                avg_sem,
            )
            scaled_penalty = penalty * (avg_sem / 100.0)

            if fundamental_agent is not None:
                new_fund = self._mutate_agent_weights(
                    fundamental_agent.dynamic_weights, scaled_penalty,
                )
                fundamental_agent.update_weights_from_opro(new_fund)
                updates_applied["FundamentalAgent"] = new_fund

            if sentiment_agent is not None:
                new_sent = self._mutate_agent_weights(
                    sentiment_agent.dynamic_weights, scaled_penalty,
                )
                sentiment_agent.update_weights_from_opro(new_sent)
                updates_applied["SentimentAgent"] = new_sent

        # 5. 執行摩擦過高 → 系統層級警告
        if avg_exec > self._ATTRIBUTION_THRESHOLD:
            logger.critical(
                "[OPRO] 執行摩擦極高 (%.1f%%)！"
                "建議：調降 RiskAgent 部位大小或放寬 ExecutionAgent 滑價容忍度。",
                avg_exec,
            )
            updates_applied["ExecutionWarning"] = {
                "avg_execution_pct": avg_exec,
                "recommendation": "調降部位大小或使用限價單",
            }

        return {
            "status": "OPTIMIZED" if updates_applied else "NO_CHANGE",
            "analyzed_failures": parsed_count,
            "average_attribution": {
                "execution": round(avg_exec, 1),
                "temporal": round(avg_temp, 1),
                "semantic": round(avg_sem, 1),
            },
            "updates": updates_applied,
        }

    @classmethod
    def _parse_attribution(
        cls,
        failure_detail: str,
        failure_layer: str = "",
    ) -> dict[str, float] | None:
        """
        從 failure_detail 字串解析三維歸因百分比。

        支援兩種格式：
        1. 英文: "Primary failure: semantic (65%) — ..."
        2. 中文: "[執行:10.0%|時序:80.0%|語義:10.0%]"

        若無法解析但有 failure_layer，退化為該維度 100%。
        """
        if not failure_detail and not failure_layer:
            return None

        # 嘗試中文格式
        m_cn = cls._RE_CHINESE.search(failure_detail or "")
        if m_cn:
            return {
                "execution": float(m_cn.group(1)),
                "temporal": float(m_cn.group(2)),
                "semantic": float(m_cn.group(3)),
            }

        # 嘗試英文格式 — 只知道 primary layer 和其佔比
        m_en = cls._RE_ENGLISH.search(failure_detail or "")
        if m_en:
            primary = m_en.group(1)
            pct = float(m_en.group(2))
            # 剩餘比例平分給其他兩個維度
            remaining = 100.0 - pct
            result = {
                "semantic": remaining / 2,
                "temporal": remaining / 2,
                "execution": remaining / 2,
            }
            result[primary] = pct
            # 重新正規化（因為上面多算了 remaining/2 一次）
            other_keys = [k for k in result if k != primary]
            for k in other_keys:
                result[k] = remaining / 2
            return result

        # 退化：只有 failure_layer 但無 detail → 100% 歸責
        if failure_layer in ("semantic", "temporal", "execution"):
            result = {"semantic": 0.0, "temporal": 0.0, "execution": 0.0}
            result[failure_layer] = 100.0
            return result

        return None

    @staticmethod
    def _mutate_agent_weights(
        current_weights: dict[str, float],
        penalty_factor: float,
    ) -> dict[str, float]:
        """
        罪魁禍首懲罰突變：削弱佔比最重的維度權重，
        並將削減的點數分配給其他維度以探索新策略。

        Gaussian 擾動避免陷入局部最佳解。
        """
        if not current_weights:
            return {}

        new_weights = dict(current_weights)

        # 找出當前最大的權重 (罪魁禍首)
        max_key = max(new_weights, key=new_weights.get)  # type: ignore[arg-type]

        # 削減最大權重
        reduction = new_weights[max_key] * penalty_factor
        new_weights[max_key] -= reduction

        # 將削減的權重分配給其他維度 (加入 Gaussian 擾動)
        other_keys = [k for k in new_weights if k != max_key]
        if other_keys:
            for k in other_keys:
                noise = random.uniform(0.8, 1.2)
                new_weights[k] += (reduction / len(other_keys)) * noise

        # 正規化確保總和 = 1.0
        total = sum(new_weights.values())
        if total > 0:
            new_weights = {k: round(v / total, 4) for k, v in new_weights.items()}

        return new_weights

    # ── persistence ───────────────────────────────────────────────────

    def _persist(self) -> None:
        state = {
            "generation": self._generation,
            "active_idx": self._active_idx,
            "candidates": [
                {
                    "id": c.candidate_id,
                    "parameters": c.parameters,
                    "score": c.score,
                    "trade_count": c.trade_count,
                    "cumulative_roi": c.cumulative_roi,
                    "active": c.active,
                }
                for c in self._candidates
            ],
            "history": self._history[-20:],  # keep last 20 generations
        }
        path = self._store_dir / "opro_state.json"
        path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")

    def load(self) -> bool:
        """Load state from disk. Returns True if loaded successfully."""
        path = self._store_dir / "opro_state.json"
        if not path.exists():
            return False
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
            self._generation = state["generation"]
            self._active_idx = state["active_idx"]
            self._candidates = [
                ParameterCandidate(
                    candidate_id=c["id"],
                    parameters=c["parameters"],
                    score=c["score"],
                    trade_count=c["trade_count"],
                    cumulative_roi=c["cumulative_roi"],
                    active=c.get("active", False),
                )
                for c in state["candidates"]
            ]
            self._history = state.get("history", [])
            logger.info(
                "OPRO loaded: gen=%d, active=%s, best_score=%.1f",
                self._generation,
                self._candidates[self._active_idx].candidate_id,
                max(c.score for c in self._candidates),
            )
            return True
        except Exception:
            logger.exception("Failed to load OPRO state")
            return False
