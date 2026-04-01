"""
Stress Testing Engine — 投資組合壓力測試
=========================================
提供三大壓力測試能力：

1. CVaR (Conditional VaR) — 尾部風險量化
   補充現有 VaR 的不足：VaR 只回答「最多虧多少」，
   CVaR 回答「如果超過 VaR 閾值，平均會虧多少」。

2. 情境分析 — 極端事件模擬
   模擬歷史上的金融危機（2008、COVID、閃崩）對當前組合的衝擊。
   每個情境定義了市場衝擊幅度、波動率放大、相關性飆升等參數。

3. 蒙特卡洛排列測試 — 策略統計顯著性驗證
   打亂交易時機 N 次，檢驗策略的夏普比率是否顯著優於隨機。
   僅用於回測驗證，不在實盤中運行。

使用者：
  - DecisionFusionAgent：decide_batch() 中的風險閘門
  - BacktestEngine：回測後的統計顯著性檢驗
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class StressTestResult:
    """完整壓力測試結果。"""
    cvar_95: float = 0.0              # 95% Conditional VaR (佔 NAV 比例)
    cvar_99: float = 0.0              # 99% Conditional VaR
    scenario_results: dict[str, float] = field(default_factory=dict)
    worst_scenario: str = ""
    worst_scenario_loss: float = 0.0
    passes_stress_test: bool = True
    reasons: list[str] = field(default_factory=list)

    @property
    def summary(self) -> str:
        status = "PASS" if self.passes_stress_test else "FAIL"
        return (
            f"StressTest[{status}] CVaR95={self.cvar_95:.2%} "
            f"CVaR99={self.cvar_99:.2%} "
            f"Worst={self.worst_scenario}({self.worst_scenario_loss:.2%})"
        )


@dataclass
class PermutationTestResult:
    """蒙特卡洛排列測試結果。"""
    strategy_sharpe: float = 0.0
    random_sharpe_mean: float = 0.0
    random_sharpe_std: float = 0.0
    p_value: float = 1.0
    is_significant: bool = False      # p_value < 0.05
    n_permutations: int = 0

    @property
    def summary(self) -> str:
        sig = "SIGNIFICANT" if self.is_significant else "NOT significant"
        return (
            f"Permutation[{sig}] strategy_sharpe={self.strategy_sharpe:.3f} "
            f"random_mean={self.random_sharpe_mean:.3f} "
            f"p={self.p_value:.4f} (n={self.n_permutations})"
        )


# ── 預定義歷史危機情境 ──────────────────────────────────────────────
SCENARIOS: dict[str, dict[str, Any]] = {
    "2008_financial_crisis": {
        "description": "Lehman Brothers collapse — 全球金融海嘯",
        "market_shock": -0.09,          # 單日 9% 跌幅
        "volatility_multiplier": 3.0,   # 波動率放大 3 倍
        "duration_days": 20,            # 持續 20 天
        "correlation_spike": 0.95,      # 相關性飆升至 0.95
    },
    "2020_covid_crash": {
        "description": "COVID-19 pandemic selloff",
        "market_shock": -0.12,
        "volatility_multiplier": 4.0,
        "duration_days": 15,
        "correlation_spike": 0.90,
    },
    "2010_flash_crash": {
        "description": "Flash crash — 閃電崩盤",
        "market_shock": -0.06,
        "volatility_multiplier": 5.0,
        "duration_days": 1,
        "correlation_spike": 0.98,
    },
    "rate_shock": {
        "description": "Sudden interest rate spike — 利率衝擊",
        "market_shock": -0.04,
        "volatility_multiplier": 2.0,
        "duration_days": 10,
        "correlation_spike": 0.70,
    },
    "tech_sector_crash": {
        "description": "Tech sector rotation — 科技股崩盤",
        "market_shock": -0.07,
        "volatility_multiplier": 2.5,
        "duration_days": 10,
        "correlation_spike": 0.85,
    },
}


class StressTester:
    """
    投資組合壓力測試引擎。

    Parameters
    ----------
    max_cvar_pct : CVaR 超過此閾值（佔 NAV 比例）則測試不通過
    max_scenario_loss_pct : 任何情境損失超過此閾值則不通過
    permutation_runs : 排列測試的蒙特卡洛次數
    """

    def __init__(
        self,
        max_cvar_pct: float = 0.05,
        max_scenario_loss_pct: float = 0.15,
        permutation_runs: int = 1000,
    ) -> None:
        self._max_cvar = max_cvar_pct
        self._max_scenario_loss = max_scenario_loss_pct
        self._perm_runs = permutation_runs

    # ══════════════════════════════════════════════════════════════
    # CVaR 計算
    # ══════════════════════════════════════════════════════════════

    @staticmethod
    def compute_cvar(
        returns: np.ndarray,
        confidence: float = 0.95,
    ) -> float:
        """
        計算 Conditional VaR (Expected Shortfall)。

        CVaR = 超過 VaR 閾值的損失的平均值。
        比 VaR 更能捕捉尾部風險。

        Parameters
        ----------
        returns : 日收益率序列
        confidence : 信賴水準（0.95 或 0.99）

        Returns
        -------
        float : CVaR（正值代表損失幅度，佔價值比例）
        """
        if len(returns) < 10:
            return 0.0

        # Sort returns ascending (worst first)
        sorted_returns = np.sort(returns)

        # VaR cutoff index
        cutoff = int(len(sorted_returns) * (1 - confidence))
        cutoff = max(1, cutoff)

        # CVaR = average of returns worse than VaR
        tail = sorted_returns[:cutoff]
        cvar = -float(np.mean(tail))  # negate so positive = loss

        return max(0.0, cvar)

    def compute_portfolio_cvar(
        self,
        portfolio_weights: dict[str, float],
        returns_map: dict[str, np.ndarray],
        confidence: float = 0.95,
    ) -> float:
        """
        計算組合層面的 CVaR。

        將各資產收益按權重加總，再計算組合收益序列的 CVaR。
        """
        tickers = [t for t in portfolio_weights if t in returns_map]
        if not tickers:
            return 0.0

        # Align returns to same length
        min_len = min(len(returns_map[t]) for t in tickers)
        if min_len < 10:
            return 0.0

        # Compute portfolio returns
        total_weight = sum(abs(portfolio_weights[t]) for t in tickers)
        if total_weight == 0:
            return 0.0

        portfolio_returns = np.zeros(min_len)
        for t in tickers:
            w = portfolio_weights[t] / total_weight
            portfolio_returns += w * returns_map[t][-min_len:]

        return self.compute_cvar(portfolio_returns, confidence)

    # ══════════════════════════════════════════════════════════════
    # 情境分析
    # ══════════════════════════════════════════════════════════════

    def run_scenario(
        self,
        portfolio_weights: dict[str, float],
        returns_map: dict[str, np.ndarray],
        scenario_name: str,
    ) -> float:
        """
        模擬單一情境下的組合損失。

        Parameters
        ----------
        portfolio_weights : ticker → 權重（佔 NAV 比例）
        returns_map : ticker → 歷史日收益率
        scenario_name : 情境名稱（需在 SCENARIOS 中定義）

        Returns
        -------
        float : 預期損失（正值 = 虧損比例）
        """
        scenario = SCENARIOS.get(scenario_name)
        if scenario is None:
            return 0.0

        shock = scenario["market_shock"]
        vol_mult = scenario["volatility_multiplier"]
        duration = scenario["duration_days"]
        corr_spike = scenario["correlation_spike"]

        tickers = [t for t in portfolio_weights if t in returns_map]
        if not tickers:
            return 0.0

        total_weight = sum(abs(portfolio_weights[t]) for t in tickers)
        if total_weight == 0:
            return 0.0

        # Compute stressed portfolio loss
        portfolio_loss = 0.0

        for t in tickers:
            w = portfolio_weights[t] / total_weight
            returns = returns_map[t]
            if len(returns) < 20:
                continue

            # Asset-specific stress: base volatility amplified
            asset_vol = float(np.std(returns, ddof=1))
            stressed_daily_loss = shock + asset_vol * vol_mult * np.random.standard_normal()

            # Correlation spike: all assets move together
            # Blend: corr_spike portion of shock + (1-corr_spike) of idiosyncratic
            correlated_component = shock * corr_spike
            idiosyncratic = stressed_daily_loss * (1 - corr_spike)
            total_daily = correlated_component + idiosyncratic

            # Multi-day compounding (rough)
            total_loss = 1 - (1 + total_daily) ** min(duration, 5)

            portfolio_loss += abs(w) * total_loss

        return max(0.0, portfolio_loss)

    # ══════════════════════════════════════════════════════════════
    # 完整壓力測試
    # ══════════════════════════════════════════════════════════════

    def full_stress_test(
        self,
        portfolio_weights: dict[str, float],
        returns_map: dict[str, np.ndarray],
    ) -> StressTestResult:
        """
        執行完整壓力測試套件。

        包含：
          1. 95% 和 99% CVaR 計算
          2. 所有預定義情境的損失模擬
          3. 通過/不通過判定

        Parameters
        ----------
        portfolio_weights : ticker → 權重佔 NAV 比例
        returns_map : ticker → 日收益率序列
        """
        result = StressTestResult()

        # 1. CVaR computation
        result.cvar_95 = self.compute_portfolio_cvar(
            portfolio_weights, returns_map, 0.95,
        )
        result.cvar_99 = self.compute_portfolio_cvar(
            portfolio_weights, returns_map, 0.99,
        )

        # 2. Scenario analysis (run each scenario multiple times for stability)
        n_runs = 5
        for scenario_name in SCENARIOS:
            losses = [
                self.run_scenario(portfolio_weights, returns_map, scenario_name)
                for _ in range(n_runs)
            ]
            avg_loss = float(np.mean(losses))
            result.scenario_results[scenario_name] = avg_loss

        # Find worst scenario
        if result.scenario_results:
            worst = max(result.scenario_results, key=result.scenario_results.get)  # type: ignore
            result.worst_scenario = worst
            result.worst_scenario_loss = result.scenario_results[worst]

        # 3. Pass/fail determination
        result.passes_stress_test = True

        if result.cvar_95 > self._max_cvar:
            result.passes_stress_test = False
            result.reasons.append(
                f"CVaR95 ({result.cvar_95:.2%}) exceeds limit ({self._max_cvar:.2%})"
            )

        if result.worst_scenario_loss > self._max_scenario_loss:
            result.passes_stress_test = False
            result.reasons.append(
                f"Worst scenario '{result.worst_scenario}' "
                f"loss ({result.worst_scenario_loss:.2%}) exceeds "
                f"limit ({self._max_scenario_loss:.2%})"
            )

        log_fn = logger.warning if not result.passes_stress_test else logger.info
        log_fn("[StressTester] %s", result.summary)

        return result

    # ══════════════════════════════════════════════════════════════
    # 蒙特卡洛排列測試
    # ══════════════════════════════════════════════════════════════

    def permutation_test(
        self,
        strategy_returns: np.ndarray,
        n_permutations: int | None = None,
    ) -> PermutationTestResult:
        """
        蒙特卡洛排列測試：驗證策略是否顯著優於隨機。

        方法：
          1. 計算策略的夏普比率
          2. 隨機打亂收益序列 N 次
          3. 計算每次打亂後的夏普
          4. p-value = 優於策略的比例

        Parameters
        ----------
        strategy_returns : 策略的日收益率序列
        n_permutations : 排列次數（預設使用建構時參數）

        注意：此方法僅用於回測驗證，不在實盤中運行。
        """
        n = n_permutations or self._perm_runs

        result = PermutationTestResult(n_permutations=n)

        if len(strategy_returns) < 30:
            result.p_value = 1.0
            return result

        # Strategy Sharpe
        strat_mean = float(np.mean(strategy_returns))
        strat_std = float(np.std(strategy_returns, ddof=1))
        if strat_std == 0:
            return result

        result.strategy_sharpe = (strat_mean / strat_std) * np.sqrt(252)

        # Permutation distribution
        random_sharpes: list[float] = []
        for _ in range(n):
            shuffled = np.random.permutation(strategy_returns)
            s_mean = float(np.mean(shuffled))
            s_std = float(np.std(shuffled, ddof=1))
            if s_std > 0:
                random_sharpes.append((s_mean / s_std) * np.sqrt(252))

        if not random_sharpes:
            return result

        result.random_sharpe_mean = float(np.mean(random_sharpes))
        result.random_sharpe_std = float(np.std(random_sharpes))

        # p-value: fraction of random Sharpes >= strategy Sharpe
        result.p_value = float(np.mean(
            [1.0 if rs >= result.strategy_sharpe else 0.0 for rs in random_sharpes]
        ))
        result.is_significant = result.p_value < 0.05

        logger.info("[StressTester] %s", result.summary)
        return result

    # ══════════════════════════════════════════════════════════════
    # 輔助：根據壓力測試縮減提案
    # ══════════════════════════════════════════════════════════════

    def compute_reduction_factor(self, stress_result: StressTestResult) -> float:
        """
        根據壓力測試結果計算提案縮減比例。

        Returns
        -------
        float : 0.0–1.0，1.0 = 不需縮減，0.0 = 全部轉 HOLD
        """
        if stress_result.passes_stress_test:
            return 1.0

        # CVaR-based reduction
        if stress_result.cvar_95 > 0 and self._max_cvar > 0:
            cvar_ratio = self._max_cvar / stress_result.cvar_95
        else:
            cvar_ratio = 1.0

        # Scenario-based reduction
        if stress_result.worst_scenario_loss > 0 and self._max_scenario_loss > 0:
            scenario_ratio = self._max_scenario_loss / stress_result.worst_scenario_loss
        else:
            scenario_ratio = 1.0

        # Use the more conservative (lower) reduction
        factor = min(cvar_ratio, scenario_ratio)
        factor = max(0.1, min(1.0, factor))  # floor at 10%

        logger.info(
            "[StressTester] Reduction factor: %.2f (cvar=%.2f scenario=%.2f)",
            factor, cvar_ratio, scenario_ratio,
        )
        return factor
