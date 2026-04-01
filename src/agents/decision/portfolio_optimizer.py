"""
Portfolio Optimizer — 投資組合優化引擎
======================================
提供三大組合優化策略：

1. Black-Litterman 模型
   結合市場均衡回報（從共變異數矩陣推算）與 AI 智能體的觀點
   （MarketView 的 fused_score / fused_confidence），
   產生最優投資組合權重。

2. Risk Parity（風險平價）
   按風險貢獻而非資金金額分配倉位，確保每個資產對組合
   風險的貢獻相等。當 AI 信號信心度普遍偏低時作為後備。

3. Dynamic Rebalancer（動態再平衡）
   定期檢查當前組合是否偏離目標配置，超過閾值時觸發再平衡。

使用者：
  - DecisionFusionAgent：decide_batch() 中的組合優化階段
  - TradingSystem：run_cycle() 中的再平衡檢查
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
# Black-Litterman Optimizer
# ══════════════════════════════════════════════════════════════════

class BlackLittermanOptimizer:
    """
    Black-Litterman 模型。

    將市場均衡回報（隱含回報）與 AI 觀點融合，
    產生後驗回報估計，再透過均值-方差優化得到最優權重。

    Parameters
    ----------
    risk_aversion : 風險趨避係數（δ），典型值 2.5
    tau : 不確定性縮放因子，控制觀點的影響程度
    """

    def __init__(
        self,
        risk_aversion: float = 2.5,
        tau: float = 0.05,
    ) -> None:
        self._delta = risk_aversion
        self._tau = tau

    def compute_equilibrium_returns(
        self,
        market_weights: np.ndarray,
        cov_matrix: np.ndarray,
    ) -> np.ndarray:
        """
        計算隱含均衡回報：Π = δ × Σ × w_mkt

        Parameters
        ----------
        market_weights : 市場均衡權重（等權或市值加權）
        cov_matrix : 共變異數矩陣
        """
        return self._delta * cov_matrix @ market_weights

    def optimize(
        self,
        market_weights: np.ndarray,
        cov_matrix: np.ndarray,
        view_returns: np.ndarray | None = None,
        view_confidence: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        執行 Black-Litterman 優化。

        Parameters
        ----------
        market_weights : 市場均衡權重 (N,)
        cov_matrix : 共變異數矩陣 (N, N)
        view_returns : AI 觀點的預期回報 (K,)，每個觀點對應一個資產
        view_confidence : 觀點信心度 (K,)，0-1

        Returns
        -------
        np.ndarray : 最優權重 (N,)，和為 1.0
        """
        n = len(market_weights)

        if n == 0:
            return np.array([])

        # 1. Equilibrium returns
        pi = self.compute_equilibrium_returns(market_weights, cov_matrix)

        # 2. If no views, return market weights
        if view_returns is None or len(view_returns) == 0:
            return market_weights.copy()

        k = len(view_returns)

        # 3. Build view matrices
        # P: pick matrix (K × N), identity for absolute views
        P = np.eye(k, n) if k <= n else np.eye(n)[:k]

        # Q: expected returns from views (K,)
        Q = view_returns[:k]

        # Omega: uncertainty diagonal matrix (K × K)
        # Lower confidence → higher uncertainty
        if view_confidence is not None and len(view_confidence) >= k:
            conf = np.clip(view_confidence[:k], 0.05, 1.0)
        else:
            conf = np.full(k, 0.5)

        # Omega_ii = tau * (1/confidence_i - 1) * diag(P @ Σ @ P.T)
        P_cov_Pt = P @ cov_matrix @ P.T
        omega_diag = self._tau * (1.0 / conf - 1.0) * np.abs(np.diag(P_cov_Pt))
        omega_diag = np.maximum(omega_diag, 1e-8)
        Omega = np.diag(omega_diag)

        # 4. Posterior return: E[R] = [(τΣ)^-1 + P'Ω^-1 P]^-1 × [(τΣ)^-1 Π + P'Ω^-1 Q]
        tau_cov = self._tau * cov_matrix
        tau_cov_inv = np.linalg.pinv(tau_cov)
        omega_inv = np.diag(1.0 / omega_diag)

        # Left term
        left = tau_cov_inv + P.T @ omega_inv @ P
        left_inv = np.linalg.pinv(left)

        # Right term
        right = tau_cov_inv @ pi + P.T @ omega_inv @ Q

        # Posterior expected return
        posterior_return = left_inv @ right

        # 5. Mean-variance optimal weights: w* = (δΣ)^-1 × E[R]
        delta_cov_inv = np.linalg.pinv(self._delta * cov_matrix)
        raw_weights = delta_cov_inv @ posterior_return

        # 6. Normalize to sum to 1, enforce non-negative (long-only)
        raw_weights = np.maximum(raw_weights, 0.0)
        total = np.sum(raw_weights)
        if total > 0:
            weights = raw_weights / total
        else:
            weights = np.full(n, 1.0 / n)

        return weights


# ══════════════════════════════════════════════════════════════════
# Risk Parity Allocator
# ══════════════════════════════════════════════════════════════════

class RiskParityAllocator:
    """
    風險平價配置器。

    分配權重使得每個資產對組合風險的貢獻相等。

    Parameters
    ----------
    max_iterations : 最大迭代次數
    tolerance : 收斂容忍度
    """

    def __init__(
        self,
        max_iterations: int = 100,
        tolerance: float = 1e-8,
    ) -> None:
        self._max_iter = max_iterations
        self._tol = tolerance

    def compute_risk_contributions(
        self,
        weights: np.ndarray,
        cov_matrix: np.ndarray,
    ) -> np.ndarray:
        """
        計算各資產的風險貢獻。

        RC_i = w_i × (Σ @ w)_i / σ_p
        """
        portfolio_var = weights @ cov_matrix @ weights
        if portfolio_var <= 0:
            return np.zeros_like(weights)

        sigma_p = np.sqrt(portfolio_var)
        marginal_risk = cov_matrix @ weights
        risk_contrib = weights * marginal_risk / sigma_p

        return risk_contrib

    def optimize(self, cov_matrix: np.ndarray) -> np.ndarray:
        """
        求解風險平價權重。

        使用反向波動率作為初始猜測，然後迭代優化。

        Parameters
        ----------
        cov_matrix : 共變異數矩陣 (N, N)

        Returns
        -------
        np.ndarray : 風險平價權重 (N,)
        """
        n = cov_matrix.shape[0]

        if n == 0:
            return np.array([])

        if n == 1:
            return np.array([1.0])

        # Initial guess: inverse volatility weights
        vols = np.sqrt(np.diag(cov_matrix))
        vols = np.maximum(vols, 1e-10)
        weights = (1.0 / vols) / np.sum(1.0 / vols)

        target_rc = 1.0 / n  # equal risk contribution

        for iteration in range(self._max_iter):
            rc = self.compute_risk_contributions(weights, cov_matrix)
            total_rc = np.sum(rc)

            if total_rc <= 0:
                break

            rc_pct = rc / total_rc

            # Update: adjust weights proportional to deviation from target
            adjustment = target_rc / np.maximum(rc_pct, 1e-10)
            new_weights = weights * adjustment
            new_weights = np.maximum(new_weights, 1e-10)
            new_weights /= np.sum(new_weights)

            # Check convergence
            if np.max(np.abs(new_weights - weights)) < self._tol:
                weights = new_weights
                break

            weights = new_weights

        return weights


# ══════════════════════════════════════════════════════════════════
# Dynamic Rebalancer
# ══════════════════════════════════════════════════════════════════

@dataclass
class RebalanceCheck:
    """再平衡檢查結果。"""
    needs_rebalance: bool = False
    drifts: dict[str, float] = field(default_factory=dict)  # ticker → 偏離量
    max_drift: float = 0.0
    max_drift_ticker: str = ""
    trades: list[dict[str, Any]] = field(default_factory=list)

    @property
    def summary(self) -> str:
        if not self.needs_rebalance:
            return "No rebalance needed"
        return (
            f"Rebalance needed: max drift={self.max_drift:.2%} "
            f"({self.max_drift_ticker}), {len(self.trades)} trades"
        )


class DynamicRebalancer:
    """
    動態再平衡器。

    定期檢查組合是否偏離目標配置，超過閾值時生成再平衡交易。

    Parameters
    ----------
    drift_threshold : 觸發再平衡的絕對偏離閾值（預設 5%）
    min_trade_value : 最小再平衡交易金額（避免微量調整）
    """

    def __init__(
        self,
        drift_threshold: float = 0.05,
        min_trade_value: float = 50.0,
    ) -> None:
        self._threshold = drift_threshold
        self._min_trade = min_trade_value
        self._last_rebalance: float = 0.0

    def check_drift(
        self,
        current_weights: dict[str, float],
        target_weights: dict[str, float],
    ) -> RebalanceCheck:
        """
        檢查當前組合是否偏離目標配置。

        Parameters
        ----------
        current_weights : ticker → 當前權重（佔 NAV 比例）
        target_weights : ticker → 目標權重

        Returns
        -------
        RebalanceCheck 含是否需要再平衡及各資產偏離量
        """
        result = RebalanceCheck()

        all_tickers = set(current_weights) | set(target_weights)

        for ticker in all_tickers:
            current = current_weights.get(ticker, 0.0)
            target = target_weights.get(ticker, 0.0)
            drift = current - target
            result.drifts[ticker] = drift

            if abs(drift) > abs(result.max_drift):
                result.max_drift = drift
                result.max_drift_ticker = ticker

        # Check if any drift exceeds threshold
        result.needs_rebalance = any(
            abs(d) > self._threshold for d in result.drifts.values()
        )

        return result

    def compute_rebalance_trades(
        self,
        current_weights: dict[str, float],
        target_weights: dict[str, float],
        total_nav: float,
        price_map: dict[str, float],
    ) -> list[dict[str, Any]]:
        """
        計算再平衡所需的交易清單。

        Returns list of:
        [{
            "ticker": str,
            "direction": "BUY" | "SELL",
            "quantity": float,
            "value": float,
            "weight_change": float
        }]
        """
        trades: list[dict[str, Any]] = []

        all_tickers = set(current_weights) | set(target_weights)

        for ticker in all_tickers:
            current = current_weights.get(ticker, 0.0)
            target = target_weights.get(ticker, 0.0)
            delta_weight = target - current

            if abs(delta_weight) < 0.001:
                continue

            price = price_map.get(ticker, 0.0)
            if price <= 0:
                continue

            trade_value = abs(delta_weight) * total_nav
            if trade_value < self._min_trade:
                continue

            quantity = trade_value / price
            direction = "BUY" if delta_weight > 0 else "SELL"

            trades.append({
                "ticker": ticker,
                "direction": direction,
                "quantity": round(quantity, 6),
                "value": trade_value,
                "weight_change": delta_weight,
            })

        # Sort: sells first (free up cash), then buys
        trades.sort(key=lambda t: (t["direction"] == "BUY", -t["value"]))

        self._last_rebalance = time.time()
        return trades

    @property
    def last_rebalance_time(self) -> float:
        return self._last_rebalance
