"""
Correlation Filter (Phase 6)
=============================
Prevents portfolio concentration in highly correlated assets.

Problem:
  Buying AAPL + MSFT + GOOGL + META looks diversified (4 stocks)
  but they're all ~0.85 correlated. A tech selloff kills all four.
  The Herfindahl index can't catch this — it only measures weight
  concentration, not return correlation.

Solution:
  Compute rolling pairwise correlation matrix across all holdings
  and candidate trades. Block or penalize new positions that are
  too correlated with existing holdings.

Key concepts:
  - Correlation matrix: 60d rolling Pearson correlation of daily returns
  - Max pairwise correlation: highest correlation between new trade
    and any existing holding
  - Portfolio average correlation: mean correlation across all pairs
  - Cluster detection: groups of 3+ stocks with avg correlation > 0.7

Integration:
  Called by DecisionFusionAgent.decide_batch() AFTER individual
  proposals are generated but BEFORE final ranking.

Thresholds:
  - BLOCK if max pairwise correlation > 0.85 (near-identical moves)
  - PENALIZE if max pairwise > 0.70 (reduce position size by 50%)
  - WARN if portfolio avg correlation > 0.60 (cluster risk)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ── Thresholds ───────────────────────────────────────────────────
_CORR_BLOCK_THRESHOLD = 0.85       # block new position above this
_CORR_PENALIZE_THRESHOLD = 0.70    # reduce size above this
_CORR_PENALTY_FACTOR = 0.50        # multiply quantity by this when penalized
_PORTFOLIO_AVG_CORR_WARN = 0.60    # warn when portfolio avg corr exceeds this
_MIN_OVERLAP_DAYS = 30             # need at least 30 overlapping return days
_ROLLING_WINDOW = 60               # use last 60 days for correlation


@dataclass
class CorrelationCheckResult:
    """Result of checking a candidate ticker against existing portfolio."""
    ticker: str = ""
    action: str = "PASS"               # PASS, PENALIZE, BLOCK
    max_pairwise_corr: float = 0.0     # highest corr with any holding
    correlated_with: str = ""          # ticker it's most correlated with
    portfolio_avg_corr: float = 0.0    # avg corr with all holdings
    num_high_corr_pairs: int = 0       # holdings with corr > penalize threshold
    reason: str = ""

    @property
    def is_blocked(self) -> bool:
        return self.action == "BLOCK"

    @property
    def is_penalized(self) -> bool:
        return self.action == "PENALIZE"


@dataclass
class CorrelationMatrix:
    """Computed correlation matrix with metadata."""
    tickers: list[str] = field(default_factory=list)
    matrix: np.ndarray = field(default_factory=lambda: np.array([]))
    overlap_days: int = 0
    timestamp: float = 0.0

    def get_correlation(self, ticker_a: str, ticker_b: str) -> float | None:
        """Get pairwise correlation between two tickers."""
        if ticker_a not in self.tickers or ticker_b not in self.tickers:
            return None
        i = self.tickers.index(ticker_a)
        j = self.tickers.index(ticker_b)
        return float(self.matrix[i, j])

    @property
    def summary(self) -> str:
        if len(self.tickers) < 2:
            return "CorrelationMatrix: <2 tickers, no pairs"
        # Get upper triangle (excluding diagonal)
        n = len(self.tickers)
        upper = [
            self.matrix[i, j]
            for i in range(n) for j in range(i + 1, n)
        ]
        if not upper:
            return "CorrelationMatrix: no pairs"
        avg = float(np.mean(upper))
        mx = float(np.max(upper))
        mn = float(np.min(upper))
        return (
            f"CorrelationMatrix: {n} tickers, {len(upper)} pairs | "
            f"avg={avg:.3f} max={mx:.3f} min={mn:.3f}"
        )


class CorrelationFilter:
    """
    Portfolio correlation risk filter.

    Computes rolling correlation matrix and checks candidate trades
    against existing holdings for excessive correlation.
    """

    def __init__(
        self,
        block_threshold: float = _CORR_BLOCK_THRESHOLD,
        penalize_threshold: float = _CORR_PENALIZE_THRESHOLD,
        penalty_factor: float = _CORR_PENALTY_FACTOR,
        portfolio_warn_threshold: float = _PORTFOLIO_AVG_CORR_WARN,
        min_overlap_days: int = _MIN_OVERLAP_DAYS,
        rolling_window: int = _ROLLING_WINDOW,
    ) -> None:
        self._block = block_threshold
        self._penalize = penalize_threshold
        self._penalty_factor = penalty_factor
        self._portfolio_warn = portfolio_warn_threshold
        self._min_overlap = min_overlap_days
        self._window = rolling_window

    # ══════════════════════════════════════════════════════════════
    # Core: Compute correlation matrix
    # ══════════════════════════════════════════════════════════════

    def compute_matrix(
        self,
        returns_map: dict[str, np.ndarray],
        tickers: list[str] | None = None,
    ) -> CorrelationMatrix:
        """
        Compute pairwise correlation matrix from return series.

        Parameters
        ----------
        returns_map : dict[str, np.ndarray]
            ticker → array of daily log returns
        tickers : list[str] | None
            subset of tickers to include (default: all in returns_map)

        Returns
        -------
        CorrelationMatrix with Pearson correlations.
        """
        if tickers is None:
            tickers = list(returns_map.keys())

        # Filter tickers with sufficient data
        valid_tickers = []
        for t in tickers:
            r = returns_map.get(t)
            if r is not None and len(r) >= self._min_overlap:
                valid_tickers.append(t)

        if len(valid_tickers) < 2:
            return CorrelationMatrix(tickers=valid_tickers)

        # Align returns to same length (use last N days)
        window = self._window
        aligned: list[np.ndarray] = []
        final_tickers: list[str] = []

        for t in valid_tickers:
            r = returns_map[t]
            if len(r) >= window:
                aligned.append(r[-window:])
                final_tickers.append(t)
            elif len(r) >= self._min_overlap:
                aligned.append(r[-self._min_overlap:])
                final_tickers.append(t)

        if len(final_tickers) < 2:
            return CorrelationMatrix(tickers=final_tickers)

        # Truncate all to same length (shortest aligned)
        min_len = min(len(a) for a in aligned)
        truncated = np.array([a[-min_len:] for a in aligned])  # shape: (n_tickers, n_days)

        # Compute Pearson correlation matrix
        # np.corrcoef expects rows as variables, columns as observations
        corr_matrix = np.corrcoef(truncated)

        # Handle NaN (constant series etc.)
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

        result = CorrelationMatrix(
            tickers=final_tickers,
            matrix=corr_matrix,
            overlap_days=min_len,
        )

        logger.debug("[CorrelationFilter] %s", result.summary)
        return result

    # ══════════════════════════════════════════════════════════════
    # Check: Should this new trade be allowed?
    # ══════════════════════════════════════════════════════════════

    def check_candidate(
        self,
        candidate_ticker: str,
        holding_tickers: list[str],
        returns_map: dict[str, np.ndarray],
    ) -> CorrelationCheckResult:
        """
        Check if a candidate ticker is too correlated with existing holdings.

        Parameters
        ----------
        candidate_ticker : ticker we want to buy
        holding_tickers : tickers we already hold
        returns_map : all available return series

        Returns
        -------
        CorrelationCheckResult with action (PASS / PENALIZE / BLOCK)
        """
        result = CorrelationCheckResult(ticker=candidate_ticker)

        if not holding_tickers:
            result.action = "PASS"
            result.reason = "No existing holdings — no correlation risk."
            return result

        if candidate_ticker not in returns_map:
            result.action = "PASS"
            result.reason = "No return data for candidate — skipping check."
            return result

        # Compute correlation between candidate and each holding
        candidate_returns = returns_map[candidate_ticker]
        if len(candidate_returns) < self._min_overlap:
            result.action = "PASS"
            result.reason = (
                f"Insufficient data ({len(candidate_returns)} days, "
                f"need {self._min_overlap})."
            )
            return result

        correlations: list[tuple[str, float]] = []

        for holding in holding_tickers:
            if holding == candidate_ticker:
                continue  # skip self
            h_returns = returns_map.get(holding)
            if h_returns is None or len(h_returns) < self._min_overlap:
                continue

            # Align to same window
            window = min(self._window, len(candidate_returns), len(h_returns))
            if window < self._min_overlap:
                continue

            c_slice = candidate_returns[-window:]
            h_slice = h_returns[-window:]

            # Pearson correlation
            corr = self._pearson(c_slice, h_slice)
            if corr is not None:
                correlations.append((holding, corr))

        if not correlations:
            result.action = "PASS"
            result.reason = "No valid correlation pairs found."
            return result

        # Find max pairwise correlation
        max_pair = max(correlations, key=lambda x: x[1])
        result.max_pairwise_corr = max_pair[1]
        result.correlated_with = max_pair[0]

        # Average correlation with all holdings
        result.portfolio_avg_corr = float(np.mean([c for _, c in correlations]))

        # Count high-correlation pairs
        result.num_high_corr_pairs = sum(
            1 for _, c in correlations if c > self._penalize
        )

        # ── Decision ─────────────────────────────────────────────
        if result.max_pairwise_corr >= self._block:
            result.action = "BLOCK"
            result.reason = (
                f"BLOCKED: {candidate_ticker} has {result.max_pairwise_corr:.3f} "
                f"correlation with {result.correlated_with} "
                f"(threshold {self._block:.2f}). "
                f"Buying both = near-duplicate exposure."
            )
            logger.warning("[CorrelationFilter] %s", result.reason)

        elif result.max_pairwise_corr >= self._penalize:
            result.action = "PENALIZE"
            result.reason = (
                f"PENALIZED: {candidate_ticker} has {result.max_pairwise_corr:.3f} "
                f"correlation with {result.correlated_with} "
                f"(threshold {self._penalize:.2f}). "
                f"Position size reduced by {(1 - self._penalty_factor):.0%}."
            )
            logger.info("[CorrelationFilter] %s", result.reason)

        else:
            result.action = "PASS"
            result.reason = (
                f"OK: max corr={result.max_pairwise_corr:.3f} with "
                f"{result.correlated_with} (below {self._penalize:.2f})."
            )
            logger.debug("[CorrelationFilter] %s %s", candidate_ticker, result.reason)

        return result

    # ══════════════════════════════════════════════════════════════
    # Batch: Filter a list of proposals
    # ══════════════════════════════════════════════════════════════

    def filter_proposals(
        self,
        proposals: list[Any],
        holding_tickers: list[str],
        returns_map: dict[str, np.ndarray],
    ) -> list[Any]:
        """
        Apply correlation filter to a batch of proposals.

        Modifies proposals in-place:
          - BLOCK: changes action to HOLD, zeroes quantity
          - PENALIZE: reduces quantity by penalty_factor

        Also prevents buying multiple correlated candidates in the
        same cycle (e.g., AAPL and MSFT both BUY in same cycle).

        Parameters
        ----------
        proposals : list of TradeProposal objects
        holding_tickers : tickers currently held in portfolio
        returns_map : all return series

        Returns
        -------
        Same list (modified in-place) for chaining.
        """
        from src.agents.decision.decision_fusion import TradeAction

        # Track what we're adding this cycle so we check against those too
        cycle_buys: list[str] = list(holding_tickers)
        blocked_count = 0
        penalized_count = 0

        for proposal in proposals:
            # Only check BUY proposals (SELL always allowed)
            if proposal.action != TradeAction.BUY:
                continue

            check = self.check_candidate(
                candidate_ticker=proposal.ticker,
                holding_tickers=cycle_buys,
                returns_map=returns_map,
            )

            if check.is_blocked:
                proposal.action = TradeAction.HOLD
                proposal.quantity = 0.0
                proposal.estimated_value = 0.0
                proposal.reasoning = (
                    f"[CORRELATION BLOCKED] {check.reason} | "
                    f"Original: {proposal.reasoning}"
                )
                blocked_count += 1
                logger.warning(
                    "[CorrelationFilter] Blocked BUY %s: corr=%.3f with %s",
                    proposal.ticker, check.max_pairwise_corr,
                    check.correlated_with,
                )

            elif check.is_penalized:
                old_qty = proposal.quantity
                proposal.quantity *= self._penalty_factor
                proposal.estimated_value = proposal.quantity * proposal.current_price
                proposal.reasoning = (
                    f"[CORRELATION PENALIZED] {check.reason} | "
                    f"Qty {old_qty:.4f} -> {proposal.quantity:.4f} | "
                    f"Original: {proposal.reasoning}"
                )
                penalized_count += 1
                # Still add to cycle_buys (we're still buying, just less)
                cycle_buys.append(proposal.ticker)
                logger.info(
                    "[CorrelationFilter] Penalized BUY %s: "
                    "qty %.4f -> %.4f (corr=%.3f with %s)",
                    proposal.ticker, old_qty, proposal.quantity,
                    check.max_pairwise_corr, check.correlated_with,
                )

            else:
                # PASS: add to cycle buys for cross-checking
                cycle_buys.append(proposal.ticker)

        if blocked_count + penalized_count > 0:
            logger.info(
                "[CorrelationFilter] Batch result: %d blocked, %d penalized, "
                "%d passed out of %d BUY proposals",
                blocked_count, penalized_count,
                sum(1 for p in proposals
                    if p.action == TradeAction.BUY) - penalized_count,
                sum(1 for p in proposals
                    if p.action in (TradeAction.BUY, TradeAction.HOLD)),
            )

        return proposals

    # ══════════════════════════════════════════════════════════════
    # Portfolio-level correlation report
    # ══════════════════════════════════════════════════════════════

    def portfolio_correlation_report(
        self,
        holding_tickers: list[str],
        returns_map: dict[str, np.ndarray],
    ) -> dict[str, Any]:
        """
        Generate a portfolio-level correlation risk report.

        Returns a dict with:
          - avg_correlation: mean pairwise correlation
          - max_correlation: highest pairwise correlation
          - max_pair: (ticker_a, ticker_b) with highest correlation
          - clusters: groups of highly correlated stocks
          - risk_level: LOW / MEDIUM / HIGH
        """
        matrix = self.compute_matrix(returns_map, holding_tickers)

        if len(matrix.tickers) < 2:
            return {
                "avg_correlation": 0.0,
                "max_correlation": 0.0,
                "max_pair": ("", ""),
                "clusters": [],
                "risk_level": "LOW",
                "n_holdings": len(holding_tickers),
            }

        n = len(matrix.tickers)
        pairs: list[tuple[str, str, float]] = []
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append((
                    matrix.tickers[i],
                    matrix.tickers[j],
                    float(matrix.matrix[i, j]),
                ))

        if not pairs:
            return {
                "avg_correlation": 0.0,
                "max_correlation": 0.0,
                "max_pair": ("", ""),
                "clusters": [],
                "risk_level": "LOW",
                "n_holdings": n,
            }

        avg_corr = float(np.mean([c for _, _, c in pairs]))
        max_pair = max(pairs, key=lambda x: x[2])

        # Find clusters: connected components where corr > penalize threshold
        clusters = self._find_clusters(matrix)

        # Risk level
        if avg_corr >= self._portfolio_warn:
            risk_level = "HIGH"
        elif max_pair[2] >= self._penalize:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        report = {
            "avg_correlation": avg_corr,
            "max_correlation": max_pair[2],
            "max_pair": (max_pair[0], max_pair[1]),
            "clusters": clusters,
            "risk_level": risk_level,
            "n_holdings": n,
            "n_pairs": len(pairs),
            "overlap_days": matrix.overlap_days,
        }

        if risk_level == "HIGH":
            logger.warning(
                "[CorrelationFilter] Portfolio correlation HIGH: "
                "avg=%.3f, max=%.3f (%s-%s), %d clusters",
                avg_corr, max_pair[2], max_pair[0], max_pair[1],
                len(clusters),
            )

        return report

    # ══════════════════════════════════════════════════════════════
    # Internal helpers
    # ══════════════════════════════════════════════════════════════

    @staticmethod
    def _pearson(a: np.ndarray, b: np.ndarray) -> float | None:
        """Compute Pearson correlation between two return series."""
        if len(a) != len(b) or len(a) < 5:
            return None
        if np.std(a) == 0 or np.std(b) == 0:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    def _find_clusters(self, matrix: CorrelationMatrix) -> list[list[str]]:
        """
        Find clusters of highly correlated stocks using simple
        connected components with correlation > penalize threshold.
        """
        n = len(matrix.tickers)
        if n < 2:
            return []

        # Build adjacency list
        adj: dict[int, set[int]] = {i: set() for i in range(n)}
        for i in range(n):
            for j in range(i + 1, n):
                if matrix.matrix[i, j] >= self._penalize:
                    adj[i].add(j)
                    adj[j].add(i)

        # BFS to find connected components
        visited: set[int] = set()
        clusters: list[list[str]] = []

        for start in range(n):
            if start in visited or not adj[start]:
                continue
            component: list[int] = []
            queue = [start]
            while queue:
                node = queue.pop(0)
                if node in visited:
                    continue
                visited.add(node)
                component.append(node)
                queue.extend(adj[node] - visited)

            if len(component) >= 2:
                clusters.append([matrix.tickers[i] for i in sorted(component)])

        return clusters
