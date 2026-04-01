"""
Risk Management Agent
=====================
Computes dynamic position sizing, VaR, and stop-loss / take-profit
boundaries. Acts as a quantitative risk gate between the intelligence
layer and the execution layer.

Key responsibilities:
  - Historical VaR (parametric + Monte Carlo)
  - Dynamic stop-loss / take-profit using ATR multiples
  - **Adaptive Kelly Criterion** — dynamic fraction scaling:
      * Confidence-scaled: low confidence → quarter-Kelly, high → three-quarter
      * Drawdown protection: auto-reduce size when account is losing
      * Volatility regime: high ATR% → shrink Kelly, low ATR% → expand
      * MTF confluence bonus: all timeframes agree → allow larger fraction
  - Portfolio-level exposure limits
  - Correlation-aware concentration checks
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.core.regime_detector import RegimeSnapshot
    from src.core.virtual_account import VirtualSubAccount
    from src.memory.episodic_memory import EpisodicMemory

logger = logging.getLogger(__name__)

# ── 帳戶層面安全常數 ──────────────────────────────────────────────
_CASH_BUFFER_PCT = 0.01        # 保留 1% 現金做滑價緩衝
_MIN_TRADEABLE_QTY = 0.0001    # T212 碎股最小精度

# ── 情節記憶融合常數 ──────────────────────────────────────────────
_MEMORY_BLEND_WEIGHT = 0.30    # 記憶最高佔 Kelly 估計的 30%
_MIN_MEMORY_TRADES = 5         # 至少 5 筆交易才啟用記憶融合

# ── 動態 Kelly 縮放常數 ──────────────────────────────────────────
# Kelly 分數乘以一個 [KELLY_FLOOR, KELLY_CEIL] 之間的動態縮放係數
_KELLY_FLOOR = 0.25            # 最保守：25% Kelly (低信心 + 高波動 + 回撤中)
_KELLY_CEIL = 0.80             # 最激進：80% Kelly (高信心 + 低波動 + MTF共振)
_KELLY_BASE = 0.50             # 基準：50% Kelly (標準條件)

# 回撤保護閾值
_DRAWDOWN_MILD = 0.03          # 3% 回撤 → 開始減倉
_DRAWDOWN_SEVERE = 0.08        # 8% 回撤 → 大幅減倉
_DRAWDOWN_CRITICAL = 0.12      # 12% 回撤 → 最小倉位

# 波動率體制閾值 (ATR 佔價格百分比)
_VOL_LOW = 1.5                 # ATR% < 1.5 = 低波動 → 可放大
_VOL_HIGH = 3.5                # ATR% > 3.5 = 高波動 → 要縮小


class RiskVerdict(Enum):
    APPROVED = auto()
    REDUCED = auto()        # approved with reduced size
    REJECTED = auto()


@dataclass
class RiskEnvelope:
    """Risk parameters computed for a single trade proposal."""
    verdict: RiskVerdict = RiskVerdict.REJECTED
    reason: str = ""

    # Position sizing
    suggested_quantity: float = 0.0
    max_position_value: float = 0.0

    # Stop / take-profit
    stop_loss_price: float = 0.0
    take_profit_price: float = 0.0
    stop_loss_pct: float = 0.0
    take_profit_pct: float = 0.0

    # Risk metrics
    var_95: float = 0.0          # 1-day 95% VaR (absolute)
    var_99: float = 0.0          # 1-day 99% VaR (absolute)
    expected_shortfall: float = 0.0
    cvar_95: float = 0.0         # 95% Conditional VaR (absolute)
    cvar_99: float = 0.0         # 99% Conditional VaR (absolute)
    risk_reward_ratio: float = 0.0

    # Kelly (Adaptive)
    kelly_fraction: float = 0.0
    half_kelly_fraction: float = 0.0        # legacy (kept for backward compat)
    kelly_scale_factor: float = 0.0         # dynamic scaling applied
    adaptive_kelly_fraction: float = 0.0    # kelly_fraction * kelly_scale_factor

    timestamp: float = field(default_factory=time.time)


@dataclass
class PortfolioRiskState:
    """Current portfolio-level risk snapshot."""
    total_nav: float = 0.0
    invested_value: float = 0.0
    cash: float = 0.0
    exposure_pct: float = 0.0           # invested / NAV
    positions: dict[str, float] = field(default_factory=dict)  # ISIN → value
    max_single_position_pct: float = 0.0

    @property
    def concentration(self) -> float:
        """Herfindahl index (0 = diversified, 1 = single stock)."""
        if self.total_nav <= 0:
            return 1.0
        weights = [v / self.total_nav for v in self.positions.values()]
        return sum(w ** 2 for w in weights)


class RiskAgent:
    """
    Deterministic + statistical risk engine.
    NOT an LLM agent — purely mathematical.
    """

    def __init__(
        self,
        max_single_position_pct: float = 0.05,     # 5% of NAV per position
        max_portfolio_exposure_pct: float = 0.95,   # max 95% invested
        max_var_pct_of_nav: float = 0.02,           # max 2% daily VaR
        atr_stop_multiplier: float = 2.0,
        atr_tp_multiplier: float = 3.0,
        var_confidence: float = 0.95,
        monte_carlo_sims: int = 10_000,
        episodic_memory: "EpisodicMemory | None" = None,
    ) -> None:
        self._max_pos_pct = max_single_position_pct
        self._max_exposure = max_portfolio_exposure_pct
        self._max_var_pct = max_var_pct_of_nav
        self._atr_stop_mult = atr_stop_multiplier
        self._atr_tp_mult = atr_tp_multiplier
        self._var_conf = var_confidence
        self._mc_sims = monte_carlo_sims
        self._memory = episodic_memory

    def evaluate(
        self,
        direction: int,                 # +1 buy, -1 sell
        current_price: float,
        returns: np.ndarray,            # historical daily returns
        atr: float | None,              # average true range (from TechnicalAgent)
        confidence: float,              # intelligence fused confidence (0-1)
        portfolio: PortfolioRiskState,
        confluence_score: float = 0.0,  # MTF confluence bonus (from TechnicalAgent)
        regime: "RegimeSnapshot | None" = None,  # Phase 5: market regime
    ) -> RiskEnvelope:
        """
        Compute full risk envelope for a proposed trade.

        Parameters
        ----------
        direction : +1 (buy) or -1 (sell/short)
        current_price : current market price
        returns : array of historical daily log returns
        atr : average true range (14-day), or None
        confidence : fused signal confidence from intelligence layer
        portfolio : current portfolio risk state
        confluence_score : multi-timeframe confluence score (±20 = full alignment)
        """
        envelope = RiskEnvelope()

        if len(returns) < 20:
            envelope.reason = "Insufficient return history (need >= 20 days)"
            return envelope

        # ── 1. VaR computation ────────────────────────────────────────
        var_95, var_99, es, cvar_95, cvar_99 = self._compute_var(returns, current_price)
        envelope.var_95 = var_95
        envelope.var_99 = var_99
        envelope.expected_shortfall = es
        envelope.cvar_95 = cvar_95
        envelope.cvar_99 = cvar_99

        # ── 2. Portfolio exposure check ───────────────────────────────
        available_for_new = portfolio.total_nav * self._max_exposure - portfolio.invested_value
        if available_for_new <= 0 and direction > 0:
            envelope.reason = (
                f"Portfolio exposure {portfolio.exposure_pct:.0%} exceeds "
                f"limit {self._max_exposure:.0%} — no room for new buys."
            )
            return envelope

        # ── 3. Max position size ──────────────────────────────────────
        max_pos_value = portfolio.total_nav * self._max_pos_pct
        max_pos_value = min(max_pos_value, available_for_new, portfolio.cash)
        envelope.max_position_value = max_pos_value

        # ── 4. VaR-adjusted size cap ──────────────────────────────────
        max_var_absolute = portfolio.total_nav * self._max_var_pct
        if var_95 > 0:
            var_per_unit = var_95 / current_price
            max_units_by_var = max_var_absolute / var_per_unit if var_per_unit > 0 else float("inf")
            var_capped_value = max_units_by_var * current_price
            max_pos_value = min(max_pos_value, var_capped_value)

        # ── 5. Kelly criterion (raw) ─────────────────────────────────
        kelly = self._kelly_fraction(returns, confidence, direction)
        envelope.kelly_fraction = kelly
        envelope.half_kelly_fraction = kelly / 2.0  # legacy compat

        # ── 6. Adaptive Kelly scaling ────────────────────────────────
        # Dynamically scale Kelly fraction based on market conditions
        atr_pct = (atr / current_price * 100) if (atr and current_price > 0) else 2.0
        scale = self._adaptive_kelly_scale(
            confidence=confidence,
            atr_pct=atr_pct,
            portfolio=portfolio,
            confluence_score=confluence_score,
            regime=regime,
        )
        envelope.kelly_scale_factor = scale
        adaptive_kelly = kelly * scale
        envelope.adaptive_kelly_fraction = adaptive_kelly

        kelly_value = portfolio.total_nav * max(adaptive_kelly, 0.0)
        final_value = min(max_pos_value, kelly_value) if kelly > 0 else 0.0

        # ── 7. Adaptive stop-loss / take-profit (ATR-based) ────────────
        if atr is not None and atr > 0:
            sl_mult, tp_mult = self._adaptive_atr_multiples(
                atr_pct=atr_pct,
                confidence=confidence,
                confluence_score=confluence_score,
            )

            if direction > 0:  # long
                envelope.stop_loss_price = current_price - sl_mult * atr
                envelope.take_profit_price = current_price + tp_mult * atr
            else:  # closing / selling
                envelope.stop_loss_price = current_price + sl_mult * atr
                envelope.take_profit_price = current_price - tp_mult * atr

            envelope.stop_loss_pct = abs(envelope.stop_loss_price - current_price) / current_price
            envelope.take_profit_pct = abs(envelope.take_profit_price - current_price) / current_price

            if envelope.take_profit_pct > 0 and envelope.stop_loss_pct > 0:
                envelope.risk_reward_ratio = envelope.take_profit_pct / envelope.stop_loss_pct

        # ── 8. Final verdict ──────────────────────────────────────────
        if final_value <= 0 or (direction > 0 and kelly <= 0):
            envelope.verdict = RiskVerdict.REJECTED
            envelope.reason = (
                f"Kelly fraction non-positive ({kelly:.4f}) or no allocable capital."
            )
            return envelope

        quantity = final_value / current_price
        envelope.suggested_quantity = round(quantity, 6)

        if final_value < max_pos_value * 0.5:
            envelope.verdict = RiskVerdict.REDUCED
            envelope.reason = (
                f"Position reduced from max {max_pos_value:.0f} to {final_value:.0f} "
                f"(VaR/Kelly constraint). "
                f"Kelly scale={scale:.2f} (conf={confidence:.2f}, "
                f"ATR%={atr_pct:.1f}, confluence={confluence_score:+.0f})"
            )
        else:
            envelope.verdict = RiskVerdict.APPROVED
            envelope.reason = (
                f"Risk OK. Kelly scale={scale:.2f} "
                f"(conf={confidence:.2f}, ATR%={atr_pct:.1f}, "
                f"confluence={confluence_score:+.0f})"
            )

        return envelope

    # ── VirtualSubAccount-aware evaluation ────────────────────────────

    def evaluate_with_account(
        self,
        direction: int,
        current_price: float,
        returns: np.ndarray,
        atr: float | None,
        confidence: float,
        virtual_account: "VirtualSubAccount",
        price_map: dict[str, float] | None = None,
        symbol: str = "",
        confluence_score: float = 0.0,
        regime: "RegimeSnapshot | None" = None,
    ) -> RiskEnvelope:
        """
        完整的帳戶感知風險評估。

        在基礎 evaluate() 之上加入三層虛擬帳戶安全閥：
        1. 現金緩衝 — 買入時保留 1% 做滑價緩衝
        2. 持倉上限 — 賣出時不超過實際持有量
        3. 自動降階 — 現金不足時自動縮減到可負擔的量

        Parameters
        ----------
        direction : +1 (buy) or -1 (sell)
        current_price : 最新市價
        returns : 歷史日收益率陣列
        atr : 14 日 ATR (可為 None)
        confidence : 情報融合信心度 (0-1)
        virtual_account : 虛擬子帳戶
        price_map : ticker → 最新市價 (用於精確估值)
        symbol : 交易標的 ticker
        confluence_score : MTF 共振分數 (from TechnicalAgent)
        """
        # 先建立 PortfolioRiskState 快照
        portfolio = self.build_virtual_portfolio_state(virtual_account, price_map)

        # 呼叫核心風險評估
        envelope = self.evaluate(
            direction=direction,
            current_price=current_price,
            returns=returns,
            atr=atr,
            confidence=confidence,
            portfolio=portfolio,
            confluence_score=confluence_score,
            regime=regime,
        )

        if envelope.verdict == RiskVerdict.REJECTED:
            return envelope

        # ── 帳戶層面安全閥 ──────────────────────────────────────
        suggested_qty = envelope.suggested_quantity
        suggested_value = suggested_qty * current_price

        if direction > 0:
            # 買入：扣掉 1% 緩衝後的安全現金
            safe_cash = virtual_account.available_cash * (1.0 - _CASH_BUFFER_PCT)

            if safe_cash <= 0:
                envelope.verdict = RiskVerdict.REJECTED
                envelope.reason = (
                    f"虛擬帳戶現金不足 (可用: {virtual_account.available_cash:.2f})，"
                    f"拒絕買入。"
                )
                envelope.suggested_quantity = 0.0
                logger.warning(
                    "[RiskAgent] %s 現金不足，拒絕買入。", symbol,
                )
                return envelope

            if suggested_value > safe_cash:
                # 自動降階到可負擔的量
                old_qty = suggested_qty
                suggested_qty = math.floor(
                    (safe_cash / current_price) * 10000
                ) / 10000.0
                suggested_value = suggested_qty * current_price

                if suggested_qty < _MIN_TRADEABLE_QTY:
                    envelope.verdict = RiskVerdict.REJECTED
                    envelope.reason = (
                        f"降階後數量 ({suggested_qty:.4f}) 低於最小可交易量，"
                        f"拒絕買入。"
                    )
                    envelope.suggested_quantity = 0.0
                    return envelope

                envelope.verdict = RiskVerdict.REDUCED
                envelope.reason = (
                    f"現金不足以買滿 (需要 {old_qty * current_price:.2f}, "
                    f"安全現金 {safe_cash:.2f})，降階至 {suggested_qty:.4f} 股。"
                )
                envelope.suggested_quantity = suggested_qty
                logger.info(
                    "[RiskAgent] %s 現金不足以買滿，降階 %.4f → %.4f",
                    symbol, old_qty, suggested_qty,
                )

        elif direction < 0:
            # 賣出：不能賣超過持有量
            owned_qty = virtual_account.get_position_qty(symbol)
            if owned_qty <= 0:
                envelope.verdict = RiskVerdict.REJECTED
                envelope.reason = (
                    f"虛擬帳戶內無 {symbol} 持倉，無法賣出。"
                )
                envelope.suggested_quantity = 0.0
                logger.warning(
                    "[RiskAgent] %s 無持倉，拒絕賣出。", symbol,
                )
                return envelope

            if suggested_qty > owned_qty:
                envelope.suggested_quantity = owned_qty
                envelope.verdict = RiskVerdict.REDUCED
                envelope.reason = (
                    f"賣出量 ({suggested_qty:.4f}) 超過持有量 "
                    f"({owned_qty:.4f})，已限制至持有量。"
                )
                logger.info(
                    "[RiskAgent] %s 賣出量受限：%.4f → %.4f",
                    symbol, suggested_qty, owned_qty,
                )

        # 更新 max_position_value 以反映實際可用額度
        envelope.max_position_value = envelope.suggested_quantity * current_price
        return envelope

    # ── Adaptive Kelly scaling ─────────────────────────────────────────

    @staticmethod
    def _adaptive_kelly_scale(
        confidence: float,
        atr_pct: float,
        portfolio: PortfolioRiskState,
        confluence_score: float = 0.0,
        regime: "RegimeSnapshot | None" = None,
    ) -> float:
        """
        計算 Kelly 動態縮放係數。

        五個維度加權決定最終縮放：
          1. 信心度 (35%) — 高信心 → 放大, 低信心 → 縮小
          2. 波動率 (20%) — 低波動 → 放大, 高波動 → 縮小
          3. 回撤保護 (20%) — 帳戶虧錢時自動收縮
          4. MTF共振 (10%) — 三框架同向 → 加碼
          5. 市場體制 (15%) — BULL → 放大, BEAR → 縮小, SIDEWAYS → 中等

        Returns
        -------
        float : 縮放係數 [_KELLY_FLOOR, _KELLY_CEIL]，通常在 0.15 ~ 0.60
        """
        # ── 1. 信心度因子 (0.0 ~ 1.0) ─────────────────────────
        # confidence: 0.0 → factor=0.0, 0.5 → 0.5, 1.0 → 1.0
        conf_factor = max(0.0, min(confidence, 1.0))

        # ── 2. 波動率因子 (inverted: high vol → low factor) ───
        if atr_pct <= _VOL_LOW:
            vol_factor = 1.0      # 低波動 → 最大
        elif atr_pct >= _VOL_HIGH:
            vol_factor = 0.3      # 高波動 → 保守但不歸零
        else:
            # 線性插值，下限 0.3
            vol_factor = max(0.3, 1.0 - (atr_pct - _VOL_LOW) / (_VOL_HIGH - _VOL_LOW))

        # ── 3. 回撤保護因子 ──────────────────────────────────
        drawdown_factor = 1.0
        if portfolio.total_nav > 0:
            cash_ratio = portfolio.cash / portfolio.total_nav
            risk_heat = portfolio.exposure_pct * (1 + portfolio.concentration)
            if risk_heat > 1.5:
                drawdown_factor = 0.5  # 集中度高 + 高曝險 → 大幅減倉
            elif risk_heat > 1.0:
                drawdown_factor = 0.75  # 中度風險

        # ── 4. MTF 共振因子 ──────────────────────────────────
        if abs(confluence_score) >= 20:
            confluence_factor = 1.0
        elif confluence_score <= -15:
            confluence_factor = 0.0
        else:
            confluence_factor = 0.5

        # ── 5. 市場體制因子 (Phase 5) ────────────────────────
        regime_factor = 0.5  # default: no regime data → neutral
        if regime is not None:
            from src.core.regime_detector import MarketRegime
            if regime.regime == MarketRegime.BULL:
                # 牛市 → 允許更大倉位，按信心度加權
                regime_factor = 0.7 + 0.3 * regime.confidence
            elif regime.regime == MarketRegime.BEAR:
                # 熊市 → 縮減倉位但保留交易能力
                regime_factor = 0.3 - 0.1 * regime.confidence
                regime_factor = max(regime_factor, 0.15)
            else:
                # SIDEWAYS → 中等
                regime_factor = 0.55

        # ── 加權融合 ──────────────────────────────────────────
        # 信心(35%) + 波動(20%) + 回撤(20%) + 共振(10%) + 體制(15%)
        raw = (
            0.35 * conf_factor
            + 0.20 * vol_factor
            + 0.20 * drawdown_factor
            + 0.10 * confluence_factor
            + 0.15 * regime_factor
        )

        # 映射到 [KELLY_FLOOR, KELLY_CEIL]
        scale = _KELLY_FLOOR + raw * (_KELLY_CEIL - _KELLY_FLOOR)

        regime_name = regime.regime.value if regime else "N/A"
        logger.debug(
            "[Kelly] scale=%.3f (conf=%.2f vol=%.2f dd=%.2f mtf=%.2f regime=%.2f[%s]) "
            "→ raw=%.3f",
            scale, conf_factor, vol_factor, drawdown_factor,
            confluence_factor, regime_factor, regime_name, raw,
        )

        return scale

    # ── Adaptive ATR-based SL/TP ──────────────────────────────────────

    def _adaptive_atr_multiples(
        self,
        atr_pct: float,
        confidence: float,
        confluence_score: float = 0.0,
    ) -> tuple[float, float]:
        """
        動態計算 ATR 止損/止盈倍數。

        三個維度影響倍數：

        1. **波動率體制** (ATR%)
           - 高波動 (ATR% > 3.5): 放寬止損避免被震出 → SL ×1.3, TP ×1.2
           - 低波動 (ATR% < 1.5): 收緊止損鎖利 → SL ×0.8, TP ×0.9

        2. **信心度**
           - 高信心 (>0.8): 收緊止損 (更有把握), 放寬止盈 (讓利潤跑)
           - 低信心 (<0.4): 收緊止盈 (快速獲利了結)

        3. **MTF 共振**
           - 三框架同向: 放寬止盈 (趨勢可能延續更遠)
           - 框架分歧: 收緊止盈 (不確定性高)

        Returns
        -------
        (sl_multiplier, tp_multiplier) : 動態調整後的 ATR 倍數
        """
        base_sl = self._atr_stop_mult  # default 2.0
        base_tp = self._atr_tp_mult    # default 3.0

        # ── 1. 波動率體制調整 ─────────────────────────────────
        if atr_pct >= _VOL_HIGH:
            # 高波動：放寬止損避免假突破震出
            vol_sl_factor = 1.3
            vol_tp_factor = 1.2
        elif atr_pct <= _VOL_LOW:
            # 低波動：收緊止損鎖定利潤
            vol_sl_factor = 0.8
            vol_tp_factor = 0.9
        else:
            # 線性插值
            t = (atr_pct - _VOL_LOW) / (_VOL_HIGH - _VOL_LOW)
            vol_sl_factor = 0.8 + t * (1.3 - 0.8)
            vol_tp_factor = 0.9 + t * (1.2 - 0.9)

        # ── 2. 信心度調整 ─────────────────────────────────────
        if confidence >= 0.8:
            # 高信心：收緊止損 (更少容忍), 放寬止盈 (讓利潤跑)
            conf_sl_factor = 0.85
            conf_tp_factor = 1.25
        elif confidence <= 0.4:
            # 低信心：放寬止損 (給更多空間), 收緊止盈 (快走)
            conf_sl_factor = 1.15
            conf_tp_factor = 0.80
        else:
            # 中等信心：線性插值
            t = (confidence - 0.4) / 0.4
            conf_sl_factor = 1.15 - t * (1.15 - 0.85)
            conf_tp_factor = 0.80 + t * (1.25 - 0.80)

        # ── 3. MTF 共振調整 ───────────────────────────────────
        if abs(confluence_score) >= 20:
            # 三框架同向：放寬止盈 (趨勢可能延續更遠)
            mtf_sl_factor = 1.0
            mtf_tp_factor = 1.30
        elif confluence_score <= -15:
            # 框架分歧：收緊止盈 (不確定性高)
            mtf_sl_factor = 1.0
            mtf_tp_factor = 0.75
        else:
            mtf_sl_factor = 1.0
            mtf_tp_factor = 1.0

        # ── 融合：各因子相乘 ──────────────────────────────────
        final_sl = base_sl * vol_sl_factor * conf_sl_factor * mtf_sl_factor
        final_tp = base_tp * vol_tp_factor * conf_tp_factor * mtf_tp_factor

        # ── 安全邊界 ──────────────────────────────────────────
        # SL 不能太小 (至少 1.0×ATR) 也不能太大 (最多 4.0×ATR)
        final_sl = max(1.0, min(4.0, final_sl))
        # TP 不能太小 (至少 1.5×ATR) 也不能太大 (最多 6.0×ATR)
        final_tp = max(1.5, min(6.0, final_tp))

        # ── 最低風險回報比保護 ────────────────────────────────
        # 確保 TP/SL >= 1.5 (風險回報比至少 1.5:1)
        if final_tp / final_sl < 1.5:
            final_tp = final_sl * 1.5

        logger.debug(
            "[SL/TP] ATR%%=%.1f conf=%.2f mtf=%+.0f → "
            "SL=%.2f×ATR (base %.1f × vol %.2f × conf %.2f) "
            "TP=%.2f×ATR (base %.1f × vol %.2f × conf %.2f × mtf %.2f) "
            "R:R=%.2f",
            atr_pct, confidence, confluence_score,
            final_sl, base_sl, vol_sl_factor, conf_sl_factor,
            final_tp, base_tp, vol_tp_factor, conf_tp_factor, mtf_tp_factor,
            final_tp / final_sl,
        )

        return final_sl, final_tp

    # ── VaR internals ─────────────────────────────────────────────────

    def _compute_var(
        self, returns: np.ndarray, price: float,
    ) -> tuple[float, float, float, float, float]:
        """
        Returns (VaR_95, VaR_99, Expected_Shortfall, CVaR_95, CVaR_99)
        as absolute values for a 1-unit position at `price`.
        """
        mu = np.mean(returns)
        sigma = np.std(returns, ddof=1)

        if sigma == 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0

        # Parametric VaR (normal assumption)
        z_95 = 1.645
        z_99 = 2.326
        var_95 = price * (z_95 * sigma - mu)
        var_99 = price * (z_99 * sigma - mu)

        # Monte Carlo VaR for validation
        mc_returns = np.random.normal(mu, sigma, self._mc_sims)
        mc_losses = -mc_returns * price
        mc_losses_sorted = np.sort(mc_losses)

        mc_var_95 = mc_losses_sorted[int(0.95 * self._mc_sims)]

        # Expected Shortfall (CVaR): average of losses beyond VaR
        tail_losses_95 = mc_losses_sorted[int(0.95 * self._mc_sims):]
        es = float(np.mean(tail_losses_95)) if len(tail_losses_95) > 0 else var_95

        # CVaR at 95% and 99% (tail risk beyond VaR)
        cvar_95 = es  # CVaR_95 = ES at 95%
        tail_losses_99 = mc_losses_sorted[int(0.99 * self._mc_sims):]
        cvar_99 = float(np.mean(tail_losses_99)) if len(tail_losses_99) > 0 else var_99

        # Use conservative estimate (max of parametric & MC)
        final_var_95 = max(var_95, mc_var_95)

        return abs(final_var_95), abs(var_99), abs(es), abs(cvar_95), abs(cvar_99)

    def _kelly_fraction(
        self, returns: np.ndarray, confidence: float, direction: int,
    ) -> float:
        """
        Kelly criterion: f* = (p*b - q) / b
        where p = win probability, b = avg_win / avg_loss, q = 1 - p.

        We bias p by the intelligence confidence score.
        When episodic memory is available, blend in real trade outcomes
        (30% memory weight) to replace hardcoded win_rate/win_loss_ratio.
        """
        if direction > 0:
            wins = returns[returns > 0]
            losses = returns[returns < 0]
        else:
            wins = returns[returns < 0]   # for shorts, down = win
            losses = returns[returns > 0]

        if len(wins) == 0 or len(losses) == 0:
            return 0.0

        # Base win rate from historical price data
        base_p = len(wins) / len(returns)
        avg_win = float(np.mean(np.abs(wins)))
        avg_loss = float(np.mean(np.abs(losses)))

        if avg_loss == 0:
            return 0.0

        b = avg_win / avg_loss

        # ── 情節記憶融合 ────────────────────────────────────────
        # 若有足夠的真實交易記憶，用記憶的勝率與盈虧比修正估計
        if self._memory is not None:
            action_filter = "BUY" if direction > 0 else "SELL"
            perf = self._memory.get_recent_performance(
                limit=30, action_filter=action_filter,
            )
            if perf["total_trades"] >= _MIN_MEMORY_TRADES:
                # 記憶權重：記憶越多越可信 (最高 30%)
                memory_weight = min(
                    perf["total_trades"] / 100.0,
                    _MEMORY_BLEND_WEIGHT,
                )
                base_p = (1.0 - memory_weight) * base_p + memory_weight * perf["win_rate"]
                b = (1.0 - memory_weight) * b + memory_weight * perf["win_loss_ratio"]

                logger.debug(
                    "[RiskAgent] 記憶融合: %d 筆交易, 勝率=%.2f, 盈虧比=%.2f, "
                    "融合權重=%.2f",
                    perf["total_trades"], perf["win_rate"],
                    perf["win_loss_ratio"], memory_weight,
                )

        # Blend with intelligence confidence (50/50 prior vs signal)
        p = 0.5 * base_p + 0.5 * confidence
        q = 1.0 - p

        kelly = (p * b - q) / b

        return max(kelly, 0.0)  # never go negative

    # ── portfolio-level risk snapshot ─────────────────────────────────

    @staticmethod
    def build_portfolio_state(
        account_info: dict[str, Any],
        portfolio_positions: list[dict[str, Any]],
    ) -> PortfolioRiskState:
        """Construct PortfolioRiskState from T212 API responses."""
        nav = float(account_info.get("value", 0))
        cash = float(account_info.get("free", 0))

        positions: dict[str, float] = {}
        invested = 0.0
        for pos in portfolio_positions:
            isin = pos.get("ticker", "")
            current_val = float(pos.get("currentPrice", 0)) * float(pos.get("quantity", 0))
            positions[isin] = current_val
            invested += current_val

        max_pos_pct = 0.0
        if nav > 0 and positions:
            max_pos_pct = max(positions.values()) / nav

        return PortfolioRiskState(
            total_nav=nav,
            invested_value=invested,
            cash=cash,
            exposure_pct=invested / nav if nav > 0 else 0.0,
            positions=positions,
            max_single_position_pct=max_pos_pct,
        )

    @staticmethod
    def build_virtual_portfolio_state(
        virtual_account: "VirtualSubAccount",
        price_map: dict[str, float] | None = None,
    ) -> PortfolioRiskState:
        """
        從虛擬子帳戶構建 PortfolioRiskState。
        風險評估時應使用此方法取代 build_portfolio_state，
        確保機械人只看到自己的虛擬資金與持倉。

        Parameters
        ----------
        virtual_account : 虛擬子帳戶實例
        price_map : ticker → 最新市價（若有），用於更準確的持倉估值
        """
        positions: dict[str, float] = {}
        invested = 0.0

        for ticker, vpos in virtual_account.positions.items():
            # 如果有最新市價，用市價估值；否則用平均成本
            if price_map and ticker in price_map:
                current_val = vpos.quantity * price_map[ticker]
            else:
                current_val = vpos.quantity * vpos.average_price
            positions[ticker] = current_val
            invested += current_val

        cash = virtual_account.available_cash
        nav = cash + invested

        max_pos_pct = 0.0
        if nav > 0 and positions:
            max_pos_pct = max(positions.values()) / nav

        return PortfolioRiskState(
            total_nav=nav,
            invested_value=invested,
            cash=cash,
            exposure_pct=invested / nav if nav > 0 else 0.0,
            positions=positions,
            max_single_position_pct=max_pos_pct,
        )
