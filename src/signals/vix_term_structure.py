"""
VIX Term Structure + HY OAS Macro Risk Switch
==============================================

State machine with hysteresis:
  NORMAL     → Full long exposure
  WARNING    → Half position size
  RISK_OFF   → 100% cash

Trigger (AND logic):
  VIX/VIX3M >= rv_trigger  AND  OAS 21d change > oas_trigger
  → RISK_OFF

Recovery (stricter):
  VIX/VIX3M < rv_recover  AND  OAS 21d change < oas_recover
  → NORMAL
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class Regime(str, Enum):
    NORMAL = "NORMAL"
    WARNING = "WARNING"
    RISK_OFF = "RISK_OFF"


@dataclass
class MacroSignal:
    date: str
    vix: float
    vix3m: float
    ratio: float
    oas: float
    oas_delta_21d: float
    regime: str
    signal: float  # 1.0 = full long, 0.5 = half, 0.0 = cash


class MacroRiskSwitch:
    """
    VIX term structure + HY OAS macro risk switch.

    Parameters are configurable for sensitivity analysis.
    """

    def __init__(
        self,
        rv_trigger: float = 1.00,    # VIX/VIX3M ratio → RISK_OFF
        rv_warning: float = 0.95,    # VIX/VIX3M ratio → WARNING
        rv_recover: float = 0.95,    # VIX/VIX3M ratio → back to NORMAL
        oas_trigger: float = 0.15,   # OAS 21d delta → RISK_OFF
        oas_recover: float = 0.10,   # OAS 21d delta → back to NORMAL
        oas_lookback: int = 21,      # days for OAS change calc
        min_hold_days: int = 5,      # minimum days before regime can change
    ):
        self.rv_trigger = rv_trigger
        self.rv_warning = rv_warning
        self.rv_recover = rv_recover
        self.oas_trigger = oas_trigger
        self.oas_recover = oas_recover
        self.oas_lookback = oas_lookback
        self.min_hold_days = min_hold_days

    def compute_signals(self, macro_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute regime signals for each date.

        Parameters
        ----------
        macro_df : pd.DataFrame
            Must contain columns: 'vix', 'vix3m', 'oas'
            Index should be DatetimeIndex.

        Returns
        -------
        pd.DataFrame with columns:
            vix, vix3m, ratio, oas, oas_sma, oas_delta_21d, regime, signal
        """
        df = macro_df[["vix", "vix3m", "oas"]].copy()
        df = df.dropna(subset=["vix", "vix3m"])

        if df.empty:
            return pd.DataFrame()

        # VIX/VIX3M ratio
        df["ratio"] = df["vix"] / df["vix3m"]

        # OAS 21-day moving average and change rate
        df["oas_sma"] = df["oas"].rolling(self.oas_lookback, min_periods=1).mean()
        df["oas_lagged"] = df["oas"].shift(self.oas_lookback)
        df["oas_delta_21d"] = np.where(
            df["oas_lagged"] > 0,
            (df["oas"] - df["oas_lagged"]) / df["oas_lagged"],
            0.0,
        )

        # State machine with hysteresis + minimum hold period
        regimes = []
        signals = []
        current_regime = Regime.NORMAL
        days_in_regime = 0

        for idx, row in df.iterrows():
            ratio = row["ratio"]
            oas_delta = row["oas_delta_21d"]
            days_in_regime += 1

            if pd.isna(ratio):
                regimes.append(current_regime.value)
                signals.append(1.0 if current_regime == Regime.NORMAL else
                              0.5 if current_regime == Regime.WARNING else 0.0)
                continue

            can_change = days_in_regime >= self.min_hold_days

            # OAS stress check (supplementary, not required for VIX-only trigger)
            oas_stressed = not pd.isna(oas_delta) and oas_delta > self.oas_trigger

            if current_regime == Regime.NORMAL and can_change:
                # RISK_OFF: VIX backwardation + OAS stress (both needed)
                if ratio >= self.rv_trigger and oas_stressed:
                    current_regime = Regime.RISK_OFF
                    days_in_regime = 0
                # WARNING: VIX approaching backwardation (VIX alone sufficient)
                elif ratio >= self.rv_warning:
                    current_regime = Regime.WARNING
                    days_in_regime = 0

            elif current_regime == Regime.WARNING and can_change:
                # Escalate to RISK_OFF if both conditions met
                if ratio >= self.rv_trigger and oas_stressed:
                    current_regime = Regime.RISK_OFF
                    days_in_regime = 0
                # Also escalate if VIX ratio is extreme (>1.05) even without OAS
                elif ratio >= self.rv_trigger + 0.05:
                    current_regime = Regime.RISK_OFF
                    days_in_regime = 0
                # De-escalate to NORMAL
                elif ratio < self.rv_recover:
                    current_regime = Regime.NORMAL
                    days_in_regime = 0

            elif current_regime == Regime.RISK_OFF and can_change:
                # Recovery requires stricter conditions
                if ratio < self.rv_recover and (pd.isna(oas_delta) or oas_delta < self.oas_recover):
                    current_regime = Regime.NORMAL
                    days_in_regime = 0

            regimes.append(current_regime.value)
            signal = {
                Regime.NORMAL: 1.0,
                Regime.WARNING: 0.5,
                Regime.RISK_OFF: 0.0,
            }[current_regime]
            signals.append(signal)

        df["regime"] = regimes
        df["signal"] = signals

        return df

    def get_signal_for_date(self, macro_df: pd.DataFrame,
                            date: pd.Timestamp) -> MacroSignal:
        """Get the macro signal for a specific date."""
        signals_df = self.compute_signals(macro_df.loc[macro_df.index <= date])

        if signals_df.empty:
            return MacroSignal(
                date=date.strftime("%Y-%m-%d"),
                vix=0, vix3m=0, ratio=0, oas=0,
                oas_delta_21d=0, regime="NORMAL", signal=1.0,
            )

        last = signals_df.iloc[-1]
        return MacroSignal(
            date=date.strftime("%Y-%m-%d"),
            vix=float(last.get("vix", 0)),
            vix3m=float(last.get("vix3m", 0)),
            ratio=float(last.get("ratio", 0)),
            oas=float(last.get("oas", 0)),
            oas_delta_21d=float(last.get("oas_delta_21d", 0)),
            regime=str(last["regime"]),
            signal=float(last["signal"]),
        )
