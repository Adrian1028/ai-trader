"""
Capital Allocation Tracker
===========================

Tracks NAV and exposure across the SPY macro signal system
and the OANDA forex bot. Enforces capital isolation.

Architecture:
  - SPY System: Trading 212 Demo → SPY/SHV toggle
  - OANDA Bot: Separate forex system (if running)
  - Total NAV = SPY NAV + OANDA NAV
  - Each system has independent risk budgets
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class SystemNAV:
    name: str
    initial_capital: float
    current_nav: float
    cash: float
    positions_value: float
    last_updated: str
    pnl_pct: float = 0.0

    def update(self, nav: float, cash: float, positions: float):
        self.current_nav = nav
        self.cash = cash
        self.positions_value = positions
        self.last_updated = datetime.utcnow().isoformat()
        self.pnl_pct = (nav / self.initial_capital - 1) * 100 if self.initial_capital > 0 else 0.0


class AllocationTracker:
    """Track capital across trading systems."""

    def __init__(self, state_file: str = "data/capital_state.json"):
        self.state_file = Path(state_file)
        self.systems: dict[str, SystemNAV] = {}
        self._load_state()

    def register_system(self, name: str, initial_capital: float):
        """Register a trading system with its allocated capital."""
        if name not in self.systems:
            self.systems[name] = SystemNAV(
                name=name,
                initial_capital=initial_capital,
                current_nav=initial_capital,
                cash=initial_capital,
                positions_value=0.0,
                last_updated=datetime.utcnow().isoformat(),
            )
            logger.info("Registered system '%s' with $%.2f capital", name, initial_capital)
        self._save_state()

    def update_nav(self, system_name: str, nav: float, cash: float, positions: float):
        """Update a system's NAV snapshot."""
        if system_name not in self.systems:
            logger.error("Unknown system: %s", system_name)
            return
        self.systems[system_name].update(nav, cash, positions)
        self._save_state()

    @property
    def total_nav(self) -> float:
        return sum(s.current_nav for s in self.systems.values())

    @property
    def total_initial(self) -> float:
        return sum(s.initial_capital for s in self.systems.values())

    @property
    def total_pnl_pct(self) -> float:
        if self.total_initial <= 0:
            return 0.0
        return (self.total_nav / self.total_initial - 1) * 100

    def get_allocation_report(self) -> dict:
        """Full allocation report for logging/Discord."""
        return {
            "total_nav": self.total_nav,
            "total_initial": self.total_initial,
            "total_pnl_pct": self.total_pnl_pct,
            "systems": {name: asdict(s) for name, s in self.systems.items()},
            "timestamp": datetime.utcnow().isoformat(),
        }

    def _save_state(self):
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        data = {name: asdict(s) for name, s in self.systems.items()}
        with open(self.state_file, "w") as f:
            json.dump(data, f, indent=2)

    def _load_state(self):
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    data = json.load(f)
                for name, d in data.items():
                    self.systems[name] = SystemNAV(**d)
                logger.info("Loaded capital state: %d systems", len(self.systems))
            except Exception as e:
                logger.warning("Failed to load capital state: %s", e)
