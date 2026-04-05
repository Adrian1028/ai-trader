"""
Gate Check Module — Automated pass/fail verification for each phase.

Usage:
    python -m backtest.gate_check --phase 1 --results-dir results/
    python -m backtest.gate_check --phase 2 --results-dir results/
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class GateResult:
    name: str
    threshold: str
    actual: str
    passed: bool


def load_result(results_dir: str, filename: str) -> dict | None:
    path = os.path.join(results_dir, filename)
    if not os.path.exists(path):
        logger.warning("Result file not found: %s", path)
        return None
    with open(path) as f:
        return json.load(f)


# ─── Phase 1 Gates ────────────────────────────────────────────────────────────

def check_phase1(results_dir: str) -> list[GateResult]:
    """
    Phase 1 pass conditions (ALL must be met on TEST set):
      1. Sharpe Ratio > 1.0
      2. Max Drawdown < 20%
      3. Return > SPY buy-and-hold
      4. Randomisation p-value < 0.05
      5. Monthly win rate > 50%
      6. Win/Loss ratio > 1.2
    """
    test = load_result(results_dir, "backtest_test.json")
    rand = load_result(results_dir, "randomisation_test.json")

    gates: list[GateResult] = []

    if test is None:
        print("ERROR: Test set results not found. Run minimal_backtest.py first.")
        return [GateResult("Results exist", "File exists", "NOT FOUND", False)]

    gates.append(GateResult(
        "Sharpe Ratio",
        "> 1.0",
        f"{test['sharpe_ratio']:.3f}",
        test["sharpe_ratio"] > 1.0,
    ))

    gates.append(GateResult(
        "Max Drawdown",
        "< 20%",
        f"{test['max_drawdown_pct']:.2f}%",
        test["max_drawdown_pct"] < 20,
    ))

    gates.append(GateResult(
        "Return > SPY",
        f"> {test['benchmark_return_pct']:.2f}%",
        f"{test['total_return_pct']:.2f}%",
        test["total_return_pct"] > test["benchmark_return_pct"],
    ))

    if rand:
        gates.append(GateResult(
            "Randomisation p-value",
            "< 0.05",
            f"{rand['p_value']:.4f}",
            rand["p_value"] < 0.05,
        ))
    else:
        gates.append(GateResult(
            "Randomisation p-value",
            "< 0.05",
            "NOT RUN",
            False,
        ))

    gates.append(GateResult(
        "Monthly Win Rate",
        "> 50%",
        f"{test['monthly_win_rate_pct']:.1f}%",
        test["monthly_win_rate_pct"] > 50,
    ))

    gates.append(GateResult(
        "Win/Loss Ratio",
        "> 1.2",
        f"{test['avg_win_loss_ratio']:.2f}",
        test["avg_win_loss_ratio"] > 1.2,
    ))

    return gates


# ─── Phase 2 Gates ────────────────────────────────────────────────────────────

def check_phase2(results_dir: str) -> list[GateResult]:
    """
    Phase 2 pass conditions:
      1. Final agent count ≤ 5
      2. Each retained agent has documented marginal contribution
      3. Overall Sharpe > Phase 1 Sharpe
      4. Overall Max DD ≤ Phase 1 Max DD
      5. LLM necessity decision is documented with data
    """
    contribution = load_result(results_dir, "agent_contribution.json")
    phase1_test = load_result(results_dir, "backtest_test.json")
    phase2_test = load_result(results_dir, "phase2_final_backtest.json")
    llm_test = load_result(results_dir, "llm_necessity.json")

    gates: list[GateResult] = []

    if contribution is None:
        gates.append(GateResult("Agent contribution data", "Exists", "NOT FOUND", False))
        return gates

    retained = [a for a in contribution.get("agents", []) if a.get("retained")]
    n_retained = len(retained)

    gates.append(GateResult(
        "Retained agents",
        "≤ 5",
        str(n_retained),
        n_retained <= 5,
    ))

    all_documented = all(
        a.get("delta_sharpe") is not None and a.get("p_value") is not None
        for a in retained
    )
    gates.append(GateResult(
        "Marginal contribution documented",
        "All retained agents",
        f"{sum(1 for a in retained if a.get('delta_sharpe') is not None)}/{n_retained}",
        all_documented,
    ))

    if phase1_test and phase2_test:
        gates.append(GateResult(
            "Sharpe improvement",
            f"> {phase1_test['sharpe_ratio']:.3f}",
            f"{phase2_test['sharpe_ratio']:.3f}",
            phase2_test["sharpe_ratio"] > phase1_test["sharpe_ratio"],
        ))
        gates.append(GateResult(
            "Max DD not worse",
            f"≤ {phase1_test['max_drawdown_pct']:.2f}%",
            f"{phase2_test['max_drawdown_pct']:.2f}%",
            phase2_test["max_drawdown_pct"] <= phase1_test["max_drawdown_pct"],
        ))
    else:
        gates.append(GateResult("Phase 2 backtest", "Exists", "NOT FOUND", False))

    if llm_test:
        gates.append(GateResult(
            "LLM decision documented",
            "Decision + data",
            f"keep={llm_test.get('keep_llm', 'unknown')}",
            llm_test.get("decision_documented", False),
        ))
    else:
        gates.append(GateResult("LLM necessity test", "Exists", "NOT FOUND", False))

    return gates


# ─── Report ───────────────────────────────────────────────────────────────────

def print_gate_report(phase: int, gates: list[GateResult]):
    print("\n" + "=" * 70)
    print(f"  PHASE {phase} GATE CHECK REPORT")
    print("=" * 70)

    all_pass = True
    for g in gates:
        status = "✓ PASS" if g.passed else "✗ FAIL"
        if not g.passed:
            all_pass = False
        print(f"  {status}  {g.name:<30} {g.actual:>12} (need {g.threshold})")

    print("-" * 70)
    if all_pass:
        print(f"  ✅ PHASE {phase} ALL GATES PASSED")
        if phase == 1:
            print("     → Proceed to Phase 2: Agent Contribution Testing")
        elif phase == 2:
            print("     → Proceed to Phase 3: Cautious Complexity Addition")
    else:
        failed = [g for g in gates if not g.passed]
        print(f"  ❌ PHASE {phase} FAILED ({len(failed)}/{len(gates)} gates failed)")
        print("\n  Failure Analysis:")
        for g in failed:
            print(f"    - {g.name}: got {g.actual}, needed {g.threshold}")

        if phase == 1:
            print("\n  Recommendations:")
            print("    1. Do NOT re-tune parameters on the test set")
            print("    2. Review signal logic fundamentally")
            print("    3. If re-testing, use a NEW data window")
            print("    4. Consider if the signal has any real edge")

    print("=" * 70)
    return all_pass


def main():
    parser = argparse.ArgumentParser(description="Gate Check")
    parser.add_argument("--phase", type=int, required=True, choices=[1, 2])
    parser.add_argument("--results-dir", default=str(PROJECT_ROOT / "results"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.phase == 1:
        gates = check_phase1(args.results_dir)
    else:
        gates = check_phase2(args.results_dir)

    passed = print_gate_report(args.phase, gates)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
