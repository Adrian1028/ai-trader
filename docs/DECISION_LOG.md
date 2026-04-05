# Decision Log — AI Trading Bot Systematic Improvement

## Phase 1: Alpha Verification

### D001: Stock Pool Reduction (40 → 5)
- **Date**: 2026-04-05
- **Decision**: Reduce from 40 stocks (US+UK) to 5 US large-caps: AAPL, MSFT, GOOGL, AMZN, NVDA
- **Rationale**:
  - Daily volume > 5M shares, market cap > $100B, spread < 0.05%
  - All available on Alpha Vantage free tier with full history
  - Eliminates FX risk and UK trading session complexity
  - Reduces API load by 87.5%
- **Data**: N/A (selection criteria based)

### D002: Agent Reduction (8 → 2 for baseline)
- **Date**: 2026-04-05
- **Decision**: Start with only Technical + Fundamental agents for alpha verification
- **Rationale**:
  - Simpler system is easier to validate
  - If 2 agents can't demonstrate edge, adding 6 more won't help
  - Each additional agent must justify its existence with data (Phase 2)
- **Data**: Pending Phase 1 results

### D003: Remove LLM from baseline
- **Date**: 2026-04-05
- **Decision**: Phase 1 baseline uses pure rule-based logic, no Gemini
- **Rationale**:
  - Rules are deterministic and reproducible
  - LLM calls introduce latency, rate limits, and non-reproducibility
  - LLM value will be tested separately in Phase 2 (Task 2.2)
- **Data**: Pending LLM necessity test

### D004: Conservative cost model
- **Date**: 2026-04-05
- **Decision**: Commission 0% (T212), slippage 5bps, spread 2bps
- **Rationale**:
  - Trading 212 offers zero commission on share dealing
  - 5bps slippage is conservative for $100B+ market cap stocks
  - Better to underestimate performance than overestimate
- **Data**: Based on T212 fee schedule and large-cap bid-ask spreads

---

## New Signal Architecture (v2.0)

### D005: Phase 1 SMA/RSI/MACD signals have NO alpha
- **Date**: 2026-04-05
- **Decision**: Abandon traditional technical indicator signals
- **Rationale**: Randomisation test p-value = 0.888. Random signals outperform real signals (random mean Sharpe 2.122 > actual 1.779). All returns were pure beta from holding mega-cap tech in a bull market.
- **Data**: `results/backtest_test.json`, `results/randomisation_test.json`

### D006: VIX Term Structure + OAS Macro Risk Switch VALIDATED
- **Date**: 2026-04-05
- **Decision**: Adopt as primary "risk on/off" signal
- **Parameters**: rv_trigger=0.98, rv_warning=0.92, rv_recover=0.90, oas_trigger=0.15, min_hold_days=5
- **Test set results**: Sharpe 1.067 vs SPY 0.570, Max DD 14.31% vs 24.50%, 2022 return -8.64% vs -18.65%
- **Robustness**: 100% of 75 parameter combinations beat SPY on training data
- **Data**: `results/macro_signal_backtest.json`, `results/macro_sensitivity.json`

### D007: Realized Amihud + Inelasticity Liquidity Filter REMOVED
- **Date**: 2026-04-05
- **Decision**: Permanently remove liquidity filter signal
- **Rationale**: Failed all 4 gate conditions. Excluded stocks had HIGHER 30d forward returns (1.41%) than retained stocks (1.18%). The filter is removing good stocks, not bad ones. For S&P 500 large-caps, liquidity is uniformly high so cross-sectional differentiation adds noise.
- **Correlation check**: Signals are independent (r=0.09) but liquidity filter has no value
- **Data**: `results/liquidity_filter_backtest.json`, `results/signal_correlation.json`

### D008: Stock Pool Expansion (5 → 20 stocks)
- **Date**: 2026-04-05
- **Decision**: Expand to 20 S&P 500 stocks across 8 sectors
- **Rationale**: Diversification reduces sector concentration risk. All 20 stocks have daily volume > 5M and are freely available via yfinance.
- **Data**: N/A (selection criteria based)

### D009: SEC Insider "Not Sold" Signal — NO VALUE
- **Date**: 2026-04-05
- **Decision**: Remove insider signal. Delta Sharpe = 0.000 (identical to baseline).
- **Rationale**: yfinance insider data is limited to recent transactions and doesn't contain enough "portfolio insider" cross-holdings to compute meaningful not-sold scores. The signal never fired differently from equal weight.
- **Data**: `results/full_system_backtest.json`, `results/signal_attribution.json`

### D010: Recommended Production Architecture
- **Date**: 2026-04-05
- **Decision**: Use macro VIX/VIX3M + OAS signal as SPY overlay, not as multi-stock allocator
- **Rationale**: Attribution analysis showed macro signal on SPY delivers Sharpe 1.067, but same signal applied to 20-stock monthly-rebalanced portfolio drops to 0.406 (below the 0.850 of equal-weight buy-and-hold). The macro signal's value is purely in **market timing** (risk-on vs cash), not in stock selection. Multi-stock rebalancing costs + sector rotation effects dilute the timing advantage.
- **Recommended system**: Simple SPY/cash toggle with VIX term structure
- **Data**: `results/signal_attribution.json`, `results/macro_signal_backtest.json`

---

## Phase 3: Production-ization

### D011: T+1 Execution — NO Look-Ahead Bias
- **Date**: 2026-04-05
- **Decision**: Signal survives T+1 execution delay. Negative Sharpe decay (-8.5%) means T+1 actually improved slightly.
- **Results (Test 2022-2024):**
  - Version A (T+0 close): Sharpe 0.343, Return 7.58%, DD 11.43%
  - Version B (T+1 Open): Sharpe 0.372, Return 8.37%, DD 11.46%
  - Version C (T+1 VWAP): Sharpe 0.379, Return 8.53%, DD 11.43%
- **Key finding**: Realistic trading Sharpe (0.37) much lower than theoretical model (1.07). The theoretical model used `strategy_returns = signals * spy_returns` (perfect daily rebalancing), while realistic model buys/sells shares with transaction costs.
- **Gate results**: 2/4 passed (2022 return and max DD passed; Sharpe < 0.70 and Ann Return < SPY failed). Signal's primary value is **drawdown reduction** and **bear market protection**, not absolute return in bull markets.
- **Data**: `results/t1_execution_backtest.json`

### D012: Multi-Index Diversification — STAY WITH SPY
- **Date**: 2026-04-05
- **Decision**: Do not diversify across indices. Stay with SPY only.
- **Results:**
  - SPY: Sharpe 0.372 (best), QQQ: 0.287, IWM: 0.212, EFA: -0.012
  - Signal effective on all 4 ETFs (DD reduction, 2022 protection)
  - Diversified portfolio (equal-weight 4 ETFs): Sharpe 0.300 < SPY 0.372
  - High correlations (0.73-0.95) explain why diversification doesn't help
- **Data**: `results/multi_index_backtest.json`

### D013: Cash Alternative — USE SHV during RISK_OFF
- **Date**: 2026-04-05
- **Decision**: During RISK_OFF periods, hold SHV instead of 100% cash.
- **Results:**
  - A) 100% Cash: Return 18.88%, Sharpe 0.550, DD 14.74%
  - B) SHV: Return 21.43%, Sharpe 0.610, DD 14.66%
  - C) SHV + costs: Return 20.87%, Sharpe 0.597, DD 14.70%
- **Rationale**: SHV earns short-term treasury yield (~5% in 2023-2024) during RISK_OFF, with minimal additional risk (DD reduction of 0.08%). Even with slippage costs, SHV adds ~2% return.
- **Data**: `results/cash_alternative.json`

### D014: Production Architecture
- **Date**: 2026-04-05
- **Decision**: Deploy as daily SPY/SHV toggle at UTC 22:00
- **Components:**
  - `src/main_macro.py` — Production daily cycle with APScheduler
  - `src/resilience/failsafe.py` — Data source fallback chain, auto-degradation
  - `src/capital/allocation_tracker.py` — NAV tracking across systems
  - `src/capital/unified_risk.py` — 5% daily / 15% monthly loss limits
  - `src/monitoring/exit_monitor.py` — Rolling Sharpe, DD, stagnation checks
  - `src/paper_trading.py` — 3-month validation with weekly/monthly Discord reports
- **Regime allocation:** NORMAL → 100% SPY, WARNING → 50% SPY + 50% SHV, RISK_OFF → 100% SHV
