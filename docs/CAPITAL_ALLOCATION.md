# Capital Allocation Framework

## Architecture

Two independent trading systems with isolated capital:

### System 1: Macro SPY/SHV (this system)
- **Capital**: Allocated at startup via `MACRO_INITIAL_CAPITAL` env var
- **Instruments**: SPY (S&P 500 ETF), SHV (Short-Term Treasury ETF)
- **Broker**: Trading 212 Demo (paper trading phase)
- **Signal**: VIX term structure + synthetic OAS macro risk switch
- **Frequency**: Daily at UTC 22:00

### System 2: OANDA Forex Bot (separate)
- **Capital**: Independently allocated
- **Instruments**: Forex pairs
- **Broker**: OANDA
- **Signal**: Independent of System 1

## Capital Isolation Rules

1. **No cross-system transfers**: Each system's P&L is independent
2. **Independent risk limits**: Each system has its own daily/monthly DD limits
3. **Unified monitoring**: Total NAV tracked across both systems for reporting
4. **No correlation assumptions**: Systems may be correlated in crisis — this is accepted

## Risk Limits

### Per-System Limits (enforced by `unified_risk.py`)
| Limit | Threshold | Action |
|-------|-----------|--------|
| Daily loss | > 5% of system NAV | Auto-pause system |
| Monthly loss | > 15% of system NAV | Auto-pause system |

### Cross-System Monitoring (not enforced, reporting only)
| Metric | Alert Threshold |
|--------|----------------|
| Total NAV drawdown | > 20% from peak |
| Correlation spike | Both systems losing > 3% same day |

## Allocation by Regime

| Regime | SPY | SHV | Cash |
|--------|-----|-----|------|
| NORMAL | 100% | 0% | 0% |
| WARNING | 50% | 50% | 0% |
| RISK_OFF | 0% | 100% | 0% |

**Note**: SHV was chosen over cash based on Step 3 backtest results (D013):
- SHV outperformed cash by ~2.5% return with lower drawdown
- SHV slippage is minimal (2 bps) due to high liquidity

## NAV Tracking

Capital state is persisted to `data/capital_state.json` and updated daily.
The `AllocationTracker` class records:
- Initial capital per system
- Current NAV, cash, positions value
- Running P&L percentage

## Degradation Rules

When data quality degrades (per `failsafe.py`):

| Data Health | Max SPY Exposure |
|-------------|-----------------|
| LIVE | 100% (per regime) |
| STALE (30min-24h) | 75% of regime target |
| DEGRADED (1-3 days) | 50% of regime target |
| DEAD (>3 days) | 0% (emergency cash) |
