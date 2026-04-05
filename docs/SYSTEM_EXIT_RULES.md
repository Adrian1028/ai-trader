# System Exit Rules

## Automated Exit Checks

The exit monitor (`src/monitoring/exit_monitor.py`) runs daily at UTC 22:30.
If any check triggers, trading is automatically paused.

### Check 1: Rolling 6-Month Sharpe
- **Threshold**: Sharpe < 0.30
- **Window**: Last 126 trading days
- **Rationale**: If annualised risk-adjusted return drops below 0.30, the signal may have lost its edge. Note: Backtest Sharpe was 0.37, so 0.30 gives 19% buffer.
- **Response**: Auto-pause + Discord alert

### Check 2: Maximum Drawdown
- **Threshold**: Current DD > 25%
- **Measurement**: NAV vs all-time high
- **Rationale**: Backtest max DD was ~11.5%. A 25% DD is 2x+ worse than expected. Could indicate regime change or signal failure.
- **Response**: Auto-pause + Discord alert

### Check 3: Signal Stagnation
- **Threshold**: Same regime for > 90 trading days
- **Rationale**: If the system stays in one regime for 4+ months, the VIX term structure data may not be updating, or the signal parameters may need recalibration.
- **Response**: Alert only (not auto-pause, since prolonged NORMAL is expected in bull markets)

### Check 4: 6-Month Excess Return
- **Threshold**: Strategy underperforms SPY by > 10% over 6 months
- **Rationale**: The signal's value is bear market protection and DD reduction. In strong bull markets, the strategy may slightly underperform SPY (due to SHV periods). But >10% underperformance suggests the signal is actively harmful.
- **Response**: Auto-pause + Discord alert

## Manual Exit Criteria

These require human judgment and cannot be fully automated:

1. **Structural market change**: If CBOE discontinues VIX3M index, or VIX calculation methodology changes significantly
2. **Synthetic OAS divergence**: If HYG/TLT ratio stops correlating with actual credit spreads (check quarterly)
3. **Regulatory change**: If short-term treasury (SHV) becomes unavailable or tax treatment changes
4. **Capital needs**: If capital is needed for other purposes

## Go-Live Criteria (Post Paper Trading)

After 3 months of paper trading, go-live requires:
- [ ] Paper trading Sharpe > 0.20
- [ ] Paper trading max DD < 20%
- [ ] Signal regime changes align with backtest expectations
- [ ] No manual overrides needed during paper trading
- [ ] Discord notifications working reliably
