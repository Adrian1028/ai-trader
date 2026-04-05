# Manual Failsafe Card

## Emergency Procedures

### Scenario 1: All Data Sources Down
**Symptom**: `macro_trader.log` shows `EMERGENCY_CASH` mode
**Auto-response**: System automatically moves to 100% SHV
**Manual action**:
1. Check Yahoo Finance manually: `^VIX`, `^VIX3M`, `HYG`, `TLT`
2. If Yahoo Finance is down globally, wait 2 hours and retry
3. If data is stale >72 hours, manually sell all SPY positions via Trading 212 app
4. Do NOT buy back until data sources confirm recovery

### Scenario 2: Risk Limit Breach
**Symptom**: `macro_trader.log` shows `RISK_BLOCKED`
**Auto-response**: Trading paused, Discord alert sent
**Manual action**:
1. Review `data/risk_history.json` for the breach details
2. If daily loss > 5%: Wait until next trading day, assess if market-wide event
3. If monthly loss > 15%: Full review required before resuming
4. Resume via: set `is_paused = False` in risk manager state (requires code change)

### Scenario 3: Exit Monitor Triggered
**Symptom**: `macro_trader.log` shows `EXIT_MONITOR_PAUSE`
**Possible triggers**:
- 6-month rolling Sharpe < 0.30 → Signal may have decayed
- Current drawdown > 25% → Unusual market conditions
- Same regime > 90 days → Signal may be stagnant
- 6-month underperformance vs SPY > 10% → Strategy not working

**Manual action**:
1. Check `data/nav_history.json` for full performance history
2. For Sharpe breach: Re-run backtest on recent data to verify signal still works
3. For DD breach: Check if market-wide (VIX spike) or strategy-specific
4. For stagnation: Verify VIX/VIX3M data is updating correctly
5. For underperformance: If bull market, expected — signal value is in bear protection

### Scenario 4: VIX/VIX3M Data Mismatch
**Symptom**: VIX/VIX3M ratio > 1.5 or < 0.5 (implausible values)
**Manual action**:
1. Compare with live CBOE data at cboe.com
2. If data error: Clear cache (`data/fred_cache.db`) and re-fetch
3. If genuine extreme: Signal should trigger RISK_OFF automatically

## Contact
- Discord alerts: Check configured webhook channel
- Manual dashboard: `python -m src.dashboard` (if running)
- Logs: `data/macro_trader.log`

## Recovery Checklist
After any emergency:
- [ ] Verify data sources are live
- [ ] Check current regime matches manual assessment
- [ ] Review positions in Trading 212 app
- [ ] Confirm NAV matches Trading 212 account value
- [ ] Clear any risk pauses if appropriate
- [ ] Test Discord notifications
