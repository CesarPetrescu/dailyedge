# Operator Guide

## What this machine is for

DailyEdge helps answer:

- “Which stocks are setting up?”
- “Where is the exact alert line?”
- “If it fires, where are the stop and target?”
- “Did this setup historically beat a baseline?”
- “Is this a swing setup or do I need intraday confirmation?”

It is **not** a buy button.

## Daily use

### Premarket

Run or read the scanner:

```bash
python3 scripts/universe_scanner.py
```

Look for:

- `SET_ALERT`: close to trigger; wait.
- `GO_PROBATION`: fresh fired setup; tiny/tryout only.
- `NO_GO`: failed referee rules.
- `EXPIRED`: stale signal; do not chase.

### During market

If a trigger fires:

1. Re-stamp with signal-bar ATR and next-open entry.
2. Run/referee the plan.
3. If still valid, operator decides whether to take a tiny/tryout-size trade.
4. Stop order goes first.
5. Log taken/skipped and result.

### For intraday entries

Run:

```bash
python3 scripts/intraday_session_model.py TICKER --period 10d --interval 15m --opening-range-minutes 60
```

Use intraday model for “right now” decisions. Use daily scanner for swing/tripwire candidates.

## Current rule

- Rule: `breakout_20_rr3`
- Meaning: 20-day breakout setup, 3R target.
- Status: tryout/incubation.
- Not proven edge yet.

## Current rule discipline

- One system change per week.
- `ema_cross` remains out until full-universe reproducibility and margin test improve.
- Backtests reject; forward ledger promotes.
