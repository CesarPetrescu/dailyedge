# Usecase: Agent operation

## Goal

Give an AI agent a stable way to understand and operate the stack without guessing paths or inventing broker behavior.

## First call

```text
GET /api/agent/context
```

The agent must observe:

- `no_auto_buy: true`
- `human_executes_orders: true`
- `restamp_required_at_fire: true`

## Common agent tasks

### “What is armed?”

```text
GET /api/state
```

### “What did scanner find?”

```text
GET /api/scanner/latest
```

or run:

```bash
python3 scripts/universe_scanner.py
```

### “Is the setup real?”

```text
GET /api/evidence/latest
```

or run:

```bash
python3 scripts/evidence_cache.py --v2
```

### “Should I enter now?”

Run the mandatory intraday script for the ticker.

```bash
python3 scripts/intraday_session_model.py TICKER --period 10d --interval 15m --opening-range-minutes 60
```

## Forbidden agent behavior

- Do not claim DailyEdge auto-buys.
- Do not make up fills or ledger rows.
- Do not promote a setup based only on backtest.
- Do not skip re-stamp at fire.
