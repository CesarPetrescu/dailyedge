# Usecase: Intraday decision

## Goal

Answer “should I get in right now?” with live session context.

## Command

```bash
python3 scripts/intraday_session_model.py KLAC --period 10d --interval 15m --opening-range-minutes 60
```

## What to inspect

- VWAP position
- Opening range high/low
- Relative volume
- Session phase
- Price versus EMA/VWAP
- Market regime conflict

## Human action

The model can say enter/wait/no-chase/only-over-reclaim, but the operator still executes manually.

## Boundary

Do not use the daily scanner alone for scalp timing. Daily scanner gives swing/tripwire context; intraday model gives session timing context.
