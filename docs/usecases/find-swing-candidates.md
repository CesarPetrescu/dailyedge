# Usecase: Find swing candidates

## Goal

Find stocks that are strong enough to watch, without chasing early.

## Command

```bash
python3 scripts/universe_scanner.py
```

## Read the output

- `SET_ALERT`: actionable watch. Add/keep tripwire.
- `WATCH_STRONG`: strong but not at a trigger; monitor only.
- `EXPIRED`: do not chase. Wait for a new setup.
- `NO_GO`: referee rejected it.

## Human action

No buy yet unless the trigger fires and the re-stamped plan passes the referee.

## API equivalent

```text
GET /api/scanner/latest
GET /api/state
```
