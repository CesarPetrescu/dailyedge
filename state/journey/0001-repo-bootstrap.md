# Journey 0001 — Repository bootstrap

Date: 2026-06-10

## Why this exists

DailyEdge started as a working Ultra Daytrader script stack. Cesar asked to turn it into a proper standalone repo with version control, docs, state/history, frontend, and a documented API so agents can operate it without guessing.

## Actions taken

- Created `/root/trading-agents/dailyedge`.
- Initialized a real git repo on branch `main`.
- Imported current DailyEdge engine code from `/root/trading-agents/ultra-daytrader`:
  - `scripts/`
  - `tests/`
  - `config/`
- Added repo structure:
  - `docs/`
  - `docs/usecases/`
  - `state/journey/`
  - `api/`
  - `frontend/`
  - `output/`
  - `journal/`
- Added FastAPI read-only/alert-only API.
- Added smart dictionary/operator frontend.
- Added Dockerfile and docker-compose.
- Added README, API docs, architecture docs, operator guide, and usecases.

## Current safety decision

The repo is alert-only. There is no broker execution code and no auto-buy endpoint.

## Current trading decision state

- Live forward rule: `breakout_20_rr3`
- `ema_cross`: not promoted / rejected by current evidence
- `breakout_20_rr3`: tryout/incubation, not proven
- Ledger/notebook: ready for forward records; promotion requires real logged trades

## Verification target

- `pytest -q`
- API contract tests
- API smoke via `/api/health`, `/api/state`, `/api/agent/context`
- Git initial commit
