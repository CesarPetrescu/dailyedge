# DailyEdge

DailyEdge is an **alert-only trading research stack** for finding and validating stock trade setups. It scans a configured universe, arms exact trigger levels, applies risk/referee checks, creates reports/charts, and records evidence so rules earn or lose trust over time.

**It does not auto-buy. There is no broker execution endpoint.** A human places or skips every trade.

## Current machine state

- Universe: `liquid_ai_semis` — 25 AI/semi/liquid tech names
- Live forward rule: `breakout_20_rr3`
- Target shape: stop = 1.5x ATR projection, target = 3R projection, re-stamped at fire
- Sizing: tryout/incubation only until enough real forward ledger trades exist
- Automation: daily scanner + weekly evidence refresh can run via cron outside this repo
- API: FastAPI read-only/alert-only interface for agents and frontend
- Frontend: static smart dictionary/operator explainer served by the API

## Repo layout

```text
api/                 FastAPI app exposing state, evidence, scanner, glossary
config/              Universe, scanner, and armed watcher configuration
docs/                Architecture, API, usecases, operator docs
frontend/            Static operator explainer UI
scripts/             DailyEdge engine scripts and models
state/journey/       Development log / decisions / implementation journal
tests/               Regression and contract tests
output/              Generated scanner/evidence/report output (ignored except .gitkeep)
journal/             Runtime ledger/journal area (ignored except .gitkeep)
```

## Quick start

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
pytest -q
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Open:

- Frontend: http://localhost:8000/
- API docs: http://localhost:8000/docs
- Machine state: http://localhost:8000/api/state
- Agent context: http://localhost:8000/api/agent/context

## Core commands

```bash
# Daily scanner
python3 scripts/universe_scanner.py

# Weekly evidence refresh
python3 scripts/evidence_cache.py --v2

# Full ticker report
python3 scripts/full_ticker_report.py KLAC

# Intraday/session read
python3 scripts/intraday_session_model.py KLAC --period 10d --interval 15m --opening-range-minutes 60

# Tests
pytest -q
```

## Safety contract

- No auto-buy.
- No autonomous real-money order placement.
- Alerts are analytical trade plans only.
- Every live signal must be re-stamped at fire with signal-bar ATR and next-open entry.
- Backtests can reject rules; only the forward ledger can promote sizing.
- Analytical framework only, not financial advice.
