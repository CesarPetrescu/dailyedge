# DailyEdge Architecture

## One-line purpose

DailyEdge is an **alert-only stock setup machine**: scan candidates, arm tripwires, validate risk, explain the plan, and measure whether rules deserve trust.

## Safety boundary

DailyEdge does **not** connect to a broker and does **not** place orders. The API intentionally has no `/order`, `/buy`, `/sell`, or broker credential surface.

## Components

### 1. Scout — universe scanner

- File: `scripts/universe_scanner.py`
- Config: `config/universe_scanner.json`
- Job: scan the configured universe, rank names, and export candidates.
- Output:
  - `output/universe_scanner_latest.json`
  - `output/universe_scanner_latest.txt`
  - `output/watcher_candidates.json`

### 2. Referee — gate/risk rules

- Files: `scripts/gate_hook.py`, `scripts/ledger.py`, `scripts/market_regime.py`
- Checks:
  - risk/reward
  - cost as R
  - stop distance / high-vol floor
  - stale signal state
  - day/week/open-risk caps
  - sector concentration
  - market regime

### 3. Tripwire/watchers

- File: `scripts/watcher_ingest.py`
- Config: `config/armed_watchers.json`
- Job: store exact alert triggers and projection values.
- Important: armed stop/target values are projections. When a trigger fires, re-stamp with signal-bar ATR and next-open entry.

### 4. Evidence machine

- File: `scripts/evidence_cache.py`
- Output: `output/evidence_v2.json`
- Records:
  - exact ticker list
  - per-setup aggregate stats
  - per-name stats
  - baseline comparison
  - bootstrap margin test
  - top-3 ticker share of R
  - regime conservation counts

### 5. Reports and charts

- Files:
  - `scripts/full_ticker_report.py`
  - `scripts/report_images.py`
  - `scripts/chart_simulation.py`
  - `scripts/analyze_ticker.py`
- Job: create Telegram/API-friendly text plus PNGs.

### 6. Intraday/session model

- File: `scripts/intraday_session_model.py`
- Job: for “should I enter now?” decisions; checks VWAP, opening range, relative volume, and session structure.

### 7. API and frontend

- API: `api/main.py`
- Frontend: `frontend/index.html`
- Purpose: read-only/alert-only interface for humans and agents.

## Current rule lifecycle

1. Backtest/evidence can reject a setup.
2. Backtest/evidence cannot promote a setup to bigger size.
3. Forward ledger trades are required for promotion.
4. Current rule `breakout_20_rr3` is tryout/incubation, not proven.
5. `ema_cross` remains disabled/rejected until a full-universe margin-tested run changes that.

## Data flow

```text
config/universe_scanner.json
  -> scripts/universe_scanner.py
  -> output/watcher_candidates.json
  -> scripts/watcher_ingest.py
  -> config/armed_watchers.json
  -> fire/re-stamp/referee
  -> ledger/journal/output
  -> weekly evidence refresh
```

## Non-goals

- No auto-buy.
- No broker custody.
- No promise of profitability.
- No using Monte Carlo as a forecast.
