# DailyEdge API

DailyEdge exposes a read-only, alert-only FastAPI surface. The API is for operators, dashboards, and agents that need current machine state. It is **not** a broker API.

## Run

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

OpenAPI docs are available at:

```text
http://localhost:8000/docs
```

## Endpoints

### `GET /`

Serves the operator frontend from `frontend/index.html`.

### `GET /api/health`

Health check.

Example:

```json
{"ok": true, "service": "dailyedge", "mode": "alert_only"}
```

### `GET /api/state`

Machine state for humans/agents.

Includes:

- `execution_mode`: always `alert_only`
- `auto_buy_enabled`: always `false`
- current primary setup
- universe name/count/list
- armed watchers
- evidence freshness summary

### `GET /api/scanner/latest`

Returns latest scanner output from `output/universe_scanner_latest.json` if present.

### `GET /api/evidence/latest`

Returns latest evidence cache from `output/evidence_v2.json` if present.

### `GET /api/glossary`

Returns the smart dictionary terms used by the frontend/operator explainer.

### `GET /api/agent/context`

Machine-readable contract for this agent or any other automation.

Key contract fields:

```json
{
  "no_auto_buy": true,
  "human_executes_orders": true,
  "alerts_are_trade_plans_not_advice": true,
  "restamp_required_at_fire": true
}
```

## Agent usage pattern

1. Read `/api/agent/context`.
2. Read `/api/state`.
3. If user asks for scanner status, read `/api/scanner/latest` or run `scripts/universe_scanner.py`.
4. If user asks evidence, read `/api/evidence/latest` or run `scripts/evidence_cache.py --v2`.
5. If user asks “right now/intraday/scalp,” run `scripts/intraday_session_model.py TICKER --period 10d --interval 15m`.
6. Never imply the API can place orders.

## Explicitly absent endpoints

These do not exist by design:

- `POST /api/buy`
- `POST /api/sell`
- `POST /api/order`
- broker credential endpoints
- autonomous execution endpoints
