#!/usr/bin/env python3
"""Forward ledger and risk controls for DailyEdge.

Backtests can enable/reject rules; this ledger is the forward ground truth.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LEDGER_COLUMNS = [
    "ts", "ticker", "rule", "side", "price", "taken", "entry", "stop", "target",
    "exit", "outcome_R", "MAE_R", "MFE_R", "regime", "confidence", "sector", "notes",
]

AI_SEMIS = {"NVDA", "MU", "SNDK", "KLAC", "ASML", "AMAT", "LRCX", "TSM", "ON", "AMD", "AVGO", "MRVL", "ARM", "SMCI"}


def sector_cluster(ticker: str) -> str:
    t = ticker.upper().strip()
    if t in AI_SEMIS:
        return "AI-semis"
    return "Other"


def ensure_ledger(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        with p.open("w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=LEDGER_COLUMNS).writeheader()
    return p


def append_signal(path: str | Path, **row: Any) -> None:
    p = ensure_ledger(path)
    full = {col: row.get(col, "") for col in LEDGER_COLUMNS}
    full["ts"] = full["ts"] or datetime.now(timezone.utc).isoformat()
    full["sector"] = full["sector"] or sector_cluster(str(full.get("ticker", "")))
    with p.open("a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=LEDGER_COLUMNS).writerow(full)


def read_rows(path: str | Path, limit: int | None = None) -> list[dict[str, str]]:
    p = ensure_ledger(path)
    with p.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[:limit] if limit else rows


def _float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x) if x not in (None, "") else default
    except Exception:
        return default


def empty_risk_state() -> dict[str, Any]:
    return {"day_R": 0.0, "week_R": 0.0, "open_risk_R": 0.0, "breaches": [], "open_positions_by_sector": {}}


def risk_state(path: str | Path | None = None) -> dict[str, Any]:
    if path is None:
        return empty_risk_state()
    rows = read_rows(path)
    state = empty_risk_state()
    # Conservative/simple: sum filled outcomes dated today/last 7 rows' explicit fields when present.
    for r in rows:
        outcome = _float(r.get("outcome_R"), 0.0)
        state["week_R"] += outcome
        state["day_R"] += outcome
        if r.get("taken", "").lower() in {"1", "true", "yes"} and not r.get("exit"):
            state["open_risk_R"] += 1.0
            sec = r.get("sector") or sector_cluster(r.get("ticker", ""))
            state["open_positions_by_sector"][sec] = state["open_positions_by_sector"].get(sec, 0) + 1
    breaches = []
    if state["day_R"] <= -2:
        breaches.append(f"day R {state['day_R']:.1f} <= -2R")
    if state["week_R"] <= -5:
        breaches.append(f"week R {state['week_R']:.1f} <= -5R")
    if state["open_risk_R"] >= 3:
        breaches.append(f"open risk {state['open_risk_R']:.1f}R >= 3R")
    state["breaches"] = breaches
    return state


def risk_breaches(state: dict[str, Any]) -> list[str]:
    breaches = list(state.get("breaches", []))
    if state.get("day_R", 0) <= -2 and not any("day R" in b for b in breaches):
        breaches.append(f"day R {state['day_R']:.1f} <= -2R")
    if state.get("week_R", 0) <= -5 and not any("week R" in b for b in breaches):
        breaches.append(f"week R {state['week_R']:.1f} <= -5R")
    if state.get("open_risk_R", 0) >= 3 and not any("open risk" in b for b in breaches):
        breaches.append(f"open risk {state['open_risk_R']:.1f}R >= 3R")
    return breaches


def rule_tier(backtest_n: int, ledger_n: int, ledger_e_cons: float | None) -> str:
    # Backtest n intentionally cannot promote a live rule.
    e = ledger_e_cons if ledger_e_cons is not None else 0.0
    if ledger_n >= 30 and e < 0:
        return "RETIRED"
    if ledger_n >= 100 and e > 0.10:
        return "PROVEN"
    if ledger_n >= 30 and e > 0:
        return "VALIDATED"
    return "INCUBATION"


def concentration_warning(rows: list[dict[str, Any]], max_per_cluster: int = 2) -> str:
    counts: dict[str, int] = {}
    for r in rows:
        if r.get("action") in {"GO", "GO_PROBATION", "SET_ALERT", "WATCH_STRONG"}:
            sec = r.get("sector") or sector_cluster(str(r.get("ticker", "")))
            counts[sec] = counts.get(sec, 0) + 1
    warnings = []
    for sec, n in counts.items():
        if n > max_per_cluster:
            warnings.append(f"{sec} concentration: {n} candidates; sector cap max {max_per_cluster} open positions")
    return "; ".join(warnings)


if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default="/root/trading-agents/ultra-daytrader/output/alerts_log.csv")
    args = ap.parse_args()
    print(json.dumps({"risk_state": risk_state(args.path), "first_rows": read_rows(args.path, 5)}, indent=2))
