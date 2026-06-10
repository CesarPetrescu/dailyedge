from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "universe_scanner.json"
ARMED = ROOT / "config" / "armed_watchers.json"
EVIDENCE = ROOT / "output" / "evidence_v2.json"
SCANNER = ROOT / "output" / "universe_scanner_latest.json"
FRONTEND = ROOT / "frontend" / "index.html"

app = FastAPI(
    title="DailyEdge API",
    version="0.1.0",
    description="Alert-only trading research API for DailyEdge v2. No broker execution or auto-buy endpoint exists.",
)


class GlossaryTerm(BaseModel):
    term: str
    plain: str
    precise: str


GLOSSARY: list[GlossaryTerm] = [
    GlossaryTerm(term="ticker", plain="The stock symbol, like KLAC or ASML.", precise="Canonical market identifier used by Yahoo/yfinance and DailyEdge configs."),
    GlossaryTerm(term="breakout", plain="A stock pushing above its recent high.", precise="Current close crosses above the prior rolling 20-bar high for breakout_20."),
    GlossaryTerm(term="trigger", plain="The exact price where an alert becomes real.", precise="A tripwire price; current DailyEdge triggers are usually prior 20-bar highs."),
    GlossaryTerm(term="tripwire", plain="A silent alert line waiting to be crossed.", precise="An armed watcher with op=cross_above and entry policy next open after trigger."),
    GlossaryTerm(term="fire", plain="The tripwire got hit.", precise="A signal-bar close crosses above trigger; trade structure is re-stamped before ledger append."),
    GlossaryTerm(term="ATR", plain="How much the stock normally wiggles.", precise="Average True Range, currently ATR14, used to set volatility-sized stop distance."),
    GlossaryTerm(term="stop loss", plain="The give-up line.", precise="Price where planned loss is taken; one stop hit equals about -1R before/after costs."),
    GlossaryTerm(term="target", plain="The take-profit line.", precise="For breakout_20_rr3, target is entry + 3 * risk distance."),
    GlossaryTerm(term="R", plain="The risk unit for one trade.", precise="R = entry minus stop distance for a long trade; outcomes are normalized in R."),
    GlossaryTerm(term="R:R", plain="Reward compared with risk.", precise="Reward-to-risk ratio; current forward rule targets 3R reward for 1R risk."),
    GlossaryTerm(term="hit rate", plain="How often target wins before stop/time exit.", precise="Winning fraction from triple-barrier outcomes."),
    GlossaryTerm(term="expectancy", plain="Average R expected per trade.", precise="E[R] = p_win * avg_win - p_loss * avg_loss - costs."),
    GlossaryTerm(term="Wilson floor", plain="The pessimistic win-rate estimate.", precise="Lower bound of Wilson confidence interval used for conservative expectancy."),
    GlossaryTerm(term="conservative expectancy", plain="Pessimistic average result.", precise="E_cons uses Wilson lower-bound hit rate with observed average win/loss."),
    GlossaryTerm(term="costs", plain="The hidden trading toll.", precise="Round-trip spread, slippage, and fees converted into R."),
    GlossaryTerm(term="cost_R", plain="Trading cost as a chunk of your risk.", precise="Round-trip percentage cost divided by stop percentage."),
    GlossaryTerm(term="breakeven hit rate", plain="Minimum win rate needed not to lose.", precise="Hit rate required for target/stop structure after costs."),
    GlossaryTerm(term="baseline", plain="The monkey benchmark.", precise="baseline_every5 enters every fifth bar with the same stop/target/time rules."),
    GlossaryTerm(term="triple barrier", plain="Stop, target, or time runs out.", precise="Backtest labeler exits at first stop/target hit or after max bars."),
    GlossaryTerm(term="backtest", plain="Historical rehearsal.", precise="Past-data simulation; can reject a setup but cannot promote it to larger size."),
    GlossaryTerm(term="ledger", plain="The notebook of every real signal and outcome.", precise="Forward evidence store used for promotion/retirement and risk state."),
    GlossaryTerm(term="tryout", plain="Tiny size while the rule proves itself live.", precise="Incubation tier before enough forward ledger trades exist."),
    GlossaryTerm(term="regime", plain="Market weather.", precise="SPY/QQQ/sector context; true bear is Close < SMA50 < SMA200."),
    GlossaryTerm(term="sector cluster", plain="Stocks that move together.", precise="Risk grouping like AI-semis used for concentration caps."),
    GlossaryTerm(term="relative strength", plain="How strong it is versus the market.", precise="Ticker return minus SPY return over 20/63 days."),
    GlossaryTerm(term="relative volume", plain="Today's volume versus normal.", precise="Current volume divided by rolling 20-bar average volume."),
    GlossaryTerm(term="Monte Carlo", plain="Many fake paths to show possible ranges.", precise="Scenario distribution from historical returns; not a forecast."),
    GlossaryTerm(term="shadow ledger", plain="Paper notebook before real confidence.", precise="Forward record of alerts/skips/outcomes, even before sizing up."),
    GlossaryTerm(term="cron", plain="Automatic scheduled run.", precise="Scheduler job for daily scanner and weekly evidence refresh."),
    GlossaryTerm(term="re-stamp", plain="Recalculate the plan when the alert really fires.", precise="Use signal-bar ATR and next open, not stale projected stop/target."),
    GlossaryTerm(term="beta", plain="How much the stock moves with the market.", precise="Market sensitivity used in some risk sizing/probation logic."),
    GlossaryTerm(term="MAE", plain="Worst pain during the trade.", precise="Maximum adverse excursion, measured in R."),
    GlossaryTerm(term="MFE", plain="Best unrealized profit during the trade.", precise="Maximum favorable excursion, measured in R."),
    GlossaryTerm(term="drawdown", plain="How deep the losing streak got.", precise="Peak-to-trough decline in account or R curve."),
]


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


@app.get("/")
def frontend() -> FileResponse:
    return FileResponse(FRONTEND)


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"ok": True, "service": "dailyedge", "mode": "alert_only"}


@app.get("/api/state")
def state() -> dict[str, Any]:
    cfg = _read_json(CONFIG, {})
    armed = _read_json(ARMED, {"armed_rules": []})
    evidence = _read_json(EVIDENCE, {})
    universe_name = cfg.get("default_universe", "liquid_ai_semis")
    tickers = cfg.get("universes", {}).get(universe_name, [])
    return {
        "name": "DailyEdge",
        "execution_mode": "alert_only",
        "auto_buy_enabled": False,
        "primary_setup": cfg.get("primary_setup", armed.get("primary_setup", "unknown")),
        "rr": cfg.get("rr"),
        "universe": universe_name,
        "universe_count": len(tickers),
        "ticker_list": tickers,
        "armed_watchers": armed.get("armed_rules", []),
        "evidence": {
            "version": evidence.get("version"),
            "generated_at": evidence.get("generated_at"),
            "ok_count": evidence.get("data_quality", {}).get("ok_count"),
            "failed_count": evidence.get("data_quality", {}).get("failed_count"),
        },
    }


@app.get("/api/scanner/latest")
def scanner_latest() -> dict[str, Any]:
    return _read_json(SCANNER, {"ok": False, "error": "scanner output not generated yet"})


@app.get("/api/evidence/latest")
def evidence_latest() -> dict[str, Any]:
    return _read_json(EVIDENCE, {"ok": False, "error": "evidence output not generated yet"})


@app.get("/api/glossary")
def glossary() -> dict[str, Any]:
    return {"terms": [term.model_dump() for term in GLOSSARY]}


@app.get("/api/agent/context")
def agent_context() -> dict[str, Any]:
    return {
        "contract": {
            "no_auto_buy": True,
            "human_executes_orders": True,
            "alerts_are_trade_plans_not_advice": True,
            "restamp_required_at_fire": True,
        },
        "capabilities": {
            "scanner": "GET /api/scanner/latest or run scripts/universe_scanner.py",
            "evidence": "GET /api/evidence/latest or run scripts/evidence_cache.py --v2",
            "state": "GET /api/state",
            "glossary": "GET /api/glossary",
            "intraday": "run scripts/intraday_session_model.py TICKER --period 10d --interval 15m",
            "ticker_report": "run scripts/full_ticker_report.py TICKER",
        },
        "paths": {
            "config": str(CONFIG.relative_to(ROOT)),
            "armed_watchers": str(ARMED.relative_to(ROOT)),
            "state_journey": "state/journey/",
            "docs": "docs/",
            "frontend": "frontend/index.html",
        },
    }
