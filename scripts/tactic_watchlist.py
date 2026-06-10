#!/usr/bin/env python3
"""Configured tactic scanner for NVDA/MU/SNDK.

This turns the pasted outcome-lab + gate-hook ideas into a repeatable tactic:
- primary daily setup: breakout_20
- disabled rules: ema_cross/pullback_trend until redesigned
- triple-barrier evidence vs baseline
- live GO/NO-GO gate with cost_R and breakeven hit-rate
- SPY/QQQ/SMH regime context

Analytical/paper-trade planning only; not financial advice.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
DEFAULT_CONFIG = ROOT / "config" / "tactic_watchlist.json"
sys.path.insert(0, str(SCRIPT_DIR))

import gate_hook
import market_regime
import outcome_lab
from edge_engine import EdgeStats, confidence_bucket


def _safe_float(x: Any, digits: int = 4):
    try:
        if x is None or pd.isna(x):
            return None
        f = float(x)
        if not math.isfinite(f):
            return None
        return round(f, digits)
    except Exception:
        return None


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["tickers"] = [str(t).upper().strip() for t in cfg.get("tickers", [])]
    if not cfg["tickers"]:
        raise ValueError("config has no tickers")
    if cfg.get("primary_setup") not in outcome_lab.SETUPS:
        raise ValueError(f"unknown primary_setup {cfg.get('primary_setup')}")
    return cfg


def fetch_frames(tickers: list[str], period: str, interval: str) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for tk in tickers:
        frames[tk] = outcome_lab.fetch(tk, period, interval)
    return frames


def detect_setup_state(df: pd.DataFrame, setup: str, max_age: int, intraday: bool = False) -> dict[str, Any]:
    fn, desc = outcome_lab.SETUPS[setup]
    sig = fn(df, intraday).fillna(False)
    latest = df.iloc[-1]
    recent = sig.tail(max_age + 1)
    signal_dates = list(sig[sig].index)
    last_signal_ts = signal_dates[-1] if signal_dates else None
    age = None
    if last_signal_ts is not None:
        age = len(df.index) - 1 - df.index.get_loc(last_signal_ts)
    trigger = None
    if setup == "breakout_20" and len(df) >= 21:
        trigger = float(df["High"].rolling(20).max().shift().iloc[-1])
    return {
        "setup": setup,
        "description": desc,
        "signal_now": bool(sig.iloc[-1]),
        "signal_recent": bool(recent.any()),
        "bars_since_signal": age,
        "last_signal_date": str(last_signal_ts) if last_signal_ts is not None else None,
        "trigger": _safe_float(trigger, 2),
        "price_vs_trigger_pct": _safe_float((float(latest["Close"]) / trigger - 1) * 100 if trigger else None, 2),
    }


def make_gate(df: pd.DataFrame, cfg: dict[str, Any], ticker: str) -> dict[str, Any]:
    latest = df.iloc[-1]
    entry = float(latest["Close"])
    atr_series = outcome_lab.atr(df)
    atr_val = float(atr_series.iloc[-1]) if pd.notna(atr_series.iloc[-1]) else None
    if not atr_val or atr_val <= 0:
        return {"go": False, "fails": ["no usable ATR"], "stamp": "⛔ NO-GO: no usable ATR"}
    risk = float(cfg.get("stop_atr", 1.5)) * atr_val
    stop = entry - risk
    target = entry + float(cfg.get("rr", 2.0)) * risk
    rsi_series = outcome_lab.rsi(df["Close"])
    sma50 = df["Close"].rolling(50).mean()
    rsi_val = float(rsi_series.iloc[-1]) if pd.notna(rsi_series.iloc[-1]) else None
    sma50_val = float(sma50.iloc[-1]) if pd.notna(sma50.iloc[-1]) else None
    ext_pct = ((entry / sma50_val - 1) * 100) if sma50_val else None
    beta = float(cfg.get("beta", {}).get(ticker, 1.0))
    g = gate_hook.gate(
        entry=entry,
        stop=stop,
        target=target,
        atr=atr_val,
        ext_pct=ext_pct,
        rsi=rsi_val,
    )
    g["stamp"] = gate_hook.stamp_line(
        entry=entry,
        stop=stop,
        target=target,
        atr=atr_val,
        ext_pct=ext_pct,
        rsi=rsi_val,
        account=cfg.get("account"),
        beta=beta,
    )
    g["entry"] = entry
    g["atr14"] = atr_val
    g["rsi14"] = rsi_val
    g["ext_pct_vs_sma50"] = ext_pct
    return {k: _safe_float(v) if isinstance(v, (int, float, np.floating)) and k not in {"go", "auto"} else v for k, v in g.items()}


def evidence_for_setup(frames: dict[str, pd.DataFrame], setup: str, cfg: dict[str, Any]) -> dict[str, Any]:
    intraday = cfg.get("interval") in outcome_lab.INTRADAY
    p = dict(
        stop_atr=float(cfg.get("stop_atr", outcome_lab.DEFAULTS["stop_atr"])),
        rr=float(cfg.get("rr", outcome_lab.DEFAULTS["rr"])),
        time_bars=int(cfg.get("time_bars", outcome_lab.DEFAULTS["time_bars"])),
        spread_bps=float(cfg.get("spread_bps", outcome_lab.DEFAULTS["spread_bps"])),
        slippage_bps=float(cfg.get("slippage_bps", outcome_lab.DEFAULTS["slippage_bps"])),
        fees_bps=float(cfg.get("fees_bps", outcome_lab.DEFAULTS["fees_bps"])),
    )
    setup_fn, _ = outcome_lab.SETUPS[setup]
    baseline_fn, _ = outcome_lab.SETUPS["baseline_every5"]
    setup_trades: list[pd.DataFrame] = []
    base_trades: list[pd.DataFrame] = []
    per_ticker: dict[str, Any] = {}
    for tk, df in frames.items():
        tr = outcome_lab.simulate(df, setup_fn(df, intraday), p, intraday)
        bt = outcome_lab.simulate(df, baseline_fn(df, intraday), p, intraday)
        if not tr.empty:
            tr["ticker"] = tk
            setup_trades.append(tr)
        if not bt.empty:
            bt["ticker"] = tk
            base_trades.append(bt)
        per_ticker[tk] = outcome_lab.report_block(tr, f"{tk}:{setup}")
    pooled = pd.concat(setup_trades) if setup_trades else pd.DataFrame()
    baseline = pd.concat(base_trades) if base_trades else pd.DataFrame()
    pooled_report = outcome_lab.report_block(pooled, setup)
    baseline_report = outcome_lab.report_block(baseline, "baseline_every5")
    stats = EdgeStats(
        n=int(pooled_report.get("n", 0)),
        expectancy_r=float(pooled_report.get("e", 0.0)) if pooled_report.get("n", 0) else 0.0,
        conservative_expectancy_r=float(pooled_report.get("ec", 0.0)) if pooled_report.get("n", 0) else None,
        baseline_expectancy_r=float(baseline_report.get("e", 0.0)) if baseline_report.get("n", 0) else None,
    )
    return {
        "setup": setup,
        "pooled": pooled_report,
        "baseline": baseline_report,
        "confidence_bucket": confidence_bucket(stats, min_n=int(cfg.get("min_oos_samples", 100))),
        "beats_baseline": stats.beats_baseline,
        "per_ticker": per_ticker,
    }


def fetch_shared_regime() -> dict[str, Any]:
    frames, _sector = market_regime.fetch_regime_frames("NVDA", period="1y", interval="1d")
    return market_regime.score_market_regime(frames, sector_etf="SMH")


def build_report(cfg: dict[str, Any]) -> dict[str, Any]:
    tickers = cfg["tickers"]
    frames = fetch_frames(tickers, cfg.get("period", "2y"), cfg.get("interval", "1d"))
    setup = cfg.get("primary_setup", "breakout_20")
    evidence = evidence_for_setup(frames, setup, cfg)
    regime = fetch_shared_regime()
    rows = []
    for tk in tickers:
        df = frames[tk]
        setup_state = detect_setup_state(df, setup, int(cfg.get("max_signal_age_bars", 5)))
        g = make_gate(df, cfg, tk)
        action = "WAIT"
        reasons: list[str] = []
        if setup_state["signal_recent"] and g.get("go"):
            action = "GO_PROBATION" if evidence["confidence_bucket"] == "LOW" else "GO"
            reasons.append(f"{setup} signal within {setup_state['bars_since_signal']} bars")
        elif not setup_state["signal_recent"]:
            reasons.append(f"no {setup} signal in last {cfg.get('max_signal_age_bars', 5)} bars")
        if not g.get("go"):
            action = "NO_GO"
            reasons.extend(g.get("fails", []))
        if regime.get("regime") == "risk_off" and action.startswith("GO"):
            action = "WAIT_REGIME_CONFLICT"
            reasons.append("market regime risk_off")
        latest = df.iloc[-1]
        rows.append({
            "ticker": tk,
            "last_date": str(df.index[-1]),
            "price": _safe_float(latest["Close"], 2),
            "action": action,
            "reasons": reasons,
            "setup_state": setup_state,
            "trade_gate": g,
        })
    return {
        "ok": True,
        "config": {k: cfg[k] for k in ["name", "tickers", "primary_setup", "stop_atr", "rr", "time_bars"] if k in cfg},
        "regime": regime,
        "evidence": evidence,
        "rows": rows,
        "disclaimer": "Analytical/paper-trade planning only; not financial advice.",
    }


def format_text(report: dict[str, Any]) -> str:
    ev = report["evidence"]
    pooled = ev["pooled"]
    base = ev["baseline"]
    lines = []
    lines.append(f"Core tactic: {report['config']['primary_setup']} on {', '.join(report['config']['tickers'])}")
    lines.append(f"Regime: {report['regime'].get('regime')} | SPY/QQQ/SMH score {report['regime'].get('weighted_score')}")
    lines.append(
        f"Evidence: n={pooled.get('n', 0)} E[R]={pooled.get('e', 0):+.2f} "
        f"E_cons={pooled.get('ec', 0):+.2f} vs baseline {base.get('e', 0):+.2f} | {ev['confidence_bucket']}"
    )
    lines.append("")
    for row in report["rows"]:
        st = row["setup_state"]
        g = row["trade_gate"]
        signal = "now" if st["signal_now"] else (f"{st['bars_since_signal']} bars ago" if st["signal_recent"] else "none recent")
        lines.append(f"{row['ticker']}: {row['action']} @ {row['price']} | signal {signal} | trigger {st.get('trigger')}")
        lines.append(f"  {g.get('stamp')}")
        if row["reasons"]:
            lines.append("  why: " + "; ".join(str(x) for x in row["reasons"]))
    lines.append("Not financial advice.")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run configured NVDA/MU/SNDK tactic scanner")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--json", action="store_true", help="print full JSON report")
    args = ap.parse_args()
    cfg = load_config(Path(args.config))
    report = build_report(cfg)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True, default=str))
    else:
        print(format_text(report))


if __name__ == "__main__":
    main()
