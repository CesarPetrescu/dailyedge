#!/usr/bin/env python3
"""Universe scanner for Ultra Daytrader tactic candidates.

Purpose:
- expand beyond NVDA/MU/SNDK without guessing
- rank liquid names by setup quality + relative strength + no-chase filters
- export top candidates for watcher/full-analysis follow-up

This is a scanner, not a trade executor. Any GO/PROBATION candidate still needs
full ticker workflow, live news/fundamental/sentiment checks, and risk review.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
DEFAULT_CONFIG = ROOT / "config" / "universe_scanner.json"
OUTPUT_DIR = ROOT / "output"
sys.path.insert(0, str(SCRIPT_DIR))

import gate_hook
import ledger
import market_regime
import outcome_lab


def decide_action_state(
    signal_age: int | None,
    dist_to_trigger_pct: float | None,
    gate: dict[str, Any],
    regime: dict[str, Any],
    risk_state: dict[str, Any],
    days_to_earnings: int | None,
) -> tuple[str, list[str]]:
    """DailyEdge final action state machine.

    Fresh GO requires the setup on the last closed bar only (age <= 1). Stale
    signals are never GO; if price is back within 0-3% below trigger we arm a
    new alert, otherwise mark EXPIRED.
    """
    reasons: list[str] = []
    breaches = ledger.risk_breaches(risk_state)
    if breaches:
        return "SUPPRESSED_RISK", breaches
    if days_to_earnings is not None and abs(days_to_earnings) <= 2:
        return "NO_GO", [f"event_window earnings in {days_to_earnings} sessions"]
    if not gate.get("go"):
        return "NO_GO", list(gate.get("fails", ["gate_fail"]))
    if regime.get("regime") == "risk_off":
        return "NO_GO", ["market regime risk_off"]

    near_trigger = dist_to_trigger_pct is not None and -3.0 <= dist_to_trigger_pct <= 0.0
    strong_near = dist_to_trigger_pct is not None and dist_to_trigger_pct < -3.0
    if signal_age is not None and signal_age <= 1:
        return "GO_PROBATION", [f"fresh signal age {signal_age}"]
    if near_trigger:
        return "SET_ALERT", ["armed 0-3% below trigger"]
    if signal_age is not None and signal_age >= 2:
        return "EXPIRED", [f"signal {signal_age} bars ago, price vs trigger {dist_to_trigger_pct}%"]
    if strong_near:
        return "WATCH_STRONG", [">3% from trigger but trend/RS candidate"]
    return "WATCH", ["no fresh signal"]


def safe_float(x: Any, digits: int = 4):
    try:
        if x is None or pd.isna(x):
            return None
        f = float(x)
        if not math.isfinite(f):
            return None
        return round(f, digits)
    except Exception:
        return None


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_ticker(t: str) -> str:
    # Yahoo uses '-' for class tickers, not '.'
    return t.upper().strip().replace(".", "-")


def unique(seq: Iterable[str]) -> list[str]:
    seen = set()
    out = []
    for item in seq:
        tk = normalize_ticker(str(item))
        if tk and tk not in seen:
            seen.add(tk)
            out.append(tk)
    return out


def fetch_wikipedia_universe(kind: str) -> list[str]:
    """Fetch current broad universe constituents from Wikipedia via pandas."""
    if kind == "sp500":
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        return unique(tables[0]["Symbol"].astype(str).tolist())
    if kind in {"ndx", "nasdaq100", "nasdaq_100"}:
        tables = pd.read_html("https://en.wikipedia.org/wiki/Nasdaq-100")
        for tb in tables:
            cols = {str(c).lower(): c for c in tb.columns}
            if "ticker" in cols:
                return unique(tb[cols["ticker"]].astype(str).tolist())
            if "symbol" in cols:
                return unique(tb[cols["symbol"]].astype(str).tolist())
        raise ValueError("could not find ticker column in Nasdaq-100 page")
    raise ValueError(f"unknown live universe {kind}")


def resolve_universe(cfg: dict[str, Any], name: str | None, tickers: list[str] | None) -> list[str]:
    if tickers:
        return unique(tickers)
    uni = name or cfg.get("default_universe", "liquid_ai_semis")
    if uni in cfg.get("universes", {}):
        return unique(cfg["universes"][uni])
    if uni in {"sp500", "ndx", "nasdaq100", "nasdaq_100"}:
        return fetch_wikipedia_universe(uni)
    raise ValueError(f"unknown universe {uni}")


def fetch_bulk(tickers: list[str], period: str, interval: str) -> dict[str, pd.DataFrame]:
    import yfinance as yf

    data = yf.download(
        tickers,
        period=period,
        interval=interval,
        group_by="ticker",
        auto_adjust=False,
        progress=False,
        threads=True,
    )
    frames: dict[str, pd.DataFrame] = {}
    if data is None or data.empty:
        return frames
    if isinstance(data.columns, pd.MultiIndex):
        for tk in tickers:
            if tk not in data.columns.get_level_values(0):
                continue
            df = data[tk].copy()
            if not df.empty and {"Open", "High", "Low", "Close", "Volume"}.issubset(df.columns):
                frames[tk] = df.dropna(subset=["Close"])
    else:
        tk = tickers[0]
        frames[tk] = data.dropna(subset=["Close"])
    return frames


def features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().dropna(subset=["Close"])
    close = out["Close"]
    out["EMA9"] = close.ewm(span=9, adjust=False).mean()
    out["EMA20"] = close.ewm(span=20, adjust=False).mean()
    out["SMA50"] = close.rolling(50).mean()
    out["SMA200"] = close.rolling(200).mean()
    out["RSI14"] = outcome_lab.rsi(close)
    out["ATR14"] = outcome_lab.atr(out)
    out["VOL_AVG20"] = out["Volume"].rolling(20).mean()
    out["DOLLAR_VOL20"] = (out["Close"] * out["Volume"]).rolling(20).mean()
    out["RET20"] = close / close.shift(20) - 1
    out["RET63"] = close / close.shift(63) - 1
    out["HIGH20_PRIOR"] = out["High"].rolling(20).max().shift()
    out["BREAKOUT_20"] = (out["Close"] > out["HIGH20_PRIOR"]) & (out["Close"].shift() <= out["HIGH20_PRIOR"].shift())
    out["ANN_VOL60"] = close.pct_change().rolling(60).std() * (252 ** 0.5) * 100
    return out


def compute_relative_strength(frames: dict[str, pd.DataFrame], benchmark: str = "SPY") -> dict[str, dict[str, float | None]]:
    rs: dict[str, dict[str, float | None]] = {}
    bench = features(frames[benchmark]) if benchmark in frames and not frames[benchmark].empty else None
    bench_ret20 = float(bench["RET20"].iloc[-1]) if bench is not None and pd.notna(bench["RET20"].iloc[-1]) else 0.0
    bench_ret63 = float(bench["RET63"].iloc[-1]) if bench is not None and pd.notna(bench["RET63"].iloc[-1]) else 0.0
    for tk, df in frames.items():
        if df.empty:
            continue
        f = features(df)
        ret20 = float(f["RET20"].iloc[-1]) if pd.notna(f["RET20"].iloc[-1]) else None
        ret63 = float(f["RET63"].iloc[-1]) if pd.notna(f["RET63"].iloc[-1]) else None
        rs[tk] = {
            "rs20_vs_spy_pct": (ret20 - bench_ret20) * 100 if ret20 is not None else None,
            "rs63_vs_spy_pct": (ret63 - bench_ret63) * 100 if ret63 is not None else None,
        }
    return rs


def score_ticker(ticker: str, df: pd.DataFrame, cfg: dict[str, Any], rs: dict[str, Any], regime: dict[str, Any], risk: dict[str, Any] | None = None) -> dict[str, Any] | None:
    if df is None or df.empty or len(df) < 80:
        return None
    f = features(df)
    last = f.iloc[-1]
    prev = f.iloc[-2]
    price = float(last["Close"])
    atr = float(last["ATR14"]) if pd.notna(last["ATR14"]) else None
    if not atr or atr <= 0:
        return None
    dollar_vol = float(last["DOLLAR_VOL20"]) if pd.notna(last["DOLLAR_VOL20"]) else 0.0
    min_dv = float(cfg.get("min_avg_dollar_vol", 50_000_000))
    liquid = dollar_vol >= min_dv
    rsi = float(last["RSI14"]) if pd.notna(last["RSI14"]) else None
    sma50 = float(last["SMA50"]) if pd.notna(last["SMA50"]) else None
    ema20 = float(last["EMA20"]) if pd.notna(last["EMA20"]) else None
    ema9 = float(last["EMA9"]) if pd.notna(last["EMA9"]) else None
    high20 = float(last["HIGH20_PRIOR"]) if pd.notna(last["HIGH20_PRIOR"]) else None
    vol_avg = float(last["VOL_AVG20"]) if pd.notna(last["VOL_AVG20"]) and last["VOL_AVG20"] else None
    rel_vol = float(last["Volume"] / vol_avg) if vol_avg else None
    ext_pct = ((price / sma50 - 1) * 100) if sma50 else None
    dist_to_breakout = ((price / high20 - 1) * 100) if high20 else None
    ret20 = float(last["RET20"]) * 100 if pd.notna(last["RET20"]) else None
    ann_vol = float(last["ANN_VOL60"]) if pd.notna(last.get("ANN_VOL60")) else None
    breakout_now = bool(last["BREAKOUT_20"])
    recent_breakout = bool(f["BREAKOUT_20"].tail(5).any())
    bars_since = None
    if f["BREAKOUT_20"].any():
        last_sig = f.index[f["BREAKOUT_20"]][-1]
        bars_since = len(f.index) - 1 - f.index.get_loc(last_sig)

    gate_entry = high20 if high20 and dist_to_breakout is not None and dist_to_breakout <= 0 else price
    stop_mult = 2.0 if ann_vol is not None and ann_vol >= 80 else float(cfg.get("stop_atr", 1.5))
    stop = gate_entry - stop_mult * atr
    target = gate_entry + float(cfg.get("rr", 2.0)) * (gate_entry - stop)
    gate = gate_hook.gate(entry=gate_entry, stop=stop, target=target, atr=atr, ann_vol=ann_vol, ext_pct=ext_pct, rsi=rsi)
    gate_stamp = None

    score = 0.0
    reasons: list[str] = []
    penalties: list[str] = []
    if liquid:
        score += 10
    else:
        score -= 25
        penalties.append("illiquid_vs_threshold")
    if breakout_now:
        score += 30
        reasons.append("breakout_20_now")
    elif recent_breakout:
        score += 18
        reasons.append(f"breakout_20_recent_{bars_since}b")
    elif dist_to_breakout is not None and -3 <= dist_to_breakout <= 0:
        score += 12
        reasons.append("near_20d_breakout")
    elif dist_to_breakout is not None and 0 < dist_to_breakout <= 5:
        score += 8
        reasons.append("above_20d_breakout")

    if ema9 and ema20 and price > ema9 > ema20:
        score += 12
        reasons.append("ema_stack_bullish")
    elif ema20 and price > ema20:
        score += 6
        reasons.append("above_ema20")
    else:
        score -= 10
        penalties.append("below_ema20")

    if sma50 and price > sma50:
        score += 8
        reasons.append("above_sma50")
    if rsi is not None:
        if 50 <= rsi <= 72:
            score += 10
            reasons.append("constructive_rsi")
        elif rsi > float(cfg.get("max_rsi_no_chase", 78)):
            score -= 15
            penalties.append("rsi_chase")
        elif rsi < 45:
            score -= 8
            penalties.append("weak_rsi")
    if rel_vol is not None:
        if rel_vol >= 1.5:
            score += 10
            reasons.append("rel_volume_hot")
        elif rel_vol >= 1.0:
            score += 4
            reasons.append("rel_volume_ok")
    rs20 = rs.get(ticker, {}).get("rs20_vs_spy_pct")
    rs63 = rs.get(ticker, {}).get("rs63_vs_spy_pct")
    if rs20 is not None:
        if rs20 > 5:
            score += 10
            reasons.append("rs20_strong_vs_spy")
        elif rs20 < -5:
            score -= 8
            penalties.append("rs20_weak_vs_spy")
    if rs63 is not None and rs63 > 10:
        score += 6
        reasons.append("rs63_strong_vs_spy")
    if ext_pct is not None and ext_pct > float(cfg.get("max_ext_pct_vs_sma50", 25)):
        score -= 18
        penalties.append("extended_vs_sma50")
    if not gate.get("go"):
        score -= 20
        penalties.extend(gate.get("fails", []))
    if regime.get("regime") == "risk_on":
        score += 5
    elif regime.get("regime") == "risk_off":
        score -= 10
        penalties.append("risk_off_regime")

    action, state_reasons = decide_action_state(
        signal_age=bars_since,
        dist_to_trigger_pct=dist_to_breakout,
        gate=gate,
        regime=regime,
        risk_state=risk or ledger.empty_risk_state(),
        days_to_earnings=None,
    )
    if not liquid:
        action = "NO_GO"
        state_reasons.append("illiquid")
    if action == "WATCH" and score >= 45:
        action = "WATCH_STRONG"
    gate_stamp = gate_hook.state_stamp(action, entry=gate_entry, stop=stop, target=target, atr=atr, ann_vol=ann_vol, ext_pct=ext_pct, rsi=rsi, account=cfg.get("account"))

    return {
        "ticker": ticker,
        "action": action,
        "score": round(score, 2),
        "price": safe_float(price, 2),
        "change_pct": safe_float((price / float(prev["Close"]) - 1) * 100 if prev["Close"] else None, 2),
        "breakout_trigger": safe_float(high20, 2),
        "dist_to_breakout_pct": safe_float(dist_to_breakout, 2),
        "breakout_now": breakout_now,
        "recent_breakout": recent_breakout,
        "bars_since_breakout": bars_since,
        "rsi14": safe_float(rsi, 2),
        "rel_volume": safe_float(rel_vol, 2),
        "ann_vol_pct": safe_float(ann_vol, 1),
        "days_to_earnings": None,
        "avg_dollar_vol20_m": safe_float(dollar_vol / 1_000_000, 1),
        "ret20_pct": safe_float(ret20, 2),
        "rs20_vs_spy_pct": safe_float(rs20, 2),
        "rs63_vs_spy_pct": safe_float(rs63, 2),
        "ext_pct_vs_sma50": safe_float(ext_pct, 2),
        "gate": {
            "go": gate.get("go"),
            "rr": safe_float(gate.get("rr"), 2),
            "cost_R": safe_float(gate.get("cost_R"), 3),
            "p_be_pct": safe_float((gate.get("p_be") or 0) * 100, 1),
            "stop": safe_float(gate.get("stop"), 2),
            "target": safe_float(gate.get("target"), 2),
            "fails": gate.get("fails", []),
            "stamp": gate_stamp,
        },
        "sector": ledger.sector_cluster(ticker),
        "reasons": state_reasons + reasons,
        "penalties": penalties,
        "last_date": str(f.index[-1]),
    }


def scan(cfg: dict[str, Any], universe: str | None = None, tickers: list[str] | None = None, limit: int | None = None) -> dict[str, Any]:
    base_tickers = resolve_universe(cfg, universe, tickers)
    # Always add SPY for relative strength and broad regime context.
    fetch_tickers = unique(base_tickers + ["SPY", "QQQ", "SMH"])
    frames = fetch_bulk(fetch_tickers, cfg.get("period", "1y"), cfg.get("interval", "1d"))
    regime_frames = {k: v for k, v in frames.items() if k in {"SPY", "QQQ", "SMH"}}
    regime = market_regime.score_market_regime(regime_frames, sector_etf="SMH") if regime_frames else {"ok": False, "regime": "unknown"}
    rs = compute_relative_strength(frames, "SPY")
    risk = ledger.risk_state(OUTPUT_DIR / "alerts_log.csv")
    rows = []
    unscored = []
    for tk in base_tickers:
        if tk not in frames or frames.get(tk) is None or frames[tk].empty:
            unscored.append({"ticker": tk, "reason": "missing_fetch_or_empty"})
            continue
        row = score_ticker(tk, frames[tk], cfg, rs, regime, risk=risk)
        if row:
            rows.append(row)
        else:
            reason = "score_filter_or_insufficient_features"
            if len(frames[tk]) < 80:
                reason = f"too_few_bars_{len(frames[tk])}"
            unscored.append({"ticker": tk, "reason": reason})
    rows.sort(key=lambda r: r["score"], reverse=True)
    if limit is None:
        limit = int(cfg.get("candidate_limit", 25))
    rows = rows[:limit]
    now = datetime.now(timezone.utc).isoformat()
    return {
        "ok": True,
        "generated_at": now,
        "universe": universe or cfg.get("default_universe"),
        "input_count": len(base_tickers),
        "scored_count": len(rows),
        "unscored_count": len(unscored),
        "unscored": unscored,
        "regime": regime,
        "risk_state": risk,
        "concentration_warning": ledger.concentration_warning(rows),
        "rows": rows,
        "disclaimer": "Scanner only; final candidates require full live ticker workflow and risk review. Not financial advice.",
    }


def export_outputs(report: dict[str, Any], cfg: dict[str, Any]) -> dict[str, str]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    latest_json = OUTPUT_DIR / "universe_scanner_latest.json"
    latest_txt = OUTPUT_DIR / "universe_scanner_latest.txt"
    watcher_json = OUTPUT_DIR / "watcher_candidates.json"
    latest_json.write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")
    latest_txt.write_text(format_text(report), encoding="utf-8")
    try:
        top3 = [r["ticker"] for r in report["rows"][:3]]
        frames = fetch_bulk(unique(top3 + ["SPY", "SMH"]), "6mo", "1d")
        import full_ticker_report
        report["rs_panel"] = full_ticker_report.render_rs_panel(top3, frames, OUTPUT_DIR)
        latest_json.write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")
    except Exception:
        pass
    export_limit = int(cfg.get("watcher_export_limit", 10))
    watcher = {
        "generated_at": report["generated_at"],
        "source": "universe_scanner.py",
        "primary_setup": cfg.get("primary_setup", "breakout_20_rr3"),
        "candidates": [r for r in report["rows"] if r["action"] in {"GO_PROBATION", "SET_ALERT", "WATCH_STRONG"}][:export_limit],
    }
    watcher_json.write_text(json.dumps(watcher, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return {"json": str(latest_json), "text": str(latest_txt), "watcher": str(watcher_json)}


def format_text(report: dict[str, Any]) -> str:
    lines = []
    lines.append(f"Universe scanner: {report['universe']} | scored {report['scored_count']}/{report['input_count']} | unscored {report.get('unscored_count', 0)} | generated {report.get('generated_at')}")
    lines.append(f"Regime: {report.get('regime', {}).get('regime')} | score {report.get('regime', {}).get('weighted_score')}")
    if report.get("concentration_warning"):
        lines.append(f"Concentration warning: {report['concentration_warning']}")
    lines.append("Top candidates:")
    for i, r in enumerate(report["rows"][:15], 1):
        lines.append(
            f"{i}. {r['ticker']} {r['action']} score {r['score']} @ {r['price']} | "
            f"trig {r['breakout_trigger']} ({r['dist_to_breakout_pct']}%) | "
            f"RSI {r['rsi14']} RVOL {r['rel_volume']} RS20 {r['rs20_vs_spy_pct']}% | "
            f"proj stop {r['gate']['stop']} proj tgt {r['gate']['target']}"
        )
        why = ", ".join(r["reasons"][:4]) if r["reasons"] else "-"
        pen = ", ".join(str(x) for x in r["penalties"][:3]) if r["penalties"] else "none"
        lines.append(f"   why: {why} | flags: {pen}")
    lines.append("Scanner only; run full ticker workflow before any trade. Not financial advice.")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Rank liquid universe for Ultra Daytrader setup candidates")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--universe", default=None, help="config universe name, sp500, or ndx")
    ap.add_argument("--tickers", nargs="*", help="explicit tickers override")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--no-export", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    report = scan(cfg, universe=args.universe, tickers=args.tickers, limit=args.limit)
    paths = {} if args.no_export else export_outputs(report, cfg)
    if args.json:
        out = dict(report, output_paths=paths)
        print(json.dumps(out, indent=2, sort_keys=True, default=str))
    else:
        print(format_text(report))
        if paths:
            print("Outputs:", paths)


if __name__ == "__main__":
    main()
