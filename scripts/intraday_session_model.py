#!/usr/bin/env python3
"""Intraday VWAP/opening-range model for Ultra Daytrader."""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


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


def build_intraday_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().dropna(subset=["Close"])
    typical = (out["High"] + out["Low"] + out["Close"]) / 3
    # For tests with high=low=close this is exactly close*volume VWAP.
    pv = typical * out["Volume"].replace(0, np.nan)
    out["VWAP"] = pv.cumsum() / out["Volume"].replace(0, np.nan).cumsum()
    out["EMA9"] = out["Close"].ewm(span=9, adjust=False).mean()
    out["EMA20"] = out["Close"].ewm(span=20, adjust=False).mean()
    out["VOL_AVG20"] = out["Volume"].rolling(20, min_periods=3).mean()
    out["REL_VOLUME"] = out["Volume"] / out["VOL_AVG20"].replace(0, np.nan)
    return out


def _ny_time(ts: pd.Timestamp) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        return t.tz_localize("America/New_York")
    return t.tz_convert("America/New_York")


def session_phase(timestamp: pd.Timestamp) -> str:
    t = _ny_time(timestamp)
    minutes = t.hour * 60 + t.minute
    if minutes < 9 * 60 + 30:
        return "premarket"
    if minutes < 10 * 60 + 30:
        return "open_drive"
    if minutes < 15 * 60:
        return "midday"
    if minutes < 16 * 60:
        return "power_hour"
    return "closed"


def opening_range_levels(df: pd.DataFrame, minutes: int = 60) -> dict[str, Any]:
    if df.empty:
        return {"opening_range_high": None, "opening_range_low": None, "opening_range_minutes": minutes}
    start = _ny_time(df.index[0])
    # 15m bars at 09:30, 09:45, 10:00: a 45m opening range includes first 3 bars.
    cutoff = start + pd.Timedelta(minutes=minutes)
    local_index = pd.DatetimeIndex([_ny_time(x) for x in df.index])
    subset = df.loc[local_index < cutoff]
    if subset.empty:
        subset = df.iloc[:1]
    return {
        "opening_range_high": safe_float(subset["High"].max(), 4),
        "opening_range_low": safe_float(subset["Low"].min(), 4),
        "opening_range_minutes": minutes,
    }


def relative_volume(df: pd.DataFrame) -> dict[str, Any]:
    f = build_intraday_features(df) if "REL_VOLUME" not in df.columns else df
    rv = safe_float(f["REL_VOLUME"].iloc[-1], 2)
    interp = "unknown"
    if rv is not None:
        if rv >= 1.5:
            interp = "high"
        elif rv >= 1.1:
            interp = "above_average"
        elif rv <= 0.7:
            interp = "low"
        else:
            interp = "normal"
    return {"relative_volume": rv, "interpretation": interp}


def detect_intraday_setup(df: pd.DataFrame, opening_minutes: int = 60) -> dict[str, Any]:
    f = build_intraday_features(df) if "VWAP" not in df.columns else df
    levels = opening_range_levels(f, opening_minutes)
    last = f.iloc[-1]
    prev = f.iloc[-2] if len(f) > 1 else last
    price = float(last["Close"])
    vwap = float(last["VWAP"])
    orh = levels["opening_range_high"]
    orl = levels["opening_range_low"]
    rv = relative_volume(f)["relative_volume"] or 1.0
    phase = session_phase(f.index[-1])

    if orh is not None and price > float(orh) and rv >= 1.0:
        return {"name": "opening_range_breakout", "direction": "long", "confidence": "high" if rv >= 1.5 else "medium"}
    if orl is not None and price < float(orl) and rv >= 1.0:
        return {"name": "opening_range_breakdown", "direction": "short", "confidence": "high" if rv >= 1.5 else "medium"}
    if float(prev["Close"]) <= float(prev["VWAP"]) and price > vwap:
        return {"name": "vwap_reclaim", "direction": "long", "confidence": "medium"}
    if float(prev["Close"]) >= float(prev["VWAP"]) and price < vwap:
        return {"name": "vwap_reject", "direction": "short", "confidence": "medium"}
    if phase == "power_hour" and price > vwap and float(last["EMA9"]) > float(last["EMA20"]):
        return {"name": "late_day_momentum", "direction": "long", "confidence": "medium"}
    return {"name": "range_chop", "direction": "neutral", "confidence": "low"}


def analyze_intraday_session(df: pd.DataFrame, ticker: str, opening_range_minutes: int = 60, outdir: str | None = None) -> dict[str, Any]:
    f = build_intraday_features(df)
    levels = opening_range_levels(f, opening_range_minutes)
    setup = detect_intraday_setup(f, opening_range_minutes)
    last = f.iloc[-1]
    direction = setup["direction"]
    bias = "neutral"
    if direction == "long":
        bias = "bullish"
    elif direction == "short":
        bias = "bearish"
    result = {
        "ok": True,
        "ticker": ticker.upper(),
        "bias": bias,
        "session_phase": session_phase(f.index[-1]),
        "setup": setup,
        "levels": {
            "last_price": safe_float(last["Close"]),
            "vwap": safe_float(last["VWAP"]),
            "ema9": safe_float(last["EMA9"]),
            "ema20": safe_float(last["EMA20"]),
            **levels,
        },
        "volume": relative_volume(f),
        "note": "Intraday session model; not financial advice.",
    }
    if outdir:
        result["chart_png"] = make_intraday_chart(ticker.upper(), f, result, Path(outdir))
    return result


def make_intraday_chart(ticker: str, df: pd.DataFrame, result: dict[str, Any], outdir: Path) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    png = outdir / f"{ticker}_{stamp}_intraday.png"
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [2.4, 0.8]}, constrained_layout=True)
    tail = df.tail(120)
    ax1.plot(tail.index, tail["Close"], label="Close", linewidth=1.5)
    ax1.plot(tail.index, tail["VWAP"], label="VWAP", linewidth=1.2)
    ax1.plot(tail.index, tail["EMA9"], label="EMA9", linewidth=0.9)
    ax1.plot(tail.index, tail["EMA20"], label="EMA20", linewidth=0.9)
    orh = result["levels"].get("opening_range_high")
    orl = result["levels"].get("opening_range_low")
    if orh is not None:
        ax1.axhline(orh, color="#2ca02c", linestyle="--", linewidth=1, label="OR high")
    if orl is not None:
        ax1.axhline(orl, color="#d62728", linestyle="--", linewidth=1, label="OR low")
    ax1.set_title(f"{ticker} intraday — {result['setup']['name']} / {result['bias']} / {result['session_phase']}")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="upper left")
    colors = np.where(tail["Close"].diff().fillna(0) >= 0, "#2ca02c", "#d62728")
    ax2.bar(tail.index, tail["Volume"], color=colors, alpha=0.5)
    ax2.set_title(f"RelVol: {result['volume'].get('relative_volume')} ({result['volume'].get('interpretation')})")
    ax2.grid(True, alpha=0.2)
    fig.savefig(png, dpi=150)
    plt.close(fig)
    return str(png)


def fetch_intraday(ticker: str, period: str, interval: str) -> pd.DataFrame:
    import yfinance as yf
    df = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=False)
    if df.empty:
        raise RuntimeError(f"no intraday data returned for {ticker}")
    return df.dropna(subset=["Close"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ticker")
    ap.add_argument("--period", default="10d")
    ap.add_argument("--interval", default="15m")
    ap.add_argument("--opening-range-minutes", type=int, default=60)
    ap.add_argument("--outdir", default="/root/trading-agents/ultra-daytrader/output")
    args = ap.parse_args()
    try:
        df = fetch_intraday(args.ticker.upper(), args.period, args.interval)
        result = analyze_intraday_session(df, args.ticker, args.opening_range_minutes, args.outdir)
        print(json.dumps(result, indent=2, sort_keys=True))
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2, sort_keys=True))
        raise SystemExit(2)


if __name__ == "__main__":
    main()
