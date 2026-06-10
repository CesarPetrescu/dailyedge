#!/usr/bin/env python3
"""Local technical snapshot for Ultra Daytrader.

Free-data helper. It is not a trading recommendation engine.
Requires: yfinance pandas numpy
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

# Allow importing sibling helper scripts when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent))


def _fail(msg: str, code: int = 2) -> None:
    print(json.dumps({"ok": False, "error": msg}, indent=2))
    raise SystemExit(code)


def rsi(close, window=14):
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, math.nan)
    return 100 - (100 / (1 + rs))


def ema(close, span):
    return close.ewm(span=span, adjust=False).mean()


def atr(df, window=14):
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = high_low.to_frame("hl").join(high_close.to_frame("hc")).join(low_close.to_frame("lc")).max(axis=1)
    return tr.rolling(window).mean()


def pivots(series, n=3, kind="support"):
    vals = []
    arr = series.dropna()
    for i in range(n, len(arr) - n):
        win = arr.iloc[i - n : i + n + 1]
        val = arr.iloc[i]
        if kind == "support" and val == win.min():
            vals.append(float(val))
        if kind == "resistance" and val == win.max():
            vals.append(float(val))
    # cluster by rounding to 2 decimals and recency; keep last unique-ish levels
    out = []
    for v in vals[-20:]:
        if all(abs(v - x) / max(abs(x), 1) > 0.01 for x in out):
            out.append(v)
    return sorted(out[-5:])


def safe_float(x):
    try:
        if x is None or (hasattr(x, "isna") and x.isna()):
            return None
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return None
        return round(float(x), 4)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ticker")
    ap.add_argument("--period", default="6mo")
    ap.add_argument("--interval", default="1d")
    args = ap.parse_args()

    try:
        import yfinance as yf
    except Exception as e:
        _fail(f"missing dependency yfinance: {e}")

    ticker = args.ticker.upper().strip()
    t = yf.Ticker(ticker)
    df = t.history(period=args.period, interval=args.interval, auto_adjust=False)
    if df.empty:
        _fail(f"no OHLCV data returned for {ticker}")

    close = df["Close"]
    df["EMA9"] = ema(close, 9)
    df["EMA20"] = ema(close, 20)
    df["SMA50"] = close.rolling(50).mean()
    df["SMA200"] = close.rolling(200).mean()
    df["RSI14"] = rsi(close)
    ema12 = ema(close, 12)
    ema26 = ema(close, 26)
    df["MACD"] = ema12 - ema26
    df["MACD_SIGNAL"] = ema(df["MACD"], 9)
    df["ATR14"] = atr(df)
    df["VOL_AVG20"] = df["Volume"].rolling(20).mean()
    # Intraday VWAP needs intraday bars; this is a close*volume approximation for period context.
    df["VWAP_APPROX"] = (df["Close"] * df["Volume"]).cumsum() / df["Volume"].replace(0, math.nan).cumsum()

    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else last
    supports = pivots(df["Low"], kind="support")
    resistances = pivots(df["High"], kind="resistance")
    try:
        import levels as auto_levels

        structure = auto_levels.detect_all(df)
        watcher_rules = auto_levels.levels_to_rules(structure.get("sr", []), structure.get("last", float(last["Close"])))
    except Exception as e:
        structure = {"ok": False, "error": str(e)}
        watcher_rules = []

    price = float(last["Close"])
    trade_gate = None
    try:
        from gate_hook import gate, stamp_line

        atr_value = float(last["ATR14"]) if not math.isnan(last["ATR14"]) else None
        rsi_value = float(last["RSI14"]) if not math.isnan(last["RSI14"]) else None
        sma50_value = float(last["SMA50"]) if not math.isnan(last["SMA50"]) else None
        ext_pct = ((price / sma50_value - 1) * 100) if sma50_value else None
        trade_gate = gate(entry=price, atr=atr_value, ext_pct=ext_pct, rsi=rsi_value)
        trade_gate["stamp"] = stamp_line(entry=price, atr=atr_value, ext_pct=ext_pct, rsi=rsi_value)
        trade_gate["note"] = "Auto-structured gate only; replace with structural stop/target when available."
    except Exception as e:
        trade_gate = {"ok": False, "error": str(e)}
    trend = "neutral"
    if price > last["EMA20"] > last["SMA50"] if not math.isnan(last["SMA50"]) else price > last["EMA20"]:
        trend = "bullish"
    elif price < last["EMA20"] < last["SMA50"] if not math.isnan(last["SMA50"]) else price < last["EMA20"]:
        trend = "bearish"

    result = {
        "ok": True,
        "ticker": ticker,
        "period": args.period,
        "interval": args.interval,
        "last_date": str(df.index[-1]),
        "price": safe_float(price),
        "change_pct_last_bar": safe_float((last["Close"] / prev["Close"] - 1) * 100 if prev["Close"] else None),
        "trend_read": trend,
        "indicators": {
            "ema9": safe_float(last["EMA9"]),
            "ema20": safe_float(last["EMA20"]),
            "sma50": safe_float(last["SMA50"]),
            "sma200": safe_float(last["SMA200"]),
            "rsi14": safe_float(last["RSI14"]),
            "macd": safe_float(last["MACD"]),
            "macd_signal": safe_float(last["MACD_SIGNAL"]),
            "atr14": safe_float(last["ATR14"]),
            "vwap_approx": safe_float(last["VWAP_APPROX"]),
            "volume": safe_float(last["Volume"]),
            "volume_vs_20d_avg_pct": safe_float((last["Volume"] / last["VOL_AVG20"] - 1) * 100 if last["VOL_AVG20"] else None),
        },
        "levels": {
            "support_candidates": [round(x, 2) for x in supports],
            "resistance_candidates": [round(x, 2) for x in resistances],
            "atr_stop_1x_below": safe_float(price - last["ATR14"] if not math.isnan(last["ATR14"]) else None),
            "atr_target_2x_above": safe_float(price + 2 * last["ATR14"] if not math.isnan(last["ATR14"]) else None),
            "auto_structure": structure,
            "watcher_rules": watcher_rules,
        },
        "trade_gate": trade_gate,
        "note": "Free-data technical snapshot only; combine with live web/news/fundamental/sentiment checks.",
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
