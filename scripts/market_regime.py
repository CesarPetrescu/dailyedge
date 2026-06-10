#!/usr/bin/env python3
"""Market regime filter for Ultra Daytrader.

Uses broad-market and optional sector ETF technical state as a confidence
modifier. This is context, not a standalone forecast.
"""
from __future__ import annotations

import argparse
import json
import math
from typing import Any

import numpy as np
import pandas as pd

SECTOR_MAP = {
    "NVDA": "SMH", "MU": "SMH", "SNDK": "SMH", "AMD": "SMH", "AVGO": "SMH", "INTC": "SMH", "TSM": "SMH",
    "AAPL": "QQQ", "MSFT": "QQQ", "META": "QQQ", "GOOGL": "QQQ", "GOOG": "QQQ", "AMZN": "QQQ", "TSLA": "QQQ",
    "XOM": "XLE", "CVX": "XLE", "OXY": "XLE",
    "JPM": "XLF", "BAC": "XLF", "GS": "XLF", "MS": "XLF",
    "UNH": "XLV", "LLY": "XLV", "JNJ": "XLV",
}


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


def ema(close: pd.Series, span: int) -> pd.Series:
    return close.ewm(span=span, adjust=False).mean()


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    out = out.mask((loss == 0) & (gain > 0), 100.0)
    out = out.mask((gain == 0) & (loss > 0), 0.0)
    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().dropna(subset=["Close"])
    close = out["Close"]
    out["EMA20"] = ema(close, 20)
    out["SMA50"] = close.rolling(50).mean()
    out["SMA200"] = close.rolling(200).mean()
    out["RSI14"] = rsi(close)
    out["RET20"] = close / close.shift(20) - 1
    out["VOL20"] = close.pct_change().rolling(20).std()
    return out


def score_index(df: pd.DataFrame, symbol: str) -> dict[str, Any]:
    f = build_features(df) if "EMA20" not in df.columns else df
    latest = f.iloc[-1]
    price = float(latest["Close"])
    score = 0
    reasons: list[str] = []
    for name, pts in [("EMA20", 1), ("SMA50", 1), ("SMA200", 1)]:
        val = latest.get(name)
        if pd.notna(val):
            if price > float(val):
                score += pts
                reasons.append(f"above_{name.lower()}")
            else:
                score -= pts
                reasons.append(f"below_{name.lower()}")
    ret20 = latest.get("RET20")
    if pd.notna(ret20):
        if float(ret20) > 0.015:
            score += 1
            reasons.append("positive_20bar_momentum")
        elif float(ret20) < -0.015:
            score -= 1
            reasons.append("negative_20bar_momentum")
    r = latest.get("RSI14")
    if pd.notna(r):
        if 45 <= float(r) <= 72:
            score += 1
            reasons.append("constructive_rsi")
        elif float(r) < 40:
            score -= 1
            reasons.append("weak_rsi")
    label = "neutral"
    if score >= 2:
        label = "bullish"
    elif score <= -2:
        label = "bearish"
    return {
        "symbol": symbol,
        "score": int(score),
        "label": label,
        "price": safe_float(price),
        "rsi14": safe_float(r, 2),
        "ret20_pct": safe_float(float(ret20) * 100 if pd.notna(ret20) else None, 2),
        "reasons": reasons,
    }


def sector_etf_for_ticker(ticker: str | None) -> str | None:
    if not ticker:
        return None
    return SECTOR_MAP.get(ticker.upper().strip(), "SPY")


def score_market_regime(frames: dict[str, pd.DataFrame], sector_etf: str | None = None) -> dict[str, Any]:
    indices = {sym: score_index(df, sym) for sym, df in frames.items() if sym in {"SPY", "QQQ"}}
    sector = None
    if sector_etf and sector_etf in frames:
        sector = score_index(frames[sector_etf], sector_etf)
    weighted = 0.0
    if "SPY" in indices:
        weighted += indices["SPY"]["score"] * 1.2
    if "QQQ" in indices:
        weighted += indices["QQQ"]["score"] * 1.0
    if sector:
        weighted += sector["score"] * 0.6
    regime = "neutral_chop"
    modifier = 0
    if weighted >= 4:
        regime = "risk_on"
        modifier = 1
    elif weighted <= -4:
        regime = "risk_off"
        modifier = -1
    return {
        "ok": True,
        "regime": regime,
        "weighted_score": round(weighted, 2),
        "confidence_modifier": modifier,
        "indices": indices,
        "sector": sector,
        "note": "Regime filter only; not a standalone forecast.",
    }


def _downgrade_confidence(conf: str) -> str:
    order = ["low", "medium", "high"]
    if conf not in order:
        return "low"
    return order[max(0, order.index(conf) - 1)]


def apply_regime_modifier(combined: dict[str, Any], regime: dict[str, Any]) -> dict[str, Any]:
    out = dict(combined)
    warnings = list(out.get("warnings", []))
    bias = out.get("bias")
    reg = regime.get("regime")
    conflict = (bias == "bullish" and reg == "risk_off") or (bias == "bearish" and reg == "risk_on")
    if conflict:
        out["confidence"] = _downgrade_confidence(str(out.get("confidence", "low")))
        warnings.append("market_regime_conflict")
    elif (bias == "bullish" and reg == "risk_on") or (bias == "bearish" and reg == "risk_off"):
        warnings.append("market_regime_aligned")
    out["warnings"] = warnings
    return out


def fetch_regime_frames(ticker: str | None = None, period: str = "1y", interval: str = "1d") -> tuple[dict[str, pd.DataFrame], str | None]:
    import yfinance as yf

    sector = sector_etf_for_ticker(ticker)
    symbols = ["SPY", "QQQ"]
    if sector and sector not in symbols:
        symbols.append(sector)
    frames: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        df = yf.Ticker(sym).history(period=period, interval=interval, auto_adjust=False)
        if not df.empty:
            frames[sym] = df.dropna(subset=["Close"])
    return frames, sector


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ticker", nargs="?", default=None)
    ap.add_argument("--period", default="1y")
    ap.add_argument("--interval", default="1d")
    args = ap.parse_args()
    try:
        frames, sector = fetch_regime_frames(args.ticker, args.period, args.interval)
        result = score_market_regime(frames, sector_etf=sector)
        print(json.dumps(result, indent=2, sort_keys=True))
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2, sort_keys=True))
        raise SystemExit(2)


if __name__ == "__main__":
    main()
