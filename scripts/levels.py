#!/usr/bin/env python3
"""Auto support/resistance + volume-profile detector for Ultra Daytrader.

Free-data helper. Daily volume profile is approximate because daily bars do not
show true intrabar volume distribution.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class Zone:
    kind: str
    low: float
    high: float
    mid: float
    touches: int
    strength: float
    last_touch: str | None
    distance_pct: float | None = None


def _num(x: Any, digits: int = 4) -> float | None:
    try:
        f = float(x)
        if math.isnan(f) or math.isinf(f):
            return None
        return round(f, digits)
    except Exception:
        return None


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    cols = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"missing OHLCV columns: {missing}")
    out = df[cols].copy().dropna()
    if out.empty:
        raise ValueError("empty OHLCV data")
    return out


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window).mean()


def swing_pivots(df: pd.DataFrame, n: int = 3) -> list[dict[str, Any]]:
    pivots: list[dict[str, Any]] = []
    highs, lows = df["High"], df["Low"]
    for i in range(n, len(df) - n):
        hwin = highs.iloc[i - n : i + n + 1]
        lwin = lows.iloc[i - n : i + n + 1]
        ts = str(df.index[i])
        if highs.iloc[i] == hwin.max():
            pivots.append({"kind": "resistance", "price": float(highs.iloc[i]), "index": i, "date": ts})
        if lows.iloc[i] == lwin.min():
            pivots.append({"kind": "support", "price": float(lows.iloc[i]), "index": i, "date": ts})
    return pivots


def cluster_zones(
    pivots: list[dict[str, Any]],
    last_price: float,
    tolerance: float,
    max_zones: int = 8,
) -> list[Zone]:
    zones: list[dict[str, Any]] = []
    for p in sorted(pivots, key=lambda x: x["price"]):
        price = float(p["price"])
        bucket = None
        for z in zones:
            if abs(price - z["mid"]) <= tolerance:
                bucket = z
                break
        if bucket is None:
            zones.append({"kind": p["kind"], "prices": [price], "dates": [p["date"]], "last_index": p["index"], "mid": price})
        else:
            bucket["prices"].append(price)
            bucket["dates"].append(p["date"])
            bucket["last_index"] = max(bucket["last_index"], p["index"])
            bucket["mid"] = float(np.mean(bucket["prices"]))
    out: list[Zone] = []
    for z in zones:
        prices = z["prices"]
        low, high, mid = min(prices), max(prices), float(np.mean(prices))
        touches = len(prices)
        recency = z["last_index"] / max(max((p["index"] for p in pivots), default=1), 1)
        strength = touches + 0.75 * recency
        kind = "support" if mid <= last_price else "resistance"
        out.append(
            Zone(
                kind=kind,
                low=low,
                high=high,
                mid=mid,
                touches=touches,
                strength=strength,
                last_touch=z["dates"][-1] if z["dates"] else None,
                distance_pct=(mid / last_price - 1) * 100 if last_price else None,
            )
        )
    return sorted(out, key=lambda z: (z.strength, z.touches), reverse=True)[:max_zones]


def volume_profile(df: pd.DataFrame, bins: int = 48, value_area_pct: float = 0.70) -> dict[str, Any]:
    lo, hi = float(df["Low"].min()), float(df["High"].max())
    if not hi > lo:
        return {}
    edges = np.linspace(lo, hi, bins + 1)
    vols = np.zeros(bins, dtype=float)
    for _, r in df.iterrows():
        rlo, rhi, vol = float(r["Low"]), float(r["High"]), float(r["Volume"])
        if vol <= 0:
            continue
        mask = (edges[:-1] <= rhi) & (edges[1:] >= rlo)
        count = int(mask.sum())
        if count:
            vols[mask] += vol / count
    centers = (edges[:-1] + edges[1:]) / 2
    poc_i = int(np.argmax(vols))
    total = float(vols.sum())
    order = list(np.argsort(vols)[::-1])
    chosen, running = [], 0.0
    for i in order:
        chosen.append(i)
        running += float(vols[i])
        if total and running / total >= value_area_pct:
            break
    vah = float(edges[max(chosen) + 1]) if chosen else None
    val = float(edges[min(chosen)]) if chosen else None
    top = sorted(
        [{"price": float(centers[i]), "volume_share_pct": float(vols[i] / total * 100) if total else 0.0} for i in range(bins)],
        key=lambda x: x["volume_share_pct"],
        reverse=True,
    )[:5]
    return {
        "poc": _num(centers[poc_i], 4),
        "value_area_low": _num(val, 4),
        "value_area_high": _num(vah, 4),
        "bins": bins,
        "top_hvn": [{"price": _num(x["price"], 4), "volume_share_pct": _num(x["volume_share_pct"], 2)} for x in top],
        "note": "Approximate when built from daily bars; sharper with intraday bars.",
    }


def round_levels(last_price: float, atr_value: float | None = None) -> list[dict[str, Any]]:
    if last_price <= 0:
        return []
    step = 1.0
    if last_price >= 1000:
        step = 50.0
    elif last_price >= 100:
        step = 10.0
    elif last_price >= 20:
        step = 5.0
    base = math.floor(last_price / step) * step
    vals = sorted(set(base + step * k for k in range(-3, 4) if base + step * k > 0))
    return [{"price": _num(v, 2), "distance_pct": _num((v / last_price - 1) * 100, 2)} for v in vals]


def regression_channel(df: pd.DataFrame, lookback: int = 60, stdevs: float = 2.0) -> dict[str, Any]:
    sub = df.tail(min(lookback, len(df)))
    if len(sub) < 10:
        return {}
    y = sub["Close"].astype(float).to_numpy()
    x = np.arange(len(y), dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    fit = slope * x + intercept
    resid = y - fit
    sd = float(np.std(resid))
    mid = float(fit[-1])
    return {
        "lookback": int(len(sub)),
        "slope_per_bar": _num(slope, 4),
        "mid": _num(mid, 4),
        "upper": _num(mid + stdevs * sd, 4),
        "lower": _num(mid - stdevs * sd, 4),
        "trend": "up" if slope > 0 else "down" if slope < 0 else "flat",
    }


def levels_to_rules(sr: list[dict[str, Any]], last_price: float, max_rules: int = 5) -> list[dict[str, Any]]:
    rules = []
    for z in sorted(sr, key=lambda x: x.get("strength", 0), reverse=True)[:max_rules]:
        kind = z["kind"]
        if kind == "support":
            trigger = z["low"]
            op = "cross_below"
        else:
            trigger = z["high"]
            op = "cross_above"
        rules.append(
            {
                "type": "price_level",
                "op": op,
                "level": _num(trigger, 4),
                "zone_mid": _num(z["mid"], 4),
                "zone": [_num(z["low"], 4), _num(z["high"], 4)],
                "strength": _num(z.get("strength"), 2),
                "touches": int(z.get("touches", 0)),
                "message": f"{kind} zone {z['low']:.2f}-{z['high']:.2f} touched {z.get('touches', 0)}x",
            }
        )
    return rules


def detect_all(df: pd.DataFrame, pivot_n: int = 3, atr_mult: float = 0.50, bins: int = 48) -> dict[str, Any]:
    df = normalize_ohlcv(df)
    last = float(df["Close"].iloc[-1])
    atr_series = atr(df)
    atr_value = float(atr_series.dropna().iloc[-1]) if not atr_series.dropna().empty else float((df["High"] - df["Low"]).tail(20).mean())
    tolerance = max(atr_value * atr_mult, last * 0.002)
    piv = swing_pivots(df, n=pivot_n)
    zones = cluster_zones(piv, last_price=last, tolerance=tolerance)
    sr = [z.__dict__ for z in zones]
    for z in sr:
        for k in ["low", "high", "mid", "strength", "distance_pct"]:
            z[k] = _num(z[k], 4 if k != "distance_pct" else 2)
    return {
        "ok": True,
        "last": _num(last, 4),
        "atr": _num(atr_value, 4),
        "tolerance": _num(tolerance, 4),
        "pivot_n": pivot_n,
        "sr": sr,
        "volume_profile": volume_profile(df, bins=bins),
        "round_levels": round_levels(last, atr_value),
        "regression_channel": regression_channel(df),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ticker")
    ap.add_argument("--period", default="6mo")
    ap.add_argument("--interval", default="1d")
    ap.add_argument("--pivot-n", type=int, default=3)
    ap.add_argument("--atr-mult", type=float, default=0.5)
    ap.add_argument("--bins", type=int, default=48)
    ap.add_argument("--rules", action="store_true")
    args = ap.parse_args()
    import yfinance as yf

    ticker = args.ticker.upper().strip()
    df = yf.download(ticker, period=args.period, interval=args.interval, progress=False, auto_adjust=False, threads=False)
    res = detect_all(df, pivot_n=args.pivot_n, atr_mult=args.atr_mult, bins=args.bins)
    res["ticker"] = ticker
    if args.rules:
        res["watcher_rules"] = levels_to_rules(res["sr"], res["last"])
    print(json.dumps(res, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
