#!/usr/bin/env python3
"""DailyEdge report image renderer: daily setup + relative strength panel."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BG, FG, GRID = "#0d1117", "#e6e6e6", "#2a2f3a"
UP, DN = "#26a69a", "#ef5350"
C_TRIG, C_STOP, C_TGT, C_ENTRY = "#ffd166", "#ef5350", "#26a69a", "#7fb3ff"


def _fetch(ticker: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    import yfinance as yf
    df = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.rename(columns=str.title).dropna()


def _rsi(c: pd.Series, n: int = 14) -> pd.Series:
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def _sr_zones(df: pd.DataFrame, atr_now: float, k: int = 3, w: int = 5) -> list[tuple[float, int]]:
    h, l = df["High"].to_numpy(), df["Low"].to_numpy()
    piv: list[float] = []
    for i in range(w, len(df) - w):
        if h[i] == h[i - w:i + w + 1].max():
            piv.append(float(h[i]))
        if l[i] == l[i - w:i + w + 1].min():
            piv.append(float(l[i]))
    if not piv:
        return []
    piv.sort()
    tol = 0.75 * atr_now
    clusters, cur = [], [piv[0]]
    for p in piv[1:]:
        if p - cur[-1] <= tol:
            cur.append(p)
        else:
            clusters.append(cur); cur = [p]
    clusters.append(cur)
    ranked = sorted(clusters, key=len, reverse=True)[:k]
    return [(float(np.mean(c)), len(c)) for c in ranked]


def _style(ax):
    ax.set_facecolor(BG)
    ax.grid(True, color=GRID, alpha=0.5, linewidth=0.6)
    ax.tick_params(colors=FG, labelsize=8)
    for s in ax.spines.values():
        s.set_color(GRID)


def render_daily_setup(ticker: str, trigger=None, stop=None, target=None, entry=None,
                       target2=None, er_days=None, action: str = "", period: str = "6mo",
                       outdir: str | Path = ".") -> str:
    df = _fetch(ticker, period)
    if df.empty:
        raise SystemExit(f"no data for {ticker}")
    o, h, l, c, v = (df[x].to_numpy() for x in ("Open", "High", "Low", "Close", "Volume"))
    x = np.arange(len(df))
    e9 = df["Close"].ewm(span=9, adjust=False).mean()
    e20 = df["Close"].ewm(span=20, adjust=False).mean()
    s50 = df["Close"].rolling(50).mean()
    rsi = _rsi(df["Close"])
    atr_now = float(_atr(df).iloc[-1])

    fig = plt.figure(figsize=(12.8, 9.0), dpi=100, facecolor=BG)
    gs = fig.add_gridspec(5, 1, hspace=0.06)
    axp = fig.add_subplot(gs[0:3, 0]); axv = fig.add_subplot(gs[3, 0], sharex=axp); axr = fig.add_subplot(gs[4, 0], sharex=axp)
    for ax in (axp, axv, axr): _style(ax)
    plt.setp(axp.get_xticklabels(), visible=False); plt.setp(axv.get_xticklabels(), visible=False)

    up = c >= o
    axp.vlines(x, l, h, color=np.where(up, UP, DN), linewidth=0.8, alpha=0.9)
    axp.bar(x[up], (c - o)[up], 0.65, bottom=o[up], color=UP, alpha=0.95)
    axp.bar(x[~up], (o - c)[~up], 0.65, bottom=c[~up], color=DN, alpha=0.95)
    axp.plot(x, e9, color="#58a6ff", lw=1.1, label="EMA9")
    axp.plot(x, e20, color="#d2a8ff", lw=1.1, label="EMA20")
    axp.plot(x, s50, color="#f0883e", lw=1.3, label="SMA50")

    explicit = [float(p) for p in (trigger, stop, target, target2, entry) if p is not None]
    for lvl, touches in _sr_zones(df, atr_now):
        if any(abs(lvl - p) < 0.5 * atr_now for p in explicit):
            continue
        axp.axhspan(lvl - 0.3 * atr_now, lvl + 0.3 * atr_now, color="#8b949e", alpha=0.12)
        axp.text(0.5, lvl, f"S/R {lvl:,.0f} ({touches}t)", color="#8b949e", fontsize=7, va="center")

    def hline(yv, color, label):
        if yv is None: return
        axp.axhline(float(yv), color=color, lw=1.4, ls="--" if label == "TRIGGER" else "-")
        axp.text(len(x) * 1.002, float(yv), f" {label} {float(yv):,.2f}", color=color, fontsize=8.5, va="center", fontweight="bold")

    if entry is not None and trigger is not None and abs(float(entry) - float(trigger)) < 1e-6:
        entry = None
    hline(trigger, C_TRIG, "TRIGGER"); hline(entry, C_ENTRY, "ENTRY"); hline(stop, C_STOP, "STOP"); hline(target, C_TGT, "T1"); hline(target2, C_TGT, "T2")
    axp.set_xlim(-1, len(x) * 1.12)
    title = f"{ticker}  ${c[-1]:,.2f}"
    if action: title += f"  —  {action}"
    title += f"  —  as of {df.index[-1].date()}"
    if er_days is not None: title += f"   |   earnings in {er_days}d"
    axp.set_title(title, color=FG, fontsize=13, fontweight="bold", loc="left")
    leg = axp.legend(loc="upper left", fontsize=8, framealpha=0.15)
    for t in leg.get_texts(): t.set_color(FG)

    axv.bar(x, v, 0.7, color=np.where(up, UP, DN), alpha=0.7); axv.bar(x[-1], v[-1], 0.7, color="#ffd166"); axv.set_ylabel("Vol", color=FG, fontsize=8)
    axr.plot(x, rsi, color="#58a6ff", lw=1.1); axr.axhline(70, color=DN, lw=0.8, alpha=0.6); axr.axhline(30, color=UP, lw=0.8, alpha=0.6)
    axr.set_ylim(0, 100); axr.set_ylabel("RSI", color=FG, fontsize=8)
    step = max(1, len(x) // 8); axr.set_xticks(x[::step]); axr.set_xticklabels([d.strftime("%b %d") for d in df.index[::step]], color=FG, fontsize=8)
    fig.text(0.99, 0.005, "DailyEdge — analytical only, not financial advice", color="#8b949e", fontsize=7, ha="right")
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    p = out / f"{ticker}_{ts}_daily_setup.png"
    fig.savefig(p, facecolor=BG, bbox_inches="tight")
    fig.savefig(out / f"{ticker}_latest_daily_setup.png", facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    return str(p)


def render_rs_panel(ticker: str, benches=("SPY", "SMH"), period: str = "6mo", outdir: str | Path = ".") -> str:
    t = _fetch(ticker, period)["Close"]
    fig, ax = plt.subplots(figsize=(12.8, 4.2), dpi=100, facecolor=BG)
    _style(ax)
    colors = ["#58a6ff", "#f0883e", "#d2a8ff"]
    for i, b in enumerate(benches):
        bc = _fetch(b, period)["Close"]
        ratio = (t / bc.reindex(t.index).ffill()).dropna()
        ratio = ratio / ratio.iloc[0] * 100
        ax.plot(ratio.index, ratio, lw=1.4, color=colors[i % 3], label=f"{ticker}/{b}  ({ratio.iloc[-1]-100:+.1f}%)")
    ax.axhline(100, color="#8b949e", lw=0.8, alpha=0.6)
    ax.set_title(f"{ticker} relative strength (rebased=100)", color=FG, fontsize=12, fontweight="bold", loc="left")
    leg = ax.legend(loc="upper left", fontsize=9, framealpha=0.15)
    for tx in leg.get_texts(): tx.set_color(FG)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    p = out / f"{ticker}_{ts}_rs_panel.png"
    fig.savefig(p, facecolor=BG, bbox_inches="tight")
    fig.savefig(out / f"{ticker}_latest_rs_panel.png", facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    return str(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ticker"); ap.add_argument("--trigger", type=float); ap.add_argument("--entry", type=float); ap.add_argument("--stop", type=float); ap.add_argument("--target", type=float); ap.add_argument("--target2", type=float); ap.add_argument("--er-days", type=int); ap.add_argument("--action", default=""); ap.add_argument("--outdir", default="."); ap.add_argument("--rs-only", action="store_true")
    a = ap.parse_args()
    if not a.rs_only:
        print(render_daily_setup(a.ticker, a.trigger, a.stop, a.target, a.entry, a.target2, a.er_days, a.action, outdir=a.outdir))
    print(render_rs_panel(a.ticker, outdir=a.outdir))

if __name__ == "__main__":
    main()
