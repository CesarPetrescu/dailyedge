#!/usr/bin/env python3
"""Generate technical charts + Monte Carlo simulation for Ultra Daytrader.

Uses free yfinance data. GPU acceleration is optional: if CuPy is installed and
`--backend auto|gpu` is selected, simulations can run on GPU; otherwise NumPy CPU
is used. This is analytical/paper-trade tooling only, not financial advice.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd


def ema(close: pd.Series, span: int) -> pd.Series:
    return close.ewm(span=span, adjust=False).mean()


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def choose_backend(kind: str):
    if kind == "cpu":
        return np, "cpu:numpy"
    try:
        import cupy as cp  # type: ignore
        # Touch device to verify runtime works.
        _ = cp.cuda.runtime.getDeviceCount()
        return cp, "gpu:cupy"
    except Exception as exc:
        if kind == "gpu":
            raise RuntimeError(f"GPU backend requested but CuPy/CUDA is unavailable: {exc}")
        return np, "cpu:numpy"


def simulate_paths(last_price: float, log_returns: np.ndarray, days: int, paths: int, backend: str, seed: int):
    xp, backend_name = choose_backend(backend)
    mu = float(np.nanmean(log_returns))
    sigma = float(np.nanstd(log_returns, ddof=1))
    if not math.isfinite(mu) or not math.isfinite(sigma) or sigma <= 0:
        raise RuntimeError("not enough valid return history for simulation")
    if backend_name.startswith("gpu"):
        xp.random.seed(seed)
        shocks = xp.random.normal(mu, sigma, size=(paths, days))
        sim = float(last_price) * xp.exp(xp.cumsum(shocks, axis=1))
        pct = xp.percentile(sim, [5, 25, 50, 75, 95], axis=0)
        terminal = sim[:, -1]
        stats = {
            "terminal_p05": float(xp.percentile(terminal, 5).get()),
            "terminal_p25": float(xp.percentile(terminal, 25).get()),
            "terminal_p50": float(xp.percentile(terminal, 50).get()),
            "terminal_p75": float(xp.percentile(terminal, 75).get()),
            "terminal_p95": float(xp.percentile(terminal, 95).get()),
            "prob_up": float((xp.mean(terminal > last_price) * 100).get()),
            "prob_down_5pct": float((xp.mean(terminal < last_price * 0.95) * 100).get()),
            "prob_up_5pct": float((xp.mean(terminal > last_price * 1.05) * 100).get()),
        }
        pct_np = pct.get()
    else:
        rng = np.random.default_rng(seed)
        shocks = rng.normal(mu, sigma, size=(paths, days))
        sim = float(last_price) * np.exp(np.cumsum(shocks, axis=1))
        pct_np = np.percentile(sim, [5, 25, 50, 75, 95], axis=0)
        terminal = sim[:, -1]
        stats = {
            "terminal_p05": float(np.percentile(terminal, 5)),
            "terminal_p25": float(np.percentile(terminal, 25)),
            "terminal_p50": float(np.percentile(terminal, 50)),
            "terminal_p75": float(np.percentile(terminal, 75)),
            "terminal_p95": float(np.percentile(terminal, 95)),
            "prob_up": float(np.mean(terminal > last_price) * 100),
            "prob_down_5pct": float(np.mean(terminal < last_price * 0.95) * 100),
            "prob_up_5pct": float(np.mean(terminal > last_price * 1.05) * 100),
        }
    stats.update({"mu_daily_log": mu, "sigma_daily_log": sigma, "backend": backend_name})
    return pct_np, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ticker")
    ap.add_argument("--period", default="1y")
    ap.add_argument("--interval", default="1d")
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--paths", type=int, default=5000)
    ap.add_argument("--backend", choices=["auto", "cpu", "gpu"], default="auto")
    ap.add_argument("--outdir", default="/root/trading-agents/ultra-daytrader/output")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import yfinance as yf
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ticker = args.ticker.upper().strip()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = yf.Ticker(ticker).history(period=args.period, interval=args.interval, auto_adjust=False)
    if df.empty:
        raise SystemExit(json.dumps({"ok": False, "error": f"no OHLCV data returned for {ticker}"}))
    df = df.dropna(subset=["Close"])
    close = df["Close"]
    df["EMA9"] = ema(close, 9)
    df["EMA20"] = ema(close, 20)
    df["SMA50"] = close.rolling(50).mean()
    df["RSI14"] = rsi(close)
    df["VOL_AVG20"] = df["Volume"].rolling(20).mean()
    log_returns = np.log(close / close.shift(1)).dropna().to_numpy()
    last_price = float(close.iloc[-1])

    percentiles, sim_stats = simulate_paths(last_price, log_returns, args.days, args.paths, args.backend, args.seed)
    future_x = np.arange(1, args.days + 1)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    png = outdir / f"{ticker}_{stamp}_chart_sim.png"
    js = outdir / f"{ticker}_{stamp}_chart_sim.json"

    fig = plt.figure(figsize=(14, 10), constrained_layout=True)
    gs = fig.add_gridspec(3, 1, height_ratios=[2.2, 0.8, 1.4])

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(df.index, df["Close"], label="Close", linewidth=1.6)
    ax1.plot(df.index, df["EMA9"], label="EMA9", linewidth=1.0)
    ax1.plot(df.index, df["EMA20"], label="EMA20", linewidth=1.0)
    ax1.plot(df.index, df["SMA50"], label="SMA50", linewidth=1.0)
    ax1.set_title(f"{ticker} technical snapshot — {args.period}/{args.interval}")
    ax1.set_ylabel("Price")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="upper left")

    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    vol_colors = np.where(df["Close"].diff().fillna(0) >= 0, "#2ca02c", "#d62728")
    ax2.bar(df.index, df["Volume"], color=vol_colors, alpha=0.45, label="Volume")
    ax2.plot(df.index, df["VOL_AVG20"], color="black", linewidth=1.0, label="20d avg vol")
    ax2.set_ylabel("Volume")
    ax2.grid(True, alpha=0.2)
    ax2.legend(loc="upper left")

    ax3 = fig.add_subplot(gs[2, 0])
    ax3.plot(future_x, percentiles[2], label="Median", color="#1f77b4")
    ax3.fill_between(future_x, percentiles[1], percentiles[3], color="#1f77b4", alpha=0.20, label="25-75%")
    ax3.fill_between(future_x, percentiles[0], percentiles[4], color="#1f77b4", alpha=0.10, label="5-95%")
    ax3.axhline(last_price, color="black", linestyle="--", linewidth=1, label=f"Last {last_price:.2f}")
    ax3.set_title(f"Monte Carlo fan: {args.paths:,} paths / {args.days} trading days / {sim_stats['backend']} — NOT A FORECAST")
    ax3.text(0.5, 0.5, "NOT A FORECAST", transform=ax3.transAxes, ha="center", va="center", fontsize=34, alpha=0.16, weight="bold")
    ax3.set_xlabel("Trading days forward")
    ax3.set_ylabel("Simulated price")
    ax3.grid(True, alpha=0.25)
    ax3.legend(loc="upper left")

    fig.savefig(png, dpi=150)
    fig.savefig(outdir / f"{ticker}_latest_chart_sim.png", dpi=150)
    plt.close(fig)

    latest = df.iloc[-1]
    result = {
        "ok": True,
        "ticker": ticker,
        "chart_png": str(png),
        "summary_json": str(js),
        "last_price": round(last_price, 4),
        "last_date": str(df.index[-1]),
        "indicators": {
            "ema9": round(float(latest["EMA9"]), 4),
            "ema20": round(float(latest["EMA20"]), 4),
            "sma50": None if pd.isna(latest["SMA50"]) else round(float(latest["SMA50"]), 4),
            "rsi14": None if pd.isna(latest["RSI14"]) else round(float(latest["RSI14"]), 4),
            "volume_vs_20d_avg_pct": None if pd.isna(latest["VOL_AVG20"]) else round((float(latest["Volume"]) / float(latest["VOL_AVG20"]) - 1) * 100, 2),
        },
        "simulation": {k: round(v, 4) if isinstance(v, float) else v for k, v in sim_stats.items()},
        "disclaimer": "Analytical simulation using historical-return assumptions; not financial advice and not a forecast.",
    }
    js.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
