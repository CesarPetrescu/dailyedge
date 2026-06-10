#!/usr/bin/env python3
"""
outcome_lab.py — DailyEdge v2 triple-barrier setup backtester
==============================================================
Answers the only question that matters:

    "If I enter on THIS setup with THIS stop and THIS target,
     was that trade structure historically profitable AFTER costs?"

This replaces price-direction prediction (up/down in N days) with
trade-outcome labeling (triple barrier):

    entry  = next bar's open after the signal       (no lookahead)
    stop   = entry − k × ATR(14 at signal bar)       (volatility-sized)
    target = entry + RR × stop_distance
    time   = exit at close after N bars (or session close for intraday)
    label  = first barrier touched; same-bar both-touch counts as STOP
             (conservative). Costs subtracted from every outcome.

Per setup it reports: n, hit rate + Wilson 95% CI, avg win/loss R,
net expectancy (point AND conservative/Wilson-LB), profit factor,
MAE/MFE percentiles (for stop/target placement), bars held.

A BASELINE setup (enter every 5th bar, same structure) is always available:
if your setup's expectancy ≈ baseline's, the setup adds nothing — the
structure is doing the work, or nothing is.

Usage:
  python3 outcome_lab.py --list-setups
  python3 outcome_lab.py NVDA --setup ema_cross --period 2y --interval 1d
  python3 outcome_lab.py NVDA --setup all --period 2y --interval 1d
  python3 outcome_lab.py NVDA --setup breakout_20 --sweep            # stop/target grid
  python3 outcome_lab.py NVDA --setup ema_cross --regime             # split by SPY trend
  python3 outcome_lab.py NVDA --setup vwap_reclaim --interval 15m --period 60d
  python3 outcome_lab.py NVDA MU SNDK --setup ema_cross --period 2y  # pooled multi-ticker

Honesty contract (printed because it must be):
  * Everything here is IN-SAMPLE on tickers you chose because they ran.
  * Sweeps guarantee lucky cells (multiple testing). Use sweeps to REJECT
    structures, never to pick the shiniest cell and size on it.
  * Backtests reject; only the forward ledger (edge_engine analyze) validates.

Not financial advice.
"""

import argparse
import math
import sys

import numpy as np
import pandas as pd

try:
    from edge_engine import PARAMS as EDGE_PARAMS, cost_in_R as engine_cost_in_R
except Exception:  # keeps the lab usable as a standalone pasted script
    EDGE_PARAMS = {}
    engine_cost_in_R = None

Z95 = 1.959964
DEFAULTS = dict(stop_atr=EDGE_PARAMS.get("min_stop_atr", 1.5), rr=2.0, time_bars=20,
                spread_bps=EDGE_PARAMS.get("spread_bps", 3.0),
                slippage_bps=EDGE_PARAMS.get("slippage_bps", 4.0),
                fees_bps=EDGE_PARAMS.get("fees_bps", 0.0))

INTRADAY = {"1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"}


# ---------------------------------------------------------------- data
def fetch(ticker: str, period: str, interval: str) -> pd.DataFrame:
    import yfinance as yf
    df = yf.Ticker(ticker).history(period=period, interval=interval,
                                   auto_adjust=True)
    if df is None or df.empty:
        sys.exit(f"No data for {ticker} ({period}/{interval}).")
    if isinstance(df.columns, pd.MultiIndex):          # yfinance quirk
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.title)[["Open", "High", "Low", "Close", "Volume"]]
    df = df.dropna()
    # data-quality gate (your step 3, automated)
    bad = (df["High"] < df["Low"]).sum()
    if bad:
        df = df[df["High"] >= df["Low"]]
        print(f"  [data] dropped {bad} inverted bars on {ticker}")
    if len(df) < 60:
        sys.exit(f"Only {len(df)} bars for {ticker} — not enough to label outcomes.")
    return df


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def session_vwap(df: pd.DataFrame) -> pd.Series:
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    day = df.index.date
    pv = (tp * df["Volume"]).groupby(day).cumsum()
    vv = df["Volume"].groupby(day).cumsum().replace(0, np.nan)
    return pv / vv


# ---------------------------------------------------------------- setups
# Each returns a boolean Series: True at the SIGNAL bar (entry = next open).
def s_ema_cross(df, intraday):
    e9 = df["Close"].ewm(span=9, adjust=False).mean()
    e20 = df["Close"].ewm(span=20, adjust=False).mean()
    return (e9 > e20) & (e9.shift() <= e20.shift())

def s_rsi_reclaim_30(df, intraday):
    r = rsi(df["Close"])
    return (r > 30) & (r.shift() <= 30)

def s_breakout_20(df, intraday):
    hh = df["High"].rolling(20).max().shift()
    return (df["Close"] > hh) & (df["Close"].shift() <= hh.shift())

def s_pullback_trend(df, intraday):
    sma50 = df["Close"].rolling(50).mean()
    e20 = df["Close"].ewm(span=20, adjust=False).mean()
    return (df["Close"] > sma50) & (df["Low"] <= e20) & (df["Close"] > e20)

def s_vwap_reclaim(df, intraday):
    if not intraday:
        return pd.Series(False, index=df.index)
    vw = session_vwap(df)
    below_run = (df["Close"] < vw).rolling(3).sum().shift()
    return (df["Close"] > vw) & (df["Close"].shift() <= vw.shift()) & (below_run >= 3)

def s_orb_60(df, intraday):
    """Break of the first-hour high (opening range breakout)."""
    if not intraday:
        return pd.Series(False, index=df.index)
    day = pd.Series(df.index.date, index=df.index)
    first_ts = day.groupby(day).transform(lambda s: s.index.min())
    mins_in = (df.index - pd.DatetimeIndex(first_ts)).total_seconds() / 60
    in_or = mins_in < 60
    or_high = df["High"].where(in_or).groupby(day).cummax().ffill()
    sig = (~in_or) & (df["Close"] > or_high) & (df["Close"].shift() <= or_high.shift())
    return sig.fillna(False)

def s_baseline_every5(df, intraday):
    s = pd.Series(False, index=df.index)
    s.iloc[::5] = True
    return s

SETUPS = {
    "ema_cross":      (s_ema_cross,      "EMA9 crosses above EMA20"),
    "rsi_reclaim_30": (s_rsi_reclaim_30, "RSI(14) reclaims 30 from below"),
    "breakout_20":    (s_breakout_20,    "Close breaks prior 20-bar high"),
    "pullback_trend": (s_pullback_trend, "Above SMA50, dip to EMA20, close back above"),
    "vwap_reclaim":   (s_vwap_reclaim,   "[intraday] reclaim session VWAP after >=3 bars below"),
    "orb_60":         (s_orb_60,         "[intraday] break of first-60-min high"),
    "baseline_every5":(s_baseline_every5,"CONTROL: enter every 5th bar, same structure"),
}


# ---------------------------------------------------------------- stats
def wilson(wins, n, z=Z95):
    if n == 0:
        return 0.0, 0.0, 1.0
    p = wins / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / den
    return p, max(0.0, c - h), min(1.0, c + h)


def cost_R(stop_pct, p):
    if engine_cost_in_R is not None:
        return engine_cost_in_R(stop_pct, p["spread_bps"], p["slippage_bps"], p["fees_bps"])
    per_side = p["spread_bps"] / 2 + p["slippage_bps"] + p["fees_bps"]
    return (2 * per_side / 100.0) / stop_pct if stop_pct > 0 else float("inf")


# ---------------------------------------------------------------- engine
def simulate(df: pd.DataFrame, sig: pd.Series, p: dict, intraday: bool) -> pd.DataFrame:
    """Triple-barrier walk. Long-only (matches current trading). No overlapping
    positions (single-position trader realism). Conservative same-bar rule."""
    a = atr(df)
    o, h, l, c = (df[x].to_numpy() for x in ("Open", "High", "Low", "Close"))
    av = a.to_numpy()
    sg = sig.fillna(False).to_numpy()
    dates = df.index
    last_of_day = None
    if intraday:
        day = np.array(dates.date)
        last_of_day = np.append(day[:-1] != day[1:], True)

    trades, i, n = [], 0, len(df)
    while i < n - 2:
        if not sg[i] or np.isnan(av[i]) or av[i] <= 0:
            i += 1
            continue
        entry = o[i + 1]
        risk = p["stop_atr"] * av[i]
        stop, target = entry - risk, entry + p["rr"] * risk
        stop_pct = risk / entry * 100
        cR = cost_R(stop_pct, p)
        mae = mfe = 0.0
        j, outcome, exit_px = i + 1, "time", None
        max_j = min(n - 1, i + p["time_bars"])
        while j <= max_j:
            mae = max(mae, (entry - l[j]) / risk)
            mfe = max(mfe, (h[j] - entry) / risk)
            hit_stop, hit_tgt = l[j] <= stop, h[j] >= target
            if hit_stop:                       # conservative: stop wins ties
                outcome, exit_px = "stop", stop
                break
            if hit_tgt:
                outcome, exit_px = "target", target
                break
            if intraday and last_of_day[j]:    # never hold intraday overnight
                outcome, exit_px = "eod", c[j]
                break
            j += 1
        if exit_px is None:
            j = max_j
            exit_px = c[j]
        r_gross = (exit_px - entry) / risk
        trades.append(dict(ts=dates[i + 1], entry=entry, stop=stop, target=target,
                           outcome=outcome, bars=j - i, R_gross=r_gross,
                           R_net=r_gross - cR, MAE_R=mae, MFE_R=mfe,
                           stop_pct=stop_pct, cost_R=cR))
        i = j + 1                              # no overlap: resume after exit
    return pd.DataFrame(trades)


def report_block(tr: pd.DataFrame, label: str) -> dict:
    n = len(tr)
    if n == 0:
        return {"label": label, "n": 0}
    rs = tr["R_net"].to_numpy()
    wins, losses = rs[rs > 0], rs[rs <= 0]
    p_hat, p_lo, p_hi = wilson(len(wins), n)
    aw = wins.mean() if len(wins) else 0.0
    al = abs(losses.mean()) if len(losses) else 1.0
    e_pt = rs.mean()
    e_cons = p_lo * aw - (1 - p_lo) * al
    gw, gl = wins.sum(), abs(losses.sum())
    pf = gw / gl if gl > 0 else float("inf")
    return dict(label=label, n=n, hit=p_hat, lo=p_lo, hi=p_hi, aw=aw, al=al,
                e=e_pt, ec=e_cons, pf=pf,
                bars=tr["bars"].mean(), cR=tr["cost_R"].mean(),
                mae_w_p80=(np.percentile(tr.loc[tr.R_net > 0, "MAE_R"], 80)
                           if (tr.R_net > 0).any() else float("nan")),
                mfe_l_p50=(np.percentile(tr.loc[tr.R_net <= 0, "MFE_R"], 50)
                           if (tr.R_net <= 0).any() else float("nan")))


def print_table(rows, title):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    hdr = (f"{'setup':<18}{'n':>4}{'hit%':>6}{'95%CI':>12}{'avgW':>6}{'avgL':>6}"
           f"{'E[R]net':>8}{'E_cons':>8}{'PF':>6}{'bars':>6}{'costR':>7}"
           f"{'MAEw80':>7}{'MFEl50':>7}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        if r["n"] == 0:
            print(f"{r['label']:<18}   0  (no signals — wrong interval or no occurrences)")
            continue
        pf = f"{r['pf']:.2f}" if np.isfinite(r["pf"]) else "inf"
        print(f"{r['label']:<18}{r['n']:>4}{r['hit']*100:>5.0f}%"
              f"[{r['lo']*100:>3.0f}–{r['hi']*100:<3.0f}]"
              f"{r['aw']:>6.2f}{-r['al']:>6.2f}{r['e']:>8.2f}{r['ec']:>8.2f}"
              f"{pf:>6}{r['bars']:>6.1f}{r['cR']:>7.3f}"
              f"{r['mae_w_p80']:>7.2f}{r['mfe_l_p50']:>7.2f}")
    print("-" * len(hdr))
    print("MAEw80 = 80th-pct adverse excursion of WINNERS  -> a stop tighter than this")
    print("         kills winning trades. MFEl50 = median favorable excursion of LOSERS")
    print("         -> partial-profit level losers reached before dying.")


# ---------------------------------------------------------------- modes
def run_setups(args, p):
    intraday = args.interval in INTRADAY
    names = list(SETUPS) if args.setup == "all" else [args.setup]
    pooled = {nm: [] for nm in names}
    for tk in args.tickers:
        df = fetch(tk, args.period, args.interval)
        print(f"{tk}: {len(df)} bars  {df.index[0].date()} -> {df.index[-1].date()}  "
              f"({args.interval})")
        for nm in names:
            fn, _ = SETUPS[nm]
            tr = simulate(df, fn(df, intraday), p, intraday)
            if not tr.empty:
                tr["ticker"] = tk
                pooled[nm].append(tr)

    rows = []
    for nm in names:
        tr = pd.concat(pooled[nm]) if pooled[nm] else pd.DataFrame()
        rows.append(report_block(tr, nm))
    rows.sort(key=lambda r: r.get("ec", -9e9), reverse=True)
    tks = "+".join(args.tickers)
    print_table(rows, f"TRIPLE-BARRIER OUTCOMES — {tks}  "
                      f"(stop {p['stop_atr']}xATR, target {p['rr']}R, "
                      f"time {p['time_bars']} bars, costs on)")

    if args.regime and not args.interval in INTRADAY:
        regime_split(args, p, names, pooled)

    print("\nHONESTY: in-sample, on tickers chosen because they ran. A setup must")
    print("beat baseline_every5 AND survive the forward ledger before it gets size.")


def regime_split(args, p, names, pooled):
    import yfinance as yf
    spy = yf.Ticker("SPY").history(period=args.period, interval="1d", auto_adjust=True)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    sma = spy["Close"].rolling(50).mean()
    up = (sma.diff(10) > 0)
    up.index = up.index.date
    for nm in names:
        if not pooled[nm]:
            continue
        tr = pd.concat(pooled[nm])
        tag = tr["ts"].map(lambda t: up.get(t.date(), np.nan))
        rows = [report_block(tr[tag == True], f"{nm} | SPY up"),
                report_block(tr[tag == False], f"{nm} | SPY down")]
        print_table(rows, f"REGIME SPLIT — {nm}")


def run_sweep(args, p):
    intraday = args.interval in INTRADAY
    fn, _ = SETUPS[args.setup]
    stops = [1.0, 1.5, 2.0, 2.5, 3.0]
    rrs = [1.5, 2.0, 3.0]
    cells = {}
    for tk in args.tickers:
        df = fetch(tk, args.period, args.interval)
        sig = fn(df, intraday)
        for k in stops:
            for q in rrs:
                pp = dict(p, stop_atr=k, rr=q)
                tr = simulate(df, sig, pp, intraday)
                cells.setdefault((k, q), []).append(tr)
    print("\n" + "=" * 78)
    print(f"STOP x TARGET SWEEP — {args.setup} on {'+'.join(args.tickers)} "
          f"({args.period}/{args.interval})   cell = E_net[R] (n)")
    print("=" * 78)
    print(f"{'stop/RR':>8}" + "".join(f"{q:>16}" for q in rrs))
    for k in stops:
        line = f"{k:>6.1f}x "
        for q in rrs:
            tr = pd.concat(cells[(k, q)]) if cells[(k, q)] else pd.DataFrame()
            if tr.empty:
                line += f"{'-':>16}"
            else:
                line += f"{tr['R_net'].mean():>+9.2f} ({len(tr):>3})"
        print(line)
    print("\nMULTIPLE-TESTING WARNING: 15 cells guarantee lucky ones. Use this grid")
    print("to REJECT structures (whole rows/cols negative) — never to cherry-pick")
    print("the best cell and size on it. Robust = a region of positive cells.")


def main():
    ap = argparse.ArgumentParser(description="Triple-barrier setup backtester")
    ap.add_argument("tickers", nargs="*", help="one or more tickers")
    ap.add_argument("--setup", default="ema_cross",
                    help=f"{list(SETUPS)} or 'all'")
    ap.add_argument("--period", default="2y")
    ap.add_argument("--interval", default="1d")
    ap.add_argument("--stop-atr", type=float, default=DEFAULTS["stop_atr"])
    ap.add_argument("--rr", type=float, default=DEFAULTS["rr"])
    ap.add_argument("--time-bars", type=int, default=DEFAULTS["time_bars"])
    ap.add_argument("--spread-bps", type=float, default=DEFAULTS["spread_bps"])
    ap.add_argument("--slippage-bps", type=float, default=DEFAULTS["slippage_bps"])
    ap.add_argument("--fees-bps", type=float, default=DEFAULTS["fees_bps"])
    ap.add_argument("--sweep", action="store_true", help="stop x target grid")
    ap.add_argument("--regime", action="store_true", help="split by SPY 50d trend")
    ap.add_argument("--list-setups", action="store_true")
    args = ap.parse_args()

    if args.list_setups:
        for k, (_, d) in SETUPS.items():
            print(f"  {k:<18} {d}")
        return
    if not args.tickers:
        ap.error("provide at least one ticker (or --list-setups)")
    if args.setup != "all" and args.setup not in SETUPS:
        ap.error(f"unknown setup {args.setup}; see --list-setups")

    p = dict(stop_atr=args.stop_atr, rr=args.rr, time_bars=args.time_bars,
             spread_bps=args.spread_bps, slippage_bps=args.slippage_bps,
             fees_bps=args.fees_bps)
    if args.sweep:
        run_sweep(args, p)
    else:
        run_setups(args, p)


if __name__ == "__main__":
    main()
