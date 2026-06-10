#!/usr/bin/env python3
"""Refresh pooled outcome evidence for active universe into output/evidence.json."""
from __future__ import annotations
import argparse, json, sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
OUTPUT = ROOT / "output" / "evidence.json"
sys.path.insert(0, str(SCRIPT_DIR))

import outcome_lab
import universe_scanner


def _period_plus_250d(period: str) -> str:
    if period.endswith("y") and period[:-1].isdigit():
        return f"{int(period[:-1]) + 1}y"
    if period.endswith("mo") and period[:-2].isdigit():
        months = int(period[:-2]) + 12
        return f"{max(1, (months + 11) // 12)}y" if months >= 12 else f"{months}mo"
    return "3y"


def _regime_masks(spy):
    f = universe_scanner.features(spy)
    sma50, sma200 = f["SMA50"], f["SMA200"]
    valid = sma50.notna() & sma200.notna() & f["Close"].notna()
    true_bear = (f["Close"] < sma50) & (sma50 < sma200) & valid
    not_true_bear = (~((f["Close"] < sma50) & (sma50 < sma200))) & valid
    masks = {"true_bear": true_bear, "not_true_bear": not_true_bear}
    for m in masks.values():
        m.index = m.index.date
    return masks


def regime_signal_counts(tr, spy) -> dict:
    import pandas as pd
    n_all = int(len(tr)) if tr is not None else 0
    if tr is None or tr.empty or spy is None or spy.empty:
        return {"n_all": n_all, "n_bear": 0, "n_not_bear": 0, "n_unclassified": n_all, "unclassified": []}
    masks = _regime_masks(spy)
    bear, not_bear = masks["true_bear"], masks["not_true_bear"]
    n_bear = n_not = n_un = 0
    unclassified = []
    for ts in tr["ts"]:
        d = ts.date() if hasattr(ts, "date") else pd.Timestamp(ts).date()
        b = bear.get(d, None)
        nb = not_bear.get(d, None)
        if b is True or (b is not None and bool(b)):
            n_bear += 1
        elif nb is True or (nb is not None and bool(nb)):
            n_not += 1
        else:
            n_un += 1
            unclassified.append(str(d))
    return {"n_all": n_all, "n_bear": n_bear, "n_not_bear": n_not, "n_unclassified": n_un, "unclassified": unclassified[:50]}


def beats_baseline_margin(setup_r, base_r, setup_ec: float, base_ec: float, seed: int = 42, n_boot: int = 2000) -> dict:
    import numpy as np
    s = np.asarray(list(setup_r), dtype=float)
    b = np.asarray(list(base_r), dtype=float)
    gap = float(setup_ec) - float(base_ec)
    if len(s) == 0 or len(b) == 0:
        return {"beats_baseline": False, "p_setup_gt_base": 0.0, "E_cons_gap": gap, "n_boot": 0}
    rng = np.random.default_rng(seed)
    s_idx = rng.integers(0, len(s), size=(int(n_boot), len(s)))
    b_idx = rng.integers(0, len(b), size=(int(n_boot), len(b)))
    diffs = s[s_idx].mean(axis=1) - b[b_idx].mean(axis=1)
    p = float((diffs > 0).mean())
    return {"beats_baseline": bool(p >= 0.80 and gap >= 0.05), "p_setup_gt_base": p, "E_cons_gap": gap, "n_boot": int(n_boot)}


def _tagged_report(tr, masks, setup):
    import pandas as pd
    out = {}
    for name, mask in masks.items():
        if tr.empty:
            part = pd.DataFrame()
        else:
            tag = tr["ts"].map(lambda t: mask.get(t.date(), None))
            part = tr[tag == True]
        out[name] = outcome_lab.report_block(part, setup)
    return out


def _per_name_block(tickers, trades_by_ticker, label):
    import pandas as pd
    return {tk: outcome_lab.report_block(trades_by_ticker.get(tk, pd.DataFrame()), label) for tk in tickers}


def _top3_share(tr) -> dict:
    if tr is None or tr.empty or "ticker" not in tr:
        return {"top3_ticker_share_R": None, "top3_tickers": [], "total_abs_R": 0.0, "total_R_net": 0.0}
    by = tr.groupby("ticker")["R_net"].sum().sort_values(key=lambda s: s.abs(), ascending=False)
    total_abs = float(tr["R_net"].abs().sum())
    top3_abs = float(by.head(3).abs().sum()) if len(by) else 0.0
    return {
        "top3_ticker_share_R": (top3_abs / total_abs) if total_abs else None,
        "top3_tickers": [{"ticker": k, "R_net": float(v)} for k, v in by.head(3).items()],
        "total_abs_R": total_abs,
        "total_R_net": float(tr["R_net"].sum()),
    }


def refresh_v2(config: str | Path = ROOT / "config" / "universe_scanner.json", universe: str | None = None) -> dict:
    """Run all daily setups over the exact full universe, with RR=3 breakout variant and true-bear counts."""
    import pandas as pd
    cfg = universe_scanner.load_config(config)
    tickers = universe_scanner.resolve_universe(cfg, universe, None)
    evidence_period = cfg.get("evidence_period", "2y")
    interval = cfg.get("interval", "1d")
    frames = universe_scanner.fetch_bulk(list(tickers), evidence_period, interval)
    spy_frames = universe_scanner.fetch_bulk(["SPY"], _period_plus_250d(evidence_period), interval)
    if "SPY" in spy_frames:
        frames["SPY"] = spy_frames["SPY"]

    base_p = dict(stop_atr=float(cfg.get("stop_atr", 1.5)), rr=2.0, time_bars=20, spread_bps=3.0, slippage_bps=4.0, fees_bps=0.0)
    setup_names = [n for n in outcome_lab.SETUPS if n not in {"baseline_every5", "vwap_reclaim", "orb_60"}]
    variants = [(n, n, base_p) for n in setup_names] + [("breakout_20_rr3", "breakout_20", dict(base_p, rr=3.0))]
    base_fn, _ = outcome_lab.SETUPS["baseline_every5"]
    base_tr = []
    base_by_ticker = {}
    by_variant = {label: [] for label, _, _ in variants}
    by_variant_ticker = {label: {} for label, _, _ in variants}
    ticker_status = {}

    for tk in tickers:
        df = frames.get(tk)
        if df is None or df.empty:
            ticker_status[tk] = {"status": "missing_fetch_or_empty", "bars": 0}
            continue
        if len(df) < 80:
            ticker_status[tk] = {"status": "too_few_bars", "bars": int(len(df))}
            continue
        ticker_status[tk] = {"status": "ok", "bars": int(len(df)), "first": str(df.index[0]), "last": str(df.index[-1])}
        bt = outcome_lab.simulate(df, base_fn(df, False), base_p, False)
        base_by_ticker[tk] = bt
        if not bt.empty:
            btc = bt.copy(); btc["ticker"] = tk; base_tr.append(btc)
        for label, setup, p in variants:
            fn, _ = outcome_lab.SETUPS[setup]
            tr = outcome_lab.simulate(df, fn(df, False), p, False)
            by_variant_ticker[label][tk] = tr
            if not tr.empty:
                tc = tr.copy(); tc["ticker"] = tk; by_variant[label].append(tc)

    baseline_all = pd.concat(base_tr) if base_tr else pd.DataFrame()
    baseline = outcome_lab.report_block(baseline_all, "baseline_every5")
    masks = _regime_masks(frames["SPY"]) if "SPY" in frames and not frames["SPY"].empty else {}
    regime_bar_counts = {name: int(mask.sum()) for name, mask in masks.items()}
    rows = []
    for label, _, _ in variants:
        tr = pd.concat(by_variant[label]) if by_variant[label] else pd.DataFrame()
        block = outcome_lab.report_block(tr, label)
        margin = beats_baseline_margin(
            tr["R_net"] if not tr.empty else [],
            baseline_all["R_net"] if not baseline_all.empty else [],
            block.get("ec", 0), baseline.get("ec", 0),
        )
        row = {
            "setup": label,
            "n": block.get("n", 0),
            "E": block.get("e", 0),
            "E_cons": block.get("ec", 0),
            "baseline_n": baseline.get("n", 0),
            "baseline_E": baseline.get("e", 0),
            "baseline_E_cons": baseline.get("ec", 0),
            "beats_baseline_cons": bool(margin["beats_baseline"]),
            "bootstrap_margin": margin,
            "per_name": _per_name_block(tickers, by_variant_ticker[label], label),
            "baseline_per_name": _per_name_block(tickers, base_by_ticker, "baseline_every5"),
            "top3_ticker_share_R": None,
            "regime_counts": regime_signal_counts(tr, frames.get("SPY")) if "SPY" in frames else {"n_all": int(block.get("n", 0)), "n_bear": 0, "n_not_bear": 0, "n_unclassified": int(block.get("n", 0)), "unclassified": []},
            "regimes": _tagged_report(tr, masks, label) if masks else {},
        }
        row.update(_top3_share(tr))
        rows.append(row)

    live_regime = "not_true_bear"
    if frames.get("SPY") is not None and not frames["SPY"].empty:
        sf = universe_scanner.features(frames["SPY"])
        live_regime = "true_bear" if bool(((sf["Close"] < sf["SMA50"]) & (sf["SMA50"] < sf["SMA200"])).iloc[-1]) else "not_true_bear"
    live_samples = regime_bar_counts.get(live_regime, 0)
    evidence = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "version": "v2",
        "universe": universe or cfg.get("default_universe"),
        "ticker_list": list(tickers),
        "evidence_period": evidence_period,
        "interval": interval,
        "rows": rows,
        "baseline": baseline,
        "baseline_per_name": _per_name_block(tickers, base_by_ticker, "baseline_every5"),
        "regime_counts": regime_bar_counts,
        "data_quality": {"ticker_status": ticker_status, "ok_count": sum(1 for s in ticker_status.values() if s.get("status") == "ok"), "failed_count": sum(1 for s in ticker_status.values() if s.get("status") != "ok")},
        "governance": {"live_regime": live_regime, "live_regime_samples": live_samples, "live_regime_sample_cap": live_samples < 30, "rule": "if live regime has <30 evidence samples, cap at probation size"},
        "note": "True-bear regime is close<SMA50<SMA200. Backtest evidence enables/rejects only; forward ledger promotes tiers. Universe list is a model parameter and is printed/versioned with every evidence table.",
    }
    out = ROOT / "output" / "evidence_v2.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(evidence, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return evidence


def format_v2_table(ev: dict) -> str:
    lines = ["setup | n | E | E_cons | base_n | base_E | base_E_cons | P(setup>base) | gap | beats_base_margin"]
    for r in ev.get("rows", []):
        bm = r.get("bootstrap_margin", {})
        lines.append(
            f"{r['setup']} | {r.get('n',0)} | {float(r.get('E',0)):+.2f} | {float(r.get('E_cons',0)):+.2f} | "
            f"{r.get('baseline_n',0)} | {float(r.get('baseline_E',0)):+.2f} | {float(r.get('baseline_E_cons',0)):+.2f} | "
            f"{float(bm.get('p_setup_gt_base',0)):.2f} | {float(bm.get('E_cons_gap',0)):+.2f} | {r.get('beats_baseline_cons')}"
        )
    lines.append(f"ticker_list: {ev.get('ticker_list')}")
    lines.append(f"regime_bar_counts: {ev.get('regime_counts')}")
    lines.append(f"data_quality: {ev.get('data_quality')}")
    lines.append(f"governance: {ev.get('governance')}")
    return "\n".join(lines)


def format_evidence_line(ev: dict) -> str:
    pooled, base = ev.get("pooled", {}), ev.get("baseline", {})
    line = (
        f"EVIDENCE: point {float(pooled.get('e', 0)):+.2f} vs baseline {float(base.get('e', 0)):+.2f}; "
        f"cons {float(pooled.get('ec', 0)):+.2f} vs baseline {float(base.get('ec', 0)):+.2f}; "
        f"edge established: {'yes' if ev.get('edge_established') else 'no'}"
    )
    reg = ev.get("regime_split") or {}
    if reg:
        parts = []
        for name, r in reg.items():
            parts.append(f"{name} setup_ec {float(r.get('setup', {}).get('ec', 0)):+.2f} base_ec {float(r.get('baseline', {}).get('ec', 0)):+.2f}")
        line += " | regime: " + "; ".join(parts)
    return line


def refresh(config: str | Path = ROOT / "config" / "universe_scanner.json", universe: str | None = None, setup: str | None = None) -> dict:
    cfg = universe_scanner.load_config(config)
    tickers = universe_scanner.resolve_universe(cfg, universe, None)
    setup = setup or cfg.get("primary_setup", "breakout_20_rr3")
    setup_name = "breakout_20" if setup == "breakout_20_rr3" else setup
    rr = 3.0 if setup == "breakout_20_rr3" else float(cfg.get("rr", 2.0))
    frames = universe_scanner.fetch_bulk(tickers, cfg.get("evidence_period", cfg.get("period", "2y")), cfg.get("interval", "1d"))
    p = dict(stop_atr=float(cfg.get("stop_atr", 1.5)), rr=rr, time_bars=20, spread_bps=3.0, slippage_bps=4.0, fees_bps=0.0)
    setup_fn, _ = outcome_lab.SETUPS[setup_name]
    base_fn, _ = outcome_lab.SETUPS["baseline_every5"]
    setup_tr = []
    base_tr = []
    per_name = {}
    for tk in tickers:
        df = frames.get(tk)
        if df is None or df.empty:
            per_name[tk] = {"status": "missing_fetch_or_empty"}
            continue
        tr = outcome_lab.simulate(df, setup_fn(df, False), p, False)
        bt = outcome_lab.simulate(df, base_fn(df, False), p, False)
        per_name[tk] = {"setup": outcome_lab.report_block(tr, setup), "baseline": outcome_lab.report_block(bt, "baseline_every5")}
        if not tr.empty:
            tr = tr.copy(); tr["ticker"] = tk; setup_tr.append(tr)
        if not bt.empty:
            bt = bt.copy(); bt["ticker"] = tk; base_tr.append(bt)
    import pandas as pd
    setup_all = pd.concat(setup_tr) if setup_tr else pd.DataFrame()
    base_all = pd.concat(base_tr) if base_tr else pd.DataFrame()
    pooled = outcome_lab.report_block(setup_all, setup)
    baseline = outcome_lab.report_block(base_all, "baseline_every5")
    margin = beats_baseline_margin(setup_all["R_net"] if not setup_all.empty else [], base_all["R_net"] if not base_all.empty else [], pooled.get("ec", 0), baseline.get("ec", 0))
    evidence = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "universe": universe or cfg.get("default_universe"),
        "ticker_list": list(tickers),
        "setup": setup,
        "pooled": pooled,
        "baseline": baseline,
        "edge_established": bool(pooled.get("n", 0) >= 100 and margin["beats_baseline"]),
        "bootstrap_margin": margin,
        "per_name": per_name,
        "top3": _top3_share(setup_all),
        "note": "Backtest evidence enables/rejects only; forward ledger promotes tiers.",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(evidence, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return evidence


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config" / "universe_scanner.json"))
    ap.add_argument("--universe", default=None)
    ap.add_argument("--setup", default=None)
    ap.add_argument("--v2", action="store_true")
    args = ap.parse_args()
    ev = refresh_v2(args.config, args.universe) if args.v2 else refresh(args.config, args.universe, args.setup)
    print(json.dumps(ev, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
