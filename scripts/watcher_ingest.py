#!/usr/bin/env python3
"""Ingest scanner watcher_candidates.json into armed watcher rules."""
from __future__ import annotations
import argparse, json, sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import gate_hook
import universe_scanner

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SRC = ROOT / "output" / "watcher_candidates.json"
DEFAULT_DST = ROOT / "config" / "armed_watchers.json"

ARM_ACTIONS = {"SET_ALERT", "GO_PROBATION", "GO"}


def ingest(src: str | Path = DEFAULT_SRC, dst: str | Path = DEFAULT_DST) -> dict:
    src, dst = Path(src), Path(dst)
    data = json.loads(src.read_text(encoding="utf-8")) if src.exists() else {"candidates": []}
    primary_setup = data.get("primary_setup") or data.get("rule") or "breakout_20_rr3"
    rule_name = "breakout_20_rr3" if primary_setup == "breakout_20_rr3" else "breakout_20_trigger"
    armed = []
    for c in data.get("candidates", []):
        if c.get("action") not in ARM_ACTIONS:
            continue
        trig = c.get("breakout_trigger")
        if trig is None:
            continue
        armed.append({
            "ticker": c["ticker"],
            "rule": rule_name,
            "setup": primary_setup,
            "rr": 3.0 if primary_setup == "breakout_20_rr3" else 2.0,
            "action": c.get("action"),
            "trigger": trig,
            "op": "cross_above",
            "entry_policy": "next_open_after_trigger_or_confirmed_break",
            "projection": True,
            "projection_note": "armed stop/target are projections; re-stamp at fire with signal ATR and next open before ledger append",
            "stop_projection": c.get("gate", {}).get("stop"),
            "target_projection": c.get("gate", {}).get("target"),
            "stop": c.get("gate", {}).get("stop"),
            "target": c.get("gate", {}).get("target"),
            "sector": c.get("sector"),
            "source_score": c.get("score"),
            "armed_at": datetime.now(timezone.utc).isoformat(),
        })
    out = {"generated_at": datetime.now(timezone.utc).isoformat(), "source": str(src), "primary_setup": primary_setup, "armed_rules": armed}
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    return out


def restamp_fired_rule(rule: dict[str, Any], df: pd.DataFrame, rr: float = 2.0, stop_atr: float = 1.5, account: float | None = None) -> dict[str, Any] | None:
    """Rebuild exact trade structure when an armed trigger fires.

    Signal bar = close crosses above trigger. Entry = next bar open. Stop and
    target use ATR from the signal bar, not the old armed projection.
    """
    if df is None or df.empty or len(df) < 3:
        return None
    trigger = float(rule["trigger"])
    f = universe_scanner.features(df)
    crosses = (f["Close"] > trigger) & (f["Close"].shift() <= trigger)
    if not crosses.any():
        return None
    sig_pos = f.index.get_loc(f.index[crosses][-1])
    if sig_pos + 1 >= len(f):
        return None
    sig = f.iloc[sig_pos]
    entry = float(f["Open"].iloc[sig_pos + 1])
    atr = float(sig["ATR14"])
    ann_vol = float(sig["ANN_VOL60"]) if pd.notna(sig.get("ANN_VOL60")) else None
    stop_mult = 2.0 if ann_vol is not None and ann_vol >= 80 else stop_atr
    stop = entry - stop_mult * atr
    target = entry + rr * (entry - stop)
    g = gate_hook.gate(entry=entry, stop=stop, target=target, atr=atr, ann_vol=ann_vol)
    stamp = gate_hook.state_stamp("GO_PROBATION" if g.get("go") else "NO_GO", entry=entry, stop=stop, target=target, atr=atr, ann_vol=ann_vol, account=account)
    return {
        "ticker": rule.get("ticker"), "rule": rule.get("rule"), "trigger": trigger,
        "signal_ts": str(f.index[sig_pos]), "entry_ts": str(f.index[sig_pos + 1]),
        "entry": entry, "atr_signal": atr, "ann_vol_pct": ann_vol,
        "stop": stop, "target": target, "gate": g, "gate_stamp": stamp,
        "projection_replaced": True,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(DEFAULT_SRC))
    ap.add_argument("--dst", default=str(DEFAULT_DST))
    args = ap.parse_args()
    print(json.dumps(ingest(args.src, args.dst), indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
