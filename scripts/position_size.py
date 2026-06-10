#!/usr/bin/env python3
"""Position sizing helper for analytical/paper-trade planning."""
from __future__ import annotations

import argparse, json, math

ap = argparse.ArgumentParser()
ap.add_argument('--equity', type=float, required=True)
ap.add_argument('--risk-pct', type=float, default=0.5)
ap.add_argument('--entry', type=float, required=True)
ap.add_argument('--stop', type=float, required=True)
ap.add_argument('--target', type=float)
args = ap.parse_args()

risk_dollars = args.equity * (args.risk_pct / 100)
per_share_risk = abs(args.entry - args.stop)
shares = math.floor(risk_dollars / per_share_risk) if per_share_risk > 0 else 0
rr = None
if args.target is not None and per_share_risk > 0:
    rr = abs(args.target - args.entry) / per_share_risk
print(json.dumps({
    'equity': args.equity,
    'risk_pct': args.risk_pct,
    'risk_dollars': round(risk_dollars, 2),
    'entry': args.entry,
    'stop': args.stop,
    'target': args.target,
    'per_share_risk': round(per_share_risk, 4),
    'shares_for_max_risk': shares,
    'notional_at_entry': round(shares * args.entry, 2),
    'risk_reward': round(rr, 2) if rr is not None else None,
    'note': 'Analytical/paper-trade sizing only; not financial advice.'
}, indent=2))
