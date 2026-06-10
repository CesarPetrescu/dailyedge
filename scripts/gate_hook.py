#!/usr/bin/env python3
"""
gate_hook.py — drop-in trade-gate stamp for dailyedge_watcher.py
=================================================================
Goal: every Telegram alert arrives pre-stamped with GO/NO-GO, R:R,
cost-in-R, the breakeven hit rate after costs, and a probation-size
share count — so the decision discipline travels with the alert.

Integration (3 lines in dailyedge_watcher.py):

    from gate_hook import stamp_line                       # 1. import

    # ... inside the function that builds the alert text, where you
    #     already know price (and ideally a level/ATR):
    msg += "\\n" + stamp_line(entry=price, atr=atr_value,  # 2. stamp
                              account=ACCOUNT, beta=BETA.get(ticker, 1.0))
    # 3. (optional) pass stop=, target= explicitly when the rule has
    #    structural levels from levels.py; otherwise auto-structure
    #    (stop = 1.5xATR, target = 2R) is used and labeled as such.

If edge_engine.py sits in the same folder its PARAMS are used, so gate
thresholds stay in one place. Otherwise safe fallbacks apply.
"""

from typing import Optional

try:                                   # single source of truth if colocated
    from edge_engine import PARAMS, cost_in_R
    _ENGINE = True
except Exception:
    _ENGINE = False
    PARAMS = {
        "spread_bps": 3.0, "slippage_bps": 4.0, "fees_bps": 0.0,
        "min_rr": 2.0, "min_rr_high_cost": 3.0,
        "cost_soft": 0.10, "cost_hard": 0.15,
        "min_stop_atr": 1.5, "min_stop_atr_highvol": 2.0, "highvol_cut": 80.0,
        "probation_risk": 0.0025,
    }

    def cost_in_R(stop_pct, spread_bps=None, slippage_bps=None, fees_bps=None):
        s = PARAMS["spread_bps"] if spread_bps is None else spread_bps
        sl = PARAMS["slippage_bps"] if slippage_bps is None else slippage_bps
        f = PARAMS["fees_bps"] if fees_bps is None else fees_bps
        rt = 2 * (s / 2 + sl + f) / 100.0
        return rt / stop_pct if stop_pct > 0 else float("inf")


def gate(entry: float,
         stop: Optional[float] = None,
         target: Optional[float] = None,
         atr: Optional[float] = None,
         ann_vol: Optional[float] = None,
         ext_pct: Optional[float] = None,
         rsi: Optional[float] = None) -> dict:
    """Returns a dict with go/no-go, reasons, and the key numbers.
    If stop/target are missing but ATR is given, auto-structure is applied
    (stop = 1.5xATR below entry, target = 2R) and flagged."""
    auto = False
    if stop is None and atr:
        stop = entry - 1.5 * atr
        auto = True
    if target is None and stop is not None:
        target = entry + 2.0 * (entry - stop)
        auto = True
    if stop is None:
        return {"go": False, "auto": False, "fails": ["no stop and no ATR"],
                "rr": 0.0, "cost_R": float("inf"), "p_be": 1.0}

    risk = abs(entry - stop)
    stop_pct = risk / entry * 100.0
    rr = abs(target - entry) / risk if risk > 0 else 0.0
    c = cost_in_R(stop_pct)
    p_be = (1 + c) / (1 + rr) if rr > 0 else 1.0

    fails = []
    rr_floor = PARAMS["min_rr_high_cost"] if c > PARAMS["cost_soft"] else PARAMS["min_rr"]
    if rr + 1e-9 < rr_floor:
        fails.append(f"R:R {rr:.2f}<{rr_floor:.1f}")
    if c > PARAMS["cost_hard"]:
        fails.append(f"cost {c:.2f}R>{PARAMS['cost_hard']:.2f}")
    if atr:
        floor = (PARAMS["min_stop_atr_highvol"]
                 if (ann_vol or 0) >= PARAMS["highvol_cut"]
                 else PARAMS["min_stop_atr"])
        if risk + 1e-9 < floor * atr:
            fails.append(f"stop {risk/atr:.1f}ATR<{floor:.1f}")
    if ext_pct is not None and rsi is not None and ext_pct > 25 and rsi > 75:
        fails.append(f"parabolic ({ext_pct:.0f}%>50d, RSI{rsi:.0f})")

    return {"go": not fails, "auto": auto, "fails": fails, "rr": rr,
            "cost_R": c, "p_be": p_be, "stop": stop, "target": target,
            "risk": risk, "stop_pct": stop_pct}


def stamp_line(entry: float,
               stop: Optional[float] = None,
               target: Optional[float] = None,
               atr: Optional[float] = None,
               ann_vol: Optional[float] = None,
               ext_pct: Optional[float] = None,
               rsi: Optional[float] = None,
               account: Optional[float] = None,
               beta: float = 1.0,
               prefix: Optional[str] = None) -> str:
    """One Telegram-ready line. Examples:
    ✅ GO[auto] R:R 2.0 | stop 191.20 (1.5ATR) | cost 0.05R | needs ≥35% | probation 1.3 sh
    ⛔ NO-GO: R:R 0.33<2.0; stop 1.4ATR<2.0; parabolic (60%>50d, RSI78)
    """
    g = gate(entry, stop, target, atr, ann_vol, ext_pct, rsi)
    if not g["go"]:
        return "⛔ NO-GO: " + "; ".join(g["fails"])
    tag = prefix if prefix is not None else ("✅ GO[auto-structure]" if g["auto"] else "✅ GO")
    parts = [f"{tag} R:R {g['rr']:.1f}",
             f"stop {g['stop']:.2f} ({g['stop_pct']:.1f}%)",
             f"tgt {g['target']:.2f}",
             f"cost {g['cost_R']:.2f}R",
             f"needs ≥{g['p_be']*100:.0f}% hit"]
    if account:
        beta_scalar = min(1.0, 1.5 / max(beta, 0.1))
        risk_frac = PARAMS["probation_risk"] * beta_scalar
        shares = account * risk_frac / g["risk"]
        parts.append(f"probation {shares:.1f} sh (${account*risk_frac:.0f} risk)")
    return " | ".join(parts)


def state_stamp(action_state: str, **kwargs) -> str:
    """State-aware gate stamp: armed alerts never display ✅ GO."""
    state = (action_state or "").upper()
    if state in {"SET_ALERT", "WATCH_STRONG"}:
        return stamp_line(prefix="🔔 IF TRIGGERED:", **kwargs)
    if state in {"GO", "GO_PROBATION"}:
        return stamp_line(prefix="✅ GO", **kwargs)
    if state in {"NO_GO", "SUPPRESSED_RISK", "EXPIRED"}:
        g = gate(kwargs.get("entry"), kwargs.get("stop"), kwargs.get("target"), kwargs.get("atr"), kwargs.get("ann_vol"), kwargs.get("ext_pct"), kwargs.get("rsi"))
        reasons = "; ".join(g.get("fails", [])) or state.replace("_", " ")
        return f"⛔ {state}: {reasons}"
    return stamp_line(**kwargs)


if __name__ == "__main__":          # smoke test
    print("engine import:", _ENGINE)
    print(stamp_line(entry=195.40, stop=191.20, target=204.00, atr=2.6,
                     account=10000, beta=1.7))
    print(stamp_line(entry=922, stop=838, target=950, atr=60, ann_vol=101,
                     ext_pct=60, rsi=78))
    print(stamp_line(entry=520.0, atr=14.0, account=10000, beta=2.0))
