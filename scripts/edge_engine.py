#!/usr/bin/env python3
"""Shared edge/risk parameters for Ultra Daytrader tactics.

This module is intentionally small: it gives gate_hook.py, outcome_lab.py,
watchers, and future scanners one source of truth for cost and gate thresholds.
It is not an order-execution engine and does not provide financial advice.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


PARAMS = {
    # Round-trip cost assumptions. spread_bps is quoted full spread; half-spread
    # is paid on entry and exit. slippage_bps/fees_bps are per side.
    "spread_bps": 3.0,
    "slippage_bps": 4.0,
    "fees_bps": 0.0,
    # Gate thresholds.
    "min_rr": 2.0,
    "min_rr_high_cost": 3.0,
    "cost_soft": 0.10,
    "cost_hard": 0.15,
    "min_stop_atr": 1.5,
    "min_stop_atr_highvol": 2.0,
    "highvol_cut": 80.0,
    # Paper/probation sizing default: 0.25% account risk, beta-adjusted.
    "probation_risk": 0.0025,
}


def round_trip_cost_pct(
    spread_bps: float | None = None,
    slippage_bps: float | None = None,
    fees_bps: float | None = None,
) -> float:
    """Return estimated round-trip transaction cost as percent of entry price."""
    s = PARAMS["spread_bps"] if spread_bps is None else spread_bps
    sl = PARAMS["slippage_bps"] if slippage_bps is None else slippage_bps
    f = PARAMS["fees_bps"] if fees_bps is None else fees_bps
    per_side_bps = s / 2 + sl + f
    return 2 * per_side_bps / 100.0


def cost_in_R(
    stop_pct: float,
    spread_bps: float | None = None,
    slippage_bps: float | None = None,
    fees_bps: float | None = None,
) -> float:
    """Convert round-trip transaction cost into R units for a stop width.

    stop_pct must be a percentage, e.g. a 2.5% stop is ``2.5``.
    """
    if stop_pct <= 0:
        return float("inf")
    return round_trip_cost_pct(spread_bps, slippage_bps, fees_bps) / stop_pct


def breakeven_hit_rate(rr: float, cost_r: float = 0.0) -> float:
    """Minimum target-hit rate needed to break even after cost, in [0, 1]."""
    if rr <= 0:
        return 1.0
    return max(0.0, min(1.0, (1.0 + cost_r) / (1.0 + rr)))


def beta_scaled_risk_fraction(beta: float = 1.0) -> float:
    """Probation risk fraction adjusted lower for high-beta names."""
    beta_scalar = min(1.0, 1.5 / max(beta, 0.1))
    return PARAMS["probation_risk"] * beta_scalar


@dataclass(frozen=True)
class EdgeStats:
    n: int
    expectancy_r: float
    conservative_expectancy_r: float | None = None
    baseline_expectancy_r: float | None = None

    @property
    def beats_baseline(self) -> bool | None:
        if self.baseline_expectancy_r is None:
            return None
        return self.expectancy_r > self.baseline_expectancy_r


def confidence_bucket(stats: EdgeStats, min_n: int = 100) -> str:
    """Simple testable confidence label for forward ledger/reporting.

    This avoids vibes-scoring: HIGH requires enough samples, positive conservative
    expectancy, and beating same-regime baseline when a baseline is supplied.
    """
    if stats.n < min_n:
        return "LOW"
    if stats.conservative_expectancy_r is not None and stats.conservative_expectancy_r <= 0:
        return "LOW"
    if stats.beats_baseline is False:
        return "LOW"
    if stats.n >= min_n * 2 and (stats.conservative_expectancy_r or stats.expectancy_r) > 0:
        return "HIGH"
    return "MED"


if __name__ == "__main__":
    for stop_pct in (1.0, 2.0, 4.0):
        c = cost_in_R(stop_pct)
        print(f"stop={stop_pct:.1f}% cost={c:.3f}R p_be@2R={breakeven_hit_rate(2.0, c)*100:.1f}%")
