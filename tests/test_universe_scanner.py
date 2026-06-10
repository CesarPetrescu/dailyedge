from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import universe_scanner


def synthetic_df(n=120, breakout=False):
    idx = pd.date_range("2026-01-01", periods=n, freq="D")
    close = pd.Series(np.linspace(100, 130, n), index=idx)
    if breakout:
        close.iloc[-1] = close.iloc[-21:-1].max() + 4
    df = pd.DataFrame(index=idx)
    df["Close"] = close
    df["Open"] = close.shift(1).fillna(close.iloc[0])
    df["High"] = df[["Open", "Close"]].max(axis=1) + 1
    df["Low"] = df[["Open", "Close"]].min(axis=1) - 1
    df["Volume"] = 2_000_000
    return df


def test_resolve_config_universe():
    cfg = {"universes": {"x": ["brk.b", "NVDA", "nvda"]}, "default_universe": "x"}
    assert universe_scanner.resolve_universe(cfg, None, None) == ["BRK-B", "NVDA"]


def test_score_ticker_breakout_candidate():
    df = synthetic_df(breakout=True)
    spy = synthetic_df()
    frames = {"TEST": df, "SPY": spy}
    rs = universe_scanner.compute_relative_strength(frames, "SPY")
    cfg = {
        "min_avg_dollar_vol": 1,
        "stop_atr": 1.5,
        "rr": 2.0,
        "max_rsi_no_chase": 90,
        "max_ext_pct_vs_sma50": 100,
    }
    row = universe_scanner.score_ticker("TEST", df, cfg, rs, {"regime": "risk_on"})
    assert row is not None
    assert row["breakout_now"] is True
    assert row["action"] in {"GO_PROBATION", "WATCH_STRONG"}
    assert row["gate"]["go"] is True


def test_format_text_contains_rows():
    report = {
        "universe": "x",
        "scored_count": 1,
        "input_count": 1,
        "regime": {"regime": "risk_on", "weighted_score": 5},
        "rows": [{
            "ticker": "TEST",
            "action": "SET_ALERT",
            "score": 50,
            "price": 100,
            "breakout_trigger": 101,
            "dist_to_breakout_pct": -1,
            "rsi14": 60,
            "rel_volume": 1.2,
            "rs20_vs_spy_pct": 3,
            "gate": {"stop": 95, "target": 110},
            "reasons": ["near_20d_breakout"],
            "penalties": [],
        }],
    }
    text = universe_scanner.format_text(report)
    assert "TEST SET_ALERT" in text
    assert "near_20d_breakout" in text
