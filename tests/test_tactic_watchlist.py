from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import tactic_watchlist


def synthetic_df(n=90, breakout_last=True):
    idx = pd.date_range("2026-01-01", periods=n, freq="D")
    close = pd.Series(np.linspace(100, 120, n), index=idx)
    if breakout_last:
        close.iloc[-1] = close.iloc[-21:-1].max() + 5
    df = pd.DataFrame(index=idx)
    df["Close"] = close
    df["Open"] = close.shift(1).fillna(close.iloc[0])
    df["High"] = df[["Open", "Close"]].max(axis=1) + 1
    df["Low"] = df[["Open", "Close"]].min(axis=1) - 1
    df["Volume"] = 1_000_000
    return df


def test_detect_setup_state_breakout_now():
    df = synthetic_df()
    state = tactic_watchlist.detect_setup_state(df, "breakout_20", max_age=5)
    assert state["signal_now"] is True
    assert state["signal_recent"] is True
    assert state["bars_since_signal"] == 0
    assert state["trigger"] is not None


def test_make_gate_returns_stamp():
    df = synthetic_df()
    cfg = {"stop_atr": 1.5, "rr": 2.0, "beta": {"NVDA": 1.7}, "account": None}
    g = tactic_watchlist.make_gate(df, cfg, "NVDA")
    assert g["go"] is True
    assert g["rr"] == 2.0
    assert "GO" in g["stamp"]
    assert g["stop"] < g["entry"] < g["target"]


def test_format_text_basic_report():
    report = {
        "config": {"primary_setup": "breakout_20", "tickers": ["NVDA"]},
        "regime": {"regime": "risk_on", "weighted_score": 5.0},
        "evidence": {
            "pooled": {"n": 120, "e": 0.2, "ec": 0.05},
            "baseline": {"e": 0.1},
            "confidence_bucket": "MED",
        },
        "rows": [{
            "ticker": "NVDA",
            "price": 100.0,
            "action": "GO",
            "reasons": ["test"],
            "setup_state": {"signal_now": True, "signal_recent": True, "bars_since_signal": 0, "trigger": 99.0},
            "trade_gate": {"stamp": "✅ GO R:R 2.0"},
        }],
    }
    text = tactic_watchlist.format_text(report)
    assert "NVDA: GO" in text
    assert "breakout_20" in text
