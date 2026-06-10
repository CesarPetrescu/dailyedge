from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import outcome_lab


def synthetic_breakout_df(n=140):
    idx = pd.date_range("2026-01-01", periods=n, freq="D")
    base = np.linspace(100, 120, n)
    close = pd.Series(base, index=idx)
    # Create repeated prior-20-high breakouts.
    close.iloc[25] = close.iloc[:25].max() + 4
    close.iloc[60] = close.iloc[:60].max() + 4
    close.iloc[95] = close.iloc[:95].max() + 4
    df = pd.DataFrame(index=idx)
    df["Close"] = close
    df["Open"] = close.shift(1).fillna(close.iloc[0])
    df["High"] = df[["Open", "Close"]].max(axis=1) + 1.5
    df["Low"] = df[["Open", "Close"]].min(axis=1) - 1.5
    df["Volume"] = 1_000_000
    return df


def test_outcome_lab_breakout_generates_trades():
    df = synthetic_breakout_df()
    sig = outcome_lab.s_breakout_20(df, intraday=False)
    p = dict(outcome_lab.DEFAULTS)
    trades = outcome_lab.simulate(df, sig, p, intraday=False)
    assert not trades.empty
    assert {"entry", "stop", "target", "R_net", "cost_R", "MAE_R", "MFE_R"}.issubset(trades.columns)
    report = outcome_lab.report_block(trades, "breakout_20")
    assert report["n"] == len(trades)
    assert "ec" in report


def test_wilson_bounds_are_ordered():
    p, lo, hi = outcome_lab.wilson(50, 100)
    assert lo < p < hi
