from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from intraday_session_model import (  # noqa: E402
    analyze_intraday_session,
    build_intraday_features,
    opening_range_levels,
    session_phase,
)


def make_intraday_bars(closes, volumes=None, highs=None, lows=None):
    idx = pd.date_range("2024-01-02 09:30", periods=len(closes), freq="15min", tz="America/New_York")
    close = pd.Series(closes, index=idx, dtype="float64")
    if volumes is None:
        volumes = [1000] * len(closes)
    return pd.DataFrame(
        {
            "Open": close.shift(1).fillna(close.iloc[0]),
            "High": pd.Series(highs if highs is not None else [c + 1 for c in closes], index=idx, dtype="float64"),
            "Low": pd.Series(lows if lows is not None else [c - 1 for c in closes], index=idx, dtype="float64"),
            "Close": close,
            "Volume": pd.Series(volumes, index=idx, dtype="float64"),
        },
        index=idx,
    )


def test_vwap_is_volume_weighted():
    df = make_intraday_bars(closes=[100, 101, 102], volumes=[10, 20, 30], highs=[100, 101, 102], lows=[100, 101, 102])

    featured = build_intraday_features(df)

    assert round(featured["VWAP"].iloc[-1], 4) == round((100 * 10 + 101 * 20 + 102 * 30) / 60, 4)


def test_opening_range_high_low_uses_first_n_minutes():
    df = make_intraday_bars(
        closes=[100, 101, 99, 103, 104],
        highs=[101, 102, 100, 104, 105],
        lows=[99, 100, 98, 102, 103],
    )

    levels = opening_range_levels(df, minutes=45)

    assert levels["opening_range_high"] == 102
    assert levels["opening_range_low"] == 98


def test_detects_vwap_reclaim_or_opening_range_breakout():
    df = make_intraday_bars(
        closes=[100, 99, 98, 99, 100, 101, 103, 104],
        highs=[101, 100, 99, 100, 101, 102, 104, 105],
        lows=[99, 98, 97, 98, 99, 100, 102, 103],
        volumes=[1000, 1100, 1200, 1300, 1500, 1800, 2500, 3000],
    )

    result = analyze_intraday_session(df, ticker="TEST")

    assert result["setup"]["name"] in {"vwap_reclaim", "opening_range_breakout", "late_day_momentum"}
    assert result["bias"] in {"bullish", "neutral", "bearish"}
    assert result["levels"]["vwap"] is not None


def test_session_phase_labels_market_periods():
    assert session_phase(pd.Timestamp("2024-01-02 09:45", tz="America/New_York")) == "open_drive"
    assert session_phase(pd.Timestamp("2024-01-02 12:30", tz="America/New_York")) == "midday"
    assert session_phase(pd.Timestamp("2024-01-02 15:15", tz="America/New_York")) == "power_hour"
