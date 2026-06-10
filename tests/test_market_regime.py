from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from market_regime import score_market_regime, sector_etf_for_ticker, apply_regime_modifier  # noqa: E402


def make_ohlcv(closes):
    idx = pd.date_range("2024-01-01", periods=len(closes), freq="D")
    close = pd.Series(closes, index=idx, dtype="float64")
    return pd.DataFrame(
        {
            "Open": close.shift(1).fillna(close.iloc[0]),
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": 1_000_000,
        },
        index=idx,
    )


def test_regime_scores_risk_on_when_spy_and_qqq_trend_up():
    spy = make_ohlcv([100 + i for i in range(240)])
    qqq = make_ohlcv([200 + i for i in range(240)])

    result = score_market_regime({"SPY": spy, "QQQ": qqq})

    assert result["regime"] == "risk_on"
    assert result["confidence_modifier"] > 0
    assert result["indices"]["SPY"]["label"] == "bullish"


def test_regime_scores_risk_off_when_indices_break_down():
    spy = make_ohlcv([300 - i for i in range(240)])
    qqq = make_ohlcv([400 - i for i in range(240)])

    result = score_market_regime({"SPY": spy, "QQQ": qqq})

    assert result["regime"] == "risk_off"
    assert result["confidence_modifier"] < 0
    assert result["indices"]["QQQ"]["label"] == "bearish"


def test_sector_etf_confirmation_is_optional():
    spy = make_ohlcv([100 + i for i in range(240)])
    qqq = make_ohlcv([200 + i for i in range(240)])

    result = score_market_regime({"SPY": spy, "QQQ": qqq}, sector_etf=None)

    assert result["sector"] is None
    assert result["regime"] in {"risk_on", "neutral_chop", "risk_off"}


def test_sector_etf_mapping_covers_semis():
    assert sector_etf_for_ticker("NVDA") == "SMH"
    assert sector_etf_for_ticker("AMD") == "SMH"


def test_regime_filter_reduces_bullish_confidence_in_risk_off_market():
    adjusted = apply_regime_modifier(
        combined={"bias": "bullish", "confidence": "high"},
        regime={"regime": "risk_off", "confidence_modifier": -1},
    )

    assert adjusted["confidence"] == "medium"
    assert "market_regime_conflict" in adjusted["warnings"]
