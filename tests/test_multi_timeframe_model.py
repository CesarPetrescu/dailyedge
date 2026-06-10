import math
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from multi_timeframe_model import (  # noqa: E402
    BacktestConfig,
    backtest_signal,
    build_features,
    combine_timeframe_scores,
    detect_setup,
    score_timeframe,
    train_test_split_backtest,
    walk_forward_backtest,
)


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


def test_score_timeframe_identifies_bullish_trend():
    df = build_features(make_ohlcv([100 + i for i in range(240)]))
    score = score_timeframe(df, "1d")

    assert score["label"] == "bullish"
    assert score["score"] > 0
    assert score["rsi14"] is not None
    assert "price_above_ema20" in score["reasons"]


def test_combine_timeframe_scores_weights_higher_timeframes():
    combined = combine_timeframe_scores(
        {
            "1mo": {"score": 2, "label": "bullish"},
            "1d": {"score": 3, "label": "bullish"},
            "1h": {"score": 1, "label": "bullish"},
            "15m": {"score": -1, "label": "bearish"},
        }
    )

    assert combined["total_score"] > 0
    assert combined["bias"] == "bullish"
    assert combined["confidence"] in {"medium", "high"}


def test_detect_setup_finds_bullish_pullback():
    setup = detect_setup(
        {
            "1mo": {"score": 2, "label": "bullish"},
            "1d": {"score": 3, "label": "bullish"},
            "1h": {"score": 0, "label": "neutral"},
            "15m": {"score": -1, "label": "bearish"},
        }
    )

    assert setup["name"] == "bullish_pullback"
    assert setup["direction"] == "long"


def test_backtest_signal_reports_sample_and_probability_stats():
    # Steady upward data should produce at least some long signals and sane stats.
    closes = [100 + i * 0.4 + math.sin(i / 3) for i in range(260)]
    df = build_features(make_ohlcv(closes))

    stats = backtest_signal(df, direction="long", config=BacktestConfig(horizon=5, min_samples=1))

    assert stats["samples"] > 0
    assert 0 <= stats["win_rate_pct"] <= 100
    assert stats["avg_return_pct"] is not None
    assert stats["profit_factor"] is None or stats["profit_factor"] >= 0


def test_train_test_split_backtest_excludes_holdout_from_in_sample_stats():
    closes = [100 + i * 0.2 + math.sin(i / 5) for i in range(300)]
    df = build_features(make_ohlcv(closes))

    split = train_test_split_backtest(
        df,
        direction="long",
        config=BacktestConfig(horizon=5, min_samples=1),
        test_fraction=0.25,
    )

    assert split["method"] == "chronological_train_test_split"
    assert split["train"]["end_index"] < split["test"]["start_index"]
    assert split["train"]["samples"] > 0
    assert split["test"]["samples"] > 0
    assert split["combined_for_display_only"]["samples"] == split["train"]["samples"] + split["test"]["samples"]


def test_train_test_split_backtest_does_not_use_future_holdout_when_training_stats_change():
    base = [100 + i * 0.2 + math.sin(i / 5) for i in range(260)]
    hostile_future = [base[-1] - i * 2.0 for i in range(1, 80)]
    original = build_features(make_ohlcv(base))
    extended = build_features(make_ohlcv(base + hostile_future))

    original_split = train_test_split_backtest(
        original,
        direction="long",
        config=BacktestConfig(horizon=5, min_samples=1),
        test_fraction=0.25,
        train_end_index=180,
    )
    extended_split = train_test_split_backtest(
        extended,
        direction="long",
        config=BacktestConfig(horizon=5, min_samples=1),
        test_fraction=0.25,
        train_end_index=180,
    )

    assert original_split["train"]["samples"] == extended_split["train"]["samples"]
    assert original_split["train"]["win_rate_pct"] == extended_split["train"]["win_rate_pct"]
    assert original_split["train"]["avg_return_pct"] == extended_split["train"]["avg_return_pct"]


def test_walk_forward_backtest_uses_embargo_between_train_and_test_folds():
    closes = [100 + i * 0.15 + math.sin(i / 4) for i in range(420)]
    df = build_features(make_ohlcv(closes))

    wf = walk_forward_backtest(
        df,
        direction="long",
        config=BacktestConfig(horizon=5, min_samples=1),
        train_size=160,
        test_size=40,
        step_size=40,
        embargo_bars=7,
    )

    assert wf["method"] == "walk_forward_embargoed"
    assert wf["fold_count"] >= 3
    for fold in wf["folds"]:
        assert fold["train_end_index"] + 7 < fold["test_start_index"]
        assert fold["test"]["window"] == "out_of_sample_test"
    assert 0 <= wf["summary"]["oos_win_rate_mean_pct"] <= 100
    assert wf["summary"]["stability_score"] in {"low", "medium", "high"}


def test_walk_forward_backtest_flags_weak_sample_confidence():
    closes = [100 + i * 0.1 for i in range(130)]
    df = build_features(make_ohlcv(closes))

    wf = walk_forward_backtest(
        df,
        direction="long",
        config=BacktestConfig(horizon=5, min_samples=50),
        train_size=90,
        test_size=20,
        step_size=20,
        embargo_bars=5,
    )

    assert wf["summary"]["confidence_adjustment"] == "low_sample_low_confidence"
    assert wf["summary"]["total_oos_samples"] < 50
