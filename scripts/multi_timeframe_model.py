#!/usr/bin/env python3
"""Multi-timeframe signal model + backtest for Ultra Daytrader.

Free-data, paper-trading research helper. It does not place orders and does not
produce financial advice. The output is a historical setup analysis, not a
forecast guarantee.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output"

TIMEFRAMES: dict[str, tuple[str, str]] = {
    "15m": ("60d", "15m"),
    "1h": ("730d", "1h"),
    "1d": ("3y", "1d"),
    "1mo": ("10y", "1mo"),
}
WEIGHTS = {"1mo": 2.0, "1d": 1.75, "1h": 1.25, "15m": 1.0}


@dataclass
class BacktestConfig:
    horizon: int = 5
    stop_atr: float = 1.5
    target_atr: float = 2.5
    min_samples: int = 20


def safe_float(value: Any, digits: int = 4):
    try:
        if value is None or pd.isna(value):
            return None
        x = float(value)
        if not math.isfinite(x):
            return None
        return round(x, digits)
    except Exception:
        return None


def ema(close: pd.Series, span: int) -> pd.Series:
    return close.ewm(span=span, adjust=False).mean()


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    # A window with gains and no losses is maximum RSI, not missing data.
    out = out.mask((loss == 0) & (gain > 0), 100.0)
    out = out.mask((gain == 0) & (loss > 0), 0.0)
    return out


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window).mean()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return OHLCV frame with common indicators."""
    if df.empty:
        raise ValueError("empty OHLCV data")
    out = df.copy()
    out = out.dropna(subset=["Close"])
    close = out["Close"]
    out["EMA9"] = ema(close, 9)
    out["EMA20"] = ema(close, 20)
    out["SMA50"] = close.rolling(50).mean()
    out["SMA200"] = close.rolling(200).mean()
    out["RSI14"] = rsi(close)
    ema12 = ema(close, 12)
    ema26 = ema(close, 26)
    out["MACD"] = ema12 - ema26
    out["MACD_SIGNAL"] = ema(out["MACD"], 9)
    out["ATR14"] = atr(out)
    out["VOL_AVG20"] = out["Volume"].rolling(20).mean() if "Volume" in out else np.nan
    out["RET_FWD_5"] = out["Close"].shift(-5) / out["Close"] - 1
    return out


def score_timeframe(df: pd.DataFrame, timeframe: str) -> dict[str, Any]:
    """Score latest candle in one timeframe, range roughly -5..+5."""
    featured = build_features(df) if "EMA20" not in df.columns else df
    latest = featured.iloc[-1]
    price = float(latest["Close"])
    score = 0
    reasons: list[str] = []

    ema20 = latest.get("EMA20")
    sma50 = latest.get("SMA50")
    sma200 = latest.get("SMA200")
    rsi14 = latest.get("RSI14")
    macd = latest.get("MACD")
    macd_signal = latest.get("MACD_SIGNAL")
    vol = latest.get("Volume")
    vol_avg = latest.get("VOL_AVG20")

    if pd.notna(ema20):
        if price > float(ema20):
            score += 1
            reasons.append("price_above_ema20")
        else:
            score -= 1
            reasons.append("price_below_ema20")
    if pd.notna(sma50):
        if price > float(sma50):
            score += 1
            reasons.append("price_above_sma50")
        else:
            score -= 1
            reasons.append("price_below_sma50")
    if pd.notna(sma200):
        if price > float(sma200):
            score += 1
            reasons.append("price_above_sma200")
        else:
            score -= 1
            reasons.append("price_below_sma200")
    if pd.notna(macd) and pd.notna(macd_signal):
        if float(macd) > float(macd_signal):
            score += 1
            reasons.append("macd_above_signal")
        else:
            score -= 1
            reasons.append("macd_below_signal")
    if pd.notna(rsi14):
        r = float(rsi14)
        if 45 <= r <= 70:
            score += 1
            reasons.append("rsi_constructive")
        elif r > 75:
            score -= 1
            reasons.append("rsi_overextended")
        elif r < 35:
            score -= 1
            reasons.append("rsi_weak_or_oversold")
    if pd.notna(vol) and pd.notna(vol_avg) and float(vol_avg) > 0:
        if float(vol) > float(vol_avg) * 1.2 and score > 0:
            score += 1
            reasons.append("bullish_volume_confirmation")
        elif float(vol) > float(vol_avg) * 1.2 and score < 0:
            score -= 1
            reasons.append("bearish_volume_confirmation")

    label = "neutral"
    if score >= 2:
        label = "bullish"
    elif score <= -2:
        label = "bearish"

    return {
        "timeframe": timeframe,
        "score": int(score),
        "label": label,
        "last_date": str(featured.index[-1]),
        "price": safe_float(price),
        "ema20": safe_float(ema20),
        "sma50": safe_float(sma50),
        "sma200": safe_float(sma200),
        "rsi14": safe_float(rsi14, 2),
        "macd": safe_float(macd),
        "atr14": safe_float(latest.get("ATR14")),
        "volume_vs_avg_pct": safe_float((float(vol) / float(vol_avg) - 1) * 100, 2) if pd.notna(vol) and pd.notna(vol_avg) and float(vol_avg) else None,
        "reasons": reasons,
    }


def combine_timeframe_scores(scores: dict[str, dict[str, Any]]) -> dict[str, Any]:
    weighted = 0.0
    max_abs = 0.0
    for tf, info in scores.items():
        weight = WEIGHTS.get(tf, 1.0)
        weighted += float(info.get("score", 0)) * weight
        max_abs += 5.0 * weight
    normalized = 0.0 if max_abs == 0 else weighted / max_abs
    if normalized >= 0.28:
        bias = "bullish"
    elif normalized <= -0.28:
        bias = "bearish"
    else:
        bias = "neutral"
    abs_norm = abs(normalized)
    confidence = "low"
    if abs_norm >= 0.55:
        confidence = "high"
    elif abs_norm >= 0.28:
        confidence = "medium"
    return {
        "total_score": round(weighted, 2),
        "normalized_score": round(normalized, 4),
        "bias": bias,
        "confidence": confidence,
    }


def detect_setup(scores: dict[str, dict[str, Any]]) -> dict[str, Any]:
    monthly = scores.get("1mo", {}).get("score", 0)
    daily = scores.get("1d", {}).get("score", 0)
    hourly = scores.get("1h", {}).get("score", 0)
    m15 = scores.get("15m", {}).get("score", 0)
    if monthly >= 1 and daily >= 2 and m15 <= 0:
        return {"name": "bullish_pullback", "direction": "long", "description": "Higher timeframes bullish while 15m is pulling back; look for confirmation rather than chasing."}
    if daily >= 2 and hourly >= 1 and m15 >= 1:
        return {"name": "bullish_momentum", "direction": "long", "description": "Daily, hourly, and 15m momentum align bullish."}
    if monthly <= -1 and daily <= -2 and m15 >= 0:
        return {"name": "bearish_bounce", "direction": "short", "description": "Higher timeframes bearish while 15m bounces; possible fade setup."}
    if daily <= -2 and hourly <= -1 and m15 <= -1:
        return {"name": "bearish_momentum", "direction": "short", "description": "Daily, hourly, and 15m momentum align bearish."}
    return {"name": "mixed_or_chop", "direction": "neutral", "description": "Timeframes are not aligned enough for a clean setup."}


def _signal_mask(df: pd.DataFrame, direction: str) -> pd.Series:
    if direction == "long":
        return (df["Close"] > df["EMA20"]) & (df["EMA20"] > df["SMA50"]) & (df["RSI14"] >= 45)
    if direction == "short":
        return (df["Close"] < df["EMA20"]) & (df["EMA20"] < df["SMA50"]) & (df["RSI14"].between(28, 55))
    return pd.Series(False, index=df.index)


def _collect_signal_returns(
    featured: pd.DataFrame,
    direction: str,
    config: BacktestConfig,
    start_index: int = 50,
    end_index: int | None = None,
) -> list[tuple[float, float, float]]:
    """Collect returns using only signals in [start_index, end_index].

    A signal at index i may only use future candles i+1..i+horizon for outcome.
    Caller controls the chronological train/test split so current or holdout bars
    cannot contaminate in-sample stats.
    """
    mask = _signal_mask(featured, direction).fillna(False)
    rows: list[tuple[float, float, float]] = []
    max_i = len(featured) - config.horizon - 1
    last_i = min(max_i, len(featured) - 1 if end_index is None else end_index)
    first_i = max(50, start_index)
    for i in range(first_i, last_i + 1):
        if not bool(mask.iloc[i]):
            continue
        entry = float(featured["Close"].iloc[i])
        future = featured.iloc[i + 1 : i + 1 + config.horizon]
        if len(future) < config.horizon:
            continue
        if direction == "long":
            exit_price = float(future["Close"].iloc[-1])
            ret = exit_price / entry - 1
            mfe = float(future["High"].max() / entry - 1)
            mae = float(future["Low"].min() / entry - 1)
        else:
            exit_price = float(future["Close"].iloc[-1])
            ret = entry / exit_price - 1
            mfe = float(entry / future["Low"].min() - 1)
            mae = float(entry / future["High"].max() - 1)
        rows.append((ret, mfe, mae))
    return rows


def _summarize_return_rows(rows: list[tuple[float, float, float]], config: BacktestConfig) -> dict[str, Any]:
    if not rows:
        return {"samples": 0, "valid": False, "reason": "no historical matching signals"}
    arr = np.array(rows, dtype=float)
    returns = arr[:, 0]
    wins = returns[returns > 0]
    losses = returns[returns <= 0]
    gross_win = float(wins.sum()) if len(wins) else 0.0
    gross_loss = abs(float(losses.sum())) if len(losses) else 0.0
    equity = np.cumprod(1 + returns)
    peaks = np.maximum.accumulate(equity)
    drawdown = equity / peaks - 1
    valid = len(returns) >= config.min_samples
    return {
        "samples": int(len(returns)),
        "valid": bool(valid),
        "reason": None if valid else f"sample size below min_samples={config.min_samples}",
        "horizon_bars": config.horizon,
        "win_rate_pct": round(float(np.mean(returns > 0) * 100), 2),
        "avg_return_pct": round(float(np.mean(returns) * 100), 3),
        "median_return_pct": round(float(np.median(returns) * 100), 3),
        "best_return_pct": round(float(np.max(returns) * 100), 3),
        "worst_return_pct": round(float(np.min(returns) * 100), 3),
        "avg_mfe_pct": round(float(np.mean(arr[:, 1]) * 100), 3),
        "avg_mae_pct": round(float(np.mean(arr[:, 2]) * 100), 3),
        "profit_factor": None if gross_loss == 0 else round(gross_win / gross_loss, 3),
        "max_drawdown_pct": round(float(np.min(drawdown) * 100), 3),
        "equity_curve": [round(float(x), 4) for x in equity[-120:]],
    }


def backtest_signal(df: pd.DataFrame, direction: str, config: BacktestConfig = BacktestConfig()) -> dict[str, Any]:
    featured = build_features(df) if "EMA20" not in df.columns else df.copy()
    if direction not in {"long", "short"}:
        return {"samples": 0, "valid": False, "reason": "neutral setup has no directional backtest"}
    rows = _collect_signal_returns(featured, direction, config)
    return _summarize_return_rows(rows, config)


def train_test_split_backtest(
    df: pd.DataFrame,
    direction: str,
    config: BacktestConfig = BacktestConfig(),
    test_fraction: float = 0.25,
    train_end_index: int | None = None,
) -> dict[str, Any]:
    """Chronological split that prevents look-ahead / train-test leakage.

    The rule set is fixed/explainable rather than fitted, but we still separate
    in-sample and out-of-sample windows. Current/future holdout bars are never
    included in `train` stats. Combined stats are marked display-only.
    """
    featured = build_features(df) if "EMA20" not in df.columns else df.copy()
    if direction not in {"long", "short"}:
        neutral = {"samples": 0, "valid": False, "reason": "neutral setup has no directional backtest"}
        return {
            "method": "chronological_train_test_split",
            "leakage_guard": "neutral setup; no fitted model and no directional split used",
            "train": neutral,
            "test": neutral,
            "combined_for_display_only": neutral,
        }
    n = len(featured)
    if n < 80:
        raise ValueError("not enough bars for chronological train/test split")
    if train_end_index is None:
        train_end_index = int(n * (1 - test_fraction)) - config.horizon - 1
    train_end_index = max(50, min(int(train_end_index), n - config.horizon - 2))
    test_start_index = train_end_index + config.horizon + 1
    test_end_index = n - config.horizon - 1

    train_rows = _collect_signal_returns(featured, direction, config, 50, train_end_index)
    test_rows = _collect_signal_returns(featured, direction, config, test_start_index, test_end_index)
    combined_rows = train_rows + test_rows
    train = _summarize_return_rows(train_rows, config)
    test = _summarize_return_rows(test_rows, config)
    combined = _summarize_return_rows(combined_rows, config)
    train.update({"start_index": 50, "end_index": train_end_index, "window": "in_sample_train"})
    test.update({"start_index": test_start_index, "end_index": test_end_index, "window": "out_of_sample_test"})
    combined.update({"window": "combined_display_only", "warning": "Do not use combined stats as trained performance; use out_of_sample_test for validation."})
    return {
        "method": "chronological_train_test_split",
        "leakage_guard": "Signals are generated only from past/current indicators; outcomes use later bars; holdout bars are excluded from train stats.",
        "train_fraction_approx": round((train_end_index + 1) / n, 3),
        "test_fraction_approx": round(max(0, test_end_index - test_start_index + 1) / n, 3),
        "train": train,
        "test": test,
        "combined_for_display_only": combined,
    }


def walk_forward_backtest(
    df: pd.DataFrame,
    direction: str,
    config: BacktestConfig = BacktestConfig(),
    train_size: int = 252,
    test_size: int = 63,
    step_size: int | None = None,
    embargo_bars: int | None = None,
) -> dict[str, Any]:
    """Embargoed walk-forward validation for directional signals.

    Each fold trains/summarizes older signals, skips an embargo gap, then tests
    the next chronological window. Probability quality should be judged from OOS
    folds, not in-sample/combined stats.
    """
    featured = build_features(df) if "EMA20" not in df.columns else df.copy()
    if direction not in {"long", "short"}:
        return {
            "method": "walk_forward_embargoed",
            "fold_count": 0,
            "folds": [],
            "summary": {"confidence_adjustment": "neutral_setup_no_directional_oos"},
        }
    n = len(featured)
    step_size = test_size if step_size is None else step_size
    embargo_bars = config.horizon if embargo_bars is None else embargo_bars
    folds: list[dict[str, Any]] = []
    start = 50
    while True:
        train_start = start
        train_end = train_start + train_size - 1
        test_start = train_end + embargo_bars + 1
        test_end = test_start + test_size - 1
        if test_start > n - config.horizon - 1:
            break
        test_end = min(test_end, n - config.horizon - 1)
        if train_end >= test_start or test_end < test_start:
            break
        train_rows = _collect_signal_returns(featured, direction, config, train_start, train_end)
        test_rows = _collect_signal_returns(featured, direction, config, test_start, test_end)
        train = _summarize_return_rows(train_rows, config)
        test = _summarize_return_rows(test_rows, config)
        train.update({"window": "in_sample_train"})
        test.update({"window": "out_of_sample_test"})
        folds.append({
            "fold": len(folds) + 1,
            "train_start_index": train_start,
            "train_end_index": train_end,
            "test_start_index": test_start,
            "test_end_index": test_end,
            "embargo_bars": embargo_bars,
            "train": {k: v for k, v in train.items() if k != "equity_curve"},
            "test": {k: v for k, v in test.items() if k != "equity_curve"},
        })
        start += step_size
        if start + train_size + embargo_bars >= n - config.horizon:
            break

    valid_tests = [f["test"] for f in folds if f["test"].get("samples", 0) > 0]
    total_oos_samples = int(sum(t.get("samples", 0) for t in valid_tests))
    win_rates = [float(t["win_rate_pct"]) for t in valid_tests if t.get("win_rate_pct") is not None]
    avg_returns = [float(t["avg_return_pct"]) for t in valid_tests if t.get("avg_return_pct") is not None]
    profit_factors = [float(t["profit_factor"]) for t in valid_tests if t.get("profit_factor") is not None]
    wr_mean = float(np.mean(win_rates)) if win_rates else None
    wr_std = float(np.std(win_rates, ddof=0)) if len(win_rates) > 1 else 0.0 if win_rates else None
    ar_mean = float(np.mean(avg_returns)) if avg_returns else None
    pf_mean = float(np.mean(profit_factors)) if profit_factors else None
    positive_folds = int(sum(1 for x in avg_returns if x > 0))

    stability_score = "low"
    if total_oos_samples >= config.min_samples and len(valid_tests) >= 3 and ar_mean is not None and ar_mean > 0:
        stability_score = "medium"
        if wr_std is not None and wr_std <= 12 and positive_folds >= max(1, int(len(valid_tests) * 0.65)):
            stability_score = "high"
    confidence_adjustment = "ok"
    if total_oos_samples < config.min_samples:
        confidence_adjustment = "low_sample_low_confidence"
    elif stability_score == "low":
        confidence_adjustment = "unstable_folds_low_confidence"

    return {
        "method": "walk_forward_embargoed",
        "leakage_guard": "Each fold tests only bars after train window plus embargo gap; OOS fold stats drive forecast framing.",
        "fold_count": len(folds),
        "train_size": train_size,
        "test_size": test_size,
        "step_size": step_size,
        "embargo_bars": embargo_bars,
        "folds": folds,
        "summary": {
            "total_oos_samples": total_oos_samples,
            "oos_win_rate_mean_pct": None if wr_mean is None else round(wr_mean, 2),
            "oos_win_rate_std_pct": None if wr_std is None else round(wr_std, 2),
            "oos_avg_return_mean_pct": None if ar_mean is None else round(ar_mean, 3),
            "oos_profit_factor_mean": None if pf_mean is None else round(pf_mean, 3),
            "positive_oos_folds": positive_folds,
            "valid_oos_folds": len(valid_tests),
            "stability_score": stability_score,
            "confidence_adjustment": confidence_adjustment,
        },
    }


def fetch_timeframes(ticker: str, overrides: dict[str, tuple[str, str]] | None = None) -> dict[str, pd.DataFrame]:
    import yfinance as yf

    out = {}
    for tf, (period, interval) in (overrides or TIMEFRAMES).items():
        df = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=False)
        if df.empty:
            raise RuntimeError(f"no OHLCV data returned for {ticker} {tf} ({period}/{interval})")
        out[tf] = df.dropna(subset=["Close"])
    return out


def make_model_chart(
    ticker: str,
    daily: pd.DataFrame,
    scores: dict[str, dict[str, Any]],
    backtest: dict[str, Any],
    outdir: Path,
    walk_forward: dict[str, Any] | None = None,
    market_regime: dict[str, Any] | None = None,
    prediction: dict[str, Any] | None = None,
) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    png = outdir / f"{ticker}_{stamp}_mtf_model.png"
    daily = build_features(daily) if "EMA20" not in daily.columns else daily

    fig = plt.figure(figsize=(15, 12), constrained_layout=True)
    gs = fig.add_gridspec(4, 2, height_ratios=[2.2, 0.9, 1.1, 0.9], width_ratios=[1.2, 0.8])
    ax1 = fig.add_subplot(gs[0, :])
    tail = daily.tail(220)
    ax1.plot(tail.index, tail["Close"], label="Close", linewidth=1.6)
    ax1.plot(tail.index, tail["EMA20"], label="EMA20", linewidth=1.0)
    ax1.plot(tail.index, tail["SMA50"], label="SMA50", linewidth=1.0)
    if tail["SMA200"].notna().any():
        ax1.plot(tail.index, tail["SMA200"], label="SMA200", linewidth=1.0)
    ax1.set_title(f"{ticker} multi-timeframe dashboard")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="upper left")

    ax2 = fig.add_subplot(gs[1, 0])
    order = [tf for tf in ["1mo", "1d", "1h", "15m"] if tf in scores]
    values = [scores[tf]["score"] for tf in order]
    colors = ["#2ca02c" if v > 0 else "#d62728" if v < 0 else "#7f7f7f" for v in values]
    ax2.bar(order, values, color=colors, alpha=0.75)
    ax2.axhline(0, color="black", linewidth=1)
    ax2.set_ylabel("Raw score")
    ax2.set_title("Timeframe alignment")
    ax2.grid(True, axis="y", alpha=0.25)

    ax3 = fig.add_subplot(gs[1, 1])
    wf = walk_forward or {}
    folds = wf.get("folds") or []
    win_rates = [f.get("test", {}).get("win_rate_pct") for f in folds if f.get("test", {}).get("win_rate_pct") is not None]
    if win_rates:
        ax3.bar(range(1, len(win_rates) + 1), win_rates, color="#1f77b4", alpha=0.75)
        ax3.axhline(50, color="black", linestyle="--", linewidth=1)
        ax3.set_ylim(0, 100)
        ax3.set_ylabel("OOS win %")
    else:
        ax3.text(0.05, 0.5, "No walk-forward folds", transform=ax3.transAxes)
    ax3.set_title("Walk-forward OOS folds")
    ax3.grid(True, axis="y", alpha=0.25)

    ax4 = fig.add_subplot(gs[2, :])
    equity = backtest.get("equity_curve") or []
    if equity:
        ax4.plot(range(len(equity)), equity, color="#1f77b4", label="Holdout/validation equity")
        ax4.legend(loc="upper left")
    else:
        ax4.text(0.02, 0.5, "No directional backtest/equity curve", transform=ax4.transAxes)
    ax4.set_title("Leak-safe validation equity")
    ax4.grid(True, alpha=0.25)

    ax5 = fig.add_subplot(gs[3, :])
    ax5.axis("off")
    wf_summary = (walk_forward or {}).get("summary", {})
    regime = market_regime or {}
    pred = prediction or {}
    box = (
        f"Regime: {regime.get('regime', 'n/a')} | regime score: {regime.get('weighted_score', 'n/a')} | "
        f"Prediction source: {pred.get('source', 'n/a')} | bull%: {pred.get('bull_probability_pct', 'n/a')} | "
        f"OOS samples: {wf_summary.get('total_oos_samples', 'n/a')} | stability: {wf_summary.get('stability_score', 'n/a')} | "
        f"confidence: {pred.get('confidence_adjustment', 'n/a')} | leak-safe: walk-forward + embargo"
    )
    ax5.text(0.01, 0.65, box, transform=ax5.transAxes, fontsize=10, va="center", bbox={"boxstyle": "round", "facecolor": "#f2f2f2", "alpha": 0.9})

    fig.savefig(png, dpi=150)
    plt.close(fig)
    return str(png)


def run_model(
    ticker: str,
    outdir: str,
    horizon: int,
    min_samples: int,
    validation: str = "walk-forward",
    train_size: int = 252,
    test_size: int = 63,
    step_size: int | None = None,
    embargo_bars: int | None = None,
) -> dict[str, Any]:
    ticker = ticker.upper().strip()
    frames = fetch_timeframes(ticker)
    features = {tf: build_features(df) for tf, df in frames.items()}
    scores = {tf: score_timeframe(df, tf) for tf, df in features.items()}
    combined_raw = combine_timeframe_scores(scores)
    market_regime = {"ok": False, "error": "regime unavailable"}
    try:
        from market_regime import apply_regime_modifier, fetch_regime_frames, score_market_regime
        regime_frames, sector = fetch_regime_frames(ticker)
        market_regime = score_market_regime(regime_frames, sector_etf=sector)
        combined = apply_regime_modifier(combined_raw, market_regime)
    except Exception as exc:
        combined = dict(combined_raw)
        combined["warnings"] = list(combined.get("warnings", [])) + ["market_regime_unavailable"]
        market_regime = {"ok": False, "error": str(exc), "regime": "unknown", "confidence_modifier": 0}
    setup = detect_setup(scores)
    config = BacktestConfig(horizon=horizon, min_samples=min_samples)
    backtest_split = train_test_split_backtest(features["1d"], setup["direction"], config)
    walk_forward = walk_forward_backtest(
        features["1d"],
        setup["direction"],
        config,
        train_size=train_size,
        test_size=test_size,
        step_size=step_size,
        embargo_bars=embargo_bars,
    )
    validation_stats = backtest_split.get("test", {})

    wf_summary = walk_forward.get("summary", {})
    bull_prob = None
    if validation == "walk-forward" and wf_summary.get("oos_win_rate_mean_pct") is not None:
        if setup["direction"] == "long":
            bull_prob = wf_summary.get("oos_win_rate_mean_pct")
        elif setup["direction"] == "short":
            bull_prob = round(100 - float(wf_summary.get("oos_win_rate_mean_pct", 50)), 2)
    elif validation_stats.get("samples", 0) > 0 and setup["direction"] == "long":
        bull_prob = validation_stats.get("win_rate_pct")
    elif validation_stats.get("samples", 0) > 0 and setup["direction"] == "short":
        bull_prob = round(100 - float(validation_stats.get("win_rate_pct", 50)), 2)
    prediction = {
        "bull_probability_pct": bull_prob,
        "bear_probability_pct": None if bull_prob is None else round(100 - float(bull_prob), 2),
        "source": "embargoed_walk_forward_oos" if validation == "walk-forward" else "chronological_holdout_oos",
        "confidence_adjustment": wf_summary.get("confidence_adjustment") if validation == "walk-forward" else ("low_sample_low_confidence" if validation_stats.get("samples", 0) < min_samples else "ok"),
        "framing": "Historical out-of-sample setup statistic, not a forecast guarantee.",
    }
    chart_stats = validation_stats if validation_stats.get("equity_curve") else backtest_split.get("combined_for_display_only", {})
    chart_png = make_model_chart(ticker, features["1d"], scores, chart_stats, Path(outdir), walk_forward=walk_forward, market_regime=market_regime, prediction=prediction)

    def strip_equity(obj: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in obj.items() if k != "equity_curve"}

    leak_safe_backtest = {
        "method": backtest_split.get("method"),
        "leakage_guard": backtest_split.get("leakage_guard"),
        "train_fraction_approx": backtest_split.get("train_fraction_approx"),
        "test_fraction_approx": backtest_split.get("test_fraction_approx"),
        "train_in_sample": strip_equity(backtest_split.get("train", {})),
        "test_out_of_sample": strip_equity(backtest_split.get("test", {})),
        "combined_for_display_only": strip_equity(backtest_split.get("combined_for_display_only", {})),
    }

    result = {
        "ok": True,
        "ticker": ticker,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "chart_png": chart_png,
        "timeframes": scores,
        "combined": combined,
        "market_regime": market_regime,
        "setup": setup,
        "backtest": leak_safe_backtest,
        "walk_forward": walk_forward,
        "prediction": prediction,
        "levels": {
            "last_price": scores["1d"].get("price"),
            "atr14_daily": scores["1d"].get("atr14"),
            "analytical_stop_long_1_5atr": safe_float(scores["1d"].get("price") - 1.5 * scores["1d"].get("atr14"), 2) if scores["1d"].get("price") and scores["1d"].get("atr14") else None,
            "analytical_target_long_2_5atr": safe_float(scores["1d"].get("price") + 2.5 * scores["1d"].get("atr14"), 2) if scores["1d"].get("price") and scores["1d"].get("atr14") else None,
        },
        "disclaimer": "Research/paper-trading model using free historical data; not financial advice.",
    }
    js = Path(outdir) / f"{ticker}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_mtf_model.json"
    js.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    result["summary_json"] = str(js)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ticker")
    ap.add_argument("--outdir", default=str(OUTPUT_DIR))
    ap.add_argument("--horizon", type=int, default=5, help="Backtest holding horizon in daily bars")
    ap.add_argument("--min-samples", type=int, default=20)
    ap.add_argument("--validation", choices=["walk-forward", "holdout"], default="walk-forward")
    ap.add_argument("--train-size", type=int, default=252, help="Walk-forward train window in daily bars")
    ap.add_argument("--test-size", type=int, default=63, help="Walk-forward OOS test window in daily bars")
    ap.add_argument("--step-size", type=int, default=None, help="Walk-forward step in bars; defaults to test-size")
    ap.add_argument("--embargo-bars", type=int, default=None, help="Gap between train and test windows; defaults to horizon")
    args = ap.parse_args()
    try:
        result = run_model(
            args.ticker,
            args.outdir,
            args.horizon,
            args.min_samples,
            validation=args.validation,
            train_size=args.train_size,
            test_size=args.test_size,
            step_size=args.step_size,
            embargo_bars=args.embargo_bars,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2, sort_keys=True))
        raise SystemExit(2)


if __name__ == "__main__":
    main()
