from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import evidence_cache
import full_ticker_report
import gate_hook
import ledger
import universe_scanner
import watcher_ingest


def test_set_alert_stamp_never_says_go():
    stamp = gate_hook.state_stamp("SET_ALERT", entry=100, stop=95, target=110, atr=2)
    assert stamp.startswith("🔔 IF TRIGGERED:")
    assert "✅ GO" not in stamp
    assert "cost" in stamp and "needs" in stamp


def test_full_report_uses_scanner_gate_and_regime_single_source(tmp_path, monkeypatch):
    idx = pd.date_range("2026-01-01", periods=180, freq="D")
    close = pd.Series(np.linspace(100, 150, 180), index=idx)
    df = pd.DataFrame({"Open": close, "High": close + 2, "Low": close - 2, "Close": close, "Volume": 1_000_000}, index=idx)
    row = {
        "ticker": "TEST", "action": "SET_ALERT", "breakout_trigger": 151.0,
        "gate": {"stop": 145.0, "target": 163.0, "stamp": "🔔 IF TRIGGERED: R:R 2.0 | stop 145.00 (4.0%) | tgt 163.00 | cost 0.01R | needs ≥34% hit"},
        "rsi14": 60, "ann_vol_pct": 55, "rel_volume": 1.2, "rs20_vs_spy_pct": 5, "rs63_vs_spy_pct": 8, "ext_pct_vs_sma50": 10, "days_to_earnings": 50,
    }
    monkeypatch.setattr(full_ticker_report, "fetch_df", lambda ticker, period="6mo", interval="1d": df)
    monkeypatch.setattr(full_ticker_report, "OUTPUT_DIR", tmp_path)
    report = full_ticker_report.build_full_report("TEST", scanner_row=row, scanner_regime={"regime": "risk_on", "weighted_score": 7.4})
    assert report["stop"] == 145.0
    assert report["target"] == 163.0
    assert report["regime"]["weighted_score"] == 7.4
    text = full_ticker_report.format_report(report)
    assert "ext50d 10" in text and "annVol 55" in text and "RVOL 1.2" in text and "RS20 5" in text
    assert "0/2 cap" in text


def test_evidence_summary_prints_same_unit_comparisons_and_regime_split():
    ev = {
        "pooled": {"e": 0.2, "ec": 0.05, "n": 100},
        "baseline": {"e": 0.4, "ec": 0.1, "n": 100},
        "regime_split": {"SPY up": {"setup": {"ec": 0.1}, "baseline": {"ec": 0.2}}, "SPY down": {"setup": {"ec": -0.3}, "baseline": {"ec": 0.0}}},
    }
    s = evidence_cache.format_evidence_line(ev)
    assert "point +0.20 vs baseline +0.40" in s
    assert "cons +0.05 vs baseline +0.10" in s
    assert "SPY down" in s


def test_img_files_are_nonempty(tmp_path):
    idx = pd.date_range("2026-01-01", periods=180, freq="D")
    close = pd.Series(np.linspace(100, 150, 180), index=idx)
    df = pd.DataFrame({"Open": close, "High": close + 2, "Low": close - 2, "Close": close, "Volume": 1_000_000}, index=idx)
    paths = full_ticker_report.render_required_images("TEST", df, "SET_ALERT", 151, 151, 145, 163, output_dir=tmp_path, render_mc_placeholder=True)
    assert {"daily_setup", "rs_panel", "mc_fan"}.issubset(paths)
    for name in ["daily_setup", "rs_panel", "mc_fan"]:
        assert Path(paths[name]).stat().st_size > 10_000


def test_watcher_ingest_arms_rr3_candidates_with_projection_label(tmp_path):
    src = tmp_path / "watcher_candidates.json"
    dst = tmp_path / "armed_watchers.json"
    src.write_text(json.dumps({"primary_setup": "breakout_20_rr3", "candidates": [
        {"ticker": "KLAC", "action": "SET_ALERT", "breakout_trigger": 2156.69, "gate": {"stop": 1982.87, "target": 2678.15}},
        {"ticker": "ON", "action": "EXPIRED", "breakout_trigger": 134.92, "gate": {"stop": 1, "target": 2}},
    ]}))
    out = watcher_ingest.ingest(src, dst)
    assert len(out["armed_rules"]) == 1
    r = out["armed_rules"][0]
    assert r["ticker"] == "KLAC"
    assert r["rule"] == "breakout_20_rr3"
    assert r["projection"] is True
    assert r["trigger"] == 2156.69
    assert r["target_projection"] == 2678.15


def test_rs_panel_top3_nonempty(tmp_path):
    idx = pd.date_range("2026-01-01", periods=180, freq="D")
    base = pd.Series(np.linspace(100, 150, 180), index=idx)
    frames = {k: pd.DataFrame({"Close": base * mult}) for k, mult in {"A":1.0,"B":1.1,"C":0.9,"SPY":0.8,"SMH":1.2}.items()}
    path = full_ticker_report.render_rs_panel(["A", "B", "C"], frames, output_dir=tmp_path)
    assert Path(path).stat().st_size > 10_000


def test_evidence_v2_records_exact_ticker_list_per_name_top3_and_full_regime_conservation(monkeypatch):
    idx = pd.date_range("2025-01-01", periods=320, freq="D")
    close = pd.Series(np.linspace(100, 180, 320) + np.sin(np.arange(320) / 5) * 5, index=idx)
    df = pd.DataFrame({"Open": close.shift(1).fillna(close.iloc[0]), "High": close + 3, "Low": close - 3, "Close": close, "Volume": 1_000_000}, index=idx)
    spy = df.copy()
    frames = {"AAA": df, "BBB": df * 1.01, "SPY": spy}
    monkeypatch.setattr(universe_scanner, "fetch_bulk", lambda tickers, period, interval: {k: v for k, v in frames.items() if k in tickers})
    monkeypatch.setattr(universe_scanner, "resolve_universe", lambda cfg, universe, tickers: ["AAA", "BBB"])
    ev = evidence_cache.refresh_v2(universe="test")
    assert ev["ticker_list"] == ["AAA", "BBB"]
    assert ev["data_quality"]["ticker_status"]["AAA"]["status"] == "ok"
    row = next(r for r in ev["rows"] if r["setup"] == "breakout_20_rr3")
    assert "per_name" in row and "AAA" in row["per_name"]
    assert "top3_ticker_share_R" in row
    rc = row["regime_counts"]
    assert rc["n_bear"] + rc["n_not_bear"] + rc["n_unclassified"] == rc["n_all"] == row["n"]
    assert "bootstrap_margin" in row


def test_scanner_documents_unscored_names(monkeypatch):
    cfg = {"period": "1y", "interval": "1d", "default_universe": "x", "universes": {"x": ["AAA", "BBB"]}, "candidate_limit": 25}
    idx = pd.date_range("2026-01-01", periods=120, freq="D")
    close = pd.Series(np.linspace(100, 120, 120), index=idx)
    good = pd.DataFrame({"Open": close, "High": close + 1, "Low": close - 1, "Close": close, "Volume": 1_000_000}, index=idx)
    monkeypatch.setattr(universe_scanner, "fetch_bulk", lambda tickers, period, interval: {"AAA": good, "SPY": good, "QQQ": good, "SMH": good})
    monkeypatch.setattr(universe_scanner.market_regime, "score_market_regime", lambda frames, sector_etf="SMH": {"regime": "risk_on"})
    report = universe_scanner.scan(cfg)
    assert report["input_count"] == 2
    assert report["scored_count"] == 1
    assert report["unscored"] == [{"ticker": "BBB", "reason": "missing_fetch_or_empty"}]
