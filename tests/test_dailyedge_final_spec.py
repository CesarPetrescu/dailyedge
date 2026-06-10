from pathlib import Path
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


def passing_gate():
    return {"go": True, "fails": [], "stamp": "✅ GO R:R 2.0 | cost 0.02R | needs >=34% hit"}


def test_t1_signal_4_bars_old_far_below_trigger_is_expired_never_go():
    action, reasons = universe_scanner.decide_action_state(
        signal_age=4,
        dist_to_trigger_pct=-13.0,
        gate=passing_gate(),
        regime={"regime": "risk_on"},
        risk_state=ledger.empty_risk_state(),
        days_to_earnings=30,
    )
    assert action == "EXPIRED"
    assert "signal 4 bars ago" in "; ".join(reasons)


def test_t2_price_below_trigger_without_fresh_signal_sets_alert_never_go():
    action, _ = universe_scanner.decide_action_state(
        signal_age=None,
        dist_to_trigger_pct=-0.8,
        gate=passing_gate(),
        regime={"regime": "risk_on"},
        risk_state=ledger.empty_risk_state(),
        days_to_earnings=30,
    )
    assert action == "SET_ALERT"


def test_t3_high_ann_vol_uses_ann_vol60_stop_floor():
    g = gate_hook.gate(entry=100, stop=98.5, target=103, atr=1, ann_vol=85)
    assert g["go"] is False
    assert any("stop 1.5ATR<2.0" in r for r in g["fails"])


def test_t4_backtest_large_but_ledger_empty_is_incubation():
    assert ledger.rule_tier(backtest_n=500, ledger_n=0, ledger_e_cons=1.0) == "INCUBATION"


def test_t5_earnings_within_2_sessions_is_no_go_event_window():
    action, reasons = universe_scanner.decide_action_state(
        signal_age=0,
        dist_to_trigger_pct=1.0,
        gate=passing_gate(),
        regime={"regime": "risk_on"},
        risk_state=ledger.empty_risk_state(),
        days_to_earnings=1,
    )
    assert action == "NO_GO"
    assert any("event_window" in r for r in reasons)


def test_t6_day_loss_suppresses_go_lines():
    risk = ledger.empty_risk_state()
    risk["day_R"] = -2.1
    action, reasons = universe_scanner.decide_action_state(
        signal_age=0,
        dist_to_trigger_pct=1.0,
        gate=passing_gate(),
        regime={"regime": "risk_on"},
        risk_state=risk,
        days_to_earnings=30,
    )
    assert action == "SUPPRESSED_RISK"
    assert any("day R" in r for r in reasons)


def test_t7_three_same_cluster_warns_and_caps():
    rows = [
        {"ticker": "KLAC", "action": "GO_PROBATION"},
        {"ticker": "ASML", "action": "SET_ALERT"},
        {"ticker": "AMAT", "action": "SET_ALERT"},
    ]
    warning = ledger.concentration_warning(rows)
    assert "AI-semis" in warning
    assert "3 candidates" in warning


def test_t8_every_go_stamp_contains_cost_and_needs():
    stamp = gate_hook.stamp_line(entry=100, stop=95, target=110, atr=2, account=10000)
    assert "cost" in stamp
    assert "needs ≥" in stamp or "needs >=" in stamp


def test_t9_real_full_report_run_produces_three_pngs_over_10kb(tmp_path, monkeypatch):
    idx = pd.date_range("2026-01-01", periods=180, freq="D")
    close = pd.Series(np.linspace(100, 180, len(idx)) + np.sin(np.arange(len(idx))) * 2, index=idx)
    df = pd.DataFrame({
        "Open": close.shift(1).fillna(close.iloc[0]),
        "High": close + 3,
        "Low": close - 3,
        "Close": close,
        "Volume": 1_000_000,
    }, index=idx)
    monkeypatch.setattr(full_ticker_report, "fetch_df", lambda ticker, period="6mo", interval="1d": df)
    monkeypatch.setattr(full_ticker_report, "OUTPUT_DIR", tmp_path)
    report = full_ticker_report.build_full_report(
        "TEST",
        scanner_row={
            "ticker": "TEST", "action": "SET_ALERT", "breakout_trigger": 181.0,
            "gate": {"stop": 169.0, "target": 205.0, "stamp": "🔔 IF TRIGGERED: R:R 3.0 | stop 169.00 | tgt 205.00 | cost 0.01R | needs ≥26% hit"},
        },
        scanner_regime={"regime": "risk_on", "weighted_score": 7.0},
    )
    paths = report.get("images", {})
    assert {"daily_setup", "rs_panel", "mc_fan"}.issubset(paths)
    for name in ["daily_setup", "rs_panel", "mc_fan"]:
        p = Path(paths[name])
        assert p.exists(), name
        assert p.stat().st_size > 10_000, (name, p.stat().st_size)


def test_restamp_fire_uses_signal_atr_and_next_open():
    idx = pd.date_range("2026-01-01", periods=25, freq="D")
    close = pd.Series([100.0] * 21 + [101.0, 106.0, 107.0, 108.0], index=idx)
    high = close + 1
    low = close - 1
    open_ = close.copy()
    open_.iloc[23] = 107.25
    df = pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": 1_000_000}, index=idx)
    fired = watcher_ingest.restamp_fired_rule({"ticker": "T", "trigger": 105.0, "rule": "breakout_20_rr3"}, df, rr=3.0, stop_atr=1.5)
    atr_signal = universe_scanner.features(df)["ATR14"].iloc[22]
    assert fired is not None
    assert fired["entry"] == open_.iloc[23]
    assert abs(fired["stop"] - (open_.iloc[23] - 1.5 * atr_signal)) < 1e-9
    assert abs(fired["target"] - (open_.iloc[23] + 3.0 * 1.5 * atr_signal)) < 1e-9


def test_ann_vol_is_60d_close_to_close_everywhere():
    idx = pd.date_range("2026-01-01", periods=90, freq="D")
    close = pd.Series(np.linspace(100, 140, 90), index=idx)
    df = pd.DataFrame({"Open": close, "High": close + 2, "Low": close - 2, "Close": close, "Volume": 1_000_000}, index=idx)
    f = universe_scanner.features(df)
    expected = close.pct_change().rolling(60).std().iloc[-1] * (252 ** 0.5) * 100
    assert abs(f["ANN_VOL60"].iloc[-1] - expected) < 1e-12


def test_kill_criterion_retires_negative_forward_edge():
    assert ledger.rule_tier(backtest_n=500, ledger_n=30, ledger_e_cons=-0.01) == "RETIRED"


def test_regime_conservation_counts_every_signal_date():
    idx = pd.date_range("2025-01-01", periods=260, freq="D")
    spy = pd.DataFrame({
        "Open": np.linspace(100, 150, 260),
        "High": np.linspace(101, 151, 260),
        "Low": np.linspace(99, 149, 260),
        "Close": np.linspace(100, 150, 260),
        "Volume": 1_000_000,
    }, index=idx)
    trades = pd.DataFrame({"ts": [idx[210], idx[220], idx[10]], "R_net": [1.0, -1.0, 0.5]})
    counts = evidence_cache.regime_signal_counts(trades, spy)
    assert counts["n_bear"] + counts["n_not_bear"] + counts["n_unclassified"] == counts["n_all"] == 3
    assert counts["n_unclassified"] == 1


def test_beats_baseline_requires_bootstrap_probability_and_gap():
    setup = pd.Series([0.20, 0.15, 0.10, 0.20, 0.15, 0.10])
    base = pd.Series([0.17, 0.14, 0.12, 0.16, 0.15, 0.13])
    weak = evidence_cache.beats_baseline_margin(setup, base, setup_ec=0.31, base_ec=0.28, seed=7, n_boot=500)
    assert weak["beats_baseline"] is False
    strong = evidence_cache.beats_baseline_margin(setup + 0.20, base, setup_ec=0.45, base_ec=0.30, seed=7, n_boot=500)
    assert strong["beats_baseline"] is True
