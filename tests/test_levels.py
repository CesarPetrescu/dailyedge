from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import levels


def sample_df():
    idx = pd.date_range("2026-01-01", periods=80, freq="D")
    rows = []
    price = 100.0
    for i in range(80):
        wave = [0, 3, 6, 3, 0, -3, -6, -3][i % 8]
        close = price + wave + i * 0.12
        rows.append(
            {
                "Open": close - 0.5,
                "High": close + 2.0,
                "Low": close - 2.0,
                "Close": close,
                "Volume": 1_000_000 + (i % 10) * 100_000,
            }
        )
    return pd.DataFrame(rows, index=idx)


def test_detect_all_outputs_structure():
    res = levels.detect_all(sample_df(), pivot_n=2, atr_mult=0.5, bins=24)
    assert res["ok"] is True
    assert res["sr"]
    assert res["volume_profile"]["poc"] is not None
    assert res["regression_channel"]["trend"] in {"up", "down", "flat"}
    assert res["round_levels"]


def test_levels_to_rules():
    res = levels.detect_all(sample_df(), pivot_n=2)
    rules = levels.levels_to_rules(res["sr"], res["last"], max_rules=3)
    assert 1 <= len(rules) <= 3
    assert all(r["type"] == "price_level" for r in rules)
    assert all(r["op"] in {"cross_below", "cross_above"} for r in rules)
