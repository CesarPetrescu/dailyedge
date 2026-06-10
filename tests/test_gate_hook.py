from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import edge_engine
import gate_hook


def test_edge_engine_cost_and_breakeven():
    c = edge_engine.cost_in_R(2.0, spread_bps=3.0, slippage_bps=4.0, fees_bps=0.0)
    assert round(c, 4) == 0.055
    assert 0.34 < edge_engine.breakeven_hit_rate(2.0, c) < 0.36


def test_gate_hook_go_auto_structure_uses_shared_params():
    g = gate_hook.gate(entry=100.0, atr=2.0)
    assert g["go"] is True
    assert g["auto"] is True
    assert round(g["rr"], 2) == 2.0
    assert round(g["stop"], 2) == 97.0
    assert "GO[auto-structure]" in gate_hook.stamp_line(entry=100.0, atr=2.0)


def test_gate_hook_rejects_high_cost_low_rr():
    g = gate_hook.gate(entry=100.0, stop=99.5, target=100.6, atr=1.0)
    assert g["go"] is False
    assert any("R:R" in reason for reason in g["fails"])
    assert any("cost" in reason for reason in g["fails"])
