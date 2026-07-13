from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


@pytest.mark.parametrize(
    "module_name",
    ["chart_simulation", "ledger", "multi_timeframe_model", "intraday_session_model"],
)
def test_default_output_dir_is_repo_local(module_name: str) -> None:
    module = importlib.import_module(module_name)
    assert module.OUTPUT_DIR == ROOT / "output"


def test_documented_and_configured_paths_are_portable() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    watcher = json.loads((ROOT / "config" / "armed_watchers.json").read_text(encoding="utf-8"))

    assert "/root/trading-agents" not in readme
    assert "/root/hermes/ultra-daytrader/dailyedge" not in readme
    assert not Path(watcher["source"]).is_absolute()
