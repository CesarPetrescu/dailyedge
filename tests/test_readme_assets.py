from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"


def test_readme_asset_references_exist() -> None:
    readme = README.read_text(encoding="utf-8")
    refs = re.findall(r"!\[[^\]]*\]\((docs/assets/[^)]+)\)", readme)
    assert refs, "README should embed docs/assets images"
    missing = [ref for ref in refs if not (ROOT / ref).exists()]
    assert missing == []


def test_readme_graphics_layout_check() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/render_readme_assets.py", "--check"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "all text blocks fit" in result.stdout
