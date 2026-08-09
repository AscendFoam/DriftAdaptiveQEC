from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from scripts.check_physics_module_size import validate


ROOT = Path(__file__).resolve().parents[1]


def test_physics_modules_respect_size_policy() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/check_physics_module_size.py"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_size_policy_rejects_an_oversize_module(tmp_path: Path) -> None:
    module = tmp_path / "nested" / "too_large.py"
    module.parent.mkdir()
    module.write_text("pass\n" * 1001, encoding="utf-8")

    failures = validate(tmp_path)

    assert len(failures) == 1
    assert "1001 lines exceeds the 1000-line policy" in failures[0]
