from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from cnn_fpga.benchmark.student_rtl_equivalence import (
    DEFAULT_JSON,
    FLOAT_ARTIFACT,
    FRACTIONAL_BITS,
    MANIFEST,
    RTL,
    FixedStudent,
    FloatStudent,
    load_coefficients,
    round_product,
)
from cnn_fpga.rtl.generate_student_fixed_memories import generate


ROOT = Path(__file__).resolve().parents[1]


def _report() -> dict:
    return json.loads(DEFAULT_JSON.read_text(encoding="utf-8"))


def test_generated_memories_are_deterministic_and_source_bound() -> None:
    before = MANIFEST.read_bytes()
    manifest = generate()
    assert MANIFEST.read_bytes() == before
    assert manifest["source_sha256"] == _sha256(FLOAT_ARTIFACT)
    assert [row["entries"] for row in manifest["files"]] == [4, 8, 8, 60, 15]
    assert max(row["maximum_absolute_quantization_error"] for row in manifest["files"]) < 3.1e-5


def _sha256(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_ties_even_rounding_is_signed_and_saturating() -> None:
    half = 1 << (FRACTIONAL_BITS - 1)
    assert round_product(2 * (1 << FRACTIONAL_BITS) + half) == 2
    assert round_product(3 * (1 << FRACTIONAL_BITS) + half) == 4
    assert round_product(-(2 * (1 << FRACTIONAL_BITS) + half)) == -2
    assert round_product(-(3 * (1 << FRACTIONAL_BITS) + half)) == -4


def test_python_fixed_student_has_exact_reset_policy() -> None:
    model = FixedStudent(load_coefficients())
    initial = list(model.state)
    assert model.step(True) != [0] * 15
    assert model.state != initial
    assert model.step(False, health_ok=False) == [0] * 15
    assert model.state == initial


def test_fixed_shadow_tracks_float_over_nonstationary_sequence() -> None:
    fixed = FixedStudent(load_coefficients())
    floating = FloatStudent()
    rng = np.random.default_rng(55391)
    maximum = 0.0
    for index in range(1000):
        health = index % 113 != 112
        outcome = bool(rng.integers(0, 2))
        fixed_values = fixed.step(outcome, health)
        float_values = floating.step(outcome, health)
        maximum = max(maximum, max(abs(code / (1 << FRACTIONAL_BITS) - value) for code, value in zip(fixed_values, float_values)))
    assert maximum < 2.0e-4


def test_rtl_is_integer_only_and_contains_real_serial_mac_and_fail_close() -> None:
    source = RTL.read_text(encoding="utf-8")
    code = "\n".join(line.split("//", 1)[0] for line in source.splitlines())
    assert " real " not in code
    assert "$readmemh" in source
    assert "multiplier_product = multiplier_a * multiplier_b" in source
    assert "round_product_ties_even" in source
    assert "else if (!health_ok)" in source
    assert "bias_mem[next_output_cursor]" in source
    assert "bias_mem[output_cursor + 1'b1]" not in source
    assert re.search(r"\bbias_mem\s*\[\s*15\s*\]", code) is None


def test_formal_cxxrtl_report_passes_every_gate() -> None:
    report = _report()
    assert report["status"] == "PASS"
    assert report["gate_summary"] == {"passed": 6, "total": 6}
    assert all(report["gates"].values())


def test_cxxrtl_report_compares_all_codes_state_and_resets() -> None:
    trace = _report()["trace"]
    assert trace["operations"] == 512
    assert trace["healthy_updates"] == 507
    assert trace["forced_resets"] == 5
    assert trace["compared_output_codes"] == 7680
    assert trace["mismatch_count"] == 0
    assert trace["maximum_absolute_fixed_minus_float_output"] < 1.5e-4


def test_cxxrtl_report_keeps_hardware_boundary_closed() -> None:
    boundary = _report()["evidence_boundary"]
    assert boundary["fixed_student_rtl"] is True
    assert boundary["cxxrtl_equivalence"] is True
    assert boundary["integrated_target_synthesis"] is False
    assert boundary["board_measured"] is False
