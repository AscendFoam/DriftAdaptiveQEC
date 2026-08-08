"""Generate signed Q3.14 memories for the frozen four-state student RTL."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "docs/t4_4_3_low_dimensional_student.json"
OUTPUT = ROOT / "cnn_fpga/rtl/generated"
WORD_BITS = 18
FRACTIONAL_BITS = 14


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def quantize(value: float) -> int:
    code = int(round(float(value) * (1 << FRACTIONAL_BITS)))
    return min(max(code, -(1 << (WORD_BITS - 1))), (1 << (WORD_BITS - 1)) - 1)


def _flatten(values: Iterable[Any]) -> list[float]:
    flattened: list[float] = []
    for value in values:
        if isinstance(value, list):
            flattened.extend(_flatten(value))
        else:
            flattened.append(float(value))
    return flattened


def _write_mem(name: str, values: list[float]) -> dict[str, Any]:
    path = OUTPUT / name
    mask = (1 << WORD_BITS) - 1
    codes = [quantize(value) for value in values]
    path.write_text("".join(f"{code & mask:05x}\n" for code in codes), encoding="ascii")
    reconstructed = [code / float(1 << FRACTIONAL_BITS) for code in codes]
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "entries": len(values),
        "sha256": _sha256(path),
        "maximum_absolute_quantization_error": max(
            abs(value - restored) for value, restored in zip(values, reconstructed)
        ),
    }


def generate() -> dict[str, Any]:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    if source["state_dimension"] != 4 or source["offline_teacher_object_embedded"]:
        raise ValueError("expected the frozen standalone four-state student artifact")
    groups = {
        "initial_state": _flatten(source["initial_state"]),
        "outcome_decays": _flatten(source["outcome_decays"]),
        "outcome_saturations": _flatten(source["outcome_saturations"]),
        "output_weights": _flatten(source["output_weights"]),
        "output_bias": _flatten(source["output_bias"]),
    }
    expected = {
        "initial_state": 4,
        "outcome_decays": 8,
        "outcome_saturations": 8,
        "output_weights": 60,
        "output_bias": 15,
    }
    if {name: len(values) for name, values in groups.items()} != expected:
        raise ValueError("student coefficient shape mismatch")
    files = [
        {"role": role, **_write_mem(f"t5_5_3_student_{role}.mem", values)}
        for role, values in groups.items()
    ]
    manifest = {
        "schema_version": "t5.5.3-four-state-student-fixed-memory-v1",
        "source_path": SOURCE.relative_to(ROOT).as_posix(),
        "source_sha256": _sha256(SOURCE),
        "word_bits": WORD_BITS,
        "fractional_bits": FRACTIONAL_BITS,
        "rounding": "nearest_ties_to_even",
        "saturation": "signed_18_bit",
        "files": files,
    }
    manifest_path = OUTPUT / "t5_5_3_student_memory_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    print(json.dumps(generate(), indent=2))


if __name__ == "__main__":
    main()
