"""CXXRTL equivalence and quantization audit for the frozen 4-state student."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RTL = ROOT / "cnn_fpga/rtl/low_dimensional_student_kernel.sv"
DRIVER = ROOT / "cnn_fpga/rtl/student_cxxrtl_driver.cc"
MANIFEST = ROOT / "cnn_fpga/rtl/generated/t5_5_3_student_memory_manifest.json"
FLOAT_ARTIFACT = ROOT / "docs/t4_4_3_low_dimensional_student.json"
DEFAULT_BUILD = ROOT / ".tmp_t5_5_3_student_equivalence"
DEFAULT_JSON = ROOT / "docs/t5_5_3_student_rtl_equivalence.json"
WORD_BITS = 18
FRACTIONAL_BITS = 14
MASK = (1 << WORD_BITS) - 1
MIN_CODE = -(1 << (WORD_BITS - 1))
MAX_CODE = (1 << (WORD_BITS - 1)) - 1


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def _find_executable(name: str, candidates: Iterable[Path]) -> Path:
    found = shutil.which(name)
    if found:
        return Path(found)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(name)


def discover_tools() -> dict[str, Path]:
    scripts = Path(sys.executable).resolve().parent / "Scripts"
    yosys = _find_executable("yowasp-yosys", (
        scripts / "yowasp-yosys.exe",
        Path(r"C:\ProgramData\anaconda3\envs\DLEnv\Scripts\yowasp-yosys.exe"),
    ))
    gpp = _find_executable("g++", (Path(r"C:\ProgramData\msys2\mingw64\bin\g++.exe"),))
    import yowasp_yosys
    include = Path(yowasp_yosys.__file__).resolve().parent / "share/include/backends/cxxrtl/runtime"
    return {"yosys": yosys, "gpp": gpp, "include": include}


def _run(command: Sequence[Any], *, env: dict[str, str], input_text: str | None = None, timeout: int = 600) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            [str(item) for item in command], cwd=ROOT, env=env, input=input_text,
            text=True, encoding="utf-8", errors="replace", capture_output=True,
            check=True, timeout=timeout,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"command failed with exit {exc.returncode}: {command}\n"
            f"stdout:\n{exc.stdout}\nstderr:\n{exc.stderr}"
        ) from exc


def _signed(code: int) -> int:
    code &= MASK
    return code - (1 << WORD_BITS) if code & (1 << (WORD_BITS - 1)) else code


def _load_mem(path: Path) -> list[int]:
    return [_signed(int(line, 16)) for line in path.read_text(encoding="ascii").splitlines()]


def load_coefficients() -> dict[str, list[int]]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    return {row["role"]: _load_mem(ROOT / row["path"]) for row in manifest["files"]}


def saturate(value: int) -> int:
    return min(max(int(value), MIN_CODE), MAX_CODE)


def round_product(value: int) -> int:
    magnitude = abs(int(value))
    quotient, remainder = divmod(magnitude, 1 << FRACTIONAL_BITS)
    half = 1 << (FRACTIONAL_BITS - 1)
    if remainder > half or (remainder == half and quotient & 1):
        quotient += 1
    return saturate(-quotient if value < 0 else quotient)


class FixedStudent:
    def __init__(self, coefficients: dict[str, list[int]]) -> None:
        self.c = coefficients
        self.state = list(coefficients["initial_state"])

    def reset(self) -> None:
        self.state = list(self.c["initial_state"])

    def step(self, outcome_e: bool, health_ok: bool = True) -> list[int]:
        if not health_ok:
            self.reset()
            return [0] * 15
        offset = 4 if outcome_e else 0
        updated = []
        for index in range(4):
            saturation = self.c["outcome_saturations"][offset + index]
            difference = saturate(self.state[index] - saturation)
            product = round_product(self.c["outcome_decays"][offset + index] * difference)
            updated.append(saturate(saturation + product))
        self.state = updated
        outputs = []
        for output in range(15):
            accumulator = self.c["output_bias"][output]
            for index in range(4):
                product = round_product(self.c["output_weights"][4 * output + index] * self.state[index])
                accumulator = saturate(accumulator + product)
            bound = (1 if output == 14 else 2) << FRACTIONAL_BITS
            outputs.append(min(max(accumulator, -bound), bound))
        return outputs


class FloatStudent:
    def __init__(self) -> None:
        artifact = json.loads(FLOAT_ARTIFACT.read_text(encoding="utf-8"))
        self.initial = np.asarray(artifact["initial_state"], dtype=np.float64)
        self.decay = np.asarray(artifact["outcome_decays"], dtype=np.float64)
        self.saturation = np.asarray(artifact["outcome_saturations"], dtype=np.float64)
        self.weights = np.asarray(artifact["output_weights"], dtype=np.float64)
        self.bias = np.asarray(artifact["output_bias"], dtype=np.float64)
        self.bounds = np.asarray(artifact["residual_bounds"], dtype=np.float64)
        self.state = self.initial.copy()

    def step(self, outcome_e: bool, health_ok: bool = True) -> np.ndarray:
        if not health_ok:
            self.state = self.initial.copy()
            return np.zeros(15, dtype=np.float64)
        index = int(outcome_e)
        self.state = self.saturation[index] + self.decay[index] * (self.state - self.saturation[index])
        return np.clip(self.bias + self.weights @ self.state, -self.bounds, self.bounds)


def build_cxxrtl(build_dir: Path, tools: dict[str, Path]) -> tuple[Path, dict[str, Any]]:
    build_dir.mkdir(parents=True, exist_ok=True)
    (build_dir / "temp").mkdir(exist_ok=True)
    (build_dir / "cache").mkdir(exist_ok=True)
    model = build_dir / "student_model.cc"
    executable = build_dir / "student_trace.exe"
    env = os.environ.copy()
    env["YOWASP_CACHE_DIR"] = str(build_dir / "cache")
    env["TEMP"] = env["TMP"] = str(build_dir / "temp")
    env["PATH"] = str(tools["gpp"].parent) + os.pathsep + env.get("PATH", "")
    script = (
        f"read_verilog -sv {_relative(RTL)}; hierarchy -check -top low_dimensional_student_kernel; "
        f"proc; check; stat; write_cxxrtl -O0 -g0 {_relative(model)}"
    )
    yosys = _run((tools["yosys"], "-Q", "-p", script), env=env)
    (build_dir / "yosys_cxxrtl.log").write_text(yosys.stdout + yosys.stderr, encoding="utf-8")
    compile_run = _run((
        tools["gpp"], "-std=c++17", "-O1", "-I", tools["include"],
        "-I", build_dir, DRIVER, "-o", executable,
    ), env=env, timeout=900)
    (build_dir / "gpp.log").write_text(compile_run.stdout + compile_run.stderr, encoding="utf-8")
    return executable, {
        "environment": env,
        "zero_structural_problems": "Found and reported 0 problems" in yosys.stdout,
        "model_sha256": _sha256(model),
        "executable_sha256": _sha256(executable),
        "multiplier_present": "$mul" in yosys.stdout,
    }


def build_trace(steps: int = 512) -> tuple[list[tuple[int, int, int]], list[dict[str, Any]], float]:
    rng = np.random.default_rng(55317)
    operations = []
    fixed = FixedStudent(load_coefficients())
    floating = FloatStudent()
    expected = []
    max_float_error = 0.0
    for step in range(steps):
        health = int(step % 97 != 96)
        outcome = int(rng.random() < (0.42 if step < steps // 2 else 0.67))
        outputs = fixed.step(bool(outcome), bool(health))
        float_outputs = floating.step(bool(outcome), bool(health))
        max_float_error = max(
            max_float_error,
            max(abs(code / (1 << FRACTIONAL_BITS) - value) for code, value in zip(outputs, float_outputs)),
        )
        operations.append((step, outcome, health))
        packed_state = sum((value & MASK) << (WORD_BITS * index) for index, value in enumerate(fixed.state))
        expected.append({
            "step": step, "outcome_e": outcome, "health_ok": health,
            "latency_cycles": 64 if health else 0,
            "state_hex": f"{packed_state:018x}",
            "outputs": outputs,
        })
    return operations, expected, float(max_float_error)


def compare(actual: list[dict[str, str]], expected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    mismatches = []
    if len(actual) != len(expected):
        return [{"kind": "row_count", "actual": len(actual), "expected": len(expected)}]
    for got, want in zip(actual, expected):
        fields = {
            "step": int(got["step"]), "outcome_e": int(got["outcome_e"]),
            "health_ok": int(got["health_ok"]), "latency_cycles": int(got["latency_cycles"]),
            "state_hex": got["state_hex"],
        }
        for name, value in fields.items():
            if value != want[name]:
                mismatches.append({"step": want["step"], "field": name, "actual": value, "expected": want[name]})
        for index, value in enumerate(want["outputs"]):
            actual_value = int(got[f"out{index}"])
            if actual_value != value:
                mismatches.append({"step": want["step"], "field": f"out{index}", "actual": actual_value, "expected": value})
        if len(mismatches) >= 20:
            break
    return mismatches


def run(build_dir: Path = DEFAULT_BUILD) -> dict[str, Any]:
    tools = discover_tools()
    executable, build = build_cxxrtl(build_dir, tools)
    operations, expected, max_float_error = build_trace()
    text = "".join(f"{step} {outcome} {health}\n" for step, outcome, health in operations)
    result = _run((executable,), env=build.pop("environment"), input_text=text, timeout=300)
    actual = list(csv.DictReader(io.StringIO(result.stdout)))
    mismatches = compare(actual, expected)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    gates = {
        "source_artifact_and_all_memories_are_hash_bound": (
            _sha256(FLOAT_ARTIFACT) == manifest["source_sha256"]
            and all(_sha256(ROOT / row["path"]) == row["sha256"] for row in manifest["files"])
        ),
        "cxxrtl_structural_check_preserves_multiplier": build["zero_structural_problems"] and build["multiplier_present"],
        "all_512_operations_match_every_state_and_output_code": not mismatches,
        "healthy_latency_is_exactly_64_cycles": all(row["latency_cycles"] == (64 if row["health_ok"] else 0) for row in expected),
        "health_failure_resets_state_and_zeroes_outputs": all(
            row["outputs"] == [0] * 15 for row in expected if not row["health_ok"]
        ),
        "fixed_q3_14_shadow_stays_within_5e_4_of_float": bool(max_float_error <= 5e-4),
    }
    return {
        "schema_version": "t5.5.3-student-cxxrtl-equivalence-v1",
        "task_id": "T5.5.3",
        "status": "PASS" if all(gates.values()) else "FAIL",
        "sources": [
            {"path": _relative(path), "sha256": _sha256(path)}
            for path in (RTL, DRIVER, MANIFEST, FLOAT_ARTIFACT)
        ],
        "trace": {
            "operations": len(expected),
            "healthy_updates": sum(row["health_ok"] for row in expected),
            "forced_resets": sum(not row["health_ok"] for row in expected),
            "compared_output_codes": len(expected) * 15,
            "mismatch_count": len(mismatches),
            "first_mismatches": mismatches,
            "maximum_absolute_fixed_minus_float_output": max_float_error,
        },
        "build": build,
        "gates": gates,
        "gate_summary": {"passed": sum(gates.values()), "total": len(gates)},
        "evidence_boundary": {
            "fixed_student_rtl": True,
            "cxxrtl_equivalence": True,
            "integrated_target_synthesis": False,
            "board_measured": False,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD)
    parser.add_argument("--output", type=Path, default=DEFAULT_JSON)
    args = parser.parse_args(argv)
    report = run(args.build_dir.resolve())
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "trace": report["trace"], "gates": report["gate_summary"]}, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
