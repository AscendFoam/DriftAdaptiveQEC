"""Compare full GRU, quantized-GRU, and distilled-student hardware feasibility.

The quantized-GRU RTL used here is deliberately an optimistic lower-bound
workload: it stores every selected-teacher parameter and consumes every weight
and bias, but it does not implement GRU dependencies or nonlinearities.  It can
therefore disqualify the route on resources/latency, but can never qualify it.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import csv
import gzip
import hashlib
import importlib.metadata
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from cnn_fpga.benchmark.target_device_synthesis import (
    CST,
    DEVICE,
    FAMILY,
    SDC,
    TARGET_MHZ,
    _read_tool_text,
    parse_nextpnr,
)


ROOT = Path(__file__).resolve().parents[2]
RUNNER = Path(__file__).resolve()
DEFAULT_BUILD = ROOT / ".tmp_t554_build"
DEFAULT_CXXRTL_BUILD = ROOT / ".tmp_t554_cxxrtl"
DEFAULT_JSON = ROOT / "docs/t5_5_4_gru_student_hardware_feasibility.json"
DEFAULT_CSV = ROOT / "docs/t5_5_4_gru_student_hardware_feasibility_source_data.csv"
CHECKPOINT = ROOT / "docs/t4_4_1_bounded_residual_rnn_teacher_checkpoints.pt"
TEACHER_VALIDATION = ROOT / "docs/t4_4_1_bounded_residual_rnn_teacher_validation.json"
GAIN_PARENT = ROOT / "docs/t4_4_4_teacher_student_gain_retention.json"
STUDENT_EQ_PARENT = ROOT / "docs/t5_5_3_student_rtl_equivalence.json"
STUDENT_PARETO_PARENT = ROOT / "docs/t5_5_3_precision_resource_pareto.json"
QUANT_MANIFEST = ROOT / "cnn_fpga/rtl/generated/t5_5_4_quantized_gru_manifest.json"
STUDENT_MANIFEST = ROOT / "cnn_fpga/rtl/generated/t5_5_3_student_memory_manifest.json"
CORE_RTL = ROOT / "cnn_fpga/rtl/gkp_fast_path_core.sv"
CORE_TOP = ROOT / "cnn_fpga/rtl/gkp_fast_path_synth_top.sv"
QUANT_RTL = ROOT / "cnn_fpga/rtl/quantized_gru_workload_kernel.sv"
QUANT_TOP = ROOT / "cnn_fpga/rtl/gkp_fast_path_gru_workload_synth_top.sv"
CXXRTL_DRIVER = ROOT / "cnn_fpga/rtl/quantized_gru_workload_cxxrtl_driver.cc"
TOP_MODULE = "gkp_fast_path_gru_workload_synth_top"
SEEDS = (1, 7, 19)
BRAM_BLOCK_BITS = 18_432
BRAM_BLOCKS_AVAILABLE = 46
MULT9_AVAILABLE = 96
STUDENT_DEADLINE_US = 5.0
WEIGHT_MACS = 72_266
BIAS_SCALARS = 587
TOTAL_PARAMETERS = WEIGHT_MACS + BIAS_SCALARS
WORKLOAD_CYCLES = 72_854
Q14_SCALE = 1 << 14
RESIDUAL_BOUNDS = np.asarray(
    [2.0] * 14 + [1.0], dtype=np.float64
)
VERDICT = "DISTILLED_STUDENT_ONLY_QUANTIZED_GRU_DROPPED_FULL_GRU_OFFLINE_TEACHER"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def _binding(path: Path) -> dict[str, Any]:
    return {"path": _relative(path), "sha256": _sha256(path), "bytes": path.stat().st_size}


def _matches(binding: dict[str, Any]) -> bool:
    path = ROOT / binding["path"]
    return path.is_file() and path.stat().st_size == binding["bytes"] and _sha256(path) == binding["sha256"]


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
    yosys = _find_executable(
        "yowasp-yosys",
        (scripts / "yowasp-yosys.exe", Path(r"C:\ProgramData\anaconda3\envs\DLEnv\Scripts\yowasp-yosys.exe")),
    )
    nextpnr = _find_executable(
        "yowasp-nextpnr-himbaechel-gowin",
        (scripts / "yowasp-nextpnr-himbaechel-gowin.exe", Path(r"C:\ProgramData\anaconda3\envs\DLEnv\Scripts\yowasp-nextpnr-himbaechel-gowin.exe")),
    )
    gpp = _find_executable("g++", (Path(r"C:\ProgramData\msys2\mingw64\bin\g++.exe"),))
    import yowasp_yosys
    include = Path(yowasp_yosys.__file__).resolve().parent / "share/include/backends/cxxrtl/runtime"
    return {"yosys": yosys, "nextpnr": nextpnr, "gpp": gpp, "include": include}


def _run(
    command: Sequence[Any], *, env: dict[str, str], timeout: int = 1200,
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            [str(value) for value in command], cwd=ROOT, env=env, input=input_text,
            text=True, encoding="utf-8", errors="replace", capture_output=True,
            timeout=timeout, check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"command failed ({exc.returncode}): {command}\nstdout:\n{exc.stdout}\nstderr:\n{exc.stderr}"
        ) from exc


def _seed_paths(build_dir: Path, seed: int) -> dict[str, Path]:
    tag = f"{seed:02d}"
    return {
        "report": build_dir / f"nextpnr_seed{tag}_report.json",
        "log": build_dir / f"nextpnr_seed{tag}.log",
        "routed": build_dir / f"gkp_gru_workload_routed_seed{tag}.json",
    }


def run_toolchain(build_dir: Path, tools: dict[str, Path]) -> None:
    """Rebuild the integrated lower-bound netlist and all three P&R seeds."""
    build_dir.mkdir(parents=True, exist_ok=True)
    cache = build_dir / "yowasp_cache"
    cache.mkdir(exist_ok=True)
    env = os.environ.copy()
    env["YOWASP_CACHE_DIR"] = str(cache)
    netlist = build_dir / "gkp_gru_workload_gw2a.json"
    script = (
        f"read_verilog -sv {_relative(CORE_RTL)} {_relative(CORE_TOP)} {_relative(QUANT_RTL)} {_relative(QUANT_TOP)}; "
        f"synth_gowin -family gw2a -no-rw-check -top {TOP_MODULE} -json {_relative(netlist)}; check; stat"
    )
    synthesis_command = (tools["yosys"], "-Q", "-p", script)
    result = _run(synthesis_command, env=env)
    (build_dir / "yosys_integrated_gru_full.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    route_commands = {}

    def route_seed(seed: int) -> None:
        paths = _seed_paths(build_dir, seed)
        command = (
            tools["nextpnr"], "--device", DEVICE,
            "-o", f"family={FAMILY}", "-o", f"cst={_relative(CST)}",
            "--json", _relative(netlist), "--top", TOP_MODULE,
            "--freq", f"{TARGET_MHZ:g}", "--sdc", _relative(SDC),
            "--seed", seed, "--report", _relative(paths["report"]),
            "--detailed-timing-report", "--write", _relative(paths["routed"]),
        )
        result = _run(command, env=env, timeout=1800)
        paths["log"].write_text(result.stdout + result.stderr, encoding="utf-8")
        route_commands[str(seed)] = [str(value) for value in command]
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(SEEDS)) as executor:
        futures = [executor.submit(route_seed, seed) for seed in SEEDS]
        for future in futures:
            future.result()
    provenance = {
        "schema_version": "t5.5.4-toolchain-execution-v1",
        "input_bindings": [_binding(path) for path in (CORE_RTL, CORE_TOP, QUANT_RTL, QUANT_TOP, SDC, CST, QUANT_MANIFEST)],
        "synthesis_command": [str(value) for value in synthesis_command],
        "route_commands": route_commands,
        "uncompressed_netlist": _binding(netlist),
        "synthesis_log": _binding(build_dir / "yosys_integrated_gru_full.log"),
        "routes": [
            {"seed": seed, "report": _binding(_seed_paths(build_dir, seed)["report"]), "log": _binding(_seed_paths(build_dir, seed)["log"])}
            for seed in SEEDS
        ],
    }
    (build_dir / "toolchain_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")


def _copy_tool_artifacts(build_dir: Path) -> tuple[Path, list[dict[str, Any]], dict[str, Any]]:
    docs = ROOT / "docs"
    provenance_path = build_dir / "toolchain_provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if not all(_matches(row) for row in provenance["input_bindings"]):
        raise ValueError("toolchain input changed after synthesis; rerun with --run-toolchain")
    if not _matches(provenance["uncompressed_netlist"]):
        raise ValueError("synthesized netlist changed after recorded tool execution")
    synthesis_source = build_dir / "yosys_integrated_gru_full.log"
    if _sha256(synthesis_source) != provenance["synthesis_log"]["sha256"]:
        raise ValueError("synthesis log changed after recorded tool execution")
    synthesis_dest = docs / "t5_5_4_yosys_quantized_gru_lower_bound.log"
    synthesis_dest.write_text(_read_tool_text(synthesis_source), encoding="utf-8")
    artifacts = [_binding(synthesis_dest)]
    for seed in SEEDS:
        tag = f"{seed:02d}"
        source = _seed_paths(build_dir, seed)
        report_dest = docs / f"t5_5_4_nextpnr_seed{tag}_report.json"
        log_dest = docs / f"t5_5_4_nextpnr_seed{tag}_place_route.log"
        shutil.copyfile(source["report"], report_dest)
        log_dest.write_text(_read_tool_text(source["log"]), encoding="utf-8")
        artifacts.extend((_binding(report_dest), _binding(log_dest)))
    netlist_source = ROOT / provenance["uncompressed_netlist"]["path"]
    netlist_dest = docs / "t5_5_4_quantized_gru_lower_bound_netlist.json.gz"
    with netlist_source.open("rb") as source_handle, netlist_dest.open("wb") as raw_handle:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_handle, mtime=0) as compressed:
            shutil.copyfileobj(source_handle, compressed)
    artifacts.append(_binding(netlist_dest))
    durable_provenance = {
        "schema_version": provenance["schema_version"],
        "input_bindings": provenance["input_bindings"],
        "synthesis_command": provenance["synthesis_command"],
        "route_commands": provenance["route_commands"],
        "uncompressed_netlist_sha256": provenance["uncompressed_netlist"]["sha256"],
        "uncompressed_netlist_bytes": provenance["uncompressed_netlist"]["bytes"],
        "compressed_netlist_artifact": _binding(netlist_dest),
        "synthesis_log_artifact": _binding(synthesis_dest),
        "route_artifacts": [
            {
                "seed": seed,
                "report": _binding(docs / f"t5_5_4_nextpnr_seed{seed:02d}_report.json"),
                "log": _binding(docs / f"t5_5_4_nextpnr_seed{seed:02d}_place_route.log"),
            }
            for seed in SEEDS
        ],
    }
    durable_path = docs / "t5_5_4_toolchain_provenance.json"
    durable_path.write_text(json.dumps(durable_provenance, indent=2) + "\n", encoding="utf-8")
    artifacts.append(_binding(durable_path))
    return synthesis_dest, artifacts, durable_provenance


def parse_synthesis_memory(path: Path) -> dict[str, Any]:
    text = _read_tool_text(path)
    counts = {}
    for name in ("SPX9", "SDPX9B", "MULT18X18", "MULT9X9"):
        matches = re.findall(rf"^\s*(\d+)\s+{name}\s*$", text, flags=re.MULTILINE)
        counts[name] = int(matches[-1]) if matches else 0
    return {
        "artifact": _binding(path),
        "primitive_counts_before_abc9": counts,
        "complete_parameter_rom_primitives": counts["SPX9"] == 33,
        "fast_path_bram_primitives": counts["SDPX9B"] == 8,
        "note": "pre-ABC9 primitive inventory; placed utilization is authoritative",
    }


def _load_parent(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = payload.get("gate_summary") or {}
    passed = int(summary.get("passed", 0))
    total = int(summary.get("total", passed + len(summary.get("failed", []))))
    return payload, {
        "artifact": _binding(path), "task_id": payload.get("task_id"),
        "status": payload.get("status"), "gate_summary": {"passed": passed, "total": total},
    }


def _load_signed_mem(path: Path, bits: int) -> list[int]:
    sign = 1 << (bits - 1)
    modulus = 1 << bits
    result = []
    for line in path.read_text(encoding="ascii").splitlines():
        value = int(line, 16)
        result.append(value - modulus if value & sign else value)
    return result


def _wrap_signed(value: int, bits: int) -> int:
    mask = (1 << bits) - 1
    code = int(value) & mask
    return code - (1 << bits) if code & (1 << (bits - 1)) else code


def _signature_rotate(value: int) -> int:
    value &= 0xFFFFFFFF
    feedback = ((value >> 31) ^ (value >> 21)) & 1
    return ((value & 0x7FFFFFFF) << 1) | feedback


def workload_reference_signature(manifest: dict[str, Any]) -> dict[str, int]:
    """Independent bit-vector replay of the RTL ROM address and signature path."""
    weights = _load_signed_mem(ROOT / manifest["weight_file"]["path"], 8)
    biases = _load_signed_mem(ROOT / manifest["bias_file"]["path"], 18)
    if len(weights) != WEIGHT_MACS or len(biases) != BIAS_SCALARS:
        raise ValueError("workload memory lengths do not match the selected GRU")
    signature = 0x243F6A88
    activation_code = 0x13579
    accumulator = 0
    for index, weight in enumerate(weights):
        activation = _wrap_signed(activation_code, 18)
        product = _wrap_signed(weight * activation, 26)
        signature = (
            _signature_rotate(signature) ^ (product & 0xFFFFFFFF) ^ index
        ) & 0xFFFFFFFF
        accumulator = _wrap_signed(accumulator + product, 40)
        feedback = ((activation_code >> 17) ^ (activation_code >> 10) ^ activation_code) & 1
        activation_code = (((activation_code & 0x1FFFF) << 1) | feedback) & 0x3FFFF
    for bias in biases:
        signature = (
            _signature_rotate(signature)
            ^ (accumulator & 0xFFFFFFFF)
            ^ (bias & 0xFFFFFFFF)
        ) & 0xFFFFFFFF
        accumulator = _wrap_signed(accumulator + bias, 40)
    return {
        "signature": signature,
        "final_accumulator_signed40": accumulator,
        "final_activation_code18": activation_code,
        "ordered_weight_entries": len(weights),
        "ordered_bias_entries": len(biases),
    }


def _require_torch() -> Any:
    import torch
    return torch


def load_teacher_states() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    torch = _require_torch()
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    index = int(payload["selected_restart_index"])
    selected = payload["restarts"][index]
    float_state = {name: value.detach().to(dtype=torch.float64).clone() for name, value in selected["state_dict"].items()}
    manifest = json.loads(QUANT_MANIFEST.read_text(encoding="utf-8"))
    if selected["checkpoint_sha256"] != manifest["selected_state_sha256"]:
        raise ValueError("quantized manifest is not bound to selected GRU state")
    weight_codes = _load_signed_mem(ROOT / manifest["weight_file"]["path"], 8)
    bias_codes = _load_signed_mem(ROOT / manifest["bias_file"]["path"], 18)
    quant_state: dict[str, Any] = {}
    for row in manifest["weight_tensors"]:
        start = int(row["offset"])
        count = int(row["entries"])
        values = torch.tensor(weight_codes[start : start + count], dtype=torch.float64)
        quant_state[row["name"]] = (values / (1 << int(row["fractional_bits"]))).reshape(row["shape"])
    for row in manifest["bias_tensors"]:
        start = int(row["offset"])
        count = int(row["entries"])
        values = torch.tensor(bias_codes[start : start + count], dtype=torch.float64)
        quant_state[row["name"]] = (values / Q14_SCALE).reshape(row["shape"])
    return float_state, quant_state, manifest


def _q14(value: Any) -> Any:
    torch = _require_torch()
    return torch.clamp(torch.round(value * Q14_SCALE), -(1 << 17), (1 << 17) - 1) / Q14_SCALE


def _gru_step(x: Any, hidden: Any, state: dict[str, Any], *, quantized: bool) -> Any:
    torch = _require_torch()
    functional = torch.nn.functional
    gi = functional.linear(x, state["gru.weight_ih"], state["gru.bias_ih"])
    gh = functional.linear(hidden, state["gru.weight_hh"], state["gru.bias_hh"])
    if quantized:
        gi, gh = _q14(gi), _q14(gh)
    gi_r, gi_z, gi_n = gi.chunk(3, dim=-1)
    gh_r, gh_z, gh_n = gh.chunk(3, dim=-1)
    r = torch.sigmoid(gi_r + gh_r)
    z = torch.sigmoid(gi_z + gh_z)
    if quantized:
        r, z = _q14(r), _q14(z)
        candidate = _q14(torch.tanh(_q14(gi_n + _q14(r * gh_n))))
        return _q14(candidate + _q14(z * _q14(hidden - candidate)))
    candidate = torch.tanh(gi_n + r * gh_n)
    return candidate + z * (hidden - candidate)


def _actions(hidden: Any, state: dict[str, Any], *, quantized: bool) -> Any:
    torch = _require_torch()
    functional = torch.nn.functional
    value = functional.linear(hidden, state["dense1.weight"], state["dense1.bias"])
    value = torch.tanh(_q14(value) if quantized else value)
    if quantized:
        value = _q14(value)
    value = functional.linear(value, state["dense2.weight"], state["dense2.bias"])
    value = torch.tanh(_q14(value) if quantized else value)
    if quantized:
        value = _q14(value)
    raw = functional.linear(value, state["output.weight"], state["output.bias"])
    if quantized:
        raw = _q14(raw)
    bounds = torch.tensor(RESIDUAL_BOUNDS, dtype=hidden.dtype, device=hidden.device)
    action = bounds * torch.tanh(raw)
    return _q14(action) if quantized else action


def _evaluate_sequences(outcomes: np.ndarray, float_state: dict[str, Any], quant_state: dict[str, Any]) -> dict[str, Any]:
    torch = _require_torch()
    data = torch.tensor(2.0 * outcomes - 1.0, dtype=torch.float64).unsqueeze(-1)
    float_hidden = torch.zeros((data.shape[0], 10), dtype=torch.float64)
    quant_hidden = torch.zeros_like(float_hidden)
    action_errors = []
    hidden_errors = []
    finite = True
    bounded = True
    with torch.no_grad():
        for index in range(data.shape[1]):
            float_hidden = _gru_step(data[:, index], float_hidden, float_state, quantized=False)
            quant_hidden = _gru_step(data[:, index], quant_hidden, quant_state, quantized=True)
            float_action = _actions(float_hidden, float_state, quantized=False)
            quant_action = _actions(quant_hidden, quant_state, quantized=True)
            action_errors.append((quant_action - float_action).detach().cpu().numpy())
            hidden_errors.append((quant_hidden - float_hidden).detach().cpu().numpy())
            finite = finite and bool(torch.isfinite(quant_hidden).all() and torch.isfinite(quant_action).all())
            bounded = bounded and bool(torch.all(torch.abs(quant_action) <= torch.tensor(RESIDUAL_BOUNDS) + 1e-12))
    actions = np.concatenate([row.reshape(-1) for row in action_errors])
    hidden = np.concatenate([row.reshape(-1) for row in hidden_errors])
    absolute = np.abs(actions)
    return {
        "sequence_count": int(outcomes.shape[0]), "steps_per_sequence": int(outcomes.shape[1]),
        "action_scalar_comparisons": int(actions.size), "hidden_scalar_comparisons": int(hidden.size),
        "action_rmse": float(np.sqrt(np.mean(actions * actions))),
        "action_maximum_absolute_error": float(absolute.max()),
        "action_p99_absolute_error": float(np.quantile(absolute, 0.99)),
        "hidden_rmse": float(np.sqrt(np.mean(hidden * hidden))),
        "hidden_maximum_absolute_error": float(np.max(np.abs(hidden))),
        "all_quantized_values_finite": finite, "all_quantized_actions_bounded": bounded,
    }


def quantized_functional_shadow() -> dict[str, Any]:
    torch = _require_torch()
    float_state, quant_state, manifest = load_teacher_states()
    cell = torch.nn.GRUCell(1, 10).to(dtype=torch.float64)
    with torch.no_grad():
        cell.load_state_dict({name.removeprefix("gru."): value for name, value in float_state.items() if name.startswith("gru.")})
    generator = torch.Generator(device="cpu")
    generator.manual_seed(55441)
    x = torch.randn((64, 1), dtype=torch.float64, generator=generator)
    hidden = torch.randn((64, 10), dtype=torch.float64, generator=generator)
    with torch.no_grad():
        manual = _gru_step(x, hidden, float_state, quantized=False)
        torch_error = float(torch.max(torch.abs(manual - cell(x, hidden))).item())
    exhaustive = np.asarray(
        [[(value >> (7 - bit)) & 1 for bit in range(8)] for value in range(256)], dtype=np.float64
    )
    rng = np.random.default_rng(55443)
    long_random = rng.integers(0, 2, size=(128, 256), dtype=np.int8).astype(np.float64)
    return {
        "model": "functional fake-quantized shadow, not RTL and not physical-gain evidence",
        "rounding_scope": "per-tensor int8 weights; Q3.14 biases and post-affine/nonlinearity state",
        "torch_grucell_equation_max_abs_error": torch_error,
        "exhaustive_length8_histories_all_prefixes": _evaluate_sequences(exhaustive, float_state, quant_state),
        "long_random_sequences": _evaluate_sequences(long_random, float_state, quant_state),
        "manifest": _binding(QUANT_MANIFEST),
        "weight_file": _binding(ROOT / manifest["weight_file"]["path"]),
        "bias_file": _binding(ROOT / manifest["bias_file"]["path"]),
        "physical_gain_retention": None,
        "physical_gain_retention_reason": "T4.4.4 physics benchmark was not rerun with this quantized functional shadow",
    }


def run_cxxrtl_trace(build_dir: Path, tools: dict[str, Path]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    build_dir.mkdir(parents=True, exist_ok=True)
    temp = build_dir / "temp"
    cache = build_dir / "cache"
    temp.mkdir(exist_ok=True)
    cache.mkdir(exist_ok=True)
    model = build_dir / "quantized_gru_workload_model.cc"
    executable = build_dir / "quantized_gru_workload_trace.exe"
    env = os.environ.copy()
    env["YOWASP_CACHE_DIR"] = str(cache)
    env["TEMP"] = env["TMP"] = str(temp)
    env["PATH"] = str(tools["gpp"].parent) + os.pathsep + env.get("PATH", "")
    script = (
        f"read_verilog -sv {_relative(QUANT_RTL)}; hierarchy -check -top quantized_gru_workload_kernel; "
        f"proc; check; stat; write_cxxrtl -O0 -g0 {_relative(model)}"
    )
    yosys = _run((tools["yosys"], "-Q", "-p", script), env=env)
    yosys_log = build_dir / "yosys_cxxrtl.log"
    yosys_log.write_text(yosys.stdout + yosys.stderr, encoding="utf-8")
    compile_result = _run(
        (tools["gpp"], "-std=c++17", "-O1", "-I", tools["include"], "-I", build_dir, CXXRTL_DRIVER, "-o", executable),
        env=env, timeout=900,
    )
    gpp_log = build_dir / "gpp.log"
    gpp_log.write_text(compile_result.stdout + compile_result.stderr, encoding="utf-8")
    trace_result = _run((executable,), env=env, timeout=300)
    reader = csv.DictReader(trace_result.stdout.splitlines())
    rows = list(reader)
    if len(rows) != 1:
        raise ValueError("expected exactly one CXXRTL workload trace row")
    row = rows[0]
    trace_path = ROOT / "docs/t5_5_4_quantized_gru_workload_trace.csv"
    trace_path.write_text(trace_result.stdout, encoding="utf-8")
    durable_yosys = ROOT / "docs/t5_5_4_quantized_gru_cxxrtl_yosys.log"
    durable_yosys.write_text(_read_tool_text(yosys_log), encoding="utf-8")
    return {
        "cycles_after_start": int(row["cycles_after_start"]),
        "weight_macs_completed": int(row["weight_macs_completed"]),
        "biases_consumed": int(row["biases_consumed"]),
        "done": bool(int(row["done"])), "busy": bool(int(row["busy"])),
        "signature": int(row["signature"]),
        "zero_structural_problems": "Found and reported 0 problems" in yosys.stdout,
        "model_sha256": _sha256(model), "executable_sha256": _sha256(executable),
        "trace_artifact": _binding(trace_path),
    }, [_binding(trace_path), _binding(durable_yosys)]


def _max_resources(routes: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    names = ("LUT4", "DFF", "BSRAM", "MULT18X18", "MULT9X9", "ALU", "IOB")
    return {
        name: {
            "used": max(row["utilization"][name]["used"] for row in routes),
            "available": routes[0]["utilization"][name]["available"],
        }
        for name in names
    }


def _student_retention(gain_parent: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for split, metrics in gain_parent["stochastic_retention"].items():
        for metric, values in metrics.items():
            rows.append({
                "split": split, "metric": metric,
                "point": float(values["point_retention_fraction"]),
                "ci_lower": float(values["ci_95"][0]), "ci_upper": float(values["ci_95"][1]),
            })
    return {
        "rows": rows,
        "minimum_point": min(row["point"] for row in rows),
        "minimum_ci_lower": min(row["ci_lower"] for row in rows),
        "threshold": float(gain_parent["retention_threshold"]["point_fraction"]),
        "ci_lower_threshold": float(gain_parent["retention_threshold"]["paired_bootstrap_ci_lower"]),
        "evidence": "T4.4.4 matched physical benchmark",
    }


def _full_gru_rows() -> list[dict[str, Any]]:
    rows = []
    optimistic_cycles = math.ceil(WEIGHT_MACS / MULT9_AVAILABLE)
    for bits in (64, 32):
        parameter_bits = TOTAL_PARAMETERS * bits
        parameter_blocks = math.ceil(parameter_bits / BRAM_BLOCK_BITS)
        rows.append({
            "candidate_id": f"full_gru_float{bits}", "role": "offline_teacher_only",
            "functional_model": True, "weight_bits": bits, "bias_bits": bits,
            "stored_parameters": TOTAL_PARAMETERS, "parameter_storage_bits": parameter_bits,
            "parameter_bram_blocks_lower_bound": parameter_blocks,
            "integrated_bram_blocks_lower_bound": parameter_blocks + 8,
            "analytic_weight_macs_per_update": WEIGHT_MACS,
            "optimistic_96_multiplier_cycle_lower_bound": optimistic_cycles,
            "optimistic_latency_us_at_27mhz_lower_bound": optimistic_cycles / TARGET_MHZ,
            "actual_target_synthesis": False,
            "synthesis_omission_reason": "parameter storage alone exceeds all target BSRAM before activations, state, and arithmetic",
            "actual_post_route_resources": None, "actual_fmax_mhz_minimum": None,
            "worst_case_latency_us": None,
            "physical_gain_retention": 1.0,
            "gain_retention_scope": "teacher reference, not a deployed implementation",
            "deadline_pass": False, "capacity_pass": False, "enhanced_route_eligible": False,
        })
    return rows


def build_report(build_dir: Path, cxxrtl_build: Path) -> dict[str, Any]:
    teacher, teacher_row = _load_parent(TEACHER_VALIDATION)
    gain, gain_row = _load_parent(GAIN_PARENT)
    student_eq, student_eq_row = _load_parent(STUDENT_EQ_PARENT)
    student_pareto, student_pareto_row = _load_parent(STUDENT_PARETO_PARENT)
    manifest = json.loads(QUANT_MANIFEST.read_text(encoding="utf-8"))
    student_manifest = json.loads(STUDENT_MANIFEST.read_text(encoding="utf-8"))
    synthesis_path, durable, toolchain_provenance = _copy_tool_artifacts(build_dir)
    tools = discover_tools()
    workload_trace, cxxrtl_artifacts = run_cxxrtl_trace(cxxrtl_build, tools)
    workload_reference = workload_reference_signature(manifest)
    workload_trace["independent_reference"] = workload_reference
    workload_trace["signature_matches_independent_reference"] = (
        workload_trace["signature"] == workload_reference["signature"]
    )
    durable.extend(cxxrtl_artifacts)
    routes = [
        parse_nextpnr(
            ROOT / f"docs/t5_5_4_nextpnr_seed{seed:02d}_report.json",
            ROOT / f"docs/t5_5_4_nextpnr_seed{seed:02d}_place_route.log", seed,
        )
        for seed in SEEDS
    ]
    resources = _max_resources(routes)
    fmax_values = [row["achieved_fmax_mhz"] for row in routes]
    shadow = quantized_functional_shadow()
    student_retention = _student_retention(gain)
    full_rows = _full_gru_rows()
    quant_bits = int(manifest["counts"]["total_quantized_parameter_bits"])
    quant_row = {
        "candidate_id": "quantized_gru_int8_q14_lower_bound", "role": "dropped_optional_enhanced_route",
        "functional_model": False, "functional_shadow_available": True,
        "weight_bits": 8, "bias_bits": 18, "stored_parameters": TOTAL_PARAMETERS,
        "parameter_storage_bits": quant_bits, "parameter_bram_blocks_analytic": math.ceil(WEIGHT_MACS * 8 / BRAM_BLOCK_BITS) + math.ceil(BIAS_SCALARS * 18 / BRAM_BLOCK_BITS),
        "integrated_bram_blocks_actual": resources["BSRAM"]["used"],
        "analytic_weight_macs_per_update": WEIGHT_MACS,
        "cxxrtl_lower_bound_cycles": workload_trace["cycles_after_start"],
        "actual_post_route_resources": {name: value["used"] for name, value in resources.items()},
        "actual_fmax_mhz_minimum": min(fmax_values),
        "latency_us_at_27mhz_lower_bound": workload_trace["cycles_after_start"] / TARGET_MHZ,
        "latency_us_at_minimum_fmax_lower_bound": workload_trace["cycles_after_start"] / min(fmax_values),
        "worst_case_latency_us": None,
        "worst_case_latency_reason": "nonfunctional workload omits dependencies, nonlinearities, and activation traffic, so only a lower bound exists",
        "physical_gain_retention": None,
        "gain_retention_scope": "not measured; functional action closeness is not physical gain retention",
        "deadline_pass": False, "capacity_pass": True, "enhanced_route_eligible": False,
        "disqualification_reasons": [
            "72,854-cycle optimistic lower bound exceeds 5 us",
            "synthesized RTL is not a functional GRU",
            "quantized physical gain retention was not established",
        ],
    }
    student_selected = student_pareto["selection"]
    student_bits = sum(int(row["entries"]) for row in student_manifest["files"]) * int(student_manifest["word_bits"])
    student_row = {
        "candidate_id": "distilled_student_q3_14_state4_serial", "role": "selected_default_mainline",
        "functional_model": True, "weight_bits": 18, "bias_bits": 18,
        "stored_parameters": sum(int(row["entries"]) for row in student_manifest["files"]),
        "parameter_storage_bits": student_bits,
        "integrated_bram_blocks_actual": int(student_selected["measured_resources"]["BSRAM"]),
        "analytic_macs_per_update_parent": 87, "rtl_multiplications_per_update": 64,
        "cxxrtl_cycles": int(student_selected["student_latency_cycles"]),
        "actual_post_route_resources": student_selected["measured_resources"],
        "actual_fmax_mhz_minimum": float(student_selected["measured_fmax_mhz_minimum"]),
        "worst_case_latency_us_at_27mhz": float(student_selected["student_latency_us_at_27mhz"]),
        "physical_gain_retention": student_retention,
        "deadline_pass": True, "capacity_pass": True, "enhanced_route_eligible": True,
    }
    source_bindings = [_binding(path) for path in (
        RUNNER, CHECKPOINT, QUANT_MANIFEST, STUDENT_MANIFEST, CORE_RTL, CORE_TOP,
        QUANT_RTL, QUANT_TOP, CXXRTL_DRIVER, SDC, CST,
    )]
    source_bindings.extend((
        _binding(ROOT / manifest["weight_file"]["path"]),
        _binding(ROOT / manifest["bias_file"]["path"]),
    ))
    report: dict[str, Any] = {
        "schema_version": "t5.5.4-gru-student-hardware-feasibility-v1",
        "task_id": "T5.5.4", "status": "PENDING", "verdict": VERDICT,
        "target": {
            "device": DEVICE, "family": FAMILY, "target_mhz": TARGET_MHZ,
            "student_deadline_us": STUDENT_DEADLINE_US,
            "bram_block_bits": BRAM_BLOCK_BITS, "bram_blocks_available": BRAM_BLOCKS_AVAILABLE,
            "mult9_available": MULT9_AVAILABLE,
        },
        "parents": [teacher_row, gain_row, student_eq_row, student_pareto_row],
        "source_bindings": source_bindings,
        "parameter_accounting": {
            "architecture": manifest["architecture"], "total_parameters": TOTAL_PARAMETERS,
            "weight_macs": WEIGHT_MACS, "bias_scalars": BIAS_SCALARS,
            "checkpoint_selected_restart_index": manifest["selected_restart_index"],
            "checkpoint_selected_state_sha256": manifest["selected_state_sha256"],
        },
        "quantized_functional_shadow": shadow,
        "quantized_lower_bound_synthesis": parse_synthesis_memory(synthesis_path),
        "quantized_lower_bound_toolchain_provenance": toolchain_provenance,
        "quantized_lower_bound_cxxrtl": workload_trace,
        "quantized_lower_bound_place_route": routes,
        "quantized_lower_bound_resources": resources,
        "quantized_lower_bound_fmax_mhz": {
            "minimum": min(fmax_values), "median": statistics.median(fmax_values), "maximum": max(fmax_values),
        },
        "candidates": [*full_rows, quant_row, student_row],
        "selection": {
            "candidate_id": student_row["candidate_id"],
            "rule": "require functional implementation, target capacity, <=5 us actual/lower-bound deadline, and measured physical gain retention",
            "full_gru_route": "offline_teacher_only",
            "quantized_gru_enhanced_route": "dropped",
            "distilled_student_route": "selected",
        },
        "durable_artifacts": durable,
        "tool_versions": {
            "python": sys.version.split()[0], "numpy": np.__version__,
            "torch": importlib.metadata.version("torch"),
            "yowasp_yosys": importlib.metadata.version("yowasp-yosys"),
            "yowasp_nextpnr_himbaechel_gowin": importlib.metadata.version("yowasp-nextpnr-himbaechel-gowin"),
        },
        "evidence_boundary": {
            "full_gru_parameter_and_mac_accounting": True,
            "full_gru_target_synthesis": False,
            "quantized_gru_functional_fake_quantization": True,
            "quantized_gru_functional_rtl": False,
            "quantized_gru_lower_bound_target_post_route": True,
            "quantized_gru_physical_gain_retention": False,
            "student_cxxrtl_equivalence": True,
            "student_target_post_route": True,
            "student_physical_gain_retention": True,
            "vendor_timing_signoff": False, "bitstream_generated": False,
            "board_measured": False, "quantum_hardware_measured": False,
        },
    }
    report["gates"] = evaluate_gates(report)
    report["mutation_audit"] = mutation_audit(report)
    report["gates"]["all_shortcut_mutations_are_rejected"] = all(row["rejected"] for row in report["mutation_audit"])
    report["gate_summary"] = {"passed": sum(report["gates"].values()), "total": len(report["gates"])}
    report["status"] = "PASS" if all(report["gates"].values()) else "FAIL"
    return report


def evaluate_gates(report: dict[str, Any]) -> dict[str, bool]:
    parents_ok = all(
        row["status"] == "PASS" and row["gate_summary"]["passed"] == row["gate_summary"]["total"]
        and _matches(row["artifact"]) for row in report["parents"]
    )
    sources_ok = all(_matches(row) for row in report["source_bindings"])
    artifacts_ok = all(_matches(row) for row in report["durable_artifacts"])
    provenance = report["quantized_lower_bound_toolchain_provenance"]
    provenance_ok = (
        all(_matches(row) for row in provenance["input_bindings"])
        and _matches(provenance["compressed_netlist_artifact"])
        and _matches(provenance["synthesis_log_artifact"])
        and all(_matches(row[key]) for row in provenance["route_artifacts"] for key in ("report", "log"))
        and [row["seed"] for row in provenance["route_artifacts"]] == list(SEEDS)
    )
    accounting = report["parameter_accounting"]
    count_ok = accounting["total_parameters"] == 72_853 and accounting["weight_macs"] == 72_266 and accounting["bias_scalars"] == 587
    shadow = report["quantized_functional_shadow"]
    shadow_ok = (
        shadow["torch_grucell_equation_max_abs_error"] <= 1e-12
        and shadow["exhaustive_length8_histories_all_prefixes"]["action_maximum_absolute_error"] <= 5e-3
        and shadow["long_random_sequences"]["action_maximum_absolute_error"] <= 5e-3
        and shadow["exhaustive_length8_histories_all_prefixes"]["all_quantized_actions_bounded"]
        and shadow["long_random_sequences"]["all_quantized_values_finite"]
        and shadow["physical_gain_retention"] is None
    )
    full = report["candidates"][:2]
    full_fail_closed = all(
        row["integrated_bram_blocks_lower_bound"] > report["target"]["bram_blocks_available"]
        and row["actual_target_synthesis"] is False and row["enhanced_route_eligible"] is False
        for row in full
    )
    synthesis = report["quantized_lower_bound_synthesis"]
    synthesis_ok = synthesis["complete_parameter_rom_primitives"] and synthesis["fast_path_bram_primitives"]
    trace = report["quantized_lower_bound_cxxrtl"]
    trace_ok = (
        trace["zero_structural_problems"] and trace["cycles_after_start"] == WORKLOAD_CYCLES
        and trace["weight_macs_completed"] == WEIGHT_MACS and trace["biases_consumed"] == BIAS_SCALARS
        and trace["done"] and not trace["busy"]
        and trace["signature_matches_independent_reference"]
        and trace["independent_reference"]["ordered_weight_entries"] == WEIGHT_MACS
        and trace["independent_reference"]["ordered_bias_entries"] == BIAS_SCALARS
    )
    routes = report["quantized_lower_bound_place_route"]
    route_ok = (
        [row["seed"] for row in routes] == list(SEEDS)
        and all(row["route_status"] == "PASS" and row["timing_pass"] for row in routes)
        and min(row["achieved_fmax_mhz"] for row in routes) >= TARGET_MHZ
    )
    resources = report["quantized_lower_bound_resources"]
    resource_ok = (
        resources["BSRAM"]["used"] == 41 and resources["BSRAM"]["available"] == 46
        and all(value["used"] <= value["available"] for value in resources.values())
    )
    quant = report["candidates"][2]
    quant_fail_closed = (
        not quant["functional_model"] and quant["capacity_pass"] and not quant["deadline_pass"]
        and quant["latency_us_at_minimum_fmax_lower_bound"] > STUDENT_DEADLINE_US
        and quant["physical_gain_retention"] is None and not quant["enhanced_route_eligible"]
        and quant["worst_case_latency_us"] is None
    )
    student = report["candidates"][3]
    student_ok = (
        student["functional_model"] and student["deadline_pass"] and student["capacity_pass"]
        and student["cxxrtl_cycles"] == 64 and student["worst_case_latency_us_at_27mhz"] <= STUDENT_DEADLINE_US
        and student["actual_post_route_resources"]["BSRAM"] == 8
        and student["physical_gain_retention"]["minimum_point"] >= student["physical_gain_retention"]["threshold"]
        and student["physical_gain_retention"]["minimum_ci_lower"] >= student["physical_gain_retention"]["threshold"]
    )
    selection_ok = (
        report["selection"]["candidate_id"] == "distilled_student_q3_14_state4_serial"
        and report["selection"]["quantized_gru_enhanced_route"] == "dropped"
        and [row["candidate_id"] for row in report["candidates"] if row["enhanced_route_eligible"]]
        == ["distilled_student_q3_14_state4_serial"]
    )
    boundary = report["evidence_boundary"]
    boundary_ok = (
        boundary["quantized_gru_lower_bound_target_post_route"]
        and not boundary["quantized_gru_functional_rtl"]
        and not boundary["quantized_gru_physical_gain_retention"]
        and boundary["student_physical_gain_retention"]
        and not boundary["vendor_timing_signoff"] and not boundary["board_measured"]
    )
    return {
        "all_four_parent_artifacts_are_hash_bound_current_passes": parents_ok,
        "all_checkpoint_rtl_memory_driver_and_constraint_sources_are_hash_bound": sources_ok,
        "gru_parameter_bias_and_mac_accounting_is_exact": count_ok,
        "quantized_functional_shadow_matches_grucell_and_stays_bounded_but_has_no_gain_claim": shadow_ok,
        "float32_and_float64_full_gru_fail_target_storage_before_synthesis": full_fail_closed,
        "quantized_lower_bound_synthesis_preserves_all_33_parameter_and_8_core_brams": synthesis_ok,
        "cxxrtl_consumes_every_weight_and_bias_in_exactly_72854_cycles": trace_ok,
        "three_quantized_lower_bound_routes_pass_27mhz": route_ok,
        "quantized_lower_bound_fits_but_uses_41_of_46_bram_blocks": resource_ok,
        "nonfunctional_quantized_lower_bound_is_disqualified_by_deadline_and_missing_gain": quant_fail_closed,
        "functional_student_meets_resource_deadline_and_physical_retention_gates": student_ok,
        "distilled_student_is_the_unique_eligible_hardware_route": selection_ok,
        "all_durable_tool_and_trace_artifacts_match_hashes": artifacts_ok,
        "toolchain_execution_binds_current_inputs_netlist_and_route_artifacts": provenance_ok,
        "post_route_lower_bound_is_not_mislabeled_as_functional_vendor_or_board_evidence": boundary_ok,
    }


def mutation_audit(report: dict[str, Any]) -> list[dict[str, Any]]:
    mutations = [
        ("break_parent_hash", lambda row: row["parents"][0]["artifact"].__setitem__("sha256", "0" * 64)),
        ("hide_full_gru_storage_failure", lambda row: row["candidates"][0].__setitem__("integrated_bram_blocks_lower_bound", 46)),
        ("skip_one_weight", lambda row: row["quantized_lower_bound_cxxrtl"].__setitem__("weight_macs_completed", 72_265)),
        ("corrupt_parameter_order_signature", lambda row: row["quantized_lower_bound_cxxrtl"].__setitem__("signature_matches_independent_reference", False)),
        ("shorten_workload_latency", lambda row: row["quantized_lower_bound_cxxrtl"].__setitem__("cycles_after_start", 100)),
        ("drop_parameter_rom", lambda row: row["quantized_lower_bound_synthesis"].__setitem__("complete_parameter_rom_primitives", False)),
        ("break_toolchain_input_provenance", lambda row: row["quantized_lower_bound_toolchain_provenance"]["input_bindings"][0].__setitem__("sha256", "0" * 64)),
        ("miss_timing", lambda row: row["quantized_lower_bound_place_route"][0].__setitem__("achieved_fmax_mhz", 26.0)),
        ("invent_quantized_gain", lambda row: row["candidates"][2].__setitem__("physical_gain_retention", 1.0)),
        ("promote_nonfunctional_gru", lambda row: row["candidates"][2].__setitem__("enhanced_route_eligible", True)),
        ("break_student_deadline", lambda row: row["candidates"][3].__setitem__("worst_case_latency_us_at_27mhz", 6.0)),
        ("claim_board_measurement", lambda row: row["evidence_boundary"].__setitem__("board_measured", True)),
    ]
    rows = []
    for name, mutate in mutations:
        candidate = copy.deepcopy(report)
        mutate(candidate)
        gates = evaluate_gates(candidate)
        rows.append({"mutation": name, "rejected": not all(gates.values()), "failed_gates": [key for key, value in gates.items() if not value]})
    return rows


def write_outputs(report: dict[str, Any], json_path: Path, csv_path: Path) -> None:
    fields = (
        "candidate_id", "role", "functional_model", "stored_parameters", "parameter_storage_bits",
        "integrated_bram_blocks", "analytic_macs", "latency_cycles", "latency_us_at_27mhz",
        "actual_fmax_mhz_minimum", "actual_lut4", "actual_dff", "actual_bram", "actual_mult18", "actual_mult9",
        "physical_gain_retention_point_minimum", "physical_gain_retention_ci_lower_minimum",
        "capacity_pass", "deadline_pass", "enhanced_route_eligible",
    )
    rows = []
    for row in report["candidates"]:
        actual = row.get("actual_post_route_resources") or {}
        retention = row.get("physical_gain_retention")
        retention_map = retention if isinstance(retention, dict) else {}
        rows.append({
            "candidate_id": row["candidate_id"], "role": row["role"], "functional_model": row["functional_model"],
            "stored_parameters": row["stored_parameters"], "parameter_storage_bits": row["parameter_storage_bits"],
            "integrated_bram_blocks": row.get("integrated_bram_blocks_actual", row.get("integrated_bram_blocks_lower_bound")),
            "analytic_macs": row.get("analytic_weight_macs_per_update", row.get("analytic_macs_per_update_parent")),
            "latency_cycles": row.get("cxxrtl_lower_bound_cycles", row.get("cxxrtl_cycles", row.get("optimistic_96_multiplier_cycle_lower_bound"))),
            "latency_us_at_27mhz": row.get("latency_us_at_27mhz_lower_bound", row.get("worst_case_latency_us_at_27mhz", row.get("optimistic_latency_us_at_27mhz_lower_bound"))),
            "actual_fmax_mhz_minimum": row.get("actual_fmax_mhz_minimum"),
            "actual_lut4": actual.get("LUT4"), "actual_dff": actual.get("DFF"), "actual_bram": actual.get("BSRAM"),
            "actual_mult18": actual.get("MULT18X18"), "actual_mult9": actual.get("MULT9X9"),
            "physical_gain_retention_point_minimum": retention_map.get("minimum_point"),
            "physical_gain_retention_ci_lower_minimum": retention_map.get("minimum_ci_lower"),
            "capacity_pass": row["capacity_pass"], "deadline_pass": row["deadline_pass"],
            "enhanced_route_eligible": row["enhanced_route_eligible"],
        })
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    report["source_data"] = {"path": _relative(csv_path), "sha256": _sha256(csv_path), "candidate_rows": len(rows)}
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD)
    parser.add_argument("--cxxrtl-build-dir", type=Path, default=DEFAULT_CXXRTL_BUILD)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--run-toolchain", action="store_true")
    args = parser.parse_args(argv)
    if args.run_toolchain:
        run_toolchain(args.build_dir, discover_tools())
    report = build_report(args.build_dir, args.cxxrtl_build_dir)
    write_outputs(report, args.output_json, args.output_csv)
    print(json.dumps({
        "status": report["status"], "verdict": report["verdict"],
        "gate_summary": report["gate_summary"], "selection": report["selection"],
    }, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
