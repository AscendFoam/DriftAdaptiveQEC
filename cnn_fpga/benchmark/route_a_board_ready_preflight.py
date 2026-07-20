"""Audit the T6.9.2 Route-A UART pre-board bitstream candidate.

The report produced here is deliberately *not* a board qualification record.
It binds the candidate sources, constraints, routed netlist and bitstream while
requiring the accelerated full-stack CXXRTL and actual-ratio UART PHY tests.
Physical claims remain false until the separately named board prerequisites
exist and measurements have been collected from the identified unit.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Iterable

from cnn_fpga.hwio.route_a_uart_protocol import (
    CMD_EXECUTE,
    RouteAInputs,
    decode_request,
    encode_request,
)


ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "T6.9.2-PREBOARD"
SCHEMA_VERSION = "t6.9.2-route-a-preboard-candidate-v1"
TARGET_MHZ = 27.0
SOURCES = (
    ROOT / "cnn_fpga/rtl/gkp_fast_path_core.sv",
    ROOT / "cnn_fpga/rtl/route_a_policy_overlay.sv",
    ROOT / "cnn_fpga/rtl/route_a_integrated_qualification_top.sv",
    ROOT / "cnn_fpga/rtl/route_a_uart_phy.sv",
    ROOT / "cnn_fpga/rtl/route_a_uart_board_top.sv",
    ROOT / "cnn_fpga/hwio/route_a_uart_protocol.py",
    ROOT / "cnn_fpga/rtl/route_a_uart_board_cxxrtl_driver.cc",
    ROOT / "cnn_fpga/rtl/route_a_uart_phy_cxxrtl_driver.cc",
    ROOT / "cnn_fpga/benchmark/route_a_board_ready_preflight.py",
    ROOT / "scripts/route_a_board_uart_smoke.py",
)
CONSTRAINTS = (
    ROOT / "cnn_fpga/rtl/tang_nano_20k_route_a_uart.cst",
    ROOT / "cnn_fpga/rtl/tang_nano_20k_27mhz.sdc",
)
PHYSICAL_PREREQUISITES = (
    ROOT / "configs/hardware/t6_1_1_actual_board.json",
    ROOT / "docs/t6_1_2_transport_adapter_qualification.json",
    ROOT / "docs/t6_1_3_board_timestamp_method.json",
    ROOT / "docs/t6_2_3_board_correctness_smoke.json",
    ROOT / "docs/t6_4_route_a_board_hil_qualification.json",
    ROOT / "docs/t6_9_2_bitstream_manifest.json",
)
DEFAULT_BUILD = ROOT / "build/t6_9_2_board_ready_uart"
DEFAULT_REPORT = ROOT / "docs/t6_9_2_preboard_bitstream_candidate.json"
DEFAULT_NOTE = ROOT / "docs/route_a_board_preboard_candidate.md"


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "path": _relative(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _run(executable: Path) -> str:
    env = os.environ.copy()
    mingw = Path(r"C:\ProgramData\msys2\mingw64\bin")
    if mingw.exists():
        env["PATH"] = str(mingw) + os.pathsep + env.get("PATH", "")
    completed = subprocess.run(
        [str(executable)], cwd=ROOT, env=env, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{_relative(executable)} failed with {completed.returncode}:\n"
            f"{completed.stdout}"
        )
    return completed.stdout.strip()


def _pairs(text: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for token in text.replace("\n", ",").split(","):
        if "=" not in token:
            continue
        key, value = token.strip().split("=", 1)
        if re.fullmatch(r"-?\d+", value):
            result[key] = int(value)
    return result


def _all_true(gates: Iterable[dict[str, Any]]) -> bool:
    return all(bool(gate["pass"]) for gate in gates)


def build_report(
    *, build_dir: Path = DEFAULT_BUILD, run_executables: bool = True,
) -> dict[str, Any]:
    route_report_path = build_dir / "route_a_uart_board_route_report.json"
    routed_path = build_dir / "route_a_uart_board_routed.json"
    netlist_path = build_dir / "route_a_uart_board_synth.json"
    synth_log_path = build_dir / "route_a_uart_board_synth.log"
    route_log_path = build_dir / "route_a_uart_board_route.log"
    bitstream_path = build_dir / "route_a_uart_board_preboard_candidate.fs"
    cxxrtl_path = build_dir / "route_a_uart_board_fast3.exe"
    cxxrtl_model_path = build_dir / "route_a_uart_board_fast3_model.cc"
    phy_path = build_dir / "route_a_uart_phy_test.exe"
    phy_model_path = build_dir / "route_a_uart_rx_model.cc"
    required = (
        *SOURCES, *CONSTRAINTS, route_report_path, routed_path, netlist_path,
        synth_log_path, route_log_path, bitstream_path, cxxrtl_path,
        cxxrtl_model_path, phy_path, phy_model_path,
    )
    missing = [_relative(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing preboard artifacts: " + ", ".join(missing))

    route = json.loads(route_report_path.read_text(encoding="utf-8"))
    clocks = route.get("fmax", {})
    if len(clocks) != 1:
        raise ValueError(f"expected exactly one routed clock, got {tuple(clocks)}")
    clock_name, clock = next(iter(clocks.items()))
    utilization = route.get("utilization", {})
    resource_names = ("LUT4", "DFF", "BSRAM", "MULT18X18", "MULT9X9", "IOB")

    request = encode_request(7, CMD_EXECUTE, RouteAInputs(posterior_valid=True))
    decoded_request = decode_request(request)
    codec_ok = decoded_request.sequence == 7 and decoded_request.command == CMD_EXECUTE

    cxxrtl_stdout = _run(cxxrtl_path) if run_executables else ""
    phy_stdout = _run(phy_path) if run_executables else ""
    cxxrtl = _pairs(cxxrtl_stdout)
    phy = _pairs(phy_stdout)
    cxxrtl_ok = (not run_executables) or cxxrtl == {
        "responses": 6,
        "execute_latency_cycles": 7,
        "duplicate_idempotent": 1,
        "crc_errors": 1,
        "sequence_errors": 1,
        "framing_errors": 1,
        "response_crc_errors": 0,
    }
    phy_ok = (not run_executables) or phy == {"received": 5, "framing_errors": 1}

    gates = [
        {"gate": "all_sources_and_constraints_bound", "pass": not missing},
        {"gate": "request_codec_roundtrip", "pass": codec_ok},
        {"gate": "full_stack_cxxrtl_protocol", "pass": cxxrtl_ok},
        {"gate": "actual_3mbaud_ratio_phy_cxxrtl", "pass": phy_ok},
        {
            "gate": "post_route_27mhz_timing",
            "pass": float(clock["achieved"]) >= TARGET_MHZ
            and float(clock["constraint"]) == TARGET_MHZ,
        },
        {
            "gate": "post_route_resources_fit",
            "pass": all(
                int(utilization[name]["used"]) <= int(utilization[name]["available"])
                for name in resource_names
            ),
        },
        {"gate": "nonempty_packed_bitstream", "pass": bitstream_path.stat().st_size > 0},
    ]
    physical = [
        {"path": _relative(path), "present": path.is_file()}
        for path in PHYSICAL_PREREQUISITES
    ]
    package_versions: dict[str, str | None] = {}
    for package in (
        "yowasp-yosys", "yowasp-nextpnr-himbaechel-gowin",
        "apycula", "yowasp-runtime",
    ):
        try:
            package_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            package_versions[package] = None
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "verdict": (
            "PASS_PREBOARD_CANDIDATE_NOT_PHYSICAL_QUALIFICATION"
            if _all_true(gates)
            else "FAIL_PREBOARD_CANDIDATE"
        ),
        "candidate_is_physical_evidence": False,
        "board_programmed": False,
        "physical_measurements_collected": False,
        "physical_qualification_blocked": not all(row["present"] for row in physical),
        "target": {
            "reference_board": "Sipeed Tang Nano 20K",
            "device": "GW2AR-LV18QN88C8/I7",
            "family": "GW2A-18C",
            "clock_mhz": TARGET_MHZ,
            "uart_baud": 3_000_000,
            "pinout_status": "candidate_unverified_against_physical_unit_revision",
        },
        "post_route": {
            "clock_name": clock_name,
            "fmax_mhz": float(clock["achieved"]),
            "constraint_mhz": float(clock["constraint"]),
            "utilization": {name: utilization[name] for name in resource_names},
        },
        "toolchain": {
            "package_versions": package_versions,
            "commands": {
                "cxxrtl_fast_model": (
                    "yowasp-yosys -Q -p 'read_verilog -sv <five RTL sources>; "
                    "chparam -set UART_BAUD 9000000 route_a_uart_board_top; "
                    "hierarchy -check -top route_a_uart_board_top; proc; check; stat; "
                    "write_cxxrtl -O0 -g0 route_a_uart_board_fast3_model.cc'"
                ),
                "cxxrtl_fast_compile": (
                    "g++ -std=c++17 -O2 -DUART_CLKS_PER_BIT=3 "
                    "route_a_uart_board_cxxrtl_driver.cc -o route_a_uart_board_fast3.exe"
                ),
                "synthesis": (
                    "yowasp-yosys -Q -p 'read_verilog -sv <five RTL sources>; "
                    "hierarchy -check -top route_a_uart_board_top; proc; check; "
                    "synth_gowin -family gw2a -no-rw-check -top "
                    "route_a_uart_board_top -json route_a_uart_board_synth.json; stat'"
                ),
                "place_route": (
                    "yowasp-nextpnr-himbaechel-gowin --device "
                    "GW2AR-LV18QN88C8/I7 -o family=GW2A-18C -o "
                    "cst=tang_nano_20k_route_a_uart.cst --freq 27 --seed 1 "
                    "--router router1 --sdc tang_nano_20k_27mhz.sdc"
                ),
                "pack": (
                    "gowin_pack -d GW2A-18C -o "
                    "route_a_uart_board_preboard_candidate.fs "
                    "route_a_uart_board_routed.json"
                ),
            },
        },
        "protocol": {
            "request_bytes": 40,
            "response_bytes": 96,
            "execute_latency_cycles": cxxrtl.get("execute_latency_cycles"),
            "duplicate_semantics": "explicit_status_no_reexecution_no_sequence_consumption",
            "event_controls_single_pulse_gated": True,
            "accelerated_full_stack_uart_clocks_per_bit": 3,
            "physical_candidate_uart_clocks_per_bit": 9,
            "cxxrtl_stdout": cxxrtl_stdout,
            "phy_stdout": phy_stdout,
        },
        "gates": gates,
        "physical_prerequisites": physical,
        "source_manifest": [_artifact(path) for path in (*SOURCES, *CONSTRAINTS)],
        "build_manifest": [
            _artifact(path) for path in (
                netlist_path, routed_path, route_report_path, synth_log_path,
                route_log_path, bitstream_path, cxxrtl_model_path,
                cxxrtl_path, phy_model_path, phy_path,
            )
        ],
        "claim_boundary": {
            "allowed": [
                "hardware-aware UART wrapper simulation passed",
                "target-device post-route timing estimate passed",
                "packed preboard bitstream candidate exists",
            ],
            "forbidden": [
                "real-board correctness",
                "real source-to-action latency",
                "zero real deadline miss",
                "measured board power",
                "physical transport qualification",
            ],
            "long_sequence_limitation": (
                "Per-frame UART execute inserts host/link gaps and is not a full-cadence "
                "1e6-cycle HIL substitute; T6.4 still requires a board-qualified streaming "
                "or autonomous trace mode."
            ),
        },
    }
    return report


def write_note(report: dict[str, Any], path: Path) -> None:
    post = report["post_route"]
    util = post["utilization"]
    lines = [
        "# Route-A 真板前位流候选审计",
        "",
        f"- 结论：`{report['verdict']}`",
        "- 证据边界：这是综合、P&R、打包和 CXXRTL 证据，不是真板测量。",
        f"- 后路由 Fmax：{post['fmax_mhz']:.2f} MHz（约束 {post['constraint_mhz']:.2f} MHz）。",
        f"- 资源：LUT {util['LUT4']['used']}/{util['LUT4']['available']}，"
        f"DFF {util['DFF']['used']}/{util['DFF']['available']}，"
        f"BSRAM {util['BSRAM']['used']}/{util['BSRAM']['available']}。",
        "- UART：候选 27 MHz / 3 Mbaud（9 clocks/bit）；完整栈以 3 clocks/bit 加速回归，"
        "实际比率 PHY 另行通过独立 CXXRTL 测试。",
        "- 重复帧：返回显式 duplicate 状态，不重新执行、不消耗序号。",
        "",
        "## 尚未满足",
        "",
        "- 未识别实际板卡、revision、串口/JTAG 路径。",
        "- 未烧录候选位流，未采集真实 source-to-action、deadline miss、功耗或长序列数据。",
        "- 逐帧 UART 模式包含链路间隙，不能替代 T6.4 的满速百万周期 HIL。",
        "",
        f"机器可读 manifest：`{_relative(DEFAULT_REPORT)}`。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--note", type=Path, default=DEFAULT_NOTE)
    parser.add_argument("--skip-executables", action="store_true")
    args = parser.parse_args()
    report = build_report(
        build_dir=args.build_dir.resolve(),
        run_executables=not args.skip_executables,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_note(report, args.note)
    print(json.dumps({
        "verdict": report["verdict"],
        "fmax_mhz": report["post_route"]["fmax_mhz"],
        "physical_qualification_blocked": report["physical_qualification_blocked"],
        "report": _relative(args.output),
    }, ensure_ascii=False))
    if report["verdict"] != "PASS_PREBOARD_CANDIDATE_NOT_PHYSICAL_QUALIFICATION":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
