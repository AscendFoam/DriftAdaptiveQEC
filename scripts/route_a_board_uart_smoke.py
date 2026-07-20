"""Run the first framed-UART smoke against a programmed Route-A board.

This command verifies transport integrity and the on-chip cycle stamp.  Host
round-trip time is diagnostic only and is not accepted as source-to-action
latency; T6.1.3 still requires the frozen board timestamp/strobe method.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cnn_fpga.hwio.route_a_uart_protocol import (
    CMD_EXECUTE,
    CMD_STATUS,
    RESPONSE_BYTES,
    STATUS_DUPLICATE_REPLAY,
    STATUS_OK,
    RouteAInputs,
    decode_response,
    encode_request,
)
from cnn_fpga.runtime.fast_production_core_reference import encode_fast_input_word


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _transaction(port: object, frame: bytes, timeout_s: float) -> tuple[object, int]:
    started = time.perf_counter_ns()
    port.reset_input_buffer()
    written = port.write(frame)
    port.flush()
    if written != len(frame):
        raise RuntimeError(f"short UART write: {written}/{len(frame)}")
    response = bytearray()
    deadline = time.monotonic() + timeout_s
    while len(response) < RESPONSE_BYTES and time.monotonic() < deadline:
        chunk = port.read(RESPONSE_BYTES - len(response))
        if chunk:
            response.extend(chunk)
    elapsed_ns = time.perf_counter_ns() - started
    if len(response) != RESPONSE_BYTES:
        raise TimeoutError(f"short UART response: {len(response)}/{RESPONSE_BYTES}")
    return decode_response(bytes(response)), elapsed_ns


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True, help="Explicit COM port; never auto-selected")
    parser.add_argument("--baud", type=int, default=3_000_000)
    parser.add_argument("--timeout", type=float, default=2.0)
    parser.add_argument("--sequence", type=int, default=0)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        import serial  # type: ignore
    except ImportError as exc:
        raise SystemExit("pyserial is required for physical UART smoke") from exc

    manifest = json.loads(args.candidate_manifest.read_text(encoding="utf-8"))
    if manifest.get("verdict") != "PASS_PREBOARD_CANDIDATE_NOT_PHYSICAL_QUALIFICATION":
        raise SystemExit("candidate manifest is not a passing preboard candidate")

    input_word = encode_fast_input_word(
        syndrome_code=0x155,
        syndrome_x_code=0,
        syndrome_z_code=0,
        phase=0,
        ood_score=0,
        parameter_age=0,
    )
    execute = encode_request(
        args.sequence,
        CMD_EXECUTE,
        RouteAInputs(in_word=input_word, posterior_valid=True, p_normal=255),
    )
    status = encode_request(args.sequence + 1, CMD_STATUS)

    with serial.Serial(
        port=args.port,
        baudrate=args.baud,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        timeout=0.05,
        write_timeout=args.timeout,
        exclusive=True if platform.system() != "Windows" else None,
    ) as uart:
        response0, execute_wall_ns = _transaction(uart, execute, args.timeout)
        duplicate, duplicate_wall_ns = _transaction(uart, execute, args.timeout)
        status_response, status_wall_ns = _transaction(uart, status, args.timeout)

    failures: list[str] = []
    if response0.status != STATUS_OK or response0.sequence != args.sequence:
        failures.append("nominal_execute_status_or_sequence")
    if response0.action_latency_cycles != 7:
        failures.append("onchip_execute_latency_not_7_cycles")
    if (response0.route_action_word & 1) == 0:
        failures.append("nominal_route_action_not_valid")
    if duplicate.status != STATUS_DUPLICATE_REPLAY:
        failures.append("duplicate_not_idempotently_rejected")
    if duplicate.route_action_word & 1:
        failures.append("duplicate_reexecuted_action")
    if status_response.status != STATUS_OK or status_response.sequence != args.sequence + 1:
        failures.append("post_duplicate_sequence_not_recovered")
    counters = {
        "rx_crc_errors": status_response.rx_crc_errors,
        "sequence_errors": status_response.sequence_errors,
        "uart_framing_errors": status_response.uart_framing_errors,
    }
    if any(counters.values()):
        failures.append("nominal_transport_counter_nonzero")

    report = {
        "schema_version": "t6.9.2-route-a-physical-uart-smoke-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "verdict": "PASS_UART_SMOKE_NOT_FULL_BOARD_QUALIFICATION" if not failures else "FAIL_UART_SMOKE",
        "port": args.port,
        "baud": args.baud,
        "candidate_manifest": str(args.candidate_manifest.resolve()),
        "candidate_manifest_sha256": _sha256(args.candidate_manifest),
        "sequence_start": args.sequence,
        "execute": {
            "onchip_action_latency_cycles": response0.action_latency_cycles,
            "route_action_word": response0.route_action_word,
            "host_roundtrip_ns_diagnostic_only": execute_wall_ns,
        },
        "duplicate_status": duplicate.status,
        "host_roundtrip_ns_diagnostic_only": {
            "duplicate": duplicate_wall_ns,
            "status": status_wall_ns,
        },
        "transport_counters": counters,
        "failures": failures,
        "claim_boundary": (
            "Passing this smoke proves framed UART exchange on the named port only. "
            "It does not prove source-to-action latency, deadline miss rate, power, "
            "or million-cycle HIL qualification."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
