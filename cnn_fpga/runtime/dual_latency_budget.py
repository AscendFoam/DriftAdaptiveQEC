"""Auditable dual latency budget for T2.4.1.

The contract deliberately keeps two non-composable lanes:

* timing facts/assumptions from external literature systems; and
* project-local cadence, software-model assumptions, capacity bounds and
  still-unmeasured target-board fields.

It is a provenance validator, not a latency simulator.  Backlog, jitter and
deadline dynamics belong to T2.4.2; fixed-point and LUT costs belong to T2.4.3.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from cnn_fpga.utils.config import load_yaml_config, save_json


SCHEMA_VERSION = "dual-latency-budget-v1"
TASK_ID = "T2.4.1"
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT = ROOT / "docs" / "dual_latency_budget.json"
DEFAULT_VALIDATION_ARTIFACT = ROOT / "docs" / "t2_4_1_dual_latency_budget_validation.json"

EVIDENCE_CLASSES = {
    "external_device_measurement",
    "external_model_assumption",
    "project_configuration_assumption",
    "project_capacity_lower_bound",
    "target_board_unmeasured",
    "not_integrated_not_measured",
}

REQUIRED_GATES = (
    "schema_and_task_identity",
    "two_lane_identity",
    "terminology_ledger_complete",
    "record_ids_unique",
    "evidence_classes_closed",
    "source_anchors_resolve",
    "external_facts_not_target_measurements",
    "sivak_control_wiring_scope",
    "sivak_measurement_breakdown",
    "sivak_measurement_aggregate_fail_closed",
    "sivak_sbs_scope_discrepancy",
    "sivak_reset_scope_discrepancy",
    "sivak_constituent_and_full_cycle_arithmetic",
    "puviani_model_half_and_full_cycle_arithmetic",
    "project_config_bindings",
    "project_slow_model_mean_arithmetic",
    "project_window_cadence_arithmetic",
    "uart_8n1_capacity_arithmetic",
    "uart_fails_twenty_ms_window",
    "target_board_frontend_fields_null",
    "target_board_latency_fields_null",
    "real_board_gate_fail_closed",
    "no_cross_lane_aggregate",
)


class BudgetValidationError(ValueError):
    """Raised when a dual-budget provenance or arithmetic gate fails."""


def load_budget(path: str | Path = DEFAULT_ARTIFACT) -> dict[str, Any]:
    artifact = Path(path)
    with artifact.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise BudgetValidationError("budget root must be a JSON object")
    return payload


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record_ids(value: Any) -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key.endswith("_id") and isinstance(child, str):
                found.append(child)
            found.extend(_record_ids(child))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for child in value:
            found.extend(_record_ids(child))
    return found


def _evidence_classes(value: Any) -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key == "evidence_class" and isinstance(child, str):
                found.append(child)
            found.extend(_evidence_classes(child))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for child in value:
            found.extend(_evidence_classes(child))
    return found


def _source_text(anchor: Mapping[str, Any], root: Path) -> str:
    source_path = root / str(anchor["source_path"])
    if not source_path.is_file():
        raise BudgetValidationError(f"source anchor path missing: {source_path}")
    lines = source_path.read_text(encoding="utf-8").splitlines()
    start = int(anchor["line_start"])
    end = int(anchor["line_end"])
    if start < 1 or end < start or end > len(lines):
        raise BudgetValidationError(
            f"invalid source line range {start}:{end} for {source_path} ({len(lines)} lines)"
        )
    return "\n".join(lines[start - 1 : end])


def _all_anchors(value: Any) -> list[Mapping[str, Any]]:
    anchors: list[Mapping[str, Any]] = []
    if isinstance(value, Mapping):
        if {"source_path", "line_start", "line_end", "expected_fragment"}.issubset(value):
            anchors.append(value)
        for child in value.values():
            anchors.extend(_all_anchors(child))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for child in value:
            anchors.extend(_all_anchors(child))
    return anchors


def _stage_map(stages: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {str(item["stage_id"]): item for item in stages}


def _find_by_id(items: Sequence[Mapping[str, Any]], key: str, value: str) -> Mapping[str, Any]:
    matches = [item for item in items if item.get(key) == value]
    if len(matches) != 1:
        raise BudgetValidationError(f"expected exactly one {key}={value!r}, got {len(matches)}")
    return matches[0]


def _close(left: float, right: float, *, atol: float = 1.0e-9) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=atol)


def audit_budget(
    payload: Mapping[str, Any] | None = None,
    *,
    artifact_path: str | Path = DEFAULT_ARTIFACT,
    root: str | Path = ROOT,
) -> dict[str, Any]:
    """Return all gate outcomes without hiding which contract failed."""

    data = copy.deepcopy(dict(payload)) if payload is not None else load_budget(artifact_path)
    repo_root = Path(root)
    gates: dict[str, bool] = {name: False for name in REQUIRED_GATES}
    failures: list[str] = []

    def gate(name: str, condition: bool, detail: str) -> None:
        gates[name] = bool(condition)
        if not condition:
            failures.append(f"{name}: {detail}")

    try:
        gate(
            "schema_and_task_identity",
            data.get("schema_version") == SCHEMA_VERSION and data.get("task_id") == TASK_ID,
            "schema_version/task_id drift",
        )
        lanes = data.get("lanes", {})
        gate(
            "two_lane_identity",
            set(lanes) == {"literature_system", "project_control_plane"}
            and lanes["literature_system"].get("composable_with_other_lane") is False
            and lanes["project_control_plane"].get("composable_with_other_lane") is False,
            "exactly two non-composable lanes are required",
        )
        canonical_terms = {row.get("canonical_term") for row in data.get("terminology_ledger", [])}
        gate(
            "terminology_ledger_complete",
            {
                "measurement chain",
                "ADC acquisition",
                "FPGA DSP",
                "waveform generation / DAC output",
                "transport latency",
                "on-chip core latency",
                "end-to-end digital replay latency",
                "physical action latency",
            }.issubset(canonical_terms),
            "canonical timing terms or their scope decisions are missing",
        )
        ids = _record_ids(data.get("lanes", {}))
        gate(
            "record_ids_unique",
            bool(ids) and len(ids) == len(set(ids)),
            "all nested *_id values must be nonempty and unique",
        )
        observed_classes = set(_evidence_classes(data.get("lanes", {})))
        gate(
            "evidence_classes_closed",
            bool(observed_classes) and observed_classes.issubset(EVIDENCE_CLASSES),
            f"unsupported evidence classes: {sorted(observed_classes - EVIDENCE_CLASSES)}",
        )
        anchors = _all_anchors(data.get("lanes", {}))
        anchor_ok = bool(anchors)
        for anchor in anchors:
            anchor_ok = anchor_ok and str(anchor["expected_fragment"]) in _source_text(anchor, repo_root)
        gate(
            "source_anchors_resolve",
            anchor_ok,
            "source path/range/expected fragment must resolve in the current worktree",
        )

        literature = lanes["literature_system"]
        systems = literature["systems"]
        external_flags = [
            record.get("measured_on_target_board")
            for record in systems
        ]
        gate(
            "external_facts_not_target_measurements",
            all(flag is False for flag in external_flags),
            "external literature may be measured on its own apparatus but never on the project target",
        )

        sivak = _find_by_id(systems, "system_id", "LIT-SIVAK-2023")
        gate(
            "sivak_control_wiring_scope",
            sivak["control_wiring"] == {
                "controller": "two X6-1000M control cards",
                "fpga": "Xilinx Virtex-6",
                "integrated_interfaces": ["DAC", "ADC", "DIO"],
                "dac_sample_rate_msps": 500,
                "dac_resolution_bits": 16,
            },
            "Sivak controller/DAC facts must retain their external apparatus identity",
        )
        sivak_budgets = sivak["budgets"]
        measurement = _find_by_id(sivak_budgets, "budget_id", "LIT-SIVAK-MEASUREMENT")
        measurement_stages = _stage_map(measurement["stages"])
        gate(
            "sivak_measurement_breakdown",
            {
                key: measurement_stages[stage_id]["duration_ns"]
                for key, stage_id in {
                    "readout_pulse": "LIT-SIVAK-MEAS-READOUT",
                    "signal_travel_delay": "LIT-SIVAK-MEAS-TRAVEL",
                    "acquisition": "LIT-SIVAK-MEAS-ACQUISITION",
                    "fpga_dsp": "LIT-SIVAK-MEAS-FPGA-DSP",
                    "bit_distribution": "LIT-SIVAK-MEAS-DISTRIBUTION",
                }.items()
            }
            == {
                "readout_pulse": 700,
                "signal_travel_delay": 300,
                "acquisition": 1400,
                "fpga_dsp": 332,
                "bit_distribution": 100,
            },
            "measurement stage values drifted from the supplement",
        )
        gate(
            "sivak_measurement_aggregate_fail_closed",
            measurement.get("aggregate_duration_ns") is None
            and measurement.get("aggregation_status") == "not_defined_due_to_overlap_semantics",
            "700 ns pulse and acquisition/travel stages must not be blindly summed",
        )
        sbs = _find_by_id(sivak_budgets, "budget_id", "LIT-SIVAK-SBS")
        gate(
            "sivak_sbs_scope_discrepancy",
            sbs["prose_sbs_ns"] == 1546
            and sum(sbs["table_layer_durations_ns"]) == 1548
            and sbs["table_sbs_block_ns"] == 1596
            and sbs["source_discrepancy_ns"] == 2,
            "1546 prose, 1548 layer sum and 1596 entered/exited block must remain distinct",
        )
        reset = _find_by_id(sivak_budgets, "budget_id", "LIT-SIVAK-RESET")
        gate(
            "sivak_reset_scope_discrepancy",
            reset["prose_subroutine_ns"] == 2332
            and sum(reset["table_block_durations_ns"]) == 2380
            and reset["table_block_ns"] == 2380,
            "2332 prose subroutine and 2380 entered/exited table block must remain distinct",
        )
        constituent = _find_by_id(sivak_budgets, "budget_id", "LIT-SIVAK-CONSTITUENT")
        constituent_sum = sum(stage["duration_ns"] for stage in constituent["stages"])
        gate(
            "sivak_constituent_and_full_cycle_arithmetic",
            constituent_sum == constituent["constituent_ns"] == 4924
            and constituent["full_xz_cycle_ns"] == 2 * constituent["constituent_ns"] == 9848,
            "Table S3 constituent/full-cycle arithmetic mismatch",
        )

        puviani = _find_by_id(systems, "system_id", "LIT-PUVIANI-2025-MODEL")
        puviani_budget = _find_by_id(puviani["budgets"], "budget_id", "LIT-PUVIANI-MODEL-CYCLE")
        half_sum = sum(stage["duration_us"] for stage in puviani_budget["half_cycle_stages"])
        gate(
            "puviani_model_half_and_full_cycle_arithmetic",
            puviani["evidence_class"] == "external_model_assumption"
            and _close(half_sum, puviani_budget["half_cycle_us"])
            and _close(puviani_budget["half_cycle_us"], 5.0)
            and _close(puviani_budget["full_cycle_us"], 10.0)
            and _close(puviani_budget["full_cycle_us"], 2 * puviani_budget["half_cycle_us"]),
            "Puviani timing is a 5 us half / 10 us model cycle, not a hardware measurement",
        )

        project = lanes["project_control_plane"]
        cfg = load_yaml_config(repo_root / project["configuration_binding"]["source_path"])
        cadence = project["cadence"]
        stages = project["software_latency_model"]
        config_ok = (
            cadence["fast_cycle_period_us"] == cfg["runtime"]["t_fast_us"]
            and cadence["window_size_samples"] == cfg["runtime"]["window_size"]
            and cadence["window_stride_cycles"] == cfg["runtime"]["window_stride"]
            and cadence["slow_update_period_ms"] == cfg["runtime"]["t_slow_update_ms"]
            and cadence["fast_action_budget_us"] == cfg["timing"]["fast_cycle_budget_us"]
            and cadence["slow_job_budget_us"] == cfg["timing"]["slow_update_budget_us"]
            and stages["fast_path"]["mean_us"] == cfg["fast_path_model"]["latency_mean_us"]
            and stages["fast_path"]["std_us"] == cfg["fast_path_model"]["latency_std_us"]
        )
        for stage in stages["slow_path_stages"]:
            prefix = stage["config_prefix"]
            config_ok = config_ok and stage["mean_us"] == cfg["latency_model"][f"{prefix}_mean_us"]
            config_ok = config_ok and stage["std_us"] == cfg["latency_model"][f"{prefix}_std_us"]
        gate("project_config_bindings", config_ok, "project budget drifted from hardware_hil.yaml")
        mean_sum = sum(stage["mean_us"] for stage in stages["slow_path_stages"])
        gate(
            "project_slow_model_mean_arithmetic",
            _close(mean_sum, stages["slow_path_mean_total_us"]) and _close(mean_sum, 995.0),
            "slow-path modeled means must sum to 995 us",
        )
        gate(
            "project_window_cadence_arithmetic",
            _close(
                cadence["window_content_duration_ms"],
                cadence["window_size_samples"] * cadence["fast_cycle_period_us"] / 1000.0,
            )
            and _close(
                cadence["window_emission_interval_ms"],
                cadence["window_stride_cycles"] * cadence["fast_cycle_period_us"] / 1000.0,
            )
            and _close(cadence["window_content_duration_ms"], 10.24)
            and _close(cadence["window_emission_interval_ms"], 20.0),
            "window content and emission cadence arithmetic mismatch",
        )
        transport = project["transport"]
        uart = _find_by_id(transport, "transport_id", "PROJECT-UART-8N1")
        raw_bound_ms = uart["raw_uint16_histogram"]["serialization_lower_bound_ms"]
        software_bound_ms = uart["software_float32_payload"]["serialization_lower_bound_ms"]
        gate(
            "uart_8n1_capacity_arithmetic",
            _close(raw_bound_ms, 1000 * 2048 * 10 / 115200, atol=1.0e-4)
            and _close(software_bound_ms, 1000 * 4096 * 10 / 115200, atol=1.0e-4)
            and uart["minimum_raw_line_rate_bps"] == 1_024_000,
            "UART 8N1 lower bounds or minimum line-rate arithmetic mismatch",
        )
        gate(
            "uart_fails_twenty_ms_window",
            raw_bound_ms > cadence["window_emission_interval_ms"]
            and uart["raw_payload_meets_window_deadline"] is False,
            "115200-baud UART cannot carry the raw histogram every 20 ms",
        )
        frontend = project["physical_frontend"]
        gate(
            "target_board_frontend_fields_null",
            all(
                field["value_us"] is None
                and field["measured_on_target_board"] is False
                and field["evidence_class"] == "not_integrated_not_measured"
                for field in frontend
            ),
            "quantum measurement/ADC/AWG-DAC/physical action fields must remain null",
        )
        board_latency = project["target_board_latency"]
        gate(
            "target_board_latency_fields_null",
            all(
                field["value_us"] is None
                and field["measured_on_target_board"] is False
                and field["evidence_class"] == "target_board_unmeasured"
                for field in board_latency
            ),
            "USB-SPI/core/transport/end-to-end target-board latency must remain null",
        )
        gate(
            "real_board_gate_fail_closed",
            project["real_board_gate"] == "NO_GO_REAL_BOARD_HOST_OR_DEVICE_PATH_UNAVAILABLE"
            and project["board_measurement_status"] == "not_started",
            "real-board evidence is not available",
        )
        cross_lane = data.get("cross_lane_comparison", {})
        gate(
            "no_cross_lane_aggregate",
            cross_lane.get("aggregate_latency_us") is None
            and cross_lane.get("difference_us") is None
            and cross_lane.get("ratio") is None
            and cross_lane.get("status") == "forbidden_noncomparable_lanes",
            "literature and project lanes must not be added, subtracted or ratioed",
        )
    except (BudgetValidationError, KeyError, TypeError, ValueError, IndexError) as exc:
        failures.append(f"structural_exception: {exc}")

    declared = tuple(data.get("audit_contract", {}).get("required_gates", []))
    if declared != REQUIRED_GATES:
        failures.append("audit_contract.required_gates does not exactly match executable gates")
    status = "PASS" if not failures and all(gates.values()) else "FAIL"
    result = {
        "audit_schema_version": "dual-latency-budget-audit-v1",
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "status": status,
        "gates": gates,
        "failures": failures,
        "gate_count": len(gates),
        "passed_gate_count": sum(gates.values()),
    }
    if payload is None:
        artifact = Path(artifact_path)
        result["artifact_path"] = artifact.relative_to(repo_root).as_posix()
        result["artifact_sha256"] = _sha256(artifact)
    return result


def validate_budget(
    payload: Mapping[str, Any] | None = None,
    *,
    artifact_path: str | Path = DEFAULT_ARTIFACT,
    root: str | Path = ROOT,
) -> dict[str, Any]:
    result = audit_budget(payload, artifact_path=artifact_path, root=root)
    if result["status"] != "PASS":
        raise BudgetValidationError("; ".join(result["failures"]) or "dual latency budget failed")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--audit", action="store_true", help="print the executable audit result")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="write a hash-bound JSON audit snapshot",
    )
    args = parser.parse_args(argv)
    result = audit_budget(artifact_path=args.artifact)
    if args.output is not None:
        save_json(args.output, result)
    if args.audit:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    elif result["status"] != "PASS":
        print("\n".join(result["failures"]))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
