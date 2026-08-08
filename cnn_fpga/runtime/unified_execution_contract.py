"""Route-A unified execution, privilege, budget, and deadline contract.

This module is deliberately stricter than a benchmark configuration file.  It
defines the only observed online packet accepted by deployable comparison
methods, keeps simulator truth in a different schema, freezes the current
bit-accurate MAP-LUT and A/B-bank capabilities, and validates per-method
manifests and accounting records fail closed.

The current synthesizable LUT is phase-conditioned (two 257-entry Q9.12 LLR
tables).  A full two-dimensional joint-MAP implementation is therefore not
silently labelled current-RTL compatible; it can conform to the observed-only
software comparison lane while remaining blocked from the RTL lane.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any, Mapping

from cnn_fpga.runtime.atomic_parameter_bank import AtomicParameterBankConfig
from cnn_fpga.runtime.bit_accurate_hardware_reference import (
    INPUT_SCHEMA,
    OUTPUT_SCHEMA,
    decode_input_word,
    encode_input_word,
)
from cnn_fpga.runtime.parametric_map_lut import ParametricMAPLUTConfig
from cnn_fpga.runtime.three_timescale_cadence import ThreeTimescaleCadenceConfig


SCHEMA_VERSION = "t6.5.2-unified-execution-contract-v1"
CONTRACT_ID = "ROUTE-A-UNIFIED-EXECUTION-CONTRACT-V1"
OBSERVED_SCHEMA_ID = "route-a-observed-syndrome-v1"
ORACLE_SCHEMA_ID = "route-a-isolated-simulator-truth-v1"
ACTION_SCHEMA_ID = OUTPUT_SCHEMA.schema_id
MAP_LUT_CONTRACT_ID = "T421-PHASE-CONDITIONED-MAP-LUT-Q9.12-V1"
BANK_CONTRACT_ID = "T432-CRC-SHA-CAS-AB-BANK-V1"
CADENCE_CONTRACT_ID = "ROUTE-A-THREE-CADENCE-V1"
BUDGET_CONTRACT_ID = "ROUTE-A-MATCHED-BUDGET-V1"
DEADLINE_CONTRACT_ID = "ROUTE-A-DEADLINE-ACCOUNTING-V1"

DEPLOYABLE_METHOD_IDS = (
    "standard_binning",
    "static_joint_map",
    "window_map",
    "ewma_adaptive_map",
    "kalman_adaptive_map",
    "legacy_cnn_residual",
    "proposed_route_a",
)
ORACLE_METHOD_ID = "hidden_state_oracle"

OBSERVED_PACKET_FIELDS = (
    "schema_id",
    "trace_id",
    "cycle_index",
    "syndrome_code",
    "syndrome_x",
    "syndrome_z",
    "quadrature_phase_bit",
    "ood_score_code",
    "parameter_age_code",
    "reset_ack",
    "observation_valid",
    "deadline_ok",
)
ORACLE_PACKET_FIELDS = (
    "schema_id",
    "trace_id",
    "cycle_index",
    "latent_displacement_q",
    "latent_displacement_p",
    "latent_mean_q",
    "latent_mean_p",
    "latent_sigma_q",
    "latent_sigma_p",
    "latent_correlation",
    "logical_x",
    "logical_z",
    "regime_label",
)
TRUTH_KEY_TOKENS = (
    "truth",
    "hidden",
    "latent",
    "drift_state",
    "logical_class",
    "regime_label",
    "future",
    "evaluation_label",
)


class ContractViolation(ValueError):
    """Structured fail-fast violation used by benchmark adapters."""

    def __init__(
        self,
        code: str,
        *,
        field: str,
        expected: object,
        actual: object,
        method_id: str | None = None,
    ) -> None:
        self.code = str(code)
        self.field = str(field)
        self.expected = expected
        self.actual = actual
        self.method_id = method_id
        prefix = f"{method_id}:" if method_id else ""
        super().__init__(f"{prefix}{self.code}:{self.field}: expected={expected!r}, actual={actual!r}")

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "field": self.field,
            "expected": self.expected,
            "actual": self.actual,
            "method_id": self.method_id,
        }


def _strict_keys(payload: Mapping[str, object], fields: tuple[str, ...], schema: str) -> None:
    actual = set(payload)
    expected = set(fields)
    if actual != expected:
        raise ContractViolation(
            "schema_keys_mismatch",
            field=schema,
            expected=sorted(expected),
            actual={"missing": sorted(expected - actual), "extra": sorted(actual - expected)},
        )


def _integer(value: object, field: str, minimum: int, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractViolation("integer_required", field=field, expected="integer", actual=type(value).__name__)
    result = int(value)
    if result < minimum or (maximum is not None and result > maximum):
        expected = f"[{minimum},{maximum}]" if maximum is not None else f">={minimum}"
        raise ContractViolation("integer_out_of_range", field=field, expected=expected, actual=result)
    return result


def _boolean(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise ContractViolation("boolean_required", field=field, expected="bool", actual=type(value).__name__)
    return value


def _finite(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractViolation("real_required", field=field, expected="finite real", actual=type(value).__name__)
    result = float(value)
    if not math.isfinite(result):
        raise ContractViolation("finite_required", field=field, expected="finite", actual=result)
    return result


def _token(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 128:
        raise ContractViolation("token_required", field=field, expected="1..128 character string", actual=value)
    if any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.:" for character in value):
        raise ContractViolation("token_not_canonical", field=field, expected="portable token", actual=value)
    return value


@dataclass(frozen=True)
class ObservedSyndromePacket:
    schema_id: str
    trace_id: str
    cycle_index: int
    syndrome_code: int
    syndrome_x: str
    syndrome_z: str
    quadrature_phase_bit: int
    ood_score_code: int
    parameter_age_code: int
    reset_ack: bool
    observation_valid: bool
    deadline_ok: bool

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "ObservedSyndromePacket":
        if not isinstance(payload, Mapping):
            raise ContractViolation("mapping_required", field="observed_packet", expected="mapping", actual=type(payload).__name__)
        _strict_keys(payload, OBSERVED_PACKET_FIELDS, OBSERVED_SCHEMA_ID)
        if payload["schema_id"] != OBSERVED_SCHEMA_ID:
            raise ContractViolation("schema_id_mismatch", field="schema_id", expected=OBSERVED_SCHEMA_ID, actual=payload["schema_id"])
        trace_id = _token(payload["trace_id"], "trace_id")
        cycle_index = _integer(payload["cycle_index"], "cycle_index", 0)
        syndrome_code = _integer(payload["syndrome_code"], "syndrome_code", 0, 1023)
        syndrome_x = payload["syndrome_x"]
        syndrome_z = payload["syndrome_z"]
        for field, value in (("syndrome_x", syndrome_x), ("syndrome_z", syndrome_z)):
            if value not in ("g", "e", "leakage"):
                raise ContractViolation("observation_label_invalid", field=field, expected=("g", "e", "leakage"), actual=value)
        return cls(
            schema_id=OBSERVED_SCHEMA_ID,
            trace_id=trace_id,
            cycle_index=cycle_index,
            syndrome_code=syndrome_code,
            syndrome_x=str(syndrome_x),
            syndrome_z=str(syndrome_z),
            quadrature_phase_bit=_integer(payload["quadrature_phase_bit"], "quadrature_phase_bit", 0, 1),
            ood_score_code=_integer(payload["ood_score_code"], "ood_score_code", 0, 255),
            parameter_age_code=_integer(payload["parameter_age_code"], "parameter_age_code", 0, 65535),
            reset_ack=_boolean(payload["reset_ack"], "reset_ack"),
            observation_valid=_boolean(payload["observation_valid"], "observation_valid"),
            deadline_ok=_boolean(payload["deadline_ok"], "deadline_ok"),
        )

    def to_wire_word(self) -> int:
        return encode_input_word(
            syndrome_code=self.syndrome_code,
            syndrome_x=self.syndrome_x,
            syndrome_z=self.syndrome_z,
            quadrature_phase_bit=self.quadrature_phase_bit,
            ood_score_code=self.ood_score_code,
            parameter_age_code=self.parameter_age_code,
            reset_ack=self.reset_ack,
            observation_valid=self.observation_valid,
            deadline_ok=self.deadline_ok,
        )

    def sha256(self) -> str:
        return hashlib.sha256(
            json.dumps(asdict(self), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
        ).hexdigest()


@dataclass(frozen=True)
class OracleTruthPacket:
    schema_id: str
    trace_id: str
    cycle_index: int
    latent_displacement_q: float
    latent_displacement_p: float
    latent_mean_q: float
    latent_mean_p: float
    latent_sigma_q: float
    latent_sigma_p: float
    latent_correlation: float
    logical_x: bool
    logical_z: bool
    regime_label: str

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "OracleTruthPacket":
        if not isinstance(payload, Mapping):
            raise ContractViolation("mapping_required", field="oracle_packet", expected="mapping", actual=type(payload).__name__)
        _strict_keys(payload, ORACLE_PACKET_FIELDS, ORACLE_SCHEMA_ID)
        if payload["schema_id"] != ORACLE_SCHEMA_ID:
            raise ContractViolation("schema_id_mismatch", field="schema_id", expected=ORACLE_SCHEMA_ID, actual=payload["schema_id"])
        sigma_q = _finite(payload["latent_sigma_q"], "latent_sigma_q")
        sigma_p = _finite(payload["latent_sigma_p"], "latent_sigma_p")
        if sigma_q <= 0.0 or sigma_p <= 0.0:
            raise ContractViolation("sigma_not_positive", field="latent_sigma", expected=">0", actual=(sigma_q, sigma_p))
        correlation = _finite(payload["latent_correlation"], "latent_correlation")
        if not -1.0 < correlation < 1.0:
            raise ContractViolation("correlation_out_of_range", field="latent_correlation", expected="(-1,1)", actual=correlation)
        regime = payload["regime_label"]
        allowed_regimes = ("normal", "smooth", "burst", "leakage", "calibration_shift", "compound")
        if regime not in allowed_regimes:
            raise ContractViolation("regime_invalid", field="regime_label", expected=allowed_regimes, actual=regime)
        return cls(
            schema_id=ORACLE_SCHEMA_ID,
            trace_id=_token(payload["trace_id"], "trace_id"),
            cycle_index=_integer(payload["cycle_index"], "cycle_index", 0),
            latent_displacement_q=_finite(payload["latent_displacement_q"], "latent_displacement_q"),
            latent_displacement_p=_finite(payload["latent_displacement_p"], "latent_displacement_p"),
            latent_mean_q=_finite(payload["latent_mean_q"], "latent_mean_q"),
            latent_mean_p=_finite(payload["latent_mean_p"], "latent_mean_p"),
            latent_sigma_q=sigma_q,
            latent_sigma_p=sigma_p,
            latent_correlation=correlation,
            logical_x=_boolean(payload["logical_x"], "logical_x"),
            logical_z=_boolean(payload["logical_z"], "logical_z"),
            regime_label=str(regime),
        )


@dataclass(frozen=True)
class MatchedBudget:
    fast_action_latency_cycles: int = 6
    fast_action_budget_us_assumption: float = 1.5
    parameter_window_samples: int = 2048
    parameter_update_period_cycles: int = 4000
    regime_window_cycles: int = 32
    regime_update_period_cycles: int = 32
    max_algorithm_macs_per_parameter_update: int = 8192
    max_private_model_state_bytes: int = 8192
    max_transient_workspace_bytes: int = 8192
    max_host_update_wallclock_us: float = 5000.0
    shared_dual_bank_lut_payload_bits: int = 22616
    shared_current_rtl_mirror_bits: int = 45232

    def __post_init__(self) -> None:
        integer_fields = (
            "fast_action_latency_cycles",
            "parameter_window_samples",
            "parameter_update_period_cycles",
            "regime_window_cycles",
            "regime_update_period_cycles",
            "max_algorithm_macs_per_parameter_update",
            "max_private_model_state_bytes",
            "max_transient_workspace_bytes",
            "shared_dual_bank_lut_payload_bits",
            "shared_current_rtl_mirror_bits",
        )
        for field in integer_fields:
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field} must be a positive integer")
        for field in ("fast_action_budget_us_assumption", "max_host_update_wallclock_us"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{field} must be finite and positive")
        if self.fast_action_latency_cycles != 6:
            raise ValueError("Route-A action path is exactly six registered cycles")
        if self.parameter_window_samples != 2048 or self.parameter_update_period_cycles != 4000:
            raise ValueError("parameter cadence must match the frozen T4.3.1 cadence")
        if self.regime_window_cycles != 32 or self.regime_update_period_cycles != 32:
            raise ValueError("regime cadence must match the frozen T4.1.1 observed-window contract")
        if self.shared_dual_bank_lut_payload_bits != 2 * 2 * 257 * 22:
            raise ValueError("shared LUT payload must describe two banks, two phases, 257 Q9.12 entries")
        if self.shared_current_rtl_mirror_bits != 2 * self.shared_dual_bank_lut_payload_bits:
            raise ValueError("current RTL mirror bits must account for the duplicated read ports")


@dataclass(frozen=True)
class MethodManifest:
    method_id: str
    online_privilege: str
    input_schema_id: str
    action_schema_id: str
    map_lut_contract_id: str
    bank_contract_id: str
    cadence_contract_id: str
    budget_contract_id: str
    deadline_contract_id: str
    parameter_update_behavior: str
    benchmark_deployability: str
    current_rtl_compatibility: str
    implementation_path: str
    adapter_entrypoint: str

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "MethodManifest":
        fields = tuple(cls.__dataclass_fields__)
        if not isinstance(payload, Mapping):
            raise ContractViolation("mapping_required", field="method_manifest", expected="mapping", actual=type(payload).__name__)
        _strict_keys(payload, fields, "method_manifest")
        return cls(**{field: str(payload[field]) for field in fields})

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


_COMMON_MANIFEST_VALUES = {
    "online_privilege": "observed_only",
    "input_schema_id": OBSERVED_SCHEMA_ID,
    "action_schema_id": ACTION_SCHEMA_ID,
    "map_lut_contract_id": MAP_LUT_CONTRACT_ID,
    "bank_contract_id": BANK_CONTRACT_ID,
    "cadence_contract_id": CADENCE_CONTRACT_ID,
    "budget_contract_id": BUDGET_CONTRACT_ID,
    "deadline_contract_id": DEADLINE_CONTRACT_ID,
    "benchmark_deployability": "observed_only_budgeted_candidate",
}


def canonical_method_manifests() -> tuple[MethodManifest, ...]:
    rows = (
        ("standard_binning", "frozen_rule_compiled_to_common_lut_grid", "standard_rule_lut_image_requires_t6_6_exhaustive_equivalence", "cnn_fpga/benchmark/standard_binning_baseline.py", "build_standard_binning_validation"),
        ("static_joint_map", "frozen_training_only_parameters", "blocked_full_2d_joint_map_not_equivalent_to_current_phase_lut", "cnn_fpga/benchmark/static_map_baseline.py", "build_static_map_validation"),
        ("window_map", "common_parameter_window_boundary", "phase_lut_candidate_requires_t6_6_adapter_proof", "cnn_fpga/benchmark/sliding_window_syndrome_estimator.py", "build_sliding_window_validation"),
        ("ewma_adaptive_map", "common_parameter_window_boundary", "phase_lut_candidate_requires_t6_6_adapter_proof", "cnn_fpga/benchmark/continuous_adaptive_map.py", "build_continuous_adaptive_validation"),
        ("kalman_adaptive_map", "common_parameter_window_boundary", "phase_lut_candidate_requires_t6_6_adapter_proof", "cnn_fpga/benchmark/continuous_adaptive_map.py", "build_continuous_adaptive_validation"),
        ("legacy_cnn_residual", "common_parameter_window_boundary", "phase_lut_candidate_only_if_checkpoint_and_budget_pass", "cnn_fpga/runtime/inference_service.py", "ArtifactHistogramPredictor.predict"),
        ("proposed_route_a", "regime_32_cycles_and_parameter_4000_cycles", "phase_lut_candidate_requires_t6_6_policy_and_adapter_proof", "cnn_fpga/runtime/unified_execution_contract.py", "route_a_adapter_pending_t6_6"),
    )
    return tuple(
        MethodManifest(
            method_id=method_id,
            parameter_update_behavior=update,
            current_rtl_compatibility=rtl,
            implementation_path=path,
            adapter_entrypoint=entrypoint,
            **_COMMON_MANIFEST_VALUES,
        )
        for method_id, update, rtl, path, entrypoint in rows
    )


def oracle_method_manifest() -> MethodManifest:
    return MethodManifest(
        method_id=ORACLE_METHOD_ID,
        online_privilege="isolated_simulator_truth",
        input_schema_id=ORACLE_SCHEMA_ID,
        action_schema_id="oracle-upper-bound-action-v1",
        map_lut_contract_id="not_applicable_privileged_reference",
        bank_contract_id="not_applicable_privileged_reference",
        cadence_contract_id="same_trace_index_not_deployable_cadence",
        budget_contract_id="reported_not_matched_not_ranked_as_deployable",
        deadline_contract_id="not_deployable_no_deadline_claim",
        parameter_update_behavior="reads_current_hidden_state",
        benchmark_deployability="nondeployable_upper_bound_only",
        current_rtl_compatibility="prohibited",
        implementation_path="cnn_fpga/benchmark/oracle_baseline.py",
        adapter_entrypoint="oracle_reference_only",
    )


def assert_method_conforms(manifest: MethodManifest) -> None:
    if not isinstance(manifest, MethodManifest):
        raise ContractViolation("manifest_type_invalid", field="manifest", expected="MethodManifest", actual=type(manifest).__name__)
    method_id = manifest.method_id
    if method_id == ORACLE_METHOD_ID:
        expected = oracle_method_manifest()
        for field in MethodManifest.__dataclass_fields__:
            actual_value = getattr(manifest, field)
            expected_value = getattr(expected, field)
            if actual_value != expected_value:
                raise ContractViolation("oracle_manifest_mismatch", field=field, expected=expected_value, actual=actual_value, method_id=method_id)
        return
    canonical = {row.method_id: row for row in canonical_method_manifests()}
    if method_id not in canonical:
        raise ContractViolation("method_not_registered", field="method_id", expected=sorted(canonical), actual=method_id, method_id=method_id)
    expected = canonical[method_id]
    for field in MethodManifest.__dataclass_fields__:
        actual_value = getattr(manifest, field)
        expected_value = getattr(expected, field)
        if actual_value != expected_value:
            raise ContractViolation("method_contract_mismatch", field=field, expected=expected_value, actual=actual_value, method_id=method_id)
    if any(token in field.lower() for field in manifest.to_dict() for token in TRUTH_KEY_TOKENS):
        raise ContractViolation("truth_named_manifest_field", field="manifest_fields", expected="no truth channel", actual=list(manifest.to_dict()), method_id=method_id)


@dataclass(frozen=True)
class ExecutionAccountingRecord:
    method_id: str
    trace_id: str
    cycle_index: int
    action_valid_cycle: int
    source_to_action_cycles: int
    logical_deadline_miss: bool
    update_due: bool
    update_macs: int
    private_model_state_bytes: int
    transient_workspace_bytes: int
    host_update_wallclock_us: float
    host_update_deadline_miss: bool
    board_measured_deadline_miss: bool | None

    def __post_init__(self) -> None:
        _token(self.method_id, "method_id")
        _token(self.trace_id, "trace_id")
        for field in (
            "cycle_index",
            "action_valid_cycle",
            "source_to_action_cycles",
            "update_macs",
            "private_model_state_bytes",
            "transient_workspace_bytes",
        ):
            _integer(getattr(self, field), field, 0)
        for field in ("logical_deadline_miss", "update_due", "host_update_deadline_miss"):
            _boolean(getattr(self, field), field)
        wallclock = _finite(self.host_update_wallclock_us, "host_update_wallclock_us")
        if wallclock < 0.0:
            raise ContractViolation(
                "negative_cost_forbidden",
                field="host_update_wallclock_us",
                expected=">=0",
                actual=wallclock,
                method_id=self.method_id,
            )
        board = self.board_measured_deadline_miss
        if board is not None and not isinstance(board, bool):
            raise ContractViolation(
                "nullable_boolean_required",
                field="board_measured_deadline_miss",
                expected="bool or null",
                actual=type(board).__name__,
                method_id=self.method_id,
            )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "ExecutionAccountingRecord":
        fields = tuple(cls.__dataclass_fields__)
        if not isinstance(payload, Mapping):
            raise ContractViolation("mapping_required", field="accounting", expected="mapping", actual=type(payload).__name__)
        _strict_keys(payload, fields, "execution_accounting")
        board = payload["board_measured_deadline_miss"]
        if board is not None and not isinstance(board, bool):
            raise ContractViolation("nullable_boolean_required", field="board_measured_deadline_miss", expected="bool or null", actual=type(board).__name__)
        return cls(
            method_id=_token(payload["method_id"], "method_id"),
            trace_id=_token(payload["trace_id"], "trace_id"),
            cycle_index=_integer(payload["cycle_index"], "cycle_index", 0),
            action_valid_cycle=_integer(payload["action_valid_cycle"], "action_valid_cycle", 0),
            source_to_action_cycles=_integer(payload["source_to_action_cycles"], "source_to_action_cycles", 0),
            logical_deadline_miss=_boolean(payload["logical_deadline_miss"], "logical_deadline_miss"),
            update_due=_boolean(payload["update_due"], "update_due"),
            update_macs=_integer(payload["update_macs"], "update_macs", 0),
            private_model_state_bytes=_integer(payload["private_model_state_bytes"], "private_model_state_bytes", 0),
            transient_workspace_bytes=_integer(payload["transient_workspace_bytes"], "transient_workspace_bytes", 0),
            host_update_wallclock_us=_finite(payload["host_update_wallclock_us"], "host_update_wallclock_us"),
            host_update_deadline_miss=_boolean(payload["host_update_deadline_miss"], "host_update_deadline_miss"),
            board_measured_deadline_miss=board,
        )


def assert_accounting_conforms(record: ExecutionAccountingRecord, budget: MatchedBudget | None = None) -> None:
    if not isinstance(record, ExecutionAccountingRecord):
        raise ContractViolation("accounting_type_invalid", field="record", expected="ExecutionAccountingRecord", actual=type(record).__name__)
    actual_budget = MatchedBudget() if budget is None else budget
    if record.method_id not in DEPLOYABLE_METHOD_IDS:
        raise ContractViolation("accounting_method_not_deployable", field="method_id", expected=DEPLOYABLE_METHOD_IDS, actual=record.method_id, method_id=record.method_id)
    expected_valid = record.cycle_index + actual_budget.fast_action_latency_cycles
    if record.action_valid_cycle != expected_valid or record.source_to_action_cycles != actual_budget.fast_action_latency_cycles:
        raise ContractViolation("action_latency_not_exact", field="action_valid_cycle", expected=expected_valid, actual=record.action_valid_cycle, method_id=record.method_id)
    if record.logical_deadline_miss:
        raise ContractViolation("logical_deadline_flag_inconsistent", field="logical_deadline_miss", expected=False, actual=True, method_id=record.method_id)
    for field, limit in (
        ("update_macs", actual_budget.max_algorithm_macs_per_parameter_update),
        ("private_model_state_bytes", actual_budget.max_private_model_state_bytes),
        ("transient_workspace_bytes", actual_budget.max_transient_workspace_bytes),
    ):
        value = getattr(record, field)
        if value > limit:
            raise ContractViolation("matched_budget_exceeded", field=field, expected=f"<={limit}", actual=value, method_id=record.method_id)
    expected_host_miss = record.host_update_wallclock_us > actual_budget.max_host_update_wallclock_us
    if record.host_update_deadline_miss != expected_host_miss:
        raise ContractViolation("host_deadline_flag_inconsistent", field="host_update_deadline_miss", expected=expected_host_miss, actual=record.host_update_deadline_miss, method_id=record.method_id)
    if expected_host_miss:
        raise ContractViolation("host_update_deadline_exceeded", field="host_update_wallclock_us", expected=f"<={actual_budget.max_host_update_wallclock_us}", actual=record.host_update_wallclock_us, method_id=record.method_id)
    if record.board_measured_deadline_miss is not None:
        raise ContractViolation("board_measurement_not_available", field="board_measured_deadline_miss", expected=None, actual=record.board_measured_deadline_miss, method_id=record.method_id)
    if not record.update_due and (record.update_macs != 0 or record.host_update_wallclock_us != 0.0):
        raise ContractViolation("cost_charged_when_update_not_due", field="update_due", expected="zero update work", actual={"macs": record.update_macs, "wallclock_us": record.host_update_wallclock_us}, method_id=record.method_id)


def validate_observed_mapping_for_deployable(method_id: str, payload: Mapping[str, object]) -> ObservedSyndromePacket:
    if method_id not in DEPLOYABLE_METHOD_IDS:
        raise ContractViolation("method_not_deployable", field="method_id", expected=DEPLOYABLE_METHOD_IDS, actual=method_id, method_id=method_id)
    lowered = {str(key).lower() for key in payload}
    leaked = sorted(key for key in lowered if any(token in key for token in TRUTH_KEY_TOKENS))
    if leaked:
        raise ContractViolation("hidden_truth_key_rejected", field="observed_packet", expected="observed fields only", actual=leaked, method_id=method_id)
    return ObservedSyndromePacket.from_mapping(payload)


def validate_wire_roundtrip(packet: ObservedSyndromePacket) -> None:
    word = packet.to_wire_word()
    decoded = decode_input_word(word)
    expected = {
        "syndrome_code": packet.syndrome_code,
        "syndrome_x": packet.syndrome_x,
        "syndrome_z": packet.syndrome_z,
        "quadrature_phase_bit": packet.quadrature_phase_bit,
        "ood_score_code": packet.ood_score_code,
        "parameter_age_code": packet.parameter_age_code,
        "reset_ack": packet.reset_ack,
        "observation_valid": packet.observation_valid,
        "deadline_ok": packet.deadline_ok,
    }
    for field, expected_value in expected.items():
        actual_value = getattr(decoded, field)
        if actual_value != expected_value:
            raise ContractViolation("wire_roundtrip_mismatch", field=field, expected=expected_value, actual=actual_value)


def contract_snapshot() -> dict[str, Any]:
    lut = ParametricMAPLUTConfig()
    cadence = ThreeTimescaleCadenceConfig()
    bank = AtomicParameterBankConfig()
    budget = MatchedBudget()
    return {
        "schema_version": SCHEMA_VERSION,
        "contract_id": CONTRACT_ID,
        "observed_schema": {"schema_id": OBSERVED_SCHEMA_ID, "fields": list(OBSERVED_PACKET_FIELDS), "wire_schema": INPUT_SCHEMA.to_dict(), "truth_key_denylist": list(TRUTH_KEY_TOKENS)},
        "oracle_schema": {"schema_id": ORACLE_SCHEMA_ID, "fields": list(ORACLE_PACKET_FIELDS), "physically_separate_from_observed": True, "deployable_access": False},
        "action_schema": {"schema_id": ACTION_SCHEMA_ID, "wire_schema": OUTPUT_SCHEMA.to_dict(), "latency_cycles": 6, "decomposition": "5-cycle MAP plus 1-cycle registered event/action"},
        "map_lut": {
            "contract_id": MAP_LUT_CONTRACT_ID,
            "capability": "phase_conditioned_X_Z_LLR_not_full_2d_joint_MAP",
            "adc_bits": lut.adc_bits,
            "address_bits": lut.address_bits,
            "fraction_bits": lut.fraction_bits,
            "entries_per_phase": lut.table_entries,
            "phase_tables": 2,
            "llr_format": "signed Q9.12",
            "llr_word_bits": lut.llr_word_bits,
            "rounding": "round_to_nearest_ties_to_even",
            "saturation_codes": [lut.llr_min_code, lut.llr_max_code],
            "pipeline_latency_cycles": lut.pipeline_latency_cycles,
            "content_semantics": "same code grid, interpolation, Q-format, CRC and bank image layout; method-specific LLR values may change only through the common cadence and commit protocol",
            "joint_map_boundary": "full_2d_joint_MAP_is_observed_only_software_comparator_until_equivalence_or_new_RTL",
        },
        "bank": {
            "contract_id": BANK_CONTRACT_ID,
            "banks": ["A", "B"],
            "version_rule": "new_version=active_version+1_with_CAS",
            "integrity": ["CRC16_wire", "CRC32_image", "SHA256_image", "manifest_CRC32", "manifest_SHA256"],
            "rollback": "last_known_good_only",
            "max_payload_age_cycles": bank.max_payload_age_cycles,
            "min_residency_cycles": bank.min_residency_cycles,
            "retired_bank_drain_cycles": 6,
            "silent_partial_publish": "prohibited",
        },
        "cadence": {
            "contract_id": CADENCE_CONTRACT_ID,
            "fast_round_period_us": cadence.t_fast_us,
            "event_evaluation_period_cycles": 1,
            "regime_window_cycles": budget.regime_window_cycles,
            "regime_update_period_cycles": budget.regime_update_period_cycles,
            "parameter_window_samples": budget.parameter_window_samples,
            "parameter_update_period_cycles": budget.parameter_update_period_cycles,
            "one_window_causal_delay_required": True,
            "commit_delay_cycles": cadence.commit_delay_cycles,
            "max_parameter_age_cycles": cadence.max_parameter_age_cycles,
        },
        "budget": {"contract_id": BUDGET_CONTRACT_ID, **asdict(budget), "shared_lut_excluded_from_private_method_memory": True, "actual_usage_reported_even_when_below_cap": True},
        "deadline": {
            "contract_id": DEADLINE_CONTRACT_ID,
            "logical_action_valid_exactly_at_input_plus_cycles": 6,
            "configured_fast_action_budget_us_assumption": budget.fast_action_budget_us_assumption,
            "host_update_wallclock_ceiling_us": budget.max_host_update_wallclock_us,
            "required_fields": list(ExecutionAccountingRecord.__dataclass_fields__),
            "board_deadline_field_must_be_null_before_measurement": True,
        },
        "methods": [row.to_dict() for row in canonical_method_manifests()],
        "oracle_method": oracle_method_manifest().to_dict(),
    }


def contract_sha256(snapshot: Mapping[str, object] | None = None) -> str:
    payload = contract_snapshot() if snapshot is None else dict(snapshot)
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    ).hexdigest()


__all__ = [
    "ACTION_SCHEMA_ID",
    "BANK_CONTRACT_ID",
    "BUDGET_CONTRACT_ID",
    "CADENCE_CONTRACT_ID",
    "CONTRACT_ID",
    "ContractViolation",
    "DEADLINE_CONTRACT_ID",
    "DEPLOYABLE_METHOD_IDS",
    "ExecutionAccountingRecord",
    "MAP_LUT_CONTRACT_ID",
    "MatchedBudget",
    "MethodManifest",
    "OBSERVED_PACKET_FIELDS",
    "OBSERVED_SCHEMA_ID",
    "ORACLE_METHOD_ID",
    "ORACLE_PACKET_FIELDS",
    "ORACLE_SCHEMA_ID",
    "ObservedSyndromePacket",
    "OracleTruthPacket",
    "SCHEMA_VERSION",
    "TRUTH_KEY_TOKENS",
    "assert_accounting_conforms",
    "assert_method_conforms",
    "canonical_method_manifests",
    "contract_sha256",
    "contract_snapshot",
    "oracle_method_manifest",
    "validate_observed_mapping_for_deployable",
    "validate_wire_roundtrip",
]
