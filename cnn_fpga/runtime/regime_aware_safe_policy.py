"""Route-A regime-aware safe adaptive policy over the production fast path.

This module is deliberately a policy/integration layer, not a second decoder.
It composes observed-only posterior evidence, compiled MAP images, the atomic
A/B bank, the six-cycle event FSM and conservative fallback.  Thresholds in
this file are integration defaults; T6.6.3 must calibrate and hash-lock them
before any evaluation claim is admitted.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from typing import Any

from cnn_fpga.runtime.atomic_parameter_bank import AtomicParameterBankConfig
from cnn_fpga.runtime.closed_loop_fault_recovery import (
    ClosedLoopCycleInput,
    ClosedLoopCycleRecord,
    ClosedLoopFaultRecoverySupervisor,
    ClosedLoopRecoveryConfig,
    parameter_image_semantics_sha256,
)
from cnn_fpga.runtime.parametric_map_lut import ParametricMAPLUTImage


MODEL_SCOPE = "route_a_integration_policy_thresholds_not_t6_6_3_frozen"
REGIME_CLASSES = ("normal", "smooth", "calibration_shift", "burst")
ADAPTIVE_OPEN = "adaptive_open"
TAIL_TRUSTED = "tail_trusted"
POSTERIOR_UNCERTAIN = "posterior_uncertain"
INTEGRITY_ROLLBACK = "integrity_rollback"
LEAKAGE_RESET = "leakage_reset"
RECOVERING = "recovering"
POLICY_MODES = (
    ADAPTIVE_OPEN,
    TAIL_TRUSTED,
    POSTERIOR_UNCERTAIN,
    INTEGRITY_ROLLBACK,
    LEAKAGE_RESET,
    RECOVERING,
)
DUAL_BANK_ROUTER_ALGORITHM_SHA256 = (
    "0ba2b4ca6d11d50cb7f28d9dc71366949ddc53dcecb2d8be12ce0b8af4d12441"
)
CANDIDATE_METHODS = ("window_map", "ewma_adaptive_map")


def _integer(value: object, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be boolean")
    return value


def _digest(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a 64-character SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{name} must be hexadecimal") from exc
    return value.lower()


@dataclass(frozen=True)
class RouteAPolicyConfig:
    regime_window_cycles: int = 32
    parameter_update_period_cycles: int = 4_000
    fast_action_latency_cycles: int = 6
    posterior_sum_tolerance: float = 1.0e-6
    regime_posterior_enter_min: float = 0.60
    regime_posterior_exit_max: float = 0.40
    uncertainty_fallback_min: float = 0.55
    ood_score_min_code: int = 192
    enter_hysteresis_windows: int = 2
    recovery_hysteresis_windows: int = 2
    emergency_confirmation_events: int | None = None
    max_parameter_age_cycles: int = 8_192
    max_update_macs: int = 8_192
    max_private_state_bytes: int = 8_192
    max_workspace_bytes: int = 8_192
    max_host_update_wallclock_us: float = 5_000.0
    smooth_bank_posterior_min: float = 0.30
    prequential_score_memory: float = 0.0
    dual_shadow_update_macs: int = 1_218
    router_algorithm_sha256: str = DUAL_BANK_ROUTER_ALGORITHM_SHA256
    threshold_lock_sha256: str | None = None
    posterior_model_sha256: str | None = None
    event_model_sha256: str | None = None
    threshold_protocol_id: str = "T6.6.3_UNFROZEN_INTEGRATION_DEFAULTS"
    model_scope: str = MODEL_SCOPE

    def __post_init__(self) -> None:
        for name in (
            "regime_window_cycles",
            "parameter_update_period_cycles",
            "fast_action_latency_cycles",
            "ood_score_min_code",
            "enter_hysteresis_windows",
            "recovery_hysteresis_windows",
            "max_parameter_age_cycles",
            "max_update_macs",
            "max_private_state_bytes",
            "max_workspace_bytes",
            "dual_shadow_update_macs",
        ):
            object.__setattr__(self, name, _integer(getattr(self, name), name, 1))
        for name in (
            "posterior_sum_tolerance",
            "regime_posterior_enter_min",
            "regime_posterior_exit_max",
            "uncertainty_fallback_min",
            "max_host_update_wallclock_us",
            "smooth_bank_posterior_min",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)
        if self.fast_action_latency_cycles != 6:
            raise ValueError("Route-A must preserve the frozen six-cycle fast action path")
        for name in (
            "regime_posterior_enter_min",
            "regime_posterior_exit_max",
            "uncertainty_fallback_min",
        ):
            if getattr(self, name) > 1.0:
                raise ValueError(f"{name} must not exceed one")
        if self.regime_posterior_exit_max >= self.regime_posterior_enter_min:
            raise ValueError("posterior exit threshold must be below enter threshold")
        score_memory = float(self.prequential_score_memory)
        if not math.isfinite(score_memory) or not 0.0 <= score_memory < 1.0:
            raise ValueError("prequential_score_memory must lie in [0,1)")
        object.__setattr__(self, "prequential_score_memory", score_memory)
        if self.dual_shadow_update_macs > self.max_update_macs:
            raise ValueError("dual-shadow router exceeds the matched update-MAC budget")
        object.__setattr__(
            self,
            "router_algorithm_sha256",
            _digest(self.router_algorithm_sha256, "router_algorithm_sha256"),
        )
        if self.ood_score_min_code > 255:
            raise ValueError("ood_score_min_code must fit the frozen 8-bit score")
        if self.emergency_confirmation_events is None:
            object.__setattr__(
                self, "emergency_confirmation_events", self.enter_hysteresis_windows
            )
        else:
            legacy_enter = _integer(
                self.emergency_confirmation_events,
                "emergency_confirmation_events",
                1,
            )
            if legacy_enter != self.enter_hysteresis_windows:
                raise ValueError(
                    "deprecated emergency_confirmation_events must equal enter_hysteresis_windows"
                )
            object.__setattr__(self, "emergency_confirmation_events", legacy_enter)
        frozen_hashes = (
            self.threshold_lock_sha256,
            self.posterior_model_sha256,
            self.event_model_sha256,
        )
        if any(value is None for value in frozen_hashes) and not all(
            value is None for value in frozen_hashes
        ):
            raise ValueError("threshold, posterior and event model hashes must be supplied together")
        if self.threshold_lock_sha256 is not None:
            object.__setattr__(
                self,
                "threshold_lock_sha256",
                _digest(self.threshold_lock_sha256, "threshold_lock_sha256"),
            )
            object.__setattr__(
                self,
                "posterior_model_sha256",
                _digest(self.posterior_model_sha256, "posterior_model_sha256"),
            )
            object.__setattr__(
                self,
                "event_model_sha256",
                _digest(self.event_model_sha256, "event_model_sha256"),
            )
            if not self.threshold_protocol_id.startswith("ROUTE-A-"):
                raise ValueError("frozen threshold_protocol_id must identify Route A")
        elif self.threshold_protocol_id != "T6.6.3_UNFROZEN_INTEGRATION_DEFAULTS":
            raise ValueError("unfrozen policy must retain the explicit integration protocol id")
        if self.model_scope != MODEL_SCOPE:
            raise ValueError(f"model_scope must be {MODEL_SCOPE!r}")


@dataclass(frozen=True)
class ObservedRegimePosterior:
    source_window_id: int
    source_start_cycle: int
    source_end_cycle: int
    available_cycle: int
    probabilities: tuple[float, float, float, float]
    model_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "source_window_id",
            "source_start_cycle",
            "source_end_cycle",
            "available_cycle",
        ):
            object.__setattr__(self, name, _integer(getattr(self, name), name))
        if self.source_end_cycle < self.source_start_cycle:
            raise ValueError("posterior source window is reversed")
        if self.available_cycle <= self.source_end_cycle:
            raise ValueError("posterior must become available strictly after its source window")
        if len(self.probabilities) != len(REGIME_CLASSES):
            raise ValueError("posterior must contain normal/smooth/calibration_shift/burst")
        checked = tuple(float(value) for value in self.probabilities)
        if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in checked):
            raise ValueError("posterior probabilities must be finite values in [0,1]")
        object.__setattr__(self, "probabilities", checked)
        object.__setattr__(self, "model_sha256", _digest(self.model_sha256, "model_sha256"))

    def named(self) -> dict[str, float]:
        return dict(zip(REGIME_CLASSES, self.probabilities, strict=True))


@dataclass(frozen=True)
class AdaptiveMAPCandidate:
    method_id: str
    source_window_id: int
    source_start_cycle: int
    source_end_cycle: int
    available_cycle: int
    image: ParametricMAPLUTImage
    update_macs: int
    private_state_bytes: int
    transient_workspace_bytes: int
    host_update_wallclock_us: float
    window_prequential_score: float
    ewma_prequential_score: float
    router_algorithm_sha256: str = DUAL_BANK_ROUTER_ALGORITHM_SHA256

    def __post_init__(self) -> None:
        if self.method_id not in CANDIDATE_METHODS:
            raise ValueError(f"method_id must be one of {CANDIDATE_METHODS}")
        for name in (
            "source_window_id",
            "source_start_cycle",
            "source_end_cycle",
            "available_cycle",
            "update_macs",
            "private_state_bytes",
            "transient_workspace_bytes",
        ):
            object.__setattr__(self, name, _integer(getattr(self, name), name))
        if self.source_end_cycle < self.source_start_cycle:
            raise ValueError("candidate source window is reversed")
        if self.available_cycle <= self.source_end_cycle:
            raise ValueError("candidate must become available strictly after its source window")
        if not isinstance(self.image, ParametricMAPLUTImage):
            raise TypeError("image must be ParametricMAPLUTImage")
        self.image.verify()
        wallclock = float(self.host_update_wallclock_us)
        if not math.isfinite(wallclock) or wallclock < 0.0:
            raise ValueError("host_update_wallclock_us must be finite and non-negative")
        object.__setattr__(self, "host_update_wallclock_us", wallclock)
        for name in ("window_prequential_score", "ewma_prequential_score"):
            score = float(getattr(self, name))
            if not math.isfinite(score):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, score)
        object.__setattr__(
            self,
            "router_algorithm_sha256",
            _digest(self.router_algorithm_sha256, "router_algorithm_sha256"),
        )


@dataclass(frozen=True)
class RouteADeadlineLedger:
    source_cycle: int
    action_cycle: int
    action_deadline_cycle: int
    source_to_action_cycles: int
    fast_deadline_miss: bool
    host_update_due: bool
    host_update_macs: int
    host_private_state_bytes: int
    host_workspace_bytes: int
    host_update_wallclock_us: float
    host_budget_violation: bool
    board_measured_deadline_miss: None = None


@dataclass(frozen=True)
class RouteACycleInput:
    fast_path: ClosedLoopCycleInput
    posterior_update: ObservedRegimePosterior | None = None
    candidate: AdaptiveMAPCandidate | None = None
    parameter_update_due: bool = False
    event_model_sha256: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.fast_path, ClosedLoopCycleInput):
            raise TypeError("fast_path must be ClosedLoopCycleInput")
        if self.posterior_update is not None and not isinstance(
            self.posterior_update, ObservedRegimePosterior
        ):
            raise TypeError("posterior_update must be ObservedRegimePosterior or None")
        if self.candidate is not None and not isinstance(self.candidate, AdaptiveMAPCandidate):
            raise TypeError("candidate must be AdaptiveMAPCandidate or None")
        object.__setattr__(
            self, "parameter_update_due", _boolean(self.parameter_update_due, "parameter_update_due")
        )
        if self.event_model_sha256 is not None:
            object.__setattr__(
                self,
                "event_model_sha256",
                _digest(self.event_model_sha256, "event_model_sha256"),
            )
        if self.candidate is not None and not self.parameter_update_due:
            raise ValueError("a candidate may only be supplied when parameter_update_due is true")


@dataclass(frozen=True)
class RouteAAction:
    cycle_index: int
    policy_mode: str
    primary_reason: str
    reason_trace: str
    posterior: tuple[float, float, float, float]
    posterior_source_window_id: int | None
    posterior_model_sha256: str | None
    active_bank: str
    active_bank_version: int
    active_semantics_sha256: str
    decision_source: str
    online_update_frozen: bool
    commit_gate_open: bool
    recovery_evidence_run: int
    fallback_active: bool
    reset_request: bool
    rollback_requested: bool
    rollback_completed: bool
    trusted_switch_requested: bool
    trusted_switch_completed: bool
    candidate_received: bool
    candidate_accepted: bool
    candidate_reason: str
    safety_stage_accepted: bool
    staged_version: int | None
    commit_status: str
    commit_reason: str
    readback_status: str
    deadline: RouteADeadlineLedger
    fast_path_record: ClosedLoopCycleRecord
    model_scope: str = MODEL_SCOPE

    def __post_init__(self) -> None:
        if self.policy_mode not in POLICY_MODES:
            raise ValueError(f"policy_mode must be one of {POLICY_MODES}")
        if not self.primary_reason or not self.reason_trace:
            raise ValueError("every Route-A action requires primary_reason and reason_trace")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["posterior"] = list(self.posterior)
        payload["fast_path_record"]["fault_flags"] = list(
            self.fast_path_record.fault_flags
        )
        return payload


def clone_parameter_image(
    prototype: ParametricMAPLUTImage, *, active_bank_version: int
) -> ParametricMAPLUTImage:
    """Republish identical semantics with a fresh monotonic version."""

    if not isinstance(prototype, ParametricMAPLUTImage):
        raise TypeError("prototype must be ParametricMAPLUTImage")
    prototype.verify()
    version = _integer(active_bank_version, "active_bank_version")
    return ParametricMAPLUTImage.create(
        config=prototype.config,
        active_bank_version=version,
        source_params_sha256=prototype.source_params_sha256,
        model_mean=prototype.model_mean,
        model_sigma=prototype.model_sigma,
        table_codes=prototype.table_codes,
        llr_saturation_count=prototype.llr_saturation_count,
    )


class RegimeAwareSafeAdaptivePolicy:
    """Causal Route-A controller with fail-closed update and recovery gates."""

    def __init__(
        self,
        trusted_static_image: ParametricMAPLUTImage,
        *,
        config: RouteAPolicyConfig | None = None,
        recovery_config: ClosedLoopRecoveryConfig | None = None,
        bank_config: AtomicParameterBankConfig | None = None,
    ) -> None:
        self.config = RouteAPolicyConfig() if config is None else config
        if not isinstance(self.config, RouteAPolicyConfig):
            raise TypeError("config must be RouteAPolicyConfig")
        if not isinstance(trusted_static_image, ParametricMAPLUTImage):
            raise TypeError("trusted_static_image must be ParametricMAPLUTImage")
        trusted_static_image.verify()
        if trusted_static_image.active_bank_version != 0:
            raise ValueError("trusted static image must start at version zero")
        recovery = (
            ClosedLoopRecoveryConfig(
                max_parameter_age_cycles=self.config.max_parameter_age_cycles,
                host_timeout_cycles=self.config.max_parameter_age_cycles,
                post_commit_guard_cycles=self.config.parameter_update_period_cycles,
            )
            if recovery_config is None
            else recovery_config
        )
        bank = (
            AtomicParameterBankConfig(
                promotion_good_windows=self.config.recovery_hysteresis_windows,
                min_residency_cycles=self.config.parameter_update_period_cycles,
                max_payload_age_cycles=self.config.max_parameter_age_cycles,
            )
            if bank_config is None
            else bank_config
        )
        if recovery.max_parameter_age_cycles != self.config.max_parameter_age_cycles:
            raise ValueError("policy and recovery max parameter ages must match")
        if bank.max_payload_age_cycles != self.config.max_parameter_age_cycles:
            raise ValueError("policy and bank max parameter ages must match")
        self.supervisor = ClosedLoopFaultRecoverySupervisor(
            (trusted_static_image,), config=recovery, bank_config=bank
        )
        self._trusted_static = trusted_static_image
        initial_semantics = parameter_image_semantics_sha256(trusted_static_image)
        self._semantics_decision_source: dict[str, str] = {
            initial_semantics: "initial_static_calibration"
        }
        self._posterior: ObservedRegimePosterior | None = None
        self._posterior_class = POSTERIOR_UNCERTAIN
        self._last_posterior_window_id = -1
        self._last_posterior_source_end: int | None = None
        self._last_cycle = self.supervisor.fast_path.contract.map_pipeline_cycles - 1
        self._bank_evidence_id = 0
        self._evidence_key = "initial"
        self._evidence_run = 0
        self._fault_latched = True
        self._commit_gate_open = False
        self._staged_version: int | None = None
        self._staged_semantics: str | None = None
        self._staged_purpose: str | None = None
        self._last_integrity_republish_version: int | None = None
        self._required_recovery_kind: str | None = None
        self._required_recovery_image: ParametricMAPLUTImage | None = None
        self._required_recovery_requires_republish = False
        self._last_action: RouteAAction | None = None
        self._history: list[RouteAAction] = []

    @property
    def history(self) -> tuple[RouteAAction, ...]:
        return tuple(self._history)

    @property
    def last_action(self) -> RouteAAction | None:
        return self._last_action

    def _validate_posterior(self, evidence: ObservedRegimePosterior, cycle: int) -> str:
        if evidence.available_cycle != cycle:
            raise ValueError("posterior must be delivered exactly at available_cycle")
        if evidence.source_window_id <= self._last_posterior_window_id:
            raise ValueError("posterior source windows must be strictly increasing")
        if evidence.source_end_cycle - evidence.source_start_cycle + 1 != self.config.regime_window_cycles:
            raise ValueError("posterior source window length violates the common cadence")
        if evidence.available_cycle != evidence.source_end_cycle + 1:
            raise ValueError("posterior must use the frozen one-cycle causal availability delay")
        if (
            self._last_posterior_source_end is not None
            and evidence.source_start_cycle != self._last_posterior_source_end + 1
        ):
            raise ValueError("posterior source windows must be contiguous without overlap or gaps")
        if abs(sum(evidence.probabilities) - 1.0) > self.config.posterior_sum_tolerance:
            raise ValueError("posterior probabilities do not sum to one")
        if (
            self.config.posterior_model_sha256 is not None
            and evidence.model_sha256 != self.config.posterior_model_sha256
        ):
            raise ValueError("posterior model hash does not match the frozen threshold lock")
        named = evidence.named()
        tail = named["calibration_shift"] + named["burst"]
        adaptive = named["normal"] + named["smooth"]
        peak = max(named.values())
        uncertainty = 1.0 - peak
        if tail >= self.config.regime_posterior_enter_min:
            return TAIL_TRUSTED
        if (
            self._posterior_class == TAIL_TRUSTED
            and tail > self.config.regime_posterior_exit_max
        ):
            return TAIL_TRUSTED
        if (
            adaptive >= self.config.regime_posterior_enter_min
            and uncertainty < self.config.uncertainty_fallback_min
        ):
            return ADAPTIVE_OPEN
        return POSTERIOR_UNCERTAIN

    def _integrity_reason(self, event: ClosedLoopCycleInput) -> str | None:
        active = self.supervisor.active_image
        if not event.observation_valid:
            return "observation_invalid"
        if not event.input_crc_ok:
            return "input_crc_mismatch"
        if not event.reported_integrity_ok:
            return "image_integrity_mismatch"
        if not event.deadline_ok:
            return "deadline_miss"
        if (
            event.reported_active_bank_version is not None
            and event.reported_active_bank_version != active.active_bank_version
        ):
            return "bank_version_mismatch"
        if (
            event.reported_parameter_age_cycles is not None
            and event.reported_parameter_age_cycles > self.config.max_parameter_age_cycles
        ):
            return "parameter_stale"
        return None

    def _observe_selection(self, key: str, *, eligible: bool) -> None:
        self._bank_evidence_id += 1
        self.supervisor.observe_selection(
            window_id=self._bank_evidence_id,
            selection_key=key,
            eligible=eligible,
        )
        if eligible and key == self._evidence_key:
            self._evidence_run += 1
        elif eligible:
            self._evidence_key = key
            self._evidence_run = 1
        else:
            self._evidence_key = key
            self._evidence_run = 0

    def _candidate_budget_violation(self, candidate: AdaptiveMAPCandidate) -> bool:
        return bool(
            candidate.update_macs > self.config.max_update_macs
            or candidate.private_state_bytes > self.config.max_private_state_bytes
            or candidate.transient_workspace_bytes > self.config.max_workspace_bytes
            or candidate.host_update_wallclock_us
            > self.config.max_host_update_wallclock_us
        )

    def _validate_candidate(self, candidate: AdaptiveMAPCandidate, cycle: int) -> str | None:
        if candidate.available_cycle != cycle:
            raise ValueError("candidate must be delivered exactly at available_cycle")
        if candidate.source_end_cycle >= cycle:
            raise ValueError("candidate may not consume the current or future cycle")
        expected_due = (
            candidate.source_end_cycle + 1
        ) % self.config.parameter_update_period_cycles == 0
        if not expected_due:
            return "candidate_source_not_common_parameter_boundary"
        if self._candidate_budget_violation(candidate):
            return "candidate_matched_budget_violation"
        if candidate.update_macs != self.config.dual_shadow_update_macs:
            return "candidate_does_not_account_for_full_dual_shadow_router"
        if candidate.router_algorithm_sha256 != self.config.router_algorithm_sha256:
            return "candidate_router_algorithm_hash_mismatch"
        if candidate.image.config != self._trusted_static.config:
            return "candidate_fixed_point_config_mismatch"
        return None

    def _stage_semantics(
        self,
        prototype: ParametricMAPLUTImage,
        *,
        cycle: int,
        source_window_id: int,
        selection_key: str,
        purpose: str,
    ) -> tuple[bool, str, int | None]:
        if self._staged_version is not None:
            return False, "another_image_is_pending", self._staged_version
        version = self.supervisor.bank.active_version + 1
        image = clone_parameter_image(prototype, active_bank_version=version)
        self.supervisor.register_image(image)
        attempt = self.supervisor.submit_update(
            image,
            transaction_id=f"route-a-{purpose}-v{version}-c{cycle}",
            selection_key=selection_key,
            # Atomic-bank hysteresis and candidate-estimator windows have
            # different cadences.  The manifest binds the bank-selection
            # evidence sequence; the estimator source window remains in the
            # typed candidate/action provenance.
            source_window_id=max(1, self._bank_evidence_id),
            created_epoch=cycle,
            apply_epoch=cycle + 1,
            purpose="candidate",
        )
        if not attempt.accepted:
            self.supervisor.unregister_unactivated_image(version)
            return False, attempt.reason, None
        semantics = parameter_image_semantics_sha256(image)
        if purpose == "window_map":
            self._semantics_decision_source[semantics] = "window_shadow"
        elif purpose == "ewma_adaptive_map":
            self._semantics_decision_source[semantics] = "trusted_ewma_shadow"
        self._staged_version = version
        self._staged_semantics = semantics
        self._staged_purpose = purpose
        return True, "staged", version

    def step(self, event: RouteACycleInput) -> RouteAAction:
        if not isinstance(event, RouteACycleInput):
            raise TypeError("event must be RouteACycleInput")
        cycle = event.fast_path.epoch
        if cycle != self._last_cycle + 1:
            raise ValueError("cycle_index must be sequential with no replay or gaps")
        self._last_cycle = cycle
        if event.parameter_update_due != (
            cycle % self.config.parameter_update_period_cycles == 0
        ):
            raise ValueError("parameter_update_due violates the common cadence")

        posterior_arrived = event.posterior_update is not None
        if posterior_arrived:
            assert event.posterior_update is not None
            self._posterior_class = self._validate_posterior(event.posterior_update, cycle)
            self._posterior = event.posterior_update
            self._last_posterior_window_id = event.posterior_update.source_window_id
            self._last_posterior_source_end = event.posterior_update.source_end_cycle

        integrity_reason = self._integrity_reason(event.fast_path)
        if (
            integrity_reason is None
            and self.config.event_model_sha256 is not None
            and event.event_model_sha256 != self.config.event_model_sha256
        ):
            integrity_reason = "event_model_hash_mismatch"
        leakage = "leakage" in (
            event.fast_path.syndrome_x,
            event.fast_path.syndrome_z,
        )
        observed_event_alert = (
            event.fast_path.ood_score_code > self.config.ood_score_min_code
        )
        if leakage:
            requested_mode = LEAKAGE_RESET
            primary_reason = "leakage_observed"
        elif integrity_reason is not None:
            requested_mode = INTEGRITY_ROLLBACK
            primary_reason = integrity_reason
        elif observed_event_alert:
            requested_mode = TAIL_TRUSTED
            primary_reason = "observed_event_score_exceeded"
        else:
            requested_mode = self._posterior_class
            primary_reason = {
                ADAPTIVE_OPEN: "normal_or_smooth_posterior",
                TAIL_TRUSTED: "calibration_shift_or_burst_posterior",
                POSTERIOR_UNCERTAIN: "posterior_uncertain",
            }[requested_mode]

        posterior_available = self._posterior is not None
        # Posterior tail/uncertainty is an algorithm-selection signal, not an
        # integrity failure.  V4 routes those states to the continuously
        # updated, validated EWMA shadow.  Only CRC/version/age/deadline/model
        # integrity failures are allowed to request an LKG rollback.
        if requested_mode == INTEGRITY_ROLLBACK:
            if self._required_recovery_kind is None:
                self._required_recovery_kind = "lkg_rollback"
                self._required_recovery_image = self.supervisor.last_known_good_image
            self._required_recovery_requires_republish = True

        selection_key = {
            ADAPTIVE_OPEN: "adaptive_candidate",
            TAIL_TRUSTED: "trusted_ewma_shadow",
            POSTERIOR_UNCERTAIN: "trusted_ewma_shadow",
            INTEGRITY_ROLLBACK: "last_known_good",
            LEAKAGE_RESET: "leakage_reset",
        }[requested_mode]
        if self._required_recovery_kind == "lkg_rollback":
            selection_key = "last_known_good"
        if posterior_arrived:
            self._observe_selection(
                selection_key, eligible=requested_mode != LEAKAGE_RESET
            )
        elif requested_mode in (INTEGRITY_ROLLBACK, LEAKAGE_RESET):
            self._observe_selection(
                selection_key, eligible=requested_mode == INTEGRITY_ROLLBACK
            )
        elif (
            self._required_recovery_kind == "lkg_rollback"
            and self._required_recovery_requires_republish
        ):
            self._observe_selection(selection_key, eligible=True)

        if requested_mode != ADAPTIVE_OPEN or self._required_recovery_kind is not None:
            self._fault_latched = True
            self._commit_gate_open = False
        elif posterior_arrived:
            ack_clear = (
                self._last_action is None
                or self._last_action.fast_path_record.awaiting_readback_version is None
            )
            if ack_clear and self._evidence_run >= self.config.recovery_hysteresis_windows:
                self._fault_latched = False
                self._commit_gate_open = True

        active_semantics_before = parameter_image_semantics_sha256(
            self.supervisor.active_image
        )
        lkg_semantics = self.supervisor.last_known_good_semantics_sha256
        target_semantics = None
        if requested_mode == INTEGRITY_ROLLBACK:
            target_semantics = lkg_semantics
        elif self._required_recovery_kind == "lkg_rollback":
            assert self._required_recovery_image is not None
            target_semantics = parameter_image_semantics_sha256(
                self._required_recovery_image
            )
        target_complete_before = target_semantics is None or active_semantics_before == target_semantics
        if self._required_recovery_requires_republish:
            target_complete_before = bool(
                target_complete_before
                and self._last_integrity_republish_version
                == self.supervisor.active_image.active_bank_version
            )

        # A pending candidate is invalidated before its commit whenever the
        # policy leaves the adaptive state.  Atomic-bank hysteresis then rejects
        # it rather than allowing a one-cycle race to publish stale semantics.
        if (
            self._staged_version is not None
            and (
                (
                    self._staged_purpose == "window_map"
                    and requested_mode != ADAPTIVE_OPEN
                )
                or (
                    self._staged_purpose in CANDIDATE_METHODS
                    and requested_mode in (INTEGRITY_ROLLBACK, LEAKAGE_RESET)
                )
            )
        ):
            self._observe_selection(selection_key, eligible=False)

        force_safe_hold = bool(
            (
                self._required_recovery_kind == "lkg_rollback"
            )
            and not target_complete_before
        )
        effective_fast_path = event.fast_path
        effective_fast_path = replace(
            effective_fast_path,
            allow_lkg_promotion=bool(
                effective_fast_path.allow_lkg_promotion
                and requested_mode == ADAPTIVE_OPEN
                and self._commit_gate_open
            ),
        )
        effective_fast_path = replace(
            effective_fast_path,
            ood_score_code=(
                255
                if observed_event_alert
                else 0
            ),
        )
        if force_safe_hold:
            effective_fast_path = replace(effective_fast_path, ood_score_code=255)

        record = self.supervisor.tick(effective_fast_path)
        completed_stage_purpose = self._staged_purpose
        rejected_staged = record.commit_status == "rejected" and self._staged_version is not None
        if rejected_staged:
            assert self._staged_version is not None
            rejected_version = self._staged_version
            self.supervisor.unregister_unactivated_image(rejected_version)
            self._staged_version = None
            self._staged_semantics = None
            self._staged_purpose = None
        elif record.commit_status == "committed":
            if completed_stage_purpose == "lkg_rollback":
                self._last_integrity_republish_version = record.active_version
            self._staged_version = None
            self._staged_semantics = None
            self._staged_purpose = None

        active_semantics = record.active_semantics_sha256
        target_complete = target_semantics is None or active_semantics == target_semantics
        if self._required_recovery_requires_republish:
            target_complete = bool(
                target_complete
                and self._last_integrity_republish_version == record.active_version
            )
        rollback_requested = bool(
            (
                requested_mode == INTEGRITY_ROLLBACK
            )
            or self._required_recovery_kind == "lkg_rollback"
            or completed_stage_purpose == "lkg_rollback"
            or self._staged_purpose == "lkg_rollback"
        )
        trusted_requested = bool(
            requested_mode in (TAIL_TRUSTED, POSTERIOR_UNCERTAIN)
            or completed_stage_purpose == "ewma_adaptive_map"
            or self._staged_purpose == "ewma_adaptive_map"
        )

        candidate_received = event.candidate is not None
        candidate_accepted = False
        safety_stage_accepted = False
        candidate_reason = "not_supplied"
        staged_version: int | None = None
        if event.candidate is not None:
            candidate_reason = self._validate_candidate(event.candidate, cycle) or "eligible"
            if candidate_reason == "eligible":
                candidate = event.candidate
                trusted_ewma = candidate.method_id == "ewma_adaptive_map"
                window_router_pass = bool(
                    candidate.method_id == "window_map"
                    and self._posterior is not None
                    and self._posterior.named()["smooth"]
                    >= self.config.smooth_bank_posterior_min
                    and candidate.window_prequential_score
                    < candidate.ewma_prequential_score
                )
                if requested_mode in (INTEGRITY_ROLLBACK, LEAKAGE_RESET):
                    candidate_reason = "online_update_frozen_by_integrity_or_reset"
                elif requested_mode in (TAIL_TRUSTED, POSTERIOR_UNCERTAIN) and not trusted_ewma:
                    candidate_reason = "tail_or_uncertain_requires_trusted_ewma_shadow"
                elif candidate.method_id == "window_map" and not window_router_pass:
                    candidate_reason = "window_router_proof_failed"
                elif (
                    candidate.method_id == "window_map"
                    and (requested_mode != ADAPTIVE_OPEN or not self._commit_gate_open)
                ):
                    candidate_reason = "window_promotion_gate_closed"
                elif record.awaiting_readback_version is not None:
                    candidate_reason = "commit_gate_closed_pending_readback"
                else:
                    candidate_accepted, candidate_reason, staged_version = self._stage_semantics(
                        candidate.image,
                        cycle=cycle,
                        source_window_id=candidate.source_window_id,
                        selection_key=(
                            "trusted_ewma_shadow"
                            if trusted_ewma
                            else "adaptive_candidate"
                        ),
                        purpose=candidate.method_id,
                    )

        emergency_ready = self._evidence_run >= self.config.enter_hysteresis_windows
        recovery_stage_requested = bool(
            requested_mode == INTEGRITY_ROLLBACK
            or self._required_recovery_kind == "lkg_rollback"
        )
        if (
            not candidate_accepted
            and recovery_stage_requested
            and not target_complete
            and emergency_ready
            and record.awaiting_readback_version is None
            and self._staged_version is None
        ):
            prototype = (
                self.supervisor.last_known_good_image
                if self._required_recovery_image is None
                else self._required_recovery_image
            )
            recovery_purpose = "lkg_rollback"
            safety_stage_accepted, stage_reason, staged_version = self._stage_semantics(
                prototype,
                cycle=cycle,
                source_window_id=(
                    self._posterior.source_window_id if self._posterior is not None else 1
                ),
                selection_key=selection_key,
                purpose=recovery_purpose,
            )
            candidate_reason = stage_reason

        posterior = (
            (0.25, 0.25, 0.25, 0.25)
            if self._posterior is None
            else self._posterior.probabilities
        )
        host_candidate = event.candidate
        budget_violation = bool(
            host_candidate is not None and self._candidate_budget_violation(host_candidate)
        )
        deadline = RouteADeadlineLedger(
            source_cycle=record.source_cycle,
            action_cycle=record.action_cycle,
            action_deadline_cycle=record.source_cycle
            + self.config.fast_action_latency_cycles,
            source_to_action_cycles=record.source_to_action_cycles,
            fast_deadline_miss=bool(
                not event.fast_path.deadline_ok
                or record.source_to_action_cycles
                != self.config.fast_action_latency_cycles
            ),
            host_update_due=event.parameter_update_due,
            host_update_macs=0 if host_candidate is None else host_candidate.update_macs,
            host_private_state_bytes=(
                0 if host_candidate is None else host_candidate.private_state_bytes
            ),
            host_workspace_bytes=(
                0 if host_candidate is None else host_candidate.transient_workspace_bytes
            ),
            host_update_wallclock_us=(
                0.0 if host_candidate is None else host_candidate.host_update_wallclock_us
            ),
            host_budget_violation=budget_violation,
        )
        if not posterior_available:
            policy_mode = RECOVERING
        elif (
            requested_mode == ADAPTIVE_OPEN
            and (self._fault_latched or self._required_recovery_kind is not None)
        ):
            policy_mode = RECOVERING
        else:
            policy_mode = requested_mode
        decision_source = self._semantics_decision_source.get(
            active_semantics,
            "last_known_good" if active_semantics == lkg_semantics else "unknown_validated_bank",
        )
        reason_trace = "|".join(
            (
                f"policy:{primary_reason}",
                f"mode:{policy_mode}",
                f"selection:{selection_key}",
                f"fast:{record.reason_trace}",
                f"candidate:{candidate_reason}",
            )
        )
        action = RouteAAction(
            cycle_index=cycle,
            policy_mode=policy_mode,
            primary_reason=primary_reason,
            reason_trace=reason_trace,
            posterior=posterior,
            posterior_source_window_id=(
                None if self._posterior is None else self._posterior.source_window_id
            ),
            posterior_model_sha256=(
                None if self._posterior is None else self._posterior.model_sha256
            ),
            active_bank=record.active_bank,
            active_bank_version=record.active_version,
            active_semantics_sha256=active_semantics,
            decision_source=decision_source,
            online_update_frozen=policy_mode != ADAPTIVE_OPEN,
            commit_gate_open=self._commit_gate_open,
            recovery_evidence_run=self._evidence_run,
            fallback_active=bool(
                requested_mode != ADAPTIVE_OPEN
                or record.conservative_action != "use_validated_map"
            ),
            reset_request=record.reset_request,
            rollback_requested=rollback_requested,
            rollback_completed=bool(
                rollback_requested
                and (
                    target_complete
                    or (
                        completed_stage_purpose == "lkg_rollback"
                        and record.commit_status == "committed"
                    )
                )
            ),
            trusted_switch_requested=trusted_requested,
            trusted_switch_completed=bool(
                trusted_requested
                and (
                    decision_source == "trusted_ewma_shadow"
                    or (
                        completed_stage_purpose == "ewma_adaptive_map"
                        and record.commit_status == "committed"
                    )
                )
            ),
            candidate_received=candidate_received,
            candidate_accepted=candidate_accepted,
            candidate_reason=candidate_reason,
            safety_stage_accepted=safety_stage_accepted,
            staged_version=staged_version,
            commit_status=record.commit_status,
            commit_reason=record.commit_reason,
            readback_status=record.readback_status,
            deadline=deadline,
            fast_path_record=record,
        )
        self._last_action = action
        self._history.append(action)
        if action.rollback_completed and self._required_recovery_kind == "lkg_rollback":
            self._required_recovery_kind = None
            self._required_recovery_image = None
            self._required_recovery_requires_republish = False
        return action


__all__ = [
    "ADAPTIVE_OPEN",
    "CANDIDATE_METHODS",
    "DUAL_BANK_ROUTER_ALGORITHM_SHA256",
    "INTEGRITY_ROLLBACK",
    "LEAKAGE_RESET",
    "MODEL_SCOPE",
    "POLICY_MODES",
    "POSTERIOR_UNCERTAIN",
    "RECOVERING",
    "REGIME_CLASSES",
    "TAIL_TRUSTED",
    "AdaptiveMAPCandidate",
    "ObservedRegimePosterior",
    "RegimeAwareSafeAdaptivePolicy",
    "RouteAAction",
    "RouteACycleInput",
    "RouteADeadlineLedger",
    "RouteAPolicyConfig",
    "clone_parameter_image",
]
