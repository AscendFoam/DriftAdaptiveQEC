from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from cnn_fpga.data.experimental_history import (
    DeployableLLRContext,
    ExperimentalHistoryBuilder,
    ExperimentalHistoryConfig,
    HistoryRuntimeStatus,
    ObservedActionRecord,
)
from cnn_fpga.decoder.hybrid_state_output import (
    CONTINUOUS_PARAMETER_NAMES,
    FORBIDDEN_DIRECT_OUTPUT_TOKENS,
    ContinuousNoiseCalibration,
    HybridStateEstimator,
    HybridStateEstimatorConfig,
    LeakageRecoveryOutput,
    ParameterBankRecommendation,
    RegimePosteriorOutput,
    UncertaintyOutput,
    schema_provenance,
    stage_parameter_bank_recommendation,
)
from cnn_fpga.runtime.param_bank import DecoderRuntimeParams, ParamBank
from physics.drift_processes import DriftState
from physics.sbs_observation_reset import PairedSyndrome
from physics.syndrome_stream import ObservedSyndromeStep


def _history(*, cycles: int = 64, eventful: bool = False, faulty: bool = False):
    builder = ExperimentalHistoryBuilder(ExperimentalHistoryConfig(history_cycles=cycles))
    x_run = z_run = leakage_run = 0
    sample = None
    for cycle in range(cycles):
        if eventful and cycle % 13 in (0, 1, 2):
            x, z = "leakage", "leakage"
        else:
            x = "e" if eventful and cycle % 5 in (0, 1) else "g"
            z = "e" if eventful and cycle % 7 in (0, 1) else "g"
        x_run = x_run + 1 if x == "e" else 0
        z_run = z_run + 1 if z == "e" else 0
        leakage_run = leakage_run + 1 if "leakage" in (x, z) else 0
        residual = (
            0.18 * np.sin(2.0 * np.pi * cycle / 29.0) + (0.22 if eventful else 0.0),
            -0.14 * np.cos(2.0 * np.pi * cycle / 31.0),
        )
        observed = ObservedSyndromeStep(
            cycle_index=cycle,
            drift_step=cycle,
            time=float(cycle),
            analog_syndrome=residual,
            residual_syndrome=residual,
            syndrome=PairedSyndrome(x=x, z=z),
            quadrature_phases_rad=(0.0, np.pi / 2.0),
            x_e_run=x_run,
            z_e_run=z_run,
            leakage_run=leakage_run,
        )
        runtime = HistoryRuntimeStatus(
            cycle_index=cycle,
            fast_deadline_ok=not faulty or cycle % 3 != 0,
            slow_deadline_ok=not faulty or cycle % 4 != 0,
            communication_available=not faulty or cycle % 5 != 0,
            update_status="none",
            update_applied=False,
            pending_update=False,
            active_bank_version=0,
            pending_window_count=0,
            crc_ok=not faulty or cycle % 2 != 0,
        )
        sample = builder.append(
            observed,
            ObservedActionRecord.neutral(cycle),
            DeployableLLRContext((0.35, 0.35)),
            runtime,
        )
    assert sample is not None
    return sample


def _config(**updates: object) -> HybridStateEstimatorConfig:
    values: dict[str, object] = {
        "minimum_samples": 32,
        "bootstrap_replicates": 32,
        "block_length_cycles": 8,
        "calibration_scope": "registered_synthetic_pilot",
        "bootstrap_seed": 77,
    }
    values.update(updates)
    return HybridStateEstimatorConfig(**values)


def _output(*, eventful: bool = False, faulty: bool = False, config=None, bank=None):
    actual_bank = ParamBank() if bank is None else bank
    estimator = HybridStateEstimator(_config() if config is None else config)
    return estimator.estimate(
        _history(eventful=eventful, faulty=faulty),
        (0.82, 0.08, 0.06, 0.04),
        regime_source="t4.1.1_registered_gaussian_hmm",
        active_params=actual_bank.read_active(),
        active_bank_version=actual_bank.active_version,
    )


@pytest.mark.parametrize(
    "factory,match",
    [
        (
            lambda: ContinuousNoiseCalibration(0, 0, 0.2, 0.2, 1.0, 0, 0, 0, 0),
            "rho_qp",
        ),
        (lambda: RegimePosteriorOutput((0.4, 0.4, 0.4, 0.0), "observed_fallback_prior"), "normalized"),
        (
            lambda: LeakageRecoveryOutput(0.5, 0.4, 2, (1.0, 0.0), 1),
            "cannot be below",
        ),
        (lambda: HybridStateEstimatorConfig(minimum_samples=7), "at least 8"),
        (
            lambda: HybridStateEstimatorConfig(minimum_samples=16, block_length_cycles=17),
            "cannot exceed",
        ),
    ],
)
def test_component_contracts_fail_closed(factory: object, match: str) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        factory()  # type: ignore[operator]


def test_schema_contains_every_required_output_and_no_cycle_action() -> None:
    provenance = schema_provenance()
    assert provenance["has_forbidden_direct_output"] is False
    assert len(provenance["fields"]["continuous"]) == len(CONTINUOUS_PARAMETER_NAMES)
    assert provenance["fields"]["regime_posterior"] == [
        "normal",
        "burst",
        "leakage",
        "calibration_shift",
    ]
    assert "recovery-burden" in provenance["recovery_semantics"]


def test_estimator_output_is_normalized_psd_future_only_and_deployable() -> None:
    output = _output()
    payload = output.to_deployable_dict()
    covariance = np.asarray(output.uncertainty.continuous_covariance)

    assert output.as_of_cycle == 63
    assert output.history_start_cycle == 0
    assert output.parameter_bank_recommendation.valid_from_cycle == 64
    assert np.min(np.linalg.eigvalsh(covariance)) >= -1.0e-12
    assert sum(output.regime.probabilities) == pytest.approx(1.0)
    assert sum(output.risk.recovery_burden_posterior) == pytest.approx(1.0)
    normalized_keys = str(payload).lower().replace("_", "")
    assert not any(token in normalized_keys for token in FORBIDDEN_DIRECT_OUTPUT_TOKENS)
    assert payload["leakage_recovery_estimate"]["truth_semantics"] is False


def test_moving_block_uncertainty_is_reproducible_across_fresh_estimators() -> None:
    first = _output()
    second = _output()
    np.testing.assert_allclose(
        first.uncertainty.continuous_covariance,
        second.uncertainty.continuous_covariance,
        rtol=0.0,
        atol=0.0,
    )
    assert first.parameter_bank_recommendation == second.parameter_bank_recommendation


def test_observed_event_history_increases_leakage_and_recovery_burden() -> None:
    calm = _output()
    eventful = _output(eventful=True)
    assert eventful.risk.leakage_probability_next_cycle > calm.risk.leakage_probability_next_cycle
    assert eventful.risk.leakage_probability_horizon > calm.risk.leakage_probability_horizon
    assert eventful.risk.expected_recovery_depth > calm.risk.expected_recovery_depth


def test_uncalibrated_or_faulty_output_holds_active_bank() -> None:
    uncalibrated = _output(config=_config(calibration_scope="uncalibrated_contract"))
    faulty = _output(faulty=True)
    assert uncalibrated.parameter_bank_recommendation.bank_action == "hold_active"
    assert "calibration_scope" in uncalibrated.parameter_bank_recommendation.hold_reason
    assert faulty.parameter_bank_recommendation.bank_action == "hold_active"
    assert "ood_gate" in faulty.parameter_bank_recommendation.hold_reason
    with pytest.raises(ValueError, match="must not be staged"):
        stage_parameter_bank_recommendation(uncalibrated, ParamBank(), staged_cycle=63)


def test_stage_then_commit_is_atomic_and_matches_recommendation() -> None:
    bank = ParamBank()
    output = _output(bank=bank, config=_config(ood_hold_threshold=1.0))
    recommendation = output.parameter_bank_recommendation
    assert recommendation.bank_action == "stage_candidate"

    pending = stage_parameter_bank_recommendation(output, bank, staged_cycle=63)
    assert pending.commit_epoch == 64
    assert bank.active_version == 0
    assert bank.commit_if_ready(63) is None
    result = bank.commit_if_ready(64)
    assert result is not None and result.version == 1
    np.testing.assert_allclose(bank.read_active().K, recommendation.gain_matrix)
    np.testing.assert_allclose(bank.read_active().b, recommendation.bias)


def test_stale_version_wrong_cycle_and_crc_corruption_are_rejected() -> None:
    bank = ParamBank()
    output = _output(bank=bank, config=_config(ood_hold_threshold=1.0))
    with pytest.raises(ValueError, match="as_of_cycle"):
        stage_parameter_bank_recommendation(output, bank, staged_cycle=62)

    stage_parameter_bank_recommendation(output, bank, staged_cycle=63)
    bank.commit_if_ready(64)
    with pytest.raises(ValueError, match="stale"):
        stage_parameter_bank_recommendation(output, bank, staged_cycle=63)

    recommendation = output.parameter_bank_recommendation
    with pytest.raises(ValueError, match="crc32"):
        replace(recommendation, payload_crc32=recommendation.payload_crc32 ^ 1)


def test_truth_object_in_output_provenance_is_rejected() -> None:
    output = _output()
    with pytest.raises(ValueError, match="forbidden truth object"):
        replace(output, provenance={"safe": DriftState()})


def test_estimator_rejects_short_history_and_bad_posterior() -> None:
    short = _history(cycles=16)
    estimator = HybridStateEstimator(_config())
    with pytest.raises(ValueError, match="fewer valid"):
        estimator.estimate(
            short,
            (0.7, 0.1, 0.1, 0.1),
            regime_source="observed_fallback_prior",
            active_params=DecoderRuntimeParams.identity(),
            active_bank_version=0,
        )
    with pytest.raises(ValueError, match="normalized"):
        estimator.estimate(
            _history(),
            (0.7, 0.2, 0.2, 0.1),
            regime_source="observed_fallback_prior",
            active_params=DecoderRuntimeParams.identity(),
            active_bank_version=0,
        )


def test_uncertainty_rejects_indefinite_covariance() -> None:
    size = len(CONTINUOUS_PARAMETER_NAMES)
    matrix = np.eye(size)
    matrix[0, 0] = -1.0
    with pytest.raises(ValueError, match="positive semidefinite"):
        UncertaintyOutput(
            continuous_covariance=tuple(tuple(row) for row in matrix),
            sample_count=32,
            bootstrap_replicates=32,
            block_length_cycles=8,
            regime_entropy_nats=0.1,
            leakage_entropy_nats=0.1,
            recovery_entropy_nats=0.1,
            ood_score=0.1,
            recommendation_confidence=0.8,
            calibration_scope="uncalibrated_contract",
        )


def test_parameter_recommendation_rejects_non_symmetric_gain() -> None:
    with pytest.raises(ValueError, match="symmetric"):
        ParameterBankRecommendation.create(
            bank_action="stage_candidate",
            recommended_mode="normal",
            gain_matrix=((0.5, 0.2), (0.0, 0.5)),
            bias=(0.0, 0.0),
            base_active_version=0,
            valid_from_cycle=1,
            expires_after_cycle=2,
            recommendation_id="bad",
            calibration_scope="registered_synthetic_pilot",
        )
