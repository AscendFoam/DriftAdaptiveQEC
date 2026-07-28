from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest

from cnn_fpga.benchmark.phase9_dual_backend_qualification import _action_words
from physics import phase9_backend_a as backend_a
from physics import phase9_backend_b as backend_b
from physics import phase9_cutoff44_runtime_adapter as adapter
from physics.phase9_reset_rao_blackwell import (
    ESTIMAND_ID,
    STRESS_ROLE,
    evaluate_expected_reset,
)


ROOT = Path(__file__).resolve().parents[1]


def _source_sha(path: str) -> str:
    return sha256((ROOT / path).read_bytes()).hexdigest()


@pytest.fixture
def verified_native_modules():
    original_a = backend_a.MAX_SUPPORTED_CUTOFF
    original_b = backend_b.MAX_SUPPORTED_CUTOFF
    old_a_attestation = getattr(backend_a, "__verified_source_sha256__", None)
    old_b_attestation = getattr(backend_b, "__verified_source_sha256__", None)
    backend_a.__verified_source_sha256__ = _source_sha("physics/phase9_backend_a.py")
    backend_b.__verified_source_sha256__ = _source_sha("physics/phase9_backend_b.py")
    try:
        yield backend_a, backend_b
    finally:
        backend_a.MAX_SUPPORTED_CUTOFF = original_a
        backend_b.MAX_SUPPORTED_CUTOFF = original_b
        if old_a_attestation is None:
            delattr(backend_a, "__verified_source_sha256__")
        else:
            backend_a.__verified_source_sha256__ = old_a_attestation
        if old_b_attestation is None:
            delattr(backend_b, "__verified_source_sha256__")
        else:
            backend_b.__verified_source_sha256__ = old_b_attestation


def test_cutoff44_adapter_is_bounded_and_self_attesting(
    verified_native_modules,
) -> None:
    module_a, module_b = verified_native_modules
    receipt = adapter.enable_verified_cutoff44(module_a, module_b)
    adapter.assert_verified_cutoff44(module_a, module_b, receipt)

    assert module_a.MAX_SUPPORTED_CUTOFF == 44
    assert module_b.MAX_SUPPORTED_CUTOFF == 44
    assert receipt["first_rejected_cutoff"] == 45
    assert receipt["numerical_kernel_source_changed"] is False
    assert receipt["claim_boundary"]["automatic_extension_beyond_44"] is False
    with pytest.raises(ValueError):
        module_a.BackendAConfig(cutoff=45)
    with pytest.raises(ValueError):
        module_b.BackendBConfig(cutoff=45)


def test_cutoff44_adapter_receipt_mutations_fail_closed(
    verified_native_modules,
) -> None:
    module_a, module_b = verified_native_modules
    receipt = adapter.enable_verified_cutoff44(module_a, module_b)
    for key, replacement in (
        ("extended_max_supported_cutoff", 48),
        ("first_rejected_cutoff", 49),
        ("numerical_kernel_source_changed", True),
        ("cutoff_45_rejected", False),
        ("analysis_sha256", "0" * 64),
    ):
        mutated = deepcopy(receipt)
        mutated[key] = replacement
        with pytest.raises(RuntimeError, match="attestation drift"):
            adapter.assert_verified_cutoff44(module_a, module_b, mutated)


@pytest.mark.parametrize("backend", ["A", "B"])
def test_rao_blackwell_reset_matches_native_branch_mixture(backend: str) -> None:
    if backend == "A":
        simulator = backend_a.Phase9BackendASimulator(
            backend_a.BackendAConfig(
                cutoff=8,
                substeps_per_segment=1,
                iq_samples=2,
                logical_grid_points=1025,
            )
        )
    else:
        simulator = backend_b.Phase9BackendBSimulator(
            backend_b.BackendBConfig(
                cutoff=8,
                split_steps_per_segment=1,
                iq_samples=2,
            )
        )
    state = simulator.initialize_fock(ancilla_state="f")
    evidence = evaluate_expected_reset(
        backend=backend,
        simulator=simulator,
        state=state,
        evaluator=None,
        action=_action_words()["RESET"],
        seed=1_750_001,
    )

    assert evidence.estimand_id == ESTIMAND_ID
    assert evidence.sampled_role == STRESS_ROLE
    assert 0.0 < evidence.success_probability < 1.0
    assert evidence.success_density is not None
    assert evidence.failure_density is not None
    reconstructed = (
        evidence.success_probability * evidence.success_density
        + (1.0 - evidence.success_probability) * evidence.failure_density
    )
    assert np.allclose(evidence.expected_density, reconstructed, atol=1.0e-13)
    assert np.allclose(
        evidence.expected_levels,
        simulator.level_probabilities(evidence.expected_density),
        atol=5.0e-12,
    )
    assert np.isclose(np.trace(evidence.expected_density), 1.0)
    assert evidence.branch_trace_distance > 0.0
    assert evidence.sampled_matches_forced_branch_trace_distance <= 2.0e-10


def test_rao_blackwell_reset_handles_deterministic_success() -> None:
    simulator = backend_a.Phase9BackendASimulator(
        backend_a.BackendAConfig(
            cutoff=8,
            substeps_per_segment=1,
            iq_samples=1,
            logical_grid_points=1025,
            reset_success_e=1.0,
            reset_success_f=1.0,
        )
    )
    evidence = evaluate_expected_reset(
        backend="A",
        simulator=simulator,
        state=simulator.initialize_fock(ancilla_state="f"),
        evaluator=None,
        action=_action_words()["RESET"],
        seed=1_750_002,
    )
    assert evidence.success_probability == pytest.approx(1.0)
    assert evidence.success_density is not None
    assert evidence.failure_density is None
    assert np.array_equal(evidence.expected_density, evidence.success_density)
    assert evidence.sampled_hidden_outcome == "success"


def test_rao_blackwell_rejects_non_reset_action() -> None:
    simulator = backend_b.Phase9BackendBSimulator(
        backend_b.BackendBConfig(
            cutoff=8,
            split_steps_per_segment=1,
            iq_samples=1,
        )
    )
    with pytest.raises(ValueError, match="requires reset_request"):
        evaluate_expected_reset(
            backend="B",
            simulator=simulator,
            state=simulator.initialize_fock(ancilla_state="f"),
            evaluator=None,
            action=_action_words()["IDLE"],
            seed=1_750_003,
        )
