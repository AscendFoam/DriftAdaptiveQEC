from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest

from physics import phase9_backend_a as backend_a
from physics import phase9_backend_b as backend_b
from physics import phase9_high_cutoff_runtime_adapter as adapter


ROOT = Path(__file__).resolve().parents[1]


def _source_sha(path: str) -> str:
    return sha256((ROOT / path).read_bytes()).hexdigest()


@pytest.fixture
def verified_native_modules():
    original_a = backend_a.MAX_SUPPORTED_CUTOFF
    original_b = backend_b.MAX_SUPPORTED_CUTOFF
    old_a_attestation = getattr(
        backend_a, "__verified_source_sha256__", None
    )
    old_b_attestation = getattr(
        backend_b, "__verified_source_sha256__", None
    )
    backend_a.__verified_source_sha256__ = _source_sha(
        "physics/phase9_backend_a.py"
    )
    backend_b.__verified_source_sha256__ = _source_sha(
        "physics/phase9_backend_b.py"
    )
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


def test_adapter_extends_only_verified_module_objects_and_rejects_37(
    verified_native_modules,
) -> None:
    module_a, module_b = verified_native_modules
    receipt = adapter.enable_verified_high_cutoff(module_a, module_b)
    adapter.assert_verified_high_cutoff(module_a, module_b, receipt)

    assert module_a.MAX_SUPPORTED_CUTOFF == 36
    assert module_b.MAX_SUPPORTED_CUTOFF == 36
    assert receipt["numerical_kernel_source_changed"] is False
    assert receipt["claim_boundary"] == adapter.CLAIM_BOUNDARY
    with pytest.raises(ValueError):
        module_a.BackendAConfig(cutoff=37)
    with pytest.raises(ValueError):
        module_b.BackendBConfig(cutoff=37)


def test_cutoff_36_executes_full_step_in_both_independent_kernels(
    verified_native_modules,
) -> None:
    module_a, module_b = verified_native_modules
    adapter.enable_verified_high_cutoff(module_a, module_b)

    config_a = module_a.BackendAConfig(
        cutoff=36,
        substeps_per_segment=1,
        iq_samples=1,
        logical_grid_points=1025,
    )
    simulator_a = module_a.Phase9BackendASimulator(config_a)
    result_a = simulator_a.step(
        simulator_a.initialize_fock(),
        module_a.diagnostic_action_word("XZ"),
        module_a.backend_a_exogenous(seed=991, round_index=0, iq_samples=1),
    )

    config_b = module_b.BackendBConfig(
        cutoff=36,
        split_steps_per_segment=1,
        iq_samples=1,
    )
    simulator_b = module_b.Phase9BackendBSimulator(config_b)
    result_b = simulator_b.step(
        simulator_b.initialize_fock(),
        module_b.diagnostic_action_word_b("XZ"),
        module_b.backend_b_random_record(
            seed=992, round_index=0, iq_samples=1
        ),
    )

    for state in (result_a.state, result_b.state):
        assert state.cutoff == 36
        assert state.joint_density.shape == (108, 108)
        assert np.isclose(np.trace(state.joint_density), 1.0, atol=1.0e-8)
        assert np.linalg.eigvalsh(
            (state.joint_density + state.joint_density.conj().T) / 2
        ).min() >= -5.0e-8


def test_adapter_rejects_unverified_or_non_native_starting_contract(
    verified_native_modules,
) -> None:
    module_a, module_b = verified_native_modules
    saved = module_a.__verified_source_sha256__
    delattr(module_a, "__verified_source_sha256__")
    with pytest.raises(RuntimeError, match="unverified"):
        adapter.enable_verified_high_cutoff(module_a, module_b)
    module_a.__verified_source_sha256__ = saved
    module_b.MAX_SUPPORTED_CUTOFF = 31
    with pytest.raises(RuntimeError, match="native backend-B"):
        adapter.enable_verified_high_cutoff(module_a, module_b)


def test_adapter_receipt_mutations_fail_closed(verified_native_modules) -> None:
    module_a, module_b = verified_native_modules
    receipt = adapter.enable_verified_high_cutoff(module_a, module_b)
    for key, replacement in (
        ("extended_max_supported_cutoff", 40),
        ("numerical_kernel_source_changed", True),
        ("cutoff_37_rejected", False),
        ("analysis_sha256", "0" * 64),
    ):
        mutated = deepcopy(receipt)
        mutated[key] = replacement
        with pytest.raises(RuntimeError, match="attestation drift"):
            adapter.assert_verified_high_cutoff(module_a, module_b, mutated)
