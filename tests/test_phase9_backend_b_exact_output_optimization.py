from __future__ import annotations

from math import sqrt

import numpy as np
import pytest
from scipy.linalg import expm

import physics.phase9_backend_b as backend_b
from physics.phase9_backend_b import (
    BackendBConfig,
    BackendBDrift,
    Phase9BackendBSimulator,
)


def _reference_split_segment(
    simulator: Phase9BackendBSimulator,
    density: backend_b.ComplexMatrix,
    duration: float,
    hamiltonian_at: object,
) -> backend_b.ComplexMatrix:
    """Pre-optimization split loop, retained as an exact-output oracle."""
    steps = simulator.config.split_steps_per_segment
    dt = duration / steps
    result = np.asarray(density, dtype=np.complex128)
    for index in range(steps):
        midpoint = (index + 0.5) / steps
        hamiltonian = np.asarray(hamiltonian_at(midpoint), dtype=np.complex128)
        half = expm(-0.5j * dt * hamiltonian)
        result = half @ result @ half.conj().T
        result = simulator._noise_channels(result, dt)
        result = half @ result @ half.conj().T
    raw_trace = complex(np.trace(result))
    result = result / raw_trace.real
    return backend_b._density(
        result,
        simulator.dimension,
        "reference_split_output",
        tolerance=1.0e-8,
    )


def test_constant_hamiltonian_reuses_expm_and_is_byte_identical(monkeypatch):
    simulator = Phase9BackendBSimulator(
        BackendBConfig(cutoff=6, split_steps_per_segment=8, iq_samples=1)
    )
    density = simulator.initialize_fock().joint_density
    hamiltonian = simulator._base_hamiltonian(BackendBDrift())
    expected = _reference_split_segment(
        simulator,
        density,
        0.08,
        lambda _fraction: hamiltonian,
    )
    calls = 0
    original = backend_b.expm

    def counted_expm(value):
        nonlocal calls
        calls += 1
        return original(value)

    monkeypatch.setattr(backend_b, "expm", counted_expm)
    actual = simulator._split_segment(
        density,
        0.08,
        lambda _fraction: hamiltonian,
    )
    assert calls == 1
    assert np.array_equal(actual, expected)


def test_varying_hamiltonian_retains_per_midpoint_expm_and_exact_output(
    monkeypatch,
):
    simulator = Phase9BackendBSimulator(
        BackendBConfig(cutoff=5, split_steps_per_segment=4, iq_samples=1)
    )
    density = simulator.initialize_fock().joint_density
    base = simulator._base_hamiltonian(BackendBDrift())
    hamiltonian_at = lambda fraction: base + fraction * simulator.joint_q
    expected = _reference_split_segment(
        simulator,
        density,
        0.07,
        hamiltonian_at,
    )
    calls = 0
    original = backend_b.expm

    def counted_expm(value):
        nonlocal calls
        calls += 1
        return original(value)

    monkeypatch.setattr(backend_b, "expm", counted_expm)
    actual = simulator._split_segment(density, 0.07, hamiltonian_at)
    assert calls == simulator.config.split_steps_per_segment
    assert np.array_equal(actual, expected)


def test_fixed_channel_terms_are_cached_readonly_and_exact():
    simulator = Phase9BackendBSimulator(BackendBConfig(cutoff=6, iq_samples=1))
    duration = 0.01
    assert simulator._pure_loss_operators(duration) is simulator._pure_loss_operators(
        duration
    )
    assert simulator._local_amplitude_operators(
        1, 0, 0.01
    ) is simulator._local_amplitude_operators(1, 0, 0.01)
    assert simulator._oscillator_dephasing_factor(
        duration
    ) is simulator._oscillator_dephasing_factor(duration)
    assert simulator._ancilla_dephasing_factor(
        duration
    ) is simulator._ancilla_dephasing_factor(duration)
    assert not simulator._pure_loss_operators(duration)[0].flags.writeable
    assert not simulator._oscillator_dephasing_factor(duration).flags.writeable

    indices = np.arange(simulator.cutoff, dtype=np.float64)
    expected_oscillator = np.exp(
        -0.5
        * simulator.config.oscillator_dephasing_rate
        * duration
        * (indices[:, None] - indices[None, :]) ** 2
    )
    assert np.array_equal(
        simulator._oscillator_dephasing_factor(duration),
        expected_oscillator,
    )


def test_structured_kraus_paths_match_dense_reference_to_roundoff():
    simulator = Phase9BackendBSimulator(BackendBConfig(cutoff=8, iq_samples=1))
    rng = np.random.default_rng(20260830)
    ket = rng.normal(size=simulator.dimension) + 1.0j * rng.normal(
        size=simulator.dimension
    )
    ket /= np.linalg.norm(ket)
    matrix = np.outer(ket, ket.conj())
    duration = 0.01

    dense_loss = simulator._apply_kraus(
        matrix,
        simulator._pure_loss_operators(duration),
    )
    structured_loss = simulator._apply_pure_loss(matrix, duration)
    assert np.max(np.abs(structured_loss - dense_loss)) <= 2.0e-16

    probability = 1.0 - np.exp(
        -simulator.config.ancilla_ge_relax_rate * duration
    )
    dense_local = simulator._apply_kraus(
        matrix,
        simulator._local_amplitude_operators(1, 0, probability),
    )
    structured_local = simulator._apply_local_amplitude(
        matrix,
        1,
        0,
        probability,
    )
    assert np.max(np.abs(structured_local - dense_local)) <= 2.0e-16


def test_density_hot_validation_matches_previous_result_and_errors():
    vector = np.array([1.0, 1.0j, -0.5], dtype=np.complex128)
    vector /= np.linalg.norm(vector)
    matrix = np.outer(vector, vector.conj())
    old_hermitian = 0.5 * (matrix + matrix.conj().T)
    assert np.array_equal(backend_b._density(matrix, 3, "state"), old_hermitian)
    with pytest.raises(ValueError, match="Hermitian"):
        backend_b._density(matrix + np.triu(np.ones((3, 3))) * 1.0e-3, 3, "bad")
    with pytest.raises(ValueError, match="unit trace"):
        backend_b._density(2.0 * matrix, 3, "bad")
    invalid = np.diag(np.array([1.1, -0.1, 0.0], dtype=np.complex128))
    with pytest.raises(ValueError, match="positive semidefinite"):
        backend_b._density(invalid, 3, "bad")


def test_precomputed_hamiltonian_and_reset_terms_are_exact_and_cached():
    simulator = Phase9BackendBSimulator(BackendBConfig(cutoff=6, iq_samples=1))
    drift = BackendBDrift(
        drive_q=0.02,
        drive_p=-0.03,
        leakage_detuning=0.01,
    )
    dispersion = simulator.level_projectors[1] + 2.0 * simulator.level_projectors[2]
    kerr = simulator.number @ (simulator.number - simulator.i_o)
    expected = (
        simulator.config.self_kerr * np.kron(kerr, simulator.i_a)
        + simulator.config.dispersive_chi
        * np.kron(simulator.number, dispersion)
        + drift.drive_q * simulator.joint_q
        + drift.drive_p * simulator.joint_p
        + drift.leakage_detuning * simulator.joint_projectors[2]
    )
    assert np.array_equal(simulator._base_hamiltonian(drift), expected)
    assert simulator._reset_operators() is simulator._reset_operators()
    assert not simulator._reset_operators()["success"][0].flags.writeable
    assert simulator.reset_completeness_error() <= 1.0e-12
    assert np.array_equal(simulator.joint_adag, np.kron(simulator.adag, simulator.i_a))
    assert simulator._reset_operators()["success"][1][0, 1] == pytest.approx(
        sqrt(simulator.config.reset_success_e)
    )
