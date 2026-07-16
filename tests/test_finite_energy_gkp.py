from __future__ import annotations

import math
import unittest

import numpy as np

from physics.constants import LATTICE_CONST
from physics.finite_energy_gkp import (
    FiniteEnergyGKPState,
    WignerLikeGrid,
    damped_projector_state,
    gaussian_envelope_state,
)


class FiniteEnergyStateFamilyTest(unittest.TestCase):
    def test_gaussian_envelope_components_follow_logical_parity(self) -> None:
        sigma = 0.3
        zero = gaussian_envelope_state("0", sigma, envelope_kappa=0.28)
        one = gaussian_envelope_state("1", sigma, envelope_kappa=0.28)

        self.assertEqual(zero.model, "gaussian_envelope")
        self.assertAlmostEqual(zero.amplitude_variance, 2.0 * sigma**2, places=15)
        np.testing.assert_array_equal(np.mod(zero.peak_table.ideal_lattice_indices, 2), 0)
        np.testing.assert_array_equal(np.mod(one.peak_table.ideal_lattice_indices, 2), 1)
        np.testing.assert_allclose(
            zero.peak_table.centers,
            zero.peak_table.ideal_lattice_indices * LATTICE_CONST,
            atol=0.0,
        )
        np.testing.assert_allclose(
            one.peak_table.centers,
            one.peak_table.ideal_lattice_indices * LATTICE_CONST,
            atol=0.0,
        )

    def test_gaussian_default_envelope_is_symmetric_parameterization(self) -> None:
        state = gaussian_envelope_state("0", 0.27)
        self.assertEqual(state.peak_sigma, 0.27)
        self.assertEqual(state.envelope_kappa, 0.27)

    def test_damped_projector_matches_mehler_kernel_parameters(self) -> None:
        delta = 0.34
        state = damped_projector_state("1", delta)
        epsilon = delta**2
        expected_variance = state.coordinate_scale**2 * math.tanh(epsilon)
        expected_contraction = 1.0 / math.cosh(epsilon)
        table = state.peak_table

        self.assertEqual(state.model, "damped_projector")
        self.assertAlmostEqual(state.amplitude_variance, expected_variance, places=15)
        np.testing.assert_allclose(
            table.centers,
            table.ideal_lattice_indices * LATTICE_CONST * expected_contraction,
            rtol=2.0e-15,
            atol=0.0,
        )

    def test_damped_wavefunction_matches_independent_mehler_kernel_sum(self) -> None:
        delta = 0.34
        state = damped_projector_state("1", delta)
        epsilon = delta**2
        q = np.linspace(-6.0, 6.0, 1201)
        scale = state.coordinate_scale
        canonical_q = q / scale
        ideal_centers = (
            state.peak_table.ideal_lattice_indices
            * LATTICE_CONST
            / scale
        )
        direct = np.zeros_like(q)
        for center in ideal_centers:
            direct += np.exp(
                -(
                    (canonical_q * canonical_q + center * center)
                    * math.cosh(epsilon)
                    - 2.0 * canonical_q * center
                )
                / (2.0 * math.sinh(epsilon))
            )
        direct /= np.linalg.norm(direct)
        implemented = np.asarray(state.wavefunction(q))
        implemented /= np.linalg.norm(implemented)

        np.testing.assert_allclose(implemented, direct, rtol=7.0e-14, atol=4.0e-15)

    def test_all_four_logical_states_are_analytically_normalized(self) -> None:
        factories = [
            lambda label: gaussian_envelope_state(label, 0.3, envelope_kappa=0.28),
            lambda label: damped_projector_state(label, 0.34),
        ]
        for factory in factories:
            states = {label: factory(label) for label in ("0", "1", "+", "-")}
            for label, state in states.items():
                with self.subTest(model=state.model, label=label):
                    self.assertAlmostEqual(state.inner_product(state), 1.0, delta=5.0e-15)
            self.assertAlmostEqual(states["+"].inner_product(states["-"]), 0.0, delta=5.0e-15)
            self.assertGreater(states["0"].inner_product(states["1"]), 0.0)
            upper = 3.0e-3 if states["0"].model == "damped_projector" else 1.0e-3
            self.assertLess(states["0"].inner_product(states["1"]), upper)

    def test_numerical_wavefunction_integral_confirms_analytic_normalization(self) -> None:
        states = [
            gaussian_envelope_state("-", 0.3, envelope_kappa=0.28),
            damped_projector_state("-", 0.34),
        ]
        for state in states:
            q = np.linspace(-state.support_radius, state.support_radius, 40_001)
            step = float(q[1] - q[0])
            mass = float(np.sum(state.probability_density(q)) * step)
            with self.subTest(model=state.model):
                self.assertAlmostEqual(mass, 1.0, delta=3.0e-11)

    def test_plus_minus_are_coherent_superpositions_not_probability_mixtures(self) -> None:
        zero = gaussian_envelope_state("0", 0.31, envelope_kappa=0.29)
        one = gaussian_envelope_state("1", 0.31, envelope_kappa=0.29)
        plus = gaussian_envelope_state("+", 0.31, envelope_kappa=0.29)
        minus = gaussian_envelope_state("-", 0.31, envelope_kappa=0.29)
        q = np.linspace(-8.0, 8.0, 2001)
        sum_wave = np.asarray(zero.wavefunction(q)) + np.asarray(one.wavefunction(q))
        difference_wave = np.asarray(zero.wavefunction(q)) - np.asarray(one.wavefunction(q))
        sum_wave /= np.linalg.norm(sum_wave)
        difference_wave /= np.linalg.norm(difference_wave)
        plus_wave = np.asarray(plus.wavefunction(q))
        minus_wave = np.asarray(minus.wavefunction(q))
        plus_wave /= np.linalg.norm(plus_wave)
        minus_wave /= np.linalg.norm(minus_wave)

        np.testing.assert_allclose(plus_wave, sum_wave, rtol=3.0e-14, atol=3.0e-15)
        np.testing.assert_allclose(minus_wave, difference_wave, rtol=3.0e-14, atol=3.0e-15)

    def test_wavefunction_scalar_and_vector_semantics(self) -> None:
        state = damped_projector_state("0", 0.34)
        scalar = state.wavefunction(0.0)
        vector = state.wavefunction([0.0, 0.1])
        scalar_density = state.probability_density(0.0)

        self.assertIsInstance(scalar, float)
        self.assertIsInstance(scalar_density, float)
        self.assertEqual(np.asarray(vector).shape, (2,))
        self.assertAlmostEqual(scalar_density, scalar * scalar, places=15)

    def test_syndrome_distribution_is_nonnegative_and_normalized(self) -> None:
        states = [
            gaussian_envelope_state(label, 0.3, envelope_kappa=0.28)
            for label in ("0", "1", "+", "-")
        ] + [damped_projector_state(label, 0.34) for label in ("0", "1", "+", "-")]
        for state in states:
            distribution = state.syndrome_distribution(2048)
            step = float(distribution.syndrome[1] - distribution.syndrome[0])
            with self.subTest(model=state.model, label=state.logical_state):
                self.assertTrue(np.all(distribution.density >= 0.0))
                self.assertAlmostEqual(
                    float(np.sum(distribution.density) * step),
                    1.0,
                    delta=8.0e-14,
                )
                self.assertAlmostEqual(distribution.captured_mass, 1.0, delta=2.0e-11)
                self.assertLess(distribution.syndrome[-1], LATTICE_CONST / 2.0)
                self.assertGreaterEqual(distribution.syndrome[0], -LATTICE_CONST / 2.0)

    def test_narrower_gaussian_peaks_narrow_syndrome_distribution(self) -> None:
        narrow = gaussian_envelope_state("0", 0.18, envelope_kappa=0.3)
        wide = gaussian_envelope_state("0", 0.36, envelope_kappa=0.3)
        narrow_distribution = narrow.syndrome_distribution(4096)
        wide_distribution = wide.syndrome_distribution(4096)
        step = float(narrow_distribution.syndrome[1] - narrow_distribution.syndrome[0])
        narrow_variance = float(
            np.sum(narrow_distribution.density * narrow_distribution.syndrome**2) * step
        )
        wide_variance = float(
            np.sum(wide_distribution.density * wide_distribution.syndrome**2) * step
        )

        self.assertAlmostEqual(narrow_variance, 0.18**2, delta=2.0e-12)
        self.assertGreater(wide_variance, 3.5 * narrow_variance)

    def test_weaker_envelope_and_smaller_projector_delta_expand_toward_ideal_comb(self) -> None:
        weak_envelope = gaussian_envelope_state("0", 0.28, envelope_kappa=0.18)
        strong_envelope = gaussian_envelope_state("0", 0.28, envelope_kappa=0.5)
        small_delta = damped_projector_state("0", 0.25)
        large_delta = damped_projector_state("0", 0.5)

        self.assertGreater(weak_envelope.component_count, strong_envelope.component_count)
        self.assertGreater(weak_envelope.support_radius, strong_envelope.support_radius)
        self.assertLess(small_delta.amplitude_variance, large_delta.amplitude_variance)
        self.assertGreater(small_delta.component_count, large_delta.component_count)
        small_table = small_delta.peak_table
        large_table = large_delta.peak_table
        small_nonzero = small_table.ideal_lattice_indices != 0
        large_nonzero = large_table.ideal_lattice_indices != 0
        small_contraction = abs(
            small_table.centers[small_nonzero][0]
            / (small_table.ideal_lattice_indices[small_nonzero][0] * LATTICE_CONST)
        )
        large_contraction = abs(
            large_table.centers[large_nonzero][0]
            / (large_table.ideal_lattice_indices[large_nonzero][0] * LATTICE_CONST)
        )
        self.assertGreater(small_contraction, large_contraction)

    def test_peak_truncation_converges_under_tighter_tail_tolerance(self) -> None:
        pairs = [
            (
                gaussian_envelope_state("+", 0.3, envelope_kappa=0.28, tail_tolerance=1e-8),
                gaussian_envelope_state("+", 0.3, envelope_kappa=0.28, tail_tolerance=1e-14),
            ),
            (
                damped_projector_state("+", 0.34, tail_tolerance=1e-8),
                damped_projector_state("+", 0.34, tail_tolerance=1e-14),
            ),
        ]
        for coarse, fine in pairs:
            with self.subTest(model=coarse.model):
                self.assertGreater(fine.component_count, coarse.component_count)
                self.assertGreater(coarse.inner_product(fine), 1.0 - 1.0e-13)

    def test_wigner_like_grid_preserves_q_marginal_and_signed_interference(self) -> None:
        states = [
            gaussian_envelope_state("+", 0.3, envelope_kappa=0.28),
            damped_projector_state("+", 0.34),
        ]
        for state in states:
            grid = state.wigner_like_grid(257)
            self.assertIsInstance(grid, WignerLikeGrid)
            step_q = float(grid.q[1] - grid.q[0])
            step_p = float(grid.p[1] - grid.p[0])
            with self.subTest(model=state.model):
                self.assertEqual(grid.values.shape, (257, 257))
                np.testing.assert_allclose(
                    grid.q_marginal,
                    grid.q_probability,
                    rtol=0.0,
                    atol=2.0e-14,
                )
                self.assertAlmostEqual(
                    float(np.sum(grid.values) * step_q * step_p),
                    grid.captured_probability,
                    delta=3.0e-12,
                )
                self.assertAlmostEqual(grid.total_mass, 1.0, delta=2.0e-8)
                self.assertGreater(grid.negative_volume, 0.1)
                self.assertLess(float(np.min(grid.values)), -0.05)

    def test_invalid_parameters_and_requests_fail_closed(self) -> None:
        invalid_calls = [
            lambda: FiniteEnergyGKPState("bad", model="gaussian_envelope", peak_sigma=0.3),  # type: ignore[arg-type]
            lambda: FiniteEnergyGKPState("0", model="bad", peak_sigma=0.3),  # type: ignore[arg-type]
            lambda: gaussian_envelope_state("0", 0.0),
            lambda: gaussian_envelope_state("0", 0.3, envelope_kappa=-0.1),
            lambda: gaussian_envelope_state("0", 1.0e-300),
            lambda: gaussian_envelope_state("0", 0.3, envelope_kappa=1.0e-300),
            lambda: damped_projector_state("0", 0.0),
            lambda: damped_projector_state("0", 1.0e-300),
            lambda: damped_projector_state("0", 1.0e308),
            lambda: FiniteEnergyGKPState(
                "0", model="gaussian_envelope", peak_sigma=0.3, projector_delta=0.3
            ),
        ]
        for call in invalid_calls:
            with self.subTest(call=call):
                with self.assertRaises(ValueError):
                    call()

        state = gaussian_envelope_state("0", 0.3)
        with self.assertRaises(ValueError):
            state.wavefunction(np.nan)
        with self.assertRaises(ValueError):
            state.syndrome_distribution(32)
        with self.assertRaises(ValueError):
            state.wigner_like_grid(128)
        with self.assertRaises(ValueError):
            state.wigner_like_grid(257, q_range=(1.0, -1.0))
        with self.assertRaises(ValueError):
            state.wigner_like_grid(257, q_range=1.0)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
