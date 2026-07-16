from __future__ import annotations

import math
import unittest

import numpy as np

from physics.constants import LATTICE_CONST
from physics.ideal_gkp_decoder import (
    MAPDecodeResult,
    MAPDecode2DResult,
    StandardBinningResult,
    coset_likelihood_1d,
    coset_likelihood_2d,
    covariance_from_sigmas,
    decode_1d,
    gaussian_logical_flip_probability,
    independent_map_decode_2d,
    llr_1d,
    map_decode_1d,
    map_decode_2d,
    standard_binning_1d,
)
from physics.syndrome_measurement import SyndromeMeasurement


class StandardBinning1DTest(unittest.TestCase):
    def test_scalar_cells_and_logical_parity(self) -> None:
        lam = LATTICE_CONST
        cases = [
            (0.49 * lam, 0, False),
            (0.51 * lam, 1, True),
            (1.49 * lam, 1, True),
            (1.51 * lam, 2, False),
            (-0.49 * lam, 0, False),
            (-0.51 * lam, -1, True),
            (-1.51 * lam, -2, False),
        ]
        for displacement, expected_index, expected_flip in cases:
            with self.subTest(displacement=displacement):
                result = standard_binning_1d(displacement)
                self.assertEqual(result.lattice_index, expected_index)
                self.assertEqual(result.logical_flip, expected_flip)
                self.assertAlmostEqual(
                    displacement,
                    result.correction + expected_index * lam,
                    places=13,
                )
                self.assertGreaterEqual(result.syndrome, -lam / 2.0)
                self.assertLess(result.syndrome, lam / 2.0)

    def test_half_open_boundary_convention_is_deterministic(self) -> None:
        lam = LATTICE_CONST
        positive = standard_binning_1d(lam / 2.0)
        negative = standard_binning_1d(-lam / 2.0)

        self.assertEqual(positive.lattice_index, 1)
        self.assertAlmostEqual(positive.syndrome, -lam / 2.0)
        self.assertEqual(negative.lattice_index, 0)
        self.assertAlmostEqual(negative.syndrome, -lam / 2.0)

    def test_scalar_result_preserves_semantic_types(self) -> None:
        result = standard_binning_1d(0.6 * LATTICE_CONST)
        self.assertIsInstance(result.syndrome, float)
        self.assertIsInstance(result.correction, float)
        self.assertIsInstance(result.lattice_index, int)
        self.assertIsInstance(result.logical_parity, int)
        self.assertIsInstance(result.logical_flip, bool)

    def test_vectorized_result_reconstructs_input(self) -> None:
        lam = LATTICE_CONST
        values = lam * np.array([-4.2, -2.51, -0.5, 0.0, 0.5, 2.49, 5.2])
        result = standard_binning_1d(values)

        np.testing.assert_allclose(
            values,
            result.correction + result.lattice_index * lam,
            rtol=0.0,
            atol=2.0e-14,
        )
        np.testing.assert_array_equal(
            result.logical_parity,
            np.mod(result.lattice_index, 2),
        )
        self.assertTrue(np.all(result.syndrome >= -lam / 2.0))
        self.assertTrue(np.all(result.syndrome < lam / 2.0))

    def test_syndrome_matches_existing_measurement_convention(self) -> None:
        lam = LATTICE_CONST
        values = lam * np.array([-2.7, -0.5, -0.1, 0.5, 1.8])
        expected = SyndromeMeasurement(lattice=lam).measure(values)
        actual = standard_binning_1d(values).syndrome
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2.0e-14)

    def test_invalid_inputs_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            standard_binning_1d(np.nan)
        with self.assertRaises(ValueError):
            standard_binning_1d([0.0, np.inf])
        with self.assertRaises(ValueError):
            standard_binning_1d(0.0, lattice=0.0)
        with self.assertRaises(ValueError):
            standard_binning_1d(1.0e308, lattice=1.0e-308)


class GaussianLogicalFlipProbabilityTest(unittest.TestCase):
    def test_zero_noise_and_probability_range(self) -> None:
        self.assertEqual(gaussian_logical_flip_probability(0.0), 0.0)
        values = [gaussian_logical_flip_probability(s) for s in (0.1, 0.4, 0.8, 2.0, 10.0)]
        self.assertTrue(all(0.0 <= value <= 0.5 for value in values))
        self.assertTrue(all(a <= b for a, b in zip(values, values[1:])))

    def test_rare_event_matches_first_odd_cell_tail(self) -> None:
        sigma = 0.18 * LATTICE_CONST
        probability = gaussian_logical_flip_probability(sigma, method="interval")
        first_odd_cells = (
            math.erfc((0.5 * LATTICE_CONST) / (math.sqrt(2.0) * sigma))
            - math.erfc((1.5 * LATTICE_CONST) / (math.sqrt(2.0) * sigma))
        )
        # 其余 odd cells 位于 2.5 lambda 之外，在该 sigma 下应可忽略。
        self.assertAlmostEqual(probability, first_odd_cells, places=14)

    def test_interval_and_fourier_forms_agree(self) -> None:
        for ratio in (0.3, 0.5, 1.0, 2.0):
            sigma = ratio * LATTICE_CONST
            with self.subTest(ratio=ratio):
                interval = gaussian_logical_flip_probability(sigma, method="interval")
                fourier = gaussian_logical_flip_probability(sigma, method="fourier")
                self.assertAlmostEqual(interval, fourier, delta=2.0e-13)

    def test_analytic_probability_matches_independent_monte_carlo(self) -> None:
        sigma = 0.32 * LATTICE_CONST
        analytic = gaussian_logical_flip_probability(sigma)

        rng = np.random.default_rng(20260714)
        samples = rng.normal(0.0, sigma, size=400_000)
        # Monte Carlo 标签由既有 modular measurement 独立构造，不调用待验函数。
        wrapped = SyndromeMeasurement(lattice=LATTICE_CONST).measure(samples)
        aliases = np.rint((samples - wrapped) / LATTICE_CONST).astype(np.int64)
        flips = np.mod(aliases, 2).astype(float)
        estimate = float(np.mean(flips))
        standard_error = math.sqrt(analytic * (1.0 - analytic) / samples.size)

        self.assertLessEqual(abs(estimate - analytic), 5.0 * standard_error + 2.0e-4)

    def test_scale_invariance(self) -> None:
        ratio = 0.41
        reference = gaussian_logical_flip_probability(ratio * LATTICE_CONST)
        scaled = gaussian_logical_flip_probability(ratio * 7.3, lattice=7.3)
        self.assertAlmostEqual(reference, scaled, places=14)

    def test_invalid_probability_inputs_are_rejected(self) -> None:
        for sigma in (-1.0, np.nan, np.inf):
            with self.subTest(sigma=sigma):
                with self.assertRaises(ValueError):
                    gaussian_logical_flip_probability(sigma)
        with self.assertRaises(ValueError):
            gaussian_logical_flip_probability(0.2, method="bad")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            gaussian_logical_flip_probability(0.2, max_terms=0)


class PeriodicGaussianMAPTest(unittest.TestCase):
    @staticmethod
    def _brute_force_likelihood(
        syndrome: float,
        sigma: float,
        parity: int,
        *,
        mean: float = 0.0,
        lattice: float = LATTICE_CONST,
        radius: int = 80,
    ) -> float:
        normalizer = sigma * math.sqrt(2.0 * math.pi)
        total = 0.0
        for alias in range(-radius, radius + 1):
            if alias % 2 != parity:
                continue
            residual = syndrome + alias * lattice - mean
            total += math.exp(-0.5 * (residual / sigma) ** 2) / normalizer
        return total

    def test_coset_likelihood_matches_independent_alias_sum(self) -> None:
        lam = LATTICE_CONST
        sigma = 0.73 * lam
        syndrome = 0.21 * lam
        mean = 2.38 * lam

        for parity in (0, 1):
            with self.subTest(parity=parity):
                expected = self._brute_force_likelihood(
                    syndrome,
                    sigma,
                    parity,
                    mean=mean,
                )
                actual = coset_likelihood_1d(
                    syndrome,
                    sigma,
                    parity,
                    mean=mean,
                )
                self.assertAlmostEqual(actual, expected, delta=2.0e-15)

    def test_log_likelihood_is_finite_when_raw_likelihood_underflows(self) -> None:
        sigma = 0.01 * LATTICE_CONST
        raw_odd = coset_likelihood_1d(0.0, sigma, 1)
        log_odd = coset_likelihood_1d(0.0, sigma, 1, log_output=True)
        result = map_decode_1d(0.0, sigma)

        self.assertEqual(raw_odd, 0.0)
        self.assertTrue(math.isfinite(log_odd))
        self.assertTrue(math.isfinite(result.llr))
        self.assertGreater(result.llr, 1000.0)
        self.assertEqual(result.posterior_even, 1.0)
        self.assertEqual(result.parity, 0)

    def test_zero_mean_likelihood_has_reflection_symmetry(self) -> None:
        lam = LATTICE_CONST
        points = lam * np.array([0.0, 0.07, 0.24, 0.49])
        for parity in (0, 1):
            positive = coset_likelihood_1d(points, 0.42 * lam, parity)
            negative = coset_likelihood_1d(-points, 0.42 * lam, parity)
            np.testing.assert_allclose(positive, negative, rtol=2.0e-14, atol=1.0e-15)

    def test_coset_density_normalizes_over_one_syndrome_period(self) -> None:
        lam = LATTICE_CONST
        count = 20_000
        grid = (-0.5 + (np.arange(count) + 0.5) / count) * lam
        for ratio, mean_cells in ((0.15, 0.0), (0.53, 1.7), (2.0, -2.4)):
            with self.subTest(ratio=ratio, mean_cells=mean_cells):
                even = coset_likelihood_1d(grid, ratio * lam, 0, mean=mean_cells * lam)
                odd = coset_likelihood_1d(grid, ratio * lam, 1, mean=mean_cells * lam)
                integral = float(np.sum(even + odd) * lam / count)
                self.assertAlmostEqual(integral, 1.0, delta=3.0e-13)

    def test_one_cell_mean_shift_swaps_logical_cosets(self) -> None:
        lam = LATTICE_CONST
        syndrome = lam * np.array([-0.41, -0.13, 0.2, 0.46])
        sigma = 0.47 * lam
        mean = -0.37 * lam
        base_even = coset_likelihood_1d(syndrome, sigma, 0, mean=mean)
        base_odd = coset_likelihood_1d(syndrome, sigma, 1, mean=mean)
        shifted_even = coset_likelihood_1d(syndrome, sigma, 0, mean=mean + lam)
        shifted_odd = coset_likelihood_1d(syndrome, sigma, 1, mean=mean + lam)

        np.testing.assert_allclose(shifted_even, base_odd, rtol=2.0e-14, atol=1.0e-15)
        np.testing.assert_allclose(shifted_odd, base_even, rtol=2.0e-14, atol=1.0e-15)

    def test_llr_matches_likelihood_ratio_and_prior_odds(self) -> None:
        lam = LATTICE_CONST
        syndrome = lam * np.array([-0.42, -0.1, 0.0, 0.31])
        sigma = 0.58 * lam
        prior_even = 0.37
        even = coset_likelihood_1d(syndrome, sigma, 0)
        odd = coset_likelihood_1d(syndrome, sigma, 1)
        expected = np.log(even) - np.log(odd) + math.log(prior_even / (1.0 - prior_even))

        actual = llr_1d(syndrome, sigma, prior_even=prior_even)
        np.testing.assert_allclose(actual, expected, rtol=2.0e-14, atol=2.0e-14)

    def test_map_result_contains_consistent_hard_and_soft_outputs(self) -> None:
        lam = LATTICE_CONST
        syndrome = lam * np.array([-0.45, -0.15, 0.0, 0.28, 0.47])
        mean = lam * np.array([0.9, -1.2, 0.0, 1.1, -0.8])
        result = map_decode_1d(syndrome, 0.39 * lam, mean=mean, prior_even=0.43)

        expected_posterior_even = 1.0 / (1.0 + np.exp(-result.llr))
        np.testing.assert_allclose(result.posterior_even, expected_posterior_even, atol=2.0e-15)
        np.testing.assert_allclose(result.posterior_even + result.posterior_odd, 1.0, atol=0.0)
        np.testing.assert_array_equal(result.parity, np.asarray(result.llr) < 0.0)
        np.testing.assert_array_equal(result.logical_flip, np.asarray(result.parity, dtype=bool))
        np.testing.assert_allclose(
            result.confidence,
            np.abs(result.posterior_even - result.posterior_odd),
            atol=0.0,
        )

    def test_nonzero_mean_can_change_map_logical_coset(self) -> None:
        lam = LATTICE_CONST
        even_centered = map_decode_1d(0.0, 0.2 * lam, mean=0.0)
        odd_centered = map_decode_1d(0.0, 0.2 * lam, mean=lam)

        self.assertEqual(even_centered.parity, 0)
        self.assertGreater(even_centered.llr, 0.0)
        self.assertEqual(odd_centered.parity, 1)
        self.assertLess(odd_centered.llr, 0.0)

    def test_prior_resolves_equal_likelihood_boundary(self) -> None:
        lam = LATTICE_CONST
        boundary = np.nextafter(lam / 2.0, -np.inf)
        even_prior = map_decode_1d(boundary, 0.31 * lam, prior_even=0.8)
        odd_prior = map_decode_1d(boundary, 0.31 * lam, prior_even=0.2)

        self.assertEqual(even_prior.parity, 0)
        self.assertEqual(odd_prior.parity, 1)
        self.assertGreater(even_prior.llr, 0.0)
        self.assertLess(odd_prior.llr, 0.0)

    def test_scalar_map_result_preserves_semantic_types(self) -> None:
        result = map_decode_1d(0.1 * LATTICE_CONST, 0.4 * LATTICE_CONST)
        self.assertIsInstance(result.syndrome, float)
        self.assertIsInstance(result.parity, int)
        self.assertIsInstance(result.logical_flip, bool)
        self.assertIsInstance(result.llr, float)
        self.assertIsInstance(result.posterior_even, float)
        self.assertIsInstance(result.confidence, float)

    def test_centered_syndrome_and_model_validation(self) -> None:
        lam = LATTICE_CONST
        invalid_calls = [
            lambda: coset_likelihood_1d(lam / 2.0, 0.2, 0),
            lambda: coset_likelihood_1d(-lam / 2.0 - 1.0e-12, 0.2, 0),
            lambda: coset_likelihood_1d(np.nan, 0.2, 0),
            lambda: coset_likelihood_1d(0.0, 0.0, 0),
            lambda: coset_likelihood_1d(0.0, 0.2, 2),  # type: ignore[arg-type]
            lambda: llr_1d(0.0, 0.2, prior_even=1.0),
            lambda: llr_1d([0.0, 0.1], 0.2, mean=[0.0, 0.1, 0.2]),
            lambda: llr_1d(0.0, 0.2, mean=1.0e300, lattice=1.0e-300),
            lambda: llr_1d(0.0, 1.0e20),
        ]
        for call in invalid_calls:
            with self.subTest(call=call):
                with self.assertRaises(ValueError):
                    call()

    def test_three_mode_entrypoint_contract(self) -> None:
        lam = LATTICE_CONST
        standard = decode_1d(0.6 * lam, mode="standard")
        mapped = decode_1d(0.1 * lam, mode="map", sigma=0.3 * lam)
        soft = decode_1d(0.1 * lam, mode="soft", sigma=0.3 * lam)

        self.assertIsInstance(standard, StandardBinningResult)
        self.assertIsInstance(mapped, MAPDecodeResult)
        self.assertAlmostEqual(soft, mapped.llr, places=14)
        with self.assertRaises(ValueError):
            decode_1d(0.0, mode="map")
        with self.assertRaises(ValueError):
            decode_1d(0.0, mode="standard", sigma=0.2)
        with self.assertRaises(ValueError):
            decode_1d(0.0, mode="unknown")  # type: ignore[arg-type]


class CorrelatedGaussianMAP2DTest(unittest.TestCase):
    @staticmethod
    def _brute_force_likelihood(
        syndrome: np.ndarray,
        covariance: np.ndarray,
        parity: tuple[int, int],
        *,
        mean: np.ndarray,
        radius: int = 30,
    ) -> float:
        inverse = np.linalg.inv(covariance)
        normalizer = 2.0 * math.pi * math.sqrt(float(np.linalg.det(covariance)))
        total = 0.0
        for alias_q in range(-radius, radius + 1):
            if alias_q % 2 != parity[0]:
                continue
            for alias_p in range(-radius, radius + 1):
                if alias_p % 2 != parity[1]:
                    continue
                residual = syndrome + LATTICE_CONST * np.array([alias_q, alias_p]) - mean
                exponent = -0.5 * float(residual @ inverse @ residual)
                total += math.exp(exponent) / normalizer
        return total

    def test_covariance_builder_matches_requested_marginals_and_rho(self) -> None:
        sigma_q = 0.31 * LATTICE_CONST
        sigma_p = 0.57 * LATTICE_CONST
        rho = -0.73
        covariance = covariance_from_sigmas(sigma_q, sigma_p, rho)

        self.assertAlmostEqual(math.sqrt(covariance[0, 0]), sigma_q, places=14)
        self.assertAlmostEqual(math.sqrt(covariance[1, 1]), sigma_p, places=14)
        actual_rho = covariance[0, 1] / math.sqrt(covariance[0, 0] * covariance[1, 1])
        self.assertAlmostEqual(actual_rho, rho, places=14)
        self.assertTrue(np.all(np.linalg.eigvalsh(covariance) > 0.0))

    def test_correlated_likelihood_matches_independent_double_alias_sum(self) -> None:
        lam = LATTICE_CONST
        covariance = covariance_from_sigmas(0.51 * lam, 0.36 * lam, 0.72)
        syndrome = lam * np.array([0.23, -0.31])
        mean = lam * np.array([1.37, -0.76])

        for parity in ((0, 0), (0, 1), (1, 0), (1, 1)):
            with self.subTest(parity=parity):
                expected = self._brute_force_likelihood(
                    syndrome,
                    covariance,
                    parity,
                    mean=mean,
                )
                actual = coset_likelihood_2d(
                    syndrome,
                    covariance,
                    parity,
                    mean=mean,
                )
                self.assertAlmostEqual(actual, expected, delta=3.0e-15)

    def test_rho_zero_likelihood_factorizes_into_two_1d_cosets(self) -> None:
        lam = LATTICE_CONST
        sigma_q = 0.34 * lam
        sigma_p = 0.61 * lam
        covariance = covariance_from_sigmas(sigma_q, sigma_p, 0.0)
        syndrome = lam * np.array(
            [[-0.41, 0.17], [-0.08, -0.32], [0.22, 0.0], [0.47, 0.39]]
        )
        mean = lam * np.array([1.2, -0.7])

        for parity_q in (0, 1):
            for parity_p in (0, 1):
                joint = coset_likelihood_2d(
                    syndrome,
                    covariance,
                    (parity_q, parity_p),
                    mean=mean,
                )
                product = coset_likelihood_1d(
                    syndrome[:, 0], sigma_q, parity_q, mean=mean[0]
                ) * coset_likelihood_1d(
                    syndrome[:, 1], sigma_p, parity_p, mean=mean[1]
                )
                np.testing.assert_allclose(joint, product, rtol=3.0e-14, atol=2.0e-16)

    def test_rho_zero_joint_map_equals_independent_axis_map(self) -> None:
        lam = LATTICE_CONST
        covariance = covariance_from_sigmas(0.39 * lam, 0.48 * lam, 0.0)
        syndrome = lam * np.array(
            [[-0.44, -0.13], [-0.2, 0.46], [0.0, 0.0], [0.37, -0.29]]
        )
        mean = lam * np.array([0.6, -1.1])
        joint = map_decode_2d(syndrome, covariance, mean=mean)
        independent = independent_map_decode_2d(syndrome, covariance, mean=mean)

        np.testing.assert_allclose(joint.log_likelihoods, independent.log_likelihoods, atol=3.0e-14)
        np.testing.assert_allclose(joint.posterior, independent.posterior, atol=3.0e-15)
        np.testing.assert_array_equal(joint.parity, independent.parity)
        np.testing.assert_array_equal(joint.logical_class, independent.logical_class)

    def test_joint_result_has_consistent_four_class_hard_and_soft_outputs(self) -> None:
        lam = LATTICE_CONST
        covariance = covariance_from_sigmas(0.43 * lam, 0.37 * lam, -0.81)
        syndrome = lam * np.array(
            [[-0.42, 0.38], [-0.1, -0.23], [0.11, 0.27], [0.45, -0.4]]
        )
        prior = np.array([[0.13, 0.27], [0.41, 0.19]])
        result = map_decode_2d(syndrome, covariance, prior=prior)

        self.assertIsInstance(result, MAPDecode2DResult)
        self.assertEqual(result.method, "joint")
        np.testing.assert_allclose(np.sum(result.posterior, axis=(-2, -1)), 1.0, atol=2.0e-15)
        expected_class = np.argmax(result.posterior.reshape((-1, 4)), axis=-1)
        np.testing.assert_array_equal(result.logical_class, expected_class)
        np.testing.assert_array_equal(result.parity[:, 0], expected_class // 2)
        np.testing.assert_array_equal(result.parity[:, 1], expected_class % 2)
        np.testing.assert_array_equal(result.logical_flips, result.parity.astype(bool))
        sorted_posterior = np.sort(result.posterior.reshape((-1, 4)), axis=-1)
        np.testing.assert_allclose(
            result.confidence,
            sorted_posterior[:, -1] - sorted_posterior[:, -2],
            atol=0.0,
        )

    def test_one_cell_q_mean_shift_swaps_q_cosets_only(self) -> None:
        lam = LATTICE_CONST
        covariance = covariance_from_sigmas(0.44 * lam, 0.52 * lam, 0.67)
        syndrome = lam * np.array([[-0.31, 0.22], [0.07, -0.43], [0.41, 0.19]])
        mean = lam * np.array([0.26, -0.38])
        for parity_q in (0, 1):
            for parity_p in (0, 1):
                shifted = coset_likelihood_2d(
                    syndrome,
                    covariance,
                    (parity_q, parity_p),
                    mean=mean + np.array([lam, 0.0]),
                )
                swapped = coset_likelihood_2d(
                    syndrome,
                    covariance,
                    (1 - parity_q, parity_p),
                    mean=mean,
                )
                np.testing.assert_allclose(shifted, swapped, rtol=4.0e-14, atol=2.0e-16)

    def test_four_coset_density_normalizes_over_syndrome_cell(self) -> None:
        lam = LATTICE_CONST
        count = 56
        axis = (-0.5 + (np.arange(count) + 0.5) / count) * lam
        q_grid, p_grid = np.meshgrid(axis, axis, indexing="ij")
        syndrome = np.stack((q_grid.ravel(), p_grid.ravel()), axis=-1)
        covariance = covariance_from_sigmas(0.38 * lam, 0.49 * lam, -0.74)
        density = np.zeros(syndrome.shape[0], dtype=np.float64)
        for parity in ((0, 0), (0, 1), (1, 0), (1, 1)):
            density += coset_likelihood_2d(syndrome, covariance, parity, mean=(0.7 * lam, -lam))
        integral = float(np.sum(density) * (lam / count) ** 2)
        self.assertAlmostEqual(integral, 1.0, delta=2.0e-10)

    def test_independent_decoder_uses_only_marginal_covariance(self) -> None:
        lam = LATTICE_CONST
        correlated = covariance_from_sigmas(0.4 * lam, 0.55 * lam, 0.88)
        diagonal = covariance_from_sigmas(0.4 * lam, 0.55 * lam, 0.0)
        syndrome = lam * np.array([[-0.3, 0.2], [0.12, -0.47], [0.43, 0.31]])
        correlated_result = independent_map_decode_2d(syndrome, correlated)
        diagonal_result = independent_map_decode_2d(syndrome, diagonal)

        np.testing.assert_allclose(correlated_result.posterior, diagonal_result.posterior, atol=0.0)
        np.testing.assert_array_equal(correlated_result.parity, diagonal_result.parity)

    def test_strong_correlation_joint_map_has_paired_monte_carlo_gain(self) -> None:
        lam = LATTICE_CONST
        covariance = covariance_from_sigmas(0.42 * lam, 0.42 * lam, 0.9)
        rng = np.random.default_rng(20260714)
        samples = rng.multivariate_normal([0.0, 0.0], covariance, size=12_000)
        q_truth = standard_binning_1d(samples[:, 0])
        p_truth = standard_binning_1d(samples[:, 1])
        truth = 2 * np.asarray(q_truth.logical_parity) + np.asarray(p_truth.logical_parity)
        syndrome = np.column_stack((q_truth.syndrome, p_truth.syndrome))

        joint_chunks = []
        independent_chunks = []
        for start in range(0, samples.shape[0], 1_500):
            chunk = syndrome[start : start + 1_500]
            joint_chunks.append(np.asarray(map_decode_2d(chunk, covariance).logical_class))
            independent_chunks.append(
                np.asarray(independent_map_decode_2d(chunk, covariance).logical_class)
            )
        joint = np.concatenate(joint_chunks)
        independent = np.concatenate(independent_chunks)
        joint_error = float(np.mean(joint != truth))
        independent_error = float(np.mean(independent != truth))
        joint_only_correct = int(np.sum((joint == truth) & (independent != truth)))
        independent_only_correct = int(np.sum((joint != truth) & (independent == truth)))
        mcnemar_z = (joint_only_correct - independent_only_correct) / math.sqrt(
            joint_only_correct + independent_only_correct
        )

        self.assertGreater(independent_error - joint_error, 0.05)
        self.assertGreater(mcnemar_z, 10.0)

    def test_invalid_2d_models_and_inputs_are_rejected(self) -> None:
        lam = LATTICE_CONST
        covariance = covariance_from_sigmas(0.4 * lam, 0.5 * lam, 0.2)
        invalid_calls = [
            lambda: covariance_from_sigmas(0.0, 1.0, 0.0),
            lambda: covariance_from_sigmas(1.0, 1.0, 1.0),
            lambda: covariance_from_sigmas(1.0, 1.0, np.nan),
            lambda: map_decode_2d([0.0], covariance),
            lambda: map_decode_2d([lam / 2.0, 0.0], covariance),
            lambda: map_decode_2d([0.0, 0.0], [[1.0, 0.2], [0.1, 1.0]]),
            lambda: map_decode_2d(
                [0.0, 0.0], [[1.0e-20, 2.0e-21], [1.0e-21, 1.0e-20]]
            ),
            lambda: map_decode_2d([0.0, 0.0], [[1.0, 2.0], [2.0, 1.0]]),
            lambda: map_decode_2d([0.0, 0.0], covariance, prior=[[1.0, 0.0], [1.0, 1.0]]),
            lambda: map_decode_2d(
                [0.0, 0.0], covariance, prior=np.full((2, 2), 1.0e308)
            ),
            lambda: coset_likelihood_2d([0.0, 0.0], covariance, (0, 2)),
            lambda: map_decode_2d([0.0, 0.0], covariance_from_sigmas(1.0e10, 1.0e10)),
            lambda: independent_map_decode_2d(
                [[0.0, 0.0], [0.1, 0.1]], covariance, mean=[0.0, 0.0, 0.0]
            ),
        ]
        for call in invalid_calls:
            with self.subTest(call=call):
                with self.assertRaises(ValueError):
                    call()


if __name__ == "__main__":
    unittest.main()
