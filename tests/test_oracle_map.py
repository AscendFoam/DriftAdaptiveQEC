from __future__ import annotations

import math
import unittest

import numpy as np

from physics.constants import LATTICE_CONST
from physics.drift_processes import DriftState, StepDriftProcess, sample_displacements
from physics.ideal_gkp_decoder import map_decode_2d, standard_binning_1d
from physics.oracle_map import (
    OracleMAPResult,
    OracleTrajectoryResult,
    oracle_log_likelihoods_2d,
    oracle_map,
    oracle_map_2d,
    oracle_map_trajectory,
)


class OracleMixtureLikelihoodTest(unittest.TestCase):
    @staticmethod
    def _gaussian_density(
        point: np.ndarray,
        mean: np.ndarray,
        covariance: np.ndarray,
    ) -> float:
        residual = point - mean
        inverse = np.linalg.inv(covariance)
        normalizer = 2.0 * math.pi * math.sqrt(float(np.linalg.det(covariance)))
        return math.exp(-0.5 * float(residual @ inverse @ residual)) / normalizer

    @classmethod
    def _brute_force_mixture_likelihood(
        cls,
        syndrome: np.ndarray,
        state: DriftState,
        parity: tuple[int, int],
        radius: int = 30,
    ) -> float:
        total = 0.0
        for alias_q in range(-radius, radius + 1):
            if alias_q % 2 != parity[0]:
                continue
            for alias_p in range(-radius, radius + 1):
                if alias_p % 2 != parity[1]:
                    continue
                point = syndrome + LATTICE_CONST * np.array([alias_q, alias_p])
                core = cls._gaussian_density(point, state.mean, state.covariance)
                outlier = cls._gaussian_density(
                    point,
                    state.mean,
                    state.outlier_covariance,
                )
                total += (1.0 - state.p_outlier) * core + state.p_outlier * outlier
        return total

    def test_oracle_mixture_matches_independent_double_alias_sum(self) -> None:
        lam = LATTICE_CONST
        state = DriftState(
            mu_q=1.17 * lam,
            mu_p=-0.82 * lam,
            sigma_q=0.48 * lam,
            sigma_p=0.36 * lam,
            rho=0.71,
            p_outlier=0.17,
            outlier_scale=2.3,
        )
        syndrome = lam * np.array([0.23, -0.31])
        logs = oracle_log_likelihoods_2d(syndrome, state)
        for parity in ((0, 0), (0, 1), (1, 0), (1, 1)):
            with self.subTest(parity=parity):
                expected = self._brute_force_mixture_likelihood(
                    syndrome,
                    state,
                    parity,
                )
                actual = math.exp(float(logs[parity]))
                self.assertAlmostEqual(actual, expected, delta=5.0e-15)

    def test_mixture_endpoints_and_identical_components_reduce_exactly(self) -> None:
        lam = LATTICE_CONST
        syndrome = lam * np.array([[0.21, -0.42], [-0.33, 0.17]])
        base_kwargs = dict(
            mu_q=0.3 * lam,
            mu_p=-0.2 * lam,
            sigma_q=0.37 * lam,
            sigma_p=0.52 * lam,
            rho=-0.63,
        )
        core_only = DriftState(**base_kwargs, p_outlier=0.0, outlier_scale=3.0)
        core_logs = map_decode_2d(
            syndrome,
            core_only.covariance,
            mean=core_only.mean,
        ).log_likelihoods
        np.testing.assert_array_equal(
            oracle_log_likelihoods_2d(syndrome, core_only),
            core_logs,
        )

        outlier_only = DriftState(**base_kwargs, p_outlier=1.0, outlier_scale=3.0)
        outlier_logs = map_decode_2d(
            syndrome,
            outlier_only.outlier_covariance,
            mean=outlier_only.mean,
        ).log_likelihoods
        np.testing.assert_array_equal(
            oracle_log_likelihoods_2d(syndrome, outlier_only),
            outlier_logs,
        )

        identical = DriftState(**base_kwargs, p_outlier=0.47, outlier_scale=1.0)
        np.testing.assert_array_equal(
            oracle_log_likelihoods_2d(syndrome, identical),
            core_logs,
        )

    def test_loss_proxy_is_explicit_and_matches_independent_covariance_construction(self) -> None:
        lam = LATTICE_CONST
        state = DriftState(
            mu_q=0.1 * lam,
            mu_p=-0.3 * lam,
            sigma_q=0.31 * lam,
            sigma_p=0.44 * lam,
            rho=0.52,
            loss_gamma=0.28,
            p_outlier=0.23,
            outlier_scale=2.7,
        )
        syndrome = lam * np.array([-0.27, 0.39])
        separate = oracle_log_likelihoods_2d(syndrome, state, loss_model="separate")
        proxied = oracle_log_likelihoods_2d(
            syndrome,
            state,
            loss_model="additive_displacement_proxy",
        )
        loss_covariance = np.eye(2) * (state.loss_gamma / 2.0)
        core = map_decode_2d(
            syndrome,
            state.covariance + loss_covariance,
            mean=state.mean,
        ).log_likelihoods
        outlier = map_decode_2d(
            syndrome,
            state.outlier_covariance + loss_covariance,
            mean=state.mean,
        ).log_likelihoods
        expected = np.logaddexp(
            math.log1p(-state.p_outlier) + core,
            math.log(state.p_outlier) + outlier,
        )
        np.testing.assert_allclose(proxied, expected, atol=0.0, rtol=0.0)
        self.assertGreater(float(np.max(np.abs(proxied - separate))), 1.0e-4)


class OracleDecisionTest(unittest.TestCase):
    def test_four_parities_map_to_consistent_pauli_actions(self) -> None:
        lam = LATTICE_CONST
        expected = {
            (0, 0): (0, "I"),
            (0, 1): (1, "Z"),
            (1, 0): (2, "X"),
            (1, 1): (3, "Y"),
        }
        for parity, (logical_class, action) in expected.items():
            with self.subTest(parity=parity):
                state = DriftState(
                    mu_q=parity[0] * lam,
                    mu_p=parity[1] * lam,
                    sigma_q=0.07 * lam,
                    sigma_p=0.07 * lam,
                    p_outlier=0.03,
                    outlier_scale=2.0,
                )
                result = oracle_map_2d([0.0, 0.0], state)
                self.assertIsInstance(result, OracleMAPResult)
                self.assertEqual(result.logical_class, logical_class)
                self.assertEqual(result.logical_action, action)
                np.testing.assert_array_equal(result.parity, parity)
                np.testing.assert_array_equal(result.analog_correction, [0.0, 0.0])
                self.assertGreater(result.confidence, 0.99)
                self.assertEqual(result.evidence_scope, "nondeployable_full_state_oracle")
                self.assertEqual(result.state_regime, "base")
                self.assertFalse(result.burst_active)

    def test_batch_posterior_hard_decision_confidence_and_prior_are_consistent(self) -> None:
        lam = LATTICE_CONST
        state = DriftState(
            mu_q=0.42 * lam,
            mu_p=-0.31 * lam,
            sigma_q=0.39 * lam,
            sigma_p=0.55 * lam,
            rho=-0.77,
            p_outlier=0.14,
            outlier_scale=2.4,
            source="unit_test",
            step=12,
        )
        syndrome = lam * np.array(
            [[-0.43, 0.31], [-0.11, -0.22], [0.19, 0.47], [0.44, -0.38]]
        )
        prior = np.array([[0.11, 0.19], [0.47, 0.23]])
        result = oracle_map(syndrome, state, prior=prior)
        self.assertEqual(result.state_step, 12)
        self.assertEqual(result.state_source, "unit_test")
        self.assertEqual(result.mixture_weights, (0.86, 0.14))
        self.assertEqual(result.loss_model, "separate")
        np.testing.assert_allclose(
            np.sum(result.posterior, axis=(-2, -1)),
            1.0,
            atol=2.0e-15,
        )
        expected_class = np.argmax(result.posterior.reshape((-1, 4)), axis=-1)
        np.testing.assert_array_equal(result.logical_class, expected_class)
        np.testing.assert_array_equal(result.parity[:, 0], expected_class // 2)
        np.testing.assert_array_equal(result.parity[:, 1], expected_class % 2)
        expected_actions = np.array(["I", "Z", "X", "Y"])[expected_class]
        np.testing.assert_array_equal(result.logical_action, expected_actions)
        ordered = np.sort(result.posterior.reshape((-1, 4)), axis=-1)
        np.testing.assert_allclose(
            result.confidence,
            ordered[:, -1] - ordered[:, -2],
            atol=0.0,
        )
        np.testing.assert_array_equal(result.analog_correction, syndrome)

    def test_one_cell_mean_translation_swaps_only_matching_parity(self) -> None:
        lam = LATTICE_CONST
        state = DriftState(
            mu_q=0.23 * lam,
            mu_p=-0.38 * lam,
            sigma_q=0.41 * lam,
            sigma_p=0.49 * lam,
            rho=0.68,
            p_outlier=0.2,
            outlier_scale=2.2,
        )
        shifted_q = DriftState(
            mu_q=state.mu_q + lam,
            mu_p=state.mu_p,
            sigma_q=state.sigma_q,
            sigma_p=state.sigma_p,
            rho=state.rho,
            p_outlier=state.p_outlier,
            outlier_scale=state.outlier_scale,
        )
        syndrome = lam * np.array([0.17, -0.29])
        original = oracle_log_likelihoods_2d(syndrome, state)
        shifted = oracle_log_likelihoods_2d(syndrome, shifted_q)
        np.testing.assert_allclose(shifted[0, :], original[1, :], atol=2.0e-14)
        np.testing.assert_allclose(shifted[1, :], original[0, :], atol=2.0e-14)

    def test_time_aligned_trajectory_consumes_each_full_state(self) -> None:
        lam = LATTICE_CONST
        process = StepDriftProcess(
            DriftState(mu_q=0.0, sigma_q=0.08 * lam, sigma_p=0.08 * lam),
            DriftState(mu_q=lam, sigma_q=0.08 * lam, sigma_p=0.08 * lam),
            change_step=3,
            seed=17,
        )
        states = process.generate(6)
        syndrome = np.zeros((6, 2), dtype=float)
        result = oracle_map_trajectory(syndrome, states)
        self.assertIsInstance(result, OracleTrajectoryResult)
        np.testing.assert_array_equal(result.logical_class, [0, 0, 0, 2, 2, 2])
        np.testing.assert_array_equal(result.logical_action, ["I", "I", "I", "X", "X", "X"])
        np.testing.assert_array_equal(result.parity[:, 0], [0, 0, 0, 1, 1, 1])
        np.testing.assert_array_equal(result.parity[:, 1], 0)
        np.testing.assert_array_equal(result.state_steps, np.arange(6))
        self.assertEqual(result.state_sources, ("step",) * 6)
        np.testing.assert_array_equal(result.analog_correction, syndrome)

    def test_mixture_posterior_bayes_risk_matches_independent_monte_carlo(self) -> None:
        lam = LATTICE_CONST
        state = DriftState(
            mu_q=0.30 * lam,
            mu_p=-0.20 * lam,
            sigma_q=0.32 * lam,
            sigma_p=0.45 * lam,
            rho=0.60,
            p_outlier=0.12,
            outlier_scale=2.5,
        )
        samples, _ = sample_displacements(state, 40_000, seed=77)
        q_truth = standard_binning_1d(samples[:, 0])
        p_truth = standard_binning_1d(samples[:, 1])
        truth = 2 * np.asarray(q_truth.logical_parity) + np.asarray(p_truth.logical_parity)
        syndrome = np.column_stack((q_truth.syndrome, p_truth.syndrome))
        decisions = []
        predicted_risk = []
        for start in range(0, len(samples), 2_000):
            result = oracle_map_2d(syndrome[start : start + 2_000], state)
            decisions.append(np.asarray(result.logical_class))
            predicted_risk.append(
                1.0 - np.max(result.posterior.reshape((-1, 4)), axis=-1)
            )
        decision = np.concatenate(decisions)
        risk = np.concatenate(predicted_risk)
        incorrect = decision != truth
        residual = incorrect.astype(float) - risk
        standard_error = float(np.std(residual, ddof=1) / math.sqrt(len(residual)))
        self.assertGreater(float(np.mean(incorrect)), 0.2)
        self.assertLess(float(np.mean(incorrect)), 0.4)
        self.assertLess(abs(float(np.mean(residual))), 4.0 * standard_error)

    def test_invalid_oracle_models_inputs_and_trajectories_fail_closed(self) -> None:
        lam = LATTICE_CONST
        state = DriftState()
        invalid_calls = [
            lambda: oracle_map_2d([0.0, 0.0], object()),  # type: ignore[arg-type]
            lambda: oracle_map_2d([0.0], state),
            lambda: oracle_map_2d([lam / 2.0, 0.0], state),
            lambda: oracle_map_2d([0.0, 0.0], state, loss_model="silent"),  # type: ignore[arg-type]
            lambda: oracle_map_2d([0.0, 0.0], state, prior=[0.5, 0.5]),
            lambda: oracle_map_2d(
                [0.0, 0.0], state, prior=[[1.0, 0.0], [1.0, 1.0]]
            ),
            lambda: oracle_map_2d(
                [0.0, 0.0], state, prior=np.full((2, 2), 1.0e308)
            ),
            lambda: oracle_map_trajectory([], []),
            lambda: oracle_map_trajectory([[0.0, 0.0]], []),
            lambda: oracle_map_trajectory([[0.0, 0.0]], [object()]),  # type: ignore[list-item]
            lambda: oracle_map_trajectory([0.0, 0.0], [state]),
        ]
        for call in invalid_calls:
            with self.subTest(call=call):
                with self.assertRaises((TypeError, ValueError)):
                    call()


if __name__ == "__main__":
    unittest.main()
