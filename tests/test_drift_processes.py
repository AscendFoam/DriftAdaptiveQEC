from __future__ import annotations

import math
import unittest
import warnings

import numpy as np

from physics.drift_processes import (
    BurstDriftProcess,
    ConstantDriftProcess,
    DriftState,
    LegacyRunWithDriftAdapter,
    LossDriftProcess,
    MeanDriftProcess,
    OutlierRateDriftProcess,
    StepDriftProcess,
    TelegraphDriftProcess,
    VarianceDriftProcess,
    as_run_with_drift_callback,
    sample_displacements,
)


class DriftStateTest(unittest.TestCase):
    def test_state_exposes_covariance_mixture_loss_and_principal_axis(self) -> None:
        state = DriftState(
            mu_q=0.2,
            mu_p=-0.1,
            sigma_q=0.3,
            sigma_p=0.5,
            rho=0.4,
            loss_gamma=0.2,
            p_outlier=0.1,
            outlier_scale=4.0,
        )
        expected = np.array([[0.09, 0.06], [0.06, 0.25]])
        np.testing.assert_allclose(state.mean, [0.2, -0.1], atol=0.0, rtol=0.0)
        np.testing.assert_allclose(state.covariance, expected, atol=1.0e-15)
        np.testing.assert_allclose(state.outlier_covariance, 16.0 * expected, atol=1.0e-15)
        np.testing.assert_allclose(state.mixture_covariance, 2.5 * expected, atol=1.0e-15)
        self.assertAlmostEqual(state.eta, math.exp(-0.2), places=15)

        covariance = state.covariance
        angle = 0.5 * math.atan2(
            2.0 * covariance[0, 1],
            covariance[0, 0] - covariance[1, 1],
        )
        self.assertAlmostEqual(state.principal_angle, angle, places=15)

    def test_state_and_workload_boundaries_fail_closed(self) -> None:
        invalid_calls = [
            lambda: DriftState(step=-1),
            lambda: DriftState(step=1.5),  # type: ignore[arg-type]
            lambda: DriftState(time=-1.0),
            lambda: DriftState(mu_q=float("nan")),
            lambda: DriftState(sigma_q=0.0),
            lambda: DriftState(sigma_p=float("inf")),
            lambda: DriftState(rho=1.0),
            lambda: DriftState(rho=-1.0),
            lambda: DriftState(loss_gamma=-0.1),
            lambda: DriftState(p_outlier=-0.1),
            lambda: DriftState(p_outlier=1.1),
            lambda: DriftState(outlier_scale=0.9),
            lambda: DriftState(source=""),
            lambda: DriftState(regime="  "),
            lambda: DriftState(seed=-1),
            lambda: DriftState(event_id=-1),
            lambda: ConstantDriftProcess().generate(10_000_001),
        ]
        for call in invalid_calls:
            with self.subTest(call=call):
                with self.assertRaises((TypeError, ValueError)):
                    call()


class DeterministicDriftProcessTest(unittest.TestCase):
    def test_mean_drift_combines_linear_and_periodic_terms(self) -> None:
        base = DriftState(mu_q=0.1, mu_p=-0.2)
        process = MeanDriftProcess(
            base=base,
            rate_q=0.03,
            rate_p=-0.04,
            amplitude_q=0.5,
            amplitude_p=0.25,
            period=4.0,
            dt=0.5,
            seed=17,
        )
        state = process.state_at(2)
        self.assertEqual(state.step, 2)
        self.assertEqual(state.time, 1.0)
        self.assertAlmostEqual(state.mu_q, 0.1 + 0.03 + 0.5, places=15)
        self.assertAlmostEqual(state.mu_p, -0.2 - 0.04 + 0.25, places=15)
        self.assertEqual(state.source, "mean")
        self.assertEqual(state.seed, 17)
        self.assertEqual(process.generate(3)[2], state)

    def test_variance_drift_uses_positive_and_positive_definite_coordinates(self) -> None:
        base = DriftState(sigma_q=0.2, sigma_p=0.6, rho=-0.3)
        process = VarianceDriftProcess(
            base=base,
            log_sigma_rate_q=math.log(2.0),
            log_sigma_rate_p=math.log(0.5),
            fisher_rho_rate=0.7,
            dt=0.2,
        )
        state = process.state_at(5)
        self.assertAlmostEqual(state.sigma_q, 0.4, places=15)
        self.assertAlmostEqual(state.sigma_p, 0.3, places=15)
        self.assertAlmostEqual(
            state.rho,
            math.tanh(math.atanh(-0.3) + 0.7),
            places=15,
        )
        self.assertGreater(np.linalg.det(state.covariance), 0.0)

    def test_loss_and_outlier_rate_drift_are_bounded_and_monotone(self) -> None:
        base = DriftState(loss_gamma=0.01, p_outlier=0.02, outlier_scale=5.0)
        loss = LossDriftProcess(base=base, target_gamma=0.31, time_constant=8.0)
        outlier = OutlierRateDriftProcess(
            base=base,
            target_probability=0.42,
            time_constant=10.0,
        )
        loss_values = np.array([state.loss_gamma for state in loss.generate(80)])
        outlier_values = np.array([state.p_outlier for state in outlier.generate(100)])
        self.assertEqual(loss_values[0], base.loss_gamma)
        self.assertEqual(outlier_values[0], base.p_outlier)
        self.assertTrue(np.all(np.diff(loss_values) > 0.0))
        self.assertTrue(np.all(np.diff(outlier_values) > 0.0))
        self.assertTrue(np.all((loss_values >= 0.01) & (loss_values < 0.31)))
        self.assertTrue(np.all((outlier_values >= 0.02) & (outlier_values < 0.42)))
        self.assertAlmostEqual(loss.state_at(400).loss_gamma, 0.31, places=15)
        self.assertAlmostEqual(outlier.state_at(500).p_outlier, 0.42, places=15)

    def test_step_drift_has_exact_boundary_and_prefix_semantics(self) -> None:
        before = DriftState(mu_q=-0.3, sigma_q=0.2, sigma_p=0.25)
        after = DriftState(mu_q=0.7, sigma_q=0.5, sigma_p=0.55, loss_gamma=0.1)
        process = StepDriftProcess(before, after, change_step=4, dt=0.25, seed=8)
        states = process.generate(7)
        self.assertEqual([state.regime for state in states], [
            "before", "before", "before", "before", "after", "after", "after"
        ])
        self.assertEqual(states[3].mu_q, -0.3)
        self.assertEqual(states[4].mu_q, 0.7)
        self.assertEqual(states[4].event_id, 1)
        self.assertEqual(states[6].time, 1.5)
        self.assertEqual(process.generate(4), process.generate(7)[:4])

    def test_invalid_process_parameters_fail_closed(self) -> None:
        state = DriftState()
        invalid_calls = [
            lambda: MeanDriftProcess(amplitude_q=0.1),
            lambda: MeanDriftProcess(period=0.0),
            lambda: VarianceDriftProcess(dt=0.0),
            lambda: VarianceDriftProcess(log_sigma_rate_q=1.0).state_at(1000),
            lambda: LossDriftProcess(target_gamma=-0.1),
            lambda: LossDriftProcess(time_constant=0.0),
            lambda: OutlierRateDriftProcess(target_probability=1.1),
            lambda: StepDriftProcess(state, state, change_step=-1),
        ]
        for call in invalid_calls:
            with self.subTest(call=call):
                with self.assertRaises((TypeError, ValueError)):
                    call()


class StatefulDriftProcessTest(unittest.TestCase):
    def test_telegraph_fixed_seed_is_prefix_reproducible_and_persistent(self) -> None:
        process = TelegraphDriftProcess(
            DriftState(mu_q=-0.4),
            DriftState(mu_q=0.8),
            p_a_to_b=0.04,
            p_b_to_a=0.04,
            seed=20260714,
        )
        short = process.generate(700)
        long = process.generate(4_000)
        self.assertEqual(short, long[:700])
        self.assertEqual(process.state_at(699), short[-1])
        regimes = np.array([state.regime == "b" for state in long], dtype=np.int8)
        switches = int(np.count_nonzero(np.diff(regimes)))
        self.assertGreater(switches, 80)
        self.assertLess(switches, 260)
        self.assertGreater(int(np.count_nonzero(regimes)), 500)
        self.assertLess(int(np.count_nonzero(regimes)), 3_500)

    def test_telegraph_transition_statistics_recover_asymmetric_markov_law(self) -> None:
        process = TelegraphDriftProcess(
            DriftState(mu_p=-1.0),
            DriftState(mu_p=1.0),
            p_a_to_b=0.08,
            p_b_to_a=0.24,
            seed=31,
        )
        states = process.generate(120_000)
        regime_b = np.array([state.regime == "b" for state in states], dtype=bool)
        prev = regime_b[:-1]
        nxt = regime_b[1:]
        empirical_ab = float(np.mean(nxt[~prev]))
        empirical_ba = float(np.mean(~nxt[prev]))
        occupancy_b = float(np.mean(regime_b[5_000:]))
        self.assertAlmostEqual(empirical_ab, 0.08, delta=0.004)
        self.assertAlmostEqual(empirical_ba, 0.24, delta=0.007)
        self.assertAlmostEqual(occupancy_b, 0.25, delta=0.015)

    def test_telegraph_probability_one_alternates_instead_of_iid_sampling(self) -> None:
        process = TelegraphDriftProcess(
            DriftState(),
            DriftState(sigma_q=0.5),
            p_a_to_b=1.0,
            p_b_to_a=1.0,
            initial_regime="a",
            seed=4,
        )
        self.assertEqual(
            [state.regime for state in process.generate(6)],
            ["a", "b", "a", "b", "a", "b"],
        )

    def test_burst_has_exact_duration_and_cooldown_state_machine(self) -> None:
        process = BurstDriftProcess(
            DriftState(sigma_q=0.2),
            DriftState(sigma_q=0.9, p_outlier=0.4, outlier_scale=6.0),
            onset_probability=1.0,
            min_duration=3,
            max_duration=3,
            cooldown_steps=2,
            seed=9,
        )
        states = process.generate(12)
        self.assertEqual(
            [state.burst_active for state in states],
            [True, True, True, False, False, True, True, True, False, False, True, True],
        )
        self.assertEqual([state.event_id for state in states[:10]], [1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
        self.assertEqual(states[2].sigma_q, 0.9)
        self.assertEqual(states[3].sigma_q, 0.2)

    def test_random_burst_prefix_run_lengths_and_gaps_are_valid(self) -> None:
        process = BurstDriftProcess(
            DriftState(p_outlier=0.01, outlier_scale=4.0),
            DriftState(p_outlier=0.5, outlier_scale=8.0),
            onset_probability=0.12,
            min_duration=2,
            max_duration=6,
            cooldown_steps=3,
            seed=2718,
        )
        short = process.generate(1_000)
        long = process.generate(20_000)
        self.assertEqual(short, long[:1_000])
        self.assertEqual(process.state_at(999), short[-1])
        active = np.array([state.burst_active for state in long], dtype=bool)
        padded = np.concatenate(([False], active, [False]))
        starts = np.flatnonzero(~padded[:-1] & padded[1:])
        ends = np.flatnonzero(padded[:-1] & ~padded[1:])
        lengths = ends - starts
        self.assertGreater(len(lengths), 500)
        self.assertTrue(np.all((lengths >= 2) & (lengths <= 6)))
        self.assertTrue(np.all(starts[1:] - ends[:-1] >= 3))

    def test_invalid_stateful_process_parameters_fail_closed(self) -> None:
        state = DriftState()
        invalid_calls = [
            lambda: TelegraphDriftProcess(state, state, p_a_to_b=-0.1),
            lambda: TelegraphDriftProcess(state, state, p_b_to_a=1.1),
            lambda: TelegraphDriftProcess(state, state, initial_regime="x"),
            lambda: BurstDriftProcess(state, state, onset_probability=1.1),
            lambda: BurstDriftProcess(state, state, min_duration=0),
            lambda: BurstDriftProcess(state, state, min_duration=4, max_duration=3),
            lambda: BurstDriftProcess(state, state, cooldown_steps=-1),
        ]
        for call in invalid_calls:
            with self.subTest(call=call):
                with self.assertRaises((TypeError, ValueError)):
                    call()


class SamplingAndLegacyAdapterTest(unittest.TestCase):
    def test_mixture_sampler_recovers_mean_covariance_and_outlier_rate(self) -> None:
        state = DriftState(
            mu_q=0.2,
            mu_p=-0.1,
            sigma_q=0.3,
            sigma_p=0.5,
            rho=0.35,
            p_outlier=0.08,
            outlier_scale=4.0,
        )
        samples, outliers = sample_displacements(state, 300_000, seed=145)
        np.testing.assert_allclose(np.mean(samples, axis=0), state.mean, atol=0.006)
        np.testing.assert_allclose(
            np.cov(samples, rowvar=False, ddof=0),
            state.mixture_covariance,
            rtol=0.025,
            atol=0.004,
        )
        standard_error = math.sqrt(state.p_outlier * (1.0 - state.p_outlier) / len(outliers))
        self.assertLess(abs(float(np.mean(outliers)) - state.p_outlier), 5.0 * standard_error)

    def test_sampler_is_reproducible_and_rejects_ambiguous_rng(self) -> None:
        state = DriftState(p_outlier=0.2, outlier_scale=3.0)
        first = sample_displacements(state, 100, seed=11)
        second = sample_displacements(state, 100, seed=11)
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])
        with self.assertRaises(ValueError):
            sample_displacements(state, 10, seed=1, rng=np.random.default_rng(1))
        with self.assertRaises(ValueError):
            sample_displacements(state, 0)

    def test_legacy_adapter_matches_rms_contract_and_old_simulator(self) -> None:
        state = DriftState(
            mu_q=0.1,
            mu_p=-0.2,
            sigma_q=0.2,
            sigma_p=0.4,
            rho=0.3,
            loss_gamma=0.02,
            p_outlier=0.1,
            outlier_scale=3.0,
        )
        process = ConstantDriftProcess(state, seed=42)
        adapter = as_run_with_drift_callback(process, delta=0.27)
        sigma, delta, theta = adapter(3)
        expected_variance = float(np.trace(state.mixture_covariance)) / 2.0
        expected_variance += (state.mu_q**2 + state.mu_p**2) / 2.0
        expected_variance += state.loss_gamma / 2.0
        self.assertAlmostEqual(sigma, math.sqrt(expected_variance), places=15)
        self.assertEqual(delta, 0.27)
        self.assertAlmostEqual(theta, state.principal_angle, places=15)
        self.assertEqual(adapter.last_state, process.state_at(3))

        class FakeCorrector:
            def __init__(self) -> None:
                self.sigmas: list[float] = []

            def evaluate_performance(self, n_samples: int, error_sigma: float) -> dict[str, float]:
                self.asserted_samples = n_samples
                self.sigmas.append(error_sigma)
                return {"logical_error_rate": error_sigma}

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Strawberry Fields not available.*",
                category=ImportWarning,
            )
            from physics.error_correction import QECSimulator

        corrector = FakeCorrector()
        result = QECSimulator(corrector=corrector).run_with_drift(
            n_timesteps=3,
            drift_model=adapter,
            recalibrate_every=50,
        )
        self.assertEqual(corrector.sigmas, [sigma, sigma, sigma])
        np.testing.assert_allclose(result["error_rates"], [sigma, sigma, sigma], atol=0.0)

    def test_invalid_legacy_adapter_fails_closed(self) -> None:
        with self.assertRaises(ValueError):
            LegacyRunWithDriftAdapter(ConstantDriftProcess(), delta=0.0)
        with self.assertRaises(TypeError):
            LegacyRunWithDriftAdapter(object())  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
