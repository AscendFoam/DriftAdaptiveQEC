from __future__ import annotations

import math
import unittest

import numpy as np

from physics.constants import LATTICE_CONST
from physics.finite_energy_gkp import damped_projector_state, gaussian_envelope_state
from physics.ideal_gkp_decoder import (
    covariance_from_sigmas,
    map_decode_2d,
    standard_binning_1d,
)
from physics.logical_channel import (
    PauliChannel,
    ParityConfusion,
    finite_energy_parity_response_1d,
    parity_confusion_from_response,
    pauli_channel_from_axis_confusions,
    pauli_channel_from_joint_confusion,
    pauli_channel_from_parity_samples,
    pauli_channel_from_residual_distribution,
)


class PauliChannelMetricsTest(unittest.TestCase):
    def test_residual_parity_to_pauli_mapping(self) -> None:
        residual = np.array([[0.4, 0.2], [0.1, 0.3]])
        channel = pauli_channel_from_residual_distribution(residual)

        self.assertAlmostEqual(channel.p_i, 0.4)
        self.assertAlmostEqual(channel.p_x, 0.1)
        self.assertAlmostEqual(channel.p_y, 0.3)
        self.assertAlmostEqual(channel.p_z, 0.2)
        np.testing.assert_allclose(channel.probabilities, [0.4, 0.1, 0.3, 0.2])

    def test_ptm_and_fidelity_identities(self) -> None:
        channel = PauliChannel(p_i=0.7, p_x=0.1, p_y=0.05, p_z=0.15)
        expected_ptm = np.diag([1.0, 0.6, 0.5, 0.7])

        np.testing.assert_allclose(channel.ptm, expected_ptm, atol=2.0e-16)
        self.assertAlmostEqual(channel.entanglement_fidelity, 0.7)
        self.assertAlmostEqual(channel.average_fidelity, 0.8)
        self.assertAlmostEqual(channel.logical_error_rate, 0.3)

    def test_identity_and_uniform_pauli_channels_have_known_metrics(self) -> None:
        identity = PauliChannel(1.0, 0.0, 0.0, 0.0)
        uniform = PauliChannel(0.25, 0.25, 0.25, 0.25)

        np.testing.assert_array_equal(identity.ptm, np.eye(4))
        np.testing.assert_array_equal(uniform.ptm, np.diag([1.0, 0.0, 0.0, 0.0]))
        self.assertEqual(identity.average_fidelity, 1.0)
        self.assertEqual(uniform.entanglement_fidelity, 0.25)
        self.assertEqual(uniform.average_fidelity, 0.5)

    def test_weighted_paired_samples_aggregate_each_residual_pauli(self) -> None:
        truth = np.zeros((4, 2), dtype=np.int64)
        decoded = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        channel = pauli_channel_from_parity_samples(
            truth,
            decoded,
            weights=[1.0, 2.0, 3.0, 4.0],
        )

        np.testing.assert_allclose(
            [channel.p_i, channel.p_x, channel.p_y, channel.p_z],
            [0.1, 0.3, 0.4, 0.2],
            atol=0.0,
        )

    def test_joint_confusion_uses_xor_residual_not_raw_decision_class(self) -> None:
        confusion = np.zeros((4, 4), dtype=np.float64)
        confusion[0, 0] = 0.4  # I
        confusion[0, 2] = 0.1  # X
        confusion[1, 0] = 0.2  # truth 01 vs decision 00 -> Z
        confusion[3, 0] = 0.3  # truth 11 vs decision 00 -> Y
        channel = pauli_channel_from_joint_confusion(confusion)

        np.testing.assert_allclose(
            [channel.p_i, channel.p_x, channel.p_y, channel.p_z],
            [0.4, 0.1, 0.3, 0.2],
            atol=0.0,
        )

    def test_axis_confusions_form_explicit_independent_product_channel(self) -> None:
        q_confusion = ParityConfusion(
            matrix=np.array([[0.6, 0.1], [0.2, 0.1]]),
            captured_mass=1.0,
            decoder_name="q",
        )
        p_confusion = ParityConfusion(
            matrix=np.array([[0.5, 0.2], [0.2, 0.1]]),
            captured_mass=1.0,
            decoder_name="p",
        )
        channel = pauli_channel_from_axis_confusions(q_confusion, p_confusion)

        self.assertAlmostEqual(q_confusion.error_probability, 0.3)
        self.assertAlmostEqual(p_confusion.error_probability, 0.4)
        np.testing.assert_allclose(
            [channel.p_i, channel.p_x, channel.p_y, channel.p_z],
            [0.42, 0.18, 0.12, 0.28],
            atol=2.0e-16,
        )


class FiniteEnergyParityResponseTest(unittest.TestCase):
    def test_response_mass_posterior_and_half_open_grid(self) -> None:
        states = [
            gaussian_envelope_state("0", 0.3, envelope_kappa=0.28),
            gaussian_envelope_state("1", 0.3, envelope_kappa=0.28),
            damped_projector_state("0", 0.34),
            damped_projector_state("1", 0.34),
        ]
        for state in states:
            response = finite_energy_parity_response_1d(
                state,
                displacement_sigma=0.3 * LATTICE_CONST,
                points=1024,
            )
            step = float(response.syndrome[1] - response.syndrome[0])
            with self.subTest(model=state.model, logical=state.logical_state):
                self.assertAlmostEqual(response.captured_mass, 1.0, delta=3.0e-11)
                self.assertAlmostEqual(
                    float(np.sum(response.joint_density) * step),
                    1.0,
                    delta=3.0e-11,
                )
                np.testing.assert_allclose(
                    np.sum(response.posterior, axis=-1),
                    1.0,
                    atol=2.0e-15,
                )
                np.testing.assert_array_equal(
                    response.map_decision,
                    response.joint_density[:, 1] > response.joint_density[:, 0],
                )
                self.assertGreaterEqual(response.syndrome[0], -LATTICE_CONST / 2.0)
                self.assertLess(response.syndrome[-1], LATTICE_CONST / 2.0)

    def test_external_displacement_increases_finite_energy_residual_error(self) -> None:
        state = damped_projector_state("0", 0.34)
        low_noise = finite_energy_parity_response_1d(
            state, displacement_sigma=0.05 * LATTICE_CONST, points=2048
        )
        high_noise = finite_energy_parity_response_1d(
            state, displacement_sigma=0.4 * LATTICE_CONST, points=2048
        )
        low_confusion = parity_confusion_from_response(low_noise)
        high_confusion = parity_confusion_from_response(high_noise)

        self.assertLess(low_confusion.error_probability, 1.0e-3)
        self.assertGreater(high_confusion.error_probability, 0.2)
        self.assertGreater(
            high_confusion.error_probability,
            400.0 * low_confusion.error_probability,
        )

    def test_arbitrary_deterministic_parity_decoder_produces_auditable_confusion(self) -> None:
        state = gaussian_envelope_state("0", 0.31, envelope_kappa=0.27)
        response = finite_energy_parity_response_1d(
            state, displacement_sigma=0.35 * LATTICE_CONST, points=1024
        )
        decisions = (response.syndrome > 0.0).astype(np.int64)
        from_array = parity_confusion_from_response(
            response, decisions, decoder_name="positive-half"
        )
        from_callable = parity_confusion_from_response(
            response,
            lambda syndrome: syndrome > 0.0,
            decoder_name="positive-half-callable",
        )

        np.testing.assert_allclose(from_array.matrix, from_callable.matrix, atol=0.0)
        self.assertAlmostEqual(float(np.sum(from_array.matrix)), 1.0, places=15)
        np.testing.assert_allclose(
            from_array.truth_distribution,
            np.sum(from_array.matrix, axis=1),
            atol=0.0,
        )

    def test_analytic_state_noise_response_matches_independent_mixture_monte_carlo(self) -> None:
        state = damped_projector_state("0", 0.34)
        noise_sigma = 0.35 * LATTICE_CONST
        response = finite_energy_parity_response_1d(
            state, displacement_sigma=noise_sigma, points=1024
        )
        confusion = parity_confusion_from_response(response)

        table = state.peak_table
        variance = table.amplitude_variance
        centers = table.centers
        coefficients = table.coefficients
        pair_centers = 0.5 * (centers[:, None] + centers[None, :])
        differences = centers[:, None] - centers[None, :]
        integrated_weights = (
            coefficients[:, None]
            * coefficients[None, :]
            * np.exp(-differences**2 / (4.0 * variance))
            * math.sqrt(math.pi * variance)
        ).reshape(-1)
        integrated_weights /= np.sum(integrated_weights)

        rng = np.random.default_rng(20260714)
        sample_count = 120_000
        component = rng.choice(
            integrated_weights.size,
            size=sample_count,
            p=integrated_weights,
        )
        clean = rng.normal(
            pair_centers.reshape(-1)[component],
            math.sqrt(variance / 2.0),
        )
        noisy = clean + rng.normal(0.0, noise_sigma, size=sample_count)
        lattice_index = np.floor(noisy / LATTICE_CONST + 0.5).astype(np.int64)
        syndrome = noisy - lattice_index * LATTICE_CONST
        truth = np.mod(lattice_index, 2)
        grid_index = np.floor(
            (syndrome / LATTICE_CONST + 0.5) * response.syndrome.size
        ).astype(np.int64)
        grid_index = np.clip(grid_index, 0, response.syndrome.size - 1)
        decision = response.map_decision[grid_index]
        empirical_error = float(np.mean(truth != decision))
        analytic_error = confusion.error_probability
        standard_error = math.sqrt(
            analytic_error * (1.0 - analytic_error) / sample_count
        )

        self.assertLessEqual(
            abs(empirical_error - analytic_error),
            5.0 * standard_error + 3.0e-4,
        )

    def test_finite_energy_axis_confusions_generate_full_pauli_metrics(self) -> None:
        q_state = damped_projector_state("0", 0.34)
        p_state = gaussian_envelope_state("0", 0.27, envelope_kappa=0.31)
        q_confusion = parity_confusion_from_response(
            finite_energy_parity_response_1d(
                q_state, displacement_sigma=0.31 * LATTICE_CONST, points=1024
            )
        )
        p_confusion = parity_confusion_from_response(
            finite_energy_parity_response_1d(
                p_state, displacement_sigma=0.24 * LATTICE_CONST, points=1024
            )
        )
        channel = pauli_channel_from_axis_confusions(q_confusion, p_confusion)

        self.assertGreater(channel.p_x, 0.0)
        self.assertGreater(channel.p_z, 0.0)
        self.assertGreater(channel.p_y, 0.0)
        self.assertAlmostEqual(float(np.sum(channel.probabilities)), 1.0, places=15)
        self.assertAlmostEqual(
            channel.average_fidelity,
            (2.0 * channel.entanglement_fidelity + 1.0) / 3.0,
            places=15,
        )

    def test_correlated_joint_map_samples_preserve_joint_y_residual(self) -> None:
        covariance = covariance_from_sigmas(
            0.42 * LATTICE_CONST,
            0.42 * LATTICE_CONST,
            0.9,
        )
        rng = np.random.default_rng(20260714)
        samples = rng.multivariate_normal([0.0, 0.0], covariance, size=20_000)
        q_truth = standard_binning_1d(samples[:, 0])
        p_truth = standard_binning_1d(samples[:, 1])
        truth = np.column_stack((q_truth.logical_parity, p_truth.logical_parity))
        syndrome = np.column_stack((q_truth.syndrome, p_truth.syndrome))
        decoded = map_decode_2d(syndrome, covariance).parity
        channel = pauli_channel_from_parity_samples(
            truth,
            decoded,
            source="correlated_joint_map_mc",
        )

        class_error = float(np.mean(np.any(truth != decoded, axis=1)))
        self.assertAlmostEqual(channel.logical_error_rate, class_error, places=15)
        self.assertGreater(channel.p_y, 0.2)
        independent_y = (channel.p_x + channel.p_y) * (channel.p_z + channel.p_y)
        self.assertGreater(channel.p_y, 4.0 * independent_y)

    def test_invalid_channels_responses_and_decoders_fail_closed(self) -> None:
        invalid_calls = [
            lambda: PauliChannel(0.8, 0.1, 0.1, 0.1),
            lambda: PauliChannel(1.0, -0.1, 0.1, 0.0),
            lambda: pauli_channel_from_residual_distribution([1.0, 0.0]),
            lambda: pauli_channel_from_residual_distribution(np.zeros((2, 2))),
            lambda: pauli_channel_from_joint_confusion(np.zeros((3, 3))),
            lambda: pauli_channel_from_parity_samples([[0, 2]], [[0, 0]]),
            lambda: pauli_channel_from_parity_samples([[0, 0]], [[0, 0]], weights=[0.0]),
            lambda: finite_energy_parity_response_1d(
                gaussian_envelope_state("+", 0.3), points=512
            ),
            lambda: finite_energy_parity_response_1d(
                gaussian_envelope_state("0", 0.3), displacement_sigma=-1.0
            ),
            lambda: finite_energy_parity_response_1d(
                gaussian_envelope_state("0", 0.3), points=64
            ),
            lambda: finite_energy_parity_response_1d(
                gaussian_envelope_state("0", 0.3), displacement_sigma=1.0e20
            ),
            lambda: finite_energy_parity_response_1d(
                gaussian_envelope_state("0", 0.3), displacement_sigma=1.0e308
            ),
        ]
        for call in invalid_calls:
            with self.subTest(call=call):
                with self.assertRaises((ValueError, TypeError)):
                    call()

        response = finite_energy_parity_response_1d(
            gaussian_envelope_state("0", 0.3), points=512
        )
        with self.assertRaises(ValueError):
            parity_confusion_from_response(response, np.zeros(10, dtype=np.int64))
        with self.assertRaises(ValueError):
            parity_confusion_from_response(response, np.full(512, 2))


if __name__ == "__main__":
    unittest.main()
