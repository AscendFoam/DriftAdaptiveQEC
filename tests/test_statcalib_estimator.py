from __future__ import annotations

import unittest

import numpy as np

from cnn_fpga.decoder.statcalib import (
    STATCALIB_REASON_DIAGNOSTIC_ERROR,
    STATCALIB_REASON_SIGNAL_INSUFFICIENT,
    STATCALIB_STATUS_DIAGNOSTIC_ERROR,
    STATCALIB_STATUS_GENERATED,
    STATCALIB_STATUS_NOT_GENERATED,
    StatCalibInput,
    run_statcalib_estimator,
)
from cnn_fpga.runtime.param_bank import DecoderRuntimeParams


class StatCalibEstimatorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.prior = DecoderRuntimeParams(
            K=np.array([[0.8, 0.1], [0.1, 0.7]], dtype=float),
            b=np.array([0.02, -0.03], dtype=float),
            metadata={"runtime_mode": "statcalib"},
        )

    def _input(
        self,
        *,
        mass: float = 1.0,
        mean_syndrome_q: float = 0.03,
        mean_syndrome_p: float = -0.02,
        valid_window: float = 1.0,
    ) -> StatCalibInput:
        return StatCalibInput(
            window_id=7,
            slow_update_index=3,
            prior_decoder_params=self.prior,
            histogram_summary={"mass": mass, "q_mean": 0.05, "p_mean": 0.02},
            calibration_features={
                "mean_syndrome_q": mean_syndrome_q,
                "mean_syndrome_p": mean_syndrome_p,
                "valid_window": valid_window,
            },
            teacher_prediction={"sigma": 0.25, "mu_q": 0.01, "mu_p": -0.02, "theta_deg": 1.5},
            source="statcalib_test",
        )

    def test_invalid_window_returns_not_generated(self) -> None:
        output = run_statcalib_estimator(self._input(valid_window=0.0))

        self.assertEqual(output.status, STATCALIB_STATUS_NOT_GENERATED)
        self.assertEqual(output.reason, STATCALIB_REASON_SIGNAL_INSUFFICIENT)

    def test_zero_histogram_mass_returns_not_generated(self) -> None:
        output = run_statcalib_estimator(self._input(mass=0.0))

        self.assertEqual(output.status, STATCALIB_STATUS_NOT_GENERATED)
        self.assertEqual(output.reason, STATCALIB_REASON_SIGNAL_INSUFFICIENT)

    def test_signal_below_threshold_returns_not_generated(self) -> None:
        output = run_statcalib_estimator(
            self._input(mean_syndrome_q=1.0e-6, mean_syndrome_p=1.0e-6),
            signal_threshold=1.0e-3,
        )

        self.assertEqual(output.status, STATCALIB_STATUS_NOT_GENERATED)
        self.assertEqual(output.reason, STATCALIB_REASON_SIGNAL_INSUFFICIENT)

    def test_clip_boundary_is_applied_to_delta_b(self) -> None:
        output = run_statcalib_estimator(
            self._input(mean_syndrome_q=0.4, mean_syndrome_p=-0.4),
            residual_scale_b=1.0,
            residual_clip_b=0.08,
        )

        self.assertEqual(output.status, STATCALIB_STATUS_GENERATED)
        np.testing.assert_allclose(output.delta_b, [0.08, -0.08], atol=1.0e-8)
        self.assertEqual(output.metadata["delta_b_pre_clip"], [0.4, -0.4])

    def test_invalid_estimator_arg_returns_diagnostic_error(self) -> None:
        output = run_statcalib_estimator(self._input(), residual_clip_b="bad")

        self.assertEqual(output.status, STATCALIB_STATUS_DIAGNOSTIC_ERROR)
        self.assertEqual(output.reason, STATCALIB_REASON_DIAGNOSTIC_ERROR)
        self.assertIn("error", output.metadata)


if __name__ == "__main__":
    unittest.main()
