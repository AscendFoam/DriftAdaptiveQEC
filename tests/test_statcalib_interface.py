from __future__ import annotations

import unittest

import numpy as np

from cnn_fpga.decoder.statcalib import (
    STATCALIB_REASON_INTERFACE_VALIDATION_FAILED,
    STATCALIB_REASON_MODE_NOT_APPLICABLE,
    STATCALIB_REASON_PARAMS_EMITTED,
    STATCALIB_REASON_SIGNAL_INSUFFICIENT,
    STATCALIB_STATUS_GENERATED,
    STATCALIB_STATUS_NOT_APPLICABLE,
    STATCALIB_STATUS_NOT_GENERATED,
    StatCalibInput,
    StatCalibOutput,
)
from cnn_fpga.runtime.param_bank import DecoderRuntimeParams


class StatCalibInterfaceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.prior = DecoderRuntimeParams(
            K=np.array([[0.8, 0.1], [0.1, 0.7]], dtype=float),
            b=np.array([0.02, -0.03], dtype=float),
            metadata={"runtime_mode": "hybrid_residual_b"},
        )

    def test_generated_output_converts_to_runtime_params(self) -> None:
        contract_input = StatCalibInput(
            window_id=7,
            slow_update_index=3,
            prior_decoder_params=self.prior,
            histogram_summary={"mass": 1.0, "q_mean": 0.05},
            calibration_features={"teacher_b_q": 0.02, "teacher_b_p": -0.03},
            teacher_prediction={"sigma": 0.25, "mu_q": 0.01, "mu_p": -0.02, "theta_deg": 1.5},
            source="statcalib_smoke",
            provenance={"task_id": "T30"},
        )
        output = StatCalibOutput.from_delta_b(
            contract_input,
            delta_b=[0.01, -0.02],
            metadata={"estimator": "contract_only"},
        )

        self.assertEqual(output.status, STATCALIB_STATUS_GENERATED)
        self.assertEqual(output.reason, STATCALIB_REASON_PARAMS_EMITTED)
        np.testing.assert_allclose(output.K, self.prior.K)
        np.testing.assert_allclose(output.b, np.array([0.03, -0.05], dtype=float))
        runtime_params = output.to_runtime_params()
        np.testing.assert_allclose(runtime_params.K, self.prior.K)
        np.testing.assert_allclose(runtime_params.b, np.array([0.03, -0.05], dtype=float))
        self.assertEqual(runtime_params.metadata["runtime_mode"], "statcalib")
        self.assertEqual(runtime_params.metadata["statcalib_status"], STATCALIB_STATUS_GENERATED)

    def test_not_generated_output_preserves_null_semantics(self) -> None:
        output = StatCalibOutput.not_generated(
            reason=STATCALIB_REASON_SIGNAL_INSUFFICIENT,
            metadata={"why": "low_signal"},
        )
        self.assertEqual(output.status, STATCALIB_STATUS_NOT_GENERATED)
        self.assertIsNone(output.K)
        self.assertIsNone(output.b)
        self.assertIsNone(output.delta_b)
        with self.assertRaises(ValueError):
            output.to_runtime_params()

    def test_not_applicable_output_is_separate_from_not_generated(self) -> None:
        output = StatCalibOutput.not_applicable(metadata={"mode": "ukf"})
        self.assertEqual(output.status, STATCALIB_STATUS_NOT_APPLICABLE)
        self.assertEqual(output.reason, STATCALIB_REASON_MODE_NOT_APPLICABLE)
        with self.assertRaises(ValueError):
            output.to_runtime_params()

    def test_invalid_reason_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            StatCalibOutput(
                status=STATCALIB_STATUS_NOT_GENERATED,
                reason="made_up_reason",
            )

    def test_invalid_input_prediction_shape_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            StatCalibInput(
                window_id=0,
                slow_update_index=0,
                prior_decoder_params=self.prior,
                histogram_summary={"mass": 1.0},
                calibration_features={"teacher_b_q": 0.02},
                teacher_prediction={"sigma": 0.25, "mu_q": 0.01},
            )

    def test_non_generated_output_cannot_carry_runtime_arrays(self) -> None:
        with self.assertRaises(ValueError):
            StatCalibOutput(
                status=STATCALIB_STATUS_NOT_GENERATED,
                reason=STATCALIB_REASON_INTERFACE_VALIDATION_FAILED,
                K=np.eye(2, dtype=float),
            )


if __name__ == "__main__":
    unittest.main()
