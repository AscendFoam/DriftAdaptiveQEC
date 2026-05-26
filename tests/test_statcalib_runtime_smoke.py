from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from cnn_fpga.decoder.param_mapper import NoisePrediction, ParamMapper
from cnn_fpga.runtime.param_bank import DecoderRuntimeParams
from cnn_fpga.runtime.scheduler import WindowFrame
from cnn_fpga.runtime.slow_loop_runtime import SlowLoopRuntime, SlowLoopRuntimeConfig


class StatCalibRuntimeSmokeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.runtime = SlowLoopRuntime(
            SlowLoopRuntimeConfig(
                mode="statcalib",
                teacher_mode="ukf",
                residual_scale_b=1.0,
                residual_clip_b=0.05,
                statcalib_signal_threshold=1.0e-3,
            ),
            param_mapper=ParamMapper.from_config({}),
            seed=123,
        )
        self.active_params = DecoderRuntimeParams(
            K=np.array([[0.7, 0.0], [0.0, 0.7]], dtype=float),
            b=np.array([0.0, 0.0], dtype=float),
            metadata={"runtime_mode": "ukf"},
        )

    def _window(self, *, mean_syndrome_q: float, mean_syndrome_p: float, valid_window: bool = True) -> WindowFrame:
        return WindowFrame(
            window_id=11,
            start_epoch=0,
            end_epoch=15,
            ready_time_us=10.0,
            payload={
                "histogram": np.array([[0.2, 0.1], [0.3, 0.4]], dtype=np.float32),
                "diagnostics": {
                    "valid_window": valid_window,
                    "overflow_ratio": 0.01,
                    "window_ler": 0.02,
                },
                "window_stats": {
                    "mean_syndrome_q": mean_syndrome_q,
                    "mean_syndrome_p": mean_syndrome_p,
                    "std_syndrome_q": 0.03,
                    "std_syndrome_p": 0.04,
                },
            },
        )

    def test_statcalib_generated_path_emits_metadata_and_params(self) -> None:
        teacher_prediction = NoisePrediction(
            sigma=0.22,
            mu_q=0.01,
            mu_p=-0.02,
            theta_deg=1.5,
            source="ukf_test",
        )
        with patch.object(self.runtime, "_predict_teacher", return_value=teacher_prediction):
            proposed = self.runtime(self._window(mean_syndrome_q=0.03, mean_syndrome_p=-0.02), self.active_params)

        self.assertEqual(proposed.metadata["runtime_mode"], "statcalib")
        self.assertEqual(proposed.metadata["teacher_mode"], "ukf")
        self.assertEqual(proposed.metadata["statcalib_status"], "generated")
        self.assertEqual(proposed.metadata["statcalib_reason"], "statcalib_params_emitted")
        self.assertEqual(proposed.metadata["statcalib_output"]["status"], "generated")
        np.testing.assert_allclose(proposed.metadata["applied_delta_b"], [0.03, -0.02], atol=1.0e-8)
        teacher_b = np.asarray(proposed.metadata["teacher_params"]["b"], dtype=float)
        np.testing.assert_allclose(proposed.b, teacher_b + np.array([0.03, -0.02], dtype=float), atol=1.0e-8)

    def test_statcalib_low_signal_falls_back_to_teacher_params(self) -> None:
        teacher_prediction = NoisePrediction(
            sigma=0.22,
            mu_q=0.01,
            mu_p=-0.02,
            theta_deg=1.5,
            source="ukf_test",
        )
        with patch.object(self.runtime, "_predict_teacher", return_value=teacher_prediction):
            proposed = self.runtime(self._window(mean_syndrome_q=1.0e-6, mean_syndrome_p=1.0e-6), self.active_params)

        self.assertEqual(proposed.metadata["runtime_mode"], "statcalib")
        self.assertEqual(proposed.metadata["statcalib_status"], "not_generated")
        self.assertEqual(proposed.metadata["statcalib_reason"], "insufficient_calibration_signal")
        self.assertEqual(proposed.metadata["statcalib_fallback"], "teacher_params")
        teacher_b = np.asarray(proposed.metadata["teacher_params"]["b"], dtype=float)
        np.testing.assert_allclose(proposed.b, teacher_b, atol=1.0e-8)

    def test_statcalib_teacher_mode_does_not_leak_into_other_modes(self) -> None:
        cfg = SlowLoopRuntimeConfig.from_config(
            {
                "slow_loop": {
                    "mode": "hybrid_residual_b",
                    "statcalib": {"teacher_mode": "ekf"},
                    "hybrid_residual_b": {},
                    "hybrid_residual_mu": {},
                    "constant_residual_mu": {},
                    "particle_filter_residual_b": {},
                    "rls_residual_b": {},
                }
            }
        )

        self.assertEqual(cfg.teacher_mode, "window_variance")

    def test_statcalib_mode_uses_statcalib_teacher_mode(self) -> None:
        cfg = SlowLoopRuntimeConfig.from_config(
            {
                "slow_loop": {
                    "mode": "statcalib",
                    "statcalib": {"teacher_mode": "ekf"},
                }
            }
        )

        self.assertEqual(cfg.teacher_mode, "ekf")


if __name__ == "__main__":
    unittest.main()
