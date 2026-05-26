from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cnn_fpga.benchmark.run_hil_suite import _aggregate_statcalib_diagnostics
from cnn_fpga.benchmark.run_p4_multiscenario_benchmark import _aggregate_status_field, _write_report


class StatCalibAggregationTest(unittest.TestCase):
    def test_hil_aggregate_defaults_to_not_applicable_when_missing(self) -> None:
        summary = _aggregate_statcalib_diagnostics(
            [
                {"proposed_params": {"metadata": {"runtime_mode": "ukf"}}},
                {"proposed_params": {"metadata": {"runtime_mode": "hybrid_residual_b"}}},
            ]
        )

        self.assertEqual(summary["statcalib_status"], "not_applicable")
        self.assertEqual(summary["statcalib_reason"], "mode_does_not_emit_statcalib")
        self.assertEqual(summary["statcalib_windows_observed"], 0)

    def test_hil_aggregate_generated_status_counts(self) -> None:
        summary = _aggregate_statcalib_diagnostics(
            [
                {
                    "proposed_params": {
                        "metadata": {
                            "statcalib_status": "generated",
                            "statcalib_reason": "statcalib_params_emitted",
                            "statcalib_metadata": {"signal_norm": 0.2},
                        }
                    }
                },
                {
                    "proposed_params": {
                        "metadata": {
                            "statcalib_status": "generated",
                            "statcalib_reason": "statcalib_params_emitted",
                            "statcalib_metadata": {"signal_norm": 0.4},
                        }
                    }
                },
            ]
        )

        self.assertEqual(summary["statcalib_status"], "generated")
        self.assertEqual(summary["statcalib_reason"], "statcalib_params_emitted")
        self.assertEqual(summary["statcalib_generated_windows"], 2)
        self.assertAlmostEqual(summary["statcalib_signal_norm_mean"], 0.3)

    def test_hil_aggregate_mixed_status_is_explicit(self) -> None:
        summary = _aggregate_statcalib_diagnostics(
            [
                {"proposed_params": {"metadata": {"statcalib_status": "generated", "statcalib_reason": "statcalib_params_emitted"}}},
                {
                    "proposed_params": {
                        "metadata": {
                            "statcalib_status": "not_generated",
                            "statcalib_reason": "insufficient_calibration_signal",
                            "statcalib_fallback": "teacher_params",
                        }
                    }
                },
            ]
        )

        self.assertEqual(summary["statcalib_status"], "mixed")
        self.assertEqual(summary["statcalib_reason"], "mixed")
        self.assertEqual(summary["statcalib_fallback_counts"], {"teacher_params": 1})

    def test_benchmark_status_field_defaults_and_mixed_are_deterministic(self) -> None:
        self.assertEqual(_aggregate_status_field([{}, {}], "statcalib_status", default="not_applicable"), "not_applicable")
        self.assertEqual(
            _aggregate_status_field(
                [{"statcalib_status": "generated"}, {"statcalib_status": "not_generated"}],
                "statcalib_status",
                default="not_applicable",
            ),
            "mixed",
        )

    def test_report_writes_statcalib_column(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "report.md"
            _write_report(
                report_path,
                protocol={"protocol_id": "test", "report_template_version": "v1", "repeats": 1, "real_board_policy": "conditional_extension"},
                comparison_rows=[
                    {
                        "scenario": "static_bias_theta",
                        "mode_label": "StatCalib Comparator",
                        "final_ler_mean": 0.1,
                        "final_ler_std": 0.0,
                        "overflow_rate_mean": 0.01,
                        "histogram_input_saturation_rate_mean": 0.01,
                        "n_commits_applied_mean": 10.0,
                        "slow_update_violation_rate_mean": 0.0,
                        "fast_cycle_violation_rate_mean": 0.0,
                        "dominant_overflow_source": "histogram_input",
                        "teacher_diagnostics_status": "not_applicable",
                        "statcalib_status": "generated",
                        "artifact_path": "",
                    }
                ],
                delta_rows=[],
                scenario_winners=[],
            )
            text = report_path.read_text(encoding="utf-8")

        self.assertIn("| Teacher Diag | Statcalib | Artifact |", text)
        self.assertIn("| static_bias_theta | StatCalib Comparator | 0.100000", text)
        self.assertIn("| not_applicable | generated |", text)


if __name__ == "__main__":
    unittest.main()
