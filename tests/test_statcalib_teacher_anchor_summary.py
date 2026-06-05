from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from cnn_fpga.benchmark.summarize_statcalib_teacher_anchor import (
    EXPECTED_ALL_MODES,
    EXPECTED_SCENARIOS,
    summarize_run,
)


class StatcalibTeacherAnchorSummaryTest(unittest.TestCase):
    def _build_fake_run_dir(self) -> Path:
        tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(tempdir.cleanup)
        run_dir = Path(tempdir.name) / "run"
        run_dir.mkdir(parents=True, exist_ok=True)

        comparison_path = run_dir / "comparison.csv"
        summary_path = run_dir / "summary.json"
        launch_plan_path = run_dir / "launch_plan.json"

        fields = [
            "scenario",
            "scenario_label",
            "mode",
            "mode_label",
            "completed_repeats",
            "expected_repeats",
            "coverage",
            "final_ler_mean",
            "final_ler_std",
            "overflow_rate_mean",
            "overflow_rate_std",
            "statcalib_status",
            "statcalib_reason",
            "statcalib_generated_repeats",
            "statcalib_generated_windows_mean",
            "statcalib_signal_norm_mean_mean",
        ]

        base_lers = {
            "ukf": 0.82,
            "hybrid_residual_b": 0.79,
            "statcalib_default_teacher_ukf": 0.44,
            "statcalib_default_teacher_window_variance": 0.43,
            "statcalib_default_teacher_ekf": 0.46,
            "statcalib_high_threshold_teacher_ukf": 0.47,
            "statcalib_high_threshold_teacher_window_variance": 0.45,
            "statcalib_high_threshold_teacher_ekf": 0.50,
        }

        rows = []
        for scenario_idx, scenario in enumerate(EXPECTED_SCENARIOS):
            offset = scenario_idx * 0.01
            for mode in EXPECTED_ALL_MODES:
                is_statcalib = mode.startswith("statcalib_")
                rows.append(
                    {
                        "scenario": scenario,
                        "scenario_label": scenario.replace("_", " ").title(),
                        "mode": mode,
                        "mode_label": mode.replace("_", " ").title(),
                        "completed_repeats": "2",
                        "expected_repeats": "2",
                        "coverage": "1.0",
                        "final_ler_mean": str(base_lers[mode] + offset),
                        "final_ler_std": "0.001",
                        "overflow_rate_mean": "0.002",
                        "overflow_rate_std": "0.0001",
                        "statcalib_status": "generated" if is_statcalib else "not_applicable",
                        "statcalib_reason": "statcalib_params_emitted" if is_statcalib else "mode_does_not_emit_statcalib",
                        "statcalib_generated_repeats": "2" if is_statcalib else "0",
                        "statcalib_generated_windows_mean": "900.0" if is_statcalib else "",
                        "statcalib_signal_norm_mean_mean": "0.18" if is_statcalib else "",
                    }
                )

        with comparison_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

        summary_payload = {
            "run_dir": str(run_dir),
            "git_commit": "deadbee",
            "missing_runs": [],
            "protocol": {
                "repeats": 2,
                "paired_seeds": True,
            },
        }
        launch_plan_payload = {
            "config": "temp.yaml",
            "run_dir": str(run_dir),
            "requested_scenarios": EXPECTED_SCENARIOS,
            "requested_modes": EXPECTED_ALL_MODES,
            "repeats": 2,
            "paired_seeds": True,
        }
        summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
        launch_plan_path.write_text(json.dumps(launch_plan_payload, indent=2), encoding="utf-8")
        return run_dir

    def test_summarize_run_builds_expected_grouped_outputs(self) -> None:
        run_dir = self._build_fake_run_dir()

        result = summarize_run(run_dir)

        self.assertEqual(len(result["scenario_summary_rows"]), 4)
        self.assertEqual(len(result["mode_summary_rows"]), 32)
        self.assertEqual(len(result["variant_summary_rows"]), 6)
        self.assertEqual(len(result["teacher_anchor_summary_rows"]), 3)
        self.assertEqual(len(result["parameter_point_summary_rows"]), 2)
        self.assertTrue(Path(result["summary_json_path"]).exists())

    def test_scenario_and_parameter_point_rankings_match_teacher_anchor_expectation(self) -> None:
        run_dir = self._build_fake_run_dir()

        result = summarize_run(run_dir)
        scenario_row = result["scenario_summary_rows"][0]
        parameter_rows = {
            row["parameter_point"]: row
            for row in result["parameter_point_summary_rows"]
        }

        self.assertEqual(scenario_row["best_statcalib_mode"], "statcalib_default_teacher_window_variance")
        self.assertEqual(scenario_row["best_statcalib_teacher_anchor"], "window_variance")
        self.assertEqual(parameter_rows["default"]["teacher_anchor_ranking"], "window_variance > ukf > ekf")
        self.assertTrue(parameter_rows["default"]["non_ukf_teacher_best"])

    def test_teacher_anchor_summary_compares_default_vs_high_threshold(self) -> None:
        run_dir = self._build_fake_run_dir()

        result = summarize_run(run_dir)
        teacher_rows = {
            row["teacher_anchor"]: row
            for row in result["teacher_anchor_summary_rows"]
        }
        window_row = teacher_rows["window_variance"]

        self.assertEqual(window_row["better_parameter_point_by_mean_ler"], "default")
        self.assertLess(window_row["default_mean_ler_minus_high_threshold_mean_ler"], 0.0)
        self.assertTrue(window_row["default_beats_both_frozen_anchors_all_scenarios"])
        self.assertTrue(window_row["high_threshold_beats_both_frozen_anchors_all_scenarios"])

    def test_summary_flags_qualifying_non_ukf_variants(self) -> None:
        run_dir = self._build_fake_run_dir()

        result = summarize_run(run_dir)
        summary_pack = result["summary_pack"]

        self.assertTrue(summary_pack["any_non_ukf_variant_beats_both_frozen_anchors_all_scenarios"])
        self.assertIn(
            "statcalib_default_teacher_window_variance",
            summary_pack["qualifying_non_ukf_variants_beating_both_frozen_anchors_all_scenarios"],
        )

    def test_summarize_run_rejects_incomplete_matrix(self) -> None:
        run_dir = self._build_fake_run_dir()
        comparison_path = run_dir / "comparison.csv"
        rows = comparison_path.read_text(encoding="utf-8").splitlines()
        comparison_path.write_text("\n".join(rows[:-1]) + "\n", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "missing comparison row"):
            summarize_run(run_dir)

    def test_summarize_run_rejects_wrong_mode_set(self) -> None:
        run_dir = self._build_fake_run_dir()
        launch_plan_path = run_dir / "launch_plan.json"
        payload = json.loads(launch_plan_path.read_text(encoding="utf-8"))
        payload["requested_modes"] = payload["requested_modes"][:-1]
        launch_plan_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "requested_modes"):
            summarize_run(run_dir)


if __name__ == "__main__":
    unittest.main()
