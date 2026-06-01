from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from cnn_fpga.benchmark.summarize_statcalib_sensitivity import (
    EXPECTED_ALL_MODES,
    EXPECTED_SCENARIOS,
    summarize_run,
)


class StatcalibSensitivitySummaryTest(unittest.TestCase):
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
            "statcalib_default": 0.44,
            "statcalib_low_scale": 0.49,
            "statcalib_high_scale": 0.46,
            "statcalib_low_clip": 0.47,
            "statcalib_high_threshold": 0.52,
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
                        "statcalib_generated_windows_mean": "900.0" if is_statcalib else "0.0",
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

    def test_summarize_run_builds_expected_rankings(self) -> None:
        run_dir = self._build_fake_run_dir()

        result = summarize_run(run_dir)

        self.assertEqual(result["summary_pack"]["best_variant_overall"]["mode"], "statcalib_default")
        self.assertEqual(len(result["scenario_summary_rows"]), 4)
        self.assertEqual(len(result["mode_summary_rows"]), 28)
        self.assertTrue(result["summary_pack"]["best_variant_beats_ukf_all_scenarios"])
        self.assertTrue(result["summary_pack"]["best_variant_beats_hybrid_all_scenarios"])
        self.assertTrue(Path(result["summary_json_path"]).exists())

    def test_mode_summary_contains_statcalib_rank_columns(self) -> None:
        run_dir = self._build_fake_run_dir()

        result = summarize_run(run_dir)
        mode_rows = {
            (row["scenario"], row["mode"]): row
            for row in result["mode_summary_rows"]
        }
        row = mode_rows[("static_bias_theta", "statcalib_default")]

        self.assertEqual(row["statcalib_rank_within_scenario"], 1)
        self.assertGreater(row["ukf_ler_minus_mode_ler"], 0.0)
        self.assertEqual(row["statcalib_status"], "generated")
        self.assertAlmostEqual(row["statcalib_generated_windows_mean"], 900.0)

    def test_best_variant_all_scenario_flags_use_same_global_variant(self) -> None:
        run_dir = self._build_fake_run_dir()
        comparison_path = run_dir / "comparison.csv"
        with comparison_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            if row["scenario"] == "periodic_drift" and row["mode"] == "statcalib_default":
                row["final_ler_mean"] = "0.95"
        with comparison_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

        result = summarize_run(run_dir)

        self.assertEqual(result["summary_pack"]["best_variant_overall"]["mode"], "statcalib_high_scale")
        self.assertTrue(result["summary_pack"]["best_variant_beats_ukf_all_scenarios"])
        self.assertTrue(result["summary_pack"]["best_variant_beats_hybrid_all_scenarios"])
        self.assertEqual(len(result["summary_pack"]["best_variant_overall_per_scenario"]), 4)

    def test_summarize_run_rejects_incomplete_matrix(self) -> None:
        run_dir = self._build_fake_run_dir()
        comparison_path = run_dir / "comparison.csv"
        rows = comparison_path.read_text(encoding="utf-8").splitlines()
        comparison_path.write_text("\n".join(rows[:-1]) + "\n", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "missing comparison row"):
            summarize_run(run_dir)

    def test_summarize_run_rejects_wrong_mode_order(self) -> None:
        run_dir = self._build_fake_run_dir()
        launch_plan_path = run_dir / "launch_plan.json"
        payload = json.loads(launch_plan_path.read_text(encoding="utf-8"))
        payload["requested_modes"] = payload["requested_modes"][:-1]
        launch_plan_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "requested_modes"):
            summarize_run(run_dir)


if __name__ == "__main__":
    unittest.main()
