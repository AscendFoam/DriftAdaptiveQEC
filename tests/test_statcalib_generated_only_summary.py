from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from cnn_fpga.benchmark.summarize_statcalib_generated_only import (
    EXPECTED_ALL_MODES,
    EXPECTED_SCENARIOS,
    summarize_run,
)


class StatcalibGeneratedOnlySummaryTest(unittest.TestCase):
    def _build_fake_run_dir(
        self,
        *,
        mode_base_ler_overrides: dict[str, float] | None = None,
        scenario_mode_ler_overrides: dict[tuple[str, str], float] | None = None,
        mixed_pairs: set[tuple[str, str]] | None = None,
        missing_runs: list[str] | None = None,
    ) -> Path:
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
            "statcalib_window_variance_t001": 0.45,
            "statcalib_window_variance_t003": 0.44,
            "statcalib_window_variance_t005": 0.44,
            "statcalib_window_variance_t010": 0.46,
            "statcalib_ekf_t001": 0.47,
            "statcalib_ekf_t003": 0.465,
            "statcalib_ekf_t005": 0.455,
            "statcalib_ekf_t010": 0.475,
        }
        if mode_base_ler_overrides:
            base_lers.update(mode_base_ler_overrides)
        scenario_mode_ler_overrides = scenario_mode_ler_overrides or {}
        mixed_pairs = mixed_pairs or set()

        rows = []
        for scenario_idx, scenario in enumerate(EXPECTED_SCENARIOS):
            offset = scenario_idx * 0.01
            for mode in EXPECTED_ALL_MODES:
                is_statcalib = mode.startswith("statcalib_")
                final_ler = scenario_mode_ler_overrides.get((scenario, mode), base_lers[mode] + offset)
                if is_statcalib and (scenario, mode) in mixed_pairs:
                    status = "mixed"
                    reason = "mixed"
                    generated_repeats = "1"
                    generated_windows = "899.5"
                elif is_statcalib:
                    status = "generated"
                    reason = "statcalib_params_emitted"
                    generated_repeats = "2"
                    generated_windows = "900.0"
                else:
                    status = "not_applicable"
                    reason = "mode_does_not_emit_statcalib"
                    generated_repeats = "0"
                    generated_windows = "0.0"
                rows.append(
                    {
                        "scenario": scenario,
                        "scenario_label": scenario.replace("_", " ").title(),
                        "mode": mode,
                        "mode_label": mode.replace("_", " ").title(),
                        "completed_repeats": "2",
                        "expected_repeats": "2",
                        "coverage": "1.0",
                        "final_ler_mean": str(final_ler),
                        "final_ler_std": "0.001",
                        "overflow_rate_mean": "0.002",
                        "overflow_rate_std": "0.0001",
                        "statcalib_status": status,
                        "statcalib_reason": reason,
                        "statcalib_generated_repeats": generated_repeats,
                        "statcalib_generated_windows_mean": generated_windows if is_statcalib else "0.0",
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
            "missing_runs": missing_runs or [],
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
        self.assertEqual(len(result["candidate_summary_rows"]), 8)
        self.assertEqual(len(result["threshold_comparison_rows"]), 4)
        self.assertEqual(len(result["teacher_anchor_summary_rows"]), 2)
        self.assertEqual(len(result["pareto_summary_rows"]), 8)
        self.assertTrue(Path(result["summary_json_path"]).exists())

    def test_tie_and_full_generated_only_winner_are_explicit(self) -> None:
        run_dir = self._build_fake_run_dir()

        result = summarize_run(run_dir)
        summary_pack = result["summary_pack"]

        self.assertTrue(summary_pack["any_full_generated_only_winner"])
        self.assertIn("statcalib_window_variance_t003", summary_pack["full_generated_only_winner_modes"])
        self.assertIn("statcalib_window_variance_t005", summary_pack["full_generated_only_winner_modes"])
        self.assertEqual(
            summary_pack["mean_best_candidates"]["ranking_with_ties"],
            "statcalib_window_variance_t003 = statcalib_window_variance_t005",
        )
        self.assertEqual(summary_pack["mean_best_and_worst_case_best_relation"], "same")

    def test_teacher_anchor_summary_reports_non_monotonic_threshold_order(self) -> None:
        run_dir = self._build_fake_run_dir()

        result = summarize_run(run_dir)
        teacher_rows = {
            row["teacher_anchor"]: row
            for row in result["teacher_anchor_summary_rows"]
        }
        window_row = teacher_rows["window_variance"]

        self.assertEqual(window_row["monotonicity"], "non_monotonic")
        self.assertTrue(window_row["threshold_ranking_with_ties"].startswith("t003 = t005"))
        self.assertIn("non_monotonic_by_threshold_order:", window_row["monotonicity_note"])

    def test_near_miss_is_reported_when_no_full_winner_exists(self) -> None:
        run_dir = self._build_fake_run_dir(
            mode_base_ler_overrides={"statcalib_window_variance_t005": 0.441},
            mixed_pairs={
                ("static_bias_theta", "statcalib_window_variance_t001"),
                ("static_bias_theta", "statcalib_window_variance_t003"),
                ("static_bias_theta", "statcalib_window_variance_t005"),
                ("static_bias_theta", "statcalib_window_variance_t010"),
            },
            scenario_mode_ler_overrides={
                ("periodic_drift", "statcalib_ekf_t001"): 0.835,
                ("periodic_drift", "statcalib_ekf_t003"): 0.835,
                ("periodic_drift", "statcalib_ekf_t005"): 0.835,
                ("periodic_drift", "statcalib_ekf_t010"): 0.835,
            },
        )

        result = summarize_run(run_dir)
        summary_pack = result["summary_pack"]

        self.assertFalse(summary_pack["any_full_generated_only_winner"])
        self.assertEqual(summary_pack["closest_near_miss_candidate"]["mode"], "statcalib_window_variance_t003")
        self.assertEqual(summary_pack["closest_near_miss_candidate"]["generated_gap"], 1)
        self.assertEqual(summary_pack["closest_near_miss_candidate"]["anchor_gap"], 0)

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

    def test_summarize_run_rejects_missing_runs(self) -> None:
        run_dir = self._build_fake_run_dir(missing_runs=["stub"])

        with self.assertRaisesRegex(ValueError, "missing runs"):
            summarize_run(run_dir)


if __name__ == "__main__":
    unittest.main()
