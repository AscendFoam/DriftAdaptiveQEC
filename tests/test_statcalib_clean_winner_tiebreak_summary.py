from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from cnn_fpga.benchmark.summarize_statcalib_clean_winner_tiebreak import (
    EXPECTED_ALL_MODES,
    EXPECTED_CANDIDATE_MODES,
    EXPECTED_SCENARIOS,
    summarize_run,
)


class StatcalibCleanWinnerTiebreakSummaryTest(unittest.TestCase):
    def _build_fake_t68_summary(self, directory: Path) -> Path:
        path = directory / "t68_summary.json"
        payload = {
            "mean_best_candidates": {
                "modes": [
                    "statcalib_window_variance_t001",
                    "statcalib_window_variance_t003",
                    "statcalib_window_variance_t005",
                ]
            },
            "full_generated_only_winner_modes": [
                "statcalib_window_variance_t001",
                "statcalib_window_variance_t003",
                "statcalib_window_variance_t005",
                "statcalib_ekf_t001",
            ],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def _build_fake_run_dir(
        self,
        *,
        mode_base_ler_overrides: dict[str, float] | None = None,
        scenario_mode_ler_overrides: dict[tuple[str, str], float] | None = None,
        mixed_pairs: set[tuple[str, str]] | None = None,
        missing_runs: list[str] | None = None,
    ) -> tuple[Path, Path]:
        tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(tempdir.cleanup)
        base_dir = Path(tempdir.name)
        run_dir = base_dir / "run"
        run_dir.mkdir(parents=True, exist_ok=True)
        t68_summary_path = self._build_fake_t68_summary(base_dir)

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
            "statcalib_window_variance_t003": 0.45,
            "statcalib_window_variance_t005": 0.45,
            "statcalib_ekf_t001": 0.455,
        }
        if mode_base_ler_overrides:
            base_lers.update(mode_base_ler_overrides)
        scenario_mode_ler_overrides = scenario_mode_ler_overrides or {}
        mixed_pairs = mixed_pairs or set()

        rows = []
        for scenario_idx, scenario in enumerate(EXPECTED_SCENARIOS):
            offset = scenario_idx * 0.01
            for mode in EXPECTED_ALL_MODES:
                is_statcalib = mode in EXPECTED_CANDIDATE_MODES
                final_ler = scenario_mode_ler_overrides.get((scenario, mode), base_lers[mode] + offset)
                if is_statcalib and (scenario, mode) in mixed_pairs:
                    status = "mixed"
                    reason = "mixed"
                    generated_repeats = "3"
                    generated_windows = "899.5"
                elif is_statcalib:
                    status = "generated"
                    reason = "statcalib_params_emitted"
                    generated_repeats = "4"
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
                        "completed_repeats": "4",
                        "expected_repeats": "4",
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
                "repeats": 4,
                "paired_seeds": True,
            },
        }
        launch_plan_payload = {
            "config": "temp.yaml",
            "run_dir": str(run_dir),
            "requested_scenarios": EXPECTED_SCENARIOS,
            "requested_modes": EXPECTED_ALL_MODES,
            "repeats": 4,
            "paired_seeds": True,
        }
        summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
        launch_plan_path.write_text(json.dumps(launch_plan_payload, indent=2), encoding="utf-8")
        return run_dir, t68_summary_path

    def test_persistent_t68_tie_set_is_explicit(self) -> None:
        run_dir, t68_summary_path = self._build_fake_run_dir()

        result = summarize_run(run_dir, t68_summary_path=t68_summary_path)
        summary_pack = result["summary_pack"]

        self.assertEqual(summary_pack["final_clean_winner_classification"], "persistent_clean_tie_set")
        self.assertEqual(summary_pack["t68_clean_tie_set_relation"], "persists")
        self.assertEqual(
            summary_pack["current_clean_answer_set"]["modes"],
            [
                "statcalib_window_variance_t001",
                "statcalib_window_variance_t003",
                "statcalib_window_variance_t005",
            ],
        )
        self.assertFalse(summary_pack["unique_clean_reference_point_exists"])
        self.assertEqual(len(result["pairwise_head_to_head_rows"]), 6)

    def test_unique_clean_reference_point_is_detected(self) -> None:
        run_dir, t68_summary_path = self._build_fake_run_dir(
            mode_base_ler_overrides={
                "statcalib_window_variance_t003": 0.44,
                "statcalib_window_variance_t001": 0.45,
                "statcalib_window_variance_t005": 0.451,
                "statcalib_ekf_t001": 0.456,
            }
        )

        result = summarize_run(run_dir, t68_summary_path=t68_summary_path)
        summary_pack = result["summary_pack"]

        self.assertEqual(summary_pack["final_clean_winner_classification"], "unique_clean_reference_point")
        self.assertEqual(summary_pack["t68_clean_tie_set_relation"], "collapses_to_unique")
        self.assertEqual(summary_pack["current_clean_answer_set"]["modes"], ["statcalib_window_variance_t003"])
        self.assertTrue(summary_pack["unique_clean_reference_point_exists"])
        self.assertEqual(summary_pack["unique_clean_reference_point_mode"], "statcalib_window_variance_t003")

    def test_reduced_clean_tie_set_is_detected(self) -> None:
        run_dir, t68_summary_path = self._build_fake_run_dir(
            mode_base_ler_overrides={
                "statcalib_window_variance_t001": 0.44,
                "statcalib_window_variance_t003": 0.44,
                "statcalib_window_variance_t005": 0.451,
                "statcalib_ekf_t001": 0.456,
            }
        )

        result = summarize_run(run_dir, t68_summary_path=t68_summary_path)
        summary_pack = result["summary_pack"]

        self.assertEqual(summary_pack["final_clean_winner_classification"], "reduced_clean_tie_set")
        self.assertEqual(summary_pack["t68_clean_tie_set_relation"], "reduces")
        self.assertEqual(
            summary_pack["current_clean_answer_set"]["modes"],
            ["statcalib_window_variance_t001", "statcalib_window_variance_t003"],
        )
        self.assertEqual(summary_pack["mean_best_and_worst_case_best_relation"], "same")

    def test_mixed_candidate_is_not_full_generated_only(self) -> None:
        run_dir, t68_summary_path = self._build_fake_run_dir(
            mixed_pairs={("periodic_drift", "statcalib_ekf_t001")}
        )

        result = summarize_run(run_dir, t68_summary_path=t68_summary_path)
        summary_pack = result["summary_pack"]
        candidate_rows = {row["mode"]: row for row in summary_pack["candidate_summaries"]}

        self.assertFalse(candidate_rows["statcalib_ekf_t001"]["full_generated_only"])
        self.assertEqual(candidate_rows["statcalib_ekf_t001"]["mixed_row_count"], 1)
        self.assertTrue(candidate_rows["statcalib_window_variance_t001"]["full_generated_only"])

    def test_summarize_run_rejects_incomplete_matrix(self) -> None:
        run_dir, t68_summary_path = self._build_fake_run_dir()
        comparison_path = run_dir / "comparison.csv"
        rows = comparison_path.read_text(encoding="utf-8").splitlines()
        comparison_path.write_text("\n".join(rows[:-1]) + "\n", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "missing comparison row"):
            summarize_run(run_dir, t68_summary_path=t68_summary_path)


if __name__ == "__main__":
    unittest.main()
