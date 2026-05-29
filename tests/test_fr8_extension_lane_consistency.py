from __future__ import annotations

import unittest
from pathlib import Path

from cnn_fpga.benchmark.audit_fr8_extension_lane_consistency import (
    _check_report_execution_shape_wording,
    _check_report_provenance_wording,
    _duplicate_running_keys,
    _format_last_write_time,
    _repo_relative,
    run_audit,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
TASK_PACKAGE = REPO_ROOT / "docs" / "tasks" / "Phase2" / "T64_fr8_statcalib_extension_lane_bounded_benchmark.md"
REPORT = REPO_ROOT / "docs" / "fr8_statcalib_extension_lane_benchmark.md"
RUN_DIR = REPO_ROOT / "runs" / "p4_benchmark" / "T64_fr8_statcalib_extension_lane_20260527_221658"
T24_RUN_DIR = REPO_ROOT / "runs" / "p4_benchmark" / "T24_formal_software_revalidation_20260510_200743"


class FR8ExtensionLaneConsistencyTest(unittest.TestCase):
    def test_duplicate_running_keys_detects_duplicates(self) -> None:
        duplicates = _duplicate_running_keys(
            [
                {"scenario": "static_bias_theta", "mode": "ekf", "repeat": 0, "status": "running"},
                {"scenario": "static_bias_theta", "mode": "ekf", "repeat": 0, "status": "running"},
                {"scenario": "static_bias_theta", "mode": "ekf", "repeat": 0, "status": "completed"},
            ]
        )

        self.assertEqual(duplicates, ["static_bias_theta/ekf/repeat_00"])

    def test_report_execution_shape_guard_rejects_detached_wording(self) -> None:
        issues = _check_report_execution_shape_wording("Execution note: one detached one-shot invocation only.")

        self.assertTrue(issues)
        self.assertIn("one full-matrix invocation under one fixed t64 run root", issues[0])

    def test_report_provenance_guard_rejects_false_summary_timestamp_wording(self) -> None:
        report_text = """
        Artifact-recorded fields
        - `summary.json["git_commit"]`: `1e59f24`
        - `summary.json["run_dir"]`: `runs/p4_benchmark/T64_fr8_statcalib_extension_lane_20260527_221658`
        - `launch_plan.json["config"]`: `cnn_fpga/config/p4_multiscenario_statcalib_extension_lane.yaml`
        Observed outside preserved artifacts
        - launch `HEAD`: `1e59f24`
        Auxiliary filesystem metadata
        - finish timestamp from `summary.json`: `2026-05-29 12:01:16 +08:00`
        """
        issues = _check_report_provenance_wording(
            report_text,
            run_dir_rel="runs/p4_benchmark/T64_fr8_statcalib_extension_lane_20260527_221658",
            config_rel="cnn_fpga/config/p4_multiscenario_statcalib_extension_lane.yaml",
            summary_git_commit="1e59f24",
            summary_last_write_time="2026-05-29 12:01:16 +08:00",
        )

        self.assertTrue(any("forbidden wording" in issue for issue in issues))

    def test_current_t64_artifacts_pass_full_audit(self) -> None:
        results = run_audit(
            task_package_path=TASK_PACKAGE,
            report_path=REPORT,
            run_dir=RUN_DIR,
            frozen_baseline_run_dir=T24_RUN_DIR,
        )

        failures = [result for result in results if not result.passed]
        self.assertFalse(failures, msg="\n".join(f"{item.check_id}: {item.detail}" for item in failures))

    def test_current_report_contains_expected_artifact_derived_provenance(self) -> None:
        summary_last_write_time = _format_last_write_time(RUN_DIR / "summary.json")
        report_text = REPORT.read_text(encoding="utf-8")
        issues = _check_report_provenance_wording(
            report_text,
            run_dir_rel=_repo_relative(RUN_DIR),
            config_rel="cnn_fpga/config/p4_multiscenario_statcalib_extension_lane.yaml",
            summary_git_commit="1e59f24",
            summary_last_write_time=summary_last_write_time,
        )

        self.assertFalse(issues, msg="\n".join(issues))


if __name__ == "__main__":
    unittest.main()
