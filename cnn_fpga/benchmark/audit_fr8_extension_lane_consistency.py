"""Audit FR8 extension-lane report consistency against preserved T64/T24 artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]

EXPECTED_SCENARIOS = [
    "static_bias_theta",
    "linear_ramp",
    "step_sigma_theta",
    "periodic_drift",
]
EXPECTED_FROZEN_MODES = [
    "ekf",
    "ukf",
    "constant_residual_mu",
    "rls_residual_b",
    "hybrid_residual_b",
]
EXPECTED_ALL_MODES = EXPECTED_FROZEN_MODES + ["statcalib"]


@dataclass(frozen=True)
class AuditCheckResult:
    """One audit check outcome."""

    check_id: str
    passed: bool
    detail: str


def _arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-package", required=True, help="Path to the T64 task package markdown.")
    parser.add_argument("--report", required=True, help="Path to the T64 report markdown.")
    parser.add_argument("--run-dir", required=True, help="Path to the preserved T64 run root.")
    parser.add_argument(
        "--frozen-baseline-run-dir",
        required=True,
        help="Path to the preserved T24 frozen-baseline run root.",
    )
    return parser


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            rows.append(json.loads(text))
    return rows


def _repo_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _format_last_write_time(path: Path) -> str:
    dt = datetime.fromtimestamp(path.stat().st_mtime).astimezone()
    offset = dt.strftime("%z")
    if offset:
        offset = f"{offset[:3]}:{offset[3:]}"
    return dt.strftime("%Y-%m-%d %H:%M:%S ") + offset


def _normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def _duplicate_running_keys(progress_rows: Iterable[Mapping[str, Any]]) -> list[str]:
    counts: dict[tuple[str, str, int], int] = {}
    for row in progress_rows:
        if row.get("status") != "running":
            continue
        key = (str(row.get("scenario")), str(row.get("mode")), int(row.get("repeat")))
        counts[key] = counts.get(key, 0) + 1
    duplicates = [f"{scenario}/{mode}/repeat_{repeat:02d}" for (scenario, mode, repeat), count in counts.items() if count > 1]
    return sorted(duplicates)


def _group_mode_order(rows: Sequence[Mapping[str, str]]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for row in rows:
        scenario = str(row["scenario"])
        grouped.setdefault(scenario, []).append(str(row["mode"]))
    return grouped


def _index_rows(rows: Sequence[Mapping[str, str]], *, allowed_modes: set[str] | None = None) -> dict[tuple[str, str], Mapping[str, str]]:
    indexed: dict[tuple[str, str], Mapping[str, str]] = {}
    for row in rows:
        mode = str(row["mode"])
        if allowed_modes is not None and mode not in allowed_modes:
            continue
        indexed[(str(row["scenario"]), mode)] = row
    return indexed


def _check_task_package_execution_shapes(task_text: str) -> list[str]:
    issues: list[str] = []
    if "one foreground invocation across the full matrix" not in task_text:
        issues.append("task package is missing the accepted one-foreground-invocation wording")
    if "repeat-range chunking only" not in task_text:
        issues.append("task package is missing the accepted repeat-range chunking wording")
    return issues


def _check_report_provenance_wording(
    report_text: str,
    *,
    run_dir_rel: str,
    config_rel: str,
    summary_git_commit: str,
    summary_last_write_time: str,
) -> list[str]:
    issues: list[str] = []
    normalized = _normalize_text(report_text)
    required_phrases = [
        f'`summary.json["git_commit"]`: `{summary_git_commit}`'.lower(),
        f'`summary.json["run_dir"]`: `{run_dir_rel}`'.lower(),
        f'`launch_plan.json["config"]`: `{config_rel}`'.lower(),
        f'`summary.json` lastwritetime: `{summary_last_write_time}`'.lower(),
        "artifact-recorded fields",
        "observed outside preserved artifacts",
        "auxiliary filesystem metadata",
    ]
    for phrase in required_phrases:
        if _normalize_text(phrase) not in normalized:
            issues.append(f"report is missing provenance phrase: {phrase}")

    forbidden_phrases = [
        "finish timestamp from `summary.json`",
        "finish timestamp from summary.json",
        "one detached one-shot invocation only",
    ]
    for phrase in forbidden_phrases:
        if _normalize_text(phrase) in normalized:
            issues.append(f"report still contains forbidden wording: {phrase}")
    return issues


def _check_report_execution_shape_wording(report_text: str) -> list[str]:
    issues: list[str] = []
    normalized = _normalize_text(report_text)
    required_phrases = [
        "one full-matrix invocation under one fixed t64 run root",
        "not repeat-range chunked",
        "not resumed into the same run root",
        "does not infer whether the invocation was foreground or detached",
    ]
    for phrase in required_phrases:
        if phrase not in normalized:
            issues.append(f"report is missing execution-shape phrase: {phrase}")
    return issues


def _check_required_boundary_statements(report_text: str) -> list[str]:
    issues: list[str] = []
    normalized = _normalize_text(report_text)
    required_phrases = [
        "mock-backed software-hil only",
        "separate extension lane",
        "not a rewrite of `t24`",
        "not `.tflite`",
        "not real-board",
    ]
    for phrase in required_phrases:
        if phrase not in normalized:
            issues.append(f"report is missing boundary phrase: {phrase}")
    return issues


def _compare_frozen_subset(
    t64_rows: Sequence[Mapping[str, str]],
    t24_rows: Sequence[Mapping[str, str]],
) -> list[str]:
    issues: list[str] = []
    t64_index = _index_rows(t64_rows, allowed_modes=set(EXPECTED_FROZEN_MODES))
    t24_index = _index_rows(t24_rows, allowed_modes=set(EXPECTED_FROZEN_MODES))

    if len(t64_index) != 20:
        issues.append(f"T64 frozen subset row count is {len(t64_index)}, expected 20")
    if len(t24_index) != 20:
        issues.append(f"T24 frozen subset row count is {len(t24_index)}, expected 20")

    for scenario in EXPECTED_SCENARIOS:
        for mode in EXPECTED_FROZEN_MODES:
            key = (scenario, mode)
            if key not in t64_index:
                issues.append(f"T64 frozen subset is missing row {scenario}/{mode}")
                continue
            if key not in t24_index:
                issues.append(f"T24 frozen subset is missing row {scenario}/{mode}")
                continue
            t64_row = t64_index[key]
            t24_row = t24_index[key]
            for field in ("final_ler_mean", "overflow_rate_mean"):
                if float(t64_row[field]) != float(t24_row[field]):
                    issues.append(
                        f"frozen subset mismatch on {scenario}/{mode}/{field}: "
                        f"T64={t64_row[field]} vs T24={t24_row[field]}"
                    )
    return issues


def run_audit(
    *,
    task_package_path: Path,
    report_path: Path,
    run_dir: Path,
    frozen_baseline_run_dir: Path,
) -> list[AuditCheckResult]:
    summary_path = run_dir / "summary.json"
    launch_plan_path = run_dir / "launch_plan.json"
    progress_path = run_dir / "progress.jsonl"
    comparison_path = run_dir / "comparison.csv"
    frozen_comparison_path = frozen_baseline_run_dir / "comparison.csv"

    task_text = task_package_path.read_text(encoding="utf-8")
    report_text = report_path.read_text(encoding="utf-8")
    summary = _read_json(summary_path)
    launch_plan = _read_json(launch_plan_path)
    progress_rows = _read_jsonl(progress_path)
    t64_rows = _read_csv_rows(comparison_path)
    t24_rows = _read_csv_rows(frozen_comparison_path)

    run_dir_rel = _repo_relative(run_dir)
    config_rel = _repo_relative(Path(str(launch_plan["config"])))
    summary_last_write_time = _format_last_write_time(summary_path)

    results: list[AuditCheckResult] = []

    task_shape_issues = _check_task_package_execution_shapes(task_text)
    results.append(
        AuditCheckResult(
            check_id="task_package_execution_shapes_present",
            passed=not task_shape_issues,
            detail="task package still contains the accepted execution-shape wording"
            if not task_shape_issues
            else "; ".join(task_shape_issues),
        )
    )

    provenance_issues = _check_report_provenance_wording(
        report_text,
        run_dir_rel=run_dir_rel,
        config_rel=config_rel,
        summary_git_commit=str(summary["git_commit"]),
        summary_last_write_time=summary_last_write_time,
    )
    results.append(
        AuditCheckResult(
            check_id="report_provenance_wording",
            passed=not provenance_issues,
            detail="report provenance wording matches artifact-recorded fields and auxiliary metadata classification"
            if not provenance_issues
            else "; ".join(provenance_issues),
        )
    )

    execution_shape_issues = _check_report_execution_shape_wording(report_text)
    results.append(
        AuditCheckResult(
            check_id="report_execution_shape_wording",
            passed=not execution_shape_issues,
            detail="report execution-shape wording matches the artifact-visible full-matrix pattern"
            if not execution_shape_issues
            else "; ".join(execution_shape_issues),
        )
    )

    boundary_issues: list[str] = []
    if launch_plan["requested_scenarios"] != EXPECTED_SCENARIOS:
        boundary_issues.append(f"requested_scenarios={launch_plan['requested_scenarios']} does not match expected {EXPECTED_SCENARIOS}")
    if launch_plan["requested_modes"] != EXPECTED_ALL_MODES:
        boundary_issues.append(f"requested_modes={launch_plan['requested_modes']} does not match expected {EXPECTED_ALL_MODES}")
    if summary["protocol"]["frozen_baseline_set"] != EXPECTED_FROZEN_MODES:
        boundary_issues.append(
            f"summary.protocol.frozen_baseline_set={summary['protocol']['frozen_baseline_set']} does not match expected {EXPECTED_FROZEN_MODES}"
        )
    grouped_orders = _group_mode_order(t64_rows)
    if list(grouped_orders) != EXPECTED_SCENARIOS:
        boundary_issues.append(f"comparison.csv scenario order={list(grouped_orders)} does not match expected {EXPECTED_SCENARIOS}")
    for scenario in EXPECTED_SCENARIOS:
        actual_order = grouped_orders.get(scenario)
        if actual_order != EXPECTED_ALL_MODES:
            boundary_issues.append(f"comparison.csv mode order for {scenario} is {actual_order}, expected {EXPECTED_ALL_MODES}")
    results.append(
        AuditCheckResult(
            check_id="locked_boundary_preserved",
            passed=not boundary_issues,
            detail="locked scenarios, frozen five-mode order, and statcalib extension lane are preserved"
            if not boundary_issues
            else "; ".join(boundary_issues),
        )
    )

    repeat_policy_issues: list[str] = []
    if not bool(launch_plan["paired_seeds"]):
        repeat_policy_issues.append("launch_plan.paired_seeds is not true")
    if int(launch_plan["repeats"]) != 2:
        repeat_policy_issues.append(f"launch_plan.repeats={launch_plan['repeats']} is not 2")
    if not bool(summary["protocol"]["paired_seeds"]):
        repeat_policy_issues.append("summary.protocol.paired_seeds is not true")
    if int(summary["protocol"]["repeats"]) != 2:
        repeat_policy_issues.append(f"summary.protocol.repeats={summary['protocol']['repeats']} is not 2")
    if bool(launch_plan["resume_only"]):
        repeat_policy_issues.append("launch_plan.resume_only is true, expected false")
    if int(launch_plan["repeat_start"]) != 0 or int(launch_plan["repeat_stop"]) != 2:
        repeat_policy_issues.append(
            f"launch_plan repeat range is {launch_plan['repeat_start']}..{launch_plan['repeat_stop']}, expected 0..2"
        )
    results.append(
        AuditCheckResult(
            check_id="paired_seed_and_repeat_policy",
            passed=not repeat_policy_issues,
            detail="paired-seed and repeat policy are preserved as paired_seeds=true and repeats=2"
            if not repeat_policy_issues
            else "; ".join(repeat_policy_issues),
        )
    )

    duplicate_running = _duplicate_running_keys(progress_rows)
    results.append(
        AuditCheckResult(
            check_id="progress_log_duplicate_running_guard",
            passed=not duplicate_running,
            detail="progress.jsonl has no duplicate running record for the same (scenario, mode, repeat) key"
            if not duplicate_running
            else f"duplicate running keys detected: {duplicate_running}",
        )
    )

    frozen_subset_issues = _compare_frozen_subset(t64_rows, t24_rows)
    results.append(
        AuditCheckResult(
            check_id="frozen_subset_matches_t24",
            passed=not frozen_subset_issues,
            detail="T64 frozen five-mode subset matches T24 on all 20 frozen rows for final_ler_mean and overflow_rate_mean"
            if not frozen_subset_issues
            else "; ".join(frozen_subset_issues),
        )
    )

    boundary_phrase_issues = _check_required_boundary_statements(report_text)
    results.append(
        AuditCheckResult(
            check_id="required_boundary_statements_present",
            passed=not boundary_phrase_issues,
            detail="report retains the required mock-backed extension-lane boundary statements"
            if not boundary_phrase_issues
            else "; ".join(boundary_phrase_issues),
        )
    )

    return results


def _render_results(results: Sequence[AuditCheckResult]) -> str:
    lines = ["FR8 extension-lane consistency audit"]
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        lines.append(f"- [{status}] {result.check_id}: {result.detail}")
    passed = sum(1 for result in results if result.passed)
    lines.append(f"Summary: {passed}/{len(results)} checks passed")
    return "\n".join(lines)


def main() -> int:
    parser = _arg_parser()
    args = parser.parse_args()

    results = run_audit(
        task_package_path=Path(args.task_package),
        report_path=Path(args.report),
        run_dir=Path(args.run_dir),
        frozen_baseline_run_dir=Path(args.frozen_baseline_run_dir),
    )
    print(_render_results(results))
    return 0 if all(result.passed for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
