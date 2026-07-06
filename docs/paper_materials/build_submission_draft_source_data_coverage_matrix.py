from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "docs" / "paper_materials"
CSV_PATH = OUT_DIR / "submission_draft_source_data_coverage_matrix.csv"
JSON_PATH = OUT_DIR / "submission_draft_source_data_coverage_matrix.json"
REPORT_PATH = OUT_DIR / "投稿稿source_data覆盖矩阵记录.md"


ROWS = [
    {
        "coverage_group": "main_performance_tables",
        "manuscript_items": "main results, paired deltas, paired uncertainty, LER advantage margin",
        "representative_labels": "tab:main-results; tab:paired-deltas; tab:paired-uncertainty; tab:ler-advantage-margin",
        "coverage_status": "mechanically_checked_against_source_csv",
        "source_files": "source_data_fig02_main_results.csv; source_data_fig02_paired_deltas.csv; submission_draft_paired_uncertainty_analysis.csv; submission_draft_ler_advantage_margin_analysis.csv",
        "audit_helper": "audit_submission_draft_source_data.py",
        "claim_boundary": "not CI/p-value/significance; not expanded benchmark",
    },
    {
        "coverage_group": "controlled_diagnostics",
        "manuscript_items": "nearest-syndrome/oracle affine checks, logical surrogate, lattice channel sanity, finite-energy toy-channel sanity, GKP boundary, sequence baselines, holdout stress, affine local-validity diagnostic, commit-lag sweep",
        "representative_labels": "tab:controlled-oracle-affine; tab:logical-channel-surrogate; tab:logical-channel-fidelity-surrogate; tab:lattice-logical-channel-sanity; tab:finite-energy-channel-sanity; tab:gkp-boundary-sensitivity; tab:sequence-controlled-baselines; tab:holdout-drift-stress; tab:affine-local-validity; tab:commit-lag-sweep",
        "coverage_status": "mechanically_checked_against_generated_csv",
        "source_files": "submission_draft_controlled_oracle_affine_analysis.csv; submission_draft_logical_channel_surrogate_analysis.csv; submission_draft_lattice_logical_channel_sanity.csv; submission_draft_finite_energy_channel_sanity.csv; submission_draft_gkp_boundary_sensitivity.csv; submission_draft_sequence_controlled_baseline_analysis.csv; submission_draft_holdout_drift_stress_analysis.csv; submission_draft_affine_local_validity_diagnostic.csv; submission_draft_commit_lag_sweep_analysis.csv",
        "audit_helper": "audit_submission_draft_source_data.py",
        "claim_boundary": "not tuned nearest-lattice decoder; nearest-syndrome rows are sanity references only; not trained-branch holdout proof; not calibrated finite-energy logical-channel tomography; not hardware latency",
    },
    {
        "coverage_group": "implementation_feasibility_tables",
        "manuscript_items": "fast-path cost, fixed-point parity, runtime discipline, validation-contract figure source",
        "representative_labels": "tab:fast-path-cost-model; tab:fixed-point-parity; tab:runtime-discipline; fig05_validation_contract",
        "coverage_status": "mechanically_checked_against_generated_csv",
        "source_files": "submission_draft_fast_path_cost_model.csv; submission_draft_fixed_point_parity_analysis.csv; submission_draft_runtime_discipline_summary.csv; source_data_fig05_validation_contract.csv",
        "audit_helper": "audit_submission_draft_source_data.py",
        "claim_boundary": "not FPGA synthesis/timing/resource/power; not source-vs-board agreement",
    },
    {
        "coverage_group": "source_and_literature_maps",
        "manuscript_items": "metric readiness, literature metric crosswalk, closest-work positioning, source-data manifest, row provenance, figure manifest",
        "representative_labels": "tab:metric-readiness; tab:external-comparison; tab:external-runtime-comparison; tab:closest-work-positioning; source-data manifest; row provenance; figure manifest",
        "coverage_status": "mechanically_checked_for_row_hash_and_citation_consistency",
        "source_files": "submission_draft_metric_readiness_matrix.csv; submission_draft_literature_metric_crosswalk.csv; submission_draft_closest_work_positioning.csv; submission_draft_source_data_manifest.csv; submission_draft_row_provenance_manifest.csv; figure_manifest.json",
        "audit_helper": "audit_submission_draft_source_data.py",
        "claim_boundary": "not a leaderboard; not recursive historical-run hash closure; not full reproducibility proof",
    },
    {
        "coverage_group": "formal_phase_a_interval",
        "manuscript_items": "completed static-bias Phase A paired interval",
        "representative_labels": "tab:phase-a-paired-interval",
        "coverage_status": "generated_from_completed_formal_phase_a_summary",
        "source_files": "submission_draft_phase_a_repeat_summary.csv; submission_draft_phase_a_paired_interval_analysis.csv; submission_draft_phase_a_paired_interval_analysis.json",
        "audit_helper": "run_phase_a_paired_interval_analysis.py",
        "claim_boundary": "not all-scenario repeat-expanded evidence; not p-value evidence; not holdout robustness; not hardware validation",
    },
    {
        "coverage_group": "planning_and_boundary_tables",
        "manuscript_items": "benchmark expansion protocol, Phase A repeat plan/summary/upgrade gate, runner smoke pair/matrix, availability, validation scope, runtime-artifact scope, hardware plan",
        "representative_labels": "tab:availability; tab:claim-validation-scope; tab:runtime-artifacts-scope; tab:hardware-plan",
        "coverage_status": "bounded_planning_or_boundary_surface",
        "source_files": "submission_draft_benchmark_expansion_protocol.csv; submission_draft_phase_a_repeat_plan.csv; submission_draft_phase_a_repeat_summary.csv; submission_draft_phase_a_upgrade_gate.csv; submission_draft_runner_smoke_pair.csv; submission_draft_runner_smoke_matrix.csv; manuscript prose tables",
        "audit_helper": "audit_submission_draft_source_data.py checks selected planning rows only",
        "claim_boundary": "not current result evidence; not hardware validation; not submission-completeness proof",
    },
]


def validate_rows() -> None:
    groups = [row["coverage_group"] for row in ROWS]
    if len(groups) != len(set(groups)):
        raise ValueError("coverage_group values must be unique")
    for row in ROWS:
        if not row["claim_boundary"].lower().startswith("not "):
            raise ValueError(f"claim boundary must be explicit for {row['coverage_group']}")
        if not row["representative_labels"] or not row["source_files"]:
            raise ValueError(f"incomplete coverage row: {row['coverage_group']}")


def write_csv() -> None:
    fieldnames = [
        "coverage_group",
        "manuscript_items",
        "representative_labels",
        "coverage_status",
        "source_files",
        "audit_helper",
        "claim_boundary",
    ]
    with CSV_PATH.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(ROWS)


def write_json() -> None:
    payload = {
        "status": "generated",
        "analysis_id": "submission_draft_source_data_coverage_matrix_v1",
        "scope": (
            "Manuscript-facing source-data coverage matrix. It records which "
            "table families are mechanically checked and which remain planning "
            "or boundary surfaces."
        ),
        "non_claims": [
            "not a new experiment",
            "not a formal benchmark",
            "not a full all-scenario confidence-interval or p-value analysis",
            "not a full reproducibility proof",
            "not hardware validation",
        ],
        "rows": ROWS,
    }
    JSON_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_report() -> None:
    lines = [
        "# 投稿稿 source-data 覆盖矩阵记录",
        "",
        "日期：2026-07-03",
        "",
        "本记录服务于 `docs/paper_notes/CNN_FPGA_GKP_submission_draft.tex`，用于把 source-data coverage 从叙述性说明转为可审计矩阵。",
        "",
        "## 生成文件",
        "",
        f"- `{CSV_PATH.relative_to(ROOT)}`",
        f"- `{JSON_PATH.relative_to(ROOT)}`",
        "",
        "## 覆盖矩阵",
        "",
        "| Coverage group | Status | Boundary |",
        "| --- | --- | --- |",
    ]
    for row in ROWS:
        lines.append(
            f"| `{row['coverage_group']}` | {row['coverage_status']} | {row['claim_boundary']} |"
        )
    lines.extend(
        [
            "",
            "## 可写边界",
            "",
            "- 可以写：当前稿件的 selected tables/source files 有机械一致性审计和 file-level manifest。",
            "- 可以写：availability、hardware plan、runtime-artifact scope 等表格是 boundary/planning surfaces，不是结果证据。",
            "- 不能写：该矩阵完成了 full reproducibility、recursive run-directory hash closure、CI/p-value、expanded benchmark 或 hardware validation。",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    validate_rows()
    write_csv()
    write_json()
    write_report()
    print(json.dumps({"status": "ok", "rows": len(ROWS), "csv": str(CSV_PATH), "json": str(JSON_PATH)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
