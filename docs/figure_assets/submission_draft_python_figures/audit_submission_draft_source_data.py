"""Audit source-data consistency for the submission draft figure package.

This helper is intentionally narrow. It checks the current submission draft
against existing CSV/JSON/TeX artifacts and writes a manuscript-facing audit
report. It does not run experiments, recompute benchmarks, or upgrade any
evidence boundary.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[3]
FIG_DIR = ROOT / "docs" / "figure_assets" / "submission_draft_python_figures"
PAPER_MATERIALS = ROOT / "docs" / "paper_materials"
TEX_PATH = ROOT / "docs" / "paper_notes" / "CNN_FPGA_GKP_submission_draft.tex"
T24_DIR = ROOT / "runs" / "p4_benchmark" / "T24_formal_software_revalidation_20260510_200743"
T24_COMPARISON = T24_DIR / "comparison.csv"
T24_SUMMARY = T24_DIR / "summary.json"
ORACLE_AFFINE_CSV = PAPER_MATERIALS / "submission_draft_controlled_oracle_affine_analysis.csv"
COST_MODEL_CSV = PAPER_MATERIALS / "submission_draft_fast_path_cost_model.csv"
FIXED_POINT_PARITY_CSV = PAPER_MATERIALS / "submission_draft_fixed_point_parity_analysis.csv"
LOGICAL_CHANNEL_SURROGATE_CSV = PAPER_MATERIALS / "submission_draft_logical_channel_surrogate_analysis.csv"
LATTICE_LOGICAL_CHANNEL_SANITY_CSV = PAPER_MATERIALS / "submission_draft_lattice_logical_channel_sanity.csv"
FINITE_ENERGY_CHANNEL_SANITY_CSV = PAPER_MATERIALS / "submission_draft_finite_energy_channel_sanity.csv"
HOLDOUT_DRIFT_STRESS_CSV = PAPER_MATERIALS / "submission_draft_holdout_drift_stress_analysis.csv"
AFFINE_LOCAL_VALIDITY_CSV = PAPER_MATERIALS / "submission_draft_affine_local_validity_diagnostic.csv"
COMMIT_LAG_SWEEP_CSV = PAPER_MATERIALS / "submission_draft_commit_lag_sweep_analysis.csv"
COMMIT_LAG_SWEEP_JSON = PAPER_MATERIALS / "submission_draft_commit_lag_sweep_analysis.json"
SEQUENCE_BASELINE_CSV = PAPER_MATERIALS / "submission_draft_sequence_controlled_baseline_analysis.csv"
PAIRED_UNCERTAINTY_CSV = PAPER_MATERIALS / "submission_draft_paired_uncertainty_analysis.csv"
LER_ADVANTAGE_MARGIN_CSV = PAPER_MATERIALS / "submission_draft_ler_advantage_margin_analysis.csv"
METRIC_READINESS_CSV = PAPER_MATERIALS / "submission_draft_metric_readiness_matrix.csv"
LITERATURE_CROSSWALK_CSV = PAPER_MATERIALS / "submission_draft_literature_metric_crosswalk.csv"
CLOSEST_WORK_POSITIONING_CSV = PAPER_MATERIALS / "submission_draft_closest_work_positioning.csv"
SOURCE_DATA_COVERAGE_MATRIX_CSV = PAPER_MATERIALS / "submission_draft_source_data_coverage_matrix.csv"
SOURCE_DATA_COVERAGE_MATRIX_JSON = PAPER_MATERIALS / "submission_draft_source_data_coverage_matrix.json"
BENCHMARK_EXPANSION_PROTOCOL_CSV = PAPER_MATERIALS / "submission_draft_benchmark_expansion_protocol.csv"
PHASE_A_REPEAT_PLAN_CSV = PAPER_MATERIALS / "submission_draft_phase_a_repeat_plan.csv"
PHASE_A_REPEAT_SUMMARY_CSV = PAPER_MATERIALS / "submission_draft_phase_a_repeat_summary.csv"
PHASE_A_PAIRED_INTERVAL_CSV = PAPER_MATERIALS / "submission_draft_phase_a_paired_interval_analysis.csv"
PHASE_A_UPGRADE_GATE_CSV = PAPER_MATERIALS / "submission_draft_phase_a_upgrade_gate.csv"
RUNNER_SMOKE_PAIR_CSV = PAPER_MATERIALS / "submission_draft_runner_smoke_pair.csv"
RUNNER_SMOKE_MATRIX_CSV = PAPER_MATERIALS / "submission_draft_runner_smoke_matrix.csv"
ROW_PROVENANCE_CSV = PAPER_MATERIALS / "submission_draft_row_provenance_manifest.csv"
ROW_PROVENANCE_JSON = PAPER_MATERIALS / "submission_draft_row_provenance_manifest.json"
RUNTIME_DISCIPLINE_CSV = PAPER_MATERIALS / "submission_draft_runtime_discipline_summary.csv"
GKP_BOUNDARY_SENSITIVITY_CSV = PAPER_MATERIALS / "submission_draft_gkp_boundary_sensitivity.csv"
SOURCE_MANIFEST_CSV = PAPER_MATERIALS / "submission_draft_source_data_manifest.csv"
SOURCE_MANIFEST_JSON = PAPER_MATERIALS / "submission_draft_source_data_manifest.json"
REPORT_JSON = PAPER_MATERIALS / "submission_draft_source_data_audit.json"
REPORT_MD = PAPER_MATERIALS / "投稿稿source_data机械审计报告.md"

TOL = 5e-7

MAIN_MODE_MAP = {
    "EKF": "ekf",
    "UKF": "ukf",
    "Const.-mu": "constant_residual_mu",
    "RLS-b": "rls_residual_b",
    "Hybrid-b": "hybrid_residual_b",
}

FIG02_MODE_MAP = {
    "EKF": "ekf",
    "UKF": "ukf",
    "Const.-mu": "constant_residual_mu",
    "RLS-b": "rls_residual_b",
    "Hybrid-b": "hybrid_residual_b",
}

PAIRED_SCENARIO_LABELS = {
    "static_bias_theta": "static_bias_theta",
    "linear_ramp": "linear_ramp",
    "step_sigma_theta": "step_sigma_theta",
    "periodic_drift": "periodic_drift",
}

PAIRED_UNCERTAINTY_LABELS = {
    **PAIRED_SCENARIO_LABELS,
    "all_scenarios": "all_scenarios",
}

ABLATION_LABEL_MAP = {
    "ukf": "UKF",
    "hybrid_full": "Hybrid full",
    "hybrid_no_hist_deltas": "No hist. deltas",
    "hybrid_no_teacher_prediction": "No teacher pred.",
    "hybrid_no_teacher_params": "No teacher params",
    "hybrid_no_teacher_deltas": "No teacher deltas",
}

ORACLE_AFFINE_METHODS = (
    "fixed_affine",
    "oracle_affine",
    "wrapped_gaussian_posterior_mean",
    "wrapped_gaussian_map",
)

CONTROLLED_AFFINE_METHODS = (
    "nearest_syndrome",
    *ORACLE_AFFINE_METHODS,
)

LOGICAL_CHANNEL_SURROGATE_METHODS = ORACLE_AFFINE_METHODS

LATTICE_LOGICAL_CHANNEL_METHOD_LABELS = {
    "Fixed affine": "fixed_affine",
    "Oracle affine": "oracle_affine",
    "Wrapped mean": "wrapped_gaussian_posterior_mean",
    "Wrapped MAP": "wrapped_gaussian_map",
}

FINITE_ENERGY_METHOD_LABELS = {
    "Hard nearest-syndrome": "hard_nearest_syndrome",
    "Fixed affine": "fixed_affine",
    "Oracle affine": "oracle_affine",
}

FINITE_ENERGY_DELTAS = ("0.18", "0.26", "0.34")

HOLDOUT_DRIFT_STRESS_METHODS = (
    "fixed_affine",
    "lagged_affine",
    "oracle_affine",
    "wrapped_gaussian_posterior_mean",
    "wrapped_gaussian_map",
)

AFFINE_LOCAL_VALIDITY_LAYER_LABELS = {
    "Short sequence": "short_sequence_controlled",
    "Holdout stress": "holdout_stress_controlled",
}

COMMIT_LAG_SWEEP_LAGS = ("0", "8", "16", "32", "64", "128")
COMMIT_LAG_SWEEP_SCENARIOS = (
    "random_walk_drift",
    "burst_reset_drift",
    "faster_than_window_oscillation",
)

METRIC_READINESS_LABELS = {
    "Logical-error proxy": "Logical-error proxy",
    "Logical-channel fidelity": "Logical-channel fidelity or infidelity",
    "Drift robustness": "Drift adaptation robustness",
    "Fast-path cost and latency": "Fast-path cost and latency",
    "Hardware-facing validation": "Hardware-facing validation",
}

LITERATURE_CROSSWALK_AXES = {
    "analog_gkp_information",
    "calibration_aware_qec",
    "logical_error_and_overhead_targets",
    "logical_channel_fidelity_and_infidelity",
    "learned_qec_modules",
    "real_time_fpga_decoders",
}

CLOSEST_WORK_FAMILIES = {
    "Analog and surface-GKP decoding",
    "Calibration-aware and learned QEC decoders",
    "Finite-energy logical-channel analyses",
    "Runtime pre-decoders and calibration-conditioned neural modules",
    "Real-time FPGA and hardware-tailored decoders",
}

SOURCE_DATA_COVERAGE_GROUPS = {
    "main_performance_tables": "Main performance tables",
    "controlled_diagnostics": "Controlled diagnostics",
    "implementation_feasibility_tables": "Implementation feasibility",
    "source_and_literature_maps": "Source and literature maps",
    "formal_phase_a_interval": "Formal Phase A interval",
    "planning_and_boundary_tables": "Planning and boundary surfaces",
}


@dataclass
class Check:
    name: str
    status: str
    detail: str


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def clean_cell(cell: str) -> str:
    value = cell.strip()
    value = re.sub(r"\\texttt\{([^{}]+)\}", r"\1", value)
    value = re.sub(r"\\textbf\{([^{}]+)\}", r"\1", value)
    value = value.replace("\\_", "_")
    value = value.replace("\\(", "").replace("\\)", "")
    value = value.replace("\\", "")
    value = value.replace("{", "").replace("}", "")
    value = value.replace("+", "")
    value = value.replace("%", "")
    return value.strip()


def strip_latex_comment(line: str) -> str:
    """Strip true LaTeX comments while preserving escaped percent signs."""
    return re.split(r"(?<!\\)%", line, maxsplit=1)[0]


def extract_table_rows(tex: str, label: str) -> list[list[str]]:
    label_token = f"\\label{{{label}}}"
    label_pos = tex.find(label_token)
    if label_pos < 0:
        raise ValueError(f"Could not find LaTeX table {label}")
    table_start = tex.rfind("\\begin{table}", 0, label_pos)
    midrule = tex.find("\\midrule", table_start, label_pos)
    bottomrule = tex.find("\\bottomrule", table_start, label_pos)
    if table_start < 0 or midrule < 0 or bottomrule < 0:
        raise ValueError(f"Could not isolate LaTeX table {label}")

    rows: list[list[str]] = []
    for raw_line in tex[midrule + len("\\midrule") : bottomrule].splitlines():
        line = raw_line.strip()
        if not line or "&" not in line:
            continue
        line = strip_latex_comment(line).strip()
        line = re.sub(r"\\\\\s*$", "", line)
        rows.append([clean_cell(part) for part in line.split("&")])
    return rows


def active_citation_keys(tex: str) -> set[str]:
    keys: set[str] = set()
    for match in re.finditer(r"\\cite\{([^{}]+)\}", tex):
        keys.update(key.strip() for key in match.group(1).split(",") if key.strip())
    return keys


def parse_float(value: str) -> float:
    return float(clean_cell(value))


def parse_interval_cell(value: str) -> tuple[float, float]:
    numbers = re.findall(r"-?\d+\.\d+", clean_cell(value))
    if len(numbers) != 2:
        raise ValueError(f"Could not parse interval cell: {value}")
    return float(numbers[0]), float(numbers[1])


def index_t24(rows: Iterable[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    return {(row["scenario"], row["mode"]): row for row in rows}


def close(a: float, b: float, tol: float = TOL) -> bool:
    return abs(a - b) <= tol


def displayed6(value: float) -> float:
    return float(Decimal(str(value)).quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP))


def add_check(checks: list[Check], name: str, ok: bool, detail: str) -> None:
    checks.append(Check(name=name, status="PASS" if ok else "FAIL", detail=detail))


def audit() -> dict[str, object]:
    checks: list[Check] = []
    tex = TEX_PATH.read_text(encoding="utf-8")
    t24_rows = read_csv(T24_COMPARISON)
    t24_by_key = index_t24(t24_rows)
    summary = json.loads(T24_SUMMARY.read_text(encoding="utf-8"))

    expected_scenarios = summary["filters"]["scenario"]
    expected_modes = summary["filters"]["mode"]
    protocol = summary["protocol"]

    add_check(
        checks,
        "T24 protocol shape",
        protocol.get("repeats") == 2
        and protocol.get("paired_seeds") is True
        and expected_scenarios
        == ["static_bias_theta", "linear_ramp", "step_sigma_theta", "periodic_drift"]
        and expected_modes
        == ["ekf", "ukf", "constant_residual_mu", "rls_residual_b", "hybrid_residual_b"],
        "summary.json records four scenarios, five predeclared modes, paired seeds and repeats=2.",
    )

    add_check(
        checks,
        "T24 row coverage",
        len(t24_rows) == 20
        and all(row["completed_repeats"] == row["expected_repeats"] == "2" for row in t24_rows)
        and all(float(row["coverage"]) == 1.0 for row in t24_rows),
        "comparison.csv contains 20 rows with completed_repeats=expected_repeats=2 and coverage=1.0.",
    )

    main_rows = extract_table_rows(tex, "tab:main-results")
    for row in main_rows:
        scenario = row[0]
        for label, value in zip(MAIN_MODE_MAP, row[1:]):
            mode = MAIN_MODE_MAP[label]
            actual = float(t24_by_key[(scenario, mode)]["final_ler_mean"])
            add_check(
                checks,
                f"TeX main table {scenario}/{mode}",
                close(parse_float(value), displayed6(actual)),
                f"TeX {parse_float(value):.6f} vs T24 rounded {displayed6(actual):.6f}.",
            )

    fig02_rows = read_csv(FIG_DIR / "source_data_fig02_main_results.csv")
    for row in fig02_rows:
        mode = FIG02_MODE_MAP[row["mode"]]
        source = t24_by_key[(row["scenario"], mode)]
        add_check(
            checks,
            f"Fig2 source mean {row['scenario']}/{mode}",
            close(float(row["final_ler_mean"]), float(source["final_ler_mean"]), 1e-11),
            f"source_data mean {row['final_ler_mean']} vs T24 {source['final_ler_mean']}.",
        )
        add_check(
            checks,
            f"Fig2 source sd {row['scenario']}/{mode}",
            close(float(row["final_ler_sd"]), float(source["final_ler_std"]), 1e-11),
            f"source_data SD {row['final_ler_sd']} vs T24 {source['final_ler_std']}.",
        )
        add_check(
            checks,
            f"Fig2 repeats {row['scenario']}/{mode}",
            row["n_repeats"] == source["completed_repeats"] == "2",
            "source_data n_repeats matches T24 completed_repeats.",
        )

    raw_rows = summary.get("raw_rows", [])
    raw_by_key = {
        (row["scenario"], row["mode"], int(row["repeat"])): row
        for row in raw_rows
    }
    paired_csv_rows = read_csv(FIG_DIR / "source_data_fig02_paired_deltas.csv")
    paired_by_scenario: dict[str, list[dict[str, str]]] = {}
    for row in paired_csv_rows:
        paired_by_scenario.setdefault(row["scenario"], []).append(row)
        ukf = raw_by_key[(row["scenario"], "ukf", int(row["repeat"]))]
        hybrid = raw_by_key[(row["scenario"], "hybrid_residual_b", int(row["repeat"]))]
        expected_delta = float(ukf["final_ler"]) - float(hybrid["final_ler"])
        expected_relative = expected_delta / float(ukf["final_ler"]) * 100.0
        add_check(
            checks,
            f"Fig2 paired source {row['scenario']}/repeat_{row['repeat']}",
            int(row["seed"]) == int(ukf["seed"]) == int(hybrid["seed"])
            and close(float(row["ukf_final_ler"]), float(ukf["final_ler"]), 1e-11)
            and close(float(row["hybrid_final_ler"]), float(hybrid["final_ler"]), 1e-11)
            and close(float(row["delta_final_ler_ukf_minus_hybrid"]), expected_delta, 1e-11)
            and close(float(row["relative_reduction_percent"]), expected_relative, 5e-7),
            (
                f"source paired delta {row['delta_final_ler_ukf_minus_hybrid']} "
                f"and relative {row['relative_reduction_percent']} vs raw_rows repeat {row['repeat']}."
            ),
        )

    paired_tex_rows = extract_table_rows(tex, "tab:paired-deltas")
    for row in paired_tex_rows:
        scenario = PAIRED_SCENARIO_LABELS[row[0]]
        rows_for_scenario = paired_by_scenario[scenario]
        mean_delta = sum(float(item["delta_final_ler_ukf_minus_hybrid"]) for item in rows_for_scenario) / len(rows_for_scenario)
        mean_relative = sum(float(item["relative_reduction_percent"]) for item in rows_for_scenario) / len(rows_for_scenario)
        min_delta = min(float(item["delta_final_ler_ukf_minus_hybrid"]) for item in rows_for_scenario)
        add_check(
            checks,
            f"TeX paired-delta table {scenario}",
            close(parse_float(row[1]), displayed6(mean_delta))
            and close(parse_float(row[2].replace("%", "")), round(mean_relative, 2), 5e-3)
            and close(parse_float(row[3]), displayed6(min_delta)),
            (
                f"TeX mean/min {row[1]}/{row[3]} and relative {row[2]} "
                f"vs source paired deltas."
            ),
        )

    paired_uncertainty_csv = {
        row["scenario"]: row
        for row in read_csv(PAIRED_UNCERTAINTY_CSV)
    }
    paired_uncertainty_tex_rows = extract_table_rows(tex, "tab:paired-uncertainty")
    for row in paired_uncertainty_tex_rows:
        scenario = PAIRED_UNCERTAINTY_LABELS[row[0]]
        source = paired_uncertainty_csv[scenario]
        direction = f"{source['directionally_positive_count']}/{source['n_paired_repeats']}"
        add_check(
            checks,
            f"TeX paired-uncertainty table {scenario}",
            int(parse_float(row[1])) == int(source["n_paired_repeats"])
            and close(parse_float(row[2]), displayed6(float(source["mean_delta_ukf_minus_hybrid"])))
            and close(parse_float(row[3]), displayed6(float(source["paired_bootstrap_span_low"])))
            and close(parse_float(row[4]), displayed6(float(source["paired_bootstrap_span_high"])))
            and row[5] == direction,
            (
                f"TeX n/mean/low/high/direction {row[1]}/{row[2]}/{row[3]}/{row[4]}/{row[5]} "
                f"vs paired uncertainty CSV."
            ),
        )

    ler_margin_csv = {
        row["scenario"]: row
        for row in read_csv(LER_ADVANTAGE_MARGIN_CSV)
    }
    ler_margin_tex_rows = extract_table_rows(tex, "tab:ler-advantage-margin")
    for row in ler_margin_tex_rows:
        scenario = PAIRED_SCENARIO_LABELS[row[0]]
        source = ler_margin_csv[scenario]
        add_check(
            checks,
            f"TeX LER advantage margin table {scenario}",
            close(parse_float(row[1]), displayed6(float(source["ukf_final_ler_mean"])))
            and close(parse_float(row[2]), displayed6(float(source["hybrid_final_ler_mean"])))
            and close(parse_float(row[3]), displayed6(float(source["mean_delta_ukf_minus_hybrid"])))
            and close(parse_float(row[4].replace("%", "")), round(float(source["mean_relative_reduction_percent"]), 2), 5e-3)
            and row[5] == source["paired_direction"]
            and close(parse_float(row[6]), round(float(source["delta_over_max_descriptive_sd"]), 2), 5e-3),
            (
                f"TeX UKF/Hybrid/delta/relative/direction/scale "
                f"{row[1]}/{row[2]}/{row[3]}/{row[4]}/{row[5]}/{row[6]} "
                f"vs LER advantage margin CSV."
            ),
        )

    gkp_boundary_rows = extract_table_rows(tex, "tab:gkp-boundary-sensitivity")
    gkp_boundary_csv = {
        f"{float(row['effective_residual_sigma']):.2f}": row
        for row in read_csv(GKP_BOUNDARY_SENSITIVITY_CSV)
    }
    for row in gkp_boundary_rows:
        sigma = f"{parse_float(row[0]):.2f}"
        csv_row = gkp_boundary_csv[sigma]
        add_check(
            checks,
            f"GKP boundary sensitivity table sigma={sigma}",
            close(parse_float(row[1]), round(float(csv_row["equivalent_modular_squeezing_db"]), 2), 5e-3)
            and close(parse_float(row[2]), displayed6(float(csv_row["single_quadrature_crossing_probability"])))
            and close(parse_float(row[3]), displayed6(float(csv_row["any_qp_crossing_probability"])))
            and close(parse_float(row[4]), displayed6(float(csv_row["pauli_surrogate_infidelity"]))),
            (
                f"TeX squeezing/p_cross/p_any/infidelity "
                f"{row[1]}/{row[2]}/{row[3]}/{row[4]} "
                "vs GKP boundary-sensitivity CSV."
            ),
        )

    ablation_rows = extract_table_rows(tex, "tab:ablation")
    ablation_csv = {
        row["label"]: row
        for row in read_csv(FIG_DIR / "source_data_fig03_ablation_mechanism.csv")
        if row["panel"] == "ablation"
    }
    for row in ablation_rows:
        label = ABLATION_LABEL_MAP[row[0]]
        csv_row = ablation_csv[label]
        add_check(
            checks,
            f"Ablation source {row[0]}",
            close(parse_float(row[1]), float(csv_row["value_a"]))
            and close(parse_float(row[2]), float(csv_row["value_b"])),
            f"TeX avg/delta {row[1]}/{row[2]} vs source {csv_row['value_a']}/{csv_row['value_b']}.",
        )

    mechanism_rows = extract_table_rows(tex, "tab:mechanism")
    mechanism_csv = {
        row["label"]: row
        for row in read_csv(FIG_DIR / "source_data_fig03_ablation_mechanism.csv")
        if row["panel"] == "mechanism"
    }
    for row in mechanism_rows:
        seed = row[0]
        csv_row = mechanism_csv[seed]
        add_check(
            checks,
            f"Mechanism source {seed}",
            close(parse_float(row[1]), float(csv_row["value_a"]))
            and close(parse_float(row[2]), float(csv_row["value_b"])),
            f"TeX deltas {row[1]}/{row[2]} vs source {csv_row['value_a']}/{csv_row['value_b']}.",
        )

    stat_rows = extract_table_rows(tex, "tab:statcalib")
    stat_csv = read_csv(FIG_DIR / "source_data_fig04_statcalib.csv")
    stat_by_key = {(row["scenario"], row["mode"]): row for row in stat_csv}
    for row in stat_rows:
        scenario = row[0]
        expected = {
            "UKF": parse_float(row[1]),
            "Hybrid-b": parse_float(row[2]),
            "StatCalib supp.": parse_float(row[3]),
        }
        for mode_label, tex_value in expected.items():
            csv_row = stat_by_key[(scenario, mode_label)]
            add_check(
                checks,
                f"StatCalib source {scenario}/{mode_label}",
                close(tex_value, float(csv_row["final_ler_mean"])),
                f"TeX {tex_value:.6f} vs source {csv_row['final_ler_mean']}.",
            )
        add_check(
            checks,
            f"StatCalib T24 anchor {scenario}",
            close(expected["UKF"], displayed6(float(t24_by_key[(scenario, "ukf")]["final_ler_mean"])))
            and close(
                expected["Hybrid-b"],
                displayed6(float(t24_by_key[(scenario, "hybrid_residual_b")]["final_ler_mean"])),
            ),
            "UKF and Hybrid-b extension-lane anchors match T24 rounded values.",
        )

    oracle_rows = extract_table_rows(tex, "tab:controlled-oracle-affine")
    oracle_csv = read_csv(ORACLE_AFFINE_CSV)
    oracle_by_key = {
        (row["scenario"], row["method"]): row
        for row in oracle_csv
        if row["method"] in CONTROLLED_AFFINE_METHODS
    }
    for row in oracle_rows:
        scenario = row[0]
        nearest = oracle_by_key[(scenario, "nearest_syndrome")]
        fixed = oracle_by_key[(scenario, "fixed_affine")]
        oracle = oracle_by_key[(scenario, "oracle_affine")]
        wrapped_mean = oracle_by_key[(scenario, "wrapped_gaussian_posterior_mean")]
        wrapped_map = oracle_by_key[(scenario, "wrapped_gaussian_map")]
        add_check(
            checks,
            f"Controlled oracle-affine table {scenario}",
            close(parse_float(row[1]), displayed6(float(nearest["residual_mse"])))
            and close(parse_float(row[2]), displayed6(float(fixed["residual_mse"])))
            and close(parse_float(row[3]), displayed6(float(oracle["residual_mse"])))
            and close(parse_float(row[4]), displayed6(float(wrapped_mean["residual_mse"])))
            and close(parse_float(row[5]), displayed6(float(wrapped_map["residual_mse"]))),
            (
                f"TeX nearest/fixed/oracle/wrapped-mean/wrapped-map MSE "
                f"{row[1]}/{row[2]}/{row[3]}/{row[4]}/{row[5]} vs controlled oracle-affine CSV."
            ),
        )

    cost_rows = extract_table_rows(tex, "tab:fast-path-cost-model")
    cost_csv = {row["table_label"]: row for row in read_csv(COST_MODEL_CSV)}
    for row in cost_rows:
        label = row[0]
        csv_row = cost_csv[label]
        add_check(
            checks,
            f"Fast-path cost model {label}",
            int(parse_float(row[1])) == int(csv_row["branch_candidates"])
            and int(parse_float(row[2])) == int(csv_row["multiplications_per_shot"])
            and int(parse_float(row[3])) == int(csv_row["additions_per_shot"])
            and int(parse_float(row[4])) == int(csv_row["nonlinear_ops_per_shot"])
            and int(parse_float(row[5])) == int(csv_row["stored_state_scalars"]),
            (
                f"TeX branches/mult/add/nonlinear/state {row[1]}/{row[2]}/{row[3]}/{row[4]}/{row[5]} "
                "vs fast-path cost CSV."
            ),
        )

    fixed_point_rows = extract_table_rows(tex, "tab:fixed-point-parity")
    fixed_point_csv = {row["scenario"]: row for row in read_csv(FIXED_POINT_PARITY_CSV)}
    for row in fixed_point_rows:
        scenario = row[0]
        csv_row = fixed_point_csv[scenario]
        add_check(
            checks,
            f"Fixed-point parity table {scenario}",
            close(parse_float(row[1]), displayed6(float(csv_row["max_abs_correction_diff"])))
            and close(parse_float(row[2]), displayed6(float(csv_row["p99_abs_correction_diff"])))
            and close(parse_float(row[3]), displayed6(float(csv_row["residual_mse_delta_fixed_minus_float"])))
            and close(parse_float(row[4]), displayed6(float(csv_row["boundary_crossing_delta_fixed_minus_float"])))
            and close(parse_float(row[5]), displayed6(float(csv_row["fixed_point_quant_saturation_rate"]))),
            (
                f"TeX max/p99/MSE-delta/crossing-delta/quant-sat "
                f"{row[1]}/{row[2]}/{row[3]}/{row[4]}/{row[5]} "
                "vs fixed-point parity CSV."
            ),
        )

    runtime_rows = extract_table_rows(tex, "tab:runtime-discipline")
    runtime_csv = {row["mode_label"]: row for row in read_csv(RUNTIME_DISCIPLINE_CSV)}
    for row in runtime_rows:
        mode_label = row[0]
        csv_row = runtime_csv[mode_label]
        add_check(
            checks,
            f"Runtime discipline table {mode_label}",
            close(parse_float(row[1]), round(float(csv_row["n_commits_applied_mean"]), 1), 5e-2)
            and close(parse_float(row[2]), float(csv_row["slow_update_violation_rate_mean"]), 5e-8)
            and close(parse_float(row[3]), float(csv_row["fast_cycle_violation_rate_mean"]), 5e-7)
            and close(parse_float(row[4]), round(float(csv_row["overflow_rate_mean"]), 6), 5e-7)
            and close(parse_float(row[5]), float(csv_row["correction_saturation_rate_mean"]), 5e-8),
            (
                f"TeX commits/slow-viol/fast-viol/overflow/correction-sat "
                f"{row[1]}/{row[2]}/{row[3]}/{row[4]}/{row[5]} "
                "vs runtime-discipline CSV."
            ),
        )

    logical_channel_rows = extract_table_rows(tex, "tab:logical-channel-surrogate")
    logical_channel_csv = {
        (row["scenario"], row["method"]): row
        for row in read_csv(LOGICAL_CHANNEL_SURROGATE_CSV)
        if row["method"] in LOGICAL_CHANNEL_SURROGATE_METHODS
    }
    for row in logical_channel_rows:
        scenario = row[0]
        fixed = logical_channel_csv[(scenario, "fixed_affine")]
        oracle = logical_channel_csv[(scenario, "oracle_affine")]
        wrapped_mean = logical_channel_csv[(scenario, "wrapped_gaussian_posterior_mean")]
        wrapped_map = logical_channel_csv[(scenario, "wrapped_gaussian_map")]
        add_check(
            checks,
            f"Logical-channel surrogate table {scenario}",
            close(parse_float(row[1]), displayed6(float(fixed["pauli_surrogate_any_crossing"])))
            and close(parse_float(row[2]), displayed6(float(oracle["pauli_surrogate_any_crossing"])))
            and close(parse_float(row[3]), displayed6(float(wrapped_mean["pauli_surrogate_any_crossing"])))
            and close(parse_float(row[4]), displayed6(float(wrapped_map["pauli_surrogate_any_crossing"]))),
            (
                f"TeX fixed/oracle/wrapped-mean/wrapped-map p_any "
                f"{row[1]}/{row[2]}/{row[3]}/{row[4]} vs logical-channel surrogate CSV."
            ),
        )

    logical_fidelity_rows = extract_table_rows(tex, "tab:logical-channel-fidelity-surrogate")
    for row in logical_fidelity_rows:
        scenario = row[0]
        fixed = logical_channel_csv[(scenario, "fixed_affine")]
        oracle = logical_channel_csv[(scenario, "oracle_affine")]
        wrapped_mean = logical_channel_csv[(scenario, "wrapped_gaussian_posterior_mean")]
        wrapped_map = logical_channel_csv[(scenario, "wrapped_gaussian_map")]
        add_check(
            checks,
            f"Logical-channel surrogate fidelity table {scenario}",
            close(parse_float(row[1]), displayed6(float(fixed["pauli_surrogate_average_fidelity"])))
            and close(parse_float(row[2]), displayed6(float(oracle["pauli_surrogate_average_fidelity"])))
            and close(parse_float(row[3]), displayed6(float(wrapped_mean["pauli_surrogate_average_fidelity"])))
            and close(parse_float(row[4]), displayed6(float(wrapped_map["pauli_surrogate_average_fidelity"]))),
            (
                f"TeX fixed/oracle/wrapped-mean/wrapped-map F_avg_surr "
                f"{row[1]}/{row[2]}/{row[3]}/{row[4]} vs logical-channel surrogate CSV."
            ),
        )

    lattice_channel_rows = extract_table_rows(tex, "tab:lattice-logical-channel-sanity")
    lattice_channel_csv = {
        row["method"]: row for row in read_csv(LATTICE_LOGICAL_CHANNEL_SANITY_CSV)
    }
    add_check(
        checks,
        "Lattice logical-channel sanity row coverage",
        len(lattice_channel_csv) == len(LATTICE_LOGICAL_CHANNEL_METHOD_LABELS)
        and set(lattice_channel_csv) == set(LATTICE_LOGICAL_CHANNEL_METHOD_LABELS.values()),
        "Lattice logical-channel sanity CSV carries the four expected controlled methods.",
    )
    add_check(
        checks,
        "Lattice logical-channel sanity boundaries",
        all(
            row["non_claim_boundary"].strip().lower().startswith("not ")
            and "finite-energy GKP logical-channel fidelity" in row["non_claim_boundary"]
            for row in lattice_channel_csv.values()
        ),
        "Every lattice logical-channel sanity row states the finite-energy channel-fidelity non-claim boundary.",
    )
    for row in lattice_channel_rows:
        method = LATTICE_LOGICAL_CHANNEL_METHOD_LABELS[row[0]]
        source = lattice_channel_csv[method]
        add_check(
            checks,
            f"Lattice logical-channel sanity table {method}",
            close(parse_float(row[1]), displayed6(float(source["mean_p_any"])))
            and row[2] == source["worst_state"]
            and close(parse_float(row[3]), displayed6(float(source["worst_state_p_any"])))
            and close(parse_float(row[4]), displayed6(float(source["mean_f_avg_surr"])))
            and close(parse_float(row[5]), displayed6(float(source["worst_state_f_avg_surr"]))),
            (
                f"TeX mean/worst p_any and mean/worst F_avg_surr "
                f"{row[1]}/{row[2]}/{row[3]}/{row[4]}/{row[5]} vs lattice logical-channel sanity CSV."
            ),
        )

    finite_energy_rows = extract_table_rows(tex, "tab:finite-energy-channel-sanity")
    finite_energy_csv_rows = read_csv(FINITE_ENERGY_CHANNEL_SANITY_CSV)
    finite_energy_csv = {
        (row["finite_energy_delta"], row["method"]): row for row in finite_energy_csv_rows
    }
    add_check(
        checks,
        "Finite-energy toy-channel sanity row coverage",
        len(finite_energy_csv_rows) == len(FINITE_ENERGY_DELTAS) * len(FINITE_ENERGY_METHOD_LABELS)
        and {row["finite_energy_delta"] for row in finite_energy_csv_rows} == set(FINITE_ENERGY_DELTAS)
        and {row["method"] for row in finite_energy_csv_rows} == set(FINITE_ENERGY_METHOD_LABELS.values()),
        "Finite-energy toy-channel sanity CSV carries three deltas across hard-nearest, fixed-affine and oracle-affine methods.",
    )
    add_check(
        checks,
        "Finite-energy toy-channel sanity boundaries",
        all(
            row["non_claim_boundary"].strip().lower().startswith("not ")
            and "finite-energy GKP logical-channel fidelity" in row["non_claim_boundary"]
            for row in finite_energy_csv_rows
        ),
        "Every finite-energy toy-channel row states the calibrated finite-energy channel-fidelity non-claim boundary.",
    )
    for row in finite_energy_rows:
        delta = row[0]
        method = FINITE_ENERGY_METHOD_LABELS[row[1]]
        source = finite_energy_csv[(delta, method)]
        add_check(
            checks,
            f"Finite-energy toy-channel table delta={delta}/{method}",
            close(parse_float(row[2]), displayed6(float(source["mean_logical_event_probability"])))
            and row[3] == source["worst_state"]
            and close(parse_float(row[4]), displayed6(float(source["worst_state_logical_event_probability"])))
            and close(parse_float(row[5]), displayed6(float(source["mean_surrogate_average_fidelity"]))),
            (
                f"TeX mean/worst p_any and mean F_avg_surr "
                f"{row[2]}/{row[3]}/{row[4]}/{row[5]} vs finite-energy toy-channel CSV."
            ),
        )

    sequence_rows = extract_table_rows(tex, "tab:sequence-controlled-baselines")
    sequence_csv = {
        (row["scenario"], row["method"]): row
        for row in read_csv(SEQUENCE_BASELINE_CSV)
    }
    for row in sequence_rows:
        scenario = row[0]
        nearest = sequence_csv[(scenario, "nearest_syndrome")]
        fixed = sequence_csv[(scenario, "fixed_affine")]
        oracle = sequence_csv[(scenario, "oracle_affine")]
        wrapped_mean = sequence_csv[(scenario, "wrapped_gaussian_posterior_mean")]
        wrapped_map = sequence_csv[(scenario, "wrapped_gaussian_map")]
        add_check(
            checks,
            f"Sequence controlled baseline {scenario}",
            close(parse_float(row[1]), displayed6(float(nearest["sequence_ler_proxy_mean"])))
            and close(parse_float(row[2]), displayed6(float(fixed["sequence_ler_proxy_mean"])))
            and close(parse_float(row[3]), displayed6(float(oracle["sequence_ler_proxy_mean"])))
            and close(parse_float(row[4]), displayed6(float(wrapped_mean["sequence_ler_proxy_mean"])))
            and close(parse_float(row[5]), displayed6(float(wrapped_map["sequence_ler_proxy_mean"]))),
            (
                f"TeX nearest/fixed/oracle/wrapped-mean/wrapped-map sequence proxy "
                f"{row[1]}/{row[2]}/{row[3]}/{row[4]}/{row[5]} vs sequence controlled baseline CSV."
            ),
        )

    holdout_rows = extract_table_rows(tex, "tab:holdout-drift-stress")
    holdout_csv = {
        (row["scenario"], row["method"]): row
        for row in read_csv(HOLDOUT_DRIFT_STRESS_CSV)
        if row["method"] in HOLDOUT_DRIFT_STRESS_METHODS
    }
    for row in holdout_rows:
        scenario = row[0]
        fixed = holdout_csv[(scenario, "fixed_affine")]
        lagged = holdout_csv[(scenario, "lagged_affine")]
        oracle = holdout_csv[(scenario, "oracle_affine")]
        wrapped_mean = holdout_csv[(scenario, "wrapped_gaussian_posterior_mean")]
        wrapped_map = holdout_csv[(scenario, "wrapped_gaussian_map")]
        add_check(
            checks,
            f"Holdout drift stress table {scenario}",
            close(parse_float(row[1]), displayed6(float(fixed["residual_mse"])))
            and close(parse_float(row[2]), displayed6(float(lagged["residual_mse"])))
            and close(parse_float(row[3]), displayed6(float(oracle["residual_mse"])))
            and close(parse_float(row[4]), displayed6(float(wrapped_mean["residual_mse"])))
            and close(parse_float(row[5]), displayed6(float(wrapped_map["residual_mse"])))
            and close(parse_float(row[6]), displayed6(float(oracle["pauli_surrogate_average_fidelity"]))),
            (
                f"TeX fixed/lagged/oracle/wrapped-mean/wrapped-map residual MSE and oracle F_avg_surr "
                f"{row[1]}/{row[2]}/{row[3]}/{row[4]}/{row[5]}/{row[6]} vs holdout drift stress CSV."
            ),
        )

    affine_validity_rows = extract_table_rows(tex, "tab:affine-local-validity")
    affine_validity_csv = {
        row["surface"]: row
        for row in read_csv(AFFINE_LOCAL_VALIDITY_CSV)
    }
    add_check(
        checks,
        "Affine local-validity diagnostic row coverage",
        len(affine_validity_csv) == 7
        and {row["evidence_layer"] for row in affine_validity_csv.values()}
        == {"short_sequence_controlled", "holdout_stress_controlled"}
        and all(row["non_claim_boundary"].strip().lower().startswith("not ") for row in affine_validity_csv.values()),
        "Affine local-validity CSV carries four short-sequence rows and three holdout-stress rows with explicit non-claim boundaries.",
    )
    for row in affine_validity_rows:
        surface = row[0]
        source = affine_validity_csv[surface]
        expected_lag_cell = "--" if not source["lag_risk_delta"] else f"{float(source['lag_risk_delta']):.6f}"
        add_check(
            checks,
            f"Affine local-validity table {surface}",
            AFFINE_LOCAL_VALIDITY_LAYER_LABELS[row[1]] == source["evidence_layer"]
            and close(parse_float(row[2]), round(float(source["oracle_mse_gain_percent"]), 2), 5e-3)
            and close(parse_float(row[3]), displayed6(float(source["branch_risk_delta"])))
            and row[4] == expected_lag_cell
            and row[5].replace(" ", "_") == source["validity_readout"],
            (
                f"TeX layer/gain/branch/lag/readout "
                f"{row[1]}/{row[2]}/{row[3]}/{row[4]}/{row[5]} "
                "vs affine local-validity CSV."
            ),
        )

    commit_lag_rows = extract_table_rows(tex, "tab:commit-lag-sweep")
    commit_lag_csv_rows = read_csv(COMMIT_LAG_SWEEP_CSV)
    commit_lag_csv = {
        (row["scenario"], row["commit_lag_steps"]): row for row in commit_lag_csv_rows
    }
    add_check(
        checks,
        "Commit-lag sweep row coverage",
        len(commit_lag_csv_rows) == len(COMMIT_LAG_SWEEP_SCENARIOS) * len(COMMIT_LAG_SWEEP_LAGS)
        and {row["scenario"] for row in commit_lag_csv_rows} == set(COMMIT_LAG_SWEEP_SCENARIOS)
        and {row["commit_lag_steps"] for row in commit_lag_csv_rows} == set(COMMIT_LAG_SWEEP_LAGS)
        and all(row["commit_interval_steps"] == "64" for row in commit_lag_csv_rows),
        "Commit-lag sweep CSV carries three holdout scenarios across six simulation-step lag settings with a fixed 64-step commit interval.",
    )
    add_check(
        checks,
        "Commit-lag sweep non-claim boundary",
        all(
            "not measured FPGA latency" in row["non_claim_boundary"]
            and "trained-branch holdout generalization" in row["non_claim_boundary"]
            for row in commit_lag_csv_rows
        ),
        "Every commit-lag sweep row states that it is simulation-step diagnostic data, not measured hardware latency or trained-branch holdout proof.",
    )
    for row in commit_lag_rows:
        scenario = row[0]
        add_check(
            checks,
            f"Commit-lag sweep table {scenario}",
            all(
                close(
                    parse_float(row[idx + 1]),
                    displayed6(float(commit_lag_csv[(scenario, lag)]["residual_mse"])),
                )
                for idx, lag in enumerate(COMMIT_LAG_SWEEP_LAGS)
            ),
            (
                f"TeX lag-0/8/16/32/64/128 residual MSE values "
                f"{'/'.join(row[1:7])} vs commit-lag sweep CSV."
            ),
        )

    metric_rows = extract_table_rows(tex, "tab:metric-readiness")
    metric_csv = {row["metric_axis"]: row for row in read_csv(METRIC_READINESS_CSV)}
    add_check(
        checks,
        "Metric readiness row coverage",
        len(metric_rows) == len(METRIC_READINESS_LABELS)
        and set(METRIC_READINESS_LABELS.values()) == set(metric_csv),
        "TeX metric-readiness table and CSV carry the same five metric axes.",
    )
    for row in metric_rows:
        csv_key = METRIC_READINESS_LABELS.get(row[0])
        csv_row = metric_csv.get(csv_key or "")
        add_check(
            checks,
            f"Metric readiness table {row[0]}",
            csv_row is not None and clean_cell(row[1]) and clean_cell(row[2]) and clean_cell(row[3]),
            "Metric-readiness TeX row is backed by the CSV axis and has non-empty current metric, supported statement and missing-evidence cells.",
        )

    literature_rows = read_csv(LITERATURE_CROSSWALK_CSV)
    literature_axes = {row["axis"] for row in literature_rows}
    literature_citations = {row["citation_key"] for row in literature_rows}
    active_citations = active_citation_keys(tex)
    add_check(
        checks,
        "Literature metric crosswalk row coverage",
        len(literature_rows) == 23 and literature_axes == LITERATURE_CROSSWALK_AXES,
        "Literature metric crosswalk carries 23 rows across the six external-comparison axes.",
    )
    add_check(
        checks,
        "Literature metric crosswalk active citations",
        literature_citations.issubset(active_citations),
        "Every citation_key in the literature crosswalk appears in the active submission-draft citation surface.",
    )
    add_check(
        checks,
        "Literature metric crosswalk boundaries",
        all(
            row["reported_metric"].strip()
            and row.get("source_anchor", "").strip()
            and row["manuscript_use"].strip()
            and row["manuscript_boundary"].strip().lower().startswith("not ")
            for row in literature_rows
        ),
        "Every literature crosswalk row has a reported metric, source anchor, manuscript use and explicit non-claim boundary.",
    )
    add_check(
        checks,
        "Literature metric crosswalk anchor policies",
        all(
            row["anchor_strength"].strip()
            and row["manuscript_number_policy"].strip()
            and row["follow_up_needed"].strip()
            for row in literature_rows
        ),
        "Every literature crosswalk row records anchor strength, manuscript number policy and final pinning follow-up.",
    )
    add_check(
        checks,
        "Literature metric crosswalk source anchors",
        all(
            "zotero_literature_review_cards.md:" in row.get("source_anchor", "")
            and (
                "Fig." in row["source_anchor"]
                or "Table" in row["source_anchor"]
                or "Eq." in row["source_anchor"]
                or "figure" in row["source_anchor"].lower()
                or "card-level" in row["source_anchor"]
                or "public PDF/HTML" in row["source_anchor"]
                or "public HTML/abstract" in row["source_anchor"]
            )
            for row in literature_rows
        ),
        "Every literature crosswalk row points back to a local literature-card anchor and either a figure/table/equation anchor or an explicit card-level/public-text limitation.",
    )
    add_check(
        checks,
        "Literature metric crosswalk hardware pinning",
        all(
            (
                "Fig." in row.get("source_anchor", "")
                or "Table" in row.get("source_anchor", "")
                or "public HTML/abstract" in row.get("source_anchor", "")
                or "pinning" in row.get("source_anchor", "")
            )
            and (
                "page/table/figure" in row["follow_up_needed"]
                or row["anchor_strength"] == "figure_or_table_checked"
            )
            for row in literature_rows
            if row["axis"] == "real_time_fpga_decoders"
        ),
        "Hardware comparison rows include a source anchor and either figure/table checked status or page/table/figure pinning before strong per-value claims.",
    )

    closest_rows = read_csv(CLOSEST_WORK_POSITIONING_CSV)
    closest_tex_rows = extract_table_rows(tex, "tab:closest-work-positioning")
    closest_families = {row["closest_work_family"] for row in closest_rows}
    closest_citations = {
        key.strip()
        for row in closest_rows
        for key in row["citation_keys"].split(";")
        if key.strip()
    }
    closest_by_family = {row["closest_work_family"]: row for row in closest_rows}
    add_check(
        checks,
        "Closest-work positioning row coverage",
        len(closest_rows) == len(CLOSEST_WORK_FAMILIES)
        and closest_families == CLOSEST_WORK_FAMILIES
        and len(closest_tex_rows) == len(CLOSEST_WORK_FAMILIES),
        "Closest-work positioning CSV and TeX table carry the same five adjacent-work families.",
    )
    add_check(
        checks,
        "Closest-work positioning active citations",
        closest_citations.issubset(active_citations),
        "Every citation key used by the closest-work positioning CSV appears in the active manuscript citation surface.",
    )
    add_check(
        checks,
        "Closest-work positioning boundaries",
        all(
            row["current_evidence_boundary"].strip().lower().startswith("not ")
            and row["source_anchor"].startswith("zotero_literature_review_cards.md")
            for row in closest_rows
        ),
        "Every closest-work row has an explicit non-claim boundary and a local literature-card source anchor.",
    )
    for row in closest_tex_rows:
        family = row[0]
        source = closest_by_family.get(family)
        add_check(
            checks,
            f"Closest-work positioning table {family}",
            source is not None
            and row[1].startswith(source["representative_metric_standard"])
            and row[2].startswith(source["manuscript_distinction"])
            and clean_cell(row[3]).lower().startswith(source["current_evidence_boundary"].lower()),
            "TeX closest-work row aligns with the generated CSV family, metric standard, distinction and boundary.",
        )

    coverage_rows = read_csv(SOURCE_DATA_COVERAGE_MATRIX_CSV)
    coverage_json = json.loads(SOURCE_DATA_COVERAGE_MATRIX_JSON.read_text(encoding="utf-8"))
    coverage_tex_rows = extract_table_rows(tex, "tab:source-data-coverage-matrix")
    add_check(
        checks,
        "Source-data coverage matrix row coverage",
        len(coverage_rows) == len(SOURCE_DATA_COVERAGE_GROUPS)
        and len(coverage_json.get("rows", [])) == len(coverage_rows)
        and {row["coverage_group"] for row in coverage_rows} == set(SOURCE_DATA_COVERAGE_GROUPS),
        "Source-data coverage matrix carries the expected manuscript coverage groups in CSV and JSON.",
    )
    add_check(
        checks,
        "Source-data coverage matrix boundaries",
        all(
            row["claim_boundary"].strip().lower().startswith("not ")
            and row["coverage_status"].strip()
            and row["source_files"].strip()
            for row in coverage_rows
        ),
        "Every source-data coverage matrix row has a status, source-file surface and explicit non-claim boundary.",
    )
    add_check(
        checks,
        "Source-data coverage matrix TeX labels",
        {row[0] for row in coverage_tex_rows} == set(SOURCE_DATA_COVERAGE_GROUPS.values()),
        "TeX source-data coverage table carries the same expected coverage-group labels as the CSV.",
    )

    benchmark_protocol_rows = read_csv(BENCHMARK_EXPANSION_PROTOCOL_CSV)
    phase_a_rows = [
        row
        for row in benchmark_protocol_rows
        if row["protocol_phase"] == "Phase A repeat-expanded anchor comparison"
    ]
    phase_b_rows = [
        row for row in benchmark_protocol_rows if row["protocol_phase"] == "Phase B holdout drift expansion"
    ]
    phase_a_scenarios = {row["scenario_family"] for row in phase_a_rows}
    add_check(
        checks,
        "Benchmark expansion protocol row coverage",
        len(benchmark_protocol_rows) == 7
        and phase_a_scenarios == {"static_bias_theta", "linear_ramp", "step_sigma_theta", "periodic_drift"},
        "Benchmark expansion protocol carries four Phase A scenario rows plus Phase A reporting, Phase B holdout and Phase C provenance rows.",
    )
    add_check(
        checks,
        "Benchmark expansion repeat budget",
        all(
            int(row["planning_min_pairs_per_scenario"]) >= 12
            and row["recommended_target_pairs_per_scenario"] == "16"
            and row["non_claim_boundary"].strip()
            for row in phase_a_rows
        ),
        "Every Phase A benchmark-expansion scenario has a minimum repeat budget, target repeat budget and non-claim boundary.",
    )
    add_check(
        checks,
        "Benchmark expansion holdout families",
        len(phase_b_rows) == 1
        and all(
            family in phase_b_rows[0]["scenario_family"]
            for family in ["random_walk_drift", "burst_reset_drift", "faster_than_window_drift"]
        ),
        "Phase B predeclares random-walk, burst/reset and faster-than-window holdout drift families without treating them as current results.",
    )

    phase_a_plan_rows = read_csv(PHASE_A_REPEAT_PLAN_CSV)
    formal_plan_rows = [row for row in phase_a_plan_rows if row["lane"] == "formal_length_phase_a"]
    smoke_plan_rows = [row for row in phase_a_plan_rows if row["lane"] == "smoke_length_feasibility"]
    add_check(
        checks,
        "Phase A repeat plan coverage",
        len(formal_plan_rows) == 12
        and len(smoke_plan_rows) == 4
        and {row["scenario"] for row in formal_plan_rows} == set(PAIRED_SCENARIO_LABELS)
        and all(row["paired_seeds"] == "true" and row["total_repeats"] == "12" for row in phase_a_plan_rows),
        "Phase A repeat plan carries three formal chunks and one smoke feasibility row for each predeclared scenario.",
    )
    add_check(
        checks,
        "Phase A repeat plan non-claim boundary",
        all("not" in row["claim_boundary"].lower() for row in phase_a_plan_rows),
        "Every Phase A plan row carries an explicit non-claim or not-yet-upgraded boundary.",
    )

    phase_a_summary_rows = read_csv(PHASE_A_REPEAT_SUMMARY_CSV)
    add_check(
        checks,
        "Phase A repeat summary boundary",
        all("not" in row["claim_boundary"].lower() for row in phase_a_summary_rows),
        "Any completed Phase A summary rows retain a non-claim boundary; an empty summary is allowed before Phase A runs exist.",
    )

    phase_a_interval_rows = read_csv(PHASE_A_PAIRED_INTERVAL_CSV)
    interval_tex_rows = extract_table_rows(tex, "tab:phase-a-paired-interval")
    completed_interval_scenarios = {row["scenario"] for row in phase_a_interval_rows}
    add_check(
        checks,
        "Phase A paired interval source boundary",
        len(phase_a_interval_rows) >= 1
        and all(row["interval_lower_bounds_positive"] == "true" for row in phase_a_interval_rows)
        and all("not all-scenario" in row["claim_boundary"].lower() for row in phase_a_interval_rows),
        (
            "Phase A paired interval source data covers completed formal "
            f"scenario(s): {', '.join(sorted(completed_interval_scenarios))}; "
            "the broader gate remains blocked."
        ),
    )
    interval_source_by_scenario = {row["scenario"]: row for row in phase_a_interval_rows}
    interval_tex_by_scenario = {row[0]: row for row in interval_tex_rows}
    interval_match = bool(phase_a_interval_rows) and (
        set(interval_source_by_scenario) == set(interval_tex_by_scenario)
    )
    if interval_match:
        for scenario, interval_source in interval_source_by_scenario.items():
            interval_tex = interval_tex_by_scenario[scenario]
            tex_t_low, tex_t_high = parse_interval_cell(interval_tex[3])
            tex_boot_low, tex_boot_high = parse_interval_cell(interval_tex[4])
            scenario_match = (
                int(interval_tex[1]) == int(interval_source["n_paired_repeats"])
                and close(parse_float(interval_tex[2]), displayed6(float(interval_source["mean_delta_ukf_minus_hybrid"])), 5e-7)
                and close(tex_t_low, displayed6(float(interval_source["paired_t_95_lower"])), 5e-7)
                and close(tex_t_high, displayed6(float(interval_source["paired_t_95_upper"])), 5e-7)
                and close(tex_boot_low, displayed6(float(interval_source["bootstrap_95_lower"])), 5e-7)
                and close(tex_boot_high, displayed6(float(interval_source["bootstrap_95_upper"])), 5e-7)
            )
            if not scenario_match:
                interval_match = False
                break
    add_check(
        checks,
        "Phase A paired interval TeX values",
        interval_match,
        "TeX formal Phase A interval table matches the generated paired-interval CSV for all completed scenarios.",
    )

    phase_a_gate_rows = read_csv(PHASE_A_UPGRADE_GATE_CSV)
    phase_a_gate_labels = {row["evidence_class"] for row in phase_a_gate_rows}
    expected_gate_labels = {
        "Current descriptive benchmark",
        "Short-run repeat rehearsal",
        "Formal Phase A repeat expansion",
        "Formal holdout drift expansion",
        "Hardware-facing measurements",
    }
    add_check(
        checks,
        "Phase A upgrade gate coverage",
        len(phase_a_gate_rows) == 5 and phase_a_gate_labels == expected_gate_labels,
        "Phase A upgrade gate separates descriptive, short-run, formal repeat, holdout and hardware evidence classes.",
    )
    add_check(
        checks,
        "Phase A upgrade gate non-claim boundary",
        all(
            "do not" in row["forbidden_inference"].lower()
            or "not " in row["forbidden_inference"].lower()
            for row in phase_a_gate_rows
        ),
        "Every Phase A upgrade-gate row states a forbidden inference.",
    )
    phase_a_gate_tex_rows = extract_table_rows(tex, "tab:phase-a-upgrade-gate")
    add_check(
        checks,
        "Phase A upgrade gate TeX labels",
        expected_gate_labels.issubset({row[0] for row in phase_a_gate_tex_rows}),
        "TeX Phase A upgrade-gate table carries all evidence-class labels from the CSV.",
    )

    runner_smoke_rows = read_csv(RUNNER_SMOKE_PAIR_CSV)
    runner_smoke_modes = {row["mode"] for row in runner_smoke_rows}
    add_check(
        checks,
        "Runner smoke pair row coverage",
        len(runner_smoke_rows) == 2
        and runner_smoke_modes == {"ukf", "hybrid_residual_b"}
        and {row["scenario"] for row in runner_smoke_rows} == {"static_bias_theta"},
        "Runner smoke-pair source data carries exactly one UKF row and one Hybrid row for one static-bias scenario.",
    )
    add_check(
        checks,
        "Runner smoke pair non-claim boundary",
        all(
            row["completed_repeats"] == "1"
            and row["expected_repeats"] == "1"
            and "not an expanded benchmark" in row["planning_boundary"]
            for row in runner_smoke_rows
        ),
        "Runner smoke-pair rows are explicitly bounded as one-repeat feasibility data, not expanded benchmark evidence.",
    )

    runner_matrix_rows = read_csv(RUNNER_SMOKE_MATRIX_CSV)
    runner_matrix_tex_rows = extract_table_rows(tex, "tab:runner-smoke-matrix")
    runner_matrix_by_scenario = {row["scenario"]: row for row in runner_matrix_rows}
    add_check(
        checks,
        "Runner smoke matrix row coverage",
        len(runner_matrix_rows) == 4
        and {row["scenario"] for row in runner_matrix_rows}
        == {"static_bias_theta", "linear_ramp", "step_sigma_theta", "periodic_drift"}
        and len(runner_matrix_tex_rows) == 4,
        "Runner smoke matrix covers all four predeclared scenarios in CSV and TeX.",
    )
    for tex_row in runner_matrix_tex_rows:
        scenario = tex_row[0]
        csv_row = runner_matrix_by_scenario[scenario]
        add_check(
            checks,
            f"Runner smoke matrix table {scenario}",
            close(float(tex_row[1]), float(csv_row["ukf_final_ler_mean"]))
            and close(float(tex_row[2]), float(csv_row["hybrid_final_ler_mean"]))
            and close(float(tex_row[3]), float(csv_row["ukf_minus_hybrid_final_ler_delta"]))
            and close(float(tex_row[4]), float(csv_row["relative_reduction_percent"]), tol=5e-3)
            and tex_row[5] == csv_row["positive_pairs"],
            f"TeX UKF/Hybrid/delta/relative/positive-pair values match the runner smoke matrix CSV for {scenario}.",
        )
    add_check(
        checks,
        "Runner smoke matrix non-claim boundary",
        all(
            row["completed_pairs"] == "2"
            and row["expected_repeats"] == "2"
            and row["coverage"] == "1.0"
            and "not an expanded benchmark" in row["planning_boundary"]
            for row in runner_matrix_rows
        ),
        "Runner smoke matrix rows are explicitly bounded as all-scenario smoke feasibility data, not expanded benchmark evidence.",
    )

    manifest = json.loads((FIG_DIR / "figure_manifest.json").read_text(encoding="utf-8"))
    manifest_sources = set(manifest.get("source_data", []))
    manifest_outputs = [FIG_DIR / out for out in manifest.get("outputs", [])]
    add_check(
        checks,
        "Figure manifest source-data list",
        manifest_sources
        == {
            "source_data_fig02_main_results.csv",
            "source_data_fig02_paired_deltas.csv",
            "source_data_fig03_ablation_mechanism.csv",
            "source_data_fig04_statcalib.csv",
            "source_data_fig05_validation_contract.csv",
        },
        "figure_manifest.json lists the source CSV files used by Fig. 2-5 and paired Fig. 2 deltas.",
    )
    add_check(
        checks,
        "Figure manifest outputs exist",
        all(path.exists() for path in manifest_outputs),
        "All figure_manifest.json PDF outputs are present.",
    )

    source_manifest_rows = read_csv(SOURCE_MANIFEST_CSV)
    source_manifest_json = json.loads(SOURCE_MANIFEST_JSON.read_text(encoding="utf-8"))
    manifest_missing_paths: list[str] = []
    manifest_hash_mismatches: list[str] = []
    for row in source_manifest_rows:
        source_path = ROOT / row["source_path"]
        if not source_path.is_file():
            manifest_missing_paths.append(row["source_path"])
            continue
        actual_hash = sha256(source_path)
        if actual_hash != row["sha256"]:
            manifest_hash_mismatches.append(row["source_path"])
    add_check(
        checks,
        "Source-data manifest row coverage",
        len(source_manifest_rows) > 0
        and len(source_manifest_json.get("rows", [])) == len(source_manifest_rows),
        f"CSV and JSON source-data manifests carry the same {len(source_manifest_rows)} manuscript-facing source/script rows.",
    )
    add_check(
        checks,
        "Source-data manifest file paths",
        not manifest_missing_paths,
        "Every source_path listed in the source-data manifest exists in the current checkout.",
    )
    add_check(
        checks,
        "Source-data manifest hashes",
        not manifest_hash_mismatches,
        "Every source-data manifest SHA-256 matches the current file content.",
    )

    hybrid_paths = {row["artifact_path"] for row in t24_rows if row["mode"] == "hybrid_residual_b"}
    baseline_artifacts_blank = all(
        not row["artifact_path"] for row in t24_rows if row["mode"] != "hybrid_residual_b"
    )
    add_check(
        checks,
        "T24 artifact path stratification",
        len(hybrid_paths) == 1 and baseline_artifacts_blank,
        "Hybrid rows share one model artifact path; non-hybrid baseline rows have no model artifact path.",
    )

    row_provenance_rows = read_csv(ROW_PROVENANCE_CSV)
    row_provenance_json = json.loads(ROW_PROVENANCE_JSON.read_text(encoding="utf-8"))
    row_provenance_scenarios = {row["scenario"] for row in row_provenance_rows}
    row_provenance_modes = {row["mode"] for row in row_provenance_rows}
    row_provenance_repeats = {row["repeat"] for row in row_provenance_rows}
    add_check(
        checks,
        "Row-level provenance coverage",
        len(row_provenance_rows) == 40
        and row_provenance_json.get("row_count") == 40
        and row_provenance_scenarios == set(PAIRED_SCENARIO_LABELS)
        and row_provenance_modes == set(MAIN_MODE_MAP.values())
        and row_provenance_repeats == {"0", "1"},
        "Row-level provenance manifest covers four scenarios, five modes and two repeats.",
    )
    add_check(
        checks,
        "Row-level provenance hashes",
        all(
            row["summary_sha256"]
            and row["launch_plan_sha256"]
            and row["comparison_sha256"]
            and row["config_sha256"]
            and row["runner_sha256"]
            and row["run_hil_summary_sha256"]
            and row["run_repeat_status_sha256"]
            for row in row_provenance_rows
        ),
        "Every row-level provenance entry has source summary, launch, comparison, config, runner and repeat summary/status hashes.",
    )
    add_check(
        checks,
        "Row-level provenance non-claim boundary",
        all("not new benchmark evidence" in row["non_claim_boundary"] for row in row_provenance_rows),
        "Every row-level provenance entry is explicitly bounded as source trace rather than new evidence.",
    )

    artifact_hashes: dict[str, str] = {}
    for raw_path in sorted(path for path in hybrid_paths if path):
        artifact = Path(raw_path)
        if not artifact.exists():
            artifact = ROOT / raw_path.replace("\\", "/")
        if artifact.exists():
            artifact_hashes[str(artifact.relative_to(ROOT))] = sha256(artifact)
    add_check(
        checks,
        "Hybrid model artifact hash",
        len(artifact_hashes) == 1,
        "The shared hybrid .npz artifact exists and has a SHA256 hash recorded in this audit.",
    )

    input_hashes = {
        "tex": sha256(TEX_PATH),
        "t24_comparison_csv": sha256(T24_COMPARISON),
        "t24_summary_json": sha256(T24_SUMMARY),
        "fig02_source_csv": sha256(FIG_DIR / "source_data_fig02_main_results.csv"),
        "fig02_paired_source_csv": sha256(FIG_DIR / "source_data_fig02_paired_deltas.csv"),
        "fig03_source_csv": sha256(FIG_DIR / "source_data_fig03_ablation_mechanism.csv"),
        "fig04_source_csv": sha256(FIG_DIR / "source_data_fig04_statcalib.csv"),
        "fig05_source_csv": sha256(FIG_DIR / "source_data_fig05_validation_contract.csv"),
        "controlled_oracle_affine_csv": sha256(ORACLE_AFFINE_CSV),
        "fast_path_cost_model_csv": sha256(COST_MODEL_CSV),
        "fixed_point_parity_csv": sha256(FIXED_POINT_PARITY_CSV),
        "logical_channel_surrogate_csv": sha256(LOGICAL_CHANNEL_SURROGATE_CSV),
        "finite_energy_channel_sanity_csv": sha256(FINITE_ENERGY_CHANNEL_SANITY_CSV),
        "holdout_drift_stress_csv": sha256(HOLDOUT_DRIFT_STRESS_CSV),
        "affine_local_validity_diagnostic_csv": sha256(AFFINE_LOCAL_VALIDITY_CSV),
        "commit_lag_sweep_csv": sha256(COMMIT_LAG_SWEEP_CSV),
        "commit_lag_sweep_json": sha256(COMMIT_LAG_SWEEP_JSON),
        "sequence_controlled_baseline_csv": sha256(SEQUENCE_BASELINE_CSV),
        "paired_uncertainty_csv": sha256(PAIRED_UNCERTAINTY_CSV),
        "ler_advantage_margin_csv": sha256(LER_ADVANTAGE_MARGIN_CSV),
        "metric_readiness_csv": sha256(METRIC_READINESS_CSV),
        "literature_metric_crosswalk_csv": sha256(LITERATURE_CROSSWALK_CSV),
        "closest_work_positioning_csv": sha256(CLOSEST_WORK_POSITIONING_CSV),
        "source_data_coverage_matrix_csv": sha256(SOURCE_DATA_COVERAGE_MATRIX_CSV),
        "source_data_coverage_matrix_json": sha256(SOURCE_DATA_COVERAGE_MATRIX_JSON),
        "benchmark_expansion_protocol_csv": sha256(BENCHMARK_EXPANSION_PROTOCOL_CSV),
        "phase_a_repeat_plan_csv": sha256(PHASE_A_REPEAT_PLAN_CSV),
        "phase_a_repeat_summary_csv": sha256(PHASE_A_REPEAT_SUMMARY_CSV),
        "phase_a_paired_interval_csv": sha256(PHASE_A_PAIRED_INTERVAL_CSV),
        "phase_a_upgrade_gate_csv": sha256(PHASE_A_UPGRADE_GATE_CSV),
        "runner_smoke_pair_csv": sha256(RUNNER_SMOKE_PAIR_CSV),
        "runner_smoke_matrix_csv": sha256(RUNNER_SMOKE_MATRIX_CSV),
        "row_provenance_manifest_csv": sha256(ROW_PROVENANCE_CSV),
        "row_provenance_manifest_json": sha256(ROW_PROVENANCE_JSON),
        "runtime_discipline_csv": sha256(RUNTIME_DISCIPLINE_CSV),
        "gkp_boundary_sensitivity_csv": sha256(GKP_BOUNDARY_SENSITIVITY_CSV),
        "submission_source_data_manifest_csv": sha256(SOURCE_MANIFEST_CSV),
        "submission_source_data_manifest_json": sha256(SOURCE_MANIFEST_JSON),
        "figure_manifest_json": sha256(FIG_DIR / "figure_manifest.json"),
    }

    failures = [check for check in checks if check.status != "PASS"]
    result = {
        "status": "PASS_WITH_LIMITATIONS" if not failures else "FAIL",
        "scope": "submission_draft_source_data_audit_v8",
        "boundary": (
            "Mechanical consistency check for current TeX tables, figure source "
            "CSV files, T24 artifacts, the controlled oracle-affine / "
            "wrapped-Gaussian CSV, the sequence-controlled baseline CSV and "
            "the GKP boundary-sensitivity, fast-path cost, fixed-point parity, runtime-discipline, logical-channel surrogate, lattice logical-channel sanity, finite-energy toy-channel sanity, holdout drift stress, affine local-validity diagnostic, commit-lag sweep, "
            "paired-uncertainty, LER advantage-margin, metric-readiness, literature-crosswalk anchor policies, closest-work positioning, source-data coverage matrix, benchmark-expansion-protocol, phase-a-repeat-plan, phase-a-repeat-summary, phase-a-upgrade-gate, runner-smoke-pair, runner-smoke-matrix and row-level provenance CSVs only; not a new formal benchmark, CI "
            "analysis, full reproducibility proof, fallback absence proof or "
            "hardware validation."
        ),
        "checks": [check.__dict__ for check in checks],
        "input_hashes": input_hashes,
        "artifact_hashes": artifact_hashes,
        "known_limitations": [
            "The helper checks selected manuscript tables and the Fig. 5 validation-contract source summary, not every table in the draft.",
            "Supplementary StatCalib values are checked against the current source CSV and TeX table; this helper does not re-open FR8 artifacts.",
            "The controlled oracle-affine and wrapped-Gaussian table is checked against its generated CSV; the helper does not make that CSV a formal benchmark, CI analysis or hardware result.",
            "The sequence-controlled baseline table is checked against its generated CSV; the helper does not make that CSV a formal benchmark, CI analysis, holdout drift run or hardware result.",
            "The GKP boundary-sensitivity table is checked against an analytical Gaussian residual-boundary CSV; the helper does not turn it into finite-energy GKP logical-channel simulation, process tomography, hardware logical error rate or benchmark evidence.",
            "The fast-path cost table is checked against an analytical count CSV; the helper does not turn it into FPGA synthesis, timing closure, power/resource measurement or hardware validation.",
            "The fixed-point parity table is checked against a software-emulation CSV; the helper does not turn it into FPGA synthesis, timing closure, power/resource measurement, source-vs-board agreement or hardware validation.",
            "The runtime-discipline table is checked against software-in-the-loop counters; the helper does not turn it into board commit latency, hardware reliability, rollback proof, source-vs-board agreement or FPGA timing/resource evidence.",
            "The logical-channel surrogate, lattice sanity and finite-energy toy-channel tables are checked against residual-boundary and toy measurement-channel CSVs; the helper does not turn them into calibrated finite-energy GKP logical-channel tomography, process fidelity or hardware fidelity.",
            "The holdout drift stress table is checked against a controlled stress-test CSV; the helper does not turn it into a formal expanded benchmark, confidence-interval analysis, trained-branch generalization proof or hardware validation.",
            "The affine local-validity diagnostic table is checked against a derived CSV; the helper does not turn it into a formal nearest-lattice or wrapped-decoder benchmark, inferential analysis, trained-branch holdout proof or hardware validation.",
            "The commit-lag sweep table is checked against a controlled simulation-step CSV; the helper does not turn it into measured FPGA latency, source-vs-board agreement, trained-branch holdout proof or hardware validation.",
            "The paired-uncertainty table is checked against a descriptive resampling CSV; the helper does not turn it into a confidence interval, standard error, p-value, statistical significance claim or robustness proof.",
            "The LER advantage-margin table is checked against a descriptive source-data CSV; the helper does not turn delta/max SD into a confidence interval, standard error, p-value, statistical significance claim, expanded benchmark or hardware evidence.",
            "The metric-readiness table is checked against a manuscript-positioning CSV; the helper does not estimate channel fidelity, hardware latency or statistical significance.",
            "The literature metric crosswalk checks manuscript-positioning and anchor-policy coverage only; it does not normalize prior results or convert them into baselines for this study.",
            "The closest-work positioning table checks adjacent-work family coverage and non-claim boundaries only; it does not normalize external metrics or convert them into this manuscript's results.",
            "The source-data coverage matrix checks coverage classification only; it does not make unchecked or planning surfaces into result evidence.",
            "The benchmark-expansion protocol checks planning coverage only; it does not run the repeat-expanded benchmark, establish holdout robustness or provide CI/p-values.",
            "The Phase A repeat plan checks command-shape and planning coverage only; it does not run benchmarks, establish holdout robustness or provide CI/p-values.",
            "The Phase A repeat summary checks completed run summaries only when present; short-run rows remain feasibility-only and are not manuscript performance evidence.",
            "The Phase A upgrade gate checks wording boundaries only; it does not run benchmarks, compute intervals or provide hardware validation.",
            "The runner smoke-pair rows check feasibility/source traceability only; they are one-scenario, one-repeat smoke data and are not used as main-text performance evidence.",
            "The runner smoke-matrix rows check all-scenario short-run feasibility/source traceability only; they are not the main benchmark, an expanded benchmark, confidence-interval evidence, holdout robustness or hardware evidence.",
            "The row-level provenance manifest checks scenario/mode/repeat source traceability for the existing software-HIL rows; it is not recursive historical run-directory hash closure and excludes hil_events.json hashes.",
            "The source-data manifest checks file-level hashes for manuscript-facing source files and scripts, not recursive historical run-directory hash closure.",
            "The audit does not provide confidence intervals, p-values, holdout drift families or repeated-run closure.",
            "The audit does not validate real-board, default-environment .tflite or deployment behavior.",
        ],
    }
    return result


def write_reports(result: dict[str, object]) -> None:
    REPORT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    checks = result["checks"]
    pass_count = sum(1 for check in checks if check["status"] == "PASS")
    fail_count = sum(1 for check in checks if check["status"] != "PASS")
    lines = [
        "# 投稿稿 source-data 机械审计报告",
        "",
        "## 作用边界",
        "",
        "本文档由 `docs/figure_assets/submission_draft_python_figures/audit_submission_draft_source_data.py` 生成，服务 `docs/paper_notes/CNN_FPGA_GKP_submission_draft.tex`。它只检查当前投稿稿表格、figure source CSV、benchmark comparison/summary、controlled analysis CSV、literature metric crosswalk、benchmark expansion protocol、runner smoke pair、runner smoke matrix、row-level provenance、图件 manifest 与 source-data manifest 的机械一致性。",
        "",
        "本文档不是新实验，不运行 benchmark，不报告 CI，不证明 full reproducibility，不证明 fallback-free runtime，也不改变 `.tflite`、real-board、statcalib 或硬件证据等级。",
        "",
        "## 审计结论",
        "",
        f"- Status: `{result['status']}`",
        f"- Checks passed: `{pass_count}`",
        f"- Checks failed: `{fail_count}`",
        "- 主结论：当前 TeX 表格、figure source data、生成分析 CSV、literature metric crosswalk、source-data coverage matrix、benchmark expansion protocol、runner smoke pair、runner smoke matrix、row-level provenance、图件 manifest 和 source-data manifest 在已检查字段上机械一致。",
        "- 解释边界：这只能支持 manuscript table/source-data traceability；不等于强统计、expanded benchmark、硬件验证或完整复现包。",
        "",
        "## 输入文件哈希",
        "",
        "| 文件 | SHA256 |",
        "| --- | --- |",
    ]
    for name, digest in result["input_hashes"].items():
        lines.append(f"| `{name}` | `{digest}` |")

    lines.extend(["", "## 模型 artifact 哈希", "", "| Artifact | SHA256 |", "| --- | --- |"])
    artifact_hashes = result.get("artifact_hashes", {})
    if artifact_hashes:
        for name, digest in artifact_hashes.items():
            lines.append(f"| `{name}` | `{digest}` |")
    else:
        lines.append("| none | none |")

    lines.extend(["", "## 检查明细", "", "| Check | Status | Detail |", "| --- | --- | --- |"])
    for check in checks:
        detail = str(check["detail"]).replace("|", "\\|")
        lines.append(f"| `{check['name']}` | `{check['status']}` | {detail} |")

    lines.extend(["", "## 已知限制", ""])
    for item in result["known_limitations"]:
        lines.append(f"- {item}")

    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    result = audit()
    write_reports(result)
    print(json.dumps({"status": result["status"], "report": str(REPORT_MD)}, ensure_ascii=False))
    return 0 if result["status"] == "PASS_WITH_LIMITATIONS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
