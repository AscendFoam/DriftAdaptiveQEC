from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
REPO = ROOT.parents[2]
T24_RUN_DIR = REPO / "runs" / "p4_benchmark" / "T24_formal_software_revalidation_20260510_200743"
T24_COMPARISON = T24_RUN_DIR / "comparison.csv"
T24_SUMMARY = T24_RUN_DIR / "summary.json"
COST_MODEL_CSV = REPO / "docs" / "paper_materials" / "submission_draft_fast_path_cost_model.csv"
FIXED_POINT_CSV = REPO / "docs" / "paper_materials" / "submission_draft_fixed_point_parity_analysis.csv"
RUNTIME_DISCIPLINE_CSV = REPO / "docs" / "paper_materials" / "submission_draft_runtime_discipline_summary.csv"

SCENARIOS = [
    "static_bias_theta",
    "linear_ramp",
    "step_sigma_theta",
    "periodic_drift",
]

MAIN_RESULTS = {
    "EKF": [0.838110, 0.819200, 0.822365, 0.832192],
    "UKF": [0.825370, 0.811201, 0.811548, 0.821558],
    "Const.-mu": [0.836658, 0.816911, 0.819784, 0.829670],
    "RLS-b": [0.837577, 0.819373, 0.821493, 0.832334],
    "Hybrid-b": [0.810902, 0.787755, 0.788800, 0.806392],
}

MAIN_RESULT_STD = {
    "EKF": [0.0008326388888889036, 0.00017013888888889328, 0.00017013888888889328, 0.0005577777777777682],
    "UKF": [0.0006729166666666897, 0.0008683333333333043, 0.0007611111111111013, 0.0018847222222221904],
    "Const.-mu": [0.000128611111111121, 0.00012444444444442704, 0.000275833333333364, 0.0003454166666666536],
    "RLS-b": [0.0007340277777777571, 0.00009208333333332597, 0.00042000000000003146, 0.00004027777777776409],
    "Hybrid-b": [0.0011884722222222366, 0.0004393055555555469, 0.001069166666666621, 0.0002891666666666737],
}

CSV_MODE_TO_FIG = {
    "ekf": "EKF",
    "ukf": "UKF",
    "constant_residual_mu": "Const.-mu",
    "rls_residual_b": "RLS-b",
    "hybrid_residual_b": "Hybrid-b",
}

ABLATION = [
    ("UKF", 0.817382, 0.000000),
    ("Hybrid full", 0.798545, -0.018837),
    ("No hist. deltas", 0.826723, +0.009341),
    ("No teacher pred.", 0.807251, -0.010131),
    ("No teacher params", 0.749621, -0.067761),
    ("No teacher deltas", 0.800329, -0.017053),
]

MECHANISM = [
    ("20260425", 0.000907, 0.163289),
    ("20260427", -0.145352, 0.287166),
    ("20260428", -0.078998, 0.057395),
    ("20260429", -0.127948, 0.322245),
    ("20260430", -0.170777, -0.024372),
    ("20260510", -0.003953, -0.035533),
]

STATCALIB = {
    "UKF": [0.825370, 0.811201, 0.811548, 0.821558],
    "Hybrid-b": [0.810902, 0.787755, 0.788800, 0.806392],
    "StatCalib supp.": [0.431708, 0.467083, 0.460016, 0.438751],
}

COLORS = {
    "ink": "#202124",
    "muted": "#6B7280",
    "blue": "#3B6FB6",
    "teal": "#2A9D8F",
    "orange": "#E76F51",
    "purple": "#7B5EA7",
    "green": "#4C956C",
    "gray": "#B8C0CC",
}


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def load_t24_main_results():
    """Load the checked-in T24 summary when available; fall back to literals."""
    if not T24_COMPARISON.exists():
        return MAIN_RESULTS, MAIN_RESULT_STD

    values = {mode: [] for mode in MAIN_RESULTS}
    stds = {mode: [] for mode in MAIN_RESULTS}
    by_key = {}
    with T24_COMPARISON.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["scenario"] in SCENARIOS and row["mode"] in CSV_MODE_TO_FIG:
                by_key[(row["scenario"], CSV_MODE_TO_FIG[row["mode"]])] = row

    for mode in MAIN_RESULTS:
        for scenario in SCENARIOS:
            row = by_key.get((scenario, mode))
            if row is None:
                return MAIN_RESULTS, MAIN_RESULT_STD
            values[mode].append(float(row["final_ler_mean"]))
            stds[mode].append(float(row["final_ler_std"]))
    return values, stds


def load_t24_paired_deltas():
    """Return UKF-vs-hybrid paired repeat deltas from the frozen T24 summary."""
    if not T24_SUMMARY.exists():
        return []

    data = json.loads(T24_SUMMARY.read_text(encoding="utf-8"))
    rows = data.get("raw_rows", [])
    by_key = {}
    for row in rows:
        key = (row["scenario"], row["mode"], int(row["repeat"]))
        by_key[key] = row

    paired = []
    for scenario in SCENARIOS:
        repeats = sorted(
            int(row["repeat"])
            for row in rows
            if row.get("scenario") == scenario and row.get("mode") == "ukf"
        )
        for repeat in repeats:
            ukf = by_key.get((scenario, "ukf", repeat))
            hybrid = by_key.get((scenario, "hybrid_residual_b", repeat))
            if ukf is None or hybrid is None:
                continue
            delta = float(ukf["final_ler"]) - float(hybrid["final_ler"])
            relative = delta / float(ukf["final_ler"]) * 100.0
            paired.append({
                "scenario": scenario,
                "repeat": repeat,
                "seed": int(ukf["seed"]),
                "ukf_final_ler": float(ukf["final_ler"]),
                "hybrid_final_ler": float(hybrid["final_ler"]),
                "delta_final_ler_ukf_minus_hybrid": delta,
                "relative_reduction_percent": relative,
                "ukf_run_dir": ukf["run_dir"],
                "hybrid_run_dir": hybrid["run_dir"],
            })
    return paired


def read_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def add_box(ax, xy, width, height, text, fc, ec=COLORS["ink"], dashed=False):
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        facecolor=fc,
        edgecolor=ec,
        linewidth=1.4,
        linestyle="--" if dashed else "-",
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=8.5,
        color=COLORS["ink"],
        wrap=True,
    )


def add_arrow(ax, start, end, color=COLORS["muted"], dashed=False):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.2,
            linestyle="--" if dashed else "-",
            color=color,
        )
    )


def fig01_architecture():
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.set_xlim(0, 10.4)
    ax.set_ylim(0, 6.4)
    ax.axis("off")

    ax.text(0.2, 6.05, "Runtime-consistent affine calibration", fontsize=12, weight="bold")
    ax.text(
        0.2,
        5.78,
        "Slow-loop estimation updates low-dimensional parameters; the per-shot path remains deterministic.",
        fontsize=9,
        color=COLORS["muted"],
    )

    add_box(ax, (0.35, 3.45), 1.55, 0.85, "Analog GKP\nsyndromes", "#EAF2FB")
    add_box(ax, (2.25, 3.45), 1.85, 0.85, "Recent syndrome\nhistogram", "#EAF2FB")
    add_box(ax, (4.45, 4.40), 2.05, 0.9, "Slow estimator\nteacher residual or\nstatistical rule", "#EAF7F3")
    add_box(ax, (7.05, 4.40), 2.0, 0.9, "Staged parameter\nbank (K,b)", "#FFF3E8")
    add_box(ax, (4.35, 1.95), 2.15, 0.9, "Fast path\nDelta = K s + b", "#F0ECFA")
    add_box(ax, (7.05, 1.95), 1.8, 0.9, "Displacement\ncorrection", "#F0ECFA")
    add_box(ax, (7.05, 0.45), 2.45, 0.85, "Planned timing,\nresource and fixed-point\nmeasurements", "#FFFFFF", dashed=True)

    add_arrow(ax, (1.9, 3.88), (2.25, 3.88))
    add_arrow(ax, (4.1, 3.88), (4.78, 4.40))
    add_arrow(ax, (6.5, 4.85), (7.05, 4.85))
    add_arrow(ax, (7.95, 4.40), (5.55, 2.85), COLORS["orange"])
    add_arrow(ax, (1.1, 3.45), (4.35, 2.4), COLORS["purple"])
    add_arrow(ax, (6.5, 2.4), (7.05, 2.4))
    add_arrow(ax, (7.95, 1.95), (7.95, 1.3), dashed=True)

    ax.text(4.95, 3.45, "slow loop", color=COLORS["teal"], fontsize=9, weight="bold")
    ax.text(4.75, 1.50, "latency-critical fast loop", color=COLORS["purple"], fontsize=9, weight="bold")
    save(fig, "fig01_dual_loop_architecture")


def fig02_main_results():
    main_results, main_std = load_t24_main_results()
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    x = np.arange(len(SCENARIOS))
    markers = ["o", "s", "^", "D", "P"]
    palette = [COLORS["gray"], COLORS["blue"], COLORS["muted"], COLORS["orange"], COLORS["green"]]
    for (mode, values), marker, color in zip(main_results.items(), markers, palette):
        lw = 2.4 if mode == "Hybrid-b" else 1.4
        ax.errorbar(
            x,
            values,
            yerr=main_std[mode],
            marker=marker,
            linewidth=lw,
            color=color,
            label=mode,
            capsize=3,
            elinewidth=0.9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(["static\nbias", "linear\nramp", "step\nsigma", "periodic\ndrift"])
    ax.set_ylabel("final_ler_mean (lower is better)")
    ax.set_title("Predeclared simulation ranking", loc="left", fontsize=12, weight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=3, frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig.subplots_adjust(bottom=0.25)
    save(fig, "fig02_main_software_hil_results")


def fig03_ablation_mechanism():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.2, 4.4), gridspec_kw={"width_ratios": [1.05, 1]})

    labels = [row[0] for row in ABLATION]
    deltas = [row[2] for row in ABLATION]
    y = np.arange(len(labels))
    colors = [COLORS["orange"] if d > 0 else COLORS["teal"] for d in deltas]
    ax1.barh(y, deltas, color=colors, alpha=0.88)
    ax1.axvline(0, color=COLORS["ink"], linewidth=0.8)
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels, fontsize=8)
    ax1.invert_yaxis()
    ax1.set_xlabel("Delta vs UKF average LER")
    ax1.set_title("Feature ablation", loc="left", fontsize=11, weight="bold")
    ax1.grid(axis="x", alpha=0.25)

    seeds = [row[0][-4:] for row in MECHANISM]
    effects = [row[2] for row in MECHANISM]
    x = np.arange(len(seeds))
    colors = [COLORS["orange"] if e > 0 else COLORS["teal"] for e in effects]
    ax2.bar(x, effects, color=colors, alpha=0.88)
    ax2.axhline(0, color=COLORS["ink"], linewidth=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(seeds, rotation=45, ha="right")
    ax2.set_ylabel("I1 - gated v5")
    ax2.set_title("Lower-clip intervention", loc="left", fontsize=11, weight="bold")
    ax2.grid(axis="y", alpha=0.25)
    ax2.text(
        0.02,
        0.04,
        "Mostly harmful or mixed; descriptive only.",
        transform=ax2.transAxes,
        fontsize=8,
        color=COLORS["muted"],
    )
    fig.suptitle("Mechanism evidence constrains the story without causal closure", x=0.08, ha="left", fontsize=12, weight="bold")
    save(fig, "fig03_ablation_and_mechanism")


def fig04_statcalib():
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    x = np.arange(len(SCENARIOS))
    width = 0.23
    colors = [COLORS["blue"], COLORS["green"], COLORS["purple"]]
    for offset, (mode, values), color in zip([-width, 0, width], STATCALIB.items(), colors):
        ax.bar(x + offset, values, width=width, label=mode, color=color, alpha=0.88)

    ax.set_xticks(x)
    ax.set_xticklabels(["static\nbias", "linear\nramp", "step\nsigma", "periodic\ndrift"])
    ax.set_ylabel("final_ler_mean (lower is better)")
    ax.set_title("Supplementary statistical calibration", loc="left", fontsize=12, weight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3)
    fig.subplots_adjust(bottom=0.25)
    save(fig, "fig04_statcalib_extension_lane")


def fig05_validation_contract():
    cost_rows = read_rows(COST_MODEL_CSV)
    fixed_rows = read_rows(FIXED_POINT_CSV)
    runtime_rows = read_rows(RUNTIME_DISCIPLINE_CSV)

    fig, axes = plt.subplots(1, 3, figsize=(11.3, 3.75), gridspec_kw={"width_ratios": [1.12, 1.05, 1.0]})
    ax1, ax2, ax3 = axes

    label_map = {
        "Affine fast path": "Affine\nfast path",
        "Wrapped MAP, 3x3 branches": "Wrapped\nMAP",
        "Wrapped posterior mean, 3x3 branches": "Wrapped\nmean",
    }
    labels = [label_map.get(row["table_label"], row["table_label"]) for row in cost_rows]
    mults = [float(row["multiplications_per_shot"]) for row in cost_rows]
    adds = [float(row["additions_per_shot"]) for row in cost_rows]
    x = np.arange(len(labels))
    ax1.bar(x - 0.18, mults, width=0.36, color=COLORS["blue"], label="Mult.")
    ax1.bar(x + 0.18, adds, width=0.36, color=COLORS["teal"], label="Add.")
    ax1.set_yscale("log")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=7)
    ax1.set_ylabel("Ops per shot (log)")
    ax1.set_title("Fast-path cost", loc="left", fontsize=10.5, weight="bold")
    ax1.grid(axis="y", alpha=0.22)
    ax1.legend(frameon=False, fontsize=7, loc="upper left")

    scenarios = [row["scenario"].replace("_", "\n") for row in fixed_rows]
    p99 = [float(row["p99_abs_correction_diff"]) for row in fixed_rows]
    crossing_delta = [float(row["boundary_crossing_delta_fixed_minus_float"]) for row in fixed_rows]
    x2 = np.arange(len(scenarios))
    ax2.bar(x2, p99, color=COLORS["purple"], alpha=0.86)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(scenarios, fontsize=7)
    ax2.set_ylabel("p99 abs. correction diff")
    ax2.set_title("Q4.20 software parity", loc="left", fontsize=10.5, weight="bold")
    ax2.grid(axis="y", alpha=0.22)
    ax2.text(
        0.02,
        0.92,
        f"Crossing-delta max: {max(abs(v) for v in crossing_delta):.1e}",
        transform=ax2.transAxes,
        fontsize=7.5,
        color=COLORS["muted"],
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.5},
    )

    modes = [row["mode_label"].replace(" Residual-", "\nResidual-") for row in runtime_rows]
    overflows = [float(row["overflow_rate_mean"]) for row in runtime_rows]
    fast_viol = [float(row["fast_cycle_violation_rate_mean"]) for row in runtime_rows]
    x3 = np.arange(len(modes))
    ax3.bar(x3 - 0.18, overflows, width=0.36, color=COLORS["orange"], label="Overflow")
    ax3.bar(x3 + 0.18, fast_viol, width=0.36, color=COLORS["gray"], label="Fast viol.")
    ax3.set_xticks(x3)
    ax3.set_xticklabels(modes, rotation=45, ha="right", fontsize=7)
    ax3.set_ylabel("Mean rate")
    ax3.set_title("Runtime counters", loc="left", fontsize=10.5, weight="bold")
    ax3.grid(axis="y", alpha=0.22)
    ax3.legend(frameon=False, fontsize=7, loc="upper left")

    fig.suptitle(
        "Software validation contract: low arithmetic, fixed-point parity and observable counters",
        x=0.06,
        ha="left",
        fontsize=12,
        weight="bold",
    )
    fig.subplots_adjust(top=0.80, bottom=0.34, wspace=0.42)
    save(fig, "fig05_validation_contract")


def write_sources():
    main_results, main_std = load_t24_main_results()
    paired_deltas = load_t24_paired_deltas()
    with (ROOT / "source_data_fig02_main_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["scenario", "mode", "final_ler_mean", "final_ler_sd", "n_repeats", "source"])
        for scenario_idx, scenario in enumerate(SCENARIOS):
            for mode in main_results:
                writer.writerow([
                    scenario,
                    mode,
                    f"{main_results[mode][scenario_idx]:.12f}",
                    f"{main_std[mode][scenario_idx]:.12f}",
                    2,
                    "runs/p4_benchmark/T24_formal_software_revalidation_20260510_200743/comparison.csv",
                ])

    with (ROOT / "source_data_fig02_paired_deltas.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "scenario",
            "repeat",
            "seed",
            "ukf_final_ler",
            "hybrid_final_ler",
            "delta_final_ler_ukf_minus_hybrid",
            "relative_reduction_percent",
            "ukf_run_dir",
            "hybrid_run_dir",
            "boundary",
        ])
        writer.writeheader()
        for row in paired_deltas:
            writer.writerow({
                "scenario": row["scenario"],
                "repeat": row["repeat"],
                "seed": row["seed"],
                "ukf_final_ler": f"{row['ukf_final_ler']:.12f}",
                "hybrid_final_ler": f"{row['hybrid_final_ler']:.12f}",
                "delta_final_ler_ukf_minus_hybrid": f"{row['delta_final_ler_ukf_minus_hybrid']:.12f}",
                "relative_reduction_percent": f"{row['relative_reduction_percent']:.6f}",
                "ukf_run_dir": row["ukf_run_dir"],
                "hybrid_run_dir": row["hybrid_run_dir"],
                "boundary": "paired descriptive delta only; n=2 is not an inferential confidence interval or hypothesis test",
            })

    with (ROOT / "source_data_fig03_ablation_mechanism.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["panel", "label", "value_a", "value_b", "boundary"])
        for label, avg_ler, delta in ABLATION:
            writer.writerow(["ablation", label, f"{avg_ler:.6f}", f"{delta:.6f}", "feature sensitivity only"])
        for seed, gated_minus_full, i1_minus_gated in MECHANISM:
            writer.writerow(["mechanism", seed, f"{gated_minus_full:.6f}", f"{i1_minus_gated:.6f}", "descriptive intervention only"])

    with (ROOT / "source_data_fig04_statcalib.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["scenario", "mode", "final_ler_mean", "boundary"])
        for scenario_idx, scenario in enumerate(SCENARIOS):
            for mode, values in STATCALIB.items():
                writer.writerow([scenario, mode, f"{values[scenario_idx]:.6f}", "supplementary analysis only"])

    with (ROOT / "source_data_fig05_validation_contract.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["panel", "label", "metric", "value", "source", "boundary"])
        for row in read_rows(COST_MODEL_CSV):
            writer.writerow([
                "fast_path_cost",
                row["table_label"],
                "multiplications_per_shot",
                row["multiplications_per_shot"],
                "docs/paper_materials/submission_draft_fast_path_cost_model.csv",
                "analytical count only; not FPGA synthesis or timing closure",
            ])
            writer.writerow([
                "fast_path_cost",
                row["table_label"],
                "additions_per_shot",
                row["additions_per_shot"],
                "docs/paper_materials/submission_draft_fast_path_cost_model.csv",
                "analytical count only; not FPGA synthesis or timing closure",
            ])
        for row in read_rows(FIXED_POINT_CSV):
            writer.writerow([
                "fixed_point_parity",
                row["scenario"],
                "p99_abs_correction_diff",
                row["p99_abs_correction_diff"],
                "docs/paper_materials/submission_draft_fixed_point_parity_analysis.csv",
                "software-emulation parity only; not source-vs-board agreement",
            ])
            writer.writerow([
                "fixed_point_parity",
                row["scenario"],
                "boundary_crossing_delta_fixed_minus_float",
                row["boundary_crossing_delta_fixed_minus_float"],
                "docs/paper_materials/submission_draft_fixed_point_parity_analysis.csv",
                "software-emulation parity only; not source-vs-board agreement",
            ])
        for row in read_rows(RUNTIME_DISCIPLINE_CSV):
            writer.writerow([
                "runtime_counters",
                row["mode_label"],
                "overflow_rate_mean",
                row["overflow_rate_mean"],
                "docs/paper_materials/submission_draft_runtime_discipline_summary.csv",
                "software-in-the-loop counters only; not board commit latency",
            ])
            writer.writerow([
                "runtime_counters",
                row["mode_label"],
                "fast_cycle_violation_rate_mean",
                row["fast_cycle_violation_rate_mean"],
                "docs/paper_materials/submission_draft_runtime_discipline_summary.csv",
                "software-in-the-loop counters only; not board commit latency",
            ])

    with (ROOT / "figure_source_map.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["asset", "numeric_source", "boundary"])
        writer.writerow([
            "fig01_dual_loop_architecture",
            "Manuscript method contract and validation-scope notes",
            "Schematic only; not a hardware result.",
        ])
        writer.writerow([
            "fig02_main_software_hil_results",
            "runs/p4_benchmark/T24_formal_software_revalidation_20260510_200743/comparison.csv and summary.json; source_data_fig02_main_results.csv; source_data_fig02_paired_deltas.csv",
            "Predeclared four-scenario simulation means with descriptive SD and paired repeat deltas only; n=2 is not an inferential CI or expanded drift claim.",
        ])
        writer.writerow([
            "fig03_ablation_and_mechanism",
            "docs/paper_notes/CNN_FPGA_GKP_submission_draft.tex Tables ablation and mechanism",
            "Feature sensitivity and descriptive intervention only; no causal closure.",
        ])
        writer.writerow([
            "fig04_statcalib_extension_lane",
            "docs/paper_notes/CNN_FPGA_GKP_submission_draft.tex Table statcalib",
            "Supplementary analysis only; not a mature main comparator.",
        ])
        writer.writerow([
            "fig05_validation_contract",
            "docs/paper_materials/submission_draft_fast_path_cost_model.csv; docs/paper_materials/submission_draft_fixed_point_parity_analysis.csv; docs/paper_materials/submission_draft_runtime_discipline_summary.csv; source_data_fig05_validation_contract.csv",
            "Software validation-contract summary only; not FPGA synthesis, timing closure, resource/power measurement, source-vs-board agreement or board commit latency.",
        ])

    manifest = {
        "package": "submission_draft_python_figures",
        "generated_by": "build_submission_draft_figures.py",
        "backend": "python/matplotlib",
        "outputs": [
            "outputs/fig01_dual_loop_architecture.pdf",
            "outputs/fig02_main_software_hil_results.pdf",
            "outputs/fig03_ablation_and_mechanism.pdf",
            "outputs/fig04_statcalib_extension_lane.pdf",
            "outputs/fig05_validation_contract.pdf",
        ],
        "source_data": [
            "source_data_fig02_main_results.csv",
            "source_data_fig02_paired_deltas.csv",
            "source_data_fig03_ablation_mechanism.csv",
            "source_data_fig04_statcalib.csv",
            "source_data_fig05_validation_contract.csv",
        ],
        "global_boundary": "No new experiments are run; figures visualize existing manuscript tables, existing paper-material CSV summaries or schematic contracts only.",
    }
    (ROOT / "figure_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig01_architecture()
    fig02_main_results()
    fig03_ablation_mechanism()
    fig04_statcalib()
    fig05_validation_contract()
    write_sources()


if __name__ == "__main__":
    main()
