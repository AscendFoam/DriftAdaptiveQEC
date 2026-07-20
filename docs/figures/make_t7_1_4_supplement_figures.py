"""Render the evidence-bounded T7.1.4 Supplement figure bundle."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cnn_fpga.benchmark import supplement_figure_contract as contract  # noqa: E402


FIGURE_DIR = contract.FIGURE_DIR
MM = 1.0 / 25.4

COLORS = {
    "blue": "#2B6CB0", "orange": "#DD6B20", "green": "#2F855A", "red": "#C53030",
    "purple": "#6B46C1", "teal": "#2C7A7B", "gray": "#718096", "black": "#1A202C",
    "pale_blue": "#EBF8FF", "pale_red": "#FFF5F5", "pale_green": "#F0FFF4", "pale_gray": "#F7FAFC",
}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _records(report: Mapping[str, Any], category: str) -> list[dict[str, Any]]:
    return [dict(row) for row in report["records"] if row["category"] == category]


def _style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7.2, "axes.titlesize": 8.2,
        "axes.labelsize": 7.2, "xtick.labelsize": 6.4, "ytick.labelsize": 6.4,
        "legend.fontsize": 5.8, "axes.linewidth": 0.7, "lines.linewidth": 1.25,
        "pdf.fonttype": 42, "svg.fonttype": "none", "savefig.facecolor": "white",
    })


def _panel(ax: plt.Axes, letter: str, title: str) -> None:
    ax.set_title(title, loc="left", pad=5, fontweight="bold")
    ax.text(-0.13, 1.04, letter, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")
    ax.grid(True, axis="y", color="#E2E8F0", linewidth=0.55, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)


def _boundary(fig: plt.Figure, text: str) -> None:
    fig.text(0.5, 0.013, text, ha="center", va="bottom", fontsize=6.2, color=COLORS["red"],
             bbox={"boxstyle": "round,pad=0.25", "facecolor": COLORS["pale_red"], "edgecolor": "#FEB2B2", "linewidth": 0.6})


def _save(fig: plt.Figure, stem: str) -> dict[str, Path]:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    paths = {suffix: FIGURE_DIR / f"{stem}.{suffix}" for suffix in ("svg", "pdf", "png", "tiff")}
    fig.savefig(paths["svg"])
    fig.savefig(paths["pdf"])
    fig.savefig(paths["png"], dpi=300)
    fig.savefig(paths["tiff"], dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    text = paths["svg"].read_text(encoding="utf-8")
    paths["svg"].write_text("\n".join(line.rstrip() for line in text.splitlines()) + "\n", encoding="utf-8")
    return paths


def _figure_s1(report: Mapping[str, Any]) -> dict[str, Path]:
    fig, axes = plt.subplots(2, 2, figsize=(183 * MM, 135 * MM))
    fig.suptitle("Supplement S1 | Numerical validation and explicit model-validity domains", fontsize=10.2, fontweight="bold", y=0.982)

    ax = axes[0, 0]
    rows = sorted(_records(report, "gradient"), key=lambda row: row["metadata"]["step"], reverse=True)
    x = np.array([row["metadata"]["step"] for row in rows])
    ax.loglog(x, [row["value"] for row in rows], "o-", label="total gradient", color=COLORS["blue"])
    ax.loglog(x, [row["metadata"]["reward_path_error"] for row in rows], "s--", label="reward path", color=COLORS["orange"])
    ax.loglog(x, [row["metadata"]["score_path_error"] for row in rows], "^:", label="score path", color=COLORS["green"])
    ax.set_xlabel("finite-difference step"); ax.set_ylabel("relative L2 error")
    ax.legend(frameon=False, ncol=1, loc="lower left")
    ax.text(0.98, 0.08, "exact decomposition error = 5.55e-17", transform=ax.transAxes, ha="right", va="bottom", fontsize=5.9, color=COLORS["gray"])
    _panel(ax, "a", "Feedback-GRAPE gradient decomposition")

    ax = axes[0, 1]
    rows = _records(report, "cutoff_feasibility")
    passed = [row for row in rows if row["status"] == "PASS"]
    failures = [row for row in rows if row["status"] != "PASS"]
    scatter = ax.scatter([row["metadata"]["cutoff"] for row in passed], [max(float(row["value"]), 1e-3) for row in passed],
                         c=[row["metadata"]["full_cycles"] for row in passed], cmap="viridis", s=[10 + 4 * math.log2(max(1, row["metadata"]["batch_size"])) for row in passed],
                         alpha=0.72, edgecolors="white", linewidths=0.3)
    for row in failures:
        ax.scatter(row["metadata"]["cutoff"], 10.8, marker="X", s=52, color=COLORS["red"], edgecolors="white", linewidths=0.5, zorder=5)
        align_right = row["metadata"]["cutoff"] > 40
        ax.annotate(row["status"].replace("_", " ").lower(), (row["metadata"]["cutoff"], 10.8),
                    xytext=(-4 if align_right else 3, -8), textcoords="offset points", ha="right" if align_right else "left",
                    fontsize=5.6, color=COLORS["red"])
    ax.axhline(10.0, color=COLORS["red"], linestyle="--", linewidth=0.9, label="runtime budget")
    ax.set_yscale("log"); ax.set_xlabel("Fock cutoff"); ax.set_ylabel("forward/backward/Adam median (s)")
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.047, pad=0.02); cbar.set_label("full cycles", fontsize=6.2); cbar.ax.tick_params(labelsize=5.8)
    ax.text(0.02, 0.05, "marker size ∝ log2(batch)", transform=ax.transAxes, va="bottom", fontsize=5.8, color=COLORS["gray"])
    _panel(ax, "b", "65-point host-specific feasibility envelope")

    ax = axes[1, 0]
    rows = sorted(_records(report, "noise_transfer_domain"), key=lambda row: row["metadata"]["squeezing_db"])
    sq = [row["metadata"]["squeezing_db"] for row in rows]
    ax.axvspan(2.5, 8.5, color=COLORS["pale_red"], zorder=0); ax.axvspan(9.5, 12.5, color=COLORS["pale_green"], zorder=0)
    ax.plot(sq, [row["metadata"]["central_probability"] for row in rows], "o-", color=COLORS["blue"], label="central probability")
    ax.plot(sq, [row["value"] for row in rows], "s--", color=COLORS["red"], label="odd-alias probability")
    ax.plot(sq, [row["metadata"]["clipping_ratio"] for row in rows], "^:", color=COLORS["purple"], label="clipping ratio")
    ax.set_ylim(-0.02, 1.04); ax.set_xticks(sq); ax.set_xlabel("squeezing (dB)"); ax.set_ylabel("probability / ratio")
    ax.legend(frameon=False, ncol=1, loc="center right")
    ax.text(0.02, 0.07, "failure domain", transform=ax.transAxes, color=COLORS["red"], fontsize=6.1, fontweight="bold")
    ax.text(0.73, 0.07, "localized", transform=ax.transAxes, color=COLORS["green"], fontsize=6.1, fontweight="bold")
    _panel(ax, "c", "Noise-transfer valid and failure domains")

    ax = axes[1, 1]
    rows = sorted(_records(report, "noise_transfer_alignment"), key=lambda row: row["metadata"]["squeezing_db"])
    sq = np.array([row["metadata"]["squeezing_db"] for row in rows])
    ax.semilogy(sq, [row["value"] for row in rows], "o-", color=COLORS["orange"], label="proxy vs direct")
    ax.semilogy(sq, [row["metadata"]["fock_to_direct_relative_error"] for row in rows], "s--", color=COLORS["blue"], label="Fock vs direct")
    ax.semilogy(sq, [row["metadata"]["state_relative_spread"] for row in rows], "^:", color=COLORS["purple"], label="direct state spread")
    ax.set_xticks(sq); ax.set_xlabel("squeezing (dB)"); ax.set_ylabel("relative error / spread")
    ax.legend(frameon=False, loc="best")
    _panel(ax, "d", "Direct–Fock–proxy alignment")

    fig.tight_layout(rect=(0.025, 0.055, 0.995, 0.945), h_pad=1.8, w_pad=1.3)
    _boundary(fig, "Analytic/host validation only • finite cutoff is not physical convergence • no FPGA timing claim")
    return _save(fig, "supplement_s1_physics_validity")


def _figure_s2(report: Mapping[str, Any]) -> dict[str, Path]:
    fig, axes = plt.subplots(2, 2, figsize=(183 * MM, 150 * MM))
    fig.suptitle("Supplement S2 | Recovery bounds, truncated MAP and six-state channel evidence", fontsize=10.0, fontweight="bold", y=0.982)
    noise_colors = {"high": COLORS["red"], "medium": COLORS["orange"], "low": COLORS["green"]}

    ax = axes[0, 0]
    rows = sorted(_records(report, "petz_small_sdp"), key=lambda row: (row["metadata"]["cutoff"], row["family"]))
    for noise in ("high", "medium", "low"):
        selected = [row for row in rows if row["family"] == noise]
        cutoffs = [row["metadata"]["cutoff"] for row in selected]
        petz_inf = [max(1e-12, 1 - row["value"]) for row in selected]
        sdp_inf = [max(1e-12, 1 - row["lower"]) for row in selected]
        ax.semilogy(cutoffs, petz_inf, "o-", color=noise_colors[noise], label=f"{noise}: Petz")
        ax.semilogy(cutoffs, sdp_inf, "x--", color=noise_colors[noise], alpha=0.85, label=f"{noise}: SDP lower")
    ax.set_xlabel("small cutoff"); ax.set_ylabel("entanglement infidelity")
    ax.legend(frameon=False, ncol=2, columnspacing=0.8, handlelength=1.5)
    ax.text(0.02, 0.04, "max certificate width 1.48e-7", transform=ax.transAxes, fontsize=5.9, color=COLORS["gray"])
    _panel(ax, "a", "Petz versus certified small-cutoff SDP")

    ax = axes[0, 1]
    rows = _records(report, "petz_cutoff_extension")
    for noise in ("high", "medium", "low"):
        selected = sorted([row for row in rows if row["family"] == noise], key=lambda row: row["metadata"]["cutoff"])
        ax.semilogy([row["metadata"]["cutoff"] for row in selected], [row["value"] for row in selected], "o-", color=noise_colors[noise], label=noise)
    ax.set_xlabel("cutoff"); ax.set_ylabel("Petz infidelity"); ax.legend(frameon=False, title="noise")
    ax.text(0.02, 0.04, "no SDP above small-cutoff validation\nnot infinite-dimensional convergence", transform=ax.transAxes, fontsize=5.9, color=COLORS["red"])
    _panel(ax, "b", "Cutoff extension of a nondeployable bound")

    ax = axes[1, 0]
    rows = _records(report, "topk_pareto")
    scenario_colors = plt.cm.tab10(np.linspace(0, 1, 6))
    for color, scenario in zip(scenario_colors, sorted({row["family"] for row in rows})):
        selected = sorted([row for row in rows if row["family"] == scenario], key=lambda row: row["metadata"]["K"])
        ax.loglog([row["metadata"]["retained_state_bits"] for row in selected], [max(float(row["value"]), 1e-16) for row in selected], "o-", markersize=3, color=color, label=scenario.replace("_", " "))
    ax.set_xlabel("retained-state bits (software cost proxy)"); ax.set_ylabel("p99 |LLR error|")
    ax.legend(frameon=False, fontsize=5.0, ncol=2, columnspacing=0.6, handlelength=1.2)
    ax.text(0.02, 0.04, "K=1…128 • convergence K=2–4\nno LUT/FF/BRAM/Fmax measurement", transform=ax.transAxes, fontsize=5.7, color=COLORS["red"])
    _panel(ax, "c", "Top-K lattice-coset accuracy–cost sensitivity")

    ax = axes[1, 1]
    rows = _records(report, "six_pauli_states")
    states = ["x_plus", "x_minus", "y_plus", "y_minus", "z_plus", "z_minus"]
    state_colors = dict(zip(states, [COLORS["blue"], "#63B3ED", COLORS["purple"], "#B794F4", COLORS["green"], "#68D391"]))
    for mode, linestyle, alpha in (("qec_off", "--", 0.72), ("qec_on", "-", 0.95)):
        for state in states:
            selected = sorted([row for row in rows if row["method"] == mode and row["family"] == state], key=lambda row: row["metadata"]["cycle"])
            ax.plot([row["metadata"]["cycle"] for row in selected], [row["value"] for row in selected], linestyle=linestyle, color=state_colors[state], alpha=alpha, linewidth=1.05)
    ax.set_xlabel("QEC cycle"); ax.set_ylabel("raw code-subspace survival"); ax.set_ylim(-0.02, 1.02)
    state_labels = {"x_plus": "X+", "x_minus": "X−", "y_plus": "Y+", "y_minus": "Y−", "z_plus": "Z+", "z_minus": "Z−"}
    state_handles = [Line2D([0], [0], color=state_colors[state], label=state_labels[state], lw=1.6) for state in states]
    mode_handles = [Line2D([0], [0], color=COLORS["black"], linestyle="-", label="QEC on"), Line2D([0], [0], color=COLORS["black"], linestyle="--", label="QEC off")]
    legend1 = ax.legend(handles=state_handles, frameon=False, ncol=3, loc="upper right", fontsize=4.9, handlelength=1.3, columnspacing=0.5)
    ax.add_artist(legend1); ax.legend(handles=mode_handles, frameon=False, loc="lower left", fontsize=5.4)
    ax.text(0.98, 0.04, "cutoff 40 • high noise\nCPTNI • no post-selection", transform=ax.transAxes, ha="right", fontsize=5.7, color=COLORS["gray"])
    _panel(ax, "d", "All six Pauli eigenstates, QEC on/off")

    fig.tight_layout(rect=(0.025, 0.055, 0.995, 0.945), h_pad=1.8, w_pad=1.3)
    _boundary(fig, "Petz/SDP = arbitrary terminal-recovery bound • top-K = unsynthesized proxy • six-state curves = finite-cutoff simulation")
    return _save(fig, "supplement_s2_bounds_maps_channel")


def _figure_s3(report: Mapping[str, Any]) -> dict[str, Path]:
    fig, axes = plt.subplots(2, 2, figsize=(183 * MM, 145 * MM))
    fig.suptitle("Supplement S3 | All formal seeds and complete registered OOD lanes", fontsize=10.1, fontweight="bold", y=0.982)

    ax = axes[0, 0]
    rows = _records(report, "all_seed_distribution")
    methods = ["standard_binning", "static_joint_map", "window_map", "ewma_adaptive_map", "kalman_adaptive_map", "proposed_route_a", "hidden_state_oracle"]
    values = [[row["value"] for row in rows if row["method"] == method] for method in methods]
    bp = ax.boxplot(values, patch_artist=True, widths=0.58, showfliers=False, medianprops={"color": COLORS["black"]})
    palette = [COLORS["gray"], COLORS["orange"], COLORS["green"], COLORS["blue"], COLORS["purple"], COLORS["red"], "#CBD5E0"]
    for patch, color in zip(bp["boxes"], palette): patch.set_facecolor(color); patch.set_alpha(0.55)
    for index, method_values in enumerate(values, start=1):
        offsets = np.linspace(-0.18, 0.18, len(method_values))
        ax.scatter(index + offsets, method_values, s=7, color=palette[index - 1], edgecolors="white", linewidths=0.2, zorder=3)
    ax.set_xticks(range(1, len(methods) + 1), ["Std", "Static", "Window", "EWMA", "Kalman", "Route-A", "Oracle"], rotation=25, ha="right")
    ax.set_ylabel("per-seed aggregate LER"); ax.text(0.02, 0.04, "24 formal seeds × all 7 methods", transform=ax.transAxes, fontsize=5.9, color=COLORS["gray"])
    _panel(ax, "a", "No seed or method is hidden")

    ax = axes[0, 1]
    rows = _records(report, "ood_drift")
    scenarios = sorted({row["family"] for row in rows})
    method_colors = {method: color for method, color in zip(sorted({row["method"] for row in rows}), plt.cm.tab10(np.linspace(0, 1, 6)))}
    for method, color in method_colors.items():
        selected = [next(row for row in rows if row["family"] == scenario and row["method"] == method) for scenario in scenarios]
        ax.errorbar(range(len(scenarios)), [row["value"] for row in selected], yerr=[[row["value"] - row["lower"] for row in selected], [row["upper"] - row["value"] for row in selected]], marker="o", markersize=3.2, linewidth=1.0, capsize=1.5, color=color, label=method)
    ax.set_xticks(range(len(scenarios)), [value.replace("_", "\n") for value in scenarios], rotation=0)
    ax.set_ylabel("logical error rate"); ax.legend(frameon=False, ncol=2, fontsize=4.8, columnspacing=0.5, handlelength=1.1)
    _panel(ax, "b", "Held-out drift families with seed-cluster CIs")

    ax = axes[1, 0]
    confusion = sorted(_records(report, "ood_measurement"), key=lambda row: row["family"])
    leakage = sorted(_records(report, "ood_leakage"), key=lambda row: row["metadata"]["intervention_rate"])
    labels = [row["family"].replace("_confusion", "") for row in confusion] + [f"leak {row['metadata']['intervention_rate']:.3g}" for row in leakage]
    values = [row["value"] for row in confusion + leakage]
    colors = [COLORS["orange"]] * len(confusion) + [COLORS["red"]] * len(leakage)
    bars = ax.bar(range(len(values)), values, color=colors, alpha=0.75, zorder=2)
    for index, row in enumerate(confusion + leakage):
        if row["lower"] is not None:
            ax.errorbar(index, row["value"], yerr=[[row["value"] - row["lower"]], [row["upper"] - row["value"]]], color=COLORS["black"], capsize=2, linewidth=0.7)
    ax.set_xticks(range(len(labels)), labels, rotation=28, ha="right"); ax.set_ylabel("error / unsafe-availability fraction")
    ax.legend([bars[0], bars[-1]], ["measurement confusion", "leakage/reset"], frameon=False, fontsize=5.5)
    _panel(ax, "c", "Measurement + leakage OOD")

    ax = axes[1, 1]
    rows = _records(report, "ood_communication")
    scenarios = ["reference", "periodic_micro_outages", "increasing_duration_flaps", "communication_jitter_burst_compound"]
    for metric, color, marker in (("logical_error_rate", COLORS["red"], "o"), ("end_to_end_control_availability", COLORS["blue"], "s")):
        selected = [next(row for row in rows if row["family"] == scenario and row["metric"] == metric) for scenario in scenarios]
        ax.plot(range(4), [row["value"] for row in selected], marker=marker, color=color, label=metric.replace("_", " "))
        ax.fill_between(range(4), [row["lower"] for row in selected], [row["upper"] for row in selected], color=color, alpha=0.12)
    ax.set_xticks(range(4), ["reference", "micro\noutages", "duration\nflaps", "jitter+burst\ncompound"])
    ax.set_ylim(-0.02, 1.02); ax.set_ylabel("probability"); ax.legend(frameon=False, loc="center left")
    ax.text(0.98, 0.96, "compound: FIFO overflow=128\nsoftware timing model, not board", transform=ax.transAxes, ha="right", va="top", fontsize=5.7, color=COLORS["red"])
    _panel(ax, "d", "Communication faults: severe degradation")

    fig.tight_layout(rect=(0.025, 0.055, 0.995, 0.945), h_pad=1.8, w_pad=1.3)
    _boundary(fig, "Lane-local held-out/OOD evidence • system robustness NOT ESTABLISHED • device robustness NOT ESTABLISHED")
    return _save(fig, "supplement_s3_seeds_and_ood")


def _figure_s4(report: Mapping[str, Any]) -> dict[str, Path]:
    fig, axes = plt.subplots(2, 2, figsize=(183 * MM, 125 * MM))
    fig.suptitle("Supplement S4 | Fixed-point sensitivity and fail-closed boundary ledger", fontsize=10.1, fontweight="bold", y=0.982)

    ax = axes[0, 0]
    rows = [row for row in _records(report, "fixed_point_oat") if row["metadata"]["fault_events"] == 0 and row["family"] != "base"]
    for color, axis in zip(plt.cm.tab10(np.linspace(0, 1, len({row["family"] for row in rows}))), sorted({row["family"] for row in rows})):
        selected = sorted([row for row in rows if row["family"] == axis], key=lambda row: row["metadata"]["axis_value"])
        ax.plot([row["metadata"]["axis_value"] for row in selected], [row["value"] for row in selected], "o-", markersize=3, color=color, label=axis)
    ax.axhline(0, color=COLORS["black"], linewidth=0.7); ax.set_xlabel("axis setting"); ax.set_ylabel("quantized − float LER")
    ax.legend(frameon=False, ncol=2, fontsize=4.8, columnspacing=0.5, handlelength=1.2)
    _panel(ax, "a", "Six-axis fixed-point OAT (8 paired seeds)")

    ax = axes[0, 1]
    rows = _records(report, "fixed_point_production")
    order = ["low_p6_a4_q5_6", "medium_p8_a6_q7_10", "selected_p10_a8_q9_12", "dense_p12_a10_q10_14"]
    selected = [next(row for row in rows if row["method"] == name) for name in order]
    ax.errorbar(range(4), [row["value"] for row in selected], yerr=[[row["value"] - row["lower"] for row in selected], [row["upper"] - row["value"] for row in selected]], fmt="o", color=COLORS["blue"], ecolor=COLORS["gray"], capsize=3)
    ax.axhline(0, color=COLORS["black"], linewidth=0.7); ax.set_xticks(range(4), ["low", "medium", "selected", "dense"], rotation=18)
    ax.set_ylabel("quantized − float LER"); ax.text(0.02, 0.04, "32 clusters/profile • 6-cycle/II=1 software contract", transform=ax.transAxes, fontsize=5.7, color=COLORS["gray"])
    _panel(ax, "b", "Production integer profiles retain float LER")

    ax = axes[1, 0]
    rows = _records(report, "fixed_point_oat")
    x = np.array([max(1, row["metadata"]["storage_bits"]) for row in rows])
    y = np.array([max(1e-8, row["metadata"]["disagreement"]) for row in rows])
    c = np.array([row["value"] for row in rows])
    scatter = ax.scatter(x, y, c=c, cmap="coolwarm", s=[22 if row["metadata"]["fault_events"] == 0 else 42 for row in rows], alpha=0.78, edgecolors="white", linewidths=0.35)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("dual-bank storage bits (representation proxy)"); ax.set_ylabel("prediction disagreement vs float")
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.047, pad=0.02); cbar.set_label("paired LER difference", fontsize=6.0); cbar.ax.tick_params(labelsize=5.5)
    ax.text(0.02, 0.95, "large markers include bank faults\nnot synthesized resources", transform=ax.transAxes, va="top", fontsize=5.7, color=COLORS["red"])
    _panel(ax, "c", "Precision–storage–disagreement map")

    ax = axes[1, 1]; ax.axis("off")
    rows = _records(report, "failure_mode")
    status_color = {"FAILURE_DOMAIN": "#FED7D7", "RESOURCE_BOUNDARY": "#FEEBC8", "NOT_SYNTHESIZED": "#E2E8F0", "INCOMPARABLE": "#E9D8FD", "NOT_ESTABLISHED_LANE_LOCAL_ONLY": "#FED7D7", "NOT_ESTABLISHED": "#FED7D7", "DROPPED": "#E2E8F0", "BLOCKED_ALL_NULL": "#FED7D7"}
    ax.text(0.0, 1.02, "d   Registered failure / non-promotion ledger", transform=ax.transAxes, fontsize=8.2, fontweight="bold", va="bottom")
    short_status = {
        "FAILURE_DOMAIN": "FAILURE DOMAIN", "RESOURCE_BOUNDARY": "RESOURCE BOUNDARY",
        "NOT_SYNTHESIZED": "NOT SYNTHESIZED", "INCOMPARABLE": "INCOMPARABLE",
        "NOT_ESTABLISHED_LANE_LOCAL_ONLY": "NOT ESTABLISHED / LANE-LOCAL",
        "NOT_ESTABLISHED": "NOT ESTABLISHED", "DROPPED": "DROPPED",
        "BLOCKED_ALL_NULL": "BLOCKED / 42 NULL",
    }
    short_name = {
        "low_squeezing_surrogate": "low-squeezing surrogate", "feasibility_memory_boundary": "host memory boundary",
        "feasibility_runtime_boundary": "host runtime boundary", "topk_hardware_unmeasured": "top-K hardware",
        "petz_teacher_student_gap": "Petz teacher/student gap", "ood_system_robustness": "OOD system robustness",
        "route_a_broad_tail_gain": "Route-A broad tail gain", "v5_execution": "V5 execution", "physical_board": "physical board",
    }
    for index, row in enumerate(rows):
        y0 = 0.94 - index * 0.102
        ax.add_patch(plt.Rectangle((0.0, y0 - 0.065), 0.98, 0.082, transform=ax.transAxes, facecolor=status_color.get(row["status"], COLORS["pale_gray"]), edgecolor="white", linewidth=0.5))
        ax.text(0.015, y0, short_name.get(row["method"], row["method"].replace("_", " ")), transform=ax.transAxes, va="center", fontsize=5.6, fontweight="bold")
        ax.text(0.98, y0, short_status.get(row["status"], row["status"].replace("_", " ")), transform=ax.transAxes, va="center", ha="right", fontsize=5.0, color=COLORS["black"])

    fig.tight_layout(rect=(0.025, 0.058, 0.995, 0.94), h_pad=1.5, w_pad=1.2)
    _boundary(fig, "Fixed-point simulation/representation evidence only • V5 dropped • 42 board fields null • no measured speed/power")
    return _save(fig, "supplement_s4_fixed_point_and_failures")


def _manifest(report: Mapping[str, Any], manual_visual_qa: str) -> dict[str, Any]:
    generated_paths: dict[str, Path] = {}
    for maker in (_figure_s1, _figure_s2, _figure_s3, _figure_s4):
        generated_paths.update({path.name: path for path in maker(report).values()})
    output_bindings = {name: contract._binding(path) for name, path in generated_paths.items()}
    for name, source in contract.LINKED_OUTPUTS.items():
        output_bindings[name] = contract._binding(contract.SOURCES[source])
    svg_paths = [ROOT / output_bindings[name]["path"] for name in output_bindings if name.endswith(".svg")]
    tiff_paths = [ROOT / output_bindings[name]["path"] for name in output_bindings if name.endswith(".tiff")]
    text_nodes = sum(path.read_text(encoding="utf-8").count("<text") for path in svg_paths)
    dimensions: dict[str, int] = {}
    for path in tiff_paths:
        with Image.open(path) as image:
            dimensions[path.name] = min(image.size)
    return {
        "task_id": contract.TASK_ID, "backend": "Python/matplotlib only",
        "contract": contract._binding(contract.DEFAULT_REPORT), "source_data": contract._binding(contract.DEFAULT_SOURCE_DATA),
        "renderer": contract._binding(Path(__file__).resolve()), "outputs": output_bindings,
        "qa": {"svg_text_nodes": text_nodes, "svg_path_text_promotion": False, "tiff_min_dimension_px": dimensions, "backend_exclusive": True, "manual_visual_qa": manual_visual_qa},
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manual-visual-qa", choices=("PENDING", "PASS"), default="PENDING")
    args = parser.parse_args(argv)
    _style()
    contract.verify_report()
    report = _load(contract.DEFAULT_REPORT)
    manifest = _manifest(report, args.manual_visual_qa)
    contract._atomic_json(manifest, contract.DEFAULT_MANIFEST)
    if args.manual_visual_qa == "PASS":
        contract.verify_bundle()
    print(json.dumps({"manifest": contract._relative(contract.DEFAULT_MANIFEST), "outputs": len(manifest["outputs"]), "qa": manifest["qa"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
