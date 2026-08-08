"""Generate evidence-bounded figures for the contract-centric Route-A note.

The script reads only repository machine artifacts and source-data CSV files.
It deliberately keeps V4, V5-headroom, auxiliary algorithm, and hardware lanes
in separate panels; no cross-lane score or ranking is computed.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "figures" / "route_a_manuscript_20260720"
OUT.mkdir(parents=True, exist_ok=True)

SOURCES = {
    "v4_smooth": ROOT / "docs" / "t6_7_1_smooth_formal_matrix.json",
    "v4_tail": ROOT / "docs" / "t6_7_2_abrupt_ood_tail_formal_matrix.json",
    "v5_headroom": ROOT / "docs" / "t6_10_1_causal_headroom.json",
    "v5_headroom_csv": ROOT / "docs" / "t6_10_1_causal_headroom_source_data.csv",
    "noh_cnot": ROOT / "docs" / "t6_17_2_noh_cnot_ci_ml_reproduction_source_data.csv",
    "multimode_cpd": ROOT / "docs" / "t6_18_3_multimode_posterior_weighted_cpd.json",
    "preboard": ROOT / "docs" / "t6_19_1_project_preboard_profiles.json",
}

for source in SOURCES.values():
    if not source.exists():
        raise FileNotFoundError(source)

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7.0,
        "axes.titlesize": 8.0,
        "axes.labelsize": 7.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)

COLORS = {
    "route": "#0F4D92",
    "adaptive": "#42949E",
    "window": "#7884B4",
    "static": "#484878",
    "standard": "#B4C0E4",
    "kalman": "#E9A6A1",
    "oracle": "#767676",
    "gain": "#2E9E44",
    "loss": "#B64342",
    "student": "#E4CCD8",
    "neutral": "#CFCECE",
    "gold": "#D69B2D",
}

DERIVED_ROWS: list[dict[str, Any]] = []


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.06,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="top",
        ha="left",
    )


def record(figure: str, panel: str, metric: str, label: str, value: float | int | str, source: str, **extra: Any) -> None:
    row: dict[str, Any] = {
        "figure": figure,
        "panel": panel,
        "metric": metric,
        "label": label,
        "value": value,
        "source": source,
    }
    row.update(extra)
    DERIVED_ROWS.append(row)


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(
        OUT / f"{stem}.tiff",
        dpi=600,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


def figure_v4_evidence_boundary() -> None:
    smooth = load_json(SOURCES["v4_smooth"])
    tail = load_json(SOURCES["v4_tail"])
    summaries = smooth["analysis"]["method_summaries"]
    by_method = {item["method_id"]: item for item in summaries}

    method_order = [
        "standard_binning",
        "static_joint_map",
        "window_map",
        "ewma_adaptive_map",
        "kalman_adaptive_map",
        "proposed_route_a",
        "hidden_state_oracle",
    ]
    method_labels = ["Standard", "Static", "Window", "EWMA", "Kalman", "Route-A", "Oracle"]
    method_colors = [
        COLORS["standard"],
        COLORS["static"],
        COLORS["window"],
        COLORS["adaptive"],
        COLORS["kalman"],
        COLORS["route"],
        COLORS["oracle"],
    ]

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6))
    ax = axes[0, 0]
    values = np.array([by_method[m]["average_ler_equal_family_seed"] for m in method_order]) * 1e3
    ci = np.array([by_method[m]["paired_formal_seed_cluster_ci95"]["p_L"] for m in method_order]) * 1e3
    yerr = np.vstack([values - ci[:, 0], ci[:, 1] - values])
    x = np.arange(len(method_order))
    ax.bar(x, values, color=method_colors, edgecolor="white", linewidth=0.5, yerr=yerr, capsize=2.0)
    ax.set_xticks(x, method_labels, rotation=34, ha="right")
    ax.set_ylabel(r"Smooth aggregate $p_L$ ($\times10^{-3}$)")
    ax.set_title("V4 was not the best deployable smooth decoder", loc="left", fontweight="bold")
    for xpos, value, method in zip(x, values, method_order):
        ax.text(xpos, value + 0.045, f"{value:.3f}", ha="center", va="bottom", fontsize=5.6)
        record(
            "v4_evidence_boundary",
            "a",
            "smooth_equal_family_seed_p_L",
            method,
            value / 1e3,
            "docs/t6_7_1_smooth_formal_matrix.json",
            ci_low=by_method[method]["paired_formal_seed_cluster_ci95"]["p_L"][0],
            ci_high=by_method[method]["paired_formal_seed_cluster_ci95"]["p_L"][1],
            decisions=by_method[method]["decisions"],
        )
    panel_label(ax, "a")

    ax = axes[0, 1]
    family = smooth["analysis"]["per_family_contrasts"]
    fam_labels = ["Mean", "Variance", "Correlation", "Periodic"]
    gain = np.array([item["estimate"] for item in family]) * 1e5
    ci_low = np.array([item["ci95_low"] for item in family]) * 1e5
    ci_high = np.array([item["ci95_high"] for item in family]) * 1e5
    yerr = np.vstack([gain - ci_low, ci_high - gain])
    colors = [COLORS["neutral"], COLORS["neutral"], COLORS["neutral"], COLORS["gain"]]
    ax.errorbar(np.arange(4), gain, yerr=yerr, fmt="none", ecolor="#4D4D4D", capsize=3, lw=1.1, zorder=1)
    ax.scatter(np.arange(4), gain, s=34, c=colors, edgecolor="white", linewidth=0.7, zorder=2)
    ax.axhline(0, color="#767676", lw=0.8, ls="--")
    ax.set_xticks(np.arange(4), fam_labels, rotation=25, ha="right")
    ax.set_ylabel(r"EWMA $-$ Route-A $p_L$ ($\times10^{-5}$)")
    ax.set_title("Only periodic drift survived family-wise testing", loc="left", fontweight="bold")
    for item in family:
        record(
            "v4_evidence_boundary",
            "b",
            "ewma_minus_route_a_p_L",
            item["family"],
            item["estimate"],
            "docs/t6_7_1_smooth_formal_matrix.json",
            ci_low=item["ci95_low"],
            ci_high=item["ci95_high"],
        )
    panel_label(ax, "b")

    ax = axes[1, 0]
    actions = [item for item in tail["analysis"]["action_metrics_by_family"] if item["family"] != "nominal_static"]
    family_labels = ["Step", "Telegraph", "Burst", "Readout/reset", "Leakage", "Compound"]
    y = np.arange(len(actions))
    fallback = np.array([item["fallback_rate"] for item in actions]) * 100
    unnecessary = np.array([item["unnecessary_fallback_rate"] for item in actions]) * 100
    height = 0.34
    ax.barh(y - height / 2, fallback, height=height, color=COLORS["route"], label="Fallback signal")
    ax.barh(y + height / 2, unnecessary, height=height, color=COLORS["neutral"], label="Unnecessary fallback")
    ax.set_yticks(y, family_labels)
    ax.set_xlabel("Decision rate (%)")
    ax.set_xlim(0, 102)
    ax.invert_yaxis()
    ax.legend(loc="upper center", bbox_to_anchor=(0.52, -0.20), ncol=2)
    ax.set_title("Tail non-inferiority required frequent intervention", loc="left", fontweight="bold")
    for item in actions:
        record(
            "v4_evidence_boundary",
            "c",
            "tail_fallback_rate",
            item["family"],
            item["fallback_rate"],
            "docs/t6_7_2_abrupt_ood_tail_formal_matrix.json",
            unnecessary_fallback_rate=item["unnecessary_fallback_rate"],
            false_updates=item["false_updates"],
            commits=item["commits"],
        )
    panel_label(ax, "c")

    ax = axes[1, 1]
    calibration = [
        item
        for item in tail["analysis"]["family_method_summaries"]
        if item["family"] == "step_calibration_shift"
        and item["method_id"] in {"static_joint_map", "window_map", "ewma_adaptive_map", "proposed_route_a"}
    ]
    order = ["static_joint_map", "window_map", "ewma_adaptive_map", "proposed_route_a"]
    cal_by_method = {item["method_id"]: item for item in calibration}
    worst = np.array([cal_by_method[m]["global_worst_window_error_count"] for m in order])
    avg = np.array([cal_by_method[m]["average_ler"] for m in order])
    labels = ["Static", "Window", "EWMA", "Route-A"]
    colors = [COLORS["static"], COLORS["window"], COLORS["adaptive"], COLORS["route"]]
    bars = ax.bar(np.arange(4), worst, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(np.arange(4), labels, rotation=25, ha="right")
    ax.set_ylabel("Worst errors per 512-decision window")
    ax.set_title("Calibration safety matched EWMA, not static MAP", loc="left", fontweight="bold")
    for bar, count, mean in zip(bars, worst, avg):
        ax.text(bar.get_x() + bar.get_width() / 2, count + 4, f"{count:.0f}", ha="center", fontsize=6)
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            max(4, count * 0.40),
            f"mean\n{mean:.4f}",
            ha="center",
            va="center",
            fontsize=5.5,
            color="white" if count > 70 else "black",
        )
    for method in order:
        item = cal_by_method[method]
        record(
            "v4_evidence_boundary",
            "d",
            "calibration_global_worst_window_errors",
            method,
            item["global_worst_window_error_count"],
            "docs/t6_7_2_abrupt_ood_tail_formal_matrix.json",
            average_ler=item["average_ler"],
            window_denominator=512,
        )
    panel_label(ax, "d")

    fig.subplots_adjust(left=0.10, right=0.99, top=0.93, bottom=0.10, hspace=0.44, wspace=0.34)
    save_figure(fig, "v4_evidence_boundary")


def figure_v5_headroom_early_stop() -> None:
    report = load_json(SOURCES["v5_headroom"])
    rows = pd.read_csv(SOURCES["v5_headroom_csv"])
    nested = report["development_audit"]["nested_audit"]
    gates = report["decision_contract"]

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4))

    ax = axes[0, 0]
    formal = rows[rows["section"] == "formal_oracle"].copy()
    names = ["Family", "Cell", "Activation", "Per-decision"]
    values = formal["relative_headroom"].to_numpy(dtype=float) * 100
    ax.bar(np.arange(4), values, color=[COLORS["neutral"], COLORS["neutral"], COLORS["gold"], COLORS["oracle"]])
    ax.set_xticks(np.arange(4), names, rotation=25, ha="right")
    ax.set_ylabel("Truth-privileged headroom (%)")
    ax.set_title("Large oracle headroom required unavailable truth", loc="left", fontweight="bold")
    for xpos, value in enumerate(values):
        ax.text(xpos, value + 0.7, f"{value:.2f}%", ha="center", fontsize=6)
    for _, item in formal.iterrows():
        record(
            "v5_headroom_early_stop",
            "a",
            "formal_truth_oracle_headroom",
            item["item"],
            float(item["relative_headroom"]),
            "docs/t6_10_1_causal_headroom_source_data.csv",
            deployable=False,
        )
    panel_label(ax, "a")

    ax = axes[0, 1]
    candidate_names = ["Strict-causal\nselector", "Held-out fixed\nmixture"]
    candidate_values = [nested["existing_expert_causal_headroom"] * 100, nested["heldout_fixed_posterior_mixture"]["relative_headroom"] * 100]
    ax.bar(np.arange(2), candidate_values, color=[COLORS["loss"], COLORS["adaptive"]], width=0.62)
    ax.axhline(gates["router_only_gate"] * 100, color=COLORS["gain"], lw=1.2, ls="--", label="10% entry gate")
    ax.axhline(0, color="#767676", lw=0.8)
    ax.set_xticks(np.arange(2), candidate_names)
    ax.set_ylabel("Headroom vs nested baseline (%)")
    ax.set_ylim(-1.2, 11.2)
    ax.legend(loc="upper left")
    ax.set_title("Deployable candidates failed the entry gate", loc="left", fontweight="bold")
    for xpos, value in enumerate(candidate_values):
        ax.text(xpos, value + (0.28 if value >= 0 else -0.55), f"{value:.3f}%", ha="center", fontsize=6)
    record("v5_headroom_early_stop", "b", "causal_router_headroom", "nested_selector", nested["existing_expert_causal_headroom"], "docs/t6_10_1_causal_headroom.json", gate=gates["router_only_gate"])
    record("v5_headroom_early_stop", "b", "fixed_mixture_headroom", "heldout_fixed_posterior_mixture", nested["heldout_fixed_posterior_mixture"]["relative_headroom"], "docs/t6_10_1_causal_headroom.json")
    panel_label(ax, "b")

    ax = axes[1, 0]
    expanded = nested["expanded_candidate_action_oracle"]
    attribution = [
        nested["hard_decision_oracle"]["errors"],
        expanded["errors"],
        expanded["incremental_errors_avoided_beyond_existing_hard_actions"],
    ]
    overall_hard = (nested["nested_strongest_baseline"]["errors"] - attribution[0]) / nested["nested_strongest_baseline"]["errors"] * 100
    overall_expanded = expanded["overall_relative_headroom_vs_baseline"] * 100
    incremental = expanded["incremental_action_space_headroom_vs_baseline"] * 100
    vals = [overall_hard, overall_expanded, incremental]
    labels = ["Hard-action\noracle", "Expanded\noracle", "Independent\naction gain"]
    ax.bar(np.arange(3), vals, color=[COLORS["oracle"], COLORS["gold"], COLORS["loss"]])
    ax.axhline(gates["action_space_upper_bound_gate"] * 100, color=COLORS["gain"], lw=1.2, ls="--", label="12% action-space gate")
    ax.set_yscale("symlog", linthresh=0.05, linscale=1.0)
    ax.set_ylim(0, 35)
    ax.set_xticks(np.arange(3), labels)
    ax.set_ylabel("Relative headroom (%)")
    ax.legend(loc="upper right")
    ax.set_title("Truth switching, not a new action, explained the oracle gain", loc="left", fontweight="bold")
    for xpos, value in enumerate(vals):
        ax.text(xpos, value * 1.22 if value > 0.1 else value + 0.012, f"{value:.4g}%", ha="center", fontsize=6)
    record("v5_headroom_early_stop", "c", "overall_hard_action_oracle_headroom", "hard_decision_oracle", overall_hard / 100, "docs/t6_10_1_causal_headroom.json", deployable=False)
    record("v5_headroom_early_stop", "c", "overall_expanded_oracle_headroom", "expanded_candidate_action_oracle", overall_expanded / 100, "docs/t6_10_1_causal_headroom.json", deployable=False)
    record("v5_headroom_early_stop", "c", "incremental_action_space_headroom", "expanded_beyond_hard_actions", incremental / 100, "docs/t6_10_1_causal_headroom.json", gate=gates["action_space_upper_bound_gate"])
    panel_label(ax, "c")

    ax = axes[1, 1]
    regret = nested["regret_decomposition"]
    regret_labels = ["Selection", "Estimation /\nactivation", "Action space"]
    regret_values = np.array(
        [regret["selection_regret_ler"], regret["estimation_regret_ler"], regret["action_space_regret_ler"]]
    )
    shares = regret_values / regret["identity_total_ler"] * 100
    bars = ax.bar(np.arange(3), shares, color=[COLORS["window"], COLORS["route"], COLORS["loss"]])
    ax.set_xticks(np.arange(3), regret_labels)
    ax.set_ylabel("Share of selector-to-oracle regret (%)")
    ax.set_title("The action family contributed negligible independent regret", loc="left", fontweight="bold")
    for bar, share in zip(bars, shares):
        ax.text(bar.get_x() + bar.get_width() / 2, share + 1.4, f"{share:.2f}%", ha="center", fontsize=6)
    for label, value, share in zip(regret_labels, regret_values, shares):
        record("v5_headroom_early_stop", "d", "regret_decomposition_ler", label.replace("\n", " "), value, "docs/t6_10_1_causal_headroom.json", share=share / 100)
    panel_label(ax, "d")

    fig.subplots_adjust(left=0.10, right=0.99, top=0.93, bottom=0.11, hspace=0.43, wspace=0.34)
    save_figure(fig, "v5_headroom_early_stop")


def figure_auxiliary_matched_evidence() -> None:
    cnot = pd.read_csv(SOURCES["noh_cnot"])
    cnot = cnot[cnot["record_type"] == "point_summary"].copy()
    cpd = load_json(SOURCES["multimode_cpd"])
    preboard = load_json(SOURCES["preboard"])

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6))

    ax = axes[0, 0]
    method_style = {
        "CI": (COLORS["static"], "o"),
        "ML": (COLORS["route"], "s"),
    }
    for method, group in cnot.groupby("method", sort=False):
        group = group.sort_values("squeezing_db")
        color, marker = method_style[method]
        values = group["probability"].to_numpy(dtype=float)
        yerr = np.vstack(
            [values - group["ci_low"].to_numpy(dtype=float), group["ci_high"].to_numpy(dtype=float) - values]
        )
        ax.errorbar(
            group["squeezing_db"],
            values,
            yerr=yerr,
            color=color,
            marker=marker,
            markersize=4.5,
            lw=1.4,
            capsize=2.5,
            label=method,
        )
        for _, item in group.iterrows():
            record(
                "auxiliary_matched_evidence",
                "a",
                "noh_cnot_gate_failure_probability",
                method,
                float(item["probability"]),
                "docs/t6_17_2_noh_cnot_ci_ml_reproduction_source_data.csv",
                squeezing_db=float(item["squeezing_db"]),
                ci_low=float(item["ci_low"]),
                ci_high=float(item["ci_high"]),
                trials=int(item["trials"]),
            )
    ax.set_yscale("log")
    ax.set_xticks([9, 12, 13])
    ax.set_xlabel("Squeezing (dB)")
    ax.set_ylabel("Two-GKP CNOT failure probability")
    ax.legend(title="Project-native reproduction")
    ax.set_title("Analog ML reduced gate-level failure in a matched model", loc="left", fontweight="bold")
    panel_label(ax, "a")

    ax = axes[0, 1]
    scenario_order = ["smooth", "calibration_shift", "telegraph"]
    scenario_labels = ["Smooth", "Calibration", "Telegraph"]
    method_order = ["static_euclidean", "weighted_static", "observed_only_posterior_predictive_weighted", "oracle_metric_upper_bound"]
    method_labels = ["Euclidean", "Weighted static", "Observed-only adaptive", "Metric oracle"]
    colors = [COLORS["static"], COLORS["kalman"], COLORS["route"], COLORS["oracle"]]
    width = 0.19
    x = np.arange(len(scenario_order))
    for idx, (method, label, color) in enumerate(zip(method_order, method_labels, colors)):
        vals = [cpd["summaries"][scenario][method]["p_L"] for scenario in scenario_order]
        ax.bar(x + (idx - 1.5) * width, vals, width=width, label=label, color=color)
        for scenario, value in zip(scenario_order, vals):
            record("auxiliary_matched_evidence", "b", "multimode_cpd_p_L", f"{scenario}:{method}", value, "docs/t6_18_3_multimode_posterior_weighted_cpd.json", evidence="PROJECT_NATIVE_MATCHED")
    ax.set_xticks(x, scenario_labels)
    ax.set_ylabel(r"Logical error rate $p_L$")
    ax.set_ylim(0, 0.33)
    ax.legend(ncol=2, loc="upper left")
    ax.set_title("Posterior weighting helped under multimode drift", loc="left", fontweight="bold")
    panel_label(ax, "b")

    ax = axes[1, 0]
    comparison_order = ["aggregate", "smooth", "calibration_shift", "telegraph"]
    comparison_labels = ["Aggregate", "Smooth", "Calibration", "Telegraph"]
    estimates = []
    lows = []
    highs = []
    for scenario in comparison_order:
        item = cpd["comparisons"][scenario]["adaptive_vs_static_euclidean"]["improvement"]
        estimates.append(item["mean"])
        lows.append(item["ci_low"])
        highs.append(item["ci_high"])
        record(
            "auxiliary_matched_evidence",
            "c",
            "static_euclidean_minus_adaptive_p_L",
            scenario,
            item["mean"],
            "docs/t6_18_3_multimode_posterior_weighted_cpd.json",
            ci_low=item["ci_low"],
            ci_high=item["ci_high"],
            clusters=item["clusters"],
        )
    estimates_np = np.array(estimates)
    yerr = np.vstack([estimates_np - np.array(lows), np.array(highs) - estimates_np])
    ax.errorbar(np.arange(4), estimates_np, yerr=yerr, fmt="none", ecolor="#4D4D4D", capsize=3, lw=1.1)
    ax.scatter(np.arange(4), estimates_np, color=[COLORS["gain"]] * 4, s=38, edgecolor="white", lw=0.7)
    ax.axhline(0, color="#767676", ls="--", lw=0.8)
    ax.set_xticks(np.arange(4), comparison_labels, rotation=24, ha="right")
    ax.set_ylabel(r"Euclidean $-$ adaptive $p_L$")
    ax.set_title("All 32 seed clusters favored the adaptive CPD lane", loc="left", fontweight="bold")
    for xpos, value in enumerate(estimates):
        ax.text(xpos, value + 0.004, f"{value:.3f}", ha="center", fontsize=6)
    panel_label(ax, "c")

    ax = axes[1, 1]
    eligible = next(item for item in preboard["hardware_profiles"] if item["method_id"] == "static_map_lut_if_rtl")
    pr = eligible["place_route"]
    seeds = [item["seed"] for item in pr]
    fmax = [item["achieved_fmax_mhz"] for item in pr]
    lut4 = [item["lut4_count"] for item in pr]
    ax.plot(seeds, fmax, color=COLORS["route"], marker="o", lw=1.4)
    ax.axhline(27.0, color=COLORS["gain"], ls="--", lw=1.0)
    ax.set_xticks(seeds)
    ax.set_xlabel("P&R seed")
    ax.set_ylabel("Fmax estimate (MHz)")
    ax.set_ylim(25, 44)
    ax.set_title("The six-cycle core cleared 27 MHz; board metrics remain null", loc="left", fontweight="bold")
    for seed, freq, lut in zip(seeds, fmax, lut4):
        ax.text(seed, freq + 0.7, f"{freq:.2f}\n{lut} LUT4", ha="center", fontsize=5.8)
        record(
            "auxiliary_matched_evidence",
            "d",
            "preboard_fmax_mhz",
            f"seed_{seed}",
            freq,
            "docs/t6_19_1_project_preboard_profiles.json",
            lut4=lut,
            ff=next(item["ff_count"] for item in pr if item["seed"] == seed),
            bram=next(item["bram_count"] for item in pr if item["seed"] == seed),
            board_measured_latency_ns=None,
        )
    ax.text(
        0.02,
        0.13,
        "dashed: 27 MHz contract clock",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.7,
        color=COLORS["gain"],
    )
    ax.text(
        0.98,
        0.02,
        "6 cycles = 222.222 ns @ 27 MHz\nboard latency / jitter / power: null",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.7,
        color="#4D4D4D",
    )
    panel_label(ax, "d")

    fig.subplots_adjust(left=0.10, right=0.99, top=0.93, bottom=0.10, hspace=0.43, wspace=0.34)
    save_figure(fig, "auxiliary_matched_evidence")


def write_source_data() -> None:
    columns = [
        "figure",
        "panel",
        "metric",
        "label",
        "value",
        "source",
        "ci_low",
        "ci_high",
        "decisions",
        "trials",
        "clusters",
        "gate",
        "share",
        "squeezing_db",
        "unnecessary_fallback_rate",
        "false_updates",
        "commits",
        "average_ler",
        "window_denominator",
        "deployable",
        "evidence",
        "lut4",
        "ff",
        "bram",
        "board_measured_latency_ns",
    ]
    with (OUT / "route_a_manuscript_figures_source_data.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(DERIVED_ROWS)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest() -> None:
    output_names = [
        f"{stem}.{suffix}"
        for stem in ("v4_evidence_boundary", "v5_headroom_early_stop", "auxiliary_matched_evidence")
        for suffix in ("svg", "pdf", "png", "tiff")
    ] + ["route_a_manuscript_figures_source_data.csv"]
    manifest = {
        "schema_version": "1.0",
        "figure_contract": {
            "backend": "Python/matplotlib",
            "width_in": 7.2,
            "purpose": "Evidence-bounded manuscript figures without cross-lane ranking",
            "evidence_boundaries": [
                "V4 restricted simulator/pre-board evidence is separate from V5 causal-headroom diagnosis",
                "Phase-6C auxiliary results cannot upgrade the V5 early-stop verdict",
                "P&R timing is an estimate; measured board latency, jitter, and power remain null",
            ],
        },
        "sources": [
            {
                "key": key,
                "path": source.relative_to(ROOT).as_posix(),
                "bytes": source.stat().st_size,
                "sha256": sha256(source),
            }
            for key, source in SOURCES.items()
        ],
        "outputs": [
            {
                "path": (OUT / name).relative_to(ROOT).as_posix(),
                "bytes": (OUT / name).stat().st_size,
                "sha256": sha256(OUT / name),
            }
            for name in output_names
        ],
        "plotted_rows": len(DERIVED_ROWS),
    }
    (OUT / "route_a_manuscript_figures_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    figure_v4_evidence_boundary()
    figure_v5_headroom_early_stop()
    figure_auxiliary_matched_evidence()
    write_source_data()
    write_manifest()
    print(f"Wrote {len(DERIVED_ROWS)} plotted source rows to {OUT}")


if __name__ == "__main__":
    main()
