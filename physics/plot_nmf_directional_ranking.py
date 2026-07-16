"""Render the T2.3.7 publication figure and traceable Source Data CSV.

Figure contract
---------------
Core conclusion:
    Under one finite-cutoff simulator, identical 10-cycle physical time and
    noise, the trained NMF family directionally exceeds MF and standard on
    held-out projected-logical-Z area-equivalent lifetime; resetting recurrent
    history removes the primary-lane advantage, and cutoff 16 preserves the
    NMF > MF > standard direction.
Archetype:
    Asymmetric quantitative grid with a trajectory hero panel.
Backend / size / exports:
    Python-only matplotlib, 183 mm x 120 mm, editable SVG/PDF, 600-dpi TIFF,
    300-dpi PNG, and a long-form CSV.
Evidence map:
    a, held-out logical-Z curves; b, paired five-agent primary lifetime and
    history ablation; c, independent cutoff-16 confirmation.
Review risks:
    Five trained agents are not five physical devices; the 10-cycle
    area-equivalent lifetime is not the paper's 1000-cycle six-state lifetime;
    state, agent and trajectory sources remain explicit in the CSV/legend.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from .nmf_directional_ranking import (
    ANALYSIS_CONTRACT_ID,
    TRAINING_PROTOCOL_ID,
    implementation_sha256,
)


# Mandatory editable-text and journal-width settings from the figure contract.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.size"] = 7
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["legend.frameon"] = False


COLORS = {
    "standard": "#606060",
    "mf": "#7884B4",
    "nmf": "#B64342",
    "nmf_latest_only": "#9A4D8E",
}
LABELS = {
    "standard": "Standard sBs",
    "mf": "MF (latest outcome)",
    "nmf": "NMF (full history)",
    "nmf_latest_only": "NMF, history reset",
}
MARKERS = {"standard": "s", "mf": "o", "nmf": "D", "nmf_latest_only": "^"}
LINESTYLES = {"standard": "--", "mf": "-", "nmf": "-", "nmf_latest_only": ":"}
METRICS = (
    "fidelity_effective_lifetime_cycles",
    "fidelity_normalized_auc",
    "logical_z_effective_lifetime_cycles",
    "logical_z_normalized_auc",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _evaluation_list(
    payload: Mapping[str, Any], lane: str, strategy: str
) -> list[Mapping[str, Any]]:
    value = payload["evaluation"][lane][strategy]
    return [value] if strategy == "standard" else list(value)


def _assert_close_sequence(
    observed: Sequence[float], expected: Sequence[float], *, label: str
) -> None:
    if len(observed) != len(expected) or not np.allclose(
        np.asarray(observed, dtype=float),
        np.asarray(expected, dtype=float),
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError(f"artifact summary is inconsistent with raw evaluation: {label}")


def load_and_audit_artifact(
    artifact_path: Path, *, verify_checkpoint: bool = True
) -> dict[str, Any]:
    """Fail closed on pilots, stale sources, inconsistent summaries or hashes."""

    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    if payload.get("task_id") != "T2.3.7" or payload.get("status") != "PASS":
        raise ValueError("input must be a passing T2.3.7 artifact")
    if payload.get("production_design") is not True:
        raise ValueError("refusing to render a pilot-sized ranking")
    if payload.get("analysis_contract_id") != ANALYSIS_CONTRACT_ID:
        raise ValueError("artifact analysis contract is stale")
    if payload.get("training_protocol_id") != TRAINING_PROTOCOL_ID:
        raise ValueError("artifact training protocol is stale")
    if payload.get("implementation_sha256") != implementation_sha256():
        raise ValueError("artifact executable-source hash does not match the current code")
    required = list(payload.get("required_directional_gates", []))
    gates = payload.get("gates", {})
    if len(required) < 14 or any(gates.get(name) is not True for name in required):
        raise ValueError("not all registered directional gates pass")
    config = payload["config"]
    if (
        int(config["full_cycles"]) != 10
        or int(config["cutoff"]) < 12
        or int(config["confirmation_cutoff"]) < 16
        or len(config["training_seeds"]) < 5
        or int(config["test_batch_size"]) * len(config["test_seeds"]) < 512
    ):
        raise ValueError("artifact violates the registered non-demo production design")
    checkpoint = payload.get("checkpoint", {})
    if (
        checkpoint.get("schema_version") != 3
        or checkpoint.get("contains_all_training_seed_models") is not True
        or checkpoint.get("all_model_hashes_match") is not True
    ):
        raise ValueError("artifact checkpoint completeness audit failed")

    for lane in ("primary", "confirmation"):
        for strategy in ("standard", "mf", "nmf", "nmf_latest_only"):
            evaluations = _evaluation_list(payload, lane, strategy)
            for metric in METRICS:
                observed = [float(item["metric_means"][metric]) for item in evaluations]
                expected = payload["summary"][lane][strategy][metric]["values"]
                _assert_close_sequence(observed, expected, label=f"{lane}/{strategy}/{metric}")

    if verify_checkpoint:
        recorded = Path(str(checkpoint["path"]))
        candidates = [recorded]
        if not recorded.is_absolute():
            candidates.extend(
                [Path.cwd() / recorded, artifact_path.resolve().parent.parent / recorded]
            )
        checkpoint_path = next((path for path in candidates if path.exists()), None)
        if checkpoint_path is None:
            raise ValueError("recorded T2.3.7 checkpoint is missing")
        if _sha256(checkpoint_path) != checkpoint["sha256"]:
            raise ValueError("checkpoint file SHA-256 differs from the artifact")
    return payload


def _agent_seed_for_index(payload: Mapping[str, Any], index: int) -> int:
    seeds = list(payload["config"]["training_seeds"])
    if index >= len(seeds):
        raise ValueError("evaluation array is longer than the registered agent-seed list")
    return int(seeds[index])


def _iter_evaluations(
    payload: Mapping[str, Any], lane: str, strategy: str
) -> Iterable[tuple[int | None, Mapping[str, Any]]]:
    for index, evaluation in enumerate(_evaluation_list(payload, lane, strategy)):
        yield (
            None if strategy == "standard" else _agent_seed_for_index(payload, index),
            evaluation,
        )


SOURCE_FIELDS = (
    "record_type",
    "lane",
    "strategy",
    "training_seed",
    "evaluation_seed",
    "cutoff",
    "metric",
    "cycle",
    "value",
    "trajectory_count",
    "statistic",
    "ci95_low",
    "ci95_high",
)


def write_source_csv(payload: Mapping[str, Any], path: Path) -> None:
    """Write every plotted/raw quantitative value in one long-form table."""

    rows: list[dict[str, Any]] = []
    for lane in ("primary", "confirmation"):
        for strategy in ("standard", "mf", "nmf", "nmf_latest_only"):
            for training_seed, evaluation in _iter_evaluations(payload, lane, strategy):
                for seed_record in evaluation["per_seed"]:
                    common = {
                        "lane": lane,
                        "strategy": strategy,
                        "training_seed": training_seed,
                        "evaluation_seed": seed_record["seed"],
                        "cutoff": evaluation["cutoff"],
                        "trajectory_count": seed_record["trajectory_count"],
                    }
                    for metric, key in (
                        ("physical_state_fidelity", "fidelity_curve"),
                        ("projected_logical_z", "logical_z_curve"),
                        ("code_survival", "code_survival_curve"),
                    ):
                        for cycle, value in enumerate(seed_record[key]):
                            rows.append(
                                {
                                    **common,
                                    "record_type": "curve",
                                    "metric": metric,
                                    "cycle": cycle,
                                    "value": value,
                                }
                            )
                    for metric, source in (
                        ("fidelity_effective_lifetime_cycles", seed_record["fidelity"]),
                        ("fidelity_normalized_auc", seed_record["fidelity"]),
                        ("logical_z_effective_lifetime_cycles", seed_record["logical_z"]),
                        ("logical_z_normalized_auc", seed_record["logical_z"]),
                    ):
                        source_key = (
                            "effective_lifetime_cycles"
                            if metric.endswith("lifetime_cycles")
                            else "normalized_auc"
                        )
                        rows.append(
                            {
                                **common,
                                "record_type": "per_evaluation_seed_metric",
                                "metric": metric,
                                "value": source[source_key],
                            }
                        )
                    for metric in (
                        "mean_ground_outcome_probability",
                        "mean_control_residual_rms",
                        "mean_control_slew_rms",
                        "maximum_trace_error",
                        "maximum_hermiticity_error",
                        "minimum_final_eigenvalue",
                    ):
                        rows.append(
                            {
                                **common,
                                "record_type": "per_evaluation_seed_diagnostic",
                                "metric": metric,
                                "value": seed_record[metric],
                            }
                        )
            for metric in METRICS:
                distribution = payload["summary"][lane][strategy][metric]
                for statistic in ("mean", "median", "q1", "q3", "minimum", "maximum"):
                    rows.append(
                        {
                            "record_type": "agent_summary",
                            "lane": lane,
                            "strategy": strategy,
                            "cutoff": payload["config"][
                                "cutoff" if lane == "primary" else "confirmation_cutoff"
                            ],
                            "metric": metric,
                            "value": distribution[statistic],
                            "statistic": statistic,
                        }
                    )
    for name, result in payload["paired_bootstrap"].items():
        rows.append(
            {
                "record_type": "paired_agent_bootstrap",
                "lane": "primary",
                "strategy": "nmf_difference",
                "cutoff": payload["config"]["cutoff"],
                "metric": name,
                "value": result["mean_difference"],
                "statistic": "mean_and_95pct_bootstrap_ci",
                "ci95_low": result["ci95_low"],
                "ci95_high": result["ci95_high"],
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SOURCE_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _bootstrap_curve(
    curves: np.ndarray, *, seed: int, repetitions: int = 20_000
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if curves.ndim != 2 or curves.shape[0] < 2:
        raise ValueError("curve bootstrap requires at least two independent units")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, curves.shape[0], size=(repetitions, curves.shape[0]))
    samples = curves[indices].mean(axis=1)
    return (
        curves.mean(axis=0),
        np.quantile(samples, 0.025, axis=0),
        np.quantile(samples, 0.975, axis=0),
    )


def _strategy_agent_curves(
    payload: Mapping[str, Any], strategy: str
) -> np.ndarray:
    evaluations = _evaluation_list(payload, "primary", strategy)
    if strategy == "standard":
        return np.asarray(
            [item["logical_z_curve"] for item in evaluations[0]["per_seed"]],
            dtype=float,
        )
    return np.asarray(
        [
            np.mean(
                np.asarray([item["logical_z_curve"] for item in evaluation["per_seed"]]),
                axis=0,
            )
            for evaluation in evaluations
        ],
        dtype=float,
    )


def _panel_label(ax: Any, label: str) -> None:
    ax.text(
        -0.08,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def build_figure(payload: Mapping[str, Any]) -> mpl.figure.Figure:
    fig = plt.figure(figsize=(183.0 / 25.4, 120.0 / 25.4), constrained_layout=False)
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=(1.35, 1.0),
        left=0.09,
        right=0.98,
        bottom=0.10,
        top=0.96,
        hspace=0.42,
        wspace=0.34,
    )
    ax_a = fig.add_subplot(grid[0, :])
    ax_b = fig.add_subplot(grid[1, 0])
    ax_c = fig.add_subplot(grid[1, 1])

    cycles = np.arange(int(payload["config"]["full_cycles"]) + 1)
    for index, strategy in enumerate(("standard", "mf", "nmf", "nmf_latest_only")):
        curves = _strategy_agent_curves(payload, strategy)
        center, lower, upper = _bootstrap_curve(curves, seed=9107 + index)
        ax_a.fill_between(cycles, lower, upper, color=COLORS[strategy], alpha=0.12, linewidth=0)
        ax_a.plot(
            cycles,
            center,
            color=COLORS[strategy],
            linestyle=LINESTYLES[strategy],
            marker=MARKERS[strategy],
            markevery=2,
            markersize=3.0,
            linewidth=1.55 if strategy == "nmf" else 1.25,
            label=LABELS[strategy],
        )
    ax_a.set_xlim(0, cycles[-1])
    ax_a.set_ylim(0.0, 1.03)
    ax_a.set_xticks(cycles)
    ax_a.set_xlabel("Full sBs cycles (10 μs per cycle)")
    ax_a.set_ylabel("Normalized projected logical-Z signal")
    ax_a.grid(axis="y", color="#D8D8D8", linewidth=0.55, alpha=0.8)
    ax_a.legend(loc="upper right", ncol=2, fontsize=6.2, columnspacing=1.2, handlelength=2.5)
    ax_a.text(
        0.01,
        0.04,
        "Shading: 95% bootstrap CI over held-out seeds (standard, n=8) or trained agents (feedback, n=5)",
        transform=ax_a.transAxes,
        fontsize=5.7,
        color="#606060",
    )
    ax_a.set_title("Held-out 10-cycle directional memory ranking", loc="left", fontsize=7.8)
    _panel_label(ax_a, "a")

    primary = payload["summary"]["primary"]
    paired_strategies = ("mf", "nmf", "nmf_latest_only")
    positions = np.arange(len(paired_strategies), dtype=float)
    paired_values = np.asarray(
        [primary[strategy]["logical_z_effective_lifetime_cycles"]["values"] for strategy in paired_strategies],
        dtype=float,
    ).T
    for row in paired_values:
        ax_b.plot(positions, row, color="#B8B8B8", linewidth=0.7, alpha=0.8, zorder=1)
    for x, strategy, values in zip(positions, paired_strategies, paired_values.T):
        ax_b.scatter(
            np.full(values.shape, x),
            values,
            color=COLORS[strategy],
            edgecolor="white",
            linewidth=0.45,
            marker=MARKERS[strategy],
            s=25,
            zorder=3,
        )
        ax_b.plot(x, np.median(values), marker="_", color="#272727", ms=11, mew=1.3, zorder=4)
    standard_lifetime = primary["standard"]["logical_z_effective_lifetime_cycles"]["mean"]
    ax_b.axhline(standard_lifetime, color=COLORS["standard"], linestyle="--", linewidth=1.0)
    ax_b.text(
        0.02,
        standard_lifetime,
        f" standard {standard_lifetime:.2f}",
        color=COLORS["standard"],
        fontsize=5.7,
        va="bottom",
    )
    delta = payload["paired_bootstrap"]["nmf_minus_mf_logical_z_lifetime"]
    ablation = payload["paired_bootstrap"][
        "nmf_minus_latest_only_ablation_logical_z_lifetime"
    ]
    ax_b.text(
        0.02,
        0.98,
        f"NMF−MF: {delta['mean_difference']:.2f} [{delta['ci95_low']:.2f}, {delta['ci95_high']:.2f}]\n"
        f"NMF−reset: {ablation['mean_difference']:.2f} [{ablation['ci95_low']:.2f}, {ablation['ci95_high']:.2f}]",
        transform=ax_b.transAxes,
        va="top",
        fontsize=5.8,
    )
    ax_b.set_xticks(positions, ("MF", "NMF", "NMF\nhistory reset"))
    ax_b.set_ylabel("Area-equivalent logical-Z lifetime (cycles)")
    ax_b.set_title("Paired trained-agent evidence (n=5)", loc="left", fontsize=7.5)
    ax_b.grid(axis="y", color="#D8D8D8", linewidth=0.55, alpha=0.8)
    _panel_label(ax_b, "b")

    cutoffs = (int(payload["config"]["cutoff"]), int(payload["config"]["confirmation_cutoff"]))
    for strategy_index, strategy in enumerate(("standard", "mf", "nmf")):
        medians: list[float] = []
        q1: list[float] = []
        q3: list[float] = []
        for lane in ("primary", "confirmation"):
            distribution = payload["summary"][lane][strategy]["logical_z_effective_lifetime_cycles"]
            medians.append(float(distribution["median"]))
            q1.append(float(distribution["q1"]))
            q3.append(float(distribution["q3"]))
        x = np.arange(2, dtype=float) + (strategy_index - 1) * 0.16
        y = np.asarray(medians)
        error = np.vstack((y - np.asarray(q1), np.asarray(q3) - y))
        ax_c.errorbar(
            x,
            y,
            yerr=error,
            color=COLORS[strategy],
            marker=MARKERS[strategy],
            markersize=4.2,
            linewidth=1.2,
            capsize=2.2,
            label=LABELS[strategy],
        )
    ax_c.set_xticks((0, 1), (f"cutoff {cutoffs[0]}\n8×64 test", f"cutoff {cutoffs[1]}\n4×32 confirm"))
    ax_c.set_ylabel("Median logical-Z lifetime (cycles)")
    ax_c.set_title("Independent cutoff confirmation", loc="left", fontsize=7.5)
    ax_c.grid(axis="y", color="#D8D8D8", linewidth=0.55, alpha=0.8)
    ax_c.legend(fontsize=5.6, loc="upper left")
    ax_c.text(
        0.98,
        0.04,
        "Bars: agent IQR; standard is one seed-averaged estimate",
        transform=ax_c.transAxes,
        ha="right",
        fontsize=5.5,
        color="#606060",
    )
    _panel_label(ax_c, "c")
    return fig


def render(
    artifact_path: Path,
    output_stem: Path,
    source_csv: Path,
    *,
    verify_checkpoint: bool = True,
) -> list[Path]:
    payload = load_and_audit_artifact(
        artifact_path, verify_checkpoint=verify_checkpoint
    )
    write_source_csv(payload, source_csv)
    figure = build_figure(payload)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for suffix, options in (
        ("svg", {}),
        ("pdf", {}),
        ("tiff", {"dpi": 600}),
        ("png", {"dpi": 300}),
    ):
        output = output_stem.with_suffix(f".{suffix}")
        figure.savefig(output, bbox_inches="tight", **options)
        outputs.append(output)
    plt.close(figure)
    return outputs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output-stem", type=Path, required=True)
    parser.add_argument("--source-csv", type=Path, required=True)
    parser.add_argument("--skip-checkpoint-hash", action="store_true")
    return parser.parse_args()


def main() -> int:
    arguments = _parse_args()
    outputs = render(
        arguments.artifact,
        arguments.output_stem,
        arguments.source_csv,
        verify_checkpoint=not arguments.skip_checkpoint_hash,
    )
    print(
        json.dumps(
            {
                "source_csv": str(arguments.source_csv),
                "outputs": [str(path) for path in outputs],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "build_figure",
    "load_and_audit_artifact",
    "render",
    "write_source_csv",
]
