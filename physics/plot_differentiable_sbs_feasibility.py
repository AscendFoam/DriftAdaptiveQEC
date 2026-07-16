"""Create the T2.3.6 publication-style feasibility figure and source CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


# Editable text and journal-width typography are mandatory for this figure.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.size"] = 7
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["legend.frameon"] = False


BLUE_MAIN = "#0F4D92"
BLUE_SECONDARY = "#3775BA"
TEAL = "#42949E"
VIOLET = "#9A4D8E"
RED = "#B64342"
NEUTRAL = "#767676"
LIGHT = "#E8E8E8"
CUTOFF_COLORS = {16: BLUE_MAIN, 24: BLUE_SECONDARY, 32: TEAL, 48: VIOLET}


def _load_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("task_id") != "T2.3.6" or payload.get("status") != "PASS":
        raise ValueError("input must be a passing T2.3.6 artifact")
    if len(payload.get("points", [])) < 60:
        raise ValueError("refusing to plot a demo-sized feasibility scan")
    return payload


def write_source_csv(payload: dict[str, Any], path: Path) -> None:
    points = payload["points"]
    repeat_count = max(len(point.get("runtime_seconds", [])) for point in points)
    fields = [
        "point_id",
        "device",
        "cutoff",
        "batch_size",
        "full_cycles",
        "half_cycles",
        "status",
        "feasible",
        "preferred",
        "runtime_median_seconds",
        "runtime_p90_seconds",
        "cuda_peak_allocated_bytes",
        "cuda_peak_reserved_bytes",
        "peak_rss_bytes",
        "rss_delta_bytes",
        "observed_memory_fraction",
        "mean_reward",
        "mean_ground_outcome_fraction",
        "minimum_trajectory_probability",
        "minimum_gradient_norm",
        "maximum_gradient_norm",
        "maximum_trace_error",
        "maximum_hermiticity_error",
        "minimum_final_eigenvalue",
        "policy_parameter_count",
        "real_dtype",
        "warmup_steps",
        "repeats",
        "grid_points",
        "score_baseline",
        "seed",
    ] + [f"runtime_repeat_{index + 1}_seconds" for index in range(repeat_count)]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for point in points:
            row = {field: point.get(field) for field in fields if not field.startswith("runtime_repeat_")}
            for index in range(repeat_count):
                runtimes = point.get("runtime_seconds", [])
                row[f"runtime_repeat_{index + 1}_seconds"] = (
                    runtimes[index] if index < len(runtimes) else None
                )
            writer.writerow(row)


def _panel_label(ax: Any, label: str) -> None:
    ax.text(
        -0.08,
        1.05,
        label,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def build_figure(payload: dict[str, Any]) -> mpl.figure.Figure:
    points = payload["points"]
    cuda = [point for point in points if point["device"] == "cuda"]
    fig = plt.figure(figsize=(7.2, 6.1), constrained_layout=False)
    grid = fig.add_gridspec(
        2,
        3,
        height_ratios=(1.20, 1.0),
        left=0.08,
        right=0.97,
        bottom=0.09,
        top=0.96,
        hspace=0.48,
        wspace=0.42,
    )
    ax_a = fig.add_subplot(grid[0, :])
    ax_b = fig.add_subplot(grid[1, 0])
    ax_c = fig.add_subplot(grid[1, 1])
    ax_d = fig.add_subplot(grid[1, 2])

    # a | Hero panel: the registered cutoff-16 horizon/batch envelope.
    batches = (8, 16, 32, 64, 128, 256, 512, 576)
    horizons = tuple(range(2, 11))
    runtime = np.full((len(batches), len(horizons)), np.nan, dtype=float)
    status: dict[tuple[int, int], str] = {}
    for point in cuda:
        if point["cutoff"] == 16 and point["batch_size"] in batches and point["full_cycles"] in horizons:
            row = batches.index(point["batch_size"])
            column = horizons.index(point["full_cycles"])
            runtime[row, column] = point["runtime_median_seconds"]
            status[(row, column)] = point["status"]
    color_map = mpl.colormaps["Blues"].copy()
    color_map.set_bad(LIGHT)
    image = ax_a.imshow(runtime, aspect="auto", cmap=color_map, vmin=0.0, vmax=10.0)
    ax_a.set_xticks(range(len(horizons)), horizons)
    ax_a.set_yticks(range(len(batches)), batches)
    ax_a.set_xlabel("Full-cycle horizon")
    ax_a.set_ylabel("Trajectory batch")
    ax_a.set_title("Cutoff 16: measured optimization-step median (s)", loc="left", fontsize=7.5)
    for row in range(len(batches)):
        for column in range(len(horizons)):
            value = runtime[row, column]
            if np.isfinite(value):
                text_color = "white" if value >= 5.2 else "#272727"
                ax_a.text(column, row, f"{value:.1f}", ha="center", va="center", fontsize=5.7, color=text_color)
                if status[(row, column)] != "pass":
                    ax_a.add_patch(
                        mpl.patches.Rectangle(
                            (column - 0.48, row - 0.48),
                            0.96,
                            0.96,
                            fill=False,
                            edgecolor=RED,
                            linewidth=1.5,
                            hatch="////",
                        )
                    )
    color_bar = fig.colorbar(image, ax=ax_a, fraction=0.020, pad=0.015)
    color_bar.set_label("Median step time (s)")
    color_bar.ax.axhline(10.0, color=RED, linewidth=1.2)
    ax_a.text(
        4.0,
        4.2,
        "not sampled",
        color=NEUTRAL,
        fontsize=6.2,
        ha="center",
        va="center",
    )
    _panel_label(ax_a, "a")

    # b/c | Ten-cycle runtime and VRAM frontiers across cutoffs.
    for cutoff, color in CUTOFF_COLORS.items():
        subset = sorted(
            (
                point for point in cuda
                if point["cutoff"] == cutoff and point["full_cycles"] == 10
            ),
            key=lambda item: item["batch_size"],
        )
        if not subset:
            continue
        x = np.asarray([point["batch_size"] for point in subset], dtype=float)
        runtime_y = np.asarray([point["runtime_median_seconds"] for point in subset], dtype=float)
        memory_y = np.asarray([point["cuda_peak_allocated_bytes"] / 1.0e9 for point in subset], dtype=float)
        ax_b.plot(x, runtime_y, color=color, marker="o", ms=3.5, lw=1.5, label=f"N={cutoff}")
        ax_c.plot(x, memory_y, color=color, marker="o", ms=3.5, lw=1.5)
        runtime_rejected = [
            index for index, point in enumerate(subset)
            if not point["within_runtime_budget"]
        ]
        memory_rejected = [
            index for index, point in enumerate(subset)
            if not point["within_memory_budget"]
        ]
        if runtime_rejected:
            ax_b.scatter(x[runtime_rejected], runtime_y[runtime_rejected], color=RED, marker="x", s=30, linewidth=1.5, zorder=5)
        if memory_rejected:
            ax_c.scatter(x[memory_rejected], memory_y[memory_rejected], color=RED, marker="x", s=30, linewidth=1.5, zorder=5)
    for ax in (ax_b, ax_c):
        ax.set_xscale("log", base=2)
        ax.grid(axis="y", color="#D8D8D8", linewidth=0.6, alpha=0.8)
        ax.set_xlabel("Trajectory batch")
    ax_b.axhline(10.0, color=RED, linestyle="--", linewidth=1.0)
    ax_b.text(0.98, 0.94, "10 s gate", transform=ax_b.transAxes, ha="right", va="top", color=RED, fontsize=6)
    ax_b.set_ylabel("Median step time (s)")
    ax_b.set_title("10-cycle runtime frontier", loc="left", fontsize=7.5)
    ax_b.legend(title="Cutoff", fontsize=5.7, title_fontsize=5.7, ncol=2, loc="upper left")
    _panel_label(ax_b, "b")

    cuda_total_gb = next(point["cuda_total_bytes"] for point in cuda if point["cuda_total_bytes"] is not None) / 1.0e9
    memory_gate_gb = 0.75 * cuda_total_gb
    ax_c.axhline(memory_gate_gb, color=RED, linestyle="--", linewidth=1.0)
    ax_c.text(0.98, 0.94, "75% VRAM gate", transform=ax_c.transAxes, ha="right", va="top", color=RED, fontsize=6)
    ax_c.set_ylabel("Peak allocated VRAM (GB)")
    ax_c.set_title("10-cycle memory frontier", loc="left", fontsize=7.5)
    _panel_label(ax_c, "c")

    # d | CPU fallback lane, same cutoff/batch and three horizons.
    cpu = sorted(
        (
            point for point in points
            if point["device"] == "cpu"
            and point["cutoff"] == 8
            and point["batch_size"] == 4
            and point["full_cycles"] in (2, 6, 10)
        ),
        key=lambda item: item["full_cycles"],
    )
    x_cpu = np.asarray([point["full_cycles"] for point in cpu], dtype=float)
    y_runtime = np.asarray([point["runtime_median_seconds"] for point in cpu], dtype=float)
    y_rss = np.asarray([point["rss_delta_bytes"] / 1.0e6 for point in cpu], dtype=float)
    ax_d.plot(x_cpu, y_runtime, color=BLUE_MAIN, marker="o", lw=1.5, ms=4)
    ax_d.set_xlabel("Full-cycle horizon")
    ax_d.set_ylabel("CPU step time (s)", color=BLUE_MAIN)
    ax_d.tick_params(axis="y", labelcolor=BLUE_MAIN)
    ax_d.set_xticks((2, 6, 10))
    twin = ax_d.twinx()
    twin.plot(x_cpu, y_rss, color=NEUTRAL, marker="s", lw=1.3, ms=3.5)
    twin.set_ylabel("RSS increase (MB)", color=NEUTRAL)
    twin.tick_params(axis="y", labelcolor=NEUTRAL)
    twin.spines["top"].set_visible(False)
    ax_d.set_title("CPU fallback scaling (N=8, B=4)", loc="left", fontsize=7.5)
    ax_d.grid(axis="y", color="#D8D8D8", linewidth=0.6, alpha=0.8)
    _panel_label(ax_d, "d")

    return fig


def render(
    artifact: Path,
    output_stem: Path,
    source_csv: Path,
) -> list[Path]:
    payload = _load_payload(artifact)
    write_source_csv(payload, source_csv)
    figure = build_figure(payload)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix, options in (
        ("svg", {}),
        ("pdf", {}),
        ("tiff", {"dpi": 600}),
        ("png", {"dpi": 300}),
    ):
        path = output_stem.with_suffix(f".{suffix}")
        figure.savefig(path, bbox_inches="tight", **options)
        outputs.append(path)
    plt.close(figure)
    svg = outputs[0].read_text(encoding="utf-8")
    if "<text" not in svg:
        raise RuntimeError("SVG text was outlined instead of remaining editable")
    return outputs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact",
        type=Path,
        default=Path("docs/t2_3_6_differentiable_sbs_feasibility.json"),
    )
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=Path("docs/figures/t2_3_6_differentiable_sbs_feasibility"),
    )
    parser.add_argument(
        "--source-csv",
        type=Path,
        default=Path("docs/t2_3_6_differentiable_sbs_feasibility.csv"),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    outputs = render(args.artifact, args.output_stem, args.source_csv)
    print(json.dumps([str(path) for path in outputs], indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
