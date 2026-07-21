"""Render the T7.1.5 Phase-6D dual-lane figure delta with Python/matplotlib only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cnn_fpga.benchmark import phase6d_claim_figure_delta as contract  # noqa: E402


OUT = contract.FIGURE_DIR
MANIFEST = contract.MANIFEST

BLUE = "#0F4D92"
BLUE_MID = "#7884B4"
BLUE_SOFT = "#DDE6F4"
TEAL = "#42949E"
RED = "#B64342"
RED_SOFT = "#F6CFCB"
GREEN = "#2E9E44"
GREEN_SOFT = "#DDF3DE"
ORANGE = "#D99718"
ORANGE_SOFT = "#F5E8C8"
GREY = "#767676"
GREY_DARK = "#4D4D4D"
GREY_SOFT = "#E8E8E8"
INK = "#272727"

# Editable-vector rules required by the selected Python backend.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
mpl.rcParams.update({
    "font.size": 7.2,
    "axes.titlesize": 9.0,
    "axes.labelsize": 7.4,
    "xtick.labelsize": 6.6,
    "ytick.labelsize": 6.6,
    "legend.fontsize": 6.3,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
})


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _binding(path: Path) -> dict[str, Any]:
    return {
        "path": path.resolve().relative_to(ROOT.resolve()).as_posix(),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }


def _elements(report: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {row["element_id"]: row for row in report["elements"]}


def _panel(ax: plt.Axes, letter: str, title: str) -> None:
    ax.text(-0.06, 1.06, letter, transform=ax.transAxes, fontsize=10.5, fontweight="bold", va="top")
    ax.set_title(title, loc="left", pad=6, fontweight="bold")


def _box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    *,
    face: str,
    edge: str,
    dashed: bool = False,
    fontsize: float = 6.7,
    weight: str = "normal",
) -> None:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.025",
        transform=ax.transAxes,
        facecolor=face,
        edgecolor=edge,
        linewidth=1.0,
        linestyle="--" if dashed else "-",
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=INK,
        fontweight=weight,
    )


def _arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], *, color: str = GREY_DARK, dashed: bool = False) -> None:
    patch = FancyArrowPatch(
        start,
        end,
        transform=ax.transAxes,
        arrowstyle="-|>",
        mutation_scale=10,
        linewidth=1.0,
        linestyle="--" if dashed else "-",
        color=color,
    )
    ax.add_patch(patch)


def _save(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.svg")
    fig.savefig(OUT / f"{stem}.pdf")
    fig.savefig(OUT / f"{stem}.png", dpi=300)
    fig.savefig(OUT / f"{stem}.tiff", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


def render_figure5(report: Mapping[str, Any]) -> None:
    element = _elements(report)
    fig = plt.figure(figsize=(183 / 25.4, 127 / 25.4))
    gs = fig.add_gridspec(
        2,
        5,
        left=0.105,
        right=0.985,
        top=0.94,
        bottom=0.08,
        hspace=0.54,
        wspace=0.72,
        height_ratios=[1.05, 0.95],
    )
    ax_a = fig.add_subplot(gs[0, :3])
    ax_b = fig.add_subplot(gs[0, 3:])
    ax_c = fig.add_subplot(gs[1, :3])
    ax_d = fig.add_subplot(gs[1, 3:])

    _panel(ax_a, "a", "Strongest deployable denominator: zero causal headroom")
    value = element["MM-E4"]["value"]
    x = [value["baseline_p_L"], value["proposed_p_L"]]
    y = [1, 0]
    ax_a.hlines(y, 0.108, x, color=[BLUE_MID, BLUE], linewidth=2.2, zorder=2)
    ax_a.scatter(x, y, s=42, color=[BLUE_MID, BLUE], edgecolor=INK, linewidth=0.65, zorder=3)
    ax_a.set_yticks(y, ["Static-mixture\nexact MLD", "Proposed\nrisk action"])
    ax_a.set_xlim(0.108, 0.116)
    ax_a.set_ylim(-0.55, 1.55)
    ax_a.set_xlabel("Logical error rate per round, $p_L$")
    ax_a.grid(axis="x", color=GREY_SOFT, linewidth=0.6, zorder=0)
    ax_a.tick_params(length=2.5)
    for yi, xv in zip(y, x):
        ax_a.text(xv + 0.00018, yi, f"{xv:.6f}", va="center", fontsize=6.4)
    ax_a.text(
        0.98,
        0.96,
        "relative point = 0.0%\npaired 95% LCB = 0.0%\nrequired ≥15% / ≥12%\nNO-GO",
        transform=ax_a.transAxes,
        ha="right",
        va="top",
        color=RED,
        fontweight="bold",
        fontsize=6.6,
        bbox={"boxstyle": "round,pad=0.30", "facecolor": RED_SOFT, "edgecolor": RED, "linewidth": 0.8},
    )
    ax_a.text(0.01, 0.02, "79,872 train-only rounds · pilot/formal not accessed", transform=ax_a.transAxes, color=GREY, fontsize=6.0)

    _panel(ax_b, "b", "Opened task-local context")
    ax_b.set_axis_off()
    opened = element["MM-E1"]["value"]
    tail = element["MM-E2"]["value"]
    compute = element["MM-E3"]["value"]
    _box(ax_b, (0.02, 0.57), 0.45, 0.29, f"LER context\n{opened['candidate_p_L']:.4f}\nvs Euclidean {opened['static_euclidean_p_L']:.4f}", face=GREY_SOFT, edge=GREY, dashed=True)
    _box(ax_b, (0.52, 0.57), 0.46, 0.29, f"Tail context\nworst {tail['candidate_worst_window_ler']:.4f}\nCVaR95 {tail['candidate_cvar95_window_ler']:.4f}", face=GREY_SOFT, edge=GREY, dashed=True)
    _box(ax_b, (0.02, 0.15), 0.96, 0.27, f"Host compute context\n{1e6 * compute['candidate_seconds_per_decode']:.1f} µs/decode · {compute['candidate_runtime_seconds']:.1f} s total", face=GREY_SOFT, edge=GREY, dashed=True)
    ax_b.text(0.5, 0.01, "9.6M cycles · opened development evidence\nnot strongest-baseline SOTA", transform=ax_b.transAxes, ha="center", va="bottom", color=RED, fontsize=5.7, fontweight="bold")

    _panel(ax_c, "c", "Evidence-state boundary: unavailable is not zero")
    ax_c.set_axis_off()
    stages = [
        ("Train-only\nheadroom", "AVAILABLE\nNO-GO", RED_SOFT, RED, False),
        ("Pilot", "NOT ACCESSED", GREY_SOFT, GREY, True),
        ("Formal", "NOT ACCESSED", GREY_SOFT, GREY, True),
        ("Scaling", "NOT RUN", GREY_SOFT, GREY, True),
    ]
    for index, (name, state, face, edge, dashed) in enumerate(stages):
        left = 0.01 + index * 0.247
        _box(ax_c, (left, 0.34), 0.205, 0.43, f"{name}\n{state}", face=face, edge=edge, dashed=dashed, fontsize=6.5, weight="bold" if index == 0 else "normal")
        if index < len(stages) - 1:
            _arrow(ax_c, (left + 0.207, 0.555), (left + 0.244, 0.555), color=GREY, dashed=True)
    ax_c.text(0.01, 0.13, "Frozen-benchmark LER/tail/scaling SOTA claim: BLOCKED", transform=ax_c.transAxes, color=RED, fontweight="bold", fontsize=6.4)
    ax_c.text(0.01, 0.02, "No post-outcome threshold, family or denominator selection is permitted.", transform=ax_c.transAxes, color=GREY, fontsize=5.9)

    _panel(ax_d, "d", "Replaceable learning module")
    ax_d.set_axis_off()
    _box(ax_d, (0.08, 0.40), 0.84, 0.38, "CNN / student\nDROPPED · ABLATION ONLY", face=GREY_SOFT, edge=GREY, dashed=True, fontsize=7.4, weight="bold")
    ax_d.text(0.5, 0.26, "no legal teacher / formal retention", transform=ax_d.transAxes, ha="center", color=GREY_DARK, fontsize=6.2)
    ax_d.text(0.5, 0.15, "does not vote on either primary lane", transform=ax_d.transAxes, ha="center", color=RED, fontsize=6.2, fontweight="bold")
    ax_d.text(0.5, 0.03, "No RTL timing claim is transferred from Figure 6.", transform=ax_d.transAxes, ha="center", color=GREY, fontsize=5.7)

    _save(fig, "figure5_multimode_software_delta")


def render_figure6(report: Mapping[str, Any]) -> None:
    element = _elements(report)
    fig = plt.figure(figsize=(183 / 25.4, 137 / 25.4))
    gs = fig.add_gridspec(
        3,
        6,
        left=0.07,
        right=0.985,
        top=0.95,
        bottom=0.065,
        hspace=0.66,
        wspace=0.92,
        height_ratios=[1.0, 1.0, 0.42],
    )
    ax_a = fig.add_subplot(gs[0, :3])
    ax_b = fig.add_subplot(gs[0, 3:])
    ax_c = fig.add_subplot(gs[1, :2])
    ax_d = fig.add_subplot(gs[1, 2:4])
    ax_e = fig.add_subplot(gs[1, 4:])
    ax_f = fig.add_subplot(gs[2, :])
    for ax in (ax_a, ax_b, ax_c, ax_f):
        ax.set_axis_off()

    _panel(ax_a, "a", "Deterministic source-to-action pipeline")
    for cycle in range(7):
        left = 0.025 + cycle * 0.137
        face = GREEN_SOFT if cycle == 6 else BLUE_SOFT
        edge = GREEN if cycle == 6 else BLUE
        ax_a.add_patch(Rectangle((left, 0.49), 0.105, 0.23, transform=ax_a.transAxes, facecolor=face, edgecolor=edge, linewidth=1.0))
        ax_a.text(left + 0.0525, 0.605, f"C{cycle}", transform=ax_a.transAxes, ha="center", va="center", color=edge, fontweight="bold")
    ax_a.text(0.025, 0.34, "syndrome / event accepted", transform=ax_a.transAxes, color=BLUE, fontsize=6.2)
    ax_a.text(0.978, 0.34, "action valid", transform=ax_a.transAxes, color=GREEN, ha="right", fontsize=6.2)
    ax_a.text(0.025, 0.17, "6 cycles · II=1", transform=ax_a.transAxes, color=INK, fontweight="bold", fontsize=8.2)
    ax_a.text(0.025, 0.055, "cycle contract; transport, physical jitter and multimode MLD excluded", transform=ax_a.transAxes, color=RED, fontsize=6.0, fontweight="bold")

    _panel(ax_b, "b", "Atomic A/B bank and fail-closed recovery")
    steps = [
        (0.01, "CRC / version\nadmission", BLUE_SOFT, BLUE),
        (0.27, "write inactive\nA/B bank", BLUE_SOFT, BLUE),
        (0.53, "safe-boundary\natomic commit", GREEN_SOFT, GREEN),
        (0.79, "LKG rollback\n+ hysteresis", ORANGE_SOFT, ORANGE),
    ]
    for index, (left, text, face, edge) in enumerate(steps):
        _box(ax_b, (left, 0.47), 0.19, 0.28, text, face=face, edge=edge, fontsize=6.1, weight="bold" if index >= 2 else "normal")
        if index < len(steps) - 1:
            _arrow(ax_b, (left + 0.192, 0.61), (steps[index + 1][0] - 0.012, 0.61), color=GREY_DARK)
    _box(ax_b, (0.12, 0.10), 0.76, 0.20, "CRC / version / age / uncertainty / leakage fault → freeze, reject, reset or LKG", face=RED_SOFT, edge=RED, dashed=True, fontsize=6.0)

    _panel(ax_c, "c", "Exact qualification")
    cards = [
        ("17 / 17", "formal gates"),
        ("21 / 21", "formal mutants"),
        ("1,000,000", "CXXRTL cycles"),
        ("0", "full-vector\nmismatch"),
    ]
    for index, (value, label) in enumerate(cards):
        row, col = divmod(index, 2)
        _box(ax_c, (0.02 + col * 0.50, 0.53 - row * 0.42), 0.45, 0.31, f"{value}\n{label}", face=GREEN_SOFT, edge=GREEN, fontsize=6.4, weight="bold")
    ax_c.text(0.02, 0.01, "148-byte public vector · independent golden", transform=ax_c.transAxes, color=GREY, fontsize=5.5)

    _panel(ax_d, "d", "Three-seed whole-harness P&R")
    fmax = element["RTL-E4"]["value"]["fmax_mhz"]
    paths = element["RTL-E4"]["value"]["critical_paths"]
    seed_values = {row["seed"]: 1000.0 / row["period_ns"] for row in paths}
    seeds = [1, 7, 19]
    values = [seed_values[seed] for seed in seeds]
    ax_d.plot(range(3), values, "o-", color=BLUE, markerfacecolor=BLUE_SOFT, markeredgecolor=BLUE, linewidth=1.3, markersize=5, zorder=3)
    ax_d.axhline(27, color=GREY, linestyle="--", linewidth=0.9, label="27-MHz target")
    ax_d.set_xticks(range(3), ["seed 1", "seed 7", "seed 19"])
    ax_d.set_ylabel("Fmax estimate (MHz)")
    ax_d.set_ylim(25, 41)
    ax_d.grid(axis="y", color=GREY_SOFT, linewidth=0.55, zorder=0)
    for index, value in enumerate(values):
        ax_d.text(index, value + 0.8, f"{value:.3f}", ha="center", fontsize=5.8)
    ax_d.text(0.03, 0.04, f"min {fmax['minimum']:.3f} MHz\npaths end in observability fold", transform=ax_d.transAxes, ha="left", va="bottom", color=RED, fontsize=5.7, fontweight="bold", bbox={"boxstyle": "round,pad=0.20", "facecolor": "white", "edgecolor": "none", "alpha": 0.86})

    _panel(ax_e, "e", "Complete-harness resources")
    resource = element["RTL-E5"]["value"]
    keys = ["LUT4", "DFF", "BSRAM", "MULT18X18", "MULT9X9", "ALU", "IOB"]
    labels = ["LUT4", "FF", "BRAM", "M18", "M9", "ALU", "IOB"]
    fractions = [100.0 * resource[key]["maximum"] / resource[key]["available"] for key in keys]
    bars = ax_e.bar(range(len(keys)), fractions, color=[BLUE_MID] * len(keys), edgecolor=INK, linewidth=0.55, zorder=2)
    ax_e.set_xticks(range(len(keys)), labels, rotation=25, ha="right")
    ax_e.set_ylabel("Used / available (%)")
    ax_e.set_ylim(0, 35)
    ax_e.grid(axis="y", color=GREY_SOFT, linewidth=0.55, zorder=0)
    for bar, key in zip(bars, keys):
        ax_e.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8, str(resource[key]["maximum"]), ha="center", va="bottom", fontsize=5.0, rotation=90)
    ax_e.text(0.98, 0.96, "post-route estimate", transform=ax_e.transAxes, ha="right", va="top", color=RED, fontsize=5.8, fontweight="bold")

    _panel(ax_f, "f", "Physical-board and speed-claim boundary")
    _box(ax_f, (0.00, 0.05), 0.49, 0.67, "BOARD: UNMEASURED\nlatency · jitter · deadline miss · power\ntransfer latency · commit latency = null", face=RED_SOFT, edge=RED, dashed=True, fontsize=6.5, weight="bold")
    _box(ax_f, (0.52, 0.05), 0.48, 0.67, "NO SPEED / SOTA CLAIM\nwhole-harness estimate ≠ bare core or board\ncurrent RTL ≠ multimode decoder", face=GREY_SOFT, edge=GREY, dashed=True, fontsize=6.5, weight="bold")

    _save(fig, "figure6_single_mode_rtl_delta")


def _image_qa(path: Path) -> tuple[int, float, float]:
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"))
    ink = np.any(rgb < 248, axis=2)
    edge = np.concatenate([ink[:5, :].ravel(), ink[-5:, :].ravel(), ink[:, :5].ravel(), ink[:, -5:].ravel()])
    return min(rgb.shape[:2]), float(np.mean(ink)), float(np.mean(edge))


def write_manifest(manual_visual_qa: str) -> dict[str, Any]:
    report = _load(contract.REPORT)
    outputs = {name: _binding(OUT / name) for name in report["output_files"]}
    svg_nodes: dict[str, int] = {}
    embedded_rasters = 0
    for name in report["output_files"]:
        if name.endswith(".svg"):
            text = (OUT / name).read_text(encoding="utf-8")
            svg_nodes[name] = len(re.findall(r"<text\b", text))
            embedded_rasters += len(re.findall(r"<image\b", text))
    tiff_dims: dict[str, int] = {}
    png_dims: dict[str, int] = {}
    nonwhite: dict[str, float] = {}
    edge_ink: dict[str, float] = {}
    for name in report["output_files"]:
        if name.endswith((".png", ".tiff")):
            minimum, fraction, edge_fraction = _image_qa(OUT / name)
            nonwhite[name] = fraction
            edge_ink[name] = edge_fraction
            if name.endswith(".tiff"):
                tiff_dims[name] = minimum
            else:
                png_dims[name] = minimum
    manifest = {
        "task_id": contract.TASK_ID,
        "schema_version": "t7.1.5-phase6d-figure-bundle-v1",
        "backend": "Python/matplotlib only",
        "contract": _binding(contract.REPORT),
        "source_data": _binding(contract.SOURCE_DATA),
        "renderer": _binding(Path(__file__).resolve()),
        "outputs": outputs,
        "historical_output_sha256": sorted(contract._historical_output_hashes()),
        "qa": {
            "svg_text_nodes": svg_nodes,
            "svg_path_text_promotion": False,
            "svg_embedded_raster_count": embedded_rasters,
            "tiff_min_dimension_px": tiff_dims,
            "png_min_dimension_px": png_dims,
            "nonwhite_fraction": nonwhite,
            "edge_ink_fraction": edge_ink,
            "backend_exclusive": True,
            "manual_visual_qa": manual_visual_qa,
        },
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--manual-visual-qa", choices=("PENDING", "PASS"), default="PENDING")
    args = parser.parse_args()
    report = _load(contract.REPORT)
    contract.verify(report)
    OUT.mkdir(parents=True, exist_ok=True)
    if not args.manifest_only:
        render_figure5(report)
        render_figure6(report)
    manifest = write_manifest(args.manual_visual_qa)
    print(json.dumps({
        "output_dir": OUT.relative_to(ROOT).as_posix(),
        "outputs": len(manifest["outputs"]),
        "manual_visual_qa": args.manual_visual_qa,
        "qa": manifest["qa"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
