"""Render T7.1.3 evidence-bounded manuscript Figures 3 and 4."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cnn_fpga.benchmark import main_result_figure_contract as contract  # noqa: E402


OUT = ROOT / "docs/figures/t7_1_3_main_figures"
MANIFEST = OUT / "figure_manifest.json"

BLUE = "#14589C"
LIGHT_BLUE = "#D8E6F5"
TEAL = "#3B98A3"
GREEN = "#2E9F47"
LIGHT_GREEN = "#DDF1DF"
ORANGE = "#D99718"
LIGHT_ORANGE = "#FAEDC9"
RED = "#BF4545"
LIGHT_RED = "#F4CBC7"
GREY = "#777777"
LIGHT_GREY = "#ECECEC"
INK = "#222222"

mpl.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 9.5,
    "axes.labelsize": 7.5, "xtick.labelsize": 6.7, "ytick.labelsize": 6.7,
    "legend.fontsize": 6.6, "axes.linewidth": 0.8, "pdf.fonttype": 42,
    "svg.fonttype": "none", "savefig.facecolor": "white", "figure.facecolor": "white",
})


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be a JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _binding(path: Path) -> dict[str, Any]:
    return {"path": path.resolve().relative_to(ROOT.resolve()).as_posix(), "sha256": _sha256(path), "bytes": path.stat().st_size}


def _axis(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=2.5, width=0.7)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.55, alpha=0.75, zorder=0)


def _panel(ax: plt.Axes, letter: str, title: str) -> None:
    ax.text(-0.065, 1.08, letter, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")
    ax.set_title(title, loc="left", fontweight="bold", pad=7)


def _box(ax: plt.Axes, xy: tuple[float, float], width: float, height: float, text: str, *, face: str, edge: str, dashed: bool = False, fontsize: float = 7.2) -> None:
    patch = FancyBboxPatch(xy, width, height, boxstyle="round,pad=0.012,rounding_size=0.025", transform=ax.transAxes, facecolor=face, edgecolor=edge, linewidth=1.25, linestyle="--" if dashed else "-")
    ax.add_patch(patch)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text, transform=ax.transAxes, ha="center", va="center", color=INK, fontsize=fontsize)


def _save(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    svg = OUT / f"{stem}.svg"
    fig.savefig(svg)
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8")
    fig.savefig(OUT / f"{stem}.pdf")
    fig.savefig(OUT / f"{stem}.png", dpi=300)
    fig.savefig(OUT / f"{stem}.tiff", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


def _records(report: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {row["record_id"]: row for row in report["records"]}


def render_figure3(report: Mapping[str, Any]) -> None:
    rec = _records(report)
    fig = plt.figure(figsize=(183 / 25.4, 137 / 25.4))
    gs = fig.add_gridspec(3, 5, height_ratios=[1.08, 1.05, 0.34], left=0.07, right=0.985, top=0.95, bottom=0.07, hspace=0.55, wspace=0.72)
    ax_a = fig.add_subplot(gs[0, :3]); ax_b = fig.add_subplot(gs[0, 3:])
    ax_c = fig.add_subplot(gs[1, :3]); ax_d = fig.add_subplot(gs[1, 3:])
    ax_e = fig.add_subplot(gs[2, :]); ax_e.set_axis_off()

    _panel(ax_a, "a", "Untouched smooth aggregate: all methods")
    methods = ["standard_binning", "static_joint_map", "window_map", "ewma_adaptive_map", "kalman_adaptive_map", "proposed_route_a", "hidden_state_oracle"]
    labels = ["Binning", "Static", "Window", "EWMA", "Kalman", "Route-A", "Oracle"]
    values = np.array([rec[f"f3a_{method}"]["value"] for method in methods]) * 1e3
    lows = np.array([rec[f"f3a_{method}"]["lower"] for method in methods]) * 1e3
    highs = np.array([rec[f"f3a_{method}"]["upper"] for method in methods]) * 1e3
    colors = ["#AFAFAF", "#7EA6C9", GREEN, ORANGE, "#9D77B5", BLUE, "#D0D0D0"]
    bars = ax_a.bar(np.arange(len(methods)), values, color=colors, edgecolor=[GREY] * 7, linewidth=0.7, zorder=2)
    bars[-1].set_hatch("///")
    ax_a.errorbar(np.arange(len(methods)), values, yerr=[values - lows, highs - values], fmt="none", ecolor=INK, elinewidth=0.7, capsize=2, zorder=3)
    ax_a.set_xticks(np.arange(len(methods)), labels, rotation=25, ha="right")
    ax_a.set_ylabel(r"Average LER ($\times 10^{-3}$)")
    ax_a.set_ylim(0, max(values) * 1.2)
    _axis(ax_a)
    ax_a.text(0.02, 0.94, "strongest deployable = Window", transform=ax_a.transAxes, color=GREEN, fontweight="bold", va="top")
    ax_a.text(0.02, 0.83, "Route-A is not global best", transform=ax_a.transAxes, color=RED, fontweight="bold", va="top")
    ax_a.text(6, values[6] + 0.14, "nondeployable", color=GREY, ha="center", va="bottom", fontsize=6.1)

    _panel(ax_b, "b", "Paired result boundaries")
    contrasts = [rec["f3b_ewma_minus_route"], rec["f3b_static_minus_route"]]
    vals = np.array([row["value"] for row in contrasts]) * 1e5
    lo = np.array([row["lower"] for row in contrasts]) * 1e5
    hi = np.array([row["upper"] for row in contrasts]) * 1e5
    ax_b.axhline(0, color=GREY, linewidth=0.8)
    ax_b.bar([0, 1], vals, color=[GREEN, RED], width=0.62, zorder=2)
    ax_b.errorbar([0, 1], vals, yerr=[vals - lo, hi - vals], fmt="none", ecolor=INK, capsize=3, linewidth=0.8, zorder=3)
    ax_b.set_xticks([0, 1], ["EWMA −\nRoute-A", "Static −\nRoute-A"])
    ax_b.set_ylabel(r"Paired $\Delta$LER ($\times 10^{-5}$)")
    ax_b.set_ylim(-4.3, 3.6)
    _axis(ax_b)
    gap = rec["f3b_oracle_gap"]
    ax_b.text(0.98, 0.96, f"static→oracle gap closure\n{100*gap['value']:.2f}% [{100*gap['lower']:.2f}, {100*gap['upper']:.2f}]% · negative", transform=ax_b.transAxes, ha="right", va="top", color=RED, fontsize=6.2, bbox={"boxstyle": "round,pad=0.25", "facecolor": LIGHT_RED, "edgecolor": RED, "linewidth": 0.7})

    _panel(ax_c, "c", "Abrupt/OOD worst window: safety, not improvement")
    families = report["tail_families"]
    short = ["Calibration", "Telegraph", "Burst", "Readout/reset", "Leakage", "Compound"]
    baseline = np.array([rec[f"f3c_{family}_ewma_adaptive_map"]["value"] for family in families])
    route = np.array([rec[f"f3c_{family}_proposed_route_a"]["value"] for family in families])
    x = np.arange(len(families))
    ax_c.plot(x - 0.06, baseline, "o", markerfacecolor="white", markeredgecolor=ORANGE, markeredgewidth=1.2, label="locked EWMA", zorder=3)
    ax_c.plot(x + 0.06, route, "x", color=BLUE, markeredgewidth=1.4, markersize=6, label="Route-A", zorder=4)
    ax_c.set_xticks(x, short, rotation=20, ha="right")
    ax_c.set_ylabel("Global worst errors / 512")
    ax_c.set_ylim(0, max(baseline) * 1.2)
    _axis(ax_c); ax_c.legend(frameon=False, loc="upper right")
    ax_c.text(0.02, 0.93, "all six paired worst counts are equal", transform=ax_c.transAxes, va="top", color=RED, fontweight="bold")

    _panel(ax_d, "d", "Fallback burden / recovery")
    fallback = np.array([rec[f"f3d_{family}_fallback"]["value"] for family in families]) * 100
    ax_d.bar(np.arange(len(families)), fallback, color=[ORANGE, "#D7A54D", "#DDB66D", "#D1A15A", "#C88D45", "#B87936"], edgecolor=GREY, linewidth=0.6, zorder=2)
    ax_d.set_xticks(np.arange(len(families)), ["Cal", "Tel", "Burst", "R/R", "Leak", "Comp"])
    ax_d.set_ylabel("Fallback rate (%)")
    ax_d.set_ylim(0, 105)
    _axis(ax_d)
    for idx, family in enumerate(families):
        recovery = rec[f"f3d_{family}_fallback"]["metadata"]["recovery_p95_decisions"]
        text = "no reopen" if recovery is None else f"p95 {recovery:.0f}"
        ax_d.text(idx, min(fallback[idx] - 2, 92), text, rotation=90, ha="center", va="top", fontsize=5.6, color=INK)
    nominal = 100 * rec["f3d_nominal_fallback"]["value"]
    ax_d.text(0.98, 0.04, f"nominal fallback = {nominal:.3f}%\nnon-inferiority; not zero cost", transform=ax_d.transAxes, ha="right", va="bottom", fontsize=6.5, color=BLUE, bbox={"boxstyle": "round,pad=0.25", "facecolor": LIGHT_BLUE, "edgecolor": BLUE, "linewidth": 0.7})

    _panel(ax_e, "e", "Main-result placement boundary")
    _box(ax_e, (0.00, 0.04), 0.31, 0.62, "RESTRICTED POSITIVE\nlocked-EWMA aggregate only", face=LIGHT_GREEN, edge=GREEN)
    _box(ax_e, (0.345, 0.04), 0.31, 0.62, "MANDATORY NEGATIVE\nWindow/static/oracle-gap and high fallback", face=LIGHT_RED, edge=RED)
    _box(ax_e, (0.69, 0.04), 0.30, 0.62, "SUPPLEMENT ONLY\nPhase 6C task-local lanes; no main ranking", face=LIGHT_GREY, edge=GREY, dashed=True)
    _save(fig, "figure3_v4_results")


def render_figure4(report: Mapping[str, Any]) -> None:
    rec = _records(report)
    fig = plt.figure(figsize=(183 / 25.4, 127 / 25.4))
    gs = fig.add_gridspec(3, 5, height_ratios=[0.9, 1.05, 0.42], left=0.07, right=0.985, top=0.95, bottom=0.07, hspace=0.55, wspace=0.72)
    ax_a = fig.add_subplot(gs[0, :3]); ax_b = fig.add_subplot(gs[0, 3:])
    ax_c = fig.add_subplot(gs[1, :3]); ax_d = fig.add_subplot(gs[1, 3:])
    ax_e = fig.add_subplot(gs[2, :]); ax_a.set_axis_off(); ax_b.set_axis_off(); ax_e.set_axis_off()

    _panel(ax_a, "a", "Million-cycle integer/CXXRTL")
    cards = [
        ("1,000,000", "qualified cycles", LIGHT_BLUE, BLUE),
        ("0", "bit\nmismatch", LIGHT_GREEN, GREEN),
        ("0", "undefined\naction", LIGHT_GREEN, GREEN),
        ("0", "silent\noverflow", LIGHT_GREEN, GREEN),
    ]
    for idx, (value, label, face, edge) in enumerate(cards):
        x = 0.01 + idx * 0.245
        _box(ax_a, (x, 0.27), 0.215, 0.47, f"{value}\n{label}", face=face, edge=edge, fontsize=7.1)
    ax_a.text(0.01, 0.13, "10 families · fixed-point integer golden · actual integrated RTL", transform=ax_a.transAxes, color=GREY, style="italic", fontsize=6.2)
    ax_a.text(0.01, 0.04, "abstract FIFO receiver; physical transport excluded", transform=ax_a.transAxes, color=GREY, style="italic", fontsize=6.2)

    _panel(ax_b, "b", "Timing ownership")
    for cycle in range(7):
        x = 0.03 + cycle * 0.13
        face = LIGHT_GREEN if cycle == 6 else LIGHT_BLUE
        edge = GREEN if cycle == 6 else BLUE
        ax_b.add_patch(Rectangle((x, 0.48), 0.105, 0.24, transform=ax_b.transAxes, facecolor=face, edgecolor=edge, linewidth=1.0))
        ax_b.text(x + 0.0525, 0.60, f"C{cycle}", transform=ax_b.transAxes, ha="center", va="center", color=edge)
    ax_b.text(0.03, 0.31, "source", transform=ax_b.transAxes, color=BLUE)
    ax_b.text(0.94, 0.31, "action valid", transform=ax_b.transAxes, color=GREEN, ha="right")
    ax_b.text(0.03, 0.15, "6 cycles · II=1", transform=ax_b.transAxes, color=INK, fontweight="bold")
    ax_b.text(0.03, 0.06, "222.222 ns at assumed 27 MHz", transform=ax_b.transAxes, color=INK, fontweight="bold", fontsize=6.5)
    ax_b.text(0.03, -0.04, "clock model; transport/jitter excluded", transform=ax_b.transAxes, color=RED, fontweight="bold", fontsize=6.1)

    _panel(ax_c, "c", "Three-seed open-source P&R estimate")
    profiles = ["route_a_core_no_student", "route_a_plus_student_sidecar"]
    labels = ["selected core", "core + optional student"]
    colors = [BLUE, ORANGE]
    seeds = [1, 7, 19]
    for profile, label, color, offset in zip(profiles, labels, colors, (-0.07, 0.07)):
        values = [rec[f"f4c_{profile}_{seed}"]["value"] for seed in seeds]
        ax_c.plot(np.arange(3) + offset, values, "o-", color=color, linewidth=1.3, markersize=4.5, label=label, zorder=3)
    ax_c.axhline(27, color=GREY, linestyle="--", linewidth=0.9, label="27-MHz target")
    ax_c.set_xticks(range(3), ["seed 1", "seed 7", "seed 19"])
    ax_c.set_ylabel("Achieved Fmax (MHz)")
    ax_c.set_ylim(25, 43)
    _axis(ax_c); ax_c.legend(frameon=False, ncol=1, loc="center left")
    ax_c.text(0.98, 0.94, "estimate ≠ vendor signoff", transform=ax_c.transAxes, ha="right", va="top", color=RED, fontweight="bold")

    _panel(ax_d, "d", "Complete-profile resources")
    resources = ["lut4", "dff", "bsram", "mult18x18", "mult9x9"]
    labels_r = ["LUT4", "FF", "BRAM", "M18", "M9"]
    x = np.arange(len(resources)); width = 0.34
    for idx, (profile, label, color) in enumerate(zip(profiles, labels, colors)):
        rows = [rec[f"f4d_{profile}_{resource}"] for resource in resources]
        fractions = np.array([row["metadata"]["fraction"] for row in rows]) * 100
        bars = ax_d.bar(x + (idx - 0.5) * width, fractions, width, color=color, alpha=0.88, label=label, zorder=2)
        for bar, row in zip(bars, rows):
            ax_d.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, str(row["value"]), ha="center", va="bottom", rotation=90, fontsize=5.2, color=INK)
    ax_d.set_xticks(x, labels_r)
    ax_d.set_ylabel("Used / available (%)")
    ax_d.set_ylim(0, 28)
    _axis(ax_d); ax_d.legend(frameon=False, fontsize=5.7, loc="upper right")

    _panel(ax_e, "e", "Physical-evidence and stopped-branch boundary")
    _box(ax_e, (0.00, 0.03), 0.49, 0.64, "PHYSICAL BOARD: BLOCKED\n42/42 measured fields null\nlatency · jitter · deadline · power · speed advantage", face=LIGHT_RED, edge=RED, dashed=True)
    _box(ax_e, (0.52, 0.03), 0.47, 0.64, "V5 HARDWARE: NOT RUN / DROPPED\nquantized · long CXXRTL · formal · multi-seed P&R\nno result panel and no inferred value", face=LIGHT_GREY, edge=GREY, dashed=True)
    _save(fig, "figure4_preboard_evidence")


def write_manifest(manual_visual_qa: str) -> dict[str, Any]:
    outputs = {name: _binding(OUT / name) for name in contract.FIGURE_OUTPUTS}
    svg_nodes = 0
    for name in contract.FIGURE_OUTPUTS:
        if name.endswith(".svg"):
            svg_nodes += len(re.findall(r"<text\b", (OUT / name).read_text(encoding="utf-8")))
    tiff_dims: dict[str, int] = {}
    for name in contract.FIGURE_OUTPUTS:
        if name.endswith(".tiff"):
            with Image.open(OUT / name) as image:
                tiff_dims[name] = min(image.size)
    manifest = {
        "task_id": contract.TASK_ID, "schema_version": "t7.1.3-main-result-figure-bundle-v1", "backend": "Python/matplotlib only",
        "contract": _binding(contract.DEFAULT_REPORT), "source_data": _binding(contract.DEFAULT_SOURCE_DATA), "renderer": _binding(Path(__file__).resolve()), "outputs": outputs,
        "qa": {"svg_text_nodes": svg_nodes, "svg_path_text_promotion": False, "tiff_min_dimension_px": tiff_dims, "backend_exclusive": True, "manual_visual_qa": manual_visual_qa},
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--manual-visual-qa", choices=("PENDING", "PASS"), default="PENDING")
    args = parser.parse_args()
    report = _load(contract.DEFAULT_REPORT)
    contract.verify_report(report)
    OUT.mkdir(parents=True, exist_ok=True)
    if not args.manifest_only:
        render_figure3(report)
        render_figure4(report)
    write_manifest(args.manual_visual_qa)
    print(json.dumps({"output_dir": OUT.relative_to(ROOT).as_posix(), "outputs": len(contract.FIGURE_OUTPUTS), "manual_visual_qa": args.manual_visual_qa}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
