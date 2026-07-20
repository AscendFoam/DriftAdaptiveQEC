"""Render T7.1.2 Figures 1--2 from the frozen machine contract (Python only)."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cnn_fpga.benchmark import main_figure_contract as contract


OUT = contract.FIGURE_DIR
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
mpl.rcParams.update({
    "font.size": 7.0,
    "axes.titlesize": 8.0,
    "axes.labelsize": 7.0,
    "axes.linewidth": 0.8,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})

COLORS = {
    "fast": "#0F4D92",
    "fast_soft": "#DCE9F6",
    "slow": "#42949E",
    "slow_soft": "#DDF1F2",
    "safe": "#D69B2D",
    "safe_soft": "#F8EBCB",
    "fault": "#B64342",
    "fault_soft": "#F6CFCB",
    "neutral": "#767676",
    "neutral_soft": "#ECECEC",
    "blocked": "#C7C7C7",
    "white": "#FFFFFF",
    "dark": "#272727",
    "green": "#2E9E44",
    "green_soft": "#DDF3DE",
}


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _binding(path: Path) -> dict[str, object]:
    return {"path": path.resolve().relative_to(ROOT.resolve()).as_posix(), "sha256": _sha256(path), "bytes": path.stat().st_size}


def _axis_off(ax: plt.Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()


def _panel_label(ax: plt.Axes, label: str, x: float = -0.02, y: float = 1.02) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=9, fontweight="bold", ha="left", va="bottom")


def _box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    face: str,
    edge: str,
    *,
    fontsize: float = 6.4,
    linewidth: float = 1.0,
    linestyle: str = "-",
    textcolor: str = COLORS["dark"],
    radius: float = 0.02,
    zorder: int = 2,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        facecolor=face, edgecolor=edge, linewidth=linewidth, linestyle=linestyle,
        transform=ax.transAxes, zorder=zorder,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, transform=ax.transAxes, ha="center", va="center", fontsize=fontsize, color=textcolor, zorder=zorder + 1)
    return patch


def _arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str,
    *,
    linestyle: str = "-",
    linewidth: float = 1.4,
    mutation_scale: float = 10,
    connectionstyle: str = "arc3",
    zorder: int = 4,
) -> FancyArrowPatch:
    arrow = FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=mutation_scale,
        linewidth=linewidth, linestyle=linestyle, color=color,
        connectionstyle=connectionstyle, transform=ax.transAxes, zorder=zorder,
    )
    ax.add_patch(arrow)
    return arrow


def _save(fig: plt.Figure, stem: str) -> None:
    svg_path = OUT / f"{stem}.svg"
    fig.savefig(svg_path)
    # Matplotlib emits multiline SVG path data with trailing spaces.  Normalize
    # those generated lines so the editable artifact also passes repository QA.
    svg_text = svg_path.read_text(encoding="utf-8")
    svg_path.write_text("\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n", encoding="utf-8")
    fig.savefig(OUT / f"{stem}.pdf")
    fig.savefig(OUT / f"{stem}.png", dpi=300)
    fig.savefig(OUT / f"{stem}.tiff", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


def render_figure1(report: Mapping[str, object]) -> None:
    fig = plt.figure(figsize=(183 / 25.4, 127 / 25.4))
    gs = fig.add_gridspec(2, 5, height_ratios=[1.75, 1.0], hspace=0.28, wspace=0.55, left=0.035, right=0.985, top=0.96, bottom=0.07)
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, :3])
    ax_c = fig.add_subplot(gs[1, 3:])
    for ax in (ax_a, ax_b, ax_c):
        _axis_off(ax)

    ax_a.text(0.0, 0.98, "Evidence-bounded Route-A dual loop", transform=ax_a.transAxes, fontsize=9, fontweight="bold", ha="left", va="top")
    ax_a.text(0.0, 0.89, "solid blue: per-round digital action     dashed teal: host parameter update", transform=ax_a.transAxes, fontsize=5.9, color=COLORS["neutral"], ha="left")

    _box(ax_a, 0.01, 0.30, 0.14, 0.25, "GKP plant /\nfrozen simulator", COLORS["neutral_soft"], COLORS["neutral"], fontsize=6.5)
    _box(ax_a, 0.19, 0.30, 0.16, 0.25, "Observed syndrome\n+ health / integrity", COLORS["fast_soft"], COLORS["fast"], fontsize=6.4)
    _box(ax_a, 0.40, 0.26, 0.20, 0.33, "FPGA fast path\nMAP-LUT → event / action\n6 cycles · II=1", COLORS["fast_soft"], COLORS["fast"], fontsize=6.7, linewidth=1.4)
    _box(ax_a, 0.65, 0.30, 0.15, 0.25, "Frame / correction\n/ reset request", COLORS["fast_soft"], COLORS["fast"], fontsize=6.4)
    _box(ax_a, 0.84, 0.30, 0.14, 0.25, "Plant response /\nnext syndrome", COLORS["neutral_soft"], COLORS["neutral"], fontsize=6.4)
    for start, end in (((0.15, 0.425), (0.19, 0.425)), ((0.35, 0.425), (0.40, 0.425)), ((0.60, 0.425), (0.65, 0.425)), ((0.80, 0.425), (0.84, 0.425))):
        _arrow(ax_a, start, end, COLORS["fast"], linewidth=1.8)
    _arrow(ax_a, (0.91, 0.29), (0.08, 0.29), COLORS["fast"], linewidth=1.1, connectionstyle="arc3,rad=-0.18")

    _box(ax_a, 0.20, 0.68, 0.17, 0.17, "Observed-only\nhost estimator", COLORS["slow_soft"], COLORS["slow"])
    _box(ax_a, 0.42, 0.68, 0.18, 0.17, "Regime-aware\ntyped safety policy", COLORS["safe_soft"], COLORS["safe"])
    _box(ax_a, 0.65, 0.68, 0.16, 0.17, "Candidate image\nCRC · SHA · version", COLORS["slow_soft"], COLORS["slow"])
    _box(ax_a, 0.65, 0.03, 0.16, 0.14, "Trusted A/B bank", COLORS["green_soft"], COLORS["green"], fontsize=6.2)
    _box(ax_a, 0.84, 0.03, 0.14, 0.14, "LKG rollback\n+ hysteresis", COLORS["safe_soft"], COLORS["safe"], fontsize=6.1)
    _arrow(ax_a, (0.285, 0.55), (0.285, 0.68), COLORS["slow"], linestyle="--")
    _arrow(ax_a, (0.37, 0.765), (0.42, 0.765), COLORS["slow"], linestyle="--")
    _arrow(ax_a, (0.60, 0.765), (0.65, 0.765), COLORS["slow"], linestyle="--")
    _arrow(ax_a, (0.73, 0.68), (0.73, 0.17), COLORS["slow"], linestyle="--")
    _arrow(ax_a, (0.65, 0.10), (0.57, 0.26), COLORS["green"], linewidth=1.5)
    _arrow(ax_a, (0.81, 0.10), (0.84, 0.10), COLORS["safe"], linewidth=1.2)
    _arrow(ax_a, (0.91, 0.17), (0.79, 0.26), COLORS["safe"], linewidth=1.2, connectionstyle="arc3,rad=0.18")
    ax_a.text(0.30, 0.61, "32-cycle observed window", transform=ax_a.transAxes, fontsize=5.4, color=COLORS["slow"], ha="center")
    ax_a.text(0.55, 0.63, "stage / freeze / switch / reset / rollback", transform=ax_a.transAxes, fontsize=5.4, color=COLORS["safe"], ha="center")
    ax_a.text(0.02, 0.06, "truth labels: evaluation only", transform=ax_a.transAxes, fontsize=5.6, color=COLORS["neutral"], style="italic")
    ax_a.text(0.20, 0.02, "CNN / teacher / student: optional ablation sidecar; not the fast action path", transform=ax_a.transAxes, fontsize=5.6, color=COLORS["neutral"], ha="left")
    _panel_label(ax_a, "a", -0.025, 0.97)

    ax_b.text(0.0, 0.98, "Evidence ladder", transform=ax_b.transAxes, fontsize=8, fontweight="bold", ha="left", va="top")
    labels = [
        ("Project-native\nsimulation", COLORS["slow_soft"], COLORS["slow"], "available"),
        ("Fixed-point\ninteger reference", COLORS["fast_soft"], COLORS["fast"], "available"),
        ("CXXRTL\n1,000,000 cycles", COLORS["fast_soft"], COLORS["fast"], "available"),
        ("3-seed P&R\nestimate", COLORS["safe_soft"], COLORS["safe"], "estimate"),
        ("Board\nmeasurement", COLORS["neutral_soft"], COLORS["fault"], "42 fields null"),
    ]
    x0, w, gap = 0.01, 0.17, 0.025
    for i, (label, face, edge, footer) in enumerate(labels):
        x = x0 + i * (w + gap)
        _box(ax_b, x, 0.30, w, 0.39, label, face, edge, fontsize=5.9, linestyle="--" if i == 4 else "-")
        ax_b.text(x + w / 2, 0.18, footer, transform=ax_b.transAxes, ha="center", va="center", fontsize=5.3, color=edge)
        if i < len(labels) - 1:
            _arrow(ax_b, (x + w, 0.495), (x + w + gap, 0.495), COLORS["neutral"], linewidth=0.8, mutation_scale=7)
    ax_b.text(0.01, 0.03, "estimate ≠ measurement; no missing value is imputed", transform=ax_b.transAxes, fontsize=5.5, color=COLORS["fault"], fontweight="bold")
    _panel_label(ax_b, "b", -0.05, 0.98)

    ax_c.text(0.0, 0.98, "Timing ownership", transform=ax_c.transAxes, fontsize=8, fontweight="bold", ha="left", va="top")
    for cycle in range(7):
        x = 0.03 + cycle * 0.105
        color = COLORS["fast"] if cycle < 6 else COLORS["green"]
        ax_c.add_patch(Rectangle((x, 0.55), 0.085, 0.18, transform=ax_c.transAxes, facecolor=COLORS["fast_soft"] if cycle < 6 else COLORS["green_soft"], edgecolor=color, linewidth=0.8))
        ax_c.text(x + 0.0425, 0.64, f"C{cycle}", transform=ax_c.transAxes, ha="center", va="center", fontsize=5.5, color=color)
    ax_c.text(0.03, 0.79, "source", transform=ax_c.transAxes, fontsize=5.5, color=COLORS["fast"])
    ax_c.text(0.70, 0.79, "action valid", transform=ax_c.transAxes, fontsize=5.5, color=COLORS["green"])
    ax_c.text(0.03, 0.42, "6-cycle clock-model latency · initiation interval = 1", transform=ax_c.transAxes, fontsize=5.8, color=COLORS["fast"], fontweight="bold")
    ax_c.plot([0.03, 0.78], [0.32, 0.32], transform=ax_c.transAxes, color=COLORS["slow"], linewidth=1.1, linestyle="--")
    ax_c.text(0.03, 0.22, "host candidate cadence: 4000 cycles", transform=ax_c.transAxes, fontsize=5.8, color=COLORS["slow"])
    ax_c.text(0.03, 0.07, "board latency / jitter / deadline / power: null", transform=ax_c.transAxes, fontsize=5.8, color=COLORS["fault"], fontweight="bold")
    _panel_label(ax_c, "c", -0.08, 0.98)

    _save(fig, "figure1_contract_system")


def render_figure2(report: Mapping[str, object]) -> None:
    fig = plt.figure(figsize=(183 / 25.4, 137 / 25.4))
    gs = fig.add_gridspec(3, 12, height_ratios=[1.72, 0.85, 0.62], hspace=0.33, wspace=0.72, left=0.035, right=0.985, top=0.96, bottom=0.06)
    ax_a = fig.add_subplot(gs[0, 0:3])
    ax_b = fig.add_subplot(gs[0, 3:7])
    ax_c = fig.add_subplot(gs[0, 7:12])
    ax_d = fig.add_subplot(gs[1, :])
    ax_e = fig.add_subplot(gs[2, :])
    for ax in (ax_a, ax_b, ax_c, ax_d, ax_e):
        _axis_off(ax)

    ax_a.text(0.0, 0.98, "Observed-only inputs", transform=ax_a.transAxes, fontsize=8, fontweight="bold", va="top")
    for y, text, color in ((0.66, "q / p syndrome codes", COLORS["fast"]), (0.39, "health · leakage · integrity", COLORS["safe"]), (0.12, "version · CRC · age · ack", COLORS["neutral"])):
        _box(ax_a, 0.07, y, 0.86, 0.17, text, COLORS["neutral_soft"] if color == COLORS["neutral"] else (COLORS["fast_soft"] if color == COLORS["fast"] else COLORS["safe_soft"]), color, fontsize=6.1)
    ax_a.text(0.50, 0.02, "no hidden regime or truth field", transform=ax_a.transAxes, fontsize=5.3, color=COLORS["fault"], ha="center", fontweight="bold")
    _panel_label(ax_a, "a", -0.09, 0.98)

    ax_b.text(0.0, 0.98, "Causal evidence branch", transform=ax_b.transAxes, fontsize=8, fontweight="bold", va="top")
    regimes = [
        (0.70, "normal / smooth", COLORS["green_soft"], COLORS["green"]),
        (0.48, "calibration shift / burst", COLORS["safe_soft"], COLORS["safe"]),
        (0.26, "leakage", COLORS["fault_soft"], COLORS["fault"]),
        (0.04, "uncertainty / CRC / version", COLORS["neutral_soft"], COLORS["neutral"]),
    ]
    for y, text, face, edge in regimes:
        _box(ax_b, 0.04, y, 0.92, 0.14, text, face, edge, fontsize=5.9)
    _panel_label(ax_b, "b", -0.07, 0.98)

    ax_c.text(0.0, 0.98, "Typed action", transform=ax_c.transAxes, fontsize=8, fontweight="bold", va="top")
    actions = [
        (0.70, "stage eligible adaptive image", COLORS["green_soft"], COLORS["green"]),
        (0.48, "freeze update · select trusted bank", COLORS["safe_soft"], COLORS["safe"]),
        (0.26, "leakage reset / frame hold", COLORS["fault_soft"], COLORS["fault"]),
        (0.04, "rollback last-known-good", COLORS["neutral_soft"], COLORS["neutral"]),
    ]
    for y, text, face, edge in actions:
        _box(ax_c, 0.03, y, 0.94, 0.14, text, face, edge, fontsize=5.9)
    for y, _, _, edge in actions:
        _arrow(ax_c, (-0.11, y + 0.07), (0.02, y + 0.07), edge, linewidth=1.2, mutation_scale=8)
    _panel_label(ax_c, "c", -0.05, 0.98)

    # Cross-panel arrows use figure coordinates and preserve row alignment.
    fig.canvas.draw()
    for frac, color in ((0.77, COLORS["green"]), (0.55, COLORS["safe"]), (0.33, COLORS["fault"]), (0.11, COLORS["neutral"])):
        b = ax_b.get_position()
        c = ax_c.get_position()
        y = b.y0 + frac * b.height
        fig.add_artist(FancyArrowPatch((b.x1 + 0.002, y), (c.x0 - 0.004, y), transform=fig.transFigure, arrowstyle="-|>", mutation_scale=8, linewidth=1.0, color=color))
    a = ax_a.get_position()
    b = ax_b.get_position()
    fig.add_artist(FancyArrowPatch((a.x1 + 0.003, a.y0 + 0.52 * a.height), (b.x0 - 0.004, b.y0 + 0.52 * b.height), transform=fig.transFigure, arrowstyle="-|>", mutation_scale=9, linewidth=1.2, color=COLORS["fast"]))

    ax_d.text(0.0, 0.98, "Atomic A/B bank and recovery transaction", transform=ax_d.transAxes, fontsize=8, fontweight="bold", va="top")
    transaction = [
        (0.02, "Candidate image\nCRC · SHA · version", COLORS["slow_soft"], COLORS["slow"]),
        (0.27, "Write inactive\nA/B bank", COLORS["fast_soft"], COLORS["fast"]),
        (0.52, "Safe-boundary\natomic commit", COLORS["green_soft"], COLORS["green"]),
        (0.77, "LKG republish +\n8-window recovery", COLORS["safe_soft"], COLORS["safe"]),
    ]
    for i, (x, text, face, edge) in enumerate(transaction):
        _box(ax_d, x, 0.28, 0.19, 0.38, text, face, edge, fontsize=6.1)
        if i < len(transaction) - 1:
            _arrow(ax_d, (x + 0.19, 0.47), (x + 0.25, 0.47), COLORS["neutral"], linewidth=1.0, mutation_scale=8)
    _arrow(ax_d, (0.90, 0.27), (0.63, 0.27), COLORS["safe"], linewidth=1.0, mutation_scale=8, connectionstyle="arc3,rad=-0.20")
    ax_d.text(0.49, 0.08, "fault or failed guard → no torn update; recover through a new monotonic version", transform=ax_d.transAxes, fontsize=5.5, color=COLORS["safe"], ha="center")
    _panel_label(ax_d, "d", -0.025, 0.98)

    ax_e.text(0.0, 0.98, "Explicit evidence boundary", transform=ax_e.transAxes, fontsize=8, fontweight="bold", va="top")
    _box(ax_e, 0.02, 0.25, 0.45, 0.43, "NOT RUN / DROPPED\nIMM · BOCPD · posterior-mixture MAP · V5 risk compiler\nV5 quantized / formal / CXXRTL / P&R", COLORS["neutral_soft"], COLORS["neutral"], fontsize=5.8, linestyle="--")
    _box(ax_e, 0.53, 0.25, 0.45, 0.43, "BLOCKED / NULL\nboard source-to-action latency · jitter · deadline · power\nno measured speed advantage", COLORS["fault_soft"], COLORS["fault"], fontsize=5.8, linestyle="--")
    ax_e.text(0.50, 0.06, "grey/red boundary boxes are not production modules and have no incoming production arrow", transform=ax_e.transAxes, fontsize=5.4, color=COLORS["fault"], ha="center", fontweight="bold")
    _panel_label(ax_e, "e", -0.025, 0.98)

    _save(fig, "figure2_safe_adaptation")


def _svg_text_count(path: Path) -> int:
    root = ET.parse(path).getroot()
    return sum(1 for element in root.iter() if element.tag.endswith("text"))


def write_manifest(manual_visual_qa: str) -> None:
    outputs = {name: _binding(OUT / name) for name in contract.FIGURE_OUTPUTS}
    tiff_dimensions = {}
    for name in ("figure1_contract_system.tiff", "figure2_safe_adaptation.tiff"):
        with Image.open(OUT / name) as image:
            tiff_dimensions[name] = min(image.size)
    text_nodes = sum(_svg_text_count(OUT / name) for name in ("figure1_contract_system.svg", "figure2_safe_adaptation.svg"))
    manifest = {
        "task_id": contract.TASK_ID,
        "schema_version": "t7.1.2-main-figure-bundle-v1",
        "backend": "Python/matplotlib only",
        "contract": _binding(contract.DEFAULT_REPORT),
        "source_data": _binding(contract.DEFAULT_SOURCE_DATA),
        "renderer": _binding(Path(__file__)),
        "outputs": outputs,
        "qa": {
            "svg_text_nodes": text_nodes,
            "svg_path_text_promotion": False,
            "tiff_min_dimension_px": tiff_dimensions,
            "backend_exclusive": True,
            "manual_visual_qa": manual_visual_qa,
        },
    }
    contract._atomic_json(manifest, contract.DEFAULT_MANIFEST)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--manual-visual-qa", choices=("PENDING", "PASS"), default="PENDING")
    args = parser.parse_args()
    report = _load(contract.DEFAULT_REPORT)
    contract.verify_report(report)
    if not args.manifest_only:
        render_figure1(report)
        render_figure2(report)
    for output in contract.FIGURE_OUTPUTS:
        if not (OUT / output).is_file():
            raise FileNotFoundError(OUT / output)
    write_manifest(args.manual_visual_qa)
    print(json.dumps({"output_dir": OUT.relative_to(ROOT).as_posix(), "outputs": len(contract.FIGURE_OUTPUTS), "manual_visual_qa": args.manual_visual_qa}, ensure_ascii=False))


if __name__ == "__main__":
    main()
