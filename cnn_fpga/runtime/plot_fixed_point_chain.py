"""Render the T2.4.3 precision/resource/LER evidence as a Python-only figure.

Figure contract
---------------
Core conclusion: quantization effects are component-specific and non-monotonic;
high precision converges to float, stale updates dominate, and severe bank-word
faults have positive paired LER effects.
Archetype: quantitative grid with a sensitivity heatmap hero panel.
Evidence: (a) six OAT precision axes, (b) joint storage/LER curve,
(c) update-period staleness, (d) bank-fault paired intervals.
Exports: 183-mm editable SVG/PDF plus 600-dpi TIFF and 300-dpi PNG.
Reviewer risks: all data are synthetic software-model evidence; representation
bits are not FPGA synthesis; the 3-address-bit Pareto point is distribution-
specific regularization and is not promoted to a universal optimum.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import SymLogNorm

from cnn_fpga.runtime.fixed_point_chain import DEFAULT_ARTIFACT, ROOT
from cnn_fpga.utils.config import save_json


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.size"] = 7
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["legend.frameon"] = False


DEFAULT_OUTPUT = ROOT / "docs" / "figures" / "t2_4_3_precision_resource_ler"
DEFAULT_FIGURE_AUDIT = ROOT / "docs" / "t2_4_3_figure_validation.json"
AXES = (
    "adc_bits",
    "lut_address_bits",
    "llr_fractional_bits",
    "threshold_fractional_bits",
    "state_bits",
    "update_period_windows",
)
AXIS_LABELS = {
    "adc_bits": "ADC / replay bits",
    "lut_address_bits": "LUT address bits",
    "llr_fractional_bits": "LLR fractional bits",
    "threshold_fractional_bits": "Threshold fractional bits",
    "state_bits": "State bits",
    "update_period_windows": "Update period (windows)",
}
BLUE = "#0F4D92"
BLUE_MID = "#3775BA"
RED = "#B64342"
TEAL = "#42949E"
NEUTRAL = "#767676"
GREEN = "#2E9E44"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_artifact(data: Mapping[str, Any]) -> None:
    if data.get("status") != "PASS":
        raise ValueError("fixed-point validation artifact must have PASS status")
    gates = data.get("gates", {})
    if not gates or not all(gates.values()):
        raise ValueError("fixed-point validation artifact contains a failed gate")
    curves = data.get("curves", {})
    expected = {axis: 6 for axis in AXES}
    expected["joint_precision"] = 5
    actual = {axis: len(curves.get(axis, [])) for axis in expected}
    if actual != expected:
        raise ValueError(f"curve coverage mismatch: {actual}")
    if data.get("target_hardware_measured") is not False:
        raise ValueError("figure input must remain non-board evidence")
    if data.get("synthesis_measured") is not False:
        raise ValueError("figure input must remain non-synthesis evidence")


def _add_panel_label(ax: mpl.axes.Axes, label: str) -> None:
    ax.text(
        -0.10,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def build_figure(data: Mapping[str, Any]) -> plt.Figure:
    _validate_artifact(data)
    float_ler = float(data["float_reference"]["mean_ler"])
    fig = plt.figure(figsize=(7.2, 5.7), constrained_layout=False)
    grid = fig.add_gridspec(
        2,
        3,
        height_ratios=[1.28, 1.0],
        hspace=0.48,
        wspace=0.48,
        left=0.12,
        right=0.97,
        bottom=0.11,
        top=0.96,
    )
    ax_a = fig.add_subplot(grid[0, :])
    ax_b = fig.add_subplot(grid[1, 0])
    ax_c = fig.add_subplot(grid[1, 1])
    ax_d = fig.add_subplot(grid[1, 2])

    matrix = np.asarray(
        [
            [row["paired_ler_minus_float"]["mean"] for row in data["curves"][axis]]
            for axis in AXES
        ],
        dtype=np.float64,
    )
    norm = SymLogNorm(linthresh=5.0e-4, linscale=0.7, vmin=-0.003, vmax=0.15)
    image = ax_a.imshow(matrix, aspect="auto", cmap="RdBu_r", norm=norm)
    ax_a.set_xticks(np.arange(6), ["1", "2", "3", "4", "5", "6"])
    ax_a.set_xlabel("Predeclared level (cell text = tested value)")
    ax_a.set_yticks(np.arange(len(AXES)), [AXIS_LABELS[axis] for axis in AXES])
    ax_a.tick_params(length=0)
    ax_a.set_title("Component precision has non-monotonic, unequal LER effects", loc="left")
    for row_index, axis in enumerate(AXES):
        for column, row in enumerate(data["curves"][axis]):
            value = row["axis_value"]
            delta = row["paired_ler_minus_float"]["mean"]
            rgba = image.cmap(image.norm(delta))
            luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            ax_a.text(
                column,
                row_index,
                f"{value:g}\n{delta:+.3f}",
                ha="center",
                va="center",
                fontsize=5.5,
                color="black" if luminance > 0.57 else "white",
            )
    colorbar = fig.colorbar(image, ax=ax_a, fraction=0.018, pad=0.018)
    colorbar.set_label(r"Paired $\Delta$LER vs float")
    colorbar.set_ticks([-0.002, 0.0, 0.001, 0.01, 0.05, 0.15])
    _add_panel_label(ax_a, "a")

    joint = data["curves"]["joint_precision"]
    storage = np.asarray(
        [row["resource_proxy"]["total_dual_bank_storage_bits"] for row in joint]
    )
    joint_ler = np.asarray([row["logical_error_rate"]["mean"] for row in joint])
    ax_b.plot(storage, joint_ler, color=NEUTRAL, ls="--", lw=1.0, zorder=1)
    ax_b.scatter(storage, joint_ler, c=np.linspace(0.2, 1.0, len(joint)), cmap="Blues", s=28, zorder=2)
    for x, y, row in zip(storage, joint_ler, joint):
        ax_b.annotate(
            row["profile_id"].replace("joint_", "").upper(),
            (x, y),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=5.8,
        )
    ax_b.axhline(float_ler, color=GREEN, lw=1.0)
    ax_b.text(
        storage[0] * 1.05,
        float_ler + 0.004,
        "float reference",
        color=GREEN,
        fontsize=5.8,
        ha="left",
    )
    ax_b.set_xscale("log")
    ax_b.set_xlabel("Dual-bank storage proxy (bits)")
    ax_b.set_ylabel("LER")
    ax_b.set_title("Joint precision is not monotonic", loc="left")
    _add_panel_label(ax_b, "b")

    updates = data["curves"]["update_period_windows"]
    update_x = np.asarray([row["axis_value"] for row in updates])
    update_ler = np.asarray([row["logical_error_rate"]["mean"] for row in updates])
    ax_c.plot(update_x, update_ler, color=BLUE, marker="o", ms=3.8, lw=1.4)
    ax_c.axhline(float_ler, color=GREEN, lw=1.0, ls="--")
    ax_c.set_xscale("log", base=2)
    ax_c.set_xticks(update_x, [str(int(value)) for value in update_x])
    ax_c.set_xlabel("Update period / max bank age (windows)")
    ax_c.set_ylabel("LER")
    ax_c.set_title("Stale state dominates", loc="left")
    _add_panel_label(ax_c, "c")

    fault_order = ["state_msb_flip", "lut_sign_burst", "stale_commit", "torn_update"]
    fault_labels = ["state MSB flip", "LUT sign burst", "stale commit", "torn update"]
    fault_rows = [data["bank_fault_aggregates"][mode] for mode in fault_order]
    effect = np.asarray(
        [row["paired_ler_minus_base_quantized"]["mean"] for row in fault_rows]
    )
    low = np.asarray(
        [row["paired_ler_minus_base_quantized"]["ci_low"] for row in fault_rows]
    )
    high = np.asarray(
        [row["paired_ler_minus_base_quantized"]["ci_high"] for row in fault_rows]
    )
    y = np.arange(len(fault_order))[::-1]
    colors = [RED, RED, BLUE_MID, BLUE_MID]
    for yi, estimate, lo, hi, color in zip(y, effect, low, high, colors):
        ax_d.plot([lo, hi], [yi, yi], color=color, lw=1.4)
        ax_d.plot(estimate, yi, marker="o", ms=4, color=color)
    ax_d.axvline(0.0, color=NEUTRAL, lw=0.9, ls="--")
    ax_d.set_yticks(y, fault_labels)
    ax_d.set_xlabel(r"Paired $\Delta$LER vs no-fault bank")
    ax_d.set_title("Only severe word faults are decisive", loc="left")
    ax_d.text(
        0.98,
        0.02,
        "points: mean\nlines: 95% paired bootstrap CI\nn = 8 seeds",
        transform=ax_d.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.4,
        color=NEUTRAL,
    )
    _add_panel_label(ax_d, "d")
    return fig


def render(
    *,
    artifact: Path = DEFAULT_ARTIFACT,
    output_base: Path = DEFAULT_OUTPUT,
    audit_path: Path = DEFAULT_FIGURE_AUDIT,
) -> dict[str, Any]:
    data = json.loads(artifact.read_text(encoding="utf-8"))
    _validate_artifact(data)
    figure = build_figure(data)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    paths = {
        "svg": output_base.with_suffix(".svg"),
        "pdf": output_base.with_suffix(".pdf"),
        "tiff": output_base.with_suffix(".tiff"),
        "png": output_base.with_suffix(".png"),
    }
    figure.savefig(paths["svg"], bbox_inches="tight")
    figure.savefig(paths["pdf"], bbox_inches="tight")
    figure.savefig(
        paths["tiff"],
        dpi=600,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    figure.savefig(paths["png"], dpi=300, bbox_inches="tight")
    plt.close(figure)

    svg_text = paths["svg"].read_text(encoding="utf-8")
    source_sha = _sha256(artifact)
    plot_sha = _sha256(Path(__file__))
    gates = {
        "source_artifact_passes": data["status"] == "PASS" and all(data["gates"].values()),
        "all_four_formats_nonempty": all(path.stat().st_size > 1000 for path in paths.values()),
        "svg_text_remains_editable": svg_text.count("<text") > 20
        and "font:" in svg_text,
        "source_is_not_board_or_synthesis": data["target_hardware_measured"] is False
        and data["synthesis_measured"] is False,
        "fault_panel_uses_paired_base_intervals": all(
            "paired_ler_minus_base_quantized" in row
            for row in data["bank_fault_aggregates"].values()
        ),
    }
    audit = {
        "task_id": "T2.4.3",
        "figure_contract": {
            "core_conclusion": "component precision is non-monotonic; high precision converges, stale updates dominate, and severe bank-word faults raise LER",
            "archetype": "quantitative_grid_with_hero_heatmap",
            "backend": "python_matplotlib_only",
            "final_width_mm": 182.88,
            "statistics": "8 paired seeds; 10000-replicate paired cluster bootstrap",
            "review_risk": "synthetic model and representation proxies are not synthesis or board evidence",
        },
        "source_artifact": artifact.relative_to(ROOT).as_posix(),
        "source_artifact_sha256": source_sha,
        "plot_source_sha256": plot_sha,
        "outputs": {
            name: {
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for name, path in paths.items()
        },
        "gates": gates,
        "status": "PASS" if all(gates.values()) else "FAIL",
    }
    save_json(audit_path, audit)
    return audit


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--output-base", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit", type=Path, default=DEFAULT_FIGURE_AUDIT)
    args = parser.parse_args(argv)
    audit = render(
        artifact=args.artifact,
        output_base=args.output_base,
        audit_path=args.audit,
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0 if audit["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
