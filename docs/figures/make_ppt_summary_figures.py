"""Generate the Chinese PPT summary figures from frozen project evidence.

Outputs
-------
docs/figures/ppt_summary_20260716/
    effective_fidelity_lifetime_cn.{svg,png,pdf}
    drift_adaptive_architecture_cn.{svg,png,pdf}

The lifetime plot is derived directly from the T4.4.4 Source Data.  The
architecture figure is a bounded presentation schematic: it distinguishes the
validated float/software path from pending student quantization, RTL and board
validation.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


# Mandatory editable-SVG and font settings.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "Arial",
    "DejaVu Sans",
    "sans-serif",
]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["legend.frameon"] = False


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DATA = REPO_ROOT / "docs" / "t4_4_4_teacher_student_gain_retention_source_data.csv"
OUTPUT_DIR = REPO_ROOT / "docs" / "figures" / "ppt_summary_20260716"


COLORS = {
    "ink": "#26313D",
    "muted": "#667483",
    "grid": "#DDE4EA",
    "grey": "#B8C2CC",
    "grey_edge": "#7D8996",
    "blue": "#6F8FB8",
    "blue_edge": "#496A93",
    "teacher": "#F1C57A",
    "teacher_edge": "#C98224",
    "student": "#4D9B73",
    "student_edge": "#2D7452",
    "teal": "#22968C",
    "teal_light": "#DFF2EF",
    "orange": "#E98118",
    "orange_light": "#FFF0D5",
    "purple": "#7854A6",
    "purple_light": "#E8E0F2",
    "pink_light": "#F8D3D4",
    "blue_light": "#E3EFF8",
    "panel": "#F4F7FA",
    "white": "#FFFFFF",
}


def _register_chinese_font() -> None:
    """Register the local Microsoft YaHei fonts when available."""

    windows_fonts = Path("C:/Windows/Fonts")
    candidates = [windows_fonts / "msyh.ttc", windows_fonts / "msyhbd.ttc"]
    for candidate in candidates:
        if candidate.exists():
            font_manager.fontManager.addfont(str(candidate))


def _read_rows() -> list[dict[str, str]]:
    with SOURCE_DATA.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _extract_lifetime_data(rows: list[dict[str, str]]):
    """Return per-seed and mean fidelity lifetime data for the four methods."""

    raw: dict[int, dict[int, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )

    for row in rows:
        if row["row_type"] != "stochastic_seed_summary":
            continue
        cutoff = int(row["cutoff"])
        seed = int(row["evaluation_seed"])
        strategy = row["strategy"]
        detail = json.loads(row["detail_json"])
        lifetime = float(detail["fidelity"]["effective_lifetime_cycles"])
        raw[cutoff][seed][strategy].append(lifetime)

    per_seed: dict[int, dict[str, list[float]]] = {}
    for cutoff in (12, 16):
        per_seed[cutoff] = {
            "standard": [],
            "mf_mean": [],
            "teacher": [],
            "student": [],
        }
        for seed in sorted(raw[cutoff]):
            strategy_rows = raw[cutoff][seed]
            per_seed[cutoff]["standard"].append(strategy_rows["standard"][0])
            mf_values = [
                values[0]
                for name, values in strategy_rows.items()
                if name.startswith("mf_agent_")
            ]
            if len(mf_values) != 5:
                raise RuntimeError(f"cutoff={cutoff}, seed={seed}: expected five MF agents")
            per_seed[cutoff]["mf_mean"].append(float(np.mean(mf_values)))
            per_seed[cutoff]["teacher"].append(strategy_rows["teacher"][0])
            per_seed[cutoff]["student"].append(strategy_rows["distilled_student"][0])

    means = {
        cutoff: {
            method: float(np.mean(values))
            for method, values in method_values.items()
        }
        for cutoff, method_values in per_seed.items()
    }

    expected = {
        12: {"standard": 3.570623, "mf_mean": 8.622787, "teacher": 8.439925, "student": 8.427107},
        16: {"standard": 5.993625, "mf_mean": 9.155712, "teacher": 9.525396, "student": 9.489395},
    }
    for cutoff, expected_methods in expected.items():
        for method, expected_value in expected_methods.items():
            if not np.isclose(means[cutoff][method], expected_value, atol=5e-7):
                raise RuntimeError(
                    f"Source Data drift for cutoff={cutoff}, method={method}: "
                    f"{means[cutoff][method]} != {expected_value}"
                )

    return per_seed, means


def _extract_summary_numbers(rows: list[dict[str, str]]):
    retentions = []
    stored_scalars: dict[str, int] = {}
    for row in rows:
        if row["row_type"] == "retention_gate" and row["lane"] in {"primary", "confirmation"}:
            retentions.append(float(row["value"]))
        if row["row_type"] == "cost_summary" and row["metric"] == "stored_scalars":
            stored_scalars[row["strategy"]] = int(float(row["value"]))

    minimum_retention = min(retentions)
    teacher_scalars = stored_scalars["fresh_gru_teacher"]
    student_scalars = stored_scalars["distilled_student"]
    compression_ratio = teacher_scalars / student_scalars
    return minimum_retention, teacher_scalars, student_scalars, compression_ratio


def _save_figure(fig: plt.Figure, stem: str, dpi: int = 300) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in (
        ("svg", {}),
        ("png", {"dpi": dpi}),
        ("pdf", {}),
    ):
        fig.savefig(
            OUTPUT_DIR / f"{stem}.{suffix}",
            bbox_inches="tight",
            facecolor="white",
            **kwargs,
        )
    plt.close(fig)


def _summary_metric(ax, y, value, title, note, color):
    ax.text(0.12, y, value, transform=ax.transAxes, fontsize=25, fontweight="bold", color=color, va="top")
    ax.text(0.12, y - 0.085, title, transform=ax.transAxes, fontsize=12.5, fontweight="bold", color=COLORS["ink"], va="top")
    ax.text(0.12, y - 0.137, note, transform=ax.transAxes, fontsize=9.7, color=COLORS["muted"], va="top", linespacing=1.35)


def make_lifetime_figure(rows: list[dict[str, str]]) -> None:
    per_seed, means = _extract_lifetime_data(rows)
    minimum_retention, teacher_scalars, student_scalars, compression_ratio = _extract_summary_numbers(rows)

    methods = ["standard", "mf_mean", "teacher", "student"]
    labels = ["固定参数", "MF强基线（5模型均值）", "离线教师模型", "4状态学生模型"]
    colors = [COLORS["grey"], COLORS["blue"], COLORS["teacher"], COLORS["student"]]
    edges = [COLORS["grey_edge"], COLORS["blue_edge"], COLORS["teacher_edge"], COLORS["student_edge"]]

    fig = plt.figure(figsize=(12.8, 7.2), facecolor="white")
    gs = fig.add_gridspec(1, 2, width_ratios=[4.35, 1.28], left=0.08, right=0.97, top=0.78, bottom=0.20, wspace=0.08)
    ax = fig.add_subplot(gs[0, 0])
    ax_summary = fig.add_subplot(gs[0, 1])

    fig.text(0.055, 0.935, "轻量学生模型基本保留教师模型的有效保真寿命", fontsize=24, fontweight="bold", color=COLORS["ink"], ha="left")
    fig.text(0.055, 0.885, "10周期物理仿真回放 · 全新配对随机种子 · 数值越高越好", fontsize=12.5, color=COLORS["muted"], ha="left")

    x = np.arange(2, dtype=float)
    width = 0.18
    offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * width
    cutoffs = [12, 16]

    for method_index, method in enumerate(methods):
        values = [means[cutoff][method] for cutoff in cutoffs]
        bars = ax.bar(
            x + offsets[method_index],
            values,
            width=width * 0.88,
            color=colors[method_index],
            edgecolor=edges[method_index],
            linewidth=1.25,
            zorder=2,
            label=labels[method_index],
        )
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.18,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=10.5,
                fontweight="bold" if method == "student" else "normal",
                color=edges[method_index],
                zorder=5,
            )

        for group_index, cutoff in enumerate(cutoffs):
            seed_values = per_seed[cutoff][method]
            jitter = np.linspace(-0.035, 0.035, len(seed_values))
            ax.scatter(
                np.full(len(seed_values), x[group_index] + offsets[method_index]) + jitter,
                seed_values,
                s=23,
                color=edges[method_index],
                edgecolor="white",
                linewidth=0.65,
                zorder=4,
            )

    ratios = [means[c]["student"] / means[c]["standard"] for c in cutoffs]
    for group_index, ratio in enumerate(ratios):
        student_x = x[group_index] + offsets[3]
        student_y = means[cutoffs[group_index]]["student"]
        ax.annotate(
            f"相对固定参数 {ratio:.2f}×",
            xy=(student_x, student_y + 0.05),
            xytext=(student_x + 0.08, student_y + 1.05),
            ha="center",
            va="bottom",
            fontsize=10.5,
            fontweight="bold",
            color=COLORS["student_edge"],
            arrowprops=dict(arrowstyle="-|>", color=COLORS["student_edge"], lw=1.15, shrinkA=4, shrinkB=5),
        )

    ax.set_ylabel("有效保真寿命（纠错周期）", fontsize=14, color=COLORS["ink"], labelpad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(["截断规模 12\n（n = 8）", "截断规模 16\n（n = 4）"], fontsize=12.5, fontweight="bold", color=COLORS["ink"])
    ax.set_ylim(0, 11.5)
    ax.set_yticks(np.arange(0, 12, 2))
    ax.tick_params(axis="y", labelsize=10.5, colors=COLORS["muted"])
    ax.tick_params(axis="x", length=0, pad=9)
    ax.spines["left"].set_color(COLORS["ink"])
    ax.spines["bottom"].set_color(COLORS["ink"])
    ax.spines["left"].set_linewidth(1.15)
    ax.spines["bottom"].set_linewidth(1.15)
    ax.yaxis.grid(True, color=COLORS["grid"], linewidth=0.9, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(-0.01, 1.095),
        ncol=2,
        fontsize=10.3,
        handlelength=1.5,
        columnspacing=1.5,
        labelspacing=0.7,
    )

    ax_summary.set_axis_off()
    panel = FancyBboxPatch(
        (0.02, 0.02), 0.96, 0.96,
        boxstyle="round,pad=0.018,rounding_size=0.05",
        transform=ax_summary.transAxes,
        facecolor=COLORS["panel"],
        edgecolor="#CFD9E3",
        linewidth=1.2,
    )
    ax_summary.add_patch(panel)
    ax_summary.text(0.12, 0.93, "核心结论", transform=ax_summary.transAxes, fontsize=13, fontweight="bold", color=COLORS["ink"], va="top")

    _summary_metric(
        ax_summary, 0.83, "2.36× / 1.58×", "有效寿命提升", "学生模型相对固定参数\n（截断规模12 / 16）", COLORS["student_edge"]
    )
    ax_summary.plot([0.12, 0.88], [0.58, 0.58], transform=ax_summary.transAxes, color="#D6DEE6", lw=1)
    _summary_metric(
        ax_summary, 0.53, f"≥ {minimum_retention * 100:.2f}%", "教师收益保持率", "三个预注册指标中的\n最低点估计", COLORS["teal"]
    )
    ax_summary.plot([0.12, 0.88], [0.29, 0.29], transform=ax_summary.transAxes, color="#D6DEE6", lw=1)
    _summary_metric(
        ax_summary, 0.24, f"约 {compression_ratio:.0f}×", "模型规模缩小", f"{teacher_scalars:,} → {student_scalars}\n个浮点标量", COLORS["orange"]
    )

    fig.text(0.055, 0.116, "●  柱为均值，点为每个配对随机种子；MF为5个冻结模型的均值。", fontsize=9.5, color=COLORS["muted"], ha="left")
    fig.text(0.055, 0.078, "●  截断规模是Fock空间的数值截断，不代表两块不同硬件。", fontsize=9.5, color=COLORS["muted"], ha="left")
    fig.text(0.055, 0.040, "有效保真寿命由10周期保真度曲线的有限时域面积换算；当前是浮点仿真证据，不是FPGA墙钟时间。", fontsize=8.7, color="#7C8792", ha="left")

    _save_figure(fig, "effective_fidelity_lifetime_cn", dpi=300)


def _box(ax, x, y, w, h, facecolor, edgecolor=COLORS["ink"], lw=1.4, radius=0.022, linestyle="-"):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=lw,
        linestyle=linestyle,
        transform=ax.transAxes,
        zorder=2,
    )
    ax.add_patch(patch)
    return patch


def _arrow(ax, start, end, color, lw=2.0, style="-|>", linestyle="-", mutation_scale=16, zorder=5, connectionstyle="arc3"):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=mutation_scale,
        linewidth=lw,
        color=color,
        linestyle=linestyle,
        connectionstyle=connectionstyle,
        transform=ax.transAxes,
        zorder=zorder,
        shrinkA=1,
        shrinkB=1,
    )
    ax.add_patch(arrow)
    return arrow


def _pill(ax, x, y, w, text, edge, fill, icon_fill=None, fontsize=9.8):
    _box(ax, x, y, w, 0.045, fill, edgecolor=edge, lw=1.15, radius=0.022)
    if icon_fill is not None:
        ax.add_patch(Circle((x + 0.020, y + 0.0225), 0.007, transform=ax.transAxes, facecolor=icon_fill, edgecolor="none", zorder=4))
    ax.text(x + 0.035, y + 0.0225, text, transform=ax.transAxes, fontsize=fontsize, color=COLORS["ink"], va="center", ha="left", fontweight="bold")


def make_architecture_figure() -> None:
    fig, ax = plt.subplots(figsize=(13.2, 7.2), facecolor="white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()

    ax.text(0.035, 0.972, "漂移自适应 GKP 纠错双回路架构", transform=ax.transAxes, fontsize=21, fontweight="bold", color=COLORS["ink"], ha="left", va="top")
    ax.text(0.035, 0.906, "离线教师蒸馏 · 在线轻量学生 · 安全原子提交 · 面向FPGA的确定性快路径", transform=ax.transAxes, fontsize=11.8, color=COLORS["muted"], ha="left", va="top")

    # Physical system.
    _box(ax, 0.035, 0.205, 0.205, 0.585, COLORS["pink_light"], edgecolor=COLORS["ink"], lw=1.6, radius=0.03)
    ax.text(0.055, 0.765, "近似 GKP 物理系统", transform=ax.transAxes, fontsize=12.2, fontweight="bold", color=COLORS["ink"], ha="left")

    ancilla = FancyBboxPatch((0.108, 0.625), 0.061, 0.073, boxstyle="round,pad=0.004,rounding_size=0.012", transform=ax.transAxes, facecolor=COLORS["orange"], edgecolor=COLORS["ink"], linewidth=1.4, zorder=4)
    ax.add_patch(ancilla)
    ax.plot([0.113, 0.164], [0.632, 0.691], transform=ax.transAxes, color=COLORS["ink"], lw=1.5, zorder=5)
    ax.plot([0.164, 0.113], [0.632, 0.691], transform=ax.transAxes, color=COLORS["ink"], lw=1.5, zorder=5)
    ax.text(0.1385, 0.718, "辅助比特", transform=ax.transAxes, fontsize=10.2, color=COLORS["ink"], ha="center")
    ax.plot([0.1385, 0.1385], [0.625, 0.550], transform=ax.transAxes, color="#E7A000", lw=7, solid_capstyle="butt", zorder=3)

    mode_center = (0.1385, 0.432)
    ax.add_patch(Circle(mode_center, 0.070, transform=ax.transAxes, facecolor=COLORS["blue_light"], edgecolor=COLORS["ink"], linewidth=1.5, zorder=3))
    for angle in np.linspace(0, 2 * np.pi, 8, endpoint=False):
        start = (mode_center[0] + 0.023 * np.cos(angle), mode_center[1] + 0.023 * np.sin(angle))
        end = (mode_center[0] + 0.055 * np.cos(angle), mode_center[1] + 0.055 * np.sin(angle))
        ax.plot([start[0], end[0]], [start[1], end[1]], transform=ax.transAxes, color="#43A6D6", lw=2.0, zorder=4)
    ax.add_patch(Circle(mode_center, 0.026, transform=ax.transAxes, facecolor="#9AA0A6", edgecolor=COLORS["ink"], linewidth=1.4, zorder=5))
    ax.text(0.1385, 0.320, "玻色模式", transform=ax.transAxes, fontsize=10.8, fontweight="bold", color=COLORS["ink"], ha="center")
    ax.text(0.1385, 0.287, "模化 q / p syndrome", transform=ax.transAxes, fontsize=8.8, color=COLORS["muted"], ha="center")

    # Recent history.
    _box(ax, 0.295, 0.600, 0.145, 0.155, COLORS["blue_light"], edgecolor=COLORS["ink"], lw=1.4, radius=0.022)
    ax.text(0.3675, 0.720, "近期 syndrome", transform=ax.transAxes, fontsize=10.7, fontweight="bold", color=COLORS["ink"], ha="center")
    ax.text(0.3675, 0.690, "历史  H_t", transform=ax.transAxes, fontsize=10.7, fontweight="bold", color=COLORS["ink"], ha="center")
    hist_x = np.linspace(0.323, 0.412, 7)
    hist_h = [0.025, 0.055, 0.083, 0.062, 0.045, 0.072, 0.035]
    for bx, bh in zip(hist_x, hist_h):
        ax.add_patch(Rectangle((bx, 0.625), 0.009, bh, transform=ax.transAxes, facecolor="#73A7D2", edgecolor="none", zorder=4))

    # Student and offline teacher.
    _box(ax, 0.490, 0.565, 0.185, 0.205, COLORS["teal_light"], edgecolor=COLORS["ink"], lw=1.5, radius=0.024)
    ax.text(0.5825, 0.724, "4状态轻量学生模型", transform=ax.transAxes, fontsize=11.7, fontweight="bold", color=COLORS["ink"], ha="center")
    ax.text(0.5825, 0.683, "95个浮点标量", transform=ax.transAxes, fontsize=10.2, color=COLORS["teal"], fontweight="bold", ha="center")
    ax.plot([0.515, 0.650], [0.657, 0.657], transform=ax.transAxes, color="#9ECFC8", lw=1.0)
    ax.text(0.5825, 0.623, "根据历史信息给出", transform=ax.transAxes, fontsize=9.5, color=COLORS["muted"], ha="center")
    ax.text(0.5825, 0.592, "慢速参数更新建议", transform=ax.transAxes, fontsize=9.5, color=COLORS["muted"], ha="center")

    _box(ax, 0.503, 0.790, 0.159, 0.070, COLORS["orange_light"], edgecolor=COLORS["teacher_edge"], lw=1.2, radius=0.020, linestyle="--")
    ax.text(0.5825, 0.826, "离线 GRU 教师模型", transform=ax.transAxes, fontsize=10.3, fontweight="bold", color=COLORS["teacher_edge"], ha="center", va="center")
    ax.text(0.5825, 0.798, "只用于训练，不进入在线回路", transform=ax.transAxes, fontsize=7.8, color=COLORS["muted"], ha="center", va="center")
    _arrow(ax, (0.5825, 0.790), (0.5825, 0.772), COLORS["teacher_edge"], lw=1.4, linestyle="--", mutation_scale=13)
    ax.text(0.603, 0.776, "离线蒸馏", transform=ax.transAxes, fontsize=8.2, color=COLORS["teacher_edge"], ha="left", va="center")

    # Safe staging.
    _box(ax, 0.720, 0.590, 0.195, 0.160, COLORS["orange_light"], edgecolor=COLORS["ink"], lw=1.5, radius=0.024)
    ax.text(0.8175, 0.708, "安全暂存 + 原子提交", transform=ax.transAxes, fontsize=11.3, fontweight="bold", color=COLORS["ink"], ha="center")
    ax.text(0.8175, 0.665, "版本号 / CRC / 完整镜像", transform=ax.transAxes, fontsize=9.5, color=COLORS["muted"], ha="center")
    ax.text(0.8175, 0.628, "仅在安全边界切换参数库", transform=ax.transAxes, fontsize=9.5, color=COLORS["muted"], ha="center")

    # Measurement and slow-loop arrows.
    _arrow(ax, (0.170, 0.662), (0.292, 0.662), "#A77043", lw=2.0)
    ax.text(0.232, 0.686, "测量结果", transform=ax.transAxes, fontsize=9.2, color="#8C603A", ha="center")
    _arrow(ax, (0.442, 0.676), (0.487, 0.676), COLORS["teal"], lw=2.2)
    ax.text(0.466, 0.707, "慢回路", transform=ax.transAxes, fontsize=8.8, color=COLORS["teal"], fontweight="bold", ha="center")
    _arrow(ax, (0.678, 0.676), (0.717, 0.676), COLORS["teal"], lw=2.2)
    ax.text(0.698, 0.707, "候选更新", transform=ax.transAxes, fontsize=8.2, color=COLORS["teal"], ha="center")

    # FPGA-facing fast path boundary.
    boundary = FancyBboxPatch(
        (0.425, 0.205), 0.548, 0.295,
        boxstyle="round,pad=0.012,rounding_size=0.025",
        facecolor="#FBFBFC",
        edgecolor=COLORS["ink"],
        linewidth=1.4,
        linestyle=(0, (4, 3)),
        transform=ax.transAxes,
        zorder=1,
    )
    ax.add_patch(boundary)
    ax.text(0.695, 0.513, "面向 FPGA 的确定性快路径（当前为软件参考）", transform=ax.transAxes, fontsize=11.0, fontweight="bold", color=COLORS["ink"], ha="center")

    _box(ax, 0.455, 0.270, 0.135, 0.155, COLORS["purple_light"], edgecolor=COLORS["ink"], lw=1.35, radius=0.020)
    ax.text(0.5225, 0.385, "双缓冲参数库", transform=ax.transAxes, fontsize=10.2, fontweight="bold", color=COLORS["ink"], ha="center")
    ax.text(0.5225, 0.348, "A   /   B", transform=ax.transAxes, fontsize=11, color=COLORS["purple"], fontweight="bold", ha="center")
    ax.text(0.5225, 0.307, "版本锁存", transform=ax.transAxes, fontsize=8.7, color=COLORS["muted"], ha="center")

    _box(ax, 0.627, 0.270, 0.135, 0.155, "#ECEAF6", edgecolor=COLORS["ink"], lw=1.35, radius=0.020)
    ax.text(0.6945, 0.385, "定点 MAP-LUT", transform=ax.transAxes, fontsize=10.3, fontweight="bold", color=COLORS["ink"], ha="center")
    ax.text(0.6945, 0.347, "地址 + 插值", transform=ax.transAxes, fontsize=9.1, color=COLORS["muted"], ha="center")
    ax.text(0.6945, 0.309, "舍入 / 饱和", transform=ax.transAxes, fontsize=9.1, color=COLORS["muted"], ha="center")

    _box(ax, 0.800, 0.270, 0.135, 0.155, COLORS["orange_light"], edgecolor=COLORS["ink"], lw=1.35, radius=0.020)
    ax.text(0.8675, 0.385, "事件 FSM + Pauli帧", transform=ax.transAxes, fontsize=9.8, fontweight="bold", color=COLORS["ink"], ha="center")
    ax.text(0.8675, 0.347, "纠正 / 保持 / 复位", transform=ax.transAxes, fontsize=8.9, color=COLORS["muted"], ha="center")
    ax.text(0.8675, 0.309, "异常时保守回退", transform=ax.transAxes, fontsize=8.9, color=COLORS["muted"], ha="center")

    _arrow(ax, (0.8175, 0.588), (0.8175, 0.500), COLORS["orange"], lw=2.1)
    ax.text(0.837, 0.545, "完整参数镜像", transform=ax.transAxes, fontsize=8.2, color=COLORS["orange"], ha="left", va="center")
    _arrow(ax, (0.592, 0.348), (0.624, 0.348), COLORS["muted"], lw=2.0)
    _arrow(ax, (0.764, 0.348), (0.797, 0.348), COLORS["muted"], lw=2.0)

    # Fast syndrome input and correction return.
    ax.plot([0.240, 0.382, 0.382, 0.452], [0.405, 0.405, 0.348, 0.348], transform=ax.transAxes, color=COLORS["purple"], lw=2.25, zorder=4)
    _arrow(ax, (0.382, 0.348), (0.452, 0.348), COLORS["purple"], lw=2.25)
    ax.text(0.328, 0.427, "模化 syndrome  s_t", transform=ax.transAxes, fontsize=9.0, color=COLORS["purple"], ha="center")
    ax.text(0.350, 0.375, "快回路 · 每次测量", transform=ax.transAxes, fontsize=8.8, color=COLORS["purple"], fontweight="bold", ha="center")

    ax.plot([0.937, 0.962, 0.962, 0.1385], [0.348, 0.348, 0.155, 0.155], transform=ax.transAxes, color=COLORS["orange"], lw=2.25, zorder=3)
    _arrow(ax, (0.1385, 0.155), (0.1385, 0.202), COLORS["orange"], lw=2.25)
    ax.text(0.630, 0.167, "位移纠正 / Pauli帧更新", transform=ax.transAxes, fontsize=9.4, color=COLORS["orange"], fontweight="bold", ha="center")

    _pill(ax, 0.270, 0.062, 0.235, "软件闭环 / 位精确参考已验证", COLORS["teal"], "#E9F6F3", icon_fill=COLORS["teal"], fontsize=9.4)
    _pill(ax, 0.525, 0.062, 0.230, "Student量化与RTL资源门待完成", COLORS["orange"], "#FFF7E9", icon_fill=None, fontsize=9.1)
    _pill(ax, 0.775, 0.062, 0.190, "真实FPGA板卡实测待完成", COLORS["muted"], "#F6F7F8", icon_fill=None, fontsize=8.8)
    ax.add_patch(Circle((0.795, 0.0845), 0.007, transform=ax.transAxes, facecolor="white", edgecolor=COLORS["muted"], linewidth=1.1, zorder=4))

    ax.text(0.035, 0.018, "说明：实线表示当前软件闭环契约；虚线模块表示候选部署边界，不代表已完成RTL或板卡验证。", transform=ax.transAxes, fontsize=8.5, color="#7C8792", ha="left", va="bottom")

    _save_figure(fig, "drift_adaptive_architecture_cn", dpi=300)


def main() -> None:
    _register_chinese_font()
    rows = _read_rows()
    make_lifetime_figure(rows)
    make_architecture_figure()
    print(f"Generated PPT figures in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
