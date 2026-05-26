from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[3]
ASSET_DIR = Path(__file__).resolve().parent
T54_CROSS_SEED = ROOT / "runs" / "T54_multi_seed_trace_phase_a_20260522" / "cross_seed_comparison.csv"
T55_INTERVENTION_SUMMARY = (
    ROOT
    / "runs"
    / "T55_multi_seed_i1_probe_20260523"
    / "analysis"
    / "intervention_summary.csv"
)
OUTPUT_CSV = ASSET_DIR / "figure_data.csv"
OUTPUT_MANIFEST = ASSET_DIR / "figure_manifest.json"
OUTPUT_CAPTION = ASSET_DIR / "caption.md"
OUTPUT_SVG = ASSET_DIR / "fr6_multi_seed_mechanism_intervention.svg"
OUTPUT_PNG = ASSET_DIR / "fr6_multi_seed_mechanism_intervention.png"

INSTABILITY_DELTA_THRESHOLD = 0.08
INSTABILITY_COMMITTED_THRESHOLD = 0.5
SEED_ORDER = ["20260425", "20260427", "20260428", "20260429", "20260430", "20260510"]

CATEGORY_COLORS = {
    "quiet": "#5b8ff9",
    "classic": "#f59f00",
    "universal": "#d94841",
    "other": "#7a7a7a",
}

VERDICT_COLORS = {
    "harmful": "#d94841",
    "mixed_or_no_clear_effect": "#f59f00",
    "helpful": "#2b8a3e",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def aggregate_t54(rows: list[dict[str, str]]) -> dict[str, dict[str, float | str | bool]]:
    grouped: dict[tuple[str, str], dict[str, float | int]] = defaultdict(
        lambda: {
            "sum_mean_window_ler": 0.0,
            "row_count": 0,
            "max_delta_b_norm": 0.0,
            "max_committed_b_norm": 0.0,
        }
    )
    for row in rows:
        key = (row["seed_source"], row["mode"])
        bucket = grouped[key]
        bucket["sum_mean_window_ler"] += float(row["mean_window_ler"])
        bucket["row_count"] += 1
        bucket["max_delta_b_norm"] = max(bucket["max_delta_b_norm"], float(row["max_delta_b_norm"]))
        bucket["max_committed_b_norm"] = max(
            bucket["max_committed_b_norm"], float(row["max_committed_b_norm"])
        )

    seed_data: dict[str, dict[str, float | str | bool]] = {}
    for seed in SEED_ORDER:
        full_key = (seed, "hybrid_full")
        gated_key = (seed, "hybrid_gated_teacher_v5")
        if full_key not in grouped or gated_key not in grouped:
            raise RuntimeError(f"missing T54 rows for seed {seed}")

        full = grouped[full_key]
        gated = grouped[gated_key]
        full_mean = float(full["sum_mean_window_ler"]) / int(full["row_count"])
        gated_mean = float(gated["sum_mean_window_ler"]) / int(gated["row_count"])
        full_instability = (
            float(full["max_delta_b_norm"]) > INSTABILITY_DELTA_THRESHOLD
            and float(full["max_committed_b_norm"]) > INSTABILITY_COMMITTED_THRESHOLD
        )
        gated_instability = (
            float(gated["max_delta_b_norm"]) > INSTABILITY_DELTA_THRESHOLD
            and float(gated["max_committed_b_norm"]) > INSTABILITY_COMMITTED_THRESHOLD
        )
        if gated_instability and not full_instability:
            category = "classic"
        elif gated_instability and full_instability:
            category = "universal"
        elif not gated_instability and not full_instability:
            category = "quiet"
        else:
            category = "other"

        seed_data[seed] = {
            "seed": seed,
            "full_mean_ler": full_mean,
            "gated_v5_mean_ler": gated_mean,
            "baseline_gap_gv5_minus_full": gated_mean - full_mean,
            "full_max_delta_b_norm": float(full["max_delta_b_norm"]),
            "full_max_committed_b_norm": float(full["max_committed_b_norm"]),
            "gated_v5_max_delta_b_norm": float(gated["max_delta_b_norm"]),
            "gated_v5_max_committed_b_norm": float(gated["max_committed_b_norm"]),
            "full_instability": full_instability,
            "gated_v5_instability": gated_instability,
            "seed_category": category,
            "t54_rows_full": int(full["row_count"]),
            "t54_rows_gated_v5": int(gated["row_count"]),
        }
    return seed_data


def attach_t55(seed_data: dict[str, dict[str, float | str | bool]], rows: list[dict[str, str]]) -> None:
    seen = set()
    for row in rows:
        seed = row["seed"]
        if seed not in seed_data:
            raise RuntimeError(f"unexpected T55 seed {seed}")
        seed_data[seed]["i1_mean_gap_minus_baseline"] = float(row["mean_gap_i1_minus_bl"])
        seed_data[seed]["i1_verdict"] = row["verdict"]
        seed_data[seed]["t55_n_scenarios"] = int(row["n_scenarios_with_data"])
        seen.add(seed)
    missing = sorted(set(seed_data) - seen)
    if missing:
        raise RuntimeError(f"missing T55 summary rows for seeds: {missing}")


def write_figure_data(seed_data: dict[str, dict[str, float | str | bool]]) -> None:
    fieldnames = [
        "seed",
        "seed_category",
        "full_mean_ler",
        "gated_v5_mean_ler",
        "baseline_gap_gv5_minus_full",
        "full_max_delta_b_norm",
        "full_max_committed_b_norm",
        "gated_v5_max_delta_b_norm",
        "gated_v5_max_committed_b_norm",
        "full_instability",
        "gated_v5_instability",
        "t54_rows_full",
        "t54_rows_gated_v5",
        "i1_mean_gap_minus_baseline",
        "i1_verdict",
        "t55_n_scenarios",
    ]
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for seed in SEED_ORDER:
            writer.writerow(seed_data[seed])


def write_caption() -> None:
    caption = """# Caption

Figure FR6 summarizes the bounded six-seed mechanism/intervention evidence already established by T54 and T55. Panel A plots the seed-wise baseline gap `mean(Gated v5) - mean(Full)` from T54, with seed labels grouped into the descriptive `quiet`, `classic`, and `universal` categories derived from the observed committed-`b` instability pattern. Negative values mean Gated v5 performs better than Full. Panel B plots the seed-wise I1 intervention delta `mean(I1) - mean(Gated v5 baseline)` from T55, where positive values mean the lower-clip intervention is worse than the original Gated v5 baseline. The figure is descriptive only: it shows that the instability pattern is broadly present across seeds and that the tested clip-reduction intervention has mixed, mostly harmful, outcomes. It must not be read as causal proof, mechanism closure, expanded benchmark evidence, `.tflite` validation, or real-board validation.
"""
    OUTPUT_CAPTION.write_text(caption, encoding="utf-8")


def write_manifest(seed_data: dict[str, dict[str, float | str | bool]]) -> None:
    manifest = {
        "task_id": "T58",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generator_script": str(Path(__file__).name),
        "output_files": [
            OUTPUT_SVG.name,
            OUTPUT_PNG.name,
            OUTPUT_CSV.name,
            OUTPUT_MANIFEST.name,
            OUTPUT_CAPTION.name,
        ],
        "sources": {
            "t54_cross_seed_comparison_csv": {
                "path": str(T54_CROSS_SEED.relative_to(ROOT)).replace("\\", "/"),
                "columns_used": [
                    "seed_source",
                    "mode",
                    "mean_window_ler",
                    "max_delta_b_norm",
                    "max_committed_b_norm",
                ],
                "aggregation": (
                    "For each seed_source and mode, take the simple mean of mean_window_ler across "
                    "all available rows; take the max of max_delta_b_norm and max_committed_b_norm."
                ),
            },
            "t55_intervention_summary_csv": {
                "path": str(T55_INTERVENTION_SUMMARY.relative_to(ROOT)).replace("\\", "/"),
                "columns_used": ["seed", "verdict", "mean_gap_i1_minus_bl", "n_scenarios_with_data"],
                "aggregation": "Use per-seed mean_gap_i1_minus_bl and verdict directly from the summary table.",
            },
        },
        "derived_definitions": {
            "baseline_gap_gv5_minus_full": "gated_v5_mean_ler - full_mean_ler; negative means Gated v5 is better.",
            "i1_mean_gap_minus_baseline": "mean(I1) - mean(Gated v5 baseline); positive means I1 is worse.",
            "instability_rule": (
                "mode is treated as unstable if max_delta_b_norm > 0.08 and "
                "max_committed_b_norm > 0.5"
            ),
            "seed_category_rule": {
                "quiet": "Neither Full nor Gated v5 meets the instability rule.",
                "classic": "Gated v5 meets the instability rule and Full does not.",
                "universal": "Both Full and Gated v5 meet the instability rule.",
                "other": "Fallback category if a new pattern appears.",
            },
        },
        "seed_rows": [seed_data[seed] for seed in SEED_ORDER],
        "non_claims": [
            "No new experiment was run to build this figure pack.",
            "The figure does not prove causality or close the mechanism story.",
            "The figure does not expand benchmark scope beyond the frozen six-seed pack.",
        ],
    }
    OUTPUT_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def plot(seed_data: dict[str, dict[str, float | str | bool]]) -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )

    seeds = SEED_ORDER
    x_positions = list(range(len(seeds)))
    baseline_gaps = [float(seed_data[seed]["baseline_gap_gv5_minus_full"]) for seed in seeds]
    baseline_colors = [
        CATEGORY_COLORS.get(str(seed_data[seed]["seed_category"]), CATEGORY_COLORS["other"]) for seed in seeds
    ]
    i1_gaps = [float(seed_data[seed]["i1_mean_gap_minus_baseline"]) for seed in seeds]
    i1_colors = [VERDICT_COLORS[str(seed_data[seed]["i1_verdict"])] for seed in seeds]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4), constrained_layout=True)

    axes[0].bar(x_positions, baseline_gaps, color=baseline_colors, edgecolor="#303030", linewidth=0.8)
    axes[0].axhline(0.0, color="#303030", linewidth=1.0)
    axes[0].set_xticks(x_positions, seeds, rotation=0)
    axes[0].set_ylabel("T54 mean(Gated v5) - mean(Full) LER")
    axes[0].set_title("Panel A. Cross-seed baseline gap from T54")
    axes[0].grid(axis="y", linestyle="--", alpha=0.35)
    for xpos, seed in zip(x_positions, seeds):
        value = float(seed_data[seed]["baseline_gap_gv5_minus_full"])
        category = str(seed_data[seed]["seed_category"])
        va = "bottom" if value >= 0 else "top"
        y_offset = 0.008 if value >= 0 else -0.008
        axes[0].text(xpos, value + y_offset, category, ha="center", va=va, fontsize=8)

    axes[1].bar(x_positions, i1_gaps, color=i1_colors, edgecolor="#303030", linewidth=0.8)
    axes[1].axhline(0.0, color="#303030", linewidth=1.0)
    axes[1].set_xticks(x_positions, seeds, rotation=0)
    axes[1].set_ylabel("T55 mean(I1) - mean(Gated v5 baseline) LER")
    axes[1].set_title("Panel B. Seed-wise I1 intervention delta from T55")
    axes[1].grid(axis="y", linestyle="--", alpha=0.35)
    for xpos, seed in zip(x_positions, seeds):
        value = float(seed_data[seed]["i1_mean_gap_minus_baseline"])
        verdict = str(seed_data[seed]["i1_verdict"])
        va = "bottom" if value >= 0 else "top"
        y_offset = 0.01 if value >= 0 else -0.01
        axes[1].text(xpos, value + y_offset, verdict.replace("_", " "), ha="center", va=va, fontsize=8)

    fig.suptitle(
        "FR6 multi-seed mechanism/intervention summary (descriptive only, not causal proof)",
        fontsize=13,
    )
    fig.savefig(OUTPUT_SVG, format="svg", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_PNG, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    t54_rows = read_csv(T54_CROSS_SEED)
    t55_rows = read_csv(T55_INTERVENTION_SUMMARY)
    seed_data = aggregate_t54(t54_rows)
    attach_t55(seed_data, t55_rows)
    write_figure_data(seed_data)
    write_caption()
    write_manifest(seed_data)
    plot(seed_data)


if __name__ == "__main__":
    main()
