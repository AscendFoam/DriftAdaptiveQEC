"""Build row-level provenance for the submission draft main benchmark.

This helper reads the preserved T24 software-HIL run metadata and emits a
row-level manifest for the manuscript-facing benchmark rows. It is not a new
benchmark, not an inferential analysis, and not a hardware provenance record.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = ROOT / "runs/p4_benchmark/T24_formal_software_revalidation_20260510_200743"
SUMMARY_JSON = RUN_DIR / "summary.json"
LAUNCH_PLAN_JSON = RUN_DIR / "launch_plan.json"
COMPARISON_CSV = RUN_DIR / "comparison.csv"
CONFIG_PATH = ROOT / "cnn_fpga/config/p4_multiscenario_strong_baselines.yaml"
RUNNER_PATH = ROOT / "cnn_fpga/benchmark/run_p4_multiscenario_benchmark.py"

OUT_CSV = ROOT / "docs/paper_materials/submission_draft_row_provenance_manifest.csv"
OUT_JSON = ROOT / "docs/paper_materials/submission_draft_row_provenance_manifest.json"
OUT_MD = ROOT / "docs/paper_materials/投稿稿row_level_provenance补强记录.md"

SCENARIOS = ("static_bias_theta", "linear_ramp", "step_sigma_theta", "periodic_drift")
MODES = ("ekf", "ukf", "constant_residual_mu", "rls_residual_b", "hybrid_residual_b")

NON_CLAIMS = [
    "not new benchmark evidence",
    "not CI, p-value, standard error or robustness evidence",
    "not holdout-drift validation",
    "not training reproducibility closure",
    "not tflite portability or HIL closure",
    "not real-board execution, source-vs-board agreement, latency, resource or FPGA evidence",
    "not statistical-calibration comparator promotion",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha_or_empty(path_text: str | None) -> str:
    if not path_text:
        return ""
    path = Path(path_text)
    if not path.is_absolute():
        path = ROOT / path
    return sha256_file(path) if path.is_file() else ""


def rel(path: Path | str | None) -> str:
    if path is None or str(path) == "":
        return ""
    path_obj = Path(path)
    try:
        return str(path_obj.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path_obj)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_comparison() -> dict[tuple[str, str], dict[str, str]]:
    with COMPARISON_CSV.open("r", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return {(row["scenario"], row["mode"]): row for row in rows}


def build_rows() -> list[dict[str, str]]:
    summary = read_json(SUMMARY_JSON)
    launch = read_json(LAUNCH_PLAN_JSON)
    comparison = read_comparison()

    summary_sha = sha256_file(SUMMARY_JSON)
    launch_sha = sha256_file(LAUNCH_PLAN_JSON)
    comparison_sha = sha256_file(COMPARISON_CSV)
    config_sha = sha256_file(CONFIG_PATH)
    runner_sha = sha256_file(RUNNER_PATH)

    rows: list[dict[str, str]] = []
    for raw in summary["raw_rows"]:
        scenario = raw["scenario"]
        mode = raw["mode"]
        repeat = str(raw["repeat"])
        if scenario not in SCENARIOS or mode not in MODES:
            continue
        run_dir = Path(raw["run_dir"])
        comparison_row = comparison[(scenario, mode)]
        artifact_path = raw.get("artifact_path") or ""
        row = {
            "row_id": f"{scenario}/{mode}/repeat_{int(repeat):02d}",
            "scenario": scenario,
            "mode": mode,
            "repeat": repeat,
            "seed": str(raw["seed"]),
            "run_dir": rel(run_dir),
            "final_ler": str(raw["final_ler"]),
            "comparison_final_ler_mean": comparison_row["final_ler_mean"],
            "comparison_completed_repeats": comparison_row["completed_repeats"],
            "comparison_expected_repeats": comparison_row["expected_repeats"],
            "comparison_coverage": comparison_row["coverage"],
            "git_commit": str(summary["git_commit"]),
            "config_hash_short": str(summary["config_hash"]),
            "protocol_id": str(summary["protocol"]["protocol_id"]),
            "paired_seeds": str(launch["paired_seeds"]).lower(),
            "summary_json": rel(SUMMARY_JSON),
            "summary_sha256": summary_sha,
            "launch_plan_json": rel(LAUNCH_PLAN_JSON),
            "launch_plan_sha256": launch_sha,
            "comparison_csv": rel(COMPARISON_CSV),
            "comparison_sha256": comparison_sha,
            "config_path": rel(CONFIG_PATH),
            "config_sha256": config_sha,
            "runner_path": rel(RUNNER_PATH),
            "runner_sha256": runner_sha,
            "run_hil_summary_sha256": sha_or_empty(str(run_dir / "hil_summary.json")),
            "run_repeat_status_sha256": sha_or_empty(str(run_dir / "repeat_status.json")),
            "artifact_path": rel(artifact_path),
            "artifact_sha256": sha_or_empty(artifact_path),
            "provenance_scope": "row-level source trace for existing software-HIL benchmark rows",
            "non_claim_boundary": "; ".join(NON_CLAIMS),
        }
        rows.append(row)

    rows.sort(key=lambda row: (SCENARIOS.index(row["scenario"]), MODES.index(row["mode"]), int(row["repeat"])))
    return rows


def write_outputs(rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "row_id",
        "scenario",
        "mode",
        "repeat",
        "seed",
        "run_dir",
        "final_ler",
        "comparison_final_ler_mean",
        "comparison_completed_repeats",
        "comparison_expected_repeats",
        "comparison_coverage",
        "git_commit",
        "config_hash_short",
        "protocol_id",
        "paired_seeds",
        "summary_json",
        "summary_sha256",
        "launch_plan_json",
        "launch_plan_sha256",
        "comparison_csv",
        "comparison_sha256",
        "config_path",
        "config_sha256",
        "runner_path",
        "runner_sha256",
        "run_hil_summary_sha256",
        "run_repeat_status_sha256",
        "artifact_path",
        "artifact_sha256",
        "provenance_scope",
        "non_claim_boundary",
    ]
    with OUT_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    OUT_JSON.write_text(
        json.dumps(
            {
                "manifest_scope": (
                    "Row-level source trace for the preserved T24 software-HIL "
                    "benchmark rows used by the submission draft."
                ),
                "row_count": len(rows),
                "scenarios": list(SCENARIOS),
                "modes": list(MODES),
                "non_claims": NON_CLAIMS,
                "excluded_from_hash_closure": [
                    "hil_events.json is not hashed here because this is not a recursive run-directory closure",
                    "hardware logs, bitstreams, DMA/MMIO traces and source-vs-board vectors are absent",
                ],
                "rows": rows,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    OUT_MD.write_text(
        "\n".join(
            [
                "# 投稿稿 row-level provenance 补强记录",
                "",
                "## 目的",
                "",
                "本记录为当前投稿稿主 benchmark 行补充 row-level source trace。它只读取既有 T24 software-HIL 运行的 summary、launch plan、comparison.csv、配置和 runner 文件；不重新运行 benchmark，也不补硬件或统计推断证据。",
                "",
                "## 输出",
                "",
                f"- `{rel(OUT_CSV)}`",
                f"- `{rel(OUT_JSON)}`",
                "",
                "## 覆盖范围",
                "",
                f"- row_count = {len(rows)}",
                "- scenarios = `static_bias_theta`, `linear_ramp`, `step_sigma_theta`, `periodic_drift`",
                "- modes = `ekf`, `ukf`, `constant_residual_mu`, `rls_residual_b`, `hybrid_residual_b`",
                "- repeats = `0`, `1`",
                "",
                "## 不可外推边界",
                "",
                *[f"- {item}" for item in NON_CLAIMS],
                "",
                "## 明确缺口",
                "",
                "- 本 manifest 不递归 hash `hil_events.json`，也不是 historical run directory 的完整 hash closure。",
                "- 本 manifest 不包含 board log、bitstream/RTL hash、DMA/MMIO trace、source-vs-board vector、latency/resource/power measurement。",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    rows = build_rows()
    expected = len(SCENARIOS) * len(MODES) * 2
    if len(rows) != expected:
        raise RuntimeError(f"expected {expected} rows, got {len(rows)}")
    write_outputs(rows)
    print(json.dumps({"status": "ok", "rows": len(rows), "csv": str(OUT_CSV), "json": str(OUT_JSON)}))


if __name__ == "__main__":
    main()
