"""Summarize the formal `features` ablation results for Hybrid-vs-UKF study."""

# 中文说明：
# - 该脚本服务于论文/汇报整理，不负责重新运行 benchmark。
# - 它优先读取最新的 `p4_hybrid_vs_ukf_ablation_features_v1_*` 结果目录，
#   自动生成 markdown/csv/json 三类摘要文件。
# - 若正式 benchmark 还未完成，它会退化为“进度汇总模式”，报告四个 feature 变体
#   当前分别完成到 dataset / train / evaluate / benchmark 的哪一步，便于后台长任务期间查看状态。

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

from cnn_fpga.benchmark.run_p4_hybrid_vs_ukf_ablation import EXPERIMENT_CONFIGS
from cnn_fpga.utils.config import ensure_dir, load_yaml_config, now_tag, save_json


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = REPO_ROOT / "runs" / "p4_benchmark"
ABALATION_RUN_ROOT = REPO_ROOT / "runs" / "p4_hybrid_vs_ukf_ablation"
DEFAULT_PREFIX = "p4_hybrid_vs_ukf_ablation_features_v1_"

MODE_ORDER = [
    "ukf",
    "hybrid_full",
    "hybrid_no_hist_deltas",
    "hybrid_no_teacher_prediction",
    "hybrid_no_teacher_params",
    "hybrid_no_teacher_deltas",
]

MODE_DESCRIPTIONS: Mapping[str, str] = {
    "ukf": "最强经典 baseline",
    "hybrid_full": "完整 teacher-guided residual-b 主模型",
    "hybrid_no_hist_deltas": "去掉 histogram delta 通道",
    "hybrid_no_teacher_prediction": "去掉 teacher prediction 通道",
    "hybrid_no_teacher_params": "去掉 teacher params 通道",
    "hybrid_no_teacher_deltas": "去掉 teacher delta 通道",
}

VARIANT_TO_EXPERIMENT: Mapping[str, str] = {
    "hybrid_no_hist_deltas": "no_hist_deltas",
    "hybrid_no_teacher_prediction": "no_teacher_prediction",
    "hybrid_no_teacher_params": "no_teacher_params",
    "hybrid_no_teacher_deltas": "no_teacher_deltas",
}

SPECIAL_MODE_PATHS: Mapping[str, Mapping[str, str]] = {
    # 中文注释：`hybrid_full` 复用正式主模型，不会在 features 组里重新 prepare。
    "hybrid_full": {
        "dataset_dir": "artifacts/datasets/runtime_b_residual_v1",
        "model_dir": "artifacts/models/runtime_b_residual_v1",
        "report_dir": "artifacts/reports/runtime_b_residual_v1",
    }
}


@dataclass(frozen=True)
class VariantProgress:
    """Track prepare-stage outputs for one ablation variant."""

    mode: str
    label: str
    dataset_ready: bool
    model_ready: bool
    eval_ready: bool
    dataset_dir: str | None
    model_dir: str | None
    report_dir: str | None

    @property
    def prepare_stage(self) -> str:
        if self.eval_ready:
            return "evaluate_done"
        if self.model_ready:
            return "train_done"
        if self.dataset_ready:
            return "dataset_done"
        return "not_started"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "label": self.label,
            "dataset_ready": self.dataset_ready,
            "model_ready": self.model_ready,
            "eval_ready": self.eval_ready,
            "prepare_stage": self.prepare_stage,
            "dataset_dir": self.dataset_dir,
            "model_dir": self.model_dir,
            "report_dir": self.report_dir,
        }


def _arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize P4 features ablation results.")
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Optional explicit features benchmark run directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional output directory. Default: runs/p4_features_summary/<timestamp>/",
    )
    return parser


def _latest_features_run() -> Path | None:
    candidates = sorted(RUN_ROOT.glob(f"{DEFAULT_PREFIX}*"), key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: List[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _load_variant_progress(mode_name: str, label: str) -> VariantProgress:
    special_paths = SPECIAL_MODE_PATHS.get(mode_name)
    if special_paths is not None:
        dataset_dir = REPO_ROOT / str(special_paths["dataset_dir"])
        model_dir = REPO_ROOT / str(special_paths["model_dir"])
        report_dir = REPO_ROOT / str(special_paths["report_dir"])
        return VariantProgress(
            mode=mode_name,
            label=label,
            dataset_ready=dataset_dir.exists() and any(dataset_dir.glob("*.npz")) and (dataset_dir / "manifest.json").exists(),
            model_ready=model_dir.exists() and any(model_dir.glob("*.npz")),
            eval_ready=report_dir.exists() and any(report_dir.glob("eval_test_*.json")),
            dataset_dir=str(dataset_dir),
            model_dir=str(model_dir),
            report_dir=str(report_dir),
        )

    experiment_key = VARIANT_TO_EXPERIMENT.get(mode_name)
    if experiment_key is None:
        return VariantProgress(
            mode=mode_name,
            label=label,
            dataset_ready=False,
            model_ready=False,
            eval_ready=False,
            dataset_dir=None,
            model_dir=None,
            report_dir=None,
        )

    cfg = load_yaml_config(EXPERIMENT_CONFIGS[experiment_key])
    paths = cfg.get("paths", {})
    dataset_dir = REPO_ROOT / str(paths.get("dataset_dir", ""))
    model_dir = REPO_ROOT / str(paths.get("model_dir", ""))
    report_dir = REPO_ROOT / str(paths.get("report_dir", ""))

    dataset_ready = dataset_dir.exists() and any(dataset_dir.glob("*.npz")) and (dataset_dir / "manifest.json").exists()
    model_ready = model_dir.exists() and any(model_dir.glob("*.npz"))
    eval_ready = report_dir.exists() and any(report_dir.glob("eval_test_*.json"))

    return VariantProgress(
        mode=mode_name,
        label=label,
        dataset_ready=dataset_ready,
        model_ready=model_ready,
        eval_ready=eval_ready,
        dataset_dir=str(dataset_dir),
        model_dir=str(model_dir),
        report_dir=str(report_dir),
    )


def _find_latest_ablation_session() -> Path | None:
    candidates = sorted(ABALATION_RUN_ROOT.glob("ablation_*"), key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _load_benchmark_config() -> Dict[str, Any]:
    cfg_path = REPO_ROOT / "cnn_fpga" / "config" / "p4_hybrid_vs_ukf_ablation_features.yaml"
    return load_yaml_config(cfg_path)


def _mode_label_map() -> Dict[str, str]:
    cfg = _load_benchmark_config()
    modes = cfg.get("p4_benchmark", {}).get("modes", [])
    return {str(item.get("name")): str(item.get("label", item.get("name"))) for item in modes}


def _collect_progress_payload() -> Dict[str, Any]:
    label_map = _mode_label_map()
    variant_progress = [_load_variant_progress(mode_name, label_map.get(mode_name, mode_name)) for mode_name in MODE_ORDER]
    latest_session = _find_latest_ablation_session()
    latest_run = _latest_features_run()
    return {
        "status": "running_or_pending",
        "latest_features_run_dir": None if latest_run is None else str(latest_run),
        "latest_ablation_session_dir": None if latest_session is None else str(latest_session),
        "variants": [item.to_dict() for item in variant_progress],
    }


def _aggregate_by_mode(rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(str(row["mode"]), []).append(row)

    if "ukf" not in grouped or "hybrid_full" not in grouped:
        raise ValueError("comparison.csv must contain both `ukf` and `hybrid_full`.")

    def _avg(items: List[Dict[str, str]], key: str) -> float:
        return float(sum(float(item[key]) for item in items) / len(items))

    ukf_avg = _avg(grouped["ukf"], "final_ler_mean")
    full_avg = _avg(grouped["hybrid_full"], "final_ler_mean")

    normalized_rows: List[Dict[str, Any]] = []
    for mode_name in MODE_ORDER:
        if mode_name not in grouped:
            continue
        items = grouped[mode_name]
        sample = items[0]
        normalized_rows.append(
            {
                "mode": mode_name,
                "mode_label": sample["mode_label"],
                "description": MODE_DESCRIPTIONS.get(mode_name, ""),
                "n_scenarios": len(items),
                "avg_ler": _avg(items, "final_ler_mean"),
                "avg_overflow": _avg(items, "overflow_rate_mean"),
                "avg_histogram_saturation": _avg(items, "histogram_input_saturation_rate_mean"),
                "avg_commits": _avg(items, "n_commits_applied_mean"),
                "avg_slow_violation": _avg(items, "slow_update_violation_rate_mean"),
                "avg_fast_violation": _avg(items, "fast_cycle_violation_rate_mean"),
                "delta_ler_vs_ukf": _avg(items, "final_ler_mean") - ukf_avg,
                "delta_ler_vs_hybrid_full": _avg(items, "final_ler_mean") - full_avg,
                "artifact_path": sample.get("artifact_path") or None,
                "dominant_overflow_sources": sorted({str(item.get("dominant_overflow_source")) for item in items}),
            }
        )
    return normalized_rows


def _summarize_completed_run(run_dir: Path) -> Dict[str, Any]:
    comparison_path = run_dir / "comparison.csv"
    summary_path = run_dir / "summary.json"
    if not comparison_path.exists() or not summary_path.exists():
        raise FileNotFoundError(f"Expected completed benchmark outputs under: {run_dir}")

    rows = _read_csv_rows(comparison_path)
    if not rows:
        raise ValueError("No feature ablation rows found in comparison.csv")
    aggregated_rows = _aggregate_by_mode(rows)
    best_row = min(aggregated_rows, key=lambda item: (float(item["avg_ler"]), float(item["avg_overflow"])))
    summary_json = json.loads(summary_path.read_text(encoding="utf-8"))
    return {
        "status": "completed",
        "run_dir": str(run_dir),
        "comparison_csv": str(comparison_path),
        "summary_json": str(summary_path),
        "best_mode": best_row["mode"],
        "best_mode_label": best_row["mode_label"],
        "aggregated_rows": aggregated_rows,
        "scenario_rows": rows,
        "protocol": summary_json.get("protocol", {}),
    }


def _write_markdown_report(path: Path, payload: Dict[str, Any]) -> None:
    lines: List[str] = [
        "# P4 Features Ablation Summary",
        "",
    ]

    if payload["status"] != "completed":
        lines.extend(
            [
                "## Status",
                "",
                "- features 正式 benchmark 还未完成。",
                f"- latest_features_run_dir: `{payload.get('latest_features_run_dir')}`",
                f"- latest_ablation_session_dir: `{payload.get('latest_ablation_session_dir')}`",
                "",
                "## Prepare Progress",
                "",
                "| Mode | Label | Prepare Stage | Dataset | Model | Eval |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
        )
        for item in payload.get("variants", []):
            lines.append(
                f"| {item['mode']} | {item['label']} | {item['prepare_stage']} | "
                f"{'yes' if item['dataset_ready'] else 'no'} | "
                f"{'yes' if item['model_ready'] else 'no'} | "
                f"{'yes' if item['eval_ready'] else 'no'} |"
            )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    lines.extend(
        [
            "## Protocol",
            "",
            f"- run_dir: `{payload['run_dir']}`",
            f"- best_mode: `{payload['best_mode_label']}`",
            "",
            "## Aggregated Table",
            "",
            "| Mode | Description | Avg LER | dLER vs UKF | dLER vs Hybrid Full | Avg Overflow | Avg Hist Sat | Avg Commits | Overflow Sources |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in payload["aggregated_rows"]:
        lines.append(
            f"| {row['mode_label']} | {row['description']} | "
            f"{float(row['avg_ler']):.6f} | {float(row['delta_ler_vs_ukf']):+.6f} | "
            f"{float(row['delta_ler_vs_hybrid_full']):+.6f} | {float(row['avg_overflow']):.6f} | "
            f"{float(row['avg_histogram_saturation']):.6f} | {float(row['avg_commits']):.2f} | "
            f"{', '.join(row['dominant_overflow_sources'])} |"
        )

    lines.extend(
        [
            "",
            "## Scenario Rows",
            "",
            "| Scenario | Mode | LER Mean | Overflow | Hist Sat | Aggressive Source |",
            "| --- | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for row in payload["scenario_rows"]:
        lines.append(
            f"| {row['scenario']} | {row['mode_label']} | {float(row['final_ler_mean']):.6f} | "
            f"{float(row['overflow_rate_mean']):.6f} | {float(row['histogram_input_saturation_rate_mean']):.6f} | "
            f"{row['dominant_overflow_source']} |"
        )

    lines.extend(
        [
            "",
            "## Reading Guide",
            "",
            "- `dLER vs UKF < 0` 说明该模式优于 `UKF`。",
            "- `dLER vs Hybrid Full > 0` 说明去掉该类输入后性能变差，意味着该输入通道是有价值的。",
            "- 若 overflow 接近但 LER 明显变差，更支持“性能差异来自估计质量而非饱和差异”。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_latex_table(path: Path, payload: Dict[str, Any]) -> None:
    if payload["status"] != "completed":
        path.write_text("% features benchmark not completed yet\n", encoding="utf-8")
        return

    lines = [
        "% Auto-generated by summarize_p4_features_ablation.py",
        "\\begin{tabular}{l l l r r r}",
        "\\hline",
        "Mode & Scenario & Description & LER & $\\Delta$ vs UKF & $\\Delta$ vs Full \\\\",
        "\\hline",
    ]
    for row in payload["aggregated_rows"]:
        lines.append(
            f"{row['mode_label']} & avg & {row['description']} & "
            f"{float(row['avg_ler']):.6f} & {float(row['delta_ler_vs_ukf']):+.6f} & "
            f"{float(row['delta_ler_vs_hybrid_full']):+.6f} \\\\"
        )
    lines.extend(["\\hline", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = _arg_parser().parse_args()
    if args.run_dir:
        run_dir = Path(args.run_dir).expanduser().resolve()
    else:
        run_dir = _latest_features_run()

    if args.output_dir:
        out_dir = ensure_dir(args.output_dir)
    else:
        out_dir = ensure_dir(REPO_ROOT / "runs" / "p4_features_summary" / f"features_summary_{now_tag()}")

    if run_dir is None or not (run_dir / "comparison.csv").exists() or not (run_dir / "summary.json").exists():
        payload = _collect_progress_payload()
    else:
        payload = _summarize_completed_run(run_dir)

    save_json(out_dir / "summary.json", payload)

    if payload["status"] == "completed":
        _write_csv(
            out_dir / "table.csv",
            payload["aggregated_rows"],
            [
                "mode",
                "mode_label",
                "description",
                "n_scenarios",
                "avg_ler",
                "delta_ler_vs_ukf",
                "delta_ler_vs_hybrid_full",
                "avg_overflow",
                "avg_histogram_saturation",
                "avg_commits",
                "avg_slow_violation",
                "avg_fast_violation",
                "dominant_overflow_sources",
                "artifact_path",
            ],
        )
        _write_csv(
            out_dir / "scenario_rows.csv",
            payload["scenario_rows"],
            [
                "scenario",
                "scenario_label",
                "mode",
                "mode_label",
                "artifact_path",
                "final_ler_mean",
                "final_ler_std",
                "overflow_rate_mean",
                "overflow_rate_std",
                "histogram_input_saturation_rate_mean",
                "histogram_input_saturation_rate_std",
                "correction_saturation_rate_mean",
                "aggressive_param_rate_mean",
                "n_commits_applied_mean",
                "slow_update_violation_rate_mean",
                "fast_cycle_violation_rate_mean",
                "dominant_overflow_source",
            ],
        )
    else:
        _write_csv(
            out_dir / "progress.csv",
            payload["variants"],
            [
                "mode",
                "label",
                "prepare_stage",
                "dataset_ready",
                "model_ready",
                "eval_ready",
                "dataset_dir",
                "model_dir",
                "report_dir",
            ],
        )

    _write_markdown_report(out_dir / "report.md", payload)
    _write_latex_table(out_dir / "table.tex", payload)

    print(f"features_summary_dir={out_dir}")
    print(f"status={payload['status']}")
    if payload["status"] == "completed":
        print(f"best_mode={payload['best_mode_label']}")
    else:
        for item in payload["variants"]:
            print(
                f"[{item['mode']}] stage={item['prepare_stage']} "
                f"dataset={item['dataset_ready']} model={item['model_ready']} eval={item['eval_ready']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
