"""Run hybrid-vs-UKF ablation suite end-to-end."""

# 中文说明：
# - 该脚本负责把 `hybrid vs ukf` 的多组 ablation 串起来执行。
# - 它复用现有模块，不重新实现训练或 benchmark 逻辑：
#   1. `runtime_dataset_builder`
#   2. `model.train`
#   3. `model.evaluate`
#   4. `benchmark.run_p4_multiscenario_benchmark`
# - 这样可以保证 ablation 与当前正式主线使用同一套运行时和统计口径。

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

from cnn_fpga.utils.config import ensure_dir, load_yaml_config, now_tag, save_json


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "cnn_fpga" / "config"

EXPERIMENT_CONFIGS: Dict[str, Path] = {
    "teacher_ukf": CONFIG_DIR / "experiment_runtime_b_residual_teacher_ukf.yaml",
    "ctx1": CONFIG_DIR / "experiment_runtime_b_residual_ctx1.yaml",
    "ctx3": CONFIG_DIR / "experiment_runtime_b_residual_ctx3.yaml",
    "no_teacher_prediction": CONFIG_DIR / "experiment_runtime_b_residual_no_teacher_prediction.yaml",
    "no_teacher_params": CONFIG_DIR / "experiment_runtime_b_residual_no_teacher_params.yaml",
    "no_teacher_deltas": CONFIG_DIR / "experiment_runtime_b_residual_no_teacher_deltas.yaml",
    "no_hist_deltas": CONFIG_DIR / "experiment_runtime_b_residual_no_hist_deltas.yaml",
}

BENCHMARK_CONFIGS: Dict[str, Path] = {
    "teacher": CONFIG_DIR / "p4_hybrid_vs_ukf_ablation_teacher.yaml",
    "features": CONFIG_DIR / "p4_hybrid_vs_ukf_ablation_features.yaml",
    "context": CONFIG_DIR / "p4_hybrid_vs_ukf_ablation_context.yaml",
}

GROUP_VARIANTS: Dict[str, List[str]] = {
    "teacher": ["teacher_ukf"],
    "features": ["no_hist_deltas", "no_teacher_prediction", "no_teacher_params", "no_teacher_deltas"],
    "context": ["ctx1", "ctx3"],
}


def _arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run hybrid-vs-UKF ablation suite.")
    parser.add_argument(
        "--group",
        action="append",
        choices=sorted(BENCHMARK_CONFIGS.keys()),
        default=[],
        help="Ablation group to run. Can be passed multiple times. Default: all groups.",
    )
    parser.add_argument(
        "--stage",
        choices=["prepare", "benchmark", "all"],
        default="all",
        help="Run dataset/train/eval, benchmark, or both.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip prepare steps for variants whose model/report outputs already exist.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=None,
        help="Optional repeats override passed to P4 benchmark runner.",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        default=[],
        help="Optional scenario filter passed to P4 benchmark runner. Can be repeated.",
    )
    return parser


def _run_module(module: str, config_path: Path, *, extra_args: List[str] | None = None) -> None:
    cmd = [sys.executable, "-m", module, "--config", str(config_path)]
    if extra_args:
        cmd.extend(extra_args)
    print(f"[run] {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def _variant_ready(variant: str) -> bool:
    config_path = EXPERIMENT_CONFIGS[variant]
    cfg = load_yaml_config(config_path)
    paths = cfg.get("paths", {})
    model_dir = paths.get("model_dir")
    report_dir = paths.get("report_dir")
    if not model_dir or not report_dir:
        return False
    return (REPO_ROOT / str(model_dir)).exists() and (REPO_ROOT / str(report_dir)).exists()


def _prepare_variant(variant: str, *, skip_existing: bool) -> Dict[str, str]:
    config_path = EXPERIMENT_CONFIGS[variant]
    if skip_existing and _variant_ready(variant):
        print(f"[skip][prepare][{variant}] existing outputs detected")
        return {"variant": variant, "status": "skipped_existing", "config": str(config_path)}
    _run_module("cnn_fpga.data.runtime_dataset_builder", config_path)
    _run_module("cnn_fpga.model.train", config_path)
    _run_module("cnn_fpga.model.evaluate", config_path)
    return {"variant": variant, "status": "completed", "config": str(config_path)}


def _run_group_benchmark(group: str, *, repeats: int | None, scenarios: List[str]) -> Dict[str, str]:
    config_path = BENCHMARK_CONFIGS[group]
    extra_args: List[str] = []
    if repeats is not None:
        extra_args.extend(["--repeats", str(repeats)])
    for scenario in scenarios:
        extra_args.extend(["--scenario", scenario])
    _run_module("cnn_fpga.benchmark.run_p4_multiscenario_benchmark", config_path, extra_args=extra_args)
    return {"group": group, "status": "completed", "config": str(config_path)}


def main() -> int:
    args = _arg_parser().parse_args()
    groups = args.group or list(BENCHMARK_CONFIGS.keys())
    run_root = ensure_dir(REPO_ROOT / "runs" / "p4_hybrid_vs_ukf_ablation")
    session_dir = ensure_dir(run_root / f"ablation_{now_tag()}")

    summary: Dict[str, object] = {
        "groups": groups,
        "stage": args.stage,
        "prepare": [],
        "benchmark": [],
    }

    if args.stage in {"prepare", "all"}:
        prepared = set()
        for group in groups:
            for variant in GROUP_VARIANTS[group]:
                if variant in prepared:
                    continue
                result = _prepare_variant(variant, skip_existing=args.skip_existing)
                summary["prepare"].append(result)
                prepared.add(variant)

    if args.stage in {"benchmark", "all"}:
        for group in groups:
            result = _run_group_benchmark(group, repeats=args.repeats, scenarios=list(args.scenario))
            summary["benchmark"].append(result)

    save_json(session_dir / "summary.json", summary)
    print("Hybrid-vs-UKF ablation run complete.")
    print(f"session_dir={session_dir}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
