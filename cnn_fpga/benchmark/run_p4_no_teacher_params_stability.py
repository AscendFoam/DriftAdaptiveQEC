"""Run a seed-sweep stability check for Hybrid Full vs No TeacherParams."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

from cnn_fpga.utils.config import ensure_dir, load_yaml_config, now_tag, save_json


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "cnn_fpga" / "config"
FULL_CONFIG = CONFIG_DIR / "experiment_runtime_b_residual.yaml"
NO_TP_CONFIG = CONFIG_DIR / "experiment_runtime_b_residual_no_teacher_params.yaml"
BENCHMARK_CONFIG = CONFIG_DIR / "p4_hybrid_vs_ukf_ablation_features.yaml"

VARIANT_CONFIGS = {
    "hybrid_full": FULL_CONFIG,
    "hybrid_no_teacher_params": NO_TP_CONFIG,
}


def _arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Seed-sweep stability check for No TeacherParams.")
    parser.add_argument(
        "--seed",
        action="append",
        type=int,
        default=[],
        help="Seed to run. Can be provided multiple times. Defaults to 20260404/05/06.",
    )
    parser.add_argument(
        "--stage",
        choices=["prepare", "benchmark", "all"],
        default="all",
        help="Run dataset/train/eval, benchmark only, or both.",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        default=[],
        help="Optional scenario filter passed to the P4 benchmark runner.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=None,
        help="Optional repeats override passed to the P4 benchmark runner.",
    )
    parser.add_argument(
        "--training-backend",
        choices=["numpy", "torch"],
        default="torch",
        help="Tiny-CNN training backend for generated configs.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Training device passed to the Tiny-CNN backend. Examples: auto, cpu, cuda.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip prepare steps when generated model/report outputs already exist.",
    )
    parser.add_argument(
        "--reuse-existing-datasets",
        action="store_true",
        help="Reuse the baseline datasets from the source configs and only retrain/re-evaluate models.",
    )
    return parser


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _save_yaml(path: Path, payload: Dict[str, Any]) -> None:
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on environment.
        raise RuntimeError("PyYAML is required to write generated configs for this script.") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)


def _run_module(module: str, config_path: Path, *, extra_args: List[str] | None = None) -> None:
    cmd = [sys.executable, "-m", module, "--config", str(config_path)]
    if extra_args:
        cmd.extend(extra_args)
    print(f"[run] {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def _variant_outputs(session_dir: Path, seed: int, variant_name: str) -> Dict[str, Path]:
    root = session_dir / "generated" / f"seed_{seed}" / variant_name
    return {
        "root": root,
        "dataset_dir": root / "datasets",
        "model_dir": root / "models",
        "report_dir": root / "reports",
    }


def _variant_ready(outputs: Dict[str, Path]) -> bool:
    return outputs["model_dir"].exists() and any(outputs["model_dir"].glob("*.npz")) and outputs["report_dir"].exists()


def _build_variant_config(
    *,
    base_config_path: Path,
    session_dir: Path,
    seed: int,
    variant_name: str,
    training_backend: str,
    device: str,
) -> tuple[Dict[str, Any], Dict[str, Path]]:
    cfg = load_yaml_config(base_config_path)
    outputs = _variant_outputs(session_dir, seed, variant_name)
    cfg.setdefault("experiment", {})
    cfg["experiment"]["seed"] = int(seed)
    cfg["experiment"]["name"] = f"{cfg['experiment'].get('name', variant_name)}_seed_{seed}"
    cfg.setdefault("paths", {})
    if not cfg["paths"].get("dataset_dir"):
        cfg["paths"]["dataset_dir"] = str(outputs["dataset_dir"])
    cfg["paths"]["model_dir"] = str(outputs["model_dir"])
    cfg["paths"]["report_dir"] = str(outputs["report_dir"])
    cfg.setdefault("training", {})
    cfg["training"]["model_type"] = "tiny_cnn"
    cfg["training"].setdefault("tiny_cnn", {})
    cfg["training"]["tiny_cnn"]["backend"] = str(training_backend)
    cfg["training"]["tiny_cnn"]["device"] = str(device)
    cfg["training"]["tiny_cnn"]["seed"] = int(seed)
    return cfg, outputs


def _build_benchmark_config(
    *,
    session_dir: Path,
    seed: int,
    model_dirs: Dict[str, Path],
) -> Dict[str, Any]:
    cfg = load_yaml_config(BENCHMARK_CONFIG)
    cfg.setdefault("experiment", {})
    cfg["experiment"]["seed"] = int(seed)
    cfg["experiment"]["name"] = f"p4ntp_s{seed}"
    cfg.setdefault("paths", {})
    cfg["paths"]["output_root"] = str(REPO_ROOT / "runs" / "p4_ntp_bench")
    cfg.setdefault("p4_benchmark", {})
    cfg["p4_benchmark"].setdefault("protocol", {})
    cfg["p4_benchmark"]["protocol"]["protocol_id"] = f"p4ntp_s{seed}"

    selected_modes = []
    for mode in cfg["p4_benchmark"].get("modes", []):
        name = str(mode.get("name"))
        if name not in {"ukf", "hybrid_full", "hybrid_no_teacher_params"}:
            continue
        mode_cfg = deepcopy(mode)
        if name in model_dirs:
            mode_cfg.setdefault("overrides", {})
            mode_cfg["overrides"].setdefault("paths", {})
            mode_cfg["overrides"]["paths"]["model_dir"] = str(model_dirs[name])
        selected_modes.append(mode_cfg)
    cfg["p4_benchmark"]["modes"] = selected_modes
    return cfg


def _collect_seed_summary(summary_path: Path) -> Dict[str, Any]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    comparison_rows = payload.get("comparison_rows", [])
    mode_scores: Dict[str, List[float]] = {}
    for row in comparison_rows:
        mode_scores.setdefault(str(row["mode"]), []).append(float(row["final_ler_mean"]))
    averaged = {mode: float(sum(values) / len(values)) for mode, values in mode_scores.items() if values}
    full_ler = averaged.get("hybrid_full")
    no_tp_ler = averaged.get("hybrid_no_teacher_params")
    return {
        "benchmark_summary": str(summary_path),
        "avg_ler_by_mode": averaged,
        "avg_gap_no_teacher_params_vs_full": None if full_ler is None or no_tp_ler is None else float(no_tp_ler - full_ler),
    }


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fields = [
        "seed",
        "training_backend",
        "device",
        "avg_ler_hybrid_full",
        "avg_ler_hybrid_no_teacher_params",
        "avg_gap_no_teacher_params_vs_full",
        "benchmark_summary",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "seed": row.get("seed"),
                    "training_backend": row.get("training_backend"),
                    "device": row.get("device"),
                    "avg_ler_hybrid_full": row.get("avg_ler_by_mode", {}).get("hybrid_full"),
                    "avg_ler_hybrid_no_teacher_params": row.get("avg_ler_by_mode", {}).get("hybrid_no_teacher_params"),
                    "avg_gap_no_teacher_params_vs_full": row.get("avg_gap_no_teacher_params_vs_full"),
                    "benchmark_summary": row.get("benchmark_summary"),
                }
            )


def main() -> int:
    args = _arg_parser().parse_args()
    seeds = args.seed or [20260404, 20260405, 20260406]
    session_dir = ensure_dir(REPO_ROOT / "runs" / "p4_no_teacher_params_stability" / f"stability_{now_tag()}")
    results: List[Dict[str, Any]] = []

    for seed in seeds:
        seed_dir = ensure_dir(session_dir / "configs" / f"seed_{seed}")
        variant_model_dirs: Dict[str, Path] = {}

        if args.stage in {"prepare", "all"}:
            for variant_name, config_path in VARIANT_CONFIGS.items():
                variant_cfg, outputs = _build_variant_config(
                    base_config_path=config_path,
                    session_dir=session_dir,
                    seed=seed,
                    variant_name=variant_name,
                    training_backend=args.training_backend,
                    device=args.device,
                )
                if args.reuse_existing_datasets:
                    source_cfg = load_yaml_config(config_path)
                    dataset_dir = source_cfg.get("paths", {}).get("dataset_dir")
                    if not dataset_dir:
                        raise ValueError(f"Source config missing dataset_dir: {config_path}")
                    variant_cfg.setdefault("paths", {})
                    variant_cfg["paths"]["dataset_dir"] = str(Path(dataset_dir).expanduser().resolve())
                variant_config_path = seed_dir / f"{variant_name}.yaml"
                _save_yaml(variant_config_path, variant_cfg)
                variant_model_dirs[variant_name] = outputs["model_dir"]
                if args.skip_existing and _variant_ready(outputs):
                    print(f"[skip][prepare][seed={seed}][{variant_name}] existing outputs detected")
                    continue
                if not args.reuse_existing_datasets:
                    _run_module("cnn_fpga.data.runtime_dataset_builder", variant_config_path)
                _run_module("cnn_fpga.model.train", variant_config_path)
                _run_module("cnn_fpga.model.evaluate", variant_config_path)
        else:
            for variant_name in VARIANT_CONFIGS:
                variant_model_dirs[variant_name] = _variant_outputs(session_dir, seed, variant_name)["model_dir"]

        seed_result: Dict[str, Any] = {
            "seed": int(seed),
            "training_backend": args.training_backend,
            "device": args.device,
            "reuse_existing_datasets": bool(args.reuse_existing_datasets),
            "variant_model_dirs": {name: str(path) for name, path in variant_model_dirs.items()},
        }

        if args.stage in {"benchmark", "all"}:
            benchmark_cfg = _build_benchmark_config(session_dir=session_dir, seed=seed, model_dirs=variant_model_dirs)
            benchmark_config_path = seed_dir / "benchmark.yaml"
            _save_yaml(benchmark_config_path, benchmark_cfg)
            extra_args: List[str] = []
            if args.repeats is not None:
                extra_args.extend(["--repeats", str(args.repeats)])
            for scenario in args.scenario:
                extra_args.extend(["--scenario", scenario])
            _run_module("cnn_fpga.benchmark.run_p4_multiscenario_benchmark", benchmark_config_path, extra_args=extra_args)

            benchmark_runs = sorted((session_dir / "benchmark_runs" / "p4_benchmark").glob(f"p4_no_teacher_params_stability_seed_{seed}_*"))
            if not benchmark_runs:
                raise FileNotFoundError(f"No benchmark run directory found for seed={seed}")
            latest_run = benchmark_runs[-1]
            summary = _collect_seed_summary(latest_run / "summary.json")
            seed_result.update(summary)

        results.append(seed_result)

    save_json(
        session_dir / "summary.json",
        {
            "session_dir": str(session_dir),
            "seeds": seeds,
            "stage": args.stage,
            "training_backend": args.training_backend,
            "device": args.device,
            "reuse_existing_datasets": bool(args.reuse_existing_datasets),
            "results": results,
        },
    )
    _write_csv(session_dir / "summary.csv", results)
    print(f"session_dir={session_dir}")
    for row in results:
        gap = row.get("avg_gap_no_teacher_params_vs_full")
        if gap is None:
            continue
        print(
            f"[seed={row['seed']}] avg_ler(full)={row['avg_ler_by_mode']['hybrid_full']:.6f} "
            f"avg_ler(no_teacher_params)={row['avg_ler_by_mode']['hybrid_no_teacher_params']:.6f} "
            f"gap={gap:.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
