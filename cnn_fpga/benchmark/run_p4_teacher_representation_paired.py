"""Run paired P4 benchmarks for teacher-representation variants anchored on Full."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

from cnn_fpga.utils.config import ensure_dir, load_yaml_config, now_tag, open_text, save_json


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "cnn_fpga" / "config"
BENCHMARK_BASE_CONFIG = CONFIG_DIR / "p4_hybrid_vs_ukf_ablation_features.yaml"


@dataclass(frozen=True)
class VariantSpec:
    key: str
    label: str
    benchmark_name: str
    config_path: Path
    is_anchor: bool = False


VARIANT_SPECS: dict[str, VariantSpec] = {
    "full": VariantSpec(
        key="full",
        label="Hybrid Full",
        benchmark_name="hybrid_full",
        config_path=CONFIG_DIR / "experiment_runtime_b_residual.yaml",
        is_anchor=True,
    ),
    "selective": VariantSpec(
        key="selective",
        label="Hybrid Selective Teacher",
        benchmark_name="hybrid_selective_teacher",
        config_path=CONFIG_DIR / "experiment_runtime_b_residual_selective_teacher.yaml",
    ),
    "gated_v2": VariantSpec(
        key="gated_v2",
        label="Hybrid Normalized + Gated Teacher v2",
        benchmark_name="hybrid_gated_teacher_v2",
        config_path=CONFIG_DIR / "experiment_runtime_b_residual_norm_gated_teacher_v2.yaml",
    ),
    "gated_v3": VariantSpec(
        key="gated_v3",
        label="Hybrid Normalized + Gated Teacher v3",
        benchmark_name="hybrid_gated_teacher_v3",
        config_path=CONFIG_DIR / "experiment_runtime_b_residual_norm_gated_teacher_v3.yaml",
    ),
    "gated_v4": VariantSpec(
        key="gated_v4",
        label="Hybrid Normalized + Gated Teacher v4",
        benchmark_name="hybrid_gated_teacher_v4",
        config_path=CONFIG_DIR / "experiment_runtime_b_residual_norm_gated_teacher_v4.yaml",
    ),
    "gated_v5": VariantSpec(
        key="gated_v5",
        label="Hybrid Normalized + Gated Teacher v5",
        benchmark_name="hybrid_gated_teacher_v5",
        config_path=CONFIG_DIR / "experiment_runtime_b_residual_norm_gated_teacher_v5.yaml",
    ),
    "gated_v6": VariantSpec(
        key="gated_v6",
        label="Hybrid Normalized + Gated Teacher v6",
        benchmark_name="hybrid_gated_teacher_v6",
        config_path=CONFIG_DIR / "experiment_runtime_b_residual_norm_gated_teacher_v6.yaml",
    ),
    "gated_v7": VariantSpec(
        key="gated_v7",
        label="Hybrid Normalized + Gated Teacher v7",
        benchmark_name="hybrid_gated_teacher_v7",
        config_path=CONFIG_DIR / "experiment_runtime_b_residual_norm_gated_teacher_v7.yaml",
    ),
    "gated_v8": VariantSpec(
        key="gated_v8",
        label="Hybrid Normalized + Gated Teacher v8",
        benchmark_name="hybrid_gated_teacher_v8",
        config_path=CONFIG_DIR / "experiment_runtime_b_residual_norm_gated_teacher_v8.yaml",
    ),
    "gated_v9": VariantSpec(
        key="gated_v9",
        label="Hybrid Normalized + Gated Teacher v9",
        benchmark_name="hybrid_gated_teacher_v9",
        config_path=CONFIG_DIR / "experiment_runtime_b_residual_norm_gated_teacher_v9.yaml",
    ),
    "minimal": VariantSpec(
        key="minimal",
        label="Hybrid Minimal Teacher",
        benchmark_name="hybrid_minimal_teacher",
        config_path=CONFIG_DIR / "experiment_runtime_b_residual_minimal_teacher.yaml",
    ),
    "no_teacher": VariantSpec(
        key="no_teacher",
        label="Hybrid No TeacherParams",
        benchmark_name="hybrid_no_teacher_params",
        config_path=CONFIG_DIR / "experiment_runtime_b_residual_no_teacher_params.yaml",
    ),
}

DEFAULT_VARIANTS = ("full", "selective", "gated_v2")
DEFAULT_BENCHMARK_SCENARIOS = ("linear_ramp", "periodic_drift")


def _arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Controlled paired benchmark for teacher-representation variants."
    )
    parser.add_argument("--seed", action="append", type=int, default=[], help="Seed to run; can be repeated.")
    parser.add_argument(
        "--variant",
        action="append",
        choices=sorted(VARIANT_SPECS),
        default=[],
        help=(
            "Variant key to include. Default is Full + Selective + Gated v2. "
            "Use this to optionally add Minimal / No Teacher auxiliary controls."
        ),
    )
    parser.add_argument("--stage", choices=["prepare", "benchmark", "all"], default="all")
    parser.add_argument(
        "--dataset-scenario",
        action="append",
        default=[],
        help="Optional scenario filter for dataset building. Omit to keep the dataset multi-scenario.",
    )
    parser.add_argument(
        "--benchmark-scenario",
        action="append",
        default=[],
        help="Scenario filter for the paired benchmark. Default: linear_ramp + periodic_drift.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=None,
        help="Optional repeat override for the paired benchmark.",
    )
    parser.add_argument(
        "--windows-per-scenario",
        type=int,
        default=None,
        help="Optional runtime-dataset windows_per_scenario override for smoke/debug runs.",
    )
    parser.add_argument("--training-backend", choices=["numpy", "torch"], default="torch")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--reuse-existing-datasets", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Write generated configs and print commands only.")
    parser.add_argument(
        "--detach",
        action="store_true",
        help="Relaunch this paired runner in the background with stdout/stderr/meta logs and exit immediately.",
    )
    parser.add_argument(
        "--detach-log-dir",
        type=str,
        default=None,
        help="Optional directory for detached-launch stdout/stderr/meta logs. Defaults to <session-root>/detached_launches.",
    )
    parser.add_argument(
        "--session-root",
        type=str,
        default="runs/teachrepr",
        help="Root directory for generated configs, models, reports, and summary files.",
    )
    parser.add_argument(
        "--output-session-dir",
        type=str,
        default=None,
        help="Optional fixed session directory for resumable prepare/benchmark runs.",
    )
    parser.add_argument(
        "--benchmark-output-root",
        type=str,
        default="runs/teachrepr",
        help="Short output root used by run_p4_multiscenario_benchmark.",
    )
    parser.add_argument(
        "--benchmark-base-config",
        type=str,
        default=None,
        help="Optional benchmark base-config override.",
    )
    parser.add_argument(
        "--model-session-dir",
        type=str,
        default=None,
        help="Existing session root to reuse prepared model outputs during benchmark-only runs.",
    )
    parser.add_argument(
        "--experiment-prefix",
        type=str,
        default="trp",
        help="Short prefix used for generated benchmark experiment names.",
    )
    parser.add_argument(
        "--independent-seeds",
        action="store_true",
        help="Disable paired seeds across modes. Default keeps same seed stream across variants.",
    )
    parser.add_argument(
        "--allow-no-anchor",
        action="store_true",
        help="Do not auto-add Full. Intended for preparing a new variant only.",
    )
    parser.add_argument(
        "--chunk-repeat-size",
        type=int,
        default=1,
        help="Run benchmark repeats in resumable chunks of this size. Default: 1.",
    )
    return parser


def _save_yaml(path: Path, payload: Dict[str, Any]) -> None:
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required to write generated configs for this script.") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)


def _safe_print(message: str) -> None:
    try:
        print(message, flush=True)
    except OSError:
        pass


def _run_module(
    module: str,
    config_path: Path,
    *,
    extra_args: Sequence[str] | None = None,
    dry_run: bool = False,
) -> None:
    cmd = [sys.executable, "-m", module, "--config", str(config_path)]
    if extra_args:
        cmd.extend(extra_args)
    _safe_print(f"[run] {' '.join(cmd)}")
    if dry_run:
        return
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def _run_module_logged(
    module: str,
    config_path: Path,
    *,
    log_dir: Path,
    log_stem: str,
    extra_args: Sequence[str] | None = None,
    dry_run: bool = False,
) -> None:
    cmd = [sys.executable, "-m", module, "--config", str(config_path)]
    if extra_args:
        cmd.extend(extra_args)
    ensure_dir(log_dir)
    stdout_path = log_dir / f"{log_stem}.stdout.log"
    stderr_path = log_dir / f"{log_stem}.stderr.log"
    meta_path = log_dir / f"{log_stem}.meta.json"
    meta_payload = {
        "module": module,
        "config_path": str(config_path),
        "cmd": cmd,
        "cwd": str(REPO_ROOT),
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
    }
    save_json(meta_path, meta_payload)
    _safe_print(f"[run-logged] {' '.join(cmd)}")
    if dry_run:
        return
    with open_text(stdout_path, "w") as stdout_file, open_text(stderr_path, "w") as stderr_file:
        subprocess.run(cmd, check=True, cwd=REPO_ROOT, stdout=stdout_file, stderr=stderr_file)


def _detach_child_argv(raw_argv: Sequence[str]) -> List[str]:
    detached: List[str] = []
    skip_next = False
    for idx, token in enumerate(raw_argv):
        if skip_next:
            skip_next = False
            continue
        if token == "--detach":
            continue
        if token == "--detach-log-dir":
            skip_next = True
            continue
        if token.startswith("--detach-log-dir="):
            continue
        detached.append(token)
    return detached


def _launch_detached_self(*, args: argparse.Namespace, raw_argv: Sequence[str]) -> int:
    log_root = (
        ensure_dir(Path(args.detach_log_dir).expanduser().resolve())
        if args.detach_log_dir
        else ensure_dir(Path(args.session_root).expanduser().resolve() / "detached_launches")
    )
    stem = f"paired_{args.stage}_{now_tag()}"
    stdout_path = log_root / f"{stem}.stdout.log"
    stderr_path = log_root / f"{stem}.stderr.log"
    meta_path = log_root / f"{stem}.meta.json"
    child_argv = _detach_child_argv(raw_argv)
    cmd = [sys.executable, "-m", "cnn_fpga.benchmark.run_p4_teacher_representation_paired", *child_argv]
    creationflags = 0
    popen_kwargs: Dict[str, Any] = {
        "cwd": str(REPO_ROOT),
        "stdout": open_text(stdout_path, "w"),
        "stderr": open_text(stderr_path, "w"),
    }
    if os.name == "nt":
        # DETACHED_PROCESS can trigger DLL-init failures for nested torch/python
        # subprocesses on Windows. Prefer a hidden background console instead.
        creationflags = int(getattr(subprocess, "CREATE_NO_WINDOW", 0)) | int(
            getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        )
        popen_kwargs["creationflags"] = creationflags
    else:
        popen_kwargs["start_new_session"] = True

    try:
        proc = subprocess.Popen(cmd, **popen_kwargs)
    finally:
        for name in ("stdout", "stderr"):
            handle = popen_kwargs.get(name)
            if handle is not None:
                handle.close()

    meta_payload = {
        "launcher_pid": os.getpid(),
        "child_pid": int(proc.pid),
        "python_executable": sys.executable,
        "cmd": cmd,
        "cwd": str(REPO_ROOT),
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "creationflags": creationflags,
    }
    save_json(meta_path, meta_payload)
    _safe_print(f"[detached] pid={proc.pid}")
    _safe_print(f"[detached] stdout={stdout_path}")
    _safe_print(f"[detached] stderr={stderr_path}")
    _safe_print(f"[detached] meta={meta_path}")
    return 0


def _sanitize_short_name(prefix: str, seed: int) -> str:
    text = "".join(ch for ch in str(prefix).lower() if ch.isalnum())
    short_prefix = text[:3] or "trp"
    digits = str(abs(int(seed)))
    tail = digits[-max(1, 8 - len(short_prefix)) :]
    return f"{short_prefix}{tail}"


def _resolved_variants(requested: Sequence[str], *, require_anchor: bool = True) -> List[VariantSpec]:
    keys = list(requested) if requested else list(DEFAULT_VARIANTS)
    ordered: List[str] = []
    if require_anchor and "full" not in keys:
        ordered.append("full")
    for key in keys:
        if key not in ordered:
            ordered.append(key)
    variants = [VARIANT_SPECS[key] for key in ordered]
    if not variants or not variants[0].is_anchor:
        variants.sort(key=lambda item: (not item.is_anchor, item.key))
    return variants


def _benchmark_scenarios(requested: Sequence[str]) -> List[str]:
    values = list(requested) if requested else list(DEFAULT_BENCHMARK_SCENARIOS)
    ordered: List[str] = []
    for value in values:
        text = str(value)
        if text not in ordered:
            ordered.append(text)
    return ordered


def _variant_outputs(session_dir: Path, seed: int, variant_key: str) -> Dict[str, Path]:
    root = session_dir / "g" / f"s{seed}" / variant_key
    return {
        "root": root,
        "dataset_dir": root / "d",
        "model_dir": root / "m",
        "report_dir": root / "r",
    }


def _variant_ready(outputs: Dict[str, Path]) -> bool:
    return outputs["model_dir"].exists() and any(outputs["model_dir"].glob("*.npz")) and outputs["report_dir"].exists()


def _dataset_ready(dataset_dir: Path) -> bool:
    required = ("train.npz", "val.npz", "test.npz", "manifest.json")
    return dataset_dir.exists() and all((dataset_dir / name).exists() for name in required)


def _build_variant_config(
    *,
    spec: VariantSpec,
    session_dir: Path,
    seed: int,
    training_backend: str,
    device: str,
    windows_per_scenario: int | None,
) -> tuple[Dict[str, Any], Dict[str, Path]]:
    cfg = load_yaml_config(spec.config_path)
    outputs = _variant_outputs(session_dir, seed, spec.key)
    cfg.setdefault("experiment", {})
    cfg["experiment"]["seed"] = int(seed)
    cfg["experiment"]["name"] = f"{spec.key}_s{seed}"
    cfg.setdefault("paths", {})
    cfg["paths"]["dataset_dir"] = str(outputs["dataset_dir"])
    cfg["paths"]["model_dir"] = str(outputs["model_dir"])
    cfg["paths"]["report_dir"] = str(outputs["report_dir"])
    cfg.setdefault("training", {})
    cfg["training"]["model_type"] = "tiny_cnn"
    cfg["training"].setdefault("tiny_cnn", {})
    cfg["training"]["tiny_cnn"]["backend"] = str(training_backend)
    cfg["training"]["tiny_cnn"]["device"] = str(device)
    cfg["training"]["tiny_cnn"]["seed"] = int(seed)
    if windows_per_scenario is not None:
        cfg.setdefault("runtime_dataset", {})
        cfg["runtime_dataset"]["windows_per_scenario"] = int(windows_per_scenario)
    return cfg, outputs


def _variant_hybrid_mode_overrides(spec: VariantSpec) -> Dict[str, Any]:
    cfg = load_yaml_config(spec.config_path)
    slow_cfg = deepcopy(dict(cfg.get("slow_loop", {})))
    hybrid_cfg = deepcopy(dict(slow_cfg.get("hybrid_residual_b", {})))
    if not hybrid_cfg:
        raise ValueError(f"Variant config missing slow_loop.hybrid_residual_b: {spec.config_path}")
    infer_cfg = deepcopy(dict(slow_cfg.get("inference_service", {})))
    infer_cfg["label_names"] = [str(item) for item in infer_cfg.get("label_names", ["b_q", "b_p"])]
    return {
        "slow_loop": {
            "inference_service": infer_cfg,
            "hybrid_residual_b": hybrid_cfg,
        }
    }


def _build_benchmark_config(
    *,
    base_config_path: Path,
    experiment_name: str,
    seed: int,
    output_root: Path,
    variants: Sequence[VariantSpec],
    model_dirs: Dict[str, Path],
    paired_seeds: bool,
    repeats: int | None,
) -> Dict[str, Any]:
    cfg = load_yaml_config(base_config_path)
    cfg.setdefault("experiment", {})
    cfg["experiment"]["seed"] = int(seed)
    cfg["experiment"]["name"] = experiment_name
    cfg.setdefault("paths", {})
    cfg["paths"]["output_root"] = str(output_root)
    benchmark_cfg = cfg.setdefault("p4_benchmark", {})
    protocol = benchmark_cfg.setdefault("protocol", {})
    protocol["protocol_id"] = experiment_name
    protocol["frozen_baseline_set"] = [spec.benchmark_name for spec in variants]
    protocol["seed_pairing"] = "paired" if paired_seeds else "independent"
    if repeats is not None:
        protocol["repeats"] = int(repeats)

    modes: List[Dict[str, Any]] = []
    for spec in variants:
        overrides = _variant_hybrid_mode_overrides(spec)
        overrides.setdefault("paths", {})
        overrides["paths"]["model_dir"] = str(model_dirs[spec.key])
        modes.append(
            {
                "name": spec.benchmark_name,
                "label": spec.label,
                "slow_loop_mode": "hybrid_residual_b",
                "artifact_selector": "latest_float",
                "overrides": overrides,
            }
        )
    benchmark_cfg["modes"] = modes
    return cfg


def _find_latest_benchmark_run(output_root: Path, experiment_name: str) -> Path:
    run_root = output_root / "p4_benchmark"
    candidates = sorted(run_root.glob(f"{experiment_name}_*"))
    if not candidates:
        raise FileNotFoundError(f"No benchmark run directory found under {run_root} for experiment={experiment_name}")
    return candidates[-1]


def _find_matching_benchmark_runs(output_root: Path, experiment_name: str) -> List[Path]:
    run_root = output_root / "p4_benchmark"
    return sorted(run_root.glob(f"{experiment_name}_*"))


def _collect_seed_summary(summary_path: Path, variants: Sequence[VariantSpec]) -> Dict[str, Any]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    comparison_rows = payload.get("comparison_rows", [])
    mode_scores: Dict[str, List[float]] = {}
    for row in comparison_rows:
        mode_scores.setdefault(str(row["mode"]), []).append(float(row["final_ler_mean"]))
    averaged = {
        mode: float(sum(values) / len(values))
        for mode, values in mode_scores.items()
        if values
    }
    anchor_mode = next(spec.benchmark_name for spec in variants if spec.is_anchor)
    anchor_ler = averaged.get(anchor_mode)
    gaps = {
        mode: None if anchor_ler is None or mode == anchor_mode else float(value - anchor_ler)
        for mode, value in averaged.items()
    }
    return {
        "benchmark_summary": str(summary_path),
        "comparison_csv": payload.get("comparison_csv"),
        "teacher_scalar_diagnostics_csv": payload.get("teacher_scalar_diagnostics_csv"),
        "avg_ler_by_mode": averaged,
        "avg_gap_vs_anchor": gaps,
    }


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]], variants: Sequence[VariantSpec]) -> None:
    anchor = next((spec for spec in variants if spec.is_anchor), None)
    anchor_mode = anchor.benchmark_name if anchor is not None else variants[0].benchmark_name
    fields = [
        "seed",
        "training_backend",
        "device",
        "benchmark_summary",
        "comparison_csv",
        "teacher_scalar_diagnostics_csv",
    ]
    fields.extend(f"avg_ler_{spec.benchmark_name}" for spec in variants)
    if anchor is not None:
        fields.extend(
            f"avg_gap_{spec.benchmark_name}_vs_{anchor_mode}"
            for spec in variants
            if not spec.is_anchor
        )

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            avg = row.get("avg_ler_by_mode", {})
            gaps = row.get("avg_gap_vs_anchor", {})
            out_row = {
                "seed": row.get("seed"),
                "training_backend": row.get("training_backend"),
                "device": row.get("device"),
                "benchmark_summary": row.get("benchmark_summary"),
                "comparison_csv": row.get("comparison_csv"),
                "teacher_scalar_diagnostics_csv": row.get("teacher_scalar_diagnostics_csv"),
            }
            for spec in variants:
                out_row[f"avg_ler_{spec.benchmark_name}"] = avg.get(spec.benchmark_name)
                if anchor is not None and not spec.is_anchor:
                    out_row[f"avg_gap_{spec.benchmark_name}_vs_{anchor_mode}"] = gaps.get(spec.benchmark_name)
            writer.writerow(out_row)


def _repeat_chunks(repeats: int, chunk_size: int) -> List[tuple[int, int]]:
    if repeats <= 0:
        raise ValueError(f"repeats must be positive, got {repeats}")
    if chunk_size <= 0:
        raise ValueError(f"chunk_repeat_size must be positive, got {chunk_size}")
    return [(start, min(repeats, start + chunk_size)) for start in range(0, repeats, chunk_size)]


def main() -> int:
    args = _arg_parser().parse_args()
    if args.detach:
        return _launch_detached_self(args=args, raw_argv=sys.argv[1:])
    variants = _resolved_variants(args.variant, require_anchor=not bool(args.allow_no_anchor))
    seeds = args.seed or [20260426]
    session_root = ensure_dir(Path(args.session_root))
    session_dir = (
        ensure_dir(Path(args.output_session_dir).expanduser().resolve())
        if args.output_session_dir
        else ensure_dir(session_root / f"paired_{now_tag()}")
    )
    model_session_dir = None if not args.model_session_dir else Path(args.model_session_dir).expanduser().resolve()
    benchmark_base_config = (
        Path(args.benchmark_base_config).expanduser().resolve()
        if args.benchmark_base_config
        else BENCHMARK_BASE_CONFIG
    )
    benchmark_output_root = ensure_dir(Path(args.benchmark_output_root))
    dataset_scenarios = [str(item) for item in args.dataset_scenario]
    benchmark_scenarios = _benchmark_scenarios(args.benchmark_scenario)
    paired_seeds = not bool(args.independent_seeds)
    chunk_repeat_size = int(args.chunk_repeat_size)
    effective_repeats = int(args.repeats) if args.repeats is not None else None
    launch_logs_dir = ensure_dir(session_root / "launch_logs")

    results: List[Dict[str, Any]] = []
    for seed in seeds:
        experiment_name = _sanitize_short_name(args.experiment_prefix, seed)
        seed_dir = ensure_dir(session_dir / "cfg" / f"s{seed}")
        variant_model_dirs: Dict[str, Path] = {}
        variant_dataset_dirs: Dict[str, Path] = {}

        if args.stage in {"prepare", "all"}:
            for spec in variants:
                if model_session_dir is not None:
                    reusable_outputs = _variant_outputs(model_session_dir, seed, spec.key)
                    if _variant_ready(reusable_outputs):
                        _safe_print(f"[reuse][prepare][seed={seed}][{spec.key}] {reusable_outputs['root']}")
                        variant_model_dirs[spec.key] = reusable_outputs["model_dir"]
                        variant_dataset_dirs[spec.key] = reusable_outputs["dataset_dir"]
                        continue
                variant_cfg, outputs = _build_variant_config(
                    spec=spec,
                    session_dir=session_dir,
                    seed=seed,
                    training_backend=args.training_backend,
                    device=args.device,
                    windows_per_scenario=args.windows_per_scenario,
                )
                if args.reuse_existing_datasets:
                    source_cfg = load_yaml_config(spec.config_path)
                    dataset_dir = source_cfg.get("paths", {}).get("dataset_dir")
                    if not dataset_dir:
                        raise ValueError(f"Source config missing dataset_dir: {spec.config_path}")
                    variant_cfg.setdefault("paths", {})
                    variant_cfg["paths"]["dataset_dir"] = str(Path(dataset_dir).expanduser().resolve())
                variant_config_path = seed_dir / f"{spec.key}.yaml"
                _save_yaml(variant_config_path, variant_cfg)
                variant_model_dirs[spec.key] = outputs["model_dir"]
                variant_dataset_dirs[spec.key] = Path(variant_cfg["paths"]["dataset_dir"]).expanduser().resolve()
                if args.skip_existing and _variant_ready(outputs):
                    _safe_print(f"[skip][prepare][seed={seed}][{spec.key}] existing outputs detected")
                    continue
                if not args.reuse_existing_datasets:
                    if args.skip_existing and _dataset_ready(outputs["dataset_dir"]):
                        _safe_print(f"[skip][dataset][seed={seed}][{spec.key}] existing dataset detected")
                    else:
                        extra_args: List[str] = []
                        for scenario in dataset_scenarios:
                            extra_args.extend(["--scenario", scenario])
                        _run_module(
                            "cnn_fpga.data.runtime_dataset_builder",
                            variant_config_path,
                            extra_args=extra_args,
                            dry_run=args.dry_run,
                        )
                _run_module("cnn_fpga.model.train", variant_config_path, dry_run=args.dry_run)
                _run_module("cnn_fpga.model.evaluate", variant_config_path, dry_run=args.dry_run)
        else:
            source_session_dir = model_session_dir or session_dir
            for spec in variants:
                outputs = _variant_outputs(source_session_dir, seed, spec.key)
                if not _variant_ready(outputs):
                    raise FileNotFoundError(
                        f"Missing existing model/report outputs for seed={seed}, variant={spec.key}: {outputs['root']}"
                    )
                variant_model_dirs[spec.key] = outputs["model_dir"]
                variant_dataset_dirs[spec.key] = outputs["dataset_dir"]

        seed_result: Dict[str, Any] = {
            "seed": int(seed),
            "training_backend": args.training_backend,
            "device": args.device,
            "paired_seeds": paired_seeds,
            "reuse_existing_datasets": bool(args.reuse_existing_datasets),
            "dataset_scenarios": list(dataset_scenarios),
            "benchmark_scenarios": list(benchmark_scenarios),
            "windows_per_scenario": args.windows_per_scenario,
            "variants": [spec.key for spec in variants],
            "variant_model_dirs": {spec.key: str(variant_model_dirs.get(spec.key, "")) for spec in variants},
            "variant_dataset_dirs": {spec.key: str(variant_dataset_dirs.get(spec.key, "")) for spec in variants},
            "model_session_dir": None if model_session_dir is None else str(model_session_dir),
        }

        if args.stage in {"benchmark", "all"}:
            benchmark_cfg = _build_benchmark_config(
                base_config_path=benchmark_base_config,
                experiment_name=experiment_name,
                seed=seed,
                output_root=benchmark_output_root,
                variants=variants,
                model_dirs=variant_model_dirs,
                paired_seeds=paired_seeds,
                repeats=args.repeats,
            )
            benchmark_config_path = seed_dir / "benchmark.yaml"
            _save_yaml(benchmark_config_path, benchmark_cfg)
            benchmark_run_dir = benchmark_output_root / "p4_benchmark" / f"{experiment_name}_resume"
            base_extra_args: List[str] = ["--run-dir", str(benchmark_run_dir)]
            if args.repeats is not None:
                base_extra_args.extend(["--repeats", str(args.repeats)])
            if paired_seeds:
                base_extra_args.append("--paired-seeds")
            for scenario in benchmark_scenarios:
                base_extra_args.extend(["--scenario", scenario])

            repeats_for_chunking = effective_repeats
            if repeats_for_chunking is None:
                repeats_for_chunking = int(benchmark_cfg.get("p4_benchmark", {}).get("protocol", {}).get("repeats", 1))
            chunk_specs = _repeat_chunks(repeats_for_chunking, chunk_repeat_size)
            seed_log_dir = ensure_dir(launch_logs_dir / f"s{seed}")
            for repeat_start, repeat_stop in chunk_specs:
                chunk_args = list(base_extra_args)
                chunk_args.extend(["--repeat-start", str(repeat_start), "--repeat-stop", str(repeat_stop)])
                _run_module_logged(
                    "cnn_fpga.benchmark.run_p4_multiscenario_benchmark",
                    benchmark_config_path,
                    log_dir=seed_log_dir,
                    log_stem=f"{experiment_name}_r{repeat_start:02d}_{repeat_stop:02d}",
                    extra_args=chunk_args,
                    dry_run=args.dry_run,
                )
            _run_module_logged(
                "cnn_fpga.benchmark.run_p4_multiscenario_benchmark",
                benchmark_config_path,
                log_dir=seed_log_dir,
                log_stem=f"{experiment_name}_aggregate",
                extra_args=[*base_extra_args, "--resume-only"],
                dry_run=args.dry_run,
            )
            if not args.dry_run:
                matching_runs = _find_matching_benchmark_runs(benchmark_output_root, experiment_name)
                if benchmark_run_dir.exists():
                    latest_run = benchmark_run_dir
                elif matching_runs:
                    latest_run = matching_runs[-1]
                else:
                    latest_run = _find_latest_benchmark_run(benchmark_output_root, experiment_name)
                seed_result["benchmark_run_dir"] = str(latest_run)
                seed_result["launch_log_dir"] = str(seed_log_dir)
                seed_result.update(_collect_seed_summary(latest_run / "summary.json", variants))

        results.append(seed_result)

    summary_payload = {
        "session_dir": str(session_dir),
        "session_root": str(session_root),
        "benchmark_output_root": str(benchmark_output_root),
        "benchmark_base_config": str(benchmark_base_config),
        "seeds": seeds,
        "stage": args.stage,
        "training_backend": args.training_backend,
        "device": args.device,
        "paired_seeds": paired_seeds,
        "dataset_scenarios": dataset_scenarios,
        "benchmark_scenarios": benchmark_scenarios,
        "windows_per_scenario": args.windows_per_scenario,
        "chunk_repeat_size": chunk_repeat_size,
        "launch_logs_dir": str(launch_logs_dir),
        "variants": [
            {
                "key": spec.key,
                "label": spec.label,
                "benchmark_name": spec.benchmark_name,
                "config_path": str(spec.config_path),
                "is_anchor": spec.is_anchor,
            }
            for spec in variants
        ],
        "results": results,
    }
    save_json(session_dir / "summary.json", summary_payload)
    _write_csv(session_dir / "summary.csv", results, variants)
    print(f"session_dir={session_dir}")
    for row in results:
        avg = row.get("avg_ler_by_mode", {})
        if not avg:
            continue
        pieces = [f"{spec.key}={avg.get(spec.benchmark_name, float('nan')):.6f}" for spec in variants]
        _safe_print(f"[seed={row['seed']}] " + " ".join(pieces))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
