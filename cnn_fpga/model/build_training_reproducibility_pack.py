"""Build the bounded T50 training reproducibility and material-regeneration pack."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from cnn_fpga.utils.config import load_yaml_config


REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CANONICAL_STATIC_MANIFEST = REPO_ROOT / "artifacts" / "datasets" / "static_theta_v2" / "manifest.json"
DEFAULT_CANONICAL_STATIC_MODEL = (
    REPO_ROOT
    / "artifacts"
    / "models"
    / "static_theta_v2"
    / "tiny_cnn_20260319_151717_b87c6c227b57.npz"
)
DEFAULT_CANONICAL_STATIC_TRAIN_REPORT = (
    REPO_ROOT
    / "artifacts"
    / "reports"
    / "static_theta_v2"
    / "tiny_cnn_20260319_151717_b87c6c227b57_train_report.json"
)
DEFAULT_CANONICAL_RUNTIME_MANIFEST = (
    REPO_ROOT / "artifacts" / "datasets" / "runtime_b_residual_v1" / "manifest.json"
)
DEFAULT_CANONICAL_RUNTIME_MODEL = (
    REPO_ROOT
    / "artifacts"
    / "models"
    / "runtime_b_residual_v1"
    / "tiny_cnn_20260401_083648_2fc740424c0d.npz"
)
DEFAULT_CANONICAL_RUNTIME_TRAIN_REPORT = (
    REPO_ROOT
    / "artifacts"
    / "reports"
    / "runtime_b_residual_v1"
    / "tiny_cnn_20260401_083648_2fc740424c0d_train_report.json"
)
DEFAULT_RUNTIME_TRAIN_CONFIG = REPO_ROOT / "cnn_fpga" / "config" / "experiment_runtime_b_residual.yaml"
DEFAULT_HIL_RECOVERY_CONFIG = REPO_ROOT / "cnn_fpga" / "config" / "hardware_hil_recovery_smoke.yaml"
DEFAULT_P4_RECOVERY_SMOKE_CONFIG = REPO_ROOT / "cnn_fpga" / "config" / "p4_multiscenario_recovery_smoke.yaml"
DEFAULT_P4_STATCALIB_EXTENSION_CONFIG = (
    REPO_ROOT / "cnn_fpga" / "config" / "p4_multiscenario_statcalib_extension_lane.yaml"
)

DEFAULT_T50_OUTPUT_ROOT = REPO_ROOT / "artifacts" / "t50_training_repro_pack"
DEFAULT_T50_MODEL_ROOT = DEFAULT_T50_OUTPUT_ROOT / "models" / "static_theta_v2"
DEFAULT_T50_REPORT_ROOT = DEFAULT_T50_OUTPUT_ROOT / "reports" / "static_theta_v2"
DEFAULT_OUTPUT_JSON = DEFAULT_T50_OUTPUT_ROOT / "training_reproducibility_pack.json"


@dataclass(frozen=True)
class PackInputs:
    canonical_static_manifest: Path = DEFAULT_CANONICAL_STATIC_MANIFEST
    canonical_static_model: Path = DEFAULT_CANONICAL_STATIC_MODEL
    canonical_static_train_report: Path = DEFAULT_CANONICAL_STATIC_TRAIN_REPORT
    canonical_runtime_manifest: Path = DEFAULT_CANONICAL_RUNTIME_MANIFEST
    canonical_runtime_model: Path = DEFAULT_CANONICAL_RUNTIME_MODEL
    canonical_runtime_train_report: Path = DEFAULT_CANONICAL_RUNTIME_TRAIN_REPORT
    runtime_train_config: Path = DEFAULT_RUNTIME_TRAIN_CONFIG
    hil_recovery_config: Path = DEFAULT_HIL_RECOVERY_CONFIG
    p4_recovery_smoke_config: Path = DEFAULT_P4_RECOVERY_SMOKE_CONFIG
    p4_statcalib_extension_config: Path = DEFAULT_P4_STATCALIB_EXTENSION_CONFIG
    rerun_train_report: Path | None = None
    rerun_eval_report: Path | None = None
    t50_output_root: Path = DEFAULT_T50_OUTPUT_ROOT


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rerun-train-report", default=None, help="Optional override for the bounded rerun train report.")
    parser.add_argument("--rerun-eval-report", default=None, help="Optional override for the bounded rerun eval report.")
    parser.add_argument("--output-json", default=None, help="Optional output path for the generated pack JSON.")
    return parser


def _resolve(path: Path | str) -> Path:
    return Path(path).expanduser().resolve()


def _resolve_repo_path(raw: Path | str) -> Path:
    candidate = Path(raw).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (REPO_ROOT / candidate).resolve()


def _norm(path: Path | str) -> str:
    return str(_resolve(path)).replace("\\", "/").lower()


def _same_path(left: Path | str, right: Path | str) -> bool:
    return _norm(left) == _norm(right)


def _is_within(path: Path | str, root: Path | str) -> bool:
    try:
        _resolve(path).relative_to(_resolve(root))
        return True
    except ValueError:
        return False


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _read_json(path: Path) -> Any:
    return json.loads(_read_text(path))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _require_existing_file(path: Path, label: str) -> Path:
    resolved = _resolve(path)
    _require(resolved.is_file(), f"Missing required {label}: {resolved}")
    return resolved


def _require_existing_dir(path: Path, label: str) -> Path:
    resolved = _resolve(path)
    _require(resolved.is_dir(), f"Missing required {label}: {resolved}")
    return resolved


def _summarize_metric_block(block: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if block is None:
        return None
    payload: dict[str, Any] = {}
    for key in ("mse", "mae", "r2_mean"):
        if key in block:
            payload[key] = block[key]
    if "per_label" in block:
        payload["per_label_keys"] = sorted(str(key) for key in block["per_label"].keys())
    return payload


def _summarize_manifest(path: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "path": str(path),
        "run_id": manifest.get("run_id"),
        "config_hash": manifest.get("config_hash"),
        "n_samples_total": manifest.get("n_samples_total"),
        "n_train": manifest.get("n_train"),
        "n_val": manifest.get("n_val"),
        "n_test": manifest.get("n_test"),
        "label_names": list(manifest.get("label_names", [])),
        "scenarios": list(manifest.get("scenarios", [])),
    }


def _summarize_train_report(path: Path, report: Mapping[str, Any]) -> dict[str, Any]:
    tiny_cnn = report.get("tiny_cnn", {})
    return {
        "path": str(path),
        "run_name": report.get("run_name"),
        "model_path_recorded": report.get("model_path"),
        "dataset_dir_recorded": report.get("dataset_dir"),
        "model_type": report.get("model_type"),
        "n_train": report.get("n_train"),
        "n_val": report.get("n_val"),
        "training_backend": report.get("training_backend"),
        "training_device": report.get("training_device"),
        "epochs": tiny_cnn.get("epochs"),
        "patience": tiny_cnn.get("patience"),
        "train_metrics": _summarize_metric_block(report.get("train_metrics")),
        "val_metrics": _summarize_metric_block(report.get("val_metrics")),
    }


def _summarize_eval_report(path: Path, report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "path": str(path),
        "run_name": report.get("run_name"),
        "model_path_recorded": report.get("model_path"),
        "model_type": report.get("model_type"),
        "split": report.get("split"),
        "n_samples": report.get("n_samples"),
        "metrics": _summarize_metric_block(report.get("metrics")),
    }


def _summarize_file_presence(paths: Sequence[Path]) -> dict[str, Any]:
    resolved = [_resolve(path) for path in paths]
    return {
        "count": len(resolved),
        "paths": [str(path) for path in resolved],
    }


def _collect_static_theta_derived_materials(model_dir: Path, report_dir: Path) -> dict[str, Any]:
    return {
        "int8_model_artifacts": _summarize_file_presence(sorted(model_dir.glob("*_int8*.npz"))),
        "tflite_model_artifacts": _summarize_file_presence(sorted(model_dir.glob("*.tflite"))),
        "tflite_json_sidecars": _summarize_file_presence(sorted(model_dir.glob("*.tflite.json"))),
        "float_eval_reports": _summarize_file_presence(sorted(report_dir.glob("eval_test_*.json"))),
        "tflite_eval_reports": _summarize_file_presence(sorted(report_dir.glob("eval_tflite_test_*.json"))),
        "quant_reports": _summarize_file_presence(sorted(report_dir.glob("*_quant_report.json"))),
        "export_reports": _summarize_file_presence(sorted(report_dir.glob("*_export_report.json"))),
        "validate_export_reports": _summarize_file_presence(sorted(report_dir.glob("validate_export_*.json"))),
    }


def _build_canonical_chain(
    *,
    chain_name: str,
    manifest_path: Path,
    model_path: Path,
    train_report_path: Path,
    include_static_derived: bool,
) -> dict[str, Any]:
    manifest_path = _require_existing_file(manifest_path, f"{chain_name} manifest")
    model_path = _require_existing_file(model_path, f"{chain_name} float model artifact")
    train_report_path = _require_existing_file(train_report_path, f"{chain_name} train report")
    manifest = _read_json(manifest_path)
    train_report = _read_json(train_report_path)
    model_dir = model_path.parent
    report_dir = train_report_path.parent
    payload = {
        "chain_complete": True,
        "dataset_manifest": _summarize_manifest(manifest_path, manifest),
        "float_model_artifact": {
            "path": str(model_path),
            "size_bytes": model_path.stat().st_size,
        },
        "train_report": _summarize_train_report(train_report_path, train_report),
    }
    if include_static_derived:
        payload["derived_materials_presence"] = _collect_static_theta_derived_materials(model_dir, report_dir)
    return payload


def _load_config(path: Path) -> dict[str, Any]:
    resolved = _require_existing_file(path, "config")
    payload = load_yaml_config(resolved)
    _require(isinstance(payload, dict), f"Config root must be a mapping: {resolved}")
    return payload


def _require_reference_path(
    *,
    config_label: str,
    reference_key: str,
    actual_path: Path,
    expected_path: Path,
) -> dict[str, Any]:
    _require(
        _same_path(actual_path, expected_path),
        f"{config_label} drifted at {reference_key}: expected {expected_path}, got {actual_path}",
    )
    _require(actual_path.exists(), f"{config_label} points {reference_key} to missing path: {actual_path}")
    return {
        "reference_key": reference_key,
        "path": str(actual_path),
        "exists": True,
    }


def _validate_runtime_training_reference(inputs: PackInputs) -> dict[str, Any]:
    cfg = _load_config(inputs.runtime_train_config)
    canonical_runtime_dataset_dir = _resolve(inputs.canonical_runtime_manifest).parent
    canonical_runtime_model_dir = _resolve(inputs.canonical_runtime_model).parent
    canonical_runtime_report_dir = _resolve(inputs.canonical_runtime_train_report).parent
    dataset_dir = _resolve_repo_path(cfg["paths"]["dataset_dir"])
    model_dir = _resolve_repo_path(cfg["paths"]["model_dir"])
    report_dir = _resolve_repo_path(cfg["paths"]["report_dir"])
    slow_loop = cfg.get("slow_loop", {})
    model_artifact = slow_loop.get("model_artifact", {})
    inference_service = slow_loop.get("inference_service", {})
    return {
        "config_path": str(_resolve(inputs.runtime_train_config)),
        "dataset_dir": _require_reference_path(
            config_label="experiment_runtime_b_residual",
            reference_key="paths.dataset_dir",
            actual_path=dataset_dir,
            expected_path=canonical_runtime_dataset_dir,
        ),
        "model_dir": _require_reference_path(
            config_label="experiment_runtime_b_residual",
            reference_key="paths.model_dir",
            actual_path=model_dir,
            expected_path=canonical_runtime_model_dir,
        ),
        "report_dir": _require_reference_path(
            config_label="experiment_runtime_b_residual",
            reference_key="paths.report_dir",
            actual_path=report_dir,
            expected_path=canonical_runtime_report_dir,
        ),
        "model_artifact_use_latest_model_dir_artifact": bool(model_artifact.get("use_latest_model_dir_artifact")),
        "model_artifact_selector": model_artifact.get("artifact_selector"),
        "inference_backend": inference_service.get("backend"),
    }


def _validate_hil_recovery_reference(inputs: PackInputs) -> dict[str, Any]:
    cfg = _load_config(inputs.hil_recovery_config)
    canonical_static_model_dir = _resolve(inputs.canonical_static_model).parent
    model_dir = _resolve_repo_path(cfg["paths"]["model_dir"])
    slow_loop = cfg.get("slow_loop", {})
    model_artifact = slow_loop.get("model_artifact", {})
    inference_service = slow_loop.get("inference_service", {})
    summary = {
        "config_path": str(_resolve(inputs.hil_recovery_config)),
        "model_dir": _require_reference_path(
            config_label="hardware_hil_recovery_smoke",
            reference_key="paths.model_dir",
            actual_path=model_dir,
            expected_path=canonical_static_model_dir,
        ),
        "slow_loop_mode": slow_loop.get("mode"),
        "use_latest_model_dir_artifact": bool(model_artifact.get("use_latest_model_dir_artifact")),
        "artifact_selector": model_artifact.get("artifact_selector"),
        "inference_backend": inference_service.get("backend"),
    }
    _require(
        summary["slow_loop_mode"] == "model_artifact",
        f"hardware_hil_recovery_smoke slow_loop.mode changed to {summary['slow_loop_mode']}",
    )
    _require(
        summary["use_latest_model_dir_artifact"],
        "hardware_hil_recovery_smoke no longer uses latest float artifact from paths.model_dir",
    )
    return summary


def _validate_p4_recovery_reference(inputs: PackInputs) -> dict[str, Any]:
    cfg = _load_config(inputs.p4_recovery_smoke_config)
    canonical_static_model = _resolve(inputs.canonical_static_model)
    canonical_static_model_dir = canonical_static_model.parent
    model_dir = _resolve_repo_path(cfg["paths"]["model_dir"])
    slow_loop = cfg.get("slow_loop", {})
    model_artifact = slow_loop.get("model_artifact", {})
    inference_service = slow_loop.get("inference_service", {})
    explicit_model_path = _resolve_repo_path(model_artifact.get("path"))
    inference_model_path = _resolve_repo_path(inference_service.get("model_path"))
    summary = {
        "config_path": str(_resolve(inputs.p4_recovery_smoke_config)),
        "model_dir": _require_reference_path(
            config_label="p4_multiscenario_recovery_smoke",
            reference_key="paths.model_dir",
            actual_path=model_dir,
            expected_path=canonical_static_model_dir,
        ),
        "explicit_model_path": _require_reference_path(
            config_label="p4_multiscenario_recovery_smoke",
            reference_key="slow_loop.model_artifact.path",
            actual_path=explicit_model_path,
            expected_path=canonical_static_model,
        ),
        "inference_service_model_path": _require_reference_path(
            config_label="p4_multiscenario_recovery_smoke",
            reference_key="slow_loop.inference_service.model_path",
            actual_path=inference_model_path,
            expected_path=canonical_static_model,
        ),
        "use_latest_model_dir_artifact": bool(model_artifact.get("use_latest_model_dir_artifact")),
        "inference_use_latest_model_dir_artifact": bool(inference_service.get("use_latest_model_dir_artifact")),
        "artifact_selector": model_artifact.get("artifact_selector"),
        "inference_backend": inference_service.get("backend"),
    }
    _require(
        not summary["use_latest_model_dir_artifact"],
        "p4_multiscenario_recovery_smoke unexpectedly switched to latest-float selection",
    )
    _require(
        not summary["inference_use_latest_model_dir_artifact"],
        "p4_multiscenario_recovery_smoke inference_service unexpectedly switched to latest-float selection",
    )
    return summary


def _validate_p4_statcalib_extension_reference(inputs: PackInputs) -> dict[str, Any]:
    cfg = _load_config(inputs.p4_statcalib_extension_config)
    canonical_runtime_model_dir = _resolve(inputs.canonical_runtime_model).parent
    modes = cfg.get("p4_benchmark", {}).get("modes", [])
    hybrid_entry = next((item for item in modes if item.get("name") == "hybrid_residual_b"), None)
    _require(hybrid_entry is not None, "p4_multiscenario_statcalib_extension_lane is missing hybrid_residual_b entry")
    overrides = hybrid_entry.get("overrides", {})
    model_dir = _resolve_repo_path(overrides.get("paths", {}).get("model_dir"))
    summary = {
        "config_path": str(_resolve(inputs.p4_statcalib_extension_config)),
        "hybrid_residual_b_model_dir": _require_reference_path(
            config_label="p4_multiscenario_statcalib_extension_lane",
            reference_key="p4_benchmark.modes[hybrid_residual_b].overrides.paths.model_dir",
            actual_path=model_dir,
            expected_path=canonical_runtime_model_dir,
        ),
        "hybrid_residual_b_artifact_selector": hybrid_entry.get("artifact_selector"),
    }
    _require(
        summary["hybrid_residual_b_artifact_selector"] == "latest_float",
        "p4_multiscenario_statcalib_extension_lane hybrid_residual_b artifact_selector changed from latest_float",
    )
    return summary


def _discover_latest(path: Path, pattern: str, label: str) -> Path:
    directory = _require_existing_dir(path, label)
    candidates = sorted(directory.glob(pattern))
    _require(candidates, f"No files matching {pattern} under {directory}")
    return _resolve(candidates[-1])


def _resolve_rerun_train_report(inputs: PackInputs) -> Path:
    if inputs.rerun_train_report is not None:
        return _require_existing_file(inputs.rerun_train_report, "rerun train report")
    return _discover_latest(_resolve(inputs.t50_output_root) / "reports" / "static_theta_v2", "*_train_report.json", "T50 report directory")


def _resolve_rerun_eval_report(inputs: PackInputs) -> Path:
    if inputs.rerun_eval_report is not None:
        return _require_existing_file(inputs.rerun_eval_report, "rerun eval report")
    return _discover_latest(_resolve(inputs.t50_output_root) / "reports" / "static_theta_v2", "eval_test_*.json", "T50 report directory")


def _validate_bounded_rerun(inputs: PackInputs) -> dict[str, Any]:
    train_report_path = _resolve_rerun_train_report(inputs)
    eval_report_path = _resolve_rerun_eval_report(inputs)
    train_report = _read_json(train_report_path)
    eval_report = _read_json(eval_report_path)
    t50_output_root = _resolve(inputs.t50_output_root)
    t50_model_root = t50_output_root / "models" / "static_theta_v2"
    t50_report_root = t50_output_root / "reports" / "static_theta_v2"
    canonical_static_dataset_dir = _resolve(inputs.canonical_static_manifest).parent
    canonical_static_model = _resolve(inputs.canonical_static_model)

    rerun_model_path = _resolve(train_report["model_path"])
    eval_model_path = _resolve(eval_report["model_path"])
    dataset_dir = _resolve(train_report["dataset_dir"])
    _require(rerun_model_path.exists(), f"T50 rerun model artifact is missing: {rerun_model_path}")
    _require(
        _same_path(rerun_model_path, eval_model_path),
        f"T50 rerun train/eval model mismatch: train={rerun_model_path}, eval={eval_model_path}",
    )
    _require(
        _is_within(rerun_model_path, t50_model_root),
        f"T50 rerun model path is outside artifacts/t50_training_repro_pack/models/static_theta_v2: {rerun_model_path}",
    )
    _require(
        _is_within(train_report_path, t50_report_root) and _is_within(eval_report_path, t50_report_root),
        "T50 rerun reports are outside artifacts/t50_training_repro_pack/reports/static_theta_v2",
    )
    _require(
        _same_path(dataset_dir, canonical_static_dataset_dir),
        f"T50 rerun dataset drifted away from static_theta_v2: {dataset_dir}",
    )
    _require(
        str(train_report.get("training_backend")).lower() == "numpy",
        f"T50 rerun backend is not numpy: {train_report.get('training_backend')}",
    )
    _require(
        str(train_report.get("training_device")).lower() == "cpu",
        f"T50 rerun device is not cpu: {train_report.get('training_device')}",
    )
    _require(str(eval_report.get("split")).lower() == "test", f"T50 rerun eval split is not test: {eval_report.get('split')}")
    return {
        "train_report": _summarize_train_report(train_report_path, train_report),
        "eval_report": _summarize_eval_report(eval_report_path, eval_report),
        "canonical_vs_rerun_relation": {
            "canonical_static_model_path": str(canonical_static_model),
            "rerun_model_path": str(rerun_model_path),
            "dataset_dir_matches_static_theta_v2": True,
            "rerun_model_matches_eval_model": True,
            "rerun_model_is_isolated_under_t50_root": True,
            "rerun_reports_are_isolated_under_t50_root": True,
            "rerun_model_is_distinct_from_canonical_static_model": not _same_path(rerun_model_path, canonical_static_model),
            "train_backend": "numpy",
            "train_device": "cpu",
        },
    }


def build_training_reproducibility_pack(inputs: PackInputs | None = None) -> dict[str, Any]:
    inputs = PackInputs() if inputs is None else inputs
    canonical_materials = {
        "static_theta_v2": _build_canonical_chain(
            chain_name="static_theta_v2",
            manifest_path=inputs.canonical_static_manifest,
            model_path=inputs.canonical_static_model,
            train_report_path=inputs.canonical_static_train_report,
            include_static_derived=True,
        ),
        "runtime_b_residual_v1": _build_canonical_chain(
            chain_name="runtime_b_residual_v1",
            manifest_path=inputs.canonical_runtime_manifest,
            model_path=inputs.canonical_runtime_model,
            train_report_path=inputs.canonical_runtime_train_report,
            include_static_derived=False,
        ),
    }
    mainline_references = {
        "runtime_b_residual_training": _validate_runtime_training_reference(inputs),
        "hil_recovery_smoke": _validate_hil_recovery_reference(inputs),
        "p4_recovery_smoke": _validate_p4_recovery_reference(inputs),
        "p4_statcalib_extension_lane": _validate_p4_statcalib_extension_reference(inputs),
    }
    bounded_rerun = _validate_bounded_rerun(inputs)
    supported_claims = [
        "canonical static_theta_v2 dataset/float-model/train-report chain is present and its historical int8/export/eval derived materials can be enumerated from preserved artifacts.",
        "canonical runtime_b_residual_v1 dataset/float-model/train-report chain is present and still backs the mainline runtime residual model_dir references.",
        "mainline P3/HIL/P4 recovery-facing configs still point to preserved static_theta_v2 or runtime_b_residual_v1 model paths/model_dirs instead of drifting to missing artifacts.",
        "one bounded clean CPU-only train rerun completed with backend=numpy and device=cpu under artifacts/t50_training_repro_pack.",
        "one bounded eval rerun completed against the new T50 model artifact under artifacts/t50_training_repro_pack.",
    ]
    unsupported_claims = [
        "full training reproducibility across repeated runs, hosts, operating systems, and environments",
        "GPU/CUDA portability",
        "Linux portability",
        ".tflite runtime correctness on this machine",
        "real-board validation",
        "benchmark/HIL superiority changes implied by this training-material pack alone",
    ]
    clean_cpu_only_boundary = {
        "supported_surface": [
            "one bounded Windows clean Python 3.12 CPU-only tiny_cnn train rerun",
            "one bounded eval rerun against the new T50 float artifact",
            "code-backed enumeration of preserved canonical material chains",
        ],
        "unsupported_surface": unsupported_claims,
    }
    summary_table = [
        {
            "category": "canonical_materials",
            "subject": "static_theta_v2",
            "summary": "canonical dataset manifest, float model, and train report exist; historical int8/export/eval derivatives are enumerated as preserved materials only.",
        },
        {
            "category": "canonical_materials",
            "subject": "runtime_b_residual_v1",
            "summary": "canonical runtime residual dataset manifest, float model, and train report exist and remain the preserved runtime residual anchor chain.",
        },
        {
            "category": "bounded_rerun_materials",
            "subject": "T50 clean CPU-only rerun",
            "summary": "rerun train/eval outputs are isolated under artifacts/t50_training_repro_pack and tied to backend=numpy, device=cpu, dataset=static_theta_v2.",
        },
        {
            "category": "supported_claims",
            "subject": "T50 scope",
            "summary": "the repository now has a code-backed training reproducibility/material-regeneration pack bounded to canonical materials plus one clean CPU-only train+eval rerun.",
        },
        {
            "category": "unsupported_claims",
            "subject": "Out-of-bound upgrades",
            "summary": "T50 does not prove full reproducibility, GPU/CUDA/Linux portability, .tflite correctness, real-board validation, or any benchmark/HIL promotion.",
        },
    ]
    historical_inputs_read_only = [
        str(_resolve(inputs.canonical_static_manifest)),
        str(_resolve(inputs.canonical_static_model)),
        str(_resolve(inputs.canonical_static_train_report)),
        str(_resolve(inputs.canonical_runtime_manifest)),
        str(_resolve(inputs.canonical_runtime_model)),
        str(_resolve(inputs.canonical_runtime_train_report)),
        str(_resolve(inputs.runtime_train_config)),
        str(_resolve(inputs.hil_recovery_config)),
        str(_resolve(inputs.p4_recovery_smoke_config)),
        str(_resolve(inputs.p4_statcalib_extension_config)),
    ]
    return {
        "task_id": "T50",
        "historical_inputs_read_only": historical_inputs_read_only,
        "canonical_materials": canonical_materials,
        "mainline_preserved_model_references": mainline_references,
        "bounded_rerun_materials": bounded_rerun,
        "supported_claims": supported_claims,
        "unsupported_claims": unsupported_claims,
        "clean_cpu_only_reproducibility_boundary": clean_cpu_only_boundary,
        "summary_table": summary_table,
    }


def main() -> int:
    args = _parser().parse_args()
    inputs = PackInputs(
        rerun_train_report=None if args.rerun_train_report is None else Path(args.rerun_train_report),
        rerun_eval_report=None if args.rerun_eval_report is None else Path(args.rerun_eval_report),
    )
    payload = build_training_reproducibility_pack(inputs)
    output_json = DEFAULT_OUTPUT_JSON if args.output_json is None else _resolve(args.output_json)
    _write_json(output_json, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"Pack JSON: {output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
