"""Build the bounded T48 true TFLite runtime gate JSON."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_T50_PACK = REPO_ROOT / "artifacts" / "t50_training_repro_pack" / "training_reproducibility_pack.json"
DEFAULT_ENV_PROBE = REPO_ROOT / "artifacts" / "t48_true_tflite_runtime_gate" / "runtime_env_probe.json"
DEFAULT_GATE_CONFIG = REPO_ROOT / "cnn_fpga" / "config" / "task_tmp" / "T48_static_theta_tflite_gate.yaml"
DEFAULT_OUTPUT_JSON = REPO_ROOT / "artifacts" / "t48_true_tflite_runtime_gate" / "t48_true_tflite_runtime_gate.json"


@dataclass(frozen=True)
class GateInputs:
    t50_pack_json: Path = DEFAULT_T50_PACK
    env_probe_json: Path = DEFAULT_ENV_PROBE
    gate_config_path: Path | None = DEFAULT_GATE_CONFIG
    compatibility_probe_json: Path | None = None
    float_tflite_path: Path | None = None
    int8_tflite_path: Path | None = None
    float_eval_report: Path | None = None
    float_validate_report: Path | None = None
    int8_eval_report: Path | None = None
    int8_validate_report: Path | None = None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t50-pack-json", default=str(DEFAULT_T50_PACK), help="Path to T50 pack JSON.")
    parser.add_argument("--env-probe-json", default=str(DEFAULT_ENV_PROBE), help="Path to runtime env probe JSON.")
    parser.add_argument("--gate-config-path", default=str(DEFAULT_GATE_CONFIG), help="Optional task-scoped config path.")
    parser.add_argument("--compatibility-probe-json", default=None, help="Optional preserved .tflite load probe JSON.")
    parser.add_argument("--float-tflite-path", default=None, help="Optional preserved float .tflite override.")
    parser.add_argument("--int8-tflite-path", default=None, help="Optional preserved int8 .tflite override.")
    parser.add_argument("--float-eval-report", default=None, help="Path to this-round float evaluate_tflite report.")
    parser.add_argument("--float-validate-report", default=None, help="Path to this-round float validate_export report.")
    parser.add_argument("--int8-eval-report", default=None, help="Path to this-round int8 evaluate_tflite report.")
    parser.add_argument("--int8-validate-report", default=None, help="Path to this-round int8 validate_export report.")
    parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT_JSON), help="Output gate JSON path.")
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


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def _read_json(path: Path) -> Any:
    return json.loads(_read_text(path))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _require_existing_file(path: Path | str, label: str) -> Path:
    resolved = _resolve_repo_path(path)
    _require(resolved.is_file(), f"Missing required {label}: {resolved}")
    return resolved


def _get_required_mapping(parent: Mapping[str, Any], key: str, label: str) -> Mapping[str, Any]:
    value = parent.get(key)
    _require(isinstance(value, Mapping), f"Missing required {label}: {key}")
    return value


def _is_stub_path(path: Path | str) -> bool:
    candidate = Path(path)
    return candidate.suffix.lower() == ".json" or [part.lower() for part in candidate.suffixes[-2:]] == [".tflite", ".json"]


def _summarize_probe(env_probe: Mapping[str, Any]) -> dict[str, Any]:
    imports = env_probe.get("imports", {})
    _require(isinstance(imports, Mapping), "runtime env probe must provide mapping field `imports`")
    available_packages = []
    unavailable_packages = []
    for name in ("tflite_runtime", "tensorflow"):
        entry = imports.get(name, {})
        if isinstance(entry, Mapping) and bool(entry.get("ok")):
            available_packages.append(
                {
                    "name": name,
                    "version": entry.get("version"),
                    "file": entry.get("file"),
                }
            )
        else:
            unavailable_packages.append(
                {
                    "name": name,
                    "error": None if not isinstance(entry, Mapping) else entry.get("error"),
                }
            )
    preferred = None
    if any(pkg["name"] == "tflite_runtime" for pkg in available_packages):
        preferred = "tflite_runtime"
    elif any(pkg["name"] == "tensorflow" for pkg in available_packages):
        preferred = "tensorflow"
    return {
        "python_executable": env_probe.get("python_executable"),
        "python_version": env_probe.get("python_version"),
        "platform": env_probe.get("platform"),
        "runtime_available": bool(available_packages),
        "preferred_runtime_package": preferred,
        "available_packages": available_packages,
        "unavailable_packages": unavailable_packages,
    }


def _compatibility_probe_results(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    payload = _read_json(_require_existing_file(path, "compatibility probe JSON"))
    _require(isinstance(payload, Mapping), "compatibility probe JSON must be a mapping")
    raw_results = payload.get("results", [])
    _require(isinstance(raw_results, Sequence), "compatibility probe JSON must provide `results` sequence")
    out: dict[str, dict[str, Any]] = {}
    for item in raw_results:
        _require(isinstance(item, Mapping), "compatibility probe entries must be JSON objects")
        raw_path = item.get("path")
        _require(raw_path is not None, "compatibility probe entry is missing `path`")
        out[_norm(raw_path)] = dict(item)
    return out


def _static_theta_pack_entry(pack: Mapping[str, Any]) -> Mapping[str, Any]:
    canonical_materials = _get_required_mapping(pack, "canonical_materials", "T50 canonical_materials")
    return _get_required_mapping(canonical_materials, "static_theta_v2", "T50 static_theta_v2 entry")


def _paths_from_entry(entry: Mapping[str, Any], key: str) -> list[Path]:
    block = _get_required_mapping(entry, key, f"T50 {key}")
    raw_paths = block.get("paths", [])
    _require(isinstance(raw_paths, Sequence), f"T50 {key}.paths must be a sequence")
    return [_resolve_repo_path(item) for item in raw_paths]


def _ensure_true_tflite_candidate(
    *,
    candidate: Path,
    preserved_true_tflites: Sequence[Path],
    rejected_stubs: Sequence[Path],
    label: str,
) -> Path:
    resolved = _require_existing_file(candidate, label)
    _require(not _is_stub_path(resolved), f"{label} must be a true .tflite artifact, not a stub manifest: {resolved}")
    _require(resolved.suffix.lower() == ".tflite", f"{label} must end with .tflite: {resolved}")
    _require(not any(_same_path(resolved, stub) for stub in rejected_stubs), f"{label} points to a rejected stub candidate: {resolved}")
    _require(
        any(_same_path(resolved, preserved) for preserved in preserved_true_tflites),
        f"{label} is not part of the preserved true .tflite set from T50: {resolved}",
    )
    return resolved


def _select_preserved_tflite(
    *,
    preserved_true_tflites: Sequence[Path],
    rejected_stubs: Sequence[Path],
    kind: str,
    explicit_path: Path | None,
) -> Path | None:
    filtered = [
        candidate
        for candidate in preserved_true_tflites
        if ("_int8_" in candidate.name) == (kind == "int8")
    ]
    if explicit_path is not None:
        return _ensure_true_tflite_candidate(
            candidate=explicit_path,
            preserved_true_tflites=preserved_true_tflites,
            rejected_stubs=rejected_stubs,
            label=f"{kind} preserved .tflite",
        )
    if not filtered:
        return None
    return _ensure_true_tflite_candidate(
        candidate=sorted(filtered)[-1],
        preserved_true_tflites=preserved_true_tflites,
        rejected_stubs=rejected_stubs,
        label=f"{kind} preserved .tflite",
    )


def _extract_artifact_selection(inputs: GateInputs, pack: Mapping[str, Any]) -> dict[str, Any]:
    static_entry = _static_theta_pack_entry(pack)
    float_artifact_block = _get_required_mapping(static_entry, "float_model_artifact", "T50 float model artifact")
    float_artifact = _require_existing_file(float_artifact_block.get("path"), "preserved float artifact")
    derived = _get_required_mapping(static_entry, "derived_materials_presence", "T50 derived_materials_presence")
    int8_artifacts = _paths_from_entry(derived, "int8_model_artifacts")
    preserved_true_tflites = _paths_from_entry(derived, "tflite_model_artifacts")
    rejected_stubs = _paths_from_entry(derived, "tflite_json_sidecars")

    int8_artifact = None if not int8_artifacts else _require_existing_file(int8_artifacts[0], "preserved int8 artifact")
    float_tflite = _select_preserved_tflite(
        preserved_true_tflites=preserved_true_tflites,
        rejected_stubs=rejected_stubs,
        kind="float",
        explicit_path=inputs.float_tflite_path,
    )
    int8_tflite = _select_preserved_tflite(
        preserved_true_tflites=preserved_true_tflites,
        rejected_stubs=rejected_stubs,
        kind="int8",
        explicit_path=inputs.int8_tflite_path,
    )
    return {
        "selection_rule": (
            "float/int8 source artifacts come from the T50 canonical static_theta_v2 pack; "
            "true .tflite candidates are selected from T50-preserved .tflite files only, "
            "preferring the latest lexical candidate of the requested kind unless an explicit override is provided; "
            ".tflite.json stub manifests are always rejected."
        ),
        "float": {
            "artifact_path": str(float_artifact),
            "tflite_path": None if float_tflite is None else str(float_tflite),
        },
        "int8": {
            "artifact_path": None if int8_artifact is None else str(int8_artifact),
            "tflite_path": None if int8_tflite is None else str(int8_tflite),
        },
        "rejected_stub_sidecars": [str(path) for path in rejected_stubs],
        "preserved_true_tflite_candidates": [str(path) for path in sorted(preserved_true_tflites)],
    }


def _summarize_eval_report(report_path: Path, *, expected_tflite: Path, label: str) -> dict[str, Any]:
    resolved = _require_existing_file(report_path, f"{label} eval report")
    report = _read_json(resolved)
    _require(isinstance(report, Mapping), f"{label} eval report must be a JSON object: {resolved}")
    report_tflite = _require_existing_file(report.get("tflite_path"), f"{label} eval report tflite_path")
    _require(
        _same_path(report_tflite, expected_tflite),
        f"{label} eval report drifted away from selected .tflite: expected {expected_tflite}, got {report_tflite}",
    )
    metrics = report.get("metrics", {})
    _require(isinstance(metrics, Mapping), f"{label} eval report must provide metrics mapping: {resolved}")
    return {
        "report_path": str(resolved),
        "run_name": report.get("run_name"),
        "split": report.get("split"),
        "n_samples": report.get("n_samples"),
        "metrics": {
            "mse": metrics.get("mse"),
            "mae": metrics.get("mae"),
            "r2_mean": metrics.get("r2_mean"),
        },
    }


def _summarize_validate_report(
    report_path: Path,
    *,
    expected_artifact: Path,
    expected_tflite: Path,
    label: str,
) -> dict[str, Any]:
    resolved = _require_existing_file(report_path, f"{label} validate report")
    report = _read_json(resolved)
    _require(isinstance(report, Mapping), f"{label} validate report must be a JSON object: {resolved}")
    artifact_path = _require_existing_file(report.get("artifact_path"), f"{label} validate report artifact_path")
    report_tflite = _require_existing_file(report.get("tflite_path"), f"{label} validate report tflite_path")
    _require(
        _same_path(artifact_path, expected_artifact),
        f"{label} validate report drifted away from selected artifact: expected {expected_artifact}, got {artifact_path}",
    )
    _require(
        _same_path(report_tflite, expected_tflite),
        f"{label} validate report drifted away from selected .tflite: expected {expected_tflite}, got {report_tflite}",
    )
    return {
        "report_path": str(resolved),
        "split": report.get("split"),
        "n_samples": report.get("n_samples"),
        "max_abs_diff": report.get("max_abs_diff"),
        "mean_abs_diff": report.get("mean_abs_diff"),
        "status": report.get("status"),
        "warn_max_abs_diff": report.get("warn_max_abs_diff"),
    }


def _not_executed_result(*, reason: str, artifact_path: str | None, tflite_path: str | None) -> dict[str, Any]:
    return {
        "executed": False,
        "reason": reason,
        "artifact_path": artifact_path,
        "tflite_path": tflite_path,
        "eval": None,
        "validate_export": None,
        "drift_summary": None,
        "compatibility_probe": None,
    }


def _executed_result(
    *,
    artifact_path: Path,
    tflite_path: Path,
    eval_report_path: Path | None,
    validate_report_path: Path | None,
    label: str,
) -> dict[str, Any]:
    _require(eval_report_path is not None and validate_report_path is not None, f"Missing key {label} runtime report files")
    eval_summary = _summarize_eval_report(eval_report_path, expected_tflite=tflite_path, label=label)
    validate_summary = _summarize_validate_report(
        validate_report_path,
        expected_artifact=artifact_path,
        expected_tflite=tflite_path,
        label=label,
    )
    return {
        "executed": True,
        "reason": None,
        "artifact_path": str(artifact_path),
        "tflite_path": str(tflite_path),
        "eval": eval_summary,
        "validate_export": validate_summary,
        "drift_summary": {
            "max_abs_diff": validate_summary["max_abs_diff"],
            "mean_abs_diff": validate_summary["mean_abs_diff"],
            "status": validate_summary["status"],
        },
    }


def _supported_claims(
    *,
    environment_truth: Mapping[str, Any],
    float_result: Mapping[str, Any],
    int8_result: Mapping[str, Any],
) -> list[str]:
    claims = [
        (
            "A real TFLite-capable interpreter environment is available on this machine."
            if environment_truth.get("runtime_available")
            else "This task established the current machine's runtime truth and found no real TFLite-capable interpreter environment."
        ),
        "The T48 gate reuses only preserved static_theta_v2 artifacts enumerated by the T50 pack and explicitly rejects .tflite.json stubs.",
    ]
    if float_result.get("executed"):
        claims.append("The preserved float .tflite artifact really executed on this machine and was compared against the source float artifact.")
    if int8_result.get("executed"):
        claims.append("The preserved int8 .tflite artifact really executed on this machine and was compared against the source int8 artifact.")
    return claims


def _unsupported_claims(
    *,
    environment_truth: Mapping[str, Any],
    float_result: Mapping[str, Any],
    int8_result: Mapping[str, Any],
) -> list[str]:
    claims = [
        "This gate does not prove real-board validation, HIL closure, or deployment closure.",
        "This gate does not upgrade any benchmark, paper, or mainline runtime claim outside the selected static_theta_v2 preserved artifacts.",
        "This gate does not prove cross-host, cross-OS, or cross-environment TFLite portability.",
    ]
    if not environment_truth.get("runtime_available"):
        claims.append("No supported claim can be made about real .tflite execution on this machine because no usable runtime environment was established.")
    if not float_result.get("executed"):
        claims.append("No supported claim can be made about preserved float .tflite execution on this machine.")
    if not int8_result.get("executed"):
        claims.append("No supported claim can be made about preserved int8 .tflite execution on this machine.")
    return claims


def build_true_tflite_runtime_gate(inputs: GateInputs | None = None) -> dict[str, Any]:
    inputs = GateInputs() if inputs is None else inputs
    t50_pack = _read_json(_require_existing_file(inputs.t50_pack_json, "T50 pack JSON"))
    env_probe = _read_json(_require_existing_file(inputs.env_probe_json, "runtime env probe JSON"))
    _require(isinstance(t50_pack, Mapping), "T50 pack JSON must be a mapping")
    _require(isinstance(env_probe, Mapping), "runtime env probe JSON must be a mapping")

    environment_truth = _summarize_probe(env_probe)
    artifact_selection = _extract_artifact_selection(inputs, t50_pack)
    compatibility_probe = _compatibility_probe_results(inputs.compatibility_probe_json)

    float_artifact_path = artifact_selection["float"]["artifact_path"]
    float_tflite_path = artifact_selection["float"]["tflite_path"]
    int8_artifact_path = artifact_selection["int8"]["artifact_path"]
    int8_tflite_path = artifact_selection["int8"]["tflite_path"]

    if float_tflite_path is None:
        float_result = _not_executed_result(
            reason="no_preserved_true_float_tflite_candidate",
            artifact_path=float_artifact_path,
            tflite_path=None,
        )
        int8_result = _not_executed_result(
            reason="float_gate_blocked",
            artifact_path=int8_artifact_path,
            tflite_path=int8_tflite_path,
        )
        verdict = "NO_GO_PRESERVED_TFLITE_ARTIFACT_INVALID_OR_STUB_ONLY"
    elif not environment_truth["runtime_available"]:
        float_result = _not_executed_result(
            reason="runtime_unavailable",
            artifact_path=float_artifact_path,
            tflite_path=float_tflite_path,
        )
        int8_result = _not_executed_result(
            reason="runtime_unavailable",
            artifact_path=int8_artifact_path,
            tflite_path=int8_tflite_path,
        )
        verdict = "NO_GO_TRUE_TFLITE_RUNTIME_UNAVAILABLE"
    elif float_tflite_path is not None and _norm(float_tflite_path) in compatibility_probe and not compatibility_probe[_norm(float_tflite_path)].get("ok"):
        float_result = _not_executed_result(
            reason="preserved_true_tflite_incompatible_with_available_runtime",
            artifact_path=float_artifact_path,
            tflite_path=float_tflite_path,
        )
        float_result["compatibility_probe"] = dict(compatibility_probe[_norm(float_tflite_path)])
        int8_result = _not_executed_result(
            reason="float_gate_blocked_by_incompatible_preserved_true_tflite",
            artifact_path=int8_artifact_path,
            tflite_path=int8_tflite_path,
        )
        if int8_tflite_path is not None and _norm(int8_tflite_path) in compatibility_probe:
            int8_result["compatibility_probe"] = dict(compatibility_probe[_norm(int8_tflite_path)])
        verdict = "NO_GO_PRESERVED_TFLITE_ARTIFACT_INVALID_OR_STUB_ONLY"
    else:
        float_result = _executed_result(
            artifact_path=_resolve_repo_path(float_artifact_path),
            tflite_path=_resolve_repo_path(float_tflite_path),
            eval_report_path=inputs.float_eval_report,
            validate_report_path=inputs.float_validate_report,
            label="float",
        )
        if int8_artifact_path is not None and int8_tflite_path is not None and inputs.int8_eval_report and inputs.int8_validate_report:
            int8_result = _executed_result(
                artifact_path=_resolve_repo_path(int8_artifact_path),
                tflite_path=_resolve_repo_path(int8_tflite_path),
                eval_report_path=inputs.int8_eval_report,
                validate_report_path=inputs.int8_validate_report,
                label="int8",
            )
            verdict = "GO_TRUE_TFLITE_RUNTIME_FLOAT_AND_INT8"
        else:
            int8_reason = "missing_int8_reports_or_not_executed"
            if int8_artifact_path is None:
                int8_reason = "no_preserved_int8_artifact"
            elif int8_tflite_path is None:
                int8_reason = "no_preserved_true_int8_tflite_candidate"
            int8_result = _not_executed_result(
                reason=int8_reason,
                artifact_path=int8_artifact_path,
                tflite_path=int8_tflite_path,
            )
            verdict = "GO_TRUE_TFLITE_RUNTIME_FLOAT_ONLY"

    supported_claims = _supported_claims(
        environment_truth=environment_truth,
        float_result=float_result,
        int8_result=int8_result,
    )
    unsupported_claims = _unsupported_claims(
        environment_truth=environment_truth,
        float_result=float_result,
        int8_result=int8_result,
    )
    summary_table = [
        {
            "category": "environment_truth",
            "subject": "runtime_env",
            "summary": (
                f"runtime_available={environment_truth['runtime_available']}, "
                f"preferred_runtime_package={environment_truth['preferred_runtime_package']}"
            ),
        },
        {
            "category": "artifact_selection",
            "subject": "static_theta_v2 preserved pair",
            "summary": (
                f"float={float_tflite_path or 'none'}, "
                f"int8={int8_tflite_path or 'none'}, "
                f"rejected_stub_count={len(artifact_selection['rejected_stub_sidecars'])}"
            ),
        },
        {
            "category": "float_runtime_result",
            "subject": "float preserved .tflite",
            "summary": (
                f"executed={float_result['executed']}, "
                f"drift={float_result['drift_summary']}"
            ),
        },
        {
            "category": "int8_runtime_result",
            "subject": "int8 preserved .tflite",
            "summary": (
                f"executed={int8_result['executed']}, "
                f"drift={int8_result['drift_summary']}"
            ),
        },
        {
            "category": "supported_claims",
            "subject": "T48 bounded support",
            "summary": supported_claims[0],
        },
        {
            "category": "unsupported_claims",
            "subject": "T48 bounded exclusions",
            "summary": unsupported_claims[0],
        },
    ]

    historical_inputs_read_only = [
        str(_resolve_repo_path(inputs.t50_pack_json)),
        str(_resolve_repo_path(inputs.env_probe_json)),
    ]
    if inputs.gate_config_path is not None:
        historical_inputs_read_only.append(str(_resolve_repo_path(inputs.gate_config_path)))
    if inputs.compatibility_probe_json is not None:
        historical_inputs_read_only.append(str(_resolve_repo_path(inputs.compatibility_probe_json)))

    payload = {
        "task_id": "T48",
        "historical_inputs_read_only": historical_inputs_read_only,
        "environment_truth": environment_truth,
        "artifact_selection": artifact_selection,
        "float_runtime_result": float_result,
        "int8_runtime_result": int8_result,
        "final_gate_verdict": verdict,
        "supported_claims": supported_claims,
        "unsupported_claims": unsupported_claims,
        "summary_table": summary_table,
    }
    return payload


def main() -> int:
    args = _parser().parse_args()
    inputs = GateInputs(
        t50_pack_json=Path(args.t50_pack_json),
        env_probe_json=Path(args.env_probe_json),
        gate_config_path=None if args.gate_config_path in {None, ""} else Path(args.gate_config_path),
        compatibility_probe_json=None if args.compatibility_probe_json is None else Path(args.compatibility_probe_json),
        float_tflite_path=None if args.float_tflite_path is None else Path(args.float_tflite_path),
        int8_tflite_path=None if args.int8_tflite_path is None else Path(args.int8_tflite_path),
        float_eval_report=None if args.float_eval_report is None else Path(args.float_eval_report),
        float_validate_report=None if args.float_validate_report is None else Path(args.float_validate_report),
        int8_eval_report=None if args.int8_eval_report is None else Path(args.int8_eval_report),
        int8_validate_report=None if args.int8_validate_report is None else Path(args.int8_validate_report),
    )
    payload = build_true_tflite_runtime_gate(inputs)
    output_json = _resolve_repo_path(args.output_json)
    _write_json(output_json, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"Gate JSON: {output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
