"""Build the bounded T49 real-board smoke execution gate JSON."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HOST_FACT_MANIFEST = REPO_ROOT / "artifacts" / "t49_real_board_smoke_execution_gate" / "host_fact_manifest.json"
DEFAULT_DEVICE_PATH_PROBE = REPO_ROOT / "artifacts" / "t49_real_board_smoke_execution_gate" / "device_path_probe.json"
DEFAULT_CODE_SIDE_AUDIT = REPO_ROOT / "artifacts" / "t49_real_board_smoke_execution_gate" / "code_side_audit.json"
DEFAULT_OUTPUT_JSON = REPO_ROOT / "artifacts" / "t49_real_board_smoke_execution_gate" / "t49_real_board_smoke_execution_gate.json"


@dataclass(frozen=True)
class GateInputs:
    host_fact_manifest_json: Path = DEFAULT_HOST_FACT_MANIFEST
    device_path_probe_json: Path = DEFAULT_DEVICE_PATH_PROBE
    code_side_audit_json: Path = DEFAULT_CODE_SIDE_AUDIT


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--host-fact-manifest-json",
        default=str(DEFAULT_HOST_FACT_MANIFEST),
        help="Path to host_fact_manifest.json.",
    )
    parser.add_argument(
        "--device-path-probe-json",
        default=str(DEFAULT_DEVICE_PATH_PROBE),
        help="Path to device_path_probe.json.",
    )
    parser.add_argument(
        "--code-side-audit-json",
        default=str(DEFAULT_CODE_SIDE_AUDIT),
        help="Path to code_side_audit.json.",
    )
    parser.add_argument(
        "--output-json",
        default=str(DEFAULT_OUTPUT_JSON),
        help="Output gate JSON path.",
    )
    return parser


def _resolve_repo_path(raw: Path | str) -> Path:
    candidate = Path(raw).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (REPO_ROOT / candidate).resolve()


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


def _get_mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = parent.get(key)
    _require(isinstance(value, Mapping), f"Missing required mapping: {key}")
    return value


def _get_list(parent: Mapping[str, Any], key: str) -> list[Any]:
    value = parent.get(key, [])
    _require(isinstance(value, list), f"Expected list field: {key}")
    return value


def _evaluate_host_environment(host_manifest: Mapping[str, Any]) -> dict[str, Any]:
    interpreter = _get_mapping(host_manifest, "interpreter")
    os_block = _get_mapping(host_manifest, "os")
    ready = bool(host_manifest.get("host_probe_completed")) and bool(interpreter.get("path")) and bool(os_block.get("system"))
    satisfied = []
    missing = []
    if host_manifest.get("host_probe_completed"):
        satisfied.append("host_probe_completed")
    else:
        missing.append("host_probe_completed")
    if interpreter.get("path"):
        satisfied.append("python_interpreter_path_recorded")
    else:
        missing.append("python_interpreter_path_missing")
    if interpreter.get("version"):
        satisfied.append("python_version_recorded")
    else:
        missing.append("python_version_missing")
    if os_block.get("system"):
        satisfied.append("os_identity_recorded")
    else:
        missing.append("os_identity_missing")
    summary = (
        f"host_probe_completed={bool(host_manifest.get('host_probe_completed'))}, "
        f"os={os_block.get('system')} {os_block.get('release')} {os_block.get('version')}, "
        f"interpreter={interpreter.get('path')}"
    )
    return {
        "status": "ready" if ready else "not_ready",
        "satisfied": satisfied,
        "missing": missing,
        "summary": summary,
    }


def _evaluate_device_path_truth(device_probe: Mapping[str, Any]) -> dict[str, Any]:
    candidates = _get_list(device_probe, "candidate_paths")
    openable_paths: list[str] = []
    missing_paths: list[str] = []
    openable_mmio_paths: list[str] = []
    openable_dma_paths: list[str] = []
    openable_unknown_role_paths: list[str] = []
    for item in candidates:
        _require(isinstance(item, Mapping), "device path probe candidate must be a JSON object")
        path = str(item.get("path", "missing_path"))
        role = str(item.get("role", "unknown")).lower()
        if bool(item.get("read_only_openable")):
            openable_paths.append(path)
            if role == "mmio":
                openable_mmio_paths.append(path)
            elif role == "dma":
                openable_dma_paths.append(path)
            else:
                openable_unknown_role_paths.append(path)
        else:
            missing_paths.append(path)
    matched_clues = _get_list(device_probe, "matched_device_clues")
    satisfied_roles = []
    missing_roles = []
    if openable_mmio_paths:
        satisfied_roles.append("mmio")
    else:
        missing_roles.append("mmio")
    if openable_dma_paths:
        satisfied_roles.append("dma")
    else:
        missing_roles.append("dma")
    ready = bool(openable_mmio_paths) and bool(openable_dma_paths)
    summary = (
        f"openable_mmio_paths={len(openable_mmio_paths)}, "
        f"openable_dma_paths={len(openable_dma_paths)}, "
        f"openable_unknown_role_paths={len(openable_unknown_role_paths)}, "
        f"matched_device_clues={len(matched_clues)}"
    )
    return {
        "status": "ready" if ready else "not_ready",
        "openable_paths": openable_paths,
        "openable_mmio_paths": openable_mmio_paths,
        "openable_dma_paths": openable_dma_paths,
        "openable_unknown_role_paths": openable_unknown_role_paths,
        "missing_paths": missing_paths,
        "satisfied_roles": satisfied_roles,
        "missing_roles": missing_roles,
        "matched_device_clues": matched_clues,
        "summary": summary,
    }


def _evaluate_bitstream_and_contract_truth(
    host_manifest: Mapping[str, Any],
    code_audit: Mapping[str, Any],
) -> dict[str, Any]:
    repo_defaults = _get_mapping(host_manifest, "repo_board_defaults")
    bitstream_evidence = _get_mapping(host_manifest, "bitstream_evidence")
    bitstream_contract = _get_mapping(code_audit, "bitstream_contract")

    bitstream_identifier = (
        bitstream_evidence.get("config_bitstream_version")
        or repo_defaults.get("bitstream_version")
        or bitstream_contract.get("config_bitstream_version")
    )
    checks = {
        "bitstream_identifier_present": bool(bitstream_identifier),
        "bitstream_alignment_confirmed": bool(bitstream_contract.get("bitstream_alignment_confirmed")),
        "rtl_address_table_confirmed": bool(bitstream_contract.get("rtl_address_table_confirmed")),
        "dma_contract_confirmed": bool(bitstream_contract.get("dma_contract_confirmed")),
        "fixed_point_contract_confirmed": bool(bitstream_contract.get("fixed_point_contract_confirmed")),
    }
    ready = all(checks.values())
    satisfied = [name for name, ok in checks.items() if ok]
    missing = [name for name, ok in checks.items() if not ok]
    summary = f"bitstream_identifier={bitstream_identifier}, satisfied={len(satisfied)}, missing={len(missing)}"
    return {
        "status": "ready" if ready else "not_ready",
        "bitstream_identifier": bitstream_identifier,
        "satisfied": satisfied,
        "missing": missing,
        "summary": summary,
    }


def _evaluate_repo_execution_path_truth(code_audit: Mapping[str, Any]) -> dict[str, Any]:
    repo_path = _get_mapping(code_audit, "repo_execution_path")
    driver_supports_board_selector = bool(repo_path.get("driver_supports_board_selector"))
    placeholder_execution_path = bool(repo_path.get("placeholder_execution_path"))
    placeholder_evidence = _get_list(repo_path, "placeholder_evidence")
    if driver_supports_board_selector and not placeholder_execution_path:
        status = "ready"
    elif placeholder_execution_path:
        status = "placeholder_only"
    else:
        status = "not_ready"
    satisfied = []
    missing = []
    if driver_supports_board_selector:
        satisfied.append("driver_supports_board_selector")
    else:
        missing.append("driver_supports_board_selector_missing")
    if placeholder_execution_path:
        missing.append("repo_execution_path_still_placeholder_only")
    else:
        satisfied.append("repo_execution_path_not_placeholder_only")
    summary = (
        f"driver_supports_board_selector={driver_supports_board_selector}, "
        f"placeholder_execution_path={placeholder_execution_path}"
    )
    return {
        "status": status,
        "satisfied": satisfied,
        "missing": missing,
        "placeholder_evidence": placeholder_evidence,
        "summary": summary,
    }


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _supported_claims(verdict: str) -> list[str]:
    claims = [
        "当前任务只完成了当前宿主只读 host/device probe、代码侧 AXI/DMA 审计和 gate 聚合，没有执行真板 smoke。",
        "仓库内 `board`/`real` 入口、AXI 地址表和 DMA 数据结构都能被代码侧准确定位与复核。",
    ]
    if verdict == "GO_REAL_BOARD_SMOKE_EXECUTION_PRECONDITIONS_READY":
        claims.append("当前 host/device/bitstream-contract/repo-path 四层前提都已齐备，可以考虑单开后续 bounded real-board smoke 执行任务。")
    elif verdict == "NO_GO_REAL_BOARD_HOST_OR_DEVICE_PATH_UNAVAILABLE":
        claims.append("当前最多只能支持“仓库存在 board-path scaffolding，但当前机器没有可读打开的真板设备路径”。")
    elif verdict == "NO_GO_REAL_BOARD_BITSTREAM_OR_AXI_DMA_CONTRACT_UNCONFIRMED":
        claims.append("当前最多只能支持“宿主与候选设备路径存在，但 bitstream/RTL/DMA contract 仍未被确认”。")
    else:
        claims.append("当前最多只能支持“宿主、设备路径和 contract 线索可能存在，但 repo 执行路径仍属于 placeholder-only”。")
    return claims


def _unsupported_claims() -> list[str]:
    return [
        "不支持把本任务写成 real-board smoke 已执行成功。",
        "不支持把本任务写成 P3 真板 HIL 已完成、board backend 已验收或 deployment closure。",
        "不支持把只读 host/device/code 审计外推成 benchmark、HIL promotion 或硬件性能结论。",
    ]


def _verdict_and_statement(
    host_environment: Mapping[str, Any],
    device_path_truth: Mapping[str, Any],
    bitstream_and_contract_truth: Mapping[str, Any],
    repo_execution_path_truth: Mapping[str, Any],
) -> tuple[str, str]:
    if host_environment.get("status") != "ready" or device_path_truth.get("status") != "ready":
        return (
            "NO_GO_REAL_BOARD_HOST_OR_DEVICE_PATH_UNAVAILABLE",
            "当前机器已完成只读宿主事实探测，但没有找到可读打开的真板设备路径，因此还不具备进入 bounded real-board smoke 的最小 host/device 前提。",
        )
    if bitstream_and_contract_truth.get("status") != "ready":
        return (
            "NO_GO_REAL_BOARD_BITSTREAM_OR_AXI_DMA_CONTRACT_UNCONFIRMED",
            "当前机器可能已经具备宿主和设备路径前提，但 bitstream 标识、RTL 地址表和 DMA/fixed-point contract 仍未完成可引用确认。",
        )
    if repo_execution_path_truth.get("status") != "ready":
        return (
            "NO_GO_REAL_BOARD_REPO_EXECUTION_PATH_PLACEHOLDER_ONLY",
            "当前宿主和 contract 线索即使齐备，仓库里的 board 执行路径仍属于 placeholder-only，不能支撑 real-board smoke 已可执行的表述。",
        )
    return (
        "GO_REAL_BOARD_SMOKE_EXECUTION_PRECONDITIONS_READY",
        "当前宿主、设备路径、bitstream/contract 和 repo execution path 四层前提都已就位，可以考虑单开后续 bounded real-board smoke 执行任务。",
    )


def build_real_board_smoke_gate(inputs: GateInputs | None = None) -> dict[str, Any]:
    inputs = GateInputs() if inputs is None else inputs
    host_manifest_path = _require_existing_file(inputs.host_fact_manifest_json, "host fact manifest JSON")
    device_probe_path = _require_existing_file(inputs.device_path_probe_json, "device path probe JSON")
    code_audit_path = _require_existing_file(inputs.code_side_audit_json, "code-side audit JSON")

    host_manifest = _read_json(host_manifest_path)
    device_probe = _read_json(device_probe_path)
    code_audit = _read_json(code_audit_path)
    _require(isinstance(host_manifest, Mapping), "host fact manifest JSON must be a mapping")
    _require(isinstance(device_probe, Mapping), "device path probe JSON must be a mapping")
    _require(isinstance(code_audit, Mapping), "code-side audit JSON must be a mapping")

    host_environment = _evaluate_host_environment(host_manifest)
    device_path_truth = _evaluate_device_path_truth(device_probe)
    bitstream_and_contract_truth = _evaluate_bitstream_and_contract_truth(host_manifest, code_audit)
    repo_execution_path_truth = _evaluate_repo_execution_path_truth(code_audit)
    verdict, strongest_statement = _verdict_and_statement(
        host_environment=host_environment,
        device_path_truth=device_path_truth,
        bitstream_and_contract_truth=bitstream_and_contract_truth,
        repo_execution_path_truth=repo_execution_path_truth,
    )

    satisfied = _dedupe(
        list(host_environment.get("satisfied", []))
        + list(device_path_truth.get("openable_paths", []))
        + list(device_path_truth.get("satisfied_roles", []))
        + list(bitstream_and_contract_truth.get("satisfied", []))
        + list(repo_execution_path_truth.get("satisfied", []))
    )
    missing = _dedupe(
        list(host_environment.get("missing", []))
        + list(device_path_truth.get("missing_paths", []))
        + list(device_path_truth.get("missing_roles", []))
        + list(bitstream_and_contract_truth.get("missing", []))
        + list(repo_execution_path_truth.get("missing", []))
    )
    supported_claims = _supported_claims(verdict)
    unsupported_claims = _unsupported_claims()

    summary_table = [
        {
            "category": "host_environment",
            "subject": "current_host",
            "summary": host_environment["summary"],
        },
        {
            "category": "device_path_truth",
            "subject": "candidate_device_paths",
            "summary": device_path_truth["summary"],
        },
        {
            "category": "bitstream_and_contract_truth",
            "subject": "bitstream_axi_dma_contract",
            "summary": bitstream_and_contract_truth["summary"],
        },
        {
            "category": "repo_execution_path_truth",
            "subject": "board_backend_path",
            "summary": repo_execution_path_truth["summary"],
        },
        {
            "category": "supported_claims",
            "subject": "T49 bounded support",
            "summary": supported_claims[0],
        },
        {
            "category": "unsupported_claims",
            "subject": "T49 bounded exclusions",
            "summary": unsupported_claims[0],
        },
    ]

    return {
        "task_id": "T49",
        "input_jsons_read_only": [
            str(host_manifest_path),
            str(device_probe_path),
            str(code_audit_path),
        ],
        "host_environment": host_environment,
        "device_path_truth": device_path_truth,
        "bitstream_and_contract_truth": bitstream_and_contract_truth,
        "repo_execution_path_truth": repo_execution_path_truth,
        "preconditions_satisfied": satisfied,
        "missing_preconditions": missing,
        "current_strongest_supported_statement": strongest_statement,
        "supported_claims": supported_claims,
        "unsupported_claims": unsupported_claims,
        "final_gate_verdict": verdict,
        "summary_table": summary_table,
    }


def main() -> int:
    args = _parser().parse_args()
    payload = build_real_board_smoke_gate(
        GateInputs(
            host_fact_manifest_json=Path(args.host_fact_manifest_json),
            device_path_probe_json=Path(args.device_path_probe_json),
            code_side_audit_json=Path(args.code_side_audit_json),
        )
    )
    output_json = _resolve_repo_path(args.output_json)
    _write_json(output_json, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"Gate JSON: {output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
