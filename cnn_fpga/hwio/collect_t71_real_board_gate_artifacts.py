"""Collect checked-in read-only artifacts for T71 real-board gate regeneration."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from cnn_fpga.hwio.axi_map import AXI_REGISTER_MAP
from cnn_fpga.hwio.board_backend import BoardFPGAConfig


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "cnn_fpga" / "config" / "hardware_hil.yaml"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "t71_real_board_gate_regeneration_pack"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to hardware_hil.yaml or equivalent board config.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to write host_fact_manifest.json, device_path_probe.json, and code_side_audit.json.",
    )
    parser.add_argument(
        "--mmio-path",
        default=None,
        help="Optional read-only override for the primary MMIO candidate path.",
    )
    parser.add_argument(
        "--dma-path",
        default=None,
        help="Optional read-only override for the primary DMA candidate path.",
    )
    return parser


def _resolve_repo_path(raw: Path | str) -> Path:
    candidate = Path(raw).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (REPO_ROOT / candidate).resolve()


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_yaml(path: Path) -> dict[str, Any]:
    return dict(yaml.safe_load(path.read_text(encoding="utf-8")) or {})


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _run_command(command: list[str]) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except OSError as exc:
        return {
            "command": " ".join(command),
            "returncode": None,
            "stdout": "",
            "stderr": str(exc),
            "ok": False,
        }
    return {
        "command": " ".join(command),
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "ok": proc.returncode == 0,
    }


def _find_line(path: Path, needle: str) -> int | None:
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if needle in line:
            return index
    return None


def _repo_relative(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def _source_ref(path: Path, needle: str) -> str:
    return f"{_repo_relative(path)}:{_find_line(path, needle)}"


def _read_windows_bios_identity() -> dict[str, str | None]:
    result = {
        "SystemManufacturer": None,
        "SystemProductName": None,
        "BaseBoardManufacturer": None,
        "BaseBoardProduct": None,
        "BIOSVendor": None,
        "BIOSVersion": None,
    }
    try:
        import winreg  # type: ignore
    except ImportError:
        return result

    try:
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\BIOS")
    except OSError:
        return result
    for name in result:
        try:
            result[name] = winreg.QueryValueEx(key, name)[0]
        except OSError:
            result[name] = None
    return result


def _read_linux_dmi_identity() -> dict[str, str | None]:
    mapping = {
        "SystemManufacturer": "/sys/devices/virtual/dmi/id/sys_vendor",
        "SystemProductName": "/sys/devices/virtual/dmi/id/product_name",
        "BaseBoardManufacturer": "/sys/devices/virtual/dmi/id/board_vendor",
        "BaseBoardProduct": "/sys/devices/virtual/dmi/id/board_name",
        "BIOSVendor": "/sys/devices/virtual/dmi/id/bios_vendor",
        "BIOSVersion": "/sys/devices/virtual/dmi/id/bios_version",
    }
    result: dict[str, str | None] = {}
    for key, raw_path in mapping.items():
        path = Path(raw_path)
        if path.is_file():
            try:
                result[key] = path.read_text(encoding="utf-8", errors="replace").strip() or None
            except OSError:
                result[key] = None
        else:
            result[key] = None
    return result


def _hardware_identity() -> dict[str, str | None]:
    if platform.system() == "Windows":
        return _read_windows_bios_identity()
    if platform.system() == "Linux":
        return _read_linux_dmi_identity()
    return {
        "SystemManufacturer": None,
        "SystemProductName": None,
        "BaseBoardManufacturer": None,
        "BaseBoardProduct": None,
        "BIOSVendor": None,
        "BIOSVersion": None,
    }


def _windows_service_name_matches() -> list[str]:
    try:
        import winreg  # type: ignore
    except ImportError:
        return []

    matches: list[str] = []
    try:
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Services")
    except OSError:
        return matches
    index = 0
    needles = ("xilinx", "xdma", "fpga", "uio", "dma")
    while True:
        try:
            name = winreg.EnumKey(key, index)
        except OSError:
            break
        if any(needle in name.lower() for needle in needles):
            matches.append(name)
        index += 1
    return matches


def _pnputil_matches() -> tuple[list[str], dict[str, Any]]:
    result = _run_command(["pnputil", "/enum-devices", "/connected"])
    matches: list[str] = []
    for line in result["stdout"].splitlines():
        normalized = line.lower()
        if any(token in normalized for token in ("xilinx", "amd", "fpga", "xdma", "uio")):
            stripped = line.strip()
            if stripped:
                matches.append(stripped)
    meta = {
        "command": result["command"],
        "returncode": result["returncode"],
        "stderr": str(result["stderr"]).strip(),
    }
    return matches, meta


def _linux_driver_clues() -> dict[str, Any]:
    lspci = _run_command(["lspci", "-nn"])
    lspci_matches = [
        line.strip()
        for line in str(lspci["stdout"]).splitlines()
        if any(token in line.lower() for token in ("xilinx", "amd", "fpga", "xdma"))
    ]
    lsmod = _run_command(["lsmod"])
    lsmod_matches = [
        line.strip()
        for line in str(lsmod["stdout"]).splitlines()
        if any(token in line.lower() for token in ("xdma", "uio", "xilinx"))
    ]
    return {
        "lspci_matches": lspci_matches,
        "lspci_probe": {
            "command": lspci["command"],
            "returncode": lspci["returncode"],
            "stderr": str(lspci["stderr"]).strip(),
        },
        "module_matches": lsmod_matches,
        "lsmod_probe": {
            "command": lsmod["command"],
            "returncode": lsmod["returncode"],
            "stderr": str(lsmod["stderr"]).strip(),
        },
    }


def _host_identity_block() -> dict[str, Any]:
    base = {
        "generated_at_utc": _now_utc(),
        "interpreter": {
            "path": sys.executable,
            "version": sys.version.split()[0],
        },
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "shell": os.environ.get("SHELL", "powershell" if platform.system() == "Windows" else "unknown"),
        },
        "hardware_identity": _hardware_identity(),
        "host_probe_completed": True,
    }
    if platform.system() == "Windows":
        ver_result = _run_command(["cmd", "/c", "ver"])
        base["os"]["cmd_ver_output"] = str(ver_result["stdout"]).strip()
        matches, meta = _pnputil_matches()
        base["board_driver_clues"] = {
            "connected_device_matches": matches,
            "pnputil_probe": meta,
            "service_name_matches": _windows_service_name_matches(),
        }
        base["probe_limitations"] = [
            "Get-CimInstance Win32_OperatingSystem access denied under current permissions.",
            "Get-CimInstance Win32_ComputerSystem access denied under current permissions.",
            "Get-PnpDevice access denied under current permissions.",
            "systeminfo access denied on this host.",
        ]
    else:
        base["board_driver_clues"] = _linux_driver_clues()
        base["probe_limitations"] = []
    return base


def _probe_path(path: str, *, source: str, role: str) -> dict[str, Any]:
    exists = os.path.exists(path)
    path_type = "missing"
    if exists:
        if os.path.isdir(path):
            path_type = "directory"
        elif os.path.isfile(path):
            path_type = "file"
        else:
            path_type = "device_or_special"
    read_only_openable = False
    open_error = None
    if exists:
        try:
            fd = os.open(path, os.O_RDONLY)
        except OSError as exc:
            open_error = str(exc)
        else:
            os.close(fd)
            read_only_openable = True
    return {
        "path": path,
        "source": source,
        "role": role,
        "exists": exists,
        "path_type": path_type,
        "read_only_openable": read_only_openable,
        "open_error": open_error,
        "status": "openable_read_only" if read_only_openable else "not_found" if not exists else "exists_but_not_openable",
    }


def _device_candidates(mmio_path: str, dma_path: str) -> list[tuple[str, str, str]]:
    candidates = [
        (mmio_path, "config_or_cli_mmio_path", "mmio"),
        (dma_path, "config_or_cli_dma_path", "dma"),
    ]
    if platform.system() == "Windows":
        candidates.extend(
            [
                (r"\\.\XilinxDMA", "windows exploratory candidate", "dma"),
                (r"\\.\XDMA", "windows exploratory candidate", "dma"),
                (r"\\.\uio0", "windows exploratory candidate", "mmio"),
                (r"\\.\uio1", "windows exploratory candidate", "dma"),
            ]
        )
    return candidates


def _find_bitstream_files() -> list[str]:
    fpga_dir = REPO_ROOT / "fpga"
    if not fpga_dir.exists():
        return []
    matches: list[str] = []
    for pattern in ("*.bit", "*.bin", "*.xsa", "*.hwh", "*.bitstream"):
        for candidate in fpga_dir.rglob(pattern):
            matches.append(_repo_relative(candidate))
    return sorted(set(matches))


def _collect_host_fact_manifest(config: dict[str, Any], board_cfg: BoardFPGAConfig, *, mmio_path: str, dma_path: str) -> dict[str, Any]:
    host = _host_identity_block()
    host["probe_kind"] = "host_fact_manifest"
    host["repo_board_defaults"] = {
        "board": config.get("hil", {}).get("board"),
        "bitstream_version": config.get("hil", {}).get("bitstream_version"),
        "candidate_mmio_path": mmio_path,
        "candidate_dma_path": dma_path,
        "dma_dtype": board_cfg.dma.dtype,
        "histogram_shape": list(board_cfg.dma.histogram_shape),
        "histogram_buffer_bytes": board_cfg.dma.size_bytes,
        "buffer_count": board_cfg.dma.buffer_count,
    }
    bitstream_files = _find_bitstream_files()
    host["bitstream_evidence"] = {
        "config_bitstream_version": config.get("hil", {}).get("bitstream_version"),
        "files_found": bitstream_files,
        "source_records": [
            "cnn_fpga/config/hardware_hil.yaml: hil.board=ZCU111",
            "cnn_fpga/config/hardware_hil.yaml: hil.bitstream_version=fpga_linear_v1",
        ],
        "status": "record_only_no_bitstream_file_in_repo" if not bitstream_files else "bitstream_files_present",
    }
    return host


def _collect_device_path_probe(mmio_path: str, dma_path: str, host_fact_manifest: dict[str, Any]) -> dict[str, Any]:
    candidate_paths = [
        _probe_path(path, source=source, role=role)
        for path, source, role in _device_candidates(mmio_path, dma_path)
    ]
    driver_clues = dict(host_fact_manifest.get("board_driver_clues", {}))
    return {
        "probe_kind": "device_path_probe",
        "generated_at_utc": _now_utc(),
        "candidate_paths": candidate_paths,
        "matched_device_clues": list(driver_clues.get("connected_device_matches") or driver_clues.get("lspci_matches") or []),
        "service_name_clues": list(driver_clues.get("service_name_matches") or driver_clues.get("module_matches") or []),
        "summary": {
            "candidate_count": len(candidate_paths),
            "openable_count": sum(1 for item in candidate_paths if item["read_only_openable"]),
        },
    }


def _collect_code_side_audit(config: dict[str, Any], board_cfg: BoardFPGAConfig) -> dict[str, Any]:
    board_backend_path = REPO_ROOT / "cnn_fpga" / "hwio" / "board_backend.py"
    fpga_driver_path = REPO_ROOT / "cnn_fpga" / "hwio" / "fpga_driver.py"
    placeholder_evidence = [
        {
            "source": _source_ref(board_backend_path, "Placeholder real-board backend using memory-mapped AXI/DMA interfaces."),
            "claim": "file header declares placeholder real-board backend",
        },
        {
            "source": _source_ref(board_backend_path, "allow_missing_device: bool = True"),
            "claim": "allow_missing_device defaults to True",
        },
        {
            "source": _source_ref(board_backend_path, "board_device_missing:"),
            "claim": "missing device paths raise board_device_missing error",
        },
        {
            "source": _source_ref(
                board_backend_path,
                'return {"target_bank": None, "commit_epoch": int(commit_epoch), "version": None, "ack_delay_us": None}',
            ),
            "claim": "schedule_commit returns None-valued target_bank/version/ack_delay_us placeholders",
        },
        {
            "source": _source_ref(board_backend_path, "return []"),
            "claim": "step() returns no board-side event stream",
        },
        {
            "source": _source_ref(fpga_driver_path, 'if backend_name in {"board", "real"}:'),
            "claim": "repo driver can route into board/real selector",
        },
        {
            "source": _source_ref(fpga_driver_path, "Unified FPGA driver facade for mock and future real HIL backends."),
            "claim": "driver header still frames real HIL path as future-facing",
        },
    ]
    return {
        "probe_kind": "code_side_audit",
        "generated_at_utc": _now_utc(),
        "axi_register_map": {
            "fixed_point_spec": AXI_REGISTER_MAP.fixed_point_spec,
            "registers": {
                "ctrl_addr": hex(AXI_REGISTER_MAP.ctrl_addr),
                "status_addr": hex(AXI_REGISTER_MAP.status_addr),
                "hist_meta_addr": hex(AXI_REGISTER_MAP.hist_meta_addr),
                "overflow_count_addr": hex(AXI_REGISTER_MAP.overflow_count_addr),
                "k11_addr": hex(AXI_REGISTER_MAP.k11_addr),
                "k12_addr": hex(AXI_REGISTER_MAP.k12_addr),
                "k21_addr": hex(AXI_REGISTER_MAP.k21_addr),
                "k22_addr": hex(AXI_REGISTER_MAP.k22_addr),
                "b1_addr": hex(AXI_REGISTER_MAP.b1_addr),
                "b2_addr": hex(AXI_REGISTER_MAP.b2_addr),
                "active_bank_addr": hex(AXI_REGISTER_MAP.active_bank_addr),
                "epoch_id_addr": hex(AXI_REGISTER_MAP.epoch_id_addr),
                "commit_epoch_addr": hex(AXI_REGISTER_MAP.commit_epoch_addr),
                "hist_seq_addr": hex(AXI_REGISTER_MAP.hist_seq_addr),
            },
            "status_masks": {
                "status_ready_mask": hex(AXI_REGISTER_MAP.status_ready_mask),
                "status_hist_ready_mask": hex(AXI_REGISTER_MAP.status_hist_ready_mask),
                "status_commit_ack_mask": hex(AXI_REGISTER_MAP.status_commit_ack_mask),
                "status_overflow_alert_mask": hex(AXI_REGISTER_MAP.status_overflow_alert_mask),
            },
            "ctrl_masks": {
                "ctrl_start_mask": hex(AXI_REGISTER_MAP.ctrl_start_mask),
                "ctrl_reset_hist_mask": hex(AXI_REGISTER_MAP.ctrl_reset_hist_mask),
                "ctrl_commit_bank_mask": hex(AXI_REGISTER_MAP.ctrl_commit_bank_mask),
            },
            "bank_encoding": {"A": 0, "B": 1},
        },
        "dma_contract": {
            "config_path_field": "hil.board_io.dma_buffer_path",
            "memory_mapped_dma_config_fields": ["path", "buffer_bytes", "buffer_count"],
            "buffer_bytes": board_cfg.dma.size_bytes,
            "buffer_count": board_cfg.dma.buffer_count,
            "histogram_shape": list(board_cfg.dma.histogram_shape),
            "dtype": board_cfg.dma.dtype,
            "expected_byte_count": board_cfg.dma.size_bytes,
            "expected_byte_count_basis": "32 x 32 float32 histogram -> 4096 bytes under current config defaults",
            "readout_fields": ["buffer_id", "byte_count", 'window.payload["histogram"]', "metadata"],
        },
        "bitstream_contract": {
            "config_bitstream_version": config.get("hil", {}).get("bitstream_version"),
            "bitstream_file_present": bool(_find_bitstream_files()),
            "bitstream_alignment_confirmed": False,
            "rtl_address_table_confirmed": False,
            "dma_contract_confirmed": False,
            "fixed_point_contract_confirmed": False,
            "missing_external_facts": [
                "current-host bitstream file or board-side version readback",
                "RTL address table source bound to this host bitstream",
                "DMA payload shape/dtype confirmation against current bitstream",
                "board-side confirmation that Q4.20 host encoding matches deployed RTL contract",
            ],
        },
        "repo_execution_path": {
            "driver_supports_board_selector": True,
            "placeholder_execution_path": True,
            "placeholder_evidence": [item["claim"] for item in placeholder_evidence],
        },
        "board_backend_placeholder_evidence": placeholder_evidence,
        "code_known_facts": [
            "AXI register addresses and masks are concrete in cnn_fpga/hwio/axi_map.py.",
            "DMA readout structure is concrete in cnn_fpga/hwio/dma_client.py.",
            "FPGADriver can select backend=board/real, but current board backend still exposes placeholder semantics.",
        ],
        "external_facts_still_required": [
            "actual board device node(s) or Windows-equivalent device entry",
            "host-specific permission model for opening MMIO/DMA resources",
            "bitstream-to-RTL register map confirmation",
            "DMA byte-count/shape/dtype confirmation on the deployed board path",
        ],
    }


def collect_real_board_gate_artifacts(
    *,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    config_path: Path | str = DEFAULT_CONFIG_PATH,
    mmio_path_override: str | None = None,
    dma_path_override: str | None = None,
) -> dict[str, Path]:
    config_path = _resolve_repo_path(config_path)
    output_dir = _resolve_repo_path(output_dir)
    config = _read_yaml(config_path)
    board_cfg = BoardFPGAConfig.from_config(config)
    mmio_path = mmio_path_override or board_cfg.mmio.path
    dma_path = dma_path_override or board_cfg.dma.path

    host_fact_manifest = _collect_host_fact_manifest(config, board_cfg, mmio_path=mmio_path, dma_path=dma_path)
    device_path_probe = _collect_device_path_probe(mmio_path, dma_path, host_fact_manifest)
    code_side_audit = _collect_code_side_audit(config, board_cfg)

    host_path = _write_json(output_dir / "host_fact_manifest.json", host_fact_manifest)
    device_path = _write_json(output_dir / "device_path_probe.json", device_path_probe)
    code_audit_path = _write_json(output_dir / "code_side_audit.json", code_side_audit)
    return {
        "output_dir": output_dir,
        "host_fact_manifest_json": host_path,
        "device_path_probe_json": device_path,
        "code_side_audit_json": code_audit_path,
    }


def main() -> int:
    args = _parser().parse_args()
    result = collect_real_board_gate_artifacts(
        output_dir=Path(args.output_dir),
        config_path=Path(args.config),
        mmio_path_override=args.mmio_path,
        dma_path_override=args.dma_path,
    )
    print(json.dumps({key: str(value) for key, value in result.items()}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
