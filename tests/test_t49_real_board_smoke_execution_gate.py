import json
import tempfile
import unittest
from pathlib import Path

from cnn_fpga.hwio.build_t49_real_board_smoke_gate import (
    GateInputs,
    build_real_board_smoke_gate,
)


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


class RealBoardSmokeGateTests(unittest.TestCase):
    def _write_host_manifest(
        self,
        root: Path,
        *,
        host_probe_completed: bool = True,
        bitstream_version: str | None = "fpga_linear_v1",
    ) -> Path:
        return _write_json(
            root / "host_fact_manifest.json",
            {
                "probe_kind": "host_fact_manifest",
                "host_probe_completed": host_probe_completed,
                "interpreter": {
                    "path": "C:/ProgramData/anaconda3/python.exe",
                    "version": "3.12.7",
                },
                "os": {
                    "system": "Windows",
                    "release": "11",
                    "version": "10.0.26200",
                },
                "repo_board_defaults": {
                    "board": "ZCU111",
                    "bitstream_version": bitstream_version,
                    "candidate_mmio_path": "/dev/uio0",
                    "candidate_dma_path": "/dev/uio1",
                },
                "bitstream_evidence": {
                    "config_bitstream_version": bitstream_version,
                    "files_found": [],
                    "source_records": ["cnn_fpga/config/hardware_hil.yaml"],
                },
            },
        )

    def _write_device_probe(
        self,
        root: Path,
        *,
        mmio_openable: bool,
        dma_openable: bool,
    ) -> Path:
        def _entry(path: str, openable: bool) -> dict:
            return {
                "path": path,
                "exists": openable,
                "path_type": "device" if openable else "missing",
                "read_only_openable": openable,
                "status": "openable_read_only" if openable else "not_found",
            }

        return _write_json(
            root / "device_path_probe.json",
            {
                "probe_kind": "device_path_probe",
                "candidate_paths": [
                    _entry("/dev/uio0", mmio_openable),
                    _entry("/dev/uio1", dma_openable),
                ],
                "matched_device_clues": [],
            },
        )

    def _write_code_audit(
        self,
        root: Path,
        *,
        bitstream_alignment_confirmed: bool,
        rtl_address_table_confirmed: bool,
        dma_contract_confirmed: bool,
        fixed_point_contract_confirmed: bool,
        placeholder_execution_path: bool,
    ) -> Path:
        return _write_json(
            root / "code_side_audit.json",
            {
                "probe_kind": "code_side_audit",
                "axi_register_map": {
                    "fixed_point_spec": "Q4.20",
                    "registers": {
                        "ctrl_addr": "0x00",
                        "status_addr": "0x04",
                        "hist_meta_addr": "0x08",
                        "overflow_count_addr": "0x0C",
                        "active_bank_addr": "0x30",
                        "epoch_id_addr": "0x34",
                        "commit_epoch_addr": "0x38",
                        "hist_seq_addr": "0x3C",
                    },
                },
                "dma_contract": {
                    "buffer_bytes": 4096,
                    "buffer_count": 2,
                    "histogram_shape": [32, 32],
                    "dtype": "float32",
                },
                "bitstream_contract": {
                    "bitstream_alignment_confirmed": bitstream_alignment_confirmed,
                    "rtl_address_table_confirmed": rtl_address_table_confirmed,
                    "dma_contract_confirmed": dma_contract_confirmed,
                    "fixed_point_contract_confirmed": fixed_point_contract_confirmed,
                },
                "repo_execution_path": {
                    "driver_supports_board_selector": True,
                    "placeholder_execution_path": placeholder_execution_path,
                    "placeholder_evidence": [
                        "board_backend.py header says placeholder real-board backend",
                        "schedule_commit returns target_bank/version/ack_delay_us None",
                        "step() returns []",
                    ],
                },
            },
        )

    def test_host_or_device_missing_yields_host_device_no_go(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            gate = build_real_board_smoke_gate(
                GateInputs(
                    host_fact_manifest_json=self._write_host_manifest(root),
                    device_path_probe_json=self._write_device_probe(root, mmio_openable=False, dma_openable=False),
                    code_side_audit_json=self._write_code_audit(
                        root,
                        bitstream_alignment_confirmed=False,
                        rtl_address_table_confirmed=False,
                        dma_contract_confirmed=False,
                        fixed_point_contract_confirmed=False,
                        placeholder_execution_path=True,
                    ),
                )
            )

        self.assertEqual(gate["final_gate_verdict"], "NO_GO_REAL_BOARD_HOST_OR_DEVICE_PATH_UNAVAILABLE")
        self.assertEqual(gate["device_path_truth"]["status"], "not_ready")

    def test_missing_bitstream_or_contract_yields_contract_no_go(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            gate = build_real_board_smoke_gate(
                GateInputs(
                    host_fact_manifest_json=self._write_host_manifest(root),
                    device_path_probe_json=self._write_device_probe(root, mmio_openable=True, dma_openable=True),
                    code_side_audit_json=self._write_code_audit(
                        root,
                        bitstream_alignment_confirmed=False,
                        rtl_address_table_confirmed=False,
                        dma_contract_confirmed=False,
                        fixed_point_contract_confirmed=False,
                        placeholder_execution_path=True,
                    ),
                )
            )

        self.assertEqual(gate["final_gate_verdict"], "NO_GO_REAL_BOARD_BITSTREAM_OR_AXI_DMA_CONTRACT_UNCONFIRMED")
        self.assertEqual(gate["bitstream_and_contract_truth"]["status"], "not_ready")

    def test_placeholder_repo_execution_path_yields_placeholder_no_go(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            gate = build_real_board_smoke_gate(
                GateInputs(
                    host_fact_manifest_json=self._write_host_manifest(root),
                    device_path_probe_json=self._write_device_probe(root, mmio_openable=True, dma_openable=True),
                    code_side_audit_json=self._write_code_audit(
                        root,
                        bitstream_alignment_confirmed=True,
                        rtl_address_table_confirmed=True,
                        dma_contract_confirmed=True,
                        fixed_point_contract_confirmed=True,
                        placeholder_execution_path=True,
                    ),
                )
            )

        self.assertEqual(gate["final_gate_verdict"], "NO_GO_REAL_BOARD_REPO_EXECUTION_PATH_PLACEHOLDER_ONLY")
        self.assertEqual(gate["repo_execution_path_truth"]["status"], "placeholder_only")

    def test_all_layers_ready_yields_go(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            gate = build_real_board_smoke_gate(
                GateInputs(
                    host_fact_manifest_json=self._write_host_manifest(root),
                    device_path_probe_json=self._write_device_probe(root, mmio_openable=True, dma_openable=True),
                    code_side_audit_json=self._write_code_audit(
                        root,
                        bitstream_alignment_confirmed=True,
                        rtl_address_table_confirmed=True,
                        dma_contract_confirmed=True,
                        fixed_point_contract_confirmed=True,
                        placeholder_execution_path=False,
                    ),
                )
            )

        self.assertEqual(gate["final_gate_verdict"], "GO_REAL_BOARD_SMOKE_EXECUTION_PRECONDITIONS_READY")
        self.assertEqual(gate["repo_execution_path_truth"]["status"], "ready")


if __name__ == "__main__":
    unittest.main()
