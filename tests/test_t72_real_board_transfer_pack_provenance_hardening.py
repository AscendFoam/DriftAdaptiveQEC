import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from cnn_fpga.hwio.build_t49_real_board_smoke_gate import GateInputs, build_real_board_smoke_gate
from cnn_fpga.hwio import collect_t71_real_board_gate_artifacts as collector


REPO_ROOT = Path(__file__).resolve().parents[1]
T49_ARTIFACT_ROOT = REPO_ROOT / "artifacts" / "t49_real_board_smoke_execution_gate"
DEFAULT_CONFIG_PATH = REPO_ROOT / "cnn_fpga" / "config" / "hardware_hil.yaml"


class TransferPackProvenanceHardeningTests(unittest.TestCase):
    def _stub_host_identity(self) -> dict:
        return {
            "generated_at_utc": "2026-06-11T00:00:00+00:00",
            "interpreter": {"path": "python", "version": "3.13.0"},
            "os": {
                "system": "Windows",
                "release": "11",
                "version": "test-version",
                "platform": "Windows-test",
                "machine": "AMD64",
                "processor": "test-cpu",
                "shell": "powershell",
            },
            "hardware_identity": {
                "SystemManufacturer": "TestVendor",
                "SystemProductName": "TestSystem",
                "BaseBoardManufacturer": "TestVendor",
                "BaseBoardProduct": "TestBoard",
                "BIOSVendor": "TestBIOS",
                "BIOSVersion": "1.0",
            },
            "host_probe_completed": True,
            "board_driver_clues": {
                "connected_device_matches": [],
                "pnputil_probe": {"command": "pnputil /enum-devices /connected", "returncode": 0, "stderr": ""},
                "service_name_matches": [],
            },
            "probe_execution_records": [],
            "probe_limitations": [],
        }

    def _write_temp_config(
        self,
        root: Path,
        *,
        board: str = "ZCU111-T72",
        bitstream_version: str = "fpga_linear_v1_t72",
        mmio_path: str = "/tmp/t72_cfg_mmio",
        dma_path: str = "/tmp/t72_cfg_dma",
        hist_bins: int = 16,
        dma_buffer_bytes: int = 1024,
        dma_buffer_count: int = 3,
        dma_dtype: str = "float32",
    ) -> Path:
        config = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
        config["hil"]["board"] = board
        config["hil"]["bitstream_version"] = bitstream_version
        config["hil"]["board_io"]["axi_uio_path"] = mmio_path
        config["hil"]["board_io"]["dma_buffer_path"] = dma_path
        config["hil"]["board_io"]["dma_dtype"] = dma_dtype
        config["fast_loop"]["histogram_bins"] = hist_bins
        config["dma"]["histogram_buffer_bytes"] = dma_buffer_bytes
        config["dma"]["buffer_count"] = dma_buffer_count
        path = root / "temp_hardware_hil.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
        return path

    def test_config_override_updates_dynamic_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = self._write_temp_config(root)
            output_dir = root / "out"
            with patch.object(collector, "_host_identity_block", return_value=self._stub_host_identity()):
                collected = collector.collect_real_board_gate_artifacts(output_dir=output_dir, config_path=config_path)

            host_manifest = json.loads(collected["host_fact_manifest_json"].read_text(encoding="utf-8"))
            code_audit = json.loads(collected["code_side_audit_json"].read_text(encoding="utf-8"))

        source_records = host_manifest["bitstream_evidence"]["source_records"]
        self.assertEqual(source_records[0]["config_path"].replace("\\", "/").split("/")[-1], "temp_hardware_hil.yaml")
        self.assertEqual(source_records[0]["field"], "hil.board")
        self.assertEqual(source_records[0]["value"], "ZCU111-T72")
        self.assertEqual(source_records[1]["field"], "hil.bitstream_version")
        self.assertEqual(source_records[1]["value"], "fpga_linear_v1_t72")

        repo_defaults = host_manifest["repo_board_defaults"]
        self.assertEqual(repo_defaults["config_path"].replace("\\", "/").split("/")[-1], "temp_hardware_hil.yaml")
        self.assertEqual(repo_defaults["config_argument_kind"], "override")

        expected_basis = code_audit["dma_contract"]["expected_byte_count_basis"]
        self.assertEqual(expected_basis["histogram_shape"], [16, 16])
        self.assertEqual(expected_basis["dtype"], "float32")
        self.assertEqual(expected_basis["computed_byte_count"], 1024)
        self.assertTrue(expected_basis["matches_configured_buffer_bytes"])
        self.assertNotIn("under current config defaults", json.dumps(expected_basis, ensure_ascii=False))

    def test_path_overrides_are_reflected_in_probe_and_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = self._write_temp_config(root, mmio_path="/tmp/from_config_mmio", dma_path="/tmp/from_config_dma")
            output_dir = root / "out"
            with patch.object(collector, "_host_identity_block", return_value=self._stub_host_identity()):
                collected = collector.collect_real_board_gate_artifacts(
                    output_dir=output_dir,
                    config_path=config_path,
                    mmio_path_override="/tmp/from_override_mmio",
                    dma_path_override="/tmp/from_override_dma",
                )

            host_manifest = json.loads(collected["host_fact_manifest_json"].read_text(encoding="utf-8"))
            device_probe = json.loads(collected["device_path_probe_json"].read_text(encoding="utf-8"))

        repo_defaults = host_manifest["repo_board_defaults"]
        self.assertEqual(repo_defaults["candidate_mmio_path"], "/tmp/from_override_mmio")
        self.assertEqual(repo_defaults["candidate_dma_path"], "/tmp/from_override_dma")
        self.assertEqual(repo_defaults["candidate_mmio_path_record"]["source_kind"], "cli_override")
        self.assertEqual(repo_defaults["candidate_mmio_path_record"]["config_value"], "/tmp/from_config_mmio")
        self.assertEqual(repo_defaults["candidate_dma_path_record"]["source_kind"], "cli_override")
        self.assertEqual(repo_defaults["candidate_dma_path_record"]["config_value"], "/tmp/from_config_dma")

        self.assertEqual(device_probe["candidate_paths"][0]["path"], "/tmp/from_override_mmio")
        self.assertEqual(device_probe["candidate_paths"][0]["source"], "cli_override_mmio_path")
        self.assertEqual(device_probe["candidate_paths"][1]["path"], "/tmp/from_override_dma")
        self.assertEqual(device_probe["candidate_paths"][1]["source"], "cli_override_dma_path")

    def test_probe_limitations_distinguish_failed_and_not_applicable_probes(self) -> None:
        def fake_run_command(command: list[str]) -> dict:
            joined = " ".join(command)
            if command[:3] == ["cmd", "/c", "ver"]:
                return {"command": joined, "returncode": 0, "stdout": "Microsoft Windows [Version test]", "stderr": "", "ok": True}
            if "Win32_OperatingSystem" in joined:
                return {"command": joined, "returncode": 5, "stdout": "", "stderr": "Access is denied.", "ok": False}
            if "Win32_ComputerSystem" in joined:
                return {"command": joined, "returncode": 0, "stdout": "Manufacturer : TestVendor", "stderr": "", "ok": True}
            if "Get-PnpDevice" in joined:
                return {"command": joined, "returncode": 1, "stdout": "", "stderr": "The term 'Get-PnpDevice' is not recognized.", "ok": False}
            if command[0] == "pnputil":
                return {"command": joined, "returncode": 0, "stdout": "", "stderr": "", "ok": True}
            if command[0] == "systeminfo":
                return {"command": joined, "returncode": 1, "stdout": "", "stderr": "ERROR: Access denied", "ok": False}
            raise AssertionError(f"Unexpected command: {joined}")

        with (
            patch.object(collector.platform, "system", return_value="Windows"),
            patch.object(collector, "_run_command", side_effect=fake_run_command),
            patch.object(collector, "_pnputil_matches", return_value=([], {"command": "pnputil /enum-devices /connected", "returncode": 0, "stderr": ""})),
            patch.object(collector, "_windows_service_name_matches", return_value=[]),
            patch.object(
                collector,
                "_hardware_identity",
                return_value={
                    "SystemManufacturer": "TestVendor",
                    "SystemProductName": "TestSystem",
                    "BaseBoardManufacturer": "TestVendor",
                    "BaseBoardProduct": "TestBoard",
                    "BIOSVendor": "TestBIOS",
                    "BIOSVersion": "1.0",
                },
            ),
        ):
            host_manifest = collector._host_identity_block()

        self.assertIn("probe_execution_records", host_manifest)
        self.assertTrue(host_manifest["probe_limitations"])
        self.assertTrue(all(isinstance(item, dict) for item in host_manifest["probe_limitations"]))

        by_probe = {item["probe"]: item for item in host_manifest["probe_execution_records"]}
        self.assertEqual(by_probe["windows_get_ciminstance_win32_operatingsystem"]["status"], "command_failed")
        self.assertEqual(by_probe["windows_get_ciminstance_win32_operatingsystem"]["returncode"], 5)
        self.assertIn("Access is denied", by_probe["windows_get_ciminstance_win32_operatingsystem"]["stderr"])
        self.assertEqual(by_probe["linux_lspci_nn"]["status"], "not_applicable")

    def test_override_focused_regeneration_keeps_no_go_verdict(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = self._write_temp_config(
                root,
                mmio_path="/tmp/from_config_mmio",
                dma_path="/tmp/from_config_dma",
                hist_bins=8,
                dma_buffer_bytes=256,
            )
            output_dir = root / "out"
            with patch.object(collector, "_host_identity_block", return_value=self._stub_host_identity()):
                collected = collector.collect_real_board_gate_artifacts(
                    output_dir=output_dir,
                    config_path=config_path,
                    mmio_path_override="/tmp/from_override_mmio",
                    dma_path_override="/tmp/from_override_dma",
                )

            gate = build_real_board_smoke_gate(
                GateInputs(
                    host_fact_manifest_json=collected["host_fact_manifest_json"],
                    device_path_probe_json=collected["device_path_probe_json"],
                    code_side_audit_json=collected["code_side_audit_json"],
                )
            )

        baseline_gate = build_real_board_smoke_gate(
            GateInputs(
                host_fact_manifest_json=T49_ARTIFACT_ROOT / "host_fact_manifest.json",
                device_path_probe_json=T49_ARTIFACT_ROOT / "device_path_probe.json",
                code_side_audit_json=T49_ARTIFACT_ROOT / "code_side_audit.json",
            )
        )
        self.assertEqual(gate["final_gate_verdict"], baseline_gate["final_gate_verdict"])
        self.assertEqual(gate["final_gate_verdict"], "NO_GO_REAL_BOARD_HOST_OR_DEVICE_PATH_UNAVAILABLE")


if __name__ == "__main__":
    unittest.main()
