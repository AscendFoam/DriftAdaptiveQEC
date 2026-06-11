import json
import tempfile
import unittest
from pathlib import Path

from cnn_fpga.hwio.build_t49_real_board_smoke_gate import GateInputs, build_real_board_smoke_gate
from cnn_fpga.hwio.collect_t71_real_board_gate_artifacts import collect_real_board_gate_artifacts


REPO_ROOT = Path(__file__).resolve().parents[1]
T49_ARTIFACT_ROOT = REPO_ROOT / "artifacts" / "t49_real_board_smoke_execution_gate"


class RealBoardGateRegenerationPackTests(unittest.TestCase):
    def test_t49_checked_in_artifact_replay_keeps_current_host_no_go(self) -> None:
        gate = build_real_board_smoke_gate(
            GateInputs(
                host_fact_manifest_json=T49_ARTIFACT_ROOT / "host_fact_manifest.json",
                device_path_probe_json=T49_ARTIFACT_ROOT / "device_path_probe.json",
                code_side_audit_json=T49_ARTIFACT_ROOT / "code_side_audit.json",
            )
        )

        self.assertEqual(gate["final_gate_verdict"], "NO_GO_REAL_BOARD_HOST_OR_DEVICE_PATH_UNAVAILABLE")
        self.assertEqual(gate["current_strongest_supported_statement"], json.loads(
            (T49_ARTIFACT_ROOT / "t49_real_board_smoke_execution_gate.json").read_text(encoding="utf-8")
        )["current_strongest_supported_statement"])

    def test_current_host_regeneration_matches_t49_verdict(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            collected = collect_real_board_gate_artifacts(output_dir=output_dir)
            gate = build_real_board_smoke_gate(
                GateInputs(
                    host_fact_manifest_json=collected["host_fact_manifest_json"],
                    device_path_probe_json=collected["device_path_probe_json"],
                    code_side_audit_json=collected["code_side_audit_json"],
                )
            )
            self.assertTrue(collected["host_fact_manifest_json"].is_file())
            self.assertTrue(collected["device_path_probe_json"].is_file())
            self.assertTrue(collected["code_side_audit_json"].is_file())

        baseline_gate = json.loads((T49_ARTIFACT_ROOT / "t49_real_board_smoke_execution_gate.json").read_text(encoding="utf-8"))
        self.assertEqual(gate["final_gate_verdict"], baseline_gate["final_gate_verdict"])
        self.assertEqual(
            gate["current_strongest_supported_statement"],
            baseline_gate["current_strongest_supported_statement"],
        )
        self.assertEqual(gate["device_path_truth"]["status"], baseline_gate["device_path_truth"]["status"])


if __name__ == "__main__":
    unittest.main()
