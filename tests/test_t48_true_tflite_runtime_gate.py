import json
import tempfile
import unittest
from pathlib import Path

from cnn_fpga.model.build_t48_true_tflite_runtime_gate import (
    GateInputs,
    build_true_tflite_runtime_gate,
)


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


class TrueTFLiteRuntimeGateTests(unittest.TestCase):
    def _write_env_probe(self, root: Path, *, runtime_available: bool) -> Path:
        imports = {
            "tflite_runtime": {"ok": False, "error": "ModuleNotFoundError: No module named 'tflite_runtime'"},
            "tensorflow": {"ok": False, "error": "ModuleNotFoundError: No module named 'tensorflow'"},
        }
        if runtime_available:
            imports["tensorflow"] = {
                "ok": True,
                "version": "2.13.0",
                "file": "C:/ProgramData/anaconda3/envs/LPNEnv/Lib/site-packages/tensorflow/__init__.py",
            }
        return _write_json(
            root / "runtime_env_probe.json",
            {
                "python_executable": "C:/ProgramData/anaconda3/envs/LPNEnv/python.exe",
                "python_version": "3.8.20",
                "platform": "Windows-11",
                "imports": imports,
            },
        )

    def _write_eval_report(self, path: Path, *, tflite_path: Path) -> Path:
        return _write_json(
            path,
            {
                "run_name": "eval_tflite_test_20260610_120000",
                "tflite_path": str(tflite_path.resolve()),
                "split": "test",
                "n_samples": 1639,
                "metrics": {
                    "mse": 0.29,
                    "mae": 0.22,
                    "r2_mean": 0.99,
                },
            },
        )

    def _write_validate_report(self, path: Path, *, artifact_path: Path, tflite_path: Path) -> Path:
        return _write_json(
            path,
            {
                "artifact_path": str(artifact_path.resolve()),
                "tflite_path": str(tflite_path.resolve()),
                "split": "test",
                "n_samples": 128,
                "max_abs_diff": 0.12,
                "mean_abs_diff": 0.008,
                "status": "ok",
            },
        )

    def _write_pack(
        self,
        root: Path,
        *,
        float_artifact: Path,
        int8_artifact: Path | None,
        float_tflite: Path,
        int8_tflite: Path | None,
        float_stub: Path | None = None,
        int8_stub: Path | None = None,
    ) -> Path:
        tflite_paths = [str(float_tflite.resolve())]
        if int8_tflite is not None:
            tflite_paths.append(str(int8_tflite.resolve()))
        stub_paths: list[str] = []
        if float_stub is not None:
            stub_paths.append(str(float_stub.resolve()))
        if int8_stub is not None:
            stub_paths.append(str(int8_stub.resolve()))
        int8_paths = [] if int8_artifact is None else [str(int8_artifact.resolve())]
        return _write_json(
            root / "training_reproducibility_pack.json",
            {
                "task_id": "T50",
                "canonical_materials": {
                    "static_theta_v2": {
                        "float_model_artifact": {"path": str(float_artifact.resolve())},
                        "derived_materials_presence": {
                            "int8_model_artifacts": {"count": len(int8_paths), "paths": int8_paths},
                            "tflite_model_artifacts": {"count": len(tflite_paths), "paths": tflite_paths},
                            "tflite_json_sidecars": {"count": len(stub_paths), "paths": stub_paths},
                        },
                    }
                },
            },
        )

    def _write_compatibility_probe(self, root: Path, *, results: list[dict]) -> Path:
        return _write_json(
            root / "preserved_tflite_load_probe.json",
            {
                "probe_kind": "preserved_tflite_load_probe",
                "python_executable": "C:/ProgramData/anaconda3/envs/LPNEnv/python.exe",
                "results": results,
            },
        )

    def test_rejects_tflite_json_stub_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            float_artifact = root / "float_model.npz"
            int8_artifact = root / "int8_model.npz"
            float_tflite = root / "float_model.tflite"
            int8_stub = root / "int8_model.tflite.json"
            for path in (float_artifact, int8_artifact, float_tflite, int8_stub):
                path.write_bytes(b"placeholder")
            pack = self._write_pack(
                root,
                float_artifact=float_artifact,
                int8_artifact=int8_artifact,
                float_tflite=float_tflite,
                int8_tflite=None,
                int8_stub=int8_stub,
            )
            env_probe = self._write_env_probe(root, runtime_available=True)
            float_eval = self._write_eval_report(root / "float_eval.json", tflite_path=float_tflite)
            float_validate = self._write_validate_report(
                root / "float_validate.json",
                artifact_path=float_artifact,
                tflite_path=float_tflite,
            )

            with self.assertRaisesRegex(ValueError, "stub"):
                build_true_tflite_runtime_gate(
                    GateInputs(
                        t50_pack_json=pack,
                        env_probe_json=env_probe,
                        float_eval_report=float_eval,
                        float_validate_report=float_validate,
                        int8_tflite_path=int8_stub,
                    )
                )

    def test_float_success_without_int8_reports_yields_float_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            float_artifact = root / "float_model.npz"
            float_tflite = root / "float_model.tflite"
            for path in (float_artifact, float_tflite):
                path.write_bytes(b"placeholder")
            pack = self._write_pack(
                root,
                float_artifact=float_artifact,
                int8_artifact=None,
                float_tflite=float_tflite,
                int8_tflite=None,
            )
            env_probe = self._write_env_probe(root, runtime_available=True)
            float_eval = self._write_eval_report(root / "float_eval.json", tflite_path=float_tflite)
            float_validate = self._write_validate_report(
                root / "float_validate.json",
                artifact_path=float_artifact,
                tflite_path=float_tflite,
            )

            gate = build_true_tflite_runtime_gate(
                GateInputs(
                    t50_pack_json=pack,
                    env_probe_json=env_probe,
                    float_eval_report=float_eval,
                    float_validate_report=float_validate,
                )
            )

        self.assertEqual(gate["final_gate_verdict"], "GO_TRUE_TFLITE_RUNTIME_FLOAT_ONLY")
        self.assertTrue(gate["environment_truth"]["runtime_available"])
        self.assertTrue(gate["float_runtime_result"]["executed"])
        self.assertFalse(gate["int8_runtime_result"]["executed"])

    def test_runtime_unavailable_yields_no_go(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            float_artifact = root / "float_model.npz"
            float_tflite = root / "float_model.tflite"
            for path in (float_artifact, float_tflite):
                path.write_bytes(b"placeholder")
            pack = self._write_pack(
                root,
                float_artifact=float_artifact,
                int8_artifact=None,
                float_tflite=float_tflite,
                int8_tflite=None,
            )
            env_probe = self._write_env_probe(root, runtime_available=False)

            gate = build_true_tflite_runtime_gate(
                GateInputs(
                    t50_pack_json=pack,
                    env_probe_json=env_probe,
                )
            )

        self.assertEqual(gate["final_gate_verdict"], "NO_GO_TRUE_TFLITE_RUNTIME_UNAVAILABLE")
        self.assertFalse(gate["environment_truth"]["runtime_available"])
        self.assertFalse(gate["float_runtime_result"]["executed"])

    def test_rejects_missing_key_float_reports_when_runtime_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            float_artifact = root / "float_model.npz"
            float_tflite = root / "float_model.tflite"
            for path in (float_artifact, float_tflite):
                path.write_bytes(b"placeholder")
            pack = self._write_pack(
                root,
                float_artifact=float_artifact,
                int8_artifact=None,
                float_tflite=float_tflite,
                int8_tflite=None,
            )
            env_probe = self._write_env_probe(root, runtime_available=True)

            with self.assertRaisesRegex(ValueError, "float.*report"):
                build_true_tflite_runtime_gate(
                    GateInputs(
                        t50_pack_json=pack,
                        env_probe_json=env_probe,
                    )
                )

    def test_incompatible_preserved_true_tflite_yields_invalid_no_go_with_probe(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            float_artifact = root / "float_model.npz"
            float_tflite = root / "float_model.tflite"
            for path in (float_artifact, float_tflite):
                path.write_bytes(b"placeholder")
            pack = self._write_pack(
                root,
                float_artifact=float_artifact,
                int8_artifact=None,
                float_tflite=float_tflite,
                int8_tflite=None,
            )
            env_probe = self._write_env_probe(root, runtime_available=True)
            compatibility_probe = self._write_compatibility_probe(
                root,
                results=[
                    {
                        "path": str(float_tflite.resolve()),
                        "ok": False,
                        "error": "ValueError: Didn't find op for builtin opcode 'FULLY_CONNECTED' version '12'.",
                    }
                ],
            )

            gate = build_true_tflite_runtime_gate(
                GateInputs(
                    t50_pack_json=pack,
                    env_probe_json=env_probe,
                    compatibility_probe_json=compatibility_probe,
                )
            )

        self.assertEqual(gate["final_gate_verdict"], "NO_GO_PRESERVED_TFLITE_ARTIFACT_INVALID_OR_STUB_ONLY")
        self.assertFalse(gate["float_runtime_result"]["executed"])
        self.assertIn("FULLY_CONNECTED", gate["float_runtime_result"]["compatibility_probe"]["error"])


if __name__ == "__main__":
    unittest.main()
