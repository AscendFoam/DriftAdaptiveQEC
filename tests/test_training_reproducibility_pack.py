import json
import tempfile
import unittest
from pathlib import Path

from cnn_fpga.model.build_training_reproducibility_pack import (
    PackInputs,
    build_training_reproducibility_pack,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
STATIC_DATASET_DIR = REPO_ROOT / "artifacts" / "datasets" / "static_theta_v2"
STATIC_MODEL_PATH = (
    REPO_ROOT
    / "artifacts"
    / "models"
    / "static_theta_v2"
    / "tiny_cnn_20260319_151717_b87c6c227b57.npz"
)
P4_RECOVERY_SMOKE_CONFIG = (
    REPO_ROOT / "cnn_fpga" / "config" / "p4_multiscenario_recovery_smoke.yaml"
)
P4_MULTISCENARIO_SMOKE_CONFIG = REPO_ROOT / "cnn_fpga" / "config" / "p4_multiscenario_smoke.yaml"


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


class TrainingReproducibilityPackTests(unittest.TestCase):
    def _write_rerun_train_report(self, root: Path, model_path: Path) -> Path:
        return _write_json(
            root / "tiny_cnn_20260610_120000_t50test_train_report.json",
            {
                "run_name": "tiny_cnn_20260610_120000_t50test",
                "config_hash": "t50test000001",
                "dataset_dir": str(STATIC_DATASET_DIR.resolve()),
                "model_path": str(model_path.resolve()),
                "model_type": "tiny_cnn",
                "train_split": "train",
                "val_split": "val",
                "n_train": 2048,
                "n_val": 512,
                "training_backend": "numpy",
                "training_device": "cpu",
                "train_metrics": {
                    "mse": 0.41,
                    "mae": 0.18,
                    "r2_mean": 0.91,
                },
                "val_metrics": {
                    "mse": 0.49,
                    "mae": 0.20,
                    "r2_mean": 0.89,
                },
                "tiny_cnn": {
                    "epochs": 5,
                    "patience": 3,
                    "backend": "numpy",
                    "device": "auto",
                },
                "history": [
                    {
                        "epoch": 1,
                        "batch_loss_mean": 0.9,
                        "train_loss_norm": 0.5,
                        "val_loss_norm": 0.6,
                    }
                ],
            },
        )

    def _write_rerun_eval_report(self, root: Path, model_path: Path) -> Path:
        return _write_json(
            root / "eval_test_20260610_120010.json",
            {
                "run_name": "eval_test_20260610_120010",
                "model_path": str(model_path.resolve()),
                "model_type": "tiny_cnn_float",
                "split": "test",
                "n_samples": 1639,
                "metrics": {
                    "mse": 0.53,
                    "mae": 0.21,
                    "r2_mean": 0.88,
                },
            },
        )

    def test_current_artifacts_build_expected_pack(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "artifacts" / "t50_training_repro_pack" / "models" / "static_theta_v2"
            report_dir = root / "artifacts" / "t50_training_repro_pack" / "reports" / "static_theta_v2"
            model_dir.mkdir(parents=True, exist_ok=True)
            report_dir.mkdir(parents=True, exist_ok=True)
            rerun_model_path = model_dir / "tiny_cnn_20260610_120000_t50test.npz"
            rerun_model_path.write_bytes(b"placeholder")
            rerun_train_report = self._write_rerun_train_report(report_dir, rerun_model_path)
            rerun_eval_report = self._write_rerun_eval_report(report_dir, rerun_model_path)

            pack = build_training_reproducibility_pack(
                PackInputs(
                    rerun_train_report=rerun_train_report,
                    rerun_eval_report=rerun_eval_report,
                    t50_output_root=root / "artifacts" / "t50_training_repro_pack",
                )
            )

        self.assertEqual(pack["task_id"], "T50")
        self.assertTrue(pack["canonical_materials"]["static_theta_v2"]["chain_complete"])
        self.assertTrue(pack["canonical_materials"]["runtime_b_residual_v1"]["chain_complete"])
        self.assertGreater(
            pack["canonical_materials"]["static_theta_v2"]["derived_materials_presence"]["int8_model_artifacts"]["count"],
            0,
        )
        self.assertEqual(
            pack["mainline_preserved_model_references"]["p4_recovery_smoke"]["explicit_model_path"]["path"],
            str(STATIC_MODEL_PATH.resolve()),
        )
        self.assertTrue(
            pack["bounded_rerun_materials"]["canonical_vs_rerun_relation"]["dataset_dir_matches_static_theta_v2"]
        )
        self.assertTrue(
            pack["bounded_rerun_materials"]["canonical_vs_rerun_relation"]["rerun_model_is_isolated_under_t50_root"]
        )
        self.assertEqual(pack["bounded_rerun_materials"]["train_report"]["training_backend"], "numpy")
        self.assertEqual(pack["bounded_rerun_materials"]["train_report"]["training_device"], "cpu")
        self.assertTrue(
            any("full training reproducibility" in item for item in pack["unsupported_claims"])
        )

    def test_rejects_missing_canonical_static_float_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "artifacts" / "t50_training_repro_pack" / "models" / "static_theta_v2"
            report_dir = root / "artifacts" / "t50_training_repro_pack" / "reports" / "static_theta_v2"
            model_dir.mkdir(parents=True, exist_ok=True)
            report_dir.mkdir(parents=True, exist_ok=True)
            rerun_model_path = model_dir / "tiny_cnn_20260610_120000_t50test.npz"
            rerun_model_path.write_bytes(b"placeholder")
            rerun_train_report = self._write_rerun_train_report(report_dir, rerun_model_path)
            rerun_eval_report = self._write_rerun_eval_report(report_dir, rerun_model_path)

            with self.assertRaisesRegex(ValueError, "static_theta_v2 float model artifact"):
                build_training_reproducibility_pack(
                    PackInputs(
                        canonical_static_model=STATIC_MODEL_PATH.with_name("missing_static_theta_model.npz"),
                        rerun_train_report=rerun_train_report,
                        rerun_eval_report=rerun_eval_report,
                        t50_output_root=root / "artifacts" / "t50_training_repro_pack",
                    )
                )

    def test_rejects_mainline_config_reference_drift_to_missing_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "artifacts" / "t50_training_repro_pack" / "models" / "static_theta_v2"
            report_dir = root / "artifacts" / "t50_training_repro_pack" / "reports" / "static_theta_v2"
            model_dir.mkdir(parents=True, exist_ok=True)
            report_dir.mkdir(parents=True, exist_ok=True)
            rerun_model_path = model_dir / "tiny_cnn_20260610_120000_t50test.npz"
            rerun_model_path.write_bytes(b"placeholder")
            rerun_train_report = self._write_rerun_train_report(report_dir, rerun_model_path)
            rerun_eval_report = self._write_rerun_eval_report(report_dir, rerun_model_path)

            mutated_config = root / "p4_multiscenario_recovery_smoke_missing.yaml"
            mutated_text = P4_RECOVERY_SMOKE_CONFIG.read_text(encoding="utf-8")
            mutated_text = mutated_text.replace(
                "base_config: p4_multiscenario_smoke.yaml",
                f"base_config: {P4_MULTISCENARIO_SMOKE_CONFIG.resolve().as_posix()}",
            )
            mutated_text = mutated_text.replace(
                "artifacts/models/static_theta_v2/tiny_cnn_20260319_151717_b87c6c227b57.npz",
                "artifacts/models/static_theta_v2/missing_t50_reference_model.npz",
            )
            mutated_config.write_text(mutated_text, encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "p4_multiscenario_recovery_smoke"):
                build_training_reproducibility_pack(
                    PackInputs(
                        rerun_train_report=rerun_train_report,
                        rerun_eval_report=rerun_eval_report,
                        p4_recovery_smoke_config=mutated_config,
                        t50_output_root=root / "artifacts" / "t50_training_repro_pack",
                    )
                )


if __name__ == "__main__":
    unittest.main()
