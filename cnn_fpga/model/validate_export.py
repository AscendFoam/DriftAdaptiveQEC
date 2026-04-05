"""Validate that exported TFLite predictions stay close to source artifacts."""

# 中文说明：
# - 该脚本对比“源 artifact 预测”和“导出 `.tflite` 预测”，检查导出是否改坏了语义。
# - 重点用于发现归一化、通道顺序、量化 scale 等部署阶段常见问题。

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Sequence

import numpy as np

from cnn_fpga.model.tiny_cnn import predict_from_artifact
from cnn_fpga.runtime.inference_service import TFLiteHistogramPredictor
from cnn_fpga.utils.config import ensure_dir, get_path, load_yaml_config, now_tag, save_json


def _arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate exported TFLite against source artifact.")
    parser.add_argument("--config", required=True, type=str, help="Path to YAML config")
    parser.add_argument("--artifact-path", required=True, type=str, help="Source artifact path (.npz)")
    parser.add_argument("--tflite-path", required=True, type=str, help="Exported TFLite path")
    parser.add_argument("--split", default=None, type=str, help="Dataset split used for validation")
    parser.add_argument("--max-samples", default=128, type=int, help="Maximum number of samples to compare")
    parser.add_argument("--warn-max-abs-diff", default=10.0, type=float, help="Warning threshold for max abs diff")
    return parser


def _load_split(dataset_dir: Path, split: str):
    """加载验证样本，默认复用 test split 的前若干条数据。"""
    path = dataset_dir / f"{split}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Dataset split not found: {path}")
    data = np.load(path)
    return (
        data["histograms"].astype(np.float32),
        data["labels"].astype(np.float64),
        [str(item) for item in data["label_names"]],
    )


def _compare_predictions(
    artifact_path: Path,
    tflite_path: Path,
    histograms: np.ndarray,
    label_names: Sequence[str],
) -> Dict:
    """逐样本比较 artifact 与 TFLite 输出，返回全局与逐标签偏差。"""
    def _extract_prediction_value(prediction, name: str) -> float:
        if hasattr(prediction, name):
            return float(getattr(prediction, name))
        metadata = dict(getattr(prediction, "metadata", {}) or {})
        raw_prediction = dict(metadata.get("raw_prediction", {}))
        return float(raw_prediction.get(name, 0.0))

    predictor = TFLiteHistogramPredictor(tflite_path, label_names)
    artifact_pred = np.asarray(predict_from_artifact(artifact_path, histograms), dtype=np.float64)
    tflite_pred = np.asarray(
        [[_extract_prediction_value(pred, name) for name in label_names] for pred in (predictor.predict(hist) for hist in histograms)],
        dtype=np.float64,
    )
    abs_diff = np.abs(artifact_pred - tflite_pred)
    return {
        "artifact_pred_shape": list(artifact_pred.shape),
        "tflite_pred_shape": list(tflite_pred.shape),
        "max_abs_diff": float(np.max(abs_diff)),
        "mean_abs_diff": float(np.mean(abs_diff)),
        "per_label": {
            str(name): {
                "max_abs_diff": float(np.max(abs_diff[:, idx])),
                "mean_abs_diff": float(np.mean(abs_diff[:, idx])),
            }
            for idx, name in enumerate(label_names)
        },
    }


def main() -> int:
    args = _arg_parser().parse_args()
    config = load_yaml_config(args.config)
    dataset_dir = get_path(config, "dataset_dir", "artifacts/datasets/default")
    report_dir = ensure_dir(get_path(config, "report_dir", "artifacts/reports/default"))

    split = args.split or str(config.get("evaluation", {}).get("target_split", "test"))
    x_eval, _, label_names = _load_split(dataset_dir, split)
    x_eval = x_eval[: int(args.max_samples)]
    label_names = tuple(label_names)

    artifact_path = Path(args.artifact_path).expanduser().resolve()
    tflite_path = Path(args.tflite_path).expanduser().resolve()
    comparison = _compare_predictions(artifact_path, tflite_path, x_eval, label_names)
    comparison["artifact_path"] = str(artifact_path)
    comparison["tflite_path"] = str(tflite_path)
    comparison["split"] = split
    comparison["n_samples"] = int(x_eval.shape[0])
    comparison["warn_max_abs_diff"] = float(args.warn_max_abs_diff)
    comparison["status"] = "warn" if comparison["max_abs_diff"] > float(args.warn_max_abs_diff) else "ok"

    report_path = report_dir / f"validate_export_{artifact_path.stem}_{now_tag()}.json"
    save_json(report_path, comparison)

    print("Export validation complete.")
    print(f"Artifact: {artifact_path}")
    print(f"TFLite: {tflite_path}")
    print(f"Split: {split}, Samples: {x_eval.shape[0]}")
    print(f"max_abs_diff: {comparison['max_abs_diff']:.6f}, mean_abs_diff: {comparison['mean_abs_diff']:.6f}")
    print(f"Status: {comparison['status']}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
