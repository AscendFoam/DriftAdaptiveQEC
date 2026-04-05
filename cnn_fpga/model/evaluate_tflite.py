"""Evaluate a real TFLite artifact against a dataset split."""

# 中文说明：
# - 该脚本用于在独立 TensorFlow / TFLite 环境中，直接评估导出的 `.tflite` 模型。
# - 目标不是验证训练脚本本身，而是验证“部署产物”在真实解释器下的回归精度。

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Sequence

import numpy as np

from cnn_fpga.runtime.inference_service import TFLiteHistogramPredictor
from cnn_fpga.utils.config import ensure_dir, get_path, load_yaml_config, now_tag, save_json


def _arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate TFLite artifact on a dataset split.")
    parser.add_argument("--config", required=True, type=str, help="Path to YAML config")
    parser.add_argument("--split", default=None, type=str, help="Dataset split to evaluate")
    parser.add_argument("--tflite-path", default=None, type=str, help="TFLite path override")
    return parser


def _find_latest_tflite(model_dir: Path) -> Path:
    """从模型目录中选择最新的 `.tflite` 产物。"""
    candidates = sorted(model_dir.glob("*.tflite"))
    if not candidates:
        raise FileNotFoundError(f"No .tflite files in {model_dir}")
    return candidates[-1]


def _load_split(dataset_dir: Path, split: str):
    """加载指定数据集 split，保持与训练/评估脚本一致的输入输出格式。"""
    path = dataset_dir / f"{split}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Dataset split not found: {path}")
    data = np.load(path)
    return (
        data["histograms"].astype(np.float32),
        data["labels"].astype(np.float64),
        [str(item) for item in data["label_names"]],
    )


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """计算单标签 R²，用于逐标签验收。"""
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot < 1.0e-12:
        return 0.0
    return 1.0 - ss_res / ss_tot


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, label_names: Sequence[str]) -> Dict:
    """汇总全局与逐标签误差指标，便于与 float/int8 artifact 直接对照。"""
    return {
        "mse": float(np.mean((y_true - y_pred) ** 2)),
        "mae": float(np.mean(np.abs(y_true - y_pred))),
        "r2_mean": float(np.mean([_r2_score(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])])),
        "per_label": {
            str(name): {
                "mse": float(np.mean((y_true[:, i] - y_pred[:, i]) ** 2)),
                "mae": float(np.mean(np.abs(y_true[:, i] - y_pred[:, i]))),
                "r2": float(_r2_score(y_true[:, i], y_pred[:, i])),
            }
            for i, name in enumerate(label_names)
        },
    }


def _extract_prediction_value(prediction, name: str) -> float:
    """Read one label from a predictor output, with fallback to raw_prediction metadata."""
    if hasattr(prediction, name):
        return float(getattr(prediction, name))
    metadata = dict(getattr(prediction, "metadata", {}) or {})
    raw_prediction = dict(metadata.get("raw_prediction", {}))
    return float(raw_prediction.get(name, 0.0))


def main() -> int:
    args = _arg_parser().parse_args()
    config = load_yaml_config(args.config)

    dataset_dir = get_path(config, "dataset_dir", "artifacts/datasets/default")
    model_dir = ensure_dir(get_path(config, "model_dir", "artifacts/models/default"))
    report_dir = ensure_dir(get_path(config, "report_dir", "artifacts/reports/default"))

    split = args.split or str(config.get("evaluation", {}).get("target_split", "test"))
    tflite_path = Path(args.tflite_path).expanduser().resolve() if args.tflite_path else _find_latest_tflite(model_dir)
    x_eval, y_eval, label_names = _load_split(dataset_dir, split)
    label_names = tuple(label_names)
    # 这里显式走 TFLite 解释器路径，而不是训练期的 NumPy artifact 推理。
    predictor = TFLiteHistogramPredictor(tflite_path, label_names)
    predictions = np.asarray(
        [[_extract_prediction_value(pred, name) for name in label_names] for pred in (predictor.predict(hist) for hist in x_eval)],
        dtype=np.float64,
    )
    metrics = _metrics(y_eval, predictions, label_names)

    run_name = f"eval_tflite_{split}_{now_tag()}"
    report = {
        "run_name": run_name,
        "tflite_path": str(tflite_path),
        "split": split,
        "n_samples": int(x_eval.shape[0]),
        "metrics": metrics,
    }
    report_path = report_dir / f"{run_name}.json"
    save_json(report_path, report)

    print("TFLite evaluation complete.")
    print(f"TFLite: {tflite_path}")
    print(f"Split: {split}, Samples: {x_eval.shape[0]}")
    print(f"MSE: {metrics['mse']:.6f}, MAE: {metrics['mae']:.6f}, R2_mean: {metrics['r2_mean']:.6f}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
