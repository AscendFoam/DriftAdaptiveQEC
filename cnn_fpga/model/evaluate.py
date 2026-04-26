"""Evaluate saved regression artifacts against a dataset split."""

# 中文说明：
# - 评估入口同时支持线性基线和 Tiny-CNN。
# - 输出整体指标和逐标签指标，便于检查主模型是否已经达到替代 baseline 的水平。

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np

from cnn_fpga.model.tiny_cnn import predict_from_artifact
from cnn_fpga.utils.config import ensure_dir, get_path, load_yaml_config, now_tag, save_json


def _find_latest_model(model_dir: Path) -> Path:
    candidates = sorted(model_dir.glob("*.npz"))
    if not candidates:
        raise FileNotFoundError(f"No model files in {model_dir}")
    return candidates[-1]


def _load_split(dataset_dir: Path, split: str):
    path = dataset_dir / f"{split}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Dataset split not found: {path}")
    data = np.load(path)
    scalar_features = data["scalar_features"].astype(np.float64) if "scalar_features" in data else np.zeros((data["histograms"].shape[0], 0), dtype=np.float64)
    return data["histograms"].astype(np.float64), scalar_features, data["labels"].astype(np.float64)


def _predict_linear(x_hist: np.ndarray, model: Dict[str, np.ndarray]) -> np.ndarray:
    x = x_hist.reshape(x_hist.shape[0], -1).astype(np.float64)
    x_norm = (x - model["x_mean"]) / model["x_std"]
    return x_norm @ model["weights"] + model["bias"]


def _predict_artifact(model_path: Path, x_hist: np.ndarray, x_scalar: np.ndarray):
    data = np.load(model_path, allow_pickle=True)
    model_type = str(data["model_type"][0])
    label_names = [str(item) for item in data["label_names"]]
    if model_type.startswith("linear_regression"):
        if model_type == "linear_regression_baseline":
            y_pred = _predict_linear(
                x_hist,
                {
                    "weights": data["weights"].astype(np.float64),
                    "bias": data["bias"].astype(np.float64),
                    "x_mean": data["x_mean"].astype(np.float64),
                    "x_std": data["x_std"].astype(np.float64),
                },
            )
        else:
            weights = data["q_weights"].astype(np.float64) * float(data["w_scale"][0])
            bias = data["q_bias"].astype(np.float64) * float(data["b_scale"][0])
            y_pred = _predict_linear(
                x_hist,
                {
                    "weights": weights,
                    "bias": bias,
                    "x_mean": data["x_mean"].astype(np.float64),
                    "x_std": data["x_std"].astype(np.float64),
                },
            )
    elif model_type.startswith("tiny_cnn"):
        y_pred = np.asarray(predict_from_artifact(model_path, (x_hist, x_scalar)), dtype=np.float64)
    else:
        raise ValueError(f"Unsupported artifact type: {model_type}")
    return model_type, label_names, y_pred


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot < 1.0e-12:
        return 0.0
    return 1.0 - ss_res / ss_tot


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, label_names):
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


def _arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate saved regression artifact.")
    parser.add_argument("--config", required=True, type=str, help="Path to YAML config")
    parser.add_argument("--split", default=None, type=str, help="Dataset split to evaluate")
    parser.add_argument("--model-path", default=None, type=str, help="Model path override")
    return parser


def main() -> int:
    args = _arg_parser().parse_args()
    config = load_yaml_config(args.config)

    dataset_dir = get_path(config, "dataset_dir", "artifacts/datasets/default")
    model_dir = ensure_dir(get_path(config, "model_dir", "artifacts/models/default"))
    report_dir = ensure_dir(get_path(config, "report_dir", "artifacts/reports/default"))

    split = args.split or str(config.get("evaluation", {}).get("target_split", "test"))
    model_path = Path(args.model_path).expanduser().resolve() if args.model_path else _find_latest_model(model_dir)

    x_eval, x_eval_scalar, y_eval = _load_split(dataset_dir, split)
    model_type, label_names, y_pred = _predict_artifact(model_path, x_eval, x_eval_scalar)
    metrics = _metrics(y_eval, y_pred, label_names)

    run_name = f"eval_{split}_{now_tag()}"
    report = {
        "run_name": run_name,
        "model_path": str(model_path),
        "model_type": model_type,
        "split": split,
        "n_samples": int(x_eval.shape[0]),
        "metrics": metrics,
    }
    report_path = report_dir / f"{run_name}.json"
    save_json(report_path, report)

    print("Evaluation complete.")
    print(f"Model: {model_path}")
    print(f"Split: {split}, Samples: {x_eval.shape[0]}")
    print(f"MSE: {metrics['mse']:.6f}, MAE: {metrics['mae']:.6f}, R2_mean: {metrics['r2_mean']:.6f}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
