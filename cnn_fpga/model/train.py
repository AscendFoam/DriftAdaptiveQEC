"""Train the selected regression model from histogram dataset splits."""

# 中文说明：
# - 该入口同时支持线性基线和 Tiny-CNN 主模型。
# - 这样可以保留 baseline 对照，同时把真正的 CNN 主路径纳入统一训练流程。

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from cnn_fpga.model.tiny_cnn import TinyCNNConfig, fit_tiny_cnn
from cnn_fpga.utils.config import config_hash, ensure_dir, get_path, load_yaml_config, now_tag, save_json


def _load_split(dataset_dir: Path, split: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = dataset_dir / f"{split}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Dataset split not found: {path}")
    data = np.load(path)
    return data["histograms"], data["labels"], data["label_names"]


def _load_dataset_manifest(dataset_dir: Path) -> Dict[str, object]:
    manifest_path = dataset_dir / "manifest.json"
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _fit_linear_regression(x: np.ndarray, y: np.ndarray, l2_reg: float):
    x_mean = x.mean(axis=0, keepdims=True)
    x_std = x.std(axis=0, keepdims=True)
    x_std = np.where(x_std < 1.0e-8, 1.0, x_std)
    x_norm = (x - x_mean) / x_std

    y_mean = y.mean(axis=0, keepdims=True)
    y_center = y - y_mean

    n_features = x_norm.shape[1]
    gram = x_norm.T @ x_norm + l2_reg * np.eye(n_features, dtype=np.float64)
    target = x_norm.T @ y_center
    weights = np.linalg.solve(gram, target)
    bias = y_mean.reshape(-1)
    return weights, bias, x_mean.reshape(-1), x_std.reshape(-1)


def _predict_linear(x: np.ndarray, weights: np.ndarray, bias: np.ndarray, x_mean: np.ndarray, x_std: np.ndarray):
    x_norm = (x - x_mean) / x_std
    return x_norm @ weights + bias


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "mse": float(np.mean((y_true - y_pred) ** 2)),
        "mae": float(np.mean(np.abs(y_true - y_pred))),
    }


def _arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train regression model from histogram dataset.")
    parser.add_argument("--config", required=True, type=str, help="Path to YAML config")
    parser.add_argument("--train-split", default="train", type=str)
    parser.add_argument("--val-split", default="val", type=str)
    return parser


def main() -> int:
    args = _arg_parser().parse_args()
    config = load_yaml_config(args.config)
    cfg_hash = config_hash(config)

    dataset_dir = get_path(config, "dataset_dir", "artifacts/datasets/default")
    model_dir = ensure_dir(get_path(config, "model_dir", "artifacts/models/default"))
    report_dir = ensure_dir(get_path(config, "report_dir", "artifacts/reports/default"))
    dataset_manifest = _load_dataset_manifest(dataset_dir)

    x_train_raw, y_train, label_names = _load_split(dataset_dir, args.train_split)
    x_val_raw, y_val, _ = _load_split(dataset_dir, args.val_split)

    train_cfg = config.get("training", {})
    model_type = str(train_cfg.get("model_type", "linear_regression_baseline")).lower()
    max_train_samples = int(train_cfg.get("max_train_samples", x_train_raw.shape[0]))
    max_val_samples = int(train_cfg.get("max_val_samples", x_val_raw.shape[0]))

    x_train_raw = x_train_raw[:max_train_samples]
    y_train = y_train[:max_train_samples].astype(np.float64)
    x_val_raw = x_val_raw[:max_val_samples]
    y_val = y_val[:max_val_samples].astype(np.float64)

    if model_type == "linear_regression_baseline":
        x_train = x_train_raw.reshape(x_train_raw.shape[0], -1).astype(np.float64)
        x_val = x_val_raw.reshape(x_val_raw.shape[0], -1).astype(np.float64)
        l2_reg = float(train_cfg.get("l2_reg", 1.0e-3))
        weights, bias, x_mean, x_std = _fit_linear_regression(x_train, y_train, l2_reg=l2_reg)
        pred_train = _predict_linear(x_train, weights, bias, x_mean, x_std)
        pred_val = _predict_linear(x_val, weights, bias, x_mean, x_std)
        train_metrics = _metrics(y_train, pred_train)
        val_metrics = _metrics(y_val, pred_val)
        run_name = f"linear_baseline_{now_tag()}_{cfg_hash}"
        model_path = model_dir / f"{run_name}.npz"
        np.savez_compressed(
            model_path,
            model_type=np.array(["linear_regression_baseline"]),
            weights=weights.astype(np.float32),
            bias=bias.astype(np.float32),
            x_mean=x_mean.astype(np.float32),
            x_std=x_std.astype(np.float32),
            label_names=label_names,
            config_hash=np.array([cfg_hash]),
        )
        report = {
            "run_name": run_name,
            "config_hash": cfg_hash,
            "dataset_dir": str(dataset_dir),
            "model_path": str(model_path),
            "model_type": model_type,
            "train_split": args.train_split,
            "val_split": args.val_split,
            "n_train": int(x_train.shape[0]),
            "n_val": int(x_val.shape[0]),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "l2_reg": l2_reg,
        }
    elif model_type == "tiny_cnn":
        cnn_cfg = TinyCNNConfig.from_config(config)
        artifact, extra_report = fit_tiny_cnn(
            x_train_raw.astype(np.float64),
            y_train,
            x_val_raw.astype(np.float64),
            y_val,
            label_names,
            cnn_cfg,
        )
        run_name = f"tiny_cnn_{now_tag()}_{cfg_hash}"
        model_path = model_dir / f"{run_name}.npz"
        artifact["config_hash"] = np.array([cfg_hash])
        if dataset_manifest:
            if "label_semantics" in dataset_manifest:
                artifact["target_semantics"] = np.array([str(dataset_manifest["label_semantics"])])
            if "teacher_mode" in dataset_manifest:
                artifact["teacher_mode"] = np.array([str(dataset_manifest["teacher_mode"])])
            if "context_windows" in dataset_manifest:
                artifact["context_windows"] = np.array([int(dataset_manifest["context_windows"])], dtype=np.int32)
            if "input_channel_names" in dataset_manifest:
                artifact["input_channel_names"] = np.asarray(dataset_manifest["input_channel_names"])
            artifact["context_strategy"] = np.array(
                [str(config.get("runtime_dataset", {}).get("context_padding", "edge_repeat"))]
            )
            artifact["residual_apply"] = np.array(
                [str(config.get("runtime_dataset", {}).get("label_semantics", "absolute"))]
            )
        np.savez_compressed(model_path, **artifact)
        train_metrics = extra_report["train_metrics"]
        val_metrics = extra_report["val_metrics"]
        report = {
            "run_name": run_name,
            "config_hash": cfg_hash,
            "dataset_dir": str(dataset_dir),
            "dataset_manifest": dataset_manifest,
            "model_path": str(model_path),
            "model_type": model_type,
            "train_split": args.train_split,
            "val_split": args.val_split,
            "n_train": int(x_train_raw.shape[0]),
            "n_val": int(x_val_raw.shape[0]),
            "training_backend": extra_report.get("backend", cnn_cfg.backend),
            "training_device": extra_report.get("device", cnn_cfg.device),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "tiny_cnn": {
                "conv_channels": cnn_cfg.conv_channels,
                "kernel_size": cnn_cfg.kernel_size,
                "hidden_dim": cnn_cfg.hidden_dim,
                "batch_size": cnn_cfg.batch_size,
                "epochs": cnn_cfg.epochs,
                "learning_rate": cnn_cfg.learning_rate,
                "weight_decay": cnn_cfg.weight_decay,
                "patience": cnn_cfg.patience,
                "backend": cnn_cfg.backend,
                "device": cnn_cfg.device,
                "label_weights": dict(cnn_cfg.label_weights or {}),
            },
            "history": extra_report["history"],
        }
    else:
        raise ValueError(f"Unsupported training model_type: {model_type}")

    save_json(report_dir / f"{run_name}_train_report.json", report)
    print("Training complete.")
    print(f"Model: {model_path}")
    print(f"Train MSE: {train_metrics['mse']:.6f}, Val MSE: {val_metrics['mse']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
