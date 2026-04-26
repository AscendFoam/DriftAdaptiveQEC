"""Quantize saved regression artifacts to int8 deployment format."""

# 中文说明：
# - 量化入口同时支持线性基线和 Tiny-CNN。
# - 当前采用逐张量对称量化，目标是先把“训练 -> 部署 -> runtime artifact 推理”链路完整跑通。

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from cnn_fpga.utils.config import ensure_dir, get_path, load_yaml_config, now_tag, save_json


def _find_latest_model(model_dir: Path) -> Path:
    candidates = sorted(model_dir.glob("*.npz"))
    if not candidates:
        raise FileNotFoundError(f"No model files in {model_dir}")
    return candidates[-1]


def _quantize_symmetric(values: np.ndarray) -> Tuple[np.ndarray, float]:
    max_abs = float(np.max(np.abs(values)))
    scale = 1.0 if max_abs < 1.0e-12 else max_abs / 127.0
    q = np.clip(np.round(values / scale), -127, 127).astype(np.int8)
    return q, scale


def _arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Quantize saved regression artifact to int8.")
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--model-path", default=None, type=str)
    return parser


def main() -> int:
    args = _arg_parser().parse_args()
    config = load_yaml_config(args.config)

    model_dir = ensure_dir(get_path(config, "model_dir", "artifacts/models/default"))
    report_dir = ensure_dir(get_path(config, "report_dir", "artifacts/reports/default"))

    model_path = Path(args.model_path).expanduser().resolve() if args.model_path else _find_latest_model(model_dir)
    data = np.load(model_path, allow_pickle=True)
    model_type = str(data["model_type"][0])

    out_name = f"{model_path.stem}_int8_{now_tag()}"
    q_model_path = model_dir / f"{out_name}.npz"

    report: Dict[str, object] = {
        "source_model": str(model_path),
        "quantized_model": str(q_model_path),
        "source_model_type": model_type,
    }

    if model_type == "linear_regression_baseline":
        weights = data["weights"].astype(np.float32)
        bias = data["bias"].astype(np.float32)
        q_weights, w_scale = _quantize_symmetric(weights)
        q_bias, b_scale = _quantize_symmetric(bias)
        deq_weights = q_weights.astype(np.float32) * w_scale
        deq_bias = q_bias.astype(np.float32) * b_scale
        report.update(
            {
                "weight_scale": w_scale,
                "bias_scale": b_scale,
                "weight_mse": float(np.mean((weights - deq_weights) ** 2)),
                "bias_mse": float(np.mean((bias - deq_bias) ** 2)),
            }
        )
        np.savez_compressed(
            q_model_path,
            model_type=np.array(["linear_regression_int8"]),
            q_weights=q_weights,
            q_bias=q_bias,
            w_scale=np.array([w_scale], dtype=np.float32),
            b_scale=np.array([b_scale], dtype=np.float32),
            x_mean=data["x_mean"],
            x_std=data["x_std"],
            label_names=data["label_names"],
            source_model=np.array([str(model_path)]),
        )
    elif model_type.startswith("tiny_cnn") and model_type.endswith("_float"):
        tensor_names = ["conv_w", "conv_b", "fc1_w", "fc1_b", "fc2_w", "fc2_b"]
        for optional_name in ("scalar_gate_w", "scalar_gate_b", "scalar_shift_w", "scalar_shift_b"):
            if optional_name in data:
                tensor_names.append(optional_name)
        q_model_type = model_type[:-6] + "_int8"
        payload: Dict[str, np.ndarray] = {
            "model_type": np.array([q_model_type]),
            "label_names": data["label_names"],
            "x_mean": data["x_mean"] if "x_mean" in data else np.zeros((int(data["input_channels"][0]),), dtype=np.float32),
            "x_std": data["x_std"] if "x_std" in data else np.ones((int(data["input_channels"][0]),), dtype=np.float32),
            "scalar_mean": data["scalar_mean"] if "scalar_mean" in data else np.zeros((int(data["scalar_feature_dim"][0]),), dtype=np.float32) if "scalar_feature_dim" in data else np.zeros((0,), dtype=np.float32),
            "scalar_std": data["scalar_std"] if "scalar_std" in data else np.ones((int(data["scalar_feature_dim"][0]),), dtype=np.float32) if "scalar_feature_dim" in data else np.ones((0,), dtype=np.float32),
            "y_mean": data["y_mean"],
            "y_std": data["y_std"],
            "input_shape": data["input_shape"],
            "input_channels": data["input_channels"] if "input_channels" in data else np.array([1], dtype=np.int32),
            "scalar_feature_dim": data["scalar_feature_dim"] if "scalar_feature_dim" in data else np.array([0], dtype=np.int32),
            "conv_channels": data["conv_channels"],
            "kernel_size": data["kernel_size"],
            "hidden_dim": data["hidden_dim"],
            "scalar_fusion_mode": data["scalar_fusion_mode"] if "scalar_fusion_mode" in data else np.array(["concat"]),
            "scalar_gate_init_bias": data["scalar_gate_init_bias"] if "scalar_gate_init_bias" in data else np.array([2.0], dtype=np.float32),
            "source_model": np.array([str(model_path)]),
        }
        for meta_key in (
            "target_semantics",
            "teacher_mode",
            "context_windows",
            "context_strategy",
            "residual_apply",
            "input_channel_names",
            "scalar_feature_names",
            "teacher_prediction_layout",
            "teacher_params_layout",
            "teacher_deltas_layout",
            "teacher_scalar_features",
        ):
            if meta_key in data:
                payload[meta_key] = data[meta_key]
        tensor_mse = {}
        for name in tensor_names:
            values = data[name].astype(np.float32)
            q_values, scale = _quantize_symmetric(values)
            deq_values = q_values.astype(np.float32) * scale
            payload[f"q_{name}"] = q_values
            payload[f"{name}_scale"] = np.array([scale], dtype=np.float32)
            tensor_mse[name] = float(np.mean((values - deq_values) ** 2))
        report["tensor_mse"] = tensor_mse
        report["mean_tensor_mse"] = float(np.mean(list(tensor_mse.values())))
        np.savez_compressed(q_model_path, **payload)
    else:
        raise ValueError(f"Unsupported model type for quantization: {model_type}")

    report_path = report_dir / f"{out_name}_quant_report.json"
    save_json(report_path, report)

    print("Quantization complete.")
    print(f"Source: {model_path}")
    print(f"Quantized: {q_model_path}")
    if "mean_tensor_mse" in report:
        print(f"Mean Tensor MSE: {float(report['mean_tensor_mse']):.8f}")
    else:
        print(
            f"Weight MSE: {float(report['weight_mse']):.8f}, "
            f"Bias MSE: {float(report['bias_mse']):.8f}"
        )
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
