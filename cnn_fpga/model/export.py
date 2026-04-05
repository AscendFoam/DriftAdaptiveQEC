"""Export trained artifacts to deployment formats such as TFLite or a TFLite-compatible stub."""

# 中文说明：
# - 真实 `.tflite` 导出依赖 TensorFlow；当前环境若缺失依赖，则自动回退为 `tflite_stub_v1` 清单。
# - 这样 `inference_service.backend=tflite` 仍可在工程链路中真实运行，只是执行后端是可追溯的 stub。

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from cnn_fpga.model.quantize import _find_latest_model
from cnn_fpga.utils.config import ensure_dir, get_path, load_yaml_config, now_tag, save_json


def _load_artifact(model_path: Path) -> Tuple[Dict[str, np.ndarray], str]:
    data = np.load(model_path, allow_pickle=True)
    model_type = str(data["model_type"][0])
    return data, model_type


def _build_tf_keras_from_tiny_cnn(data: Dict[str, np.ndarray]):
    import tensorflow as tf  # type: ignore

    input_h = int(data["input_shape"][0])
    input_w = int(data["input_shape"][1])
    input_channels = int(data["input_channels"][0]) if "input_channels" in data else 1
    conv_channels = int(data["conv_channels"][0])
    kernel_size = int(data["kernel_size"][0])
    hidden_dim = int(data["hidden_dim"][0])
    output_dim = int(data["y_mean"].shape[0])

    inputs = tf.keras.Input(shape=(input_h, input_w, input_channels), name="histogram")
    x_mean = data["x_mean"].astype(np.float32).reshape((1, 1, 1, input_channels)) if "x_mean" in data else np.zeros((1, 1, 1, input_channels), dtype=np.float32)
    x_std = data["x_std"].astype(np.float32).reshape((1, 1, 1, input_channels)) if "x_std" in data else np.ones((1, 1, 1, input_channels), dtype=np.float32)
    x = tf.keras.layers.Lambda(
        lambda tensor, mean, std: (tensor - mean) / std,
        arguments={"mean": x_mean, "std": x_std},
        name="input_norm",
    )(inputs)
    x = tf.keras.layers.Conv2D(conv_channels, kernel_size, padding="same", activation="relu", name="conv")(x)
    x = tf.keras.layers.AveragePooling2D(pool_size=2, name="pool")(x)
    # 中文注释：
    # - NumPy Tiny-CNN 在池化后使用的是 `(N, C, H, W)` 内存布局再直接 reshape。
    # - Keras 默认是 `(N, H, W, C)`；若直接 Flatten，会导致 `fc1` 看到的特征顺序错误。
    x = tf.keras.layers.Permute((3, 1, 2), name="to_nchw_order")(x)
    x = tf.keras.layers.Flatten(name="flatten")(x)
    x = tf.keras.layers.Dense(hidden_dim, activation="relu", name="fc1")(x)
    outputs_norm = tf.keras.layers.Dense(output_dim, activation=None, name="fc2")(x)

    y_mean = data["y_mean"].astype(np.float32).reshape((1, output_dim))
    y_std = data["y_std"].astype(np.float32).reshape((1, output_dim))

    # 中文注释：
    # - Tiny-CNN artifact 保存的是“训练域 -> 物理域”的反归一化参数 `pred = pred_norm * y_std + y_mean`。
    # - 若导出时遗漏这一步，TFLite 输出会停留在归一化空间，和运行时/benchmark 的物理语义不一致。
    outputs = tf.keras.layers.Lambda(
        lambda tensor, scale, offset: tensor * scale + offset,
        arguments={"scale": y_std, "offset": y_mean},
        name="output_denorm",
    )(outputs_norm)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model_type = str(data["model_type"][0])
    if model_type.endswith("_float"):
        conv_w = data["conv_w"].astype(np.float32)
        conv_b = data["conv_b"].astype(np.float32)
        fc1_w = data["fc1_w"].astype(np.float32)
        fc1_b = data["fc1_b"].astype(np.float32)
        fc2_w = data["fc2_w"].astype(np.float32)
        fc2_b = data["fc2_b"].astype(np.float32)
    else:
        conv_w = data["q_conv_w"].astype(np.float32) * float(data["conv_w_scale"][0])
        conv_b = data["q_conv_b"].astype(np.float32) * float(data["conv_b_scale"][0])
        fc1_w = data["q_fc1_w"].astype(np.float32) * float(data["fc1_w_scale"][0])
        fc1_b = data["q_fc1_b"].astype(np.float32) * float(data["fc1_b_scale"][0])
        fc2_w = data["q_fc2_w"].astype(np.float32) * float(data["fc2_w_scale"][0])
        fc2_b = data["q_fc2_b"].astype(np.float32) * float(data["fc2_b_scale"][0])

    # 中文注释：NumPy Tiny-CNN 存的是 OIHW / (in, out)；Keras 需要 HWIO / (in, out)。
    model.get_layer("conv").set_weights([np.transpose(conv_w, (2, 3, 1, 0)), conv_b])
    model.get_layer("fc1").set_weights([fc1_w, fc1_b])
    model.get_layer("fc2").set_weights([fc2_w, fc2_b])
    return model


def _export_true_tflite(data: Dict[str, np.ndarray], model_path: Path, out_path: Path) -> Dict:
    import tensorflow as tf  # type: ignore

    model_type = str(data["model_type"][0])
    if not (model_type.startswith("tiny_cnn") and (model_type.endswith("_float") or model_type.endswith("_int8"))):
        raise ValueError(f"Unsupported model type for TFLite export: {model_type}")

    keras_model = _build_tf_keras_from_tiny_cnn(data)
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_bytes = converter.convert()
    out_path.write_bytes(tflite_bytes)
    return {
        "export_mode": "true_tflite",
        "output_path": str(out_path),
        "source_model": str(model_path),
        "source_model_type": model_type,
    }


def _export_tflite_stub(data: Dict[str, np.ndarray], model_path: Path, out_path: Path) -> Dict:
    manifest = {
        "format": "tflite_stub_v1",
        "source_model": str(model_path),
        "source_model_type": str(data["model_type"][0]),
        "weights_path": str(model_path),
        "label_names": [str(item) for item in data["label_names"]],
        "input_shape": [int(x) for x in data["input_shape"]],
        "note": "TensorFlow/TFLite runtime unavailable in current environment; use artifact-backed stub for pipeline validation.",
    }
    out_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "export_mode": "tflite_stub",
        "output_path": str(out_path),
        "source_model": str(model_path),
        "source_model_type": str(data["model_type"][0]),
        "manifest": manifest,
    }


def _arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export trained model artifact to deployment formats.")
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--model-path", default=None, type=str)
    parser.add_argument("--backend", default="tflite", type=str, choices=["tflite"])
    return parser


def main() -> int:
    args = _arg_parser().parse_args()
    config = load_yaml_config(args.config)

    model_dir = ensure_dir(get_path(config, "model_dir", "artifacts/models/default"))
    report_dir = ensure_dir(get_path(config, "report_dir", "artifacts/reports/default"))
    model_path = Path(args.model_path).expanduser().resolve() if args.model_path else _find_latest_model(model_dir)
    data, model_type = _load_artifact(model_path)

    if not (model_type.startswith("tiny_cnn") and (model_type.endswith("_float") or model_type.endswith("_int8"))):
        raise ValueError(f"Only Tiny-CNN artifacts are supported for export, got: {model_type}")

    out_stem = f"{model_path.stem}_tflite_{now_tag()}"
    true_tflite_path = model_dir / f"{out_stem}.tflite"
    stub_tflite_path = model_dir / f"{out_stem}.tflite.json"

    try:
        report = _export_true_tflite(data, model_path, true_tflite_path)
    except Exception as exc:
        report = _export_tflite_stub(data, model_path, stub_tflite_path)
        report["fallback_reason"] = str(exc)

    report_path = report_dir / f"{out_stem}_export_report.json"
    save_json(report_path, report)

    print("Export complete.")
    print(f"Source: {model_path}")
    print(f"Output: {report['output_path']}")
    print(f"Mode: {report['export_mode']}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
