"""Evaluate baseline model against a dataset split."""

# 中文说明：
# - 读取已训练模型与指定分片数据，输出总体指标与逐标签指标。
# - 主要用于快速检查训练产物是否达到可用水平。

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np

from cnn_fpga.utils.config import ensure_dir, get_path, load_yaml_config, now_tag, save_json


def _load_model(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    data = np.load(model_path)
    return {
        "weights": data["weights"].astype(np.float64),
        "bias": data["bias"].astype(np.float64),
        "x_mean": data["x_mean"].astype(np.float64),
        "x_std": data["x_std"].astype(np.float64),
        "label_names": [str(x) for x in data["label_names"]],
    }


def _load_split(dataset_dir: Path, split: str):
    path = dataset_dir / f"{split}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Dataset split not found: {path}")
    data = np.load(path)
    x = data["histograms"].reshape(data["histograms"].shape[0], -1).astype(np.float64)
    y = data["labels"].astype(np.float64)
    return x, y


def _predict(x: np.ndarray, model: Dict[str, np.ndarray]):
    # 中文注释：推理时使用训练期保存的标准化参数，保证前后处理一致。
    x_norm = (x - model["x_mean"]) / model["x_std"]
    return x_norm @ model["weights"] + model["bias"]


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot < 1e-12:
        return 0.0
    return 1.0 - ss_res / ss_tot


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, label_names):
    # 中文注释：同时给出整体误差和每个标签维度的误差，便于定位问题来源。
    out = {
        "mse": float(np.mean((y_true - y_pred) ** 2)),
        "mae": float(np.mean(np.abs(y_true - y_pred))),
        "r2_mean": float(np.mean([_r2_score(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])])),
        "per_label": {},
    }
    for i, name in enumerate(label_names):
        out["per_label"][name] = {
            "mse": float(np.mean((y_true[:, i] - y_pred[:, i]) ** 2)),
            "mae": float(np.mean(np.abs(y_true[:, i] - y_pred[:, i]))),
            "r2": float(_r2_score(y_true[:, i], y_pred[:, i])),
        }
    return out


def _arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate baseline model.")
    parser.add_argument("--config", required=True, type=str, help="Path to YAML config")
    parser.add_argument("--split", default=None, type=str, help="Dataset split to evaluate")
    parser.add_argument("--model-path", default=None, type=str, help="Model path override")
    return parser


def _find_latest_model(model_dir: Path) -> Path:
    candidates = sorted(model_dir.glob("*.npz"))
    if not candidates:
        raise FileNotFoundError(f"No model files in {model_dir}")
    return candidates[-1]


def main() -> int:
    args = _arg_parser().parse_args()
    config = load_yaml_config(args.config)

    dataset_dir = get_path(config, "dataset_dir", "artifacts/datasets/default")
    model_dir = ensure_dir(get_path(config, "model_dir", "artifacts/models/default"))
    report_dir = ensure_dir(get_path(config, "report_dir", "artifacts/reports/default"))

    # 中文注释：命令行优先级高于配置文件，便于临时切换评估分片。
    split = args.split or str(config.get("evaluation", {}).get("target_split", "test"))
    model_path = Path(args.model_path).expanduser().resolve() if args.model_path else _find_latest_model(model_dir)

    model = _load_model(model_path)
    x_eval, y_eval = _load_split(dataset_dir, split=split)
    y_pred = _predict(x_eval, model)
    metrics = _metrics(y_eval, y_pred, model["label_names"])

    run_name = f"eval_{split}_{now_tag()}"
    report = {
        "run_name": run_name,
        "model_path": str(model_path),
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

"""
### 代码整体功能解析
这段代码的核心作用是**评估一个已训练的基线模型在指定数据集分片（如测试集）上的性能**，输出整体和逐标签的评估指标（MSE、MAE、R²），并将评估结果保存为JSON报告，用于快速验证模型是否达到可用水平。

---

### 核心模块逐段解析
#### 1. 基础函数：加载资源
- `_load_model(model_path)`：读取保存为`.npz`格式的模型文件，解析出模型权重、偏置、数据标准化参数（均值/标准差）和标签名，确保数据类型为`float64`以保证精度。
- `_load_split(dataset_dir, split)`：读取指定分片的数据集（如`test.npz`），将特征数据展平为二维数组（样本数×特征数），同时加载标签数据，为模型推理做准备。

#### 2. 模型推理与指标计算
- `_predict(x, model)`：对输入特征先做**标准化**（使用训练时保存的均值/标准差，保证前后处理一致），再通过线性计算（`x_norm @ weights + bias`）得到预测值，这是典型的线性模型推理逻辑。
- `_r2_score(y_true, y_pred)`：计算决定系数R²，衡量模型对数据的拟合程度（R²越接近1，拟合效果越好；若分母过小则返回0避免除零错误）。
- `_metrics(...)`：核心指标计算函数：
  - 整体指标：均方误差（MSE）、平均绝对误差（MAE）、所有标签R²的均值（r2_mean）；
  - 逐标签指标：为每个标签单独计算MSE、MAE、R²，便于定位某类标签的预测问题。

#### 3. 命令行参数与路径处理
- `_arg_parser()`：定义命令行参数（配置文件路径、数据集分片名、模型路径），支持灵活指定评估参数。
- `_find_latest_model(model_dir)`：自动查找模型目录下最新的`.npz`模型文件，无需手动指定路径。
- 路径优先级：命令行传入的参数（如分片名、模型路径）> 配置文件中的参数，方便临时切换评估目标。

#### 4. 主流程（main函数）
1. 解析命令行参数，加载YAML配置文件；
2. 确定数据集、模型、报告的存储路径；
3. 加载模型和指定分片的数据集；
4. 执行模型推理，计算评估指标；
5. 生成包含评估信息的JSON报告（保存路径、样本数、指标等），并打印关键指标；
6. 返回执行状态码，完成评估。

---

### 总结
1. **核心目标**：快速评估线性基线模型在指定数据集分片上的性能，输出整体+逐标签的MSE/MAE/R²指标，验证模型可用性；
2. **关键设计**：使用训练期的标准化参数保证推理一致性，支持命令行灵活指定参数，自动保存评估报告便于追溯；
3. **适用场景**：模型训练后快速校验，定位某类标签的预测误差问题，判断模型是否达到基础可用标准。
"""