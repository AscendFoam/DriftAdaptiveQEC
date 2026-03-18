"""Train a minimal baseline regressor from histogram dataset splits."""

# 中文说明：
# - 当前是“可运行脚手架”训练入口，先用线性回归打通流程。
# - 后续可在保持接口不变的前提下替换为 Tiny-CNN 等模型。

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np

from cnn_fpga.utils.config import config_hash, ensure_dir, get_path, load_yaml_config, now_tag, save_json


def _load_split(dataset_dir: Path, split: str):
    path = dataset_dir / f"{split}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Dataset split not found: {path}")
    data = np.load(path)
    return data["histograms"], data["labels"], data["label_names"]


def _fit_linear_regression(x: np.ndarray, y: np.ndarray, l2_reg: float):
    # 中文注释：先做特征标准化，再解带 L2 正则的最小二乘闭式解。
    x_mean = x.mean(axis=0, keepdims=True)
    x_std = x.std(axis=0, keepdims=True)
    x_std = np.where(x_std < 1e-8, 1.0, x_std)
    x_norm = (x - x_mean) / x_std

    y_mean = y.mean(axis=0, keepdims=True)
    y_center = y - y_mean

    n_features = x_norm.shape[1]
    gram = x_norm.T @ x_norm + l2_reg * np.eye(n_features, dtype=np.float64)
    target = x_norm.T @ y_center
    weights = np.linalg.solve(gram, target)
    bias = y_mean.reshape(-1)
    return weights, bias, x_mean.reshape(-1), x_std.reshape(-1)


def _predict(x: np.ndarray, weights: np.ndarray, bias: np.ndarray, x_mean: np.ndarray, x_std: np.ndarray):
    x_norm = (x - x_mean) / x_std
    return x_norm @ weights + bias


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mse = float(np.mean((y_true - y_pred) ** 2))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    return {"mse": mse, "mae": mae}


def _arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train baseline linear regressor from histogram dataset.")
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

    x_train_raw, y_train, label_names = _load_split(dataset_dir, args.train_split)
    x_val_raw, y_val, _ = _load_split(dataset_dir, args.val_split)

    train_cfg = config.get("training", {})
    max_train_samples = int(train_cfg.get("max_train_samples", x_train_raw.shape[0]))
    l2_reg = float(train_cfg.get("l2_reg", 1.0e-3))

    # 中文注释：限制训练样本量，防止脚手架阶段内存/耗时过大。
    x_train_raw = x_train_raw[:max_train_samples]
    y_train = y_train[:max_train_samples]

    x_train = x_train_raw.reshape(x_train_raw.shape[0], -1).astype(np.float64)
    x_val = x_val_raw.reshape(x_val_raw.shape[0], -1).astype(np.float64)
    y_train = y_train.astype(np.float64)
    y_val = y_val.astype(np.float64)

    weights, bias, x_mean, x_std = _fit_linear_regression(x_train, y_train, l2_reg=l2_reg)
    pred_train = _predict(x_train, weights, bias, x_mean, x_std)
    pred_val = _predict(x_val, weights, bias, x_mean, x_std)

    train_metrics = _metrics(y_train, pred_train)
    val_metrics = _metrics(y_val, pred_val)

    # 中文注释：run_name 带时间戳与配置哈希，便于追溯。
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
        "train_split": args.train_split,
        "val_split": args.val_split,
        "n_train": int(x_train.shape[0]),
        "n_val": int(x_val.shape[0]),
        "l2_reg": l2_reg,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
    }
    save_json(report_dir / f"{run_name}_train_report.json", report)

    print("Training complete.")
    print(f"Model: {model_path}")
    print(f"Train MSE: {train_metrics['mse']:.6f}, Val MSE: {val_metrics['mse']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""
### 代码整体功能解析
这段代码的核心是**训练一个基于线性回归的极简基线回归模型**，数据源是分块存储的直方图数据集（.npz 格式），主要用于快速打通“数据加载→模型训练→评估→保存”的完整流程，为后续替换成 Tiny-CNN 等更复杂模型提供可复用的脚手架。

---

### 核心模块逐段解析
#### 1. 数据加载：`_load_split` 函数
- 功能：从指定目录加载训练/验证集的 `.npz` 文件，读取其中的 `histograms`（直方图特征）、`labels`（回归目标值）、`label_names`（标签名称）。
- 关键逻辑：检查文件是否存在，不存在则抛出异常，确保数据链路的健壮性。

#### 2. 线性回归训练：`_fit_linear_regression` 函数
这是核心训练逻辑，实现带 L2 正则的线性回归闭式解（最小二乘法）：
1. **特征标准化**：对输入特征 `x` 做均值减、方差除（避免数值量级问题），对标签 `y` 做中心化（减均值）；
2. **正则化求解**：构造 Gram 矩阵 `X^T·X + λI`（`λ` 是 L2 正则系数，防止过拟合），通过 `np.linalg.solve` 求解权重 `weights`；
3. **偏置计算**：偏置 `bias` 直接取标签的均值，最终返回权重、偏置、特征均值/方差（用于后续预测时的标准化）。

#### 3. 预测与评估：`_predict` + `_metrics` 函数
- `_predict`：用训练好的参数对新特征做标准化，再计算 `X·weights + bias` 得到预测值；
- `_metrics`：计算回归任务的核心指标——均方误差（MSE）、平均绝对误差（MAE），评估模型效果。

#### 4. 主流程：`main` 函数
1. **参数解析**：读取配置文件路径、训练/验证集名称；
2. **路径初始化**：创建模型、报告的保存目录；
3. **数据预处理**：
   - 加载训练/验证集，限制最大训练样本数（避免脚手架阶段内存/耗时过高）；
   - 将直方图特征从多维展平为一维（如 (N, H, W) → (N, H*W)），转为 float64 提升计算精度；
4. **模型训练**：调用 `_fit_linear_regression` 训练线性回归模型；
5. **评估与保存**：
   - 计算训练/验证集的 MSE/MAE；
   - 保存模型参数（权重、偏置、特征均值/方差）到 `.npz` 文件，文件名带时间戳+配置哈希（便于追溯）；
   - 保存训练报告（数据集路径、样本量、指标等）到 JSON 文件。

---

### 关键设计亮点
1. **脚手架特性**：接口标准化，后续替换为 Tiny-CNN 时无需修改数据加载、评估、保存等逻辑，只需替换 `_fit_linear_regression` 函数；
2. **可追溯性**：模型/报告命名带时间戳（`now_tag`）和配置哈希（`config_hash`），能精准定位训练时的配置和时间；
3. **鲁棒性**：特征标准化时处理了方差为 0 的情况（`np.where(x_std < 1e-8, 1.0, x_std)`），避免除以 0 错误；
4. **资源控制**：限制最大训练样本数，适配脚手架阶段的轻量需求。

---

### 总结
1. 核心目标：用线性回归快速搭建“直方图数据→回归模型”的可运行流程，作为后续复杂模型的基线；
2. 核心逻辑：带 L2 正则的线性回归闭式解，通过特征标准化提升训练稳定性，用 MSE/MAE 评估回归效果；
3. 工程设计：路径管理规范化、模型/报告可追溯、接口可扩展（便于替换为 CNN 等模型）。
"""