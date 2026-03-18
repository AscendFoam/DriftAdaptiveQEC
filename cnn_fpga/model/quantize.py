"""Quantize baseline linear model to int8 as first deployment scaffold."""

# 中文说明：
# - 该脚本实现首版 int8 量化流程，便于验证“训练 -> 部署”链路完整性。
# - 当前采用对称量化并输出量化误差，后续可替换为更复杂方案（如QAT）。

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np

from cnn_fpga.utils.config import ensure_dir, get_path, load_yaml_config, now_tag, save_json


def _find_latest_model(model_dir: Path) -> Path:
    candidates = sorted(model_dir.glob("*.npz"))
    if not candidates:
        raise FileNotFoundError(f"No model files in {model_dir}")
    return candidates[-1]


def _quantize_symmetric(values: np.ndarray) -> Tuple[np.ndarray, float]:
    # 中文注释：以最大绝对值确定 scale，映射到 int8 范围 [-127,127]。
    max_abs = float(np.max(np.abs(values)))
    if max_abs < 1e-12:
        scale = 1.0
    else:
        scale = max_abs / 127.0
    q = np.clip(np.round(values / scale), -127, 127).astype(np.int8)
    return q, scale


def _arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Quantize baseline model to int8.")
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--model-path", default=None, type=str)
    return parser


def main() -> int:
    args = _arg_parser().parse_args()
    config = load_yaml_config(args.config)

    model_dir = ensure_dir(get_path(config, "model_dir", "artifacts/models/default"))
    report_dir = ensure_dir(get_path(config, "report_dir", "artifacts/reports/default"))

    model_path = Path(args.model_path).expanduser().resolve() if args.model_path else _find_latest_model(model_dir)
    data = np.load(model_path)

    weights = data["weights"].astype(np.float32)
    bias = data["bias"].astype(np.float32)

    q_weights, w_scale = _quantize_symmetric(weights)
    q_bias, b_scale = _quantize_symmetric(bias)

    # 中文注释：反量化用于估计量化引入的参数失真。
    deq_weights = q_weights.astype(np.float32) * w_scale
    deq_bias = q_bias.astype(np.float32) * b_scale

    weight_mse = float(np.mean((weights - deq_weights) ** 2))
    bias_mse = float(np.mean((bias - deq_bias) ** 2))

    out_name = f"{model_path.stem}_int8_{now_tag()}"
    q_model_path = model_dir / f"{out_name}.npz"
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

    report = {
        "source_model": str(model_path),
        "quantized_model": str(q_model_path),
        "weight_scale": w_scale,
        "bias_scale": b_scale,
        "weight_mse": weight_mse,
        "bias_mse": bias_mse,
    }
    report_path = report_dir / f"{out_name}_quant_report.json"
    save_json(report_path, report)

    print("Quantization complete.")
    print(f"Source: {model_path}")
    print(f"Quantized: {q_model_path}")
    print(f"Weight MSE: {weight_mse:.8f}, Bias MSE: {bias_mse:.8f}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
### 代码整体功能解析
这段代码的核心是**将训练好的基线线性模型从浮点精度（float32）量化为INT8整数精度**，并生成量化报告，作为模型部署的第一版脚手架，验证“训练→部署”全链路的完整性。

### 核心逻辑分步解析
#### 1. 基础工具函数
- `_find_latest_model`: 从指定目录找到最新的`.npz`格式模型文件（按文件名排序取最后一个），如果目录无模型文件则抛出异常。
- `_arg_parser`: 定义命令行参数，支持指定配置文件（必填）和模型路径（可选）。
- 辅助函数（`ensure_dir`/`get_path`/`load_yaml_config`等）：处理路径、配置加载、目录创建等基础操作。

#### 2. 核心量化函数 `_quantize_symmetric`
实现**对称量化**（核心逻辑）：
1. 计算输入数组（权重/偏置）的最大绝对值 `max_abs`；
2. 计算量化缩放因子`scale`：`scale = 最大绝对值 / 127`（因为INT8的有效范围是[-127, 127]，避开-128防止溢出）；
3. 量化过程：`浮点值 / scale` → 四舍五入 → 裁剪到[-127, 127]范围 → 转为`int8`类型；
4. 返回量化后的INT8数组和缩放因子（后续反量化/推理时需要用）。

#### 3. 主流程 `main` 函数
1. **参数与配置加载**：解析命令行参数，加载YAML配置文件，确定模型目录、报告目录并确保目录存在。
2. **模型加载**：优先使用用户指定的模型路径，未指定则找最新模型；加载`.npz`模型文件，提取权重（`weights`）、偏置（`bias`）及预处理参数（`x_mean`/`x_std`等）。
3. **执行量化**：
   - 对权重、偏置分别调用`_quantize_symmetric`完成INT8量化，得到量化后的数据（`q_weights`/`q_bias`）和缩放因子（`w_scale`/`b_scale`）；
   - 反量化验证：将INT8数据转回float32（`量化值 × scale`），计算与原始浮点值的**均方误差（MSE）**，评估量化失真程度。
4. **保存量化模型**：将量化后的权重/偏置、缩放因子、预处理参数等保存为新的`.npz`文件，记录源模型路径、模型类型等元信息。
5. **生成量化报告**：将源模型路径、量化模型路径、缩放因子、MSE误差等信息保存为JSON报告，便于后续分析量化效果。
6. **输出日志**：打印量化完成信息、误差值、文件路径等，直观展示量化结果。

#### 4. 执行入口
`if __name__ == "__main__"` 中调用`main`函数，通过`SystemExit`返回执行状态码，符合命令行脚本的规范。

### 关键细节说明
1. **对称量化特点**：以0为中心，仅用一个缩放因子（scale），计算简单、部署友好，适合线性模型；
2. **量化误差评估**：通过MSE（均方误差）衡量量化后参数与原始参数的差异，MSE越小说明量化失真越小；
3. **保留预处理参数**：量化模型仍保留`x_mean`/`x_std`（输入数据的均值/标准差），保证推理时数据预处理与训练一致；
4. **可扩展性**：当前仅实现基础对称量化，后续可替换为QAT（量化感知训练）等更复杂的量化方案。

### 总结
1. 核心目标：将浮点线性模型量化为INT8，验证训练到部署的链路完整性，降低模型存储/计算成本；
2. 核心方法：对称量化（基于最大绝对值计算scale，映射到INT8范围[-127,127]），并通过MSE评估量化失真；
3. 输出产物：量化后的INT8模型文件（.npz）+ 量化报告（JSON），包含缩放因子、误差等关键信息，便于部署验证。
"""