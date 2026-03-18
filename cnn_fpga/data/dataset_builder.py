"""Generate first-version training datasets for CNN-FPGA experiments."""

# 中文说明：
# - 该脚本根据 YAML 配置合成 syndrome 直方图，并输出 train/val/test 三个分片。
# - 当前标签为 (sigma, mu_q, mu_p, theta_deg)，用于回归任务基线。

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from physics.gkp_state import LATTICE_CONST

from cnn_fpga.utils.config import config_hash, ensure_dir, get_path, load_yaml_config, save_json


LABEL_NAMES = ("sigma", "mu_q", "mu_p", "theta_deg")


def _sample_param(rng: np.random.Generator, bounds: Tuple[float, float]) -> float:
    return float(rng.uniform(bounds[0], bounds[1]))


def _wrap_to_fundamental(value: np.ndarray, lattice: float) -> np.ndarray:
    # 中文注释：把误差映射到基本晶胞区间 [-lambda/2, +lambda/2]。
    return np.mod(value + lattice / 2.0, lattice) - lattice / 2.0


def _histogram_from_params(
    rng: np.random.Generator,
    sigma: float,
    mu_q: float,
    mu_p: float,
    theta_deg: float,
    points_per_sample: int,
    bins: int,
    lattice: float = LATTICE_CONST,
) -> np.ndarray:
    # 中文注释：给定一组参数，采样点云后统计为固定大小直方图输入。
    theta = np.deg2rad(theta_deg)
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype=float)
    base = rng.normal(0.0, sigma, size=(points_per_sample, 2))
    rotated = (rotation @ base.T).T
    displaced = rotated + np.array([mu_q, mu_p], dtype=float)
    wrapped = _wrap_to_fundamental(displaced, lattice)

    hist, _, _ = np.histogram2d(
        wrapped[:, 0],
        wrapped[:, 1],
        bins=bins,
        range=[[-lattice / 2.0, lattice / 2.0], [-lattice / 2.0, lattice / 2.0]],
    )
    hist_sum = float(hist.sum())
    if hist_sum > 0.0:
        # 中文注释：归一化后更适合后续模型训练，降低样本尺度差异。
        hist = hist / hist_sum
    return hist.astype(np.float32)


def _build_dataset(config: Dict, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    dataset_cfg = config["dataset"]
    ranges = dataset_cfg["parameter_ranges"]
    n_cfg = int(dataset_cfg["n_configurations"])
    samples_per_cfg = int(dataset_cfg["samples_per_configuration"])
    points_per_sample = int(dataset_cfg["points_per_sample"])
    bins = int(dataset_cfg["histogram_bins"])

    total = n_cfg * samples_per_cfg
    histograms = np.zeros((total, bins, bins), dtype=np.float32)
    labels = np.zeros((total, len(LABEL_NAMES)), dtype=np.float32)

    # 中文注释：先采样“配置级参数”，再在每个配置下生成多条样本。
    idx = 0
    for _ in range(n_cfg):
        sigma = _sample_param(rng, tuple(ranges["sigma"]))
        mu_q = _sample_param(rng, tuple(ranges["mu_q"]))
        mu_p = _sample_param(rng, tuple(ranges["mu_p"]))
        theta_deg = _sample_param(rng, tuple(ranges["theta_deg"]))

        for _ in range(samples_per_cfg):
            hist = _histogram_from_params(
                rng=rng,
                sigma=sigma,
                mu_q=mu_q,
                mu_p=mu_p,
                theta_deg=theta_deg,
                points_per_sample=points_per_sample,
                bins=bins,
            )
            histograms[idx] = hist
            labels[idx] = np.array([sigma, mu_q, mu_p, theta_deg], dtype=np.float32)
            idx += 1

    return histograms, labels


def _split_indices(n_samples: int, train_ratio: float, val_ratio: float, rng: np.random.Generator):
    # 中文注释：按随机置乱后的索引切分 train/val/test。
    indices = rng.permutation(n_samples)
    train_end = int(n_samples * train_ratio)
    val_end = train_end + int(n_samples * val_ratio)
    return indices[:train_end], indices[train_end:val_end], indices[val_end:]


def _save_split(
    out_dir: Path,
    split_name: str,
    histograms: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
) -> Path:
    split_path = out_dir / f"{split_name}.npz"
    np.savez_compressed(
        split_path,
        histograms=histograms[indices],
        labels=labels[indices],
        label_names=np.array(LABEL_NAMES),
    )
    return split_path


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build synthetic histogram dataset from YAML config.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--dry-run", action="store_true", help="Only validate config and print plan")
    return parser


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()
    config = load_yaml_config(args.config)

    experiment = config.get("experiment", {})
    seed = int(experiment.get("seed", 1234))
    rng = np.random.default_rng(seed)

    dataset_dir = ensure_dir(get_path(config, "dataset_dir", "artifacts/datasets/default"))
    run_id = config_hash(config)

    dataset_cfg = config["dataset"]
    split_cfg = dataset_cfg["split"]

    # 中文注释：先打印构建计划，方便核对配置是否符合预期。
    plan = {
        "dataset_dir": str(dataset_dir),
        "run_id": run_id,
        "seed": seed,
        "n_configurations": int(dataset_cfg["n_configurations"]),
        "samples_per_configuration": int(dataset_cfg["samples_per_configuration"]),
        "histogram_bins": int(dataset_cfg["histogram_bins"]),
        "points_per_sample": int(dataset_cfg["points_per_sample"]),
        "split": split_cfg,
    }
    print("Dataset build plan:", plan)

    if args.dry_run:
        return 0

    histograms, labels = _build_dataset(config, rng)
    n_samples = histograms.shape[0]

    train_idx, val_idx, test_idx = _split_indices(
        n_samples=n_samples,
        train_ratio=float(split_cfg["train"]),
        val_ratio=float(split_cfg["val"]),
        rng=rng,
    )

    train_path = _save_split(dataset_dir, "train", histograms, labels, train_idx)
    val_path = _save_split(dataset_dir, "val", histograms, labels, val_idx)
    test_path = _save_split(dataset_dir, "test", histograms, labels, test_idx)

    # 中文注释：manifest 记录数据规模、标签名与文件路径，便于后续复现实验。
    manifest = {
        "run_id": run_id,
        "seed": seed,
        "n_samples_total": int(n_samples),
        "n_train": int(train_idx.size),
        "n_val": int(val_idx.size),
        "n_test": int(test_idx.size),
        "label_names": list(LABEL_NAMES),
        "files": {
            "train": str(train_path),
            "val": str(val_path),
            "test": str(test_path),
        },
    }
    save_json(dataset_dir / "manifest.json", manifest)
    print(f"Dataset written to {dataset_dir}")
    print(f"Manifest: {dataset_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""
### 代码整体功能解析
这段Python脚本的核心作用是**为CNN-FPGA实验生成第一版的合成训练数据集**，具体是基于物理参数生成 syndrome 直方图数据，并按训练/验证/测试集切分，用于后续的回归任务基线训练。

---

### 核心逻辑拆解
#### 1. 关键函数与流程
| 函数/模块                | 核心作用                                                                 |
|--------------------------|--------------------------------------------------------------------------|
| `_sample_param`          | 从指定数值区间内随机采样单个物理参数（如sigma、mu_q等）                  |
| `_wrap_to_fundamental`   | 将物理误差值映射到基本晶胞区间 `[-lambda/2, +lambda/2]`，保证数值合理性   |
| `_histogram_from_params` | 核心生成逻辑：
                            1. 将随机生成的二维正态分布点云做旋转变换（theta角度）
                            2. 叠加偏移量（mu_q/mu_p）
                            3. 映射到晶胞区间后生成2D直方图
                            4. 直方图归一化（消除样本尺度差异
| `_build_dataset`         | 批量生成数据集：
                            1. 先采样多组“配置级参数”（sigma/mu_q/mu_p/theta）
                            2. 每组配置下生成多条样本，输出直方图+对应标签 
| `_split_indices`         | 随机打乱样本索引，按比例切分train/val/test集（避免数据分布偏差）          |
| `_save_split`            | 将切分后的数据集以压缩npz格式保存，包含直方图、标签、标签名              |
| `main`                   | 主流程：加载配置→初始化随机数→生成数据集→切分→保存+生成实验记录（manifest） |

#### 2. 数据生成规则
- **输入**：YAML配置文件（指定参数范围、样本数量、直方图分箱数、数据集切分比例等）
- **输出**：
  - train.npz/val.npz/test.npz：压缩的numpy文件，包含`histograms`（二维归一化直方图）和`labels`（对应4个物理参数）
  - manifest.json：记录数据集基本信息（样本数量、文件路径、标签名等），方便实验复现
- **标签维度**：`(sigma, mu_q, mu_p, theta_deg)`，对应4个回归任务目标

#### 3. 关键细节
- 随机数种子固定（默认1234），保证数据集可复现；
- 直方图归一化：将每个直方图的像素值求和后取商，消除不同样本点数带来的尺度差异；
- 数据切分前先随机打乱索引，避免数据分布不均；
- 支持“dry-run”模式：仅验证配置并打印计划，不实际生成数据。

---

### 总结
1. **核心目标**：基于物理参数（sigma/mu_q/mu_p/theta）合成2D直方图数据集，用于CNN-FPGA的回归任务基线训练；
2. **生成逻辑**：随机采样参数→生成带旋转/偏移的正态分布点云→映射到晶胞区间→生成归一化2D直方图→按比例切分数据集；
3. **工程特性**：支持配置驱动、可复现（固定种子）、数据归一化、实验记录（manifest），符合机器学习数据集生成的最佳实践。
"""