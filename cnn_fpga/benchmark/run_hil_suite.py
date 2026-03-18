"""Run first-version HIL suite scaffold (mock backend by default)."""

# 中文说明：
# - 该脚本用于 HIL 流程脚手架验证，当前默认 mock 后端（不依赖真实板卡）。
# - 目标是先打通时延统计、预算违约率、通信错误注入这条工程链路。

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np

from cnn_fpga.utils.config import config_hash, ensure_dir, get_path, load_yaml_config, now_tag, save_json


def _sample_positive_normal(rng: np.random.Generator, mean: float, std: float, size: int) -> np.ndarray:
    # 中文注释：时延不能为负值，因此做下限裁剪。
    values = rng.normal(mean, std, size=size)
    return np.clip(values, 0.0, None)


def _run_mock_hil(config: Dict, run_dir: Path) -> Dict:
    experiment = config.get("experiment", {})
    seed = int(experiment.get("seed", 1234))
    rng = np.random.default_rng(seed)

    hil_cfg = config.get("hil", {})
    if str(hil_cfg.get("backend", "mock")).lower() != "mock":
        raise NotImplementedError("Only mock backend is implemented in this scaffold.")

    timing_cfg = config.get("timing", {})
    n_updates = int(timing_cfg.get("n_slow_updates", 1000))
    n_fast_cycles = int(timing_cfg.get("n_fast_cycles", 200000))

    slow_budget_us = float(timing_cfg.get("slow_update_budget_us", 5000.0))
    fast_budget_us = float(timing_cfg.get("fast_cycle_budget_us", 1.5))

    latency_cfg = config.get("latency_model", {})
    dma = _sample_positive_normal(rng, float(latency_cfg.get("dma_mean_us", 10.0)), float(latency_cfg.get("dma_std_us", 2.0)), n_updates)
    preprocess = _sample_positive_normal(rng, float(latency_cfg.get("preprocess_mean_us", 50.0)), float(latency_cfg.get("preprocess_std_us", 10.0)), n_updates)
    inference = _sample_positive_normal(rng, float(latency_cfg.get("inference_mean_us", 800.0)), float(latency_cfg.get("inference_std_us", 120.0)), n_updates)
    writeback = _sample_positive_normal(rng, float(latency_cfg.get("writeback_mean_us", 20.0)), float(latency_cfg.get("writeback_std_us", 5.0)), n_updates)

    # 中文注释：慢回路总延迟由 DMA、预处理、推理、写回四部分组成。
    slow_total = dma + preprocess + inference + writeback
    slow_violation_mask = slow_total > slow_budget_us

    fast_cfg = config.get("fast_path_model", {})
    fast_latency = _sample_positive_normal(
        rng,
        float(fast_cfg.get("latency_mean_us", 1.0)),
        float(fast_cfg.get("latency_std_us", 0.15)),
        n_fast_cycles,
    )
    fast_violation_mask = fast_latency > fast_budget_us

    # 中文注释：按配置概率注入通信与推理失败事件，用于评估容错策略。
    err_cfg = config.get("error_model", {})
    dma_timeout_prob = float(err_cfg.get("dma_timeout_prob", 1e-4))
    axi_write_fail_prob = float(err_cfg.get("axi_write_fail_prob", 5e-5))
    cnn_infer_fail_prob = float(err_cfg.get("cnn_infer_fail_prob", 2e-5))

    dma_timeouts = int(np.sum(rng.random(n_updates) < dma_timeout_prob))
    axi_write_failures = int(np.sum(rng.random(n_updates) < axi_write_fail_prob))
    cnn_infer_failures = int(np.sum(rng.random(n_updates) < cnn_infer_fail_prob))

    summary = {
        "n_slow_updates": n_updates,
        "n_fast_cycles": n_fast_cycles,
        "slow_update_budget_us": slow_budget_us,
        "fast_cycle_budget_us": fast_budget_us,
        "slow_update_mean_us": float(np.mean(slow_total)),
        "slow_update_p95_us": float(np.percentile(slow_total, 95)),
        "slow_update_p99_us": float(np.percentile(slow_total, 99)),
        "slow_update_violation_rate": float(np.mean(slow_violation_mask)),
        "fast_cycle_mean_us": float(np.mean(fast_latency)),
        "fast_cycle_p95_us": float(np.percentile(fast_latency, 95)),
        "fast_cycle_p99_us": float(np.percentile(fast_latency, 99)),
        "fast_cycle_violation_rate": float(np.mean(fast_violation_mask)),
        "dma_timeouts": dma_timeouts,
        "axi_write_failures": axi_write_failures,
        "cnn_infer_failures": cnn_infer_failures,
    }
    save_json(run_dir / "hil_mock_summary.json", summary)
    return summary


def _arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run HIL suite scaffold (mock backend).")
    parser.add_argument("--config", required=True, type=str, help="Path to YAML config")
    return parser


def main() -> int:
    args = _arg_parser().parse_args()
    config = load_yaml_config(args.config)
    cfg_hash = config_hash(config)

    out_root = ensure_dir(get_path(config, "output_root", "runs"))
    name = str(config.get("experiment", {}).get("name", "hil_suite"))
    run_dir = ensure_dir(out_root / "hil_suite" / f"{name}_{now_tag()}_{cfg_hash}")

    # 中文注释：执行 mock HIL 并将摘要结果写入 run_dir。
    summary = _run_mock_hil(config, run_dir=run_dir)
    print("HIL suite complete (mock backend).")
    print(f"run_dir={run_dir}")
    print(f"slow_update_violation_rate={summary['slow_update_violation_rate']:.6e}")
    print(f"fast_cycle_violation_rate={summary['fast_cycle_violation_rate']:.6e}")
    print(f"dma_timeouts={summary['dma_timeouts']}, axi_write_failures={summary['axi_write_failures']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
