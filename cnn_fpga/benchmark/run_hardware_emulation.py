"""Run P2 hardware-behavior emulation using the dual-loop runtime scaffold."""

# 中文说明：
# - 该入口用于 P2“硬件行为仿真”，目标是验证快慢回路、窗口、stage/commit 和延迟预算是否协同正常。
# - 它不同于 HIL mock：这里不强调板级 I/O，而是强调运行时链路与物理闭环的行为正确性。

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np

from cnn_fpga.runtime import DualLoopScheduler, FastLoopEmulator, ParamBank, SlowLoopRuntime
from cnn_fpga.utils.config import config_hash, ensure_dir, get_path, load_yaml_config, now_tag, save_json


def _build_noise_provider(config: Dict, scenario: Dict, seed: int):
    model_cfg = config.get("model", {})
    base_mu_q = float(model_cfg.get("mu_q", 0.0))
    base_mu_p = float(model_cfg.get("mu_p", 0.0))
    base_theta_deg = float(model_cfg.get("theta_deg", 0.0))

    params = scenario.get("params", {})
    kind = str(scenario.get("type", "linear")).lower()
    rng = np.random.default_rng(seed)

    sigma_state = float(params.get("sigma0", 0.2))

    def _provider(epoch_id: int, _time_us: float) -> Dict:
        nonlocal sigma_state
        t = float(epoch_id - 1)

        if kind == "linear":
            sigma_value = sigma_state + float(params.get("alpha", 0.0)) * t
        elif kind == "step":
            sigma_value = sigma_state
            if epoch_id >= int(params.get("jump_round", 1000)):
                sigma_value += float(params.get("delta_sigma", 0.1))
        elif kind in {"sin", "sine", "periodic"}:
            sigma_value = sigma_state + float(params.get("amplitude", 0.05)) * np.sin(
                2.0 * np.pi * float(params.get("frequency", 0.001)) * t + float(params.get("phase", 0.0))
            )
        elif kind in {"random_walk", "rw"}:
            if epoch_id > 1:
                sigma_state = float(sigma_state + rng.normal(0.0, float(params.get("step_std", 0.002))))
            sigma_value = sigma_state
        else:
            raise ValueError(f"Unsupported drift type: {scenario.get('type')}")

        sigma_value = float(
            np.clip(
                sigma_value,
                float(params.get("sigma_min", 0.05)),
                float(params.get("sigma_max", 0.70)),
            )
        )
        return {
            "sigma": sigma_value,
            "mu_q": float(params.get("mu_q", base_mu_q)),
            "mu_p": float(params.get("mu_p", base_mu_p)),
            "theta_deg": float(params.get("theta_deg", base_theta_deg)),
            "metadata": {
                "scenario": str(scenario.get("name", scenario.get("type", "scenario"))),
                "type": kind,
            },
        }

    return _provider


def _required_fast_cycles(config: Dict, scheduler: DualLoopScheduler) -> int:
    timing_cfg = config.get("timing", {})
    requested_fast_cycles = int(timing_cfg.get("n_fast_cycles", config.get("simulation", {}).get("n_rounds", 4000)))
    requested_slow_updates = int(timing_cfg.get("n_slow_updates", 0))
    if requested_slow_updates <= 0:
        return requested_fast_cycles

    cycles_per_slow = int(np.ceil(scheduler.config.slow_update_period_us / scheduler.config.t_fast_us))
    cycles_for_window_target = scheduler.config.window_size + max(0, requested_slow_updates - 1) * scheduler.window_stride
    cycles_for_slow_target = requested_slow_updates * cycles_per_slow
    return max(requested_fast_cycles, cycles_for_window_target, cycles_for_slow_target)


def _write_ler_curve(path: Path, curve: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["round", "cumulative_ler"])
        for idx, value in enumerate(curve, start=1):
            writer.writerow([idx, float(value)])


def _run_repeat(config: Dict, scenario: Dict, repeat_idx: int, run_dir: Path, seed: int) -> Dict:
    param_bank = ParamBank()
    slow_loop = SlowLoopRuntime.from_config(config, seed=seed + 101)
    scheduler = DualLoopScheduler.from_config(config, param_bank=param_bank, slow_path_fn=slow_loop)
    fast_loop = FastLoopEmulator.from_config(
        config,
        param_bank=param_bank,
        noise_provider=_build_noise_provider(config, scenario, seed=seed + 7),
        seed=seed + 13,
    )

    n_cycles = _required_fast_cycles(config, scheduler)
    events = scheduler.run(n_cycles, fast_path_fn=fast_loop)
    event_counts = Counter(event.kind for event in events)
    fast_summary = fast_loop.summary()
    ler_curve = fast_loop.get_ler_curve()

    scenario_name = str(scenario.get("name", scenario.get("type", "scenario")))
    curve_path = run_dir / f"{scenario_name}_repeat{repeat_idx:02d}_ler.csv"
    _write_ler_curve(curve_path, ler_curve)

    summary = {
        "scenario": scenario_name,
        "repeat": repeat_idx,
        "seed": seed,
        "n_fast_cycles": n_cycles,
        "n_windows_ready": int(event_counts.get("window_ready", 0)),
        "n_slow_updates_started": int(event_counts.get("slow_update_started", 0)),
        "n_slow_updates_finished": int(event_counts.get("slow_update_finished", 0)),
        "n_slow_updates_failed": int(event_counts.get("slow_update_failed", 0)),
        "n_commits_applied": int(event_counts.get("commit_applied", 0)),
        "window_drop_count": scheduler.dropped_windows,
        "pending_windows_at_end": scheduler.pending_windows,
        "fast_cycle_violation_rate": float(scheduler.fast_cycle_budget_violations / max(1, n_cycles)),
        "slow_update_violation_rate": float(
            scheduler.slow_update_budget_violations / max(1, event_counts.get("slow_update_started", 0))
        ),
        "final_ler": float(fast_summary["final_logical_error_rate"]),
        "total_logical_errors": int(fast_summary["tracker"]["total_logical_errors"]),
        "overflow_rate": float(fast_summary["overflow_rate"]),
        "final_active_version": int(scheduler.param_bank.active_version),
        "ler_curve_csv": str(curve_path),
        "slow_loop_mode": str(config.get("slow_loop", {}).get("mode", "mock")),
        "artifact_path": None
        if getattr(slow_loop, "artifact_predictor", None) is None
        else str(slow_loop.artifact_predictor.artifact_path),
        "fast_loop_summary": fast_summary,
        "final_snapshot": scheduler.snapshot(),
    }
    save_json(run_dir / f"{scenario_name}_repeat{repeat_idx:02d}_summary.json", summary)
    return summary


def _aggregate_summaries(config: Dict, run_dir: Path, summaries: List[Dict]) -> Dict:
    aggregated: Dict[str, List[Dict]] = {}
    for item in summaries:
        aggregated.setdefault(item["scenario"], []).append(item)

    scenario_reports = []
    for scenario_name, items in aggregated.items():
        final_ler = np.asarray([entry["final_ler"] for entry in items], dtype=float)
        overflow_rate = np.asarray([entry["overflow_rate"] for entry in items], dtype=float)
        commits = np.asarray([entry["n_commits_applied"] for entry in items], dtype=float)
        scenario_reports.append(
            {
                "scenario": scenario_name,
                "repeats": len(items),
                "final_ler_mean": float(np.mean(final_ler)),
                "final_ler_std": float(np.std(final_ler)),
                "overflow_rate_mean": float(np.mean(overflow_rate)),
                "commit_count_mean": float(np.mean(commits)),
                "repeat_summaries": [str(run_dir / f"{scenario_name}_repeat{entry['repeat']:02d}_summary.json") for entry in items],
            }
        )

    top_summary = {
        "config_hash": config_hash(config),
        "run_dir": str(run_dir),
        "stage": str(config.get("experiment", {}).get("stage", "p2_emulation")),
        "slow_loop_mode": str(config.get("slow_loop", {}).get("mode", "mock")),
        "scenarios": scenario_reports,
    }
    save_json(run_dir / "summary.json", top_summary)
    return top_summary


def _arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run hardware-behavior emulation suite.")
    parser.add_argument("--config", required=True, type=str, help="Path to YAML config")
    return parser


def main() -> int:
    args = _arg_parser().parse_args()
    config = load_yaml_config(args.config)
    experiment = config.get("experiment", {})
    cfg_hash = config_hash(config)
    seed_base = int(experiment.get("seed", 1234))
    repeats = int(config.get("simulation", {}).get("repeats", 1))
    scenarios = config.get("drift", {}).get("scenarios", [])
    if not scenarios:
        raise ValueError("No drift scenarios configured for hardware emulation.")

    out_root = ensure_dir(get_path(config, "output_root", "runs"))
    run_dir = ensure_dir(out_root / "hardware_emulation" / f"{experiment.get('name', 'hardware_emulation')}_{now_tag()}_{cfg_hash}")

    all_summaries: List[Dict] = []
    for scenario_idx, scenario in enumerate(scenarios):
        scenario_name = str(scenario.get("name", scenario.get("type", "scenario")))
        for repeat_idx in range(repeats):
            seed = seed_base + scenario_idx * 1000 + repeat_idx
            summary = _run_repeat(config, scenario, repeat_idx, run_dir, seed)
            all_summaries.append(summary)
            print(
                f"[{scenario_name}][rep={repeat_idx:02d}] "
                f"LER={summary['final_ler']:.6f}, commits={summary['n_commits_applied']}, "
                f"overflow={summary['overflow_rate']:.6f}"
            )

    aggregated = _aggregate_summaries(config, run_dir, all_summaries)
    print("Hardware emulation complete.")
    print(f"run_dir={run_dir}")
    for item in aggregated["scenarios"]:
        print(
            f"{item['scenario']}: final_ler_mean={item['final_ler_mean']:.6f}, "
            f"final_ler_std={item['final_ler_std']:.6f}, "
            f"commit_count_mean={item['commit_count_mean']:.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
