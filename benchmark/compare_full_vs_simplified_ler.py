#!/usr/bin/env python3
"""Minimal benchmark for full_qec vs simplified logical tracking models."""

# 中文说明：
# - 该脚本用于直接比较 full_qec 与 simplified 两种模型的累计 LER 曲线差异。
# - 输出包含 CSV/JSON（以及可选 PNG），便于后续报告与回归测试。

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


def _setup_import_path() -> Path:
    """Ensure repository root is available for local imports."""
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return repo_root


REPO_ROOT = _setup_import_path()

from physics.logical_tracking import simulate_error_accumulation  # noqa: E402


def _curve_from_history(history) -> np.ndarray:
    """Convert per-round history to cumulative LER curve."""
    if not history:
        return np.array([], dtype=float)

    per_round_err = np.fromiter(
        (1.0 if (item["x_error"] or item["z_error"]) else 0.0 for item in history),
        dtype=float,
        count=len(history),
    )
    # 中文注释：把每轮错误标记转成“到当前轮为止的累计平均错误率”。
    return np.cumsum(per_round_err) / np.arange(1, per_round_err.size + 1, dtype=float)


def _run_mode(args: argparse.Namespace, use_full_qec_model: bool) -> Tuple[np.ndarray, np.ndarray]:
    """Run repeated simulations for one model mode."""
    curves = []
    final_rates = []

    for rep in range(args.repeats):
        # 中文注释：每次重复使用不同 seed，避免单次随机性主导结果。
        seed = args.seed_base + rep
        result = simulate_error_accumulation(
            n_rounds=args.n_rounds,
            sigma_error=args.sigma_error,
            sigma_measurement=args.sigma_measurement,
            gain=args.gain,
            delta=args.delta,
            measurement_efficiency=args.measurement_efficiency,
            ancilla_error_rate=args.ancilla_error_rate,
            add_shot_noise=(not args.no_shot_noise),
            sigma_error_p=args.sigma_error_p,
            theta=args.theta,
            error_bias=np.array([args.mu_q, args.mu_p], dtype=float),
            use_full_qec_model=use_full_qec_model,
            return_history=True,
            seed=seed,
        )

        curve = _curve_from_history(result.get("history", []))
        if curve.size != args.n_rounds:
            raise RuntimeError(
                f"Unexpected curve length {curve.size}, expected {args.n_rounds}."
            )

        curves.append(curve)
        final_rates.append(float(result["total_error_rate"]))

    return np.vstack(curves), np.asarray(final_rates, dtype=float)


def _write_curve_csv(path: Path,
                     rounds: np.ndarray,
                     full_mean: np.ndarray,
                     full_std: np.ndarray,
                     simplified_mean: np.ndarray,
                     simplified_std: np.ndarray,
                     gap_mean: np.ndarray,
                     gap_std: np.ndarray) -> None:
    """Write per-round statistics to CSV."""
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "round",
            "full_mean",
            "full_std",
            "simplified_mean",
            "simplified_std",
            "gap_mean",
            "gap_std",
        ])
        for i in range(rounds.size):
            writer.writerow([
                int(rounds[i]),
                float(full_mean[i]),
                float(full_std[i]),
                float(simplified_mean[i]),
                float(simplified_std[i]),
                float(gap_mean[i]),
                float(gap_std[i]),
            ])


def _plot_curves(path: Path,
                 rounds: np.ndarray,
                 full_mean: np.ndarray,
                 full_std: np.ndarray,
                 simplified_mean: np.ndarray,
                 simplified_std: np.ndarray,
                 gap_mean: np.ndarray,
                 gap_std: np.ndarray) -> bool:
    """Plot and save curve figure. Returns True if plot was saved."""
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax = axes[0]
    ax.plot(rounds, full_mean, label="full_qec", linewidth=2)
    ax.fill_between(rounds, full_mean - full_std, full_mean + full_std, alpha=0.2)

    ax.plot(rounds, simplified_mean, label="simplified", linewidth=2)
    ax.fill_between(
        rounds,
        simplified_mean - simplified_std,
        simplified_mean + simplified_std,
        alpha=0.2,
    )

    ax.set_ylabel("Cumulative LER")
    ax.set_title("Full QEC vs Simplified: Cumulative LER")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax2 = axes[1]
    ax2.plot(rounds, gap_mean, label="gap = full - simplified", color="tab:red", linewidth=2)
    ax2.fill_between(rounds, gap_mean - gap_std, gap_mean + gap_std, color="tab:red", alpha=0.2)
    ax2.axhline(0.0, color="black", linewidth=1, alpha=0.4)
    ax2.set_xlabel("Round")
    ax2.set_ylabel("LER Gap")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare cumulative LER curves between full_qec and simplified modes."
    )

    parser.add_argument("--n-rounds", type=int, default=2000)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed-base", type=int, default=1234)

    parser.add_argument("--sigma-error", type=float, default=0.25)
    parser.add_argument("--sigma-error-p", type=float, default=None)
    parser.add_argument("--sigma-measurement", type=float, default=0.05)
    parser.add_argument("--gain", type=float, default=0.9)

    parser.add_argument("--delta", type=float, default=0.3)
    parser.add_argument("--measurement-efficiency", type=float, default=0.95)
    parser.add_argument("--ancilla-error-rate", type=float, default=0.01)
    parser.add_argument("--no-shot-noise", action="store_true")

    parser.add_argument("--theta", type=float, default=0.0)
    parser.add_argument("--mu-q", type=float, default=0.0)
    parser.add_argument("--mu-p", type=float, default=0.0)

    parser.add_argument("--output-dir", type=Path, default=Path("benchmark/results"))
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--no-plot", action="store_true")

    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.n_rounds <= 0:
        raise ValueError("--n-rounds must be positive")
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    full_curves, full_finals = _run_mode(args, use_full_qec_model=True)
    simple_curves, simple_finals = _run_mode(args, use_full_qec_model=False)

    rounds = np.arange(1, args.n_rounds + 1, dtype=int)

    full_mean = full_curves.mean(axis=0)
    full_std = full_curves.std(axis=0)

    simple_mean = simple_curves.mean(axis=0)
    simple_std = simple_curves.std(axis=0)

    gap_curves = full_curves - simple_curves
    gap_mean = gap_curves.mean(axis=0)
    gap_std = gap_curves.std(axis=0)

    tag = args.tag.strip() if args.tag else f"n{args.n_rounds}_r{args.repeats}_s{args.sigma_error:.3f}"
    csv_path = args.output_dir / f"{tag}_ler_curve_compare.csv"
    json_path = args.output_dir / f"{tag}_summary.json"
    plot_path = args.output_dir / f"{tag}_ler_curve_compare.png"

    _write_curve_csv(
        csv_path,
        rounds,
        full_mean,
        full_std,
        simple_mean,
        simple_std,
        gap_mean,
        gap_std,
    )

    plot_saved = False
    if not args.no_plot:
        plot_saved = _plot_curves(
            plot_path,
            rounds,
            full_mean,
            full_std,
            simple_mean,
            simple_std,
            gap_mean,
            gap_std,
        )

    summary: Dict[str, object] = {
        "n_rounds": args.n_rounds,
        "repeats": args.repeats,
        "seed_base": args.seed_base,
        "params": {
            "sigma_error": args.sigma_error,
            "sigma_error_p": args.sigma_error_p,
            "sigma_measurement": args.sigma_measurement,
            "gain": args.gain,
            "delta": args.delta,
            "measurement_efficiency": args.measurement_efficiency,
            "ancilla_error_rate": args.ancilla_error_rate,
            "add_shot_noise": not args.no_shot_noise,
            "theta": args.theta,
            "mu_q": args.mu_q,
            "mu_p": args.mu_p,
        },
        "full_final_ler_mean": float(full_finals.mean()),
        "full_final_ler_std": float(full_finals.std()),
        "simplified_final_ler_mean": float(simple_finals.mean()),
        "simplified_final_ler_std": float(simple_finals.std()),
        "final_gap_mean": float((full_finals - simple_finals).mean()),
        "final_gap_std": float((full_finals - simple_finals).std()),
        "auc_gap_mean": float(gap_mean.mean()),
        "auc_gap_std": float(gap_std.mean()),
        "artifacts": {
            "csv": str(csv_path),
            "json": str(json_path),
            "plot": str(plot_path) if plot_saved else None,
        },
    }

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Benchmark complete.")
    print(f"full_final_ler_mean={summary['full_final_ler_mean']:.6f}")
    print(f"simplified_final_ler_mean={summary['simplified_final_ler_mean']:.6f}")
    print(f"final_gap_mean={summary['final_gap_mean']:.6f}")
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")
    if plot_saved:
        print(f"Plot: {plot_path}")
    else:
        if args.no_plot:
            print("Plot skipped (--no-plot).")
        else:
            print("Plot not generated (matplotlib unavailable).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
