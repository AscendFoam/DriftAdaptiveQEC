from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cnn_fpga.decoder.param_mapper import NoisePrediction, ParamMapper  # noqa: E402
from cnn_fpga.runtime.param_bank import DecoderRuntimeParams  # noqa: E402
from docs.paper_materials.run_controlled_oracle_affine_analysis import (  # noqa: E402
    CONFIG,
    SCENARIOS,
    _sample_errors,
    _wrap,
)
from physics.constants import LATTICE_CONST  # noqa: E402


OUT_DIR = ROOT / "docs" / "paper_materials"
CSV_PATH = OUT_DIR / "submission_draft_finite_energy_channel_sanity.csv"
JSON_PATH = OUT_DIR / "submission_draft_finite_energy_channel_sanity.json"
REPORT_PATH = OUT_DIR / "投稿稿finite_energy_channel_sanity记录.md"

SEED = 20260706
N_SAMPLES = 60_000
FINITE_ENERGY_DELTAS = (0.18, 0.26, 0.34)
MEASUREMENT_EFFICIENCY = float(CONFIG["measurement"]["measurement_efficiency"])

METHODS = (
    "hard_nearest_syndrome",
    "fixed_affine",
    "oracle_affine",
)

METHOD_LABELS = {
    "hard_nearest_syndrome": "Hard nearest-syndrome",
    "fixed_affine": "Fixed affine",
    "oracle_affine": "Oracle affine",
}


def _measurement_std(delta: float) -> float:
    inefficiency_var = (1.0 - MEASUREMENT_EFFICIENCY) / (2.0 * MEASUREMENT_EFFICIENCY)
    return math.sqrt(delta**2 + inefficiency_var)


def _prediction(row: dict[str, float | str], source: str) -> NoisePrediction:
    return NoisePrediction(
        sigma=float(row["sigma"]),
        mu_q=float(row["mu_q"]),
        mu_p=float(row["mu_p"]),
        theta_deg=float(row["theta_deg"]),
        source=source,
        metadata={"finite_energy_channel_sanity": True},
    )


def _apply_params(syndromes: np.ndarray, params: DecoderRuntimeParams) -> np.ndarray:
    syndrome_limit = float(CONFIG["fast_loop"]["syndrome_limit"])
    correction_limit = float(CONFIG["fast_loop"]["correction_limit"])
    clipped = np.clip(syndromes, -syndrome_limit, syndrome_limit)
    raw = clipped @ params.K.T + params.b
    return np.clip(raw, -correction_limit, correction_limit)


def _summarize_residuals(
    *,
    delta: float,
    scenario: dict[str, float | str],
    method: str,
    residuals: np.ndarray,
) -> dict[str, object]:
    boundary = LATTICE_CONST / 2.0
    q_cross = np.abs(residuals[:, 0]) > boundary
    p_cross = np.abs(residuals[:, 1]) > boundary
    any_cross = np.logical_or(q_cross, p_cross)
    p_any = float(np.mean(any_cross))
    p_identity = 1.0 - p_any
    f_avg_surr = (1.0 + 2.0 * p_identity) / 3.0
    return {
        "finite_energy_delta": delta,
        "equivalent_modular_squeezing_db": -10.0 * math.log10(delta**2),
        "mean_photon_proxy": 1.0 / (2.0 * delta**2),
        "scenario": scenario["scenario"],
        "scenario_label": scenario["label"],
        "method": method,
        "method_label": METHOD_LABELS[method],
        "n_samples": N_SAMPLES,
        "logical_event_probability": p_any,
        "q_event_probability": float(np.mean(q_cross)),
        "p_event_probability": float(np.mean(p_cross)),
        "surrogate_average_fidelity": f_avg_surr,
        "residual_mse": float(np.mean(np.sum(residuals**2, axis=1))),
    }


def _scenario_rows() -> list[dict[str, object]]:
    rng = np.random.default_rng(SEED)
    mapper = ParamMapper.from_config(CONFIG)
    fixed_prediction = NoisePrediction(
        sigma=0.22,
        mu_q=0.0,
        mu_p=0.0,
        theta_deg=0.0,
        source="fixed_nominal",
        metadata={"finite_energy_channel_sanity": True},
    )
    fixed_params = mapper.map_prediction(fixed_prediction)
    hard_params = DecoderRuntimeParams.identity()

    rows: list[dict[str, object]] = []
    for delta in FINITE_ENERGY_DELTAS:
        measurement_std = _measurement_std(delta)
        for scenario in SCENARIOS:
            errors = _sample_errors(scenario, rng)
            syndromes = _wrap(errors) + rng.normal(0.0, measurement_std, size=errors.shape)
            params_by_method = {
                "hard_nearest_syndrome": hard_params,
                "fixed_affine": fixed_params,
                "oracle_affine": mapper.map_prediction(_prediction(scenario, "oracle_affine")),
            }
            for method, params in params_by_method.items():
                corrections = _apply_params(syndromes, params)
                rows.append(
                    _summarize_residuals(
                        delta=delta,
                        scenario=scenario,
                        method=method,
                        residuals=errors - corrections,
                    )
                )
    return rows


def _aggregate_rows(scenario_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for delta in FINITE_ENERGY_DELTAS:
        for method in METHODS:
            items = [
                row
                for row in scenario_rows
                if row["finite_energy_delta"] == delta and row["method"] == method
            ]
            if len(items) != len(SCENARIOS):
                raise ValueError(f"missing rows for delta={delta}, method={method}")
            worst = max(items, key=lambda row: float(row["logical_event_probability"]))
            rows.append(
                {
                    "finite_energy_delta": delta,
                    "equivalent_modular_squeezing_db": -10.0 * math.log10(delta**2),
                    "mean_photon_proxy": 1.0 / (2.0 * delta**2),
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "n_controlled_states": len(items),
                    "n_samples_per_state": N_SAMPLES,
                    "mean_logical_event_probability": sum(
                        float(row["logical_event_probability"]) for row in items
                    )
                    / len(items),
                    "worst_state": worst["scenario"],
                    "worst_state_logical_event_probability": worst["logical_event_probability"],
                    "mean_surrogate_average_fidelity": sum(
                        float(row["surrogate_average_fidelity"]) for row in items
                    )
                    / len(items),
                    "mean_residual_mse": sum(float(row["residual_mse"]) for row in items)
                    / len(items),
                    "non_claim_boundary": (
                        "not calibrated finite-energy GKP logical-channel fidelity; "
                        "finite-squeezing toy-channel sanity only; not process "
                        "tomography; not a formal benchmark; not hardware evidence"
                    ),
                }
            )
    return rows


def _write_csv(rows: list[dict[str, object]]) -> None:
    fields = [
        "finite_energy_delta",
        "equivalent_modular_squeezing_db",
        "mean_photon_proxy",
        "method",
        "method_label",
        "n_controlled_states",
        "n_samples_per_state",
        "mean_logical_event_probability",
        "worst_state",
        "worst_state_logical_event_probability",
        "mean_surrogate_average_fidelity",
        "mean_residual_mse",
        "non_claim_boundary",
    ]
    with CSV_PATH.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(aggregate_rows: list[dict[str, object]], scenario_rows: list[dict[str, object]]) -> None:
    payload = {
        "status": "generated",
        "analysis_id": "submission_draft_finite_energy_channel_sanity_v1",
        "seed": SEED,
        "n_samples_per_state": N_SAMPLES,
        "finite_energy_deltas": FINITE_ENERGY_DELTAS,
        "measurement_efficiency": MEASUREMENT_EFFICIENCY,
        "formulae": {
            "syndrome": "wrap(error) + Normal(0, sqrt(delta^2 + (1-eta)/(2 eta)))",
            "logical_event_probability": "Pr(|residual_q| > lambda/2 or |residual_p| > lambda/2)",
            "surrogate_average_fidelity": "(1 + 2*(1-p_any))/3",
            "mean_photon_proxy": "1/(2 delta^2)",
        },
        "scope": (
            "Toy finite-squeezing measurement-channel sanity check over the same "
            "controlled local-Gaussian states used elsewhere in the submission draft. "
            "It uses the fast-path affine parameter mapper and half-lattice logical "
            "event rule, but it is not a calibrated finite-energy physical device "
            "channel and not process tomography."
        ),
        "non_claims": [
            "not finite-energy GKP logical-channel tomography",
            "not process fidelity",
            "not a calibrated physical-device channel",
            "not a formal software-HIL benchmark",
            "not hardware evidence",
            "not inferential statistics",
        ],
        "aggregate_rows": aggregate_rows,
        "scenario_rows": scenario_rows,
    }
    JSON_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_report(rows: list[dict[str, object]]) -> None:
    lines = [
        "# 投稿稿 finite-energy channel sanity 记录",
        "",
        "日期：2026-07-06",
        "",
        "本记录服务于 `docs/paper_notes/CNN_FPGA_GKP_submission_draft.tex`。它在同一组 controlled local-Gaussian states 上加入一个 finite-squeezing measurement-channel toy sanity check，用来缩小 residual-boundary surrogate 与 finite-energy logical-channel 之间的解释缺口。",
        "",
        "## 生成文件",
        "",
        f"- `{CSV_PATH.relative_to(ROOT)}`",
        f"- `{JSON_PATH.relative_to(ROOT)}`",
        "",
        "## 协议",
        "",
        f"- random seed: `{SEED}`",
        f"- samples per state: `{N_SAMPLES}`",
        f"- finite-energy delta values: `{', '.join(str(v) for v in FINITE_ENERGY_DELTAS)}`",
        "- syndrome model: `wrap(error) + Normal(0, sqrt(delta^2 + (1-eta)/(2 eta)))`",
        "- compared methods: hard nearest-syndrome, fixed affine, oracle affine",
        "- metric: q/p half-lattice logical-event probability plus surrogate average-fidelity readout",
        "",
        "## 聚合结果",
        "",
        "| delta | method | mean p_any | worst state | worst p_any | mean F_avg^surr |",
        "| ---: | --- | ---: | --- | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {float(row['finite_energy_delta']):.2f} | "
            f"{row['method_label']} | "
            f"{float(row['mean_logical_event_probability']):.6f} | "
            f"`{row['worst_state']}` | "
            f"{float(row['worst_state_logical_event_probability']):.6f} | "
            f"{float(row['mean_surrogate_average_fidelity']):.6f} |"
        )
    lines.extend(
        [
            "",
            "## 可写边界",
            "",
            "- 可以写：该 sanity check 在一个简化 finite-squeezing measurement channel 下比较了 hard nearest-syndrome、fixed affine 和 oracle affine 的 half-lattice logical-event probability。",
            "- 可以写：该表比纯 residual-boundary surrogate 更接近 approximate-GKP channel language，但仍然只是 toy-channel bridge。",
            "- 不能写：该表完成了 finite-energy GKP logical-channel tomography、process fidelity、真实物理器件 calibration、正式 software-HIL benchmark、统计显著性或硬件验证。",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    scenario_rows = _scenario_rows()
    aggregate_rows = _aggregate_rows(scenario_rows)
    _write_csv(aggregate_rows)
    _write_json(aggregate_rows, scenario_rows)
    _write_report(aggregate_rows)
    print(
        json.dumps(
            {
                "status": "generated",
                "rows": len(aggregate_rows),
                "csv": str(CSV_PATH),
                "json": str(JSON_PATH),
                "report": str(REPORT_PATH),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
