"""Export and summarize per-window trace rows from a bounded T38 benchmark rerun."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Sequence


TRACE_FIELDS: List[str] = [
    "scenario",
    "scenario_label",
    "mode",
    "mode_label",
    "repeat",
    "seed",
    "window_id",
    "started_epoch",
    "completed_epoch",
    "commit_epoch",
    "commit_target_bank",
    "commit_version",
    "runtime_mode",
    "teacher_prediction_sigma",
    "teacher_prediction_mu_q",
    "teacher_prediction_mu_p",
    "teacher_b_q",
    "teacher_b_p",
    "raw_delta_b_q",
    "raw_delta_b_p",
    "delta_b_q",
    "delta_b_p",
    "committed_b_q",
    "committed_b_p",
    "committed_minus_teacher_b_q",
    "committed_minus_teacher_b_p",
    "window_ler",
    "mean_correction_utilization",
    "mean_correction_norm",
    "overflow_ratio",
    "overflow_any_ratio",
    "histogram_input_saturation_ratio",
    "decoder_input_saturation_ratio",
    "correction_saturation_ratio",
    "correction_clip_saturation_ratio",
    "correction_fixed_point_saturation_ratio",
    "aggressive_param_ratio",
    "aggressive_param_correction_ratio",
    "dominant_overflow_source",
    "teacher_diag_status",
    "teacher_contribution_l2",
    "prediction_without_teacher_b_q",
    "prediction_without_teacher_b_p",
]


FIELD_SPECS: List[Dict[str, str]] = [
    {"field": "scenario", "source": "summary.raw_rows[].scenario", "required": "yes"},
    {"field": "mode", "source": "summary.raw_rows[].mode", "required": "yes"},
    {"field": "repeat", "source": "summary.raw_rows[].repeat", "required": "yes"},
    {"field": "seed", "source": "summary.raw_rows[].seed", "required": "yes"},
    {"field": "window_id", "source": "host_events[].readout.window.window_id", "required": "yes"},
    {"field": "teacher_b_q", "source": "host_events[].proposed_params.metadata.teacher_params.b[0]", "required": "yes"},
    {"field": "teacher_b_p", "source": "host_events[].proposed_params.metadata.teacher_params.b[1]", "required": "yes"},
    {"field": "delta_b_q", "source": "host_events[].proposed_params.metadata.applied_delta_b[0]", "required": "yes"},
    {"field": "delta_b_p", "source": "host_events[].proposed_params.metadata.applied_delta_b[1]", "required": "yes"},
    {"field": "committed_b_q", "source": "host_events[].proposed_params.b[0]", "required": "yes"},
    {"field": "committed_b_p", "source": "host_events[].proposed_params.b[1]", "required": "yes"},
    {"field": "commit_target_bank", "source": "host_events[].commit.target_bank", "required": "yes"},
    {"field": "commit_epoch", "source": "host_events[].commit.commit_epoch", "required": "yes"},
    {"field": "commit_version", "source": "host_events[].commit.version", "required": "yes"},
    {"field": "window_ler", "source": "host_events[].proposed_params.metadata.window_diagnostics.window_ler", "required": "yes"},
    {
        "field": "mean_correction_utilization",
        "source": "host_events[].proposed_params.metadata.window_diagnostics.mean_correction_utilization",
        "required": "yes",
    },
    {
        "field": "overflow_ratio",
        "source": "host_events[].proposed_params.metadata.window_diagnostics.overflow_ratio",
        "required": "yes",
    },
    {
        "field": "correction_saturation_ratio",
        "source": "host_events[].proposed_params.metadata.window_diagnostics.correction_saturation_ratio",
        "required": "yes",
    },
    {
        "field": "dominant_overflow_source",
        "source": "host_events[].proposed_params.metadata.window_diagnostics.dominant_overflow_source",
        "required": "yes",
    },
]


def _arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=str, help="Benchmark run dir containing summary.json and repeat hil_events.json files.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional output dir. Defaults to <run-dir>/trace_export.",
    )
    return parser


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _ensure_output_dir(run_dir: Path, output_dir_arg: str | None) -> Path:
    output_dir = Path(output_dir_arg).expanduser().resolve() if output_dir_arg else (run_dir / "trace_export")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _pair_at(values: Any, index: int) -> float | None:
    if not isinstance(values, Sequence) or len(values) <= index:
        return None
    item = values[index]
    if item is None:
        return None
    return float(item)


def _norm2(*values: Any) -> float | None:
    parts: List[float] = []
    for value in values:
        if value is None:
            return None
        parts.append(float(value))
    return math.sqrt(sum(part * part for part in parts))


def _non_null(values: Iterable[float | None]) -> List[float]:
    return [float(value) for value in values if value is not None]


def _sign_flips(values: Sequence[float | None]) -> int:
    flips = 0
    previous_sign = 0
    for value in values:
        if value is None:
            continue
        sign = 1 if value > 0 else -1 if value < 0 else 0
        if sign == 0:
            continue
        if previous_sign != 0 and sign != previous_sign:
            flips += 1
        previous_sign = sign
    return flips


def _window_trace_rows(summary: Mapping[str, Any]) -> List[Dict[str, Any]]:
    trace_rows: List[Dict[str, Any]] = []
    for raw_row in summary.get("raw_rows", []):
        repeat_run_dir = Path(str(raw_row["run_dir"])).expanduser().resolve()
        hil_events_path = repeat_run_dir / "hil_events.json"
        payload = _read_json(hil_events_path)
        host_events = payload.get("host_events", [])
        for event in host_events:
            if event.get("kind") != "slow_update_finished":
                continue
            metadata = dict(event.get("proposed_params", {}).get("metadata", {}))
            teacher_prediction = dict(metadata.get("teacher_prediction", {}))
            teacher_params = dict(metadata.get("teacher_params", {}))
            residual_prediction = dict(metadata.get("residual_prediction", {}))
            residual_meta = dict(residual_prediction.get("metadata", {}))
            raw_prediction = dict(residual_meta.get("raw_prediction", {}))
            window_diag = dict(metadata.get("window_diagnostics", {}))
            commit = dict(event.get("commit", {}))
            prediction_without_teacher = residual_meta.get("teacher_branch_diagnostics", {}) or metadata.get("teacher_branch_diagnostics", {})
            teacher_diag = dict(prediction_without_teacher)
            teacher_b_q = _pair_at(teacher_params.get("b"), 0)
            teacher_b_p = _pair_at(teacher_params.get("b"), 1)
            delta_b_q = _pair_at(metadata.get("applied_delta_b"), 0)
            delta_b_p = _pair_at(metadata.get("applied_delta_b"), 1)
            committed_b_q = _pair_at(event.get("proposed_params", {}).get("b"), 0)
            committed_b_p = _pair_at(event.get("proposed_params", {}).get("b"), 1)
            predicted_without_teacher = teacher_diag.get("prediction_without_teacher")
            trace_rows.append(
                {
                    "scenario": raw_row["scenario"],
                    "scenario_label": raw_row.get("scenario_label", raw_row["scenario"]),
                    "mode": raw_row["mode"],
                    "mode_label": raw_row.get("mode_label", raw_row["mode"]),
                    "repeat": int(raw_row["repeat"]),
                    "seed": int(raw_row["seed"]),
                    "window_id": int(event.get("readout", {}).get("window", {}).get("window_id")),
                    "started_epoch": int(event.get("started_epoch")),
                    "completed_epoch": int(event.get("completed_epoch")),
                    "commit_epoch": int(commit.get("commit_epoch")) if commit.get("commit_epoch") is not None else None,
                    "commit_target_bank": commit.get("target_bank"),
                    "commit_version": int(commit.get("version")) if commit.get("version") is not None else None,
                    "runtime_mode": metadata.get("runtime_mode"),
                    "teacher_prediction_sigma": _as_float(teacher_prediction.get("sigma")),
                    "teacher_prediction_mu_q": _as_float(teacher_prediction.get("mu_q")),
                    "teacher_prediction_mu_p": _as_float(teacher_prediction.get("mu_p")),
                    "teacher_b_q": teacher_b_q,
                    "teacher_b_p": teacher_b_p,
                    "raw_delta_b_q": _as_float(raw_prediction.get("b_q", raw_prediction.get("mu_q"))),
                    "raw_delta_b_p": _as_float(raw_prediction.get("b_p", raw_prediction.get("mu_p"))),
                    "delta_b_q": delta_b_q,
                    "delta_b_p": delta_b_p,
                    "committed_b_q": committed_b_q,
                    "committed_b_p": committed_b_p,
                    "committed_minus_teacher_b_q": None if committed_b_q is None or teacher_b_q is None else committed_b_q - teacher_b_q,
                    "committed_minus_teacher_b_p": None if committed_b_p is None or teacher_b_p is None else committed_b_p - teacher_b_p,
                    "window_ler": _as_float(window_diag.get("window_ler")),
                    "mean_correction_utilization": _as_float(window_diag.get("mean_correction_utilization")),
                    "mean_correction_norm": _as_float(window_diag.get("mean_correction_norm")),
                    "overflow_ratio": _as_float(window_diag.get("overflow_ratio")),
                    "overflow_any_ratio": _as_float(window_diag.get("overflow_any_ratio")),
                    "histogram_input_saturation_ratio": _as_float(window_diag.get("histogram_input_saturation_ratio")),
                    "decoder_input_saturation_ratio": _as_float(window_diag.get("decoder_input_saturation_ratio")),
                    "correction_saturation_ratio": _as_float(window_diag.get("correction_saturation_ratio")),
                    "correction_clip_saturation_ratio": _as_float(window_diag.get("correction_clip_saturation_ratio")),
                    "correction_fixed_point_saturation_ratio": _as_float(window_diag.get("correction_fixed_point_saturation_ratio")),
                    "aggressive_param_ratio": _as_float(window_diag.get("aggressive_param_ratio")),
                    "aggressive_param_correction_ratio": _as_float(window_diag.get("aggressive_param_correction_ratio")),
                    "dominant_overflow_source": window_diag.get("dominant_overflow_source"),
                    "teacher_diag_status": teacher_diag.get("teacher_diagnostics_status"),
                    "teacher_contribution_l2": _as_float(teacher_diag.get("teacher_contribution_l2")),
                    "prediction_without_teacher_b_q": _pair_at(predicted_without_teacher, 0),
                    "prediction_without_teacher_b_p": _pair_at(predicted_without_teacher, 1),
                }
            )
    trace_rows.sort(key=lambda item: (str(item["scenario"]), str(item["mode"]), int(item["repeat"]), int(item["window_id"])))
    return trace_rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def _field_availability(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    total = len(rows)
    availability: List[Dict[str, Any]] = []
    for spec in FIELD_SPECS:
        field = spec["field"]
        present = sum(1 for row in rows if row.get(field) not in (None, ""))
        availability.append(
            {
                **spec,
                "present_count": present,
                "total_count": total,
                "availability_ratio": None if total == 0 else float(present / total),
                "status": "present" if present == total else "partial" if present > 0 else "missing",
            }
        )
    return availability


def _repeat_summary(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    buckets: Dict[tuple[str, str, int], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[(str(row["scenario"]), str(row["mode"]), int(row["repeat"]))].append(row)

    summary_rows: List[Dict[str, Any]] = []
    for (scenario, mode, repeat), bucket in sorted(buckets.items()):
        bucket = sorted(bucket, key=lambda item: int(item["window_id"]))
        teacher_b_q = [row.get("teacher_b_q") for row in bucket]
        teacher_b_p = [row.get("teacher_b_p") for row in bucket]
        delta_b_q = [row.get("delta_b_q") for row in bucket]
        delta_b_p = [row.get("delta_b_p") for row in bucket]
        committed_b_q = [row.get("committed_b_q") for row in bucket]
        committed_b_p = [row.get("committed_b_p") for row in bucket]
        window_lers = _non_null(row.get("window_ler") for row in bucket)
        correction_utilization = _non_null(row.get("mean_correction_utilization") for row in bucket)
        teacher_contrib = _non_null(row.get("teacher_contribution_l2") for row in bucket)

        teacher_norms = _non_null(_norm2(q, p) for q, p in zip(teacher_b_q, teacher_b_p))
        delta_norms = _non_null(_norm2(q, p) for q, p in zip(delta_b_q, delta_b_p))
        committed_norms = _non_null(_norm2(q, p) for q, p in zip(committed_b_q, committed_b_p))

        summary_rows.append(
            {
                "scenario": scenario,
                "scenario_label": bucket[0]["scenario_label"],
                "mode": mode,
                "mode_label": bucket[0]["mode_label"],
                "repeat": repeat,
                "seed": int(bucket[0]["seed"]),
                "n_windows": len(bucket),
                "first_window_id": int(bucket[0]["window_id"]),
                "last_window_id": int(bucket[-1]["window_id"]),
                "final_window_ler": window_lers[-1] if window_lers else None,
                "mean_window_ler": None if not window_lers else float(mean(window_lers)),
                "max_window_ler": None if not window_lers else float(max(window_lers)),
                "mean_correction_utilization": None if not correction_utilization else float(mean(correction_utilization)),
                "max_abs_teacher_b": None if not teacher_norms else float(max(teacher_norms)),
                "final_abs_teacher_b": None if not teacher_norms else float(teacher_norms[-1]),
                "max_abs_delta_b": None if not delta_norms else float(max(delta_norms)),
                "final_abs_delta_b": None if not delta_norms else float(delta_norms[-1]),
                "max_abs_committed_b": None if not committed_norms else float(max(committed_norms)),
                "final_abs_committed_b": None if not committed_norms else float(committed_norms[-1]),
                "teacher_b_q_sign_flips": _sign_flips(teacher_b_q),
                "teacher_b_p_sign_flips": _sign_flips(teacher_b_p),
                "delta_b_q_sign_flips": _sign_flips(delta_b_q),
                "delta_b_p_sign_flips": _sign_flips(delta_b_p),
                "committed_b_q_sign_flips": _sign_flips(committed_b_q),
                "committed_b_p_sign_flips": _sign_flips(committed_b_p),
                "teacher_contribution_l2_mean": None if not teacher_contrib else float(mean(teacher_contrib)),
            }
        )
    return summary_rows


def _scenario_mode_summary(repeat_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    buckets: Dict[tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in repeat_rows:
        buckets[(str(row["scenario"]), str(row["mode"]))].append(row)

    metric_names = [
        "final_window_ler",
        "mean_window_ler",
        "max_window_ler",
        "mean_correction_utilization",
        "max_abs_teacher_b",
        "final_abs_teacher_b",
        "max_abs_delta_b",
        "final_abs_delta_b",
        "max_abs_committed_b",
        "final_abs_committed_b",
        "teacher_contribution_l2_mean",
    ]
    summary_rows: List[Dict[str, Any]] = []
    for (scenario, mode), bucket in sorted(buckets.items()):
        out: Dict[str, Any] = {
            "scenario": scenario,
            "scenario_label": bucket[0]["scenario_label"],
            "mode": mode,
            "mode_label": bucket[0]["mode_label"],
            "repeats": len(bucket),
        }
        for metric_name in metric_names:
            values = _non_null(row.get(metric_name) for row in bucket)
            out[f"{metric_name}_mean"] = None if not values else float(mean(values))
            out[f"{metric_name}_max"] = None if not values else float(max(values))
        out["delta_b_q_sign_flips_total"] = int(sum(int(row["delta_b_q_sign_flips"]) for row in bucket))
        out["delta_b_p_sign_flips_total"] = int(sum(int(row["delta_b_p_sign_flips"]) for row in bucket))
        out["committed_b_q_sign_flips_total"] = int(sum(int(row["committed_b_q_sign_flips"]) for row in bucket))
        out["committed_b_p_sign_flips_total"] = int(sum(int(row["committed_b_p_sign_flips"]) for row in bucket))
        summary_rows.append(out)
    return summary_rows


def _paired_repeat_comparison(repeat_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    buckets: Dict[tuple[str, int], Dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in repeat_rows:
        buckets[(str(row["scenario"]), int(row["repeat"]))][str(row["mode"])] = row

    comparison_rows: List[Dict[str, Any]] = []
    for (scenario, repeat), bucket in sorted(buckets.items()):
        full_row = bucket.get("hybrid_full")
        gated_row = bucket.get("hybrid_gated_teacher_v5")
        if full_row is None or gated_row is None:
            continue
        comparison_rows.append(
            {
                "scenario": scenario,
                "scenario_label": full_row["scenario_label"],
                "repeat": repeat,
                "seed_full": int(full_row["seed"]),
                "seed_gated_v5": int(gated_row["seed"]),
                "full_final_window_ler": full_row.get("final_window_ler"),
                "gated_v5_final_window_ler": gated_row.get("final_window_ler"),
                "gated_minus_full_final_window_ler": None
                if full_row.get("final_window_ler") is None or gated_row.get("final_window_ler") is None
                else float(gated_row["final_window_ler"] - full_row["final_window_ler"]),
                "full_max_abs_teacher_b": full_row.get("max_abs_teacher_b"),
                "gated_v5_max_abs_teacher_b": gated_row.get("max_abs_teacher_b"),
                "full_max_abs_delta_b": full_row.get("max_abs_delta_b"),
                "gated_v5_max_abs_delta_b": gated_row.get("max_abs_delta_b"),
                "full_max_abs_committed_b": full_row.get("max_abs_committed_b"),
                "gated_v5_max_abs_committed_b": gated_row.get("max_abs_committed_b"),
                "gated_minus_full_max_abs_delta_b": None
                if full_row.get("max_abs_delta_b") is None or gated_row.get("max_abs_delta_b") is None
                else float(gated_row["max_abs_delta_b"] - full_row["max_abs_delta_b"]),
                "gated_minus_full_max_abs_committed_b": None
                if full_row.get("max_abs_committed_b") is None or gated_row.get("max_abs_committed_b") is None
                else float(gated_row["max_abs_committed_b"] - full_row["max_abs_committed_b"]),
                "full_teacher_contribution_l2_mean": full_row.get("teacher_contribution_l2_mean"),
                "gated_v5_teacher_contribution_l2_mean": gated_row.get("teacher_contribution_l2_mean"),
            }
        )
    return comparison_rows


def _stdout_summary(
    run_dir: Path,
    output_dir: Path,
    trace_rows: Sequence[Mapping[str, Any]],
    field_rows: Sequence[Mapping[str, Any]],
    paired_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    return {
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "trace_row_count": len(trace_rows),
        "field_availability": {row["field"]: row["status"] for row in field_rows},
        "paired_repeat_highlights": paired_rows,
    }


def main() -> int:
    args = _arg_parser().parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    summary = _read_json(run_dir / "summary.json")
    output_dir = _ensure_output_dir(run_dir, args.output_dir)

    trace_rows = _window_trace_rows(summary)
    field_rows = _field_availability(trace_rows)
    repeat_rows = _repeat_summary(trace_rows)
    scenario_mode_rows = _scenario_mode_summary(repeat_rows)
    paired_rows = _paired_repeat_comparison(repeat_rows)

    _write_csv(output_dir / "trace_rows.csv", trace_rows, TRACE_FIELDS)
    _write_csv(
        output_dir / "repeat_summary.csv",
        repeat_rows,
        [
            "scenario",
            "scenario_label",
            "mode",
            "mode_label",
            "repeat",
            "seed",
            "n_windows",
            "first_window_id",
            "last_window_id",
            "final_window_ler",
            "mean_window_ler",
            "max_window_ler",
            "mean_correction_utilization",
            "max_abs_teacher_b",
            "final_abs_teacher_b",
            "max_abs_delta_b",
            "final_abs_delta_b",
            "max_abs_committed_b",
            "final_abs_committed_b",
            "teacher_b_q_sign_flips",
            "teacher_b_p_sign_flips",
            "delta_b_q_sign_flips",
            "delta_b_p_sign_flips",
            "committed_b_q_sign_flips",
            "committed_b_p_sign_flips",
            "teacher_contribution_l2_mean",
        ],
    )
    _write_csv(
        output_dir / "scenario_mode_summary.csv",
        scenario_mode_rows,
        [
            "scenario",
            "scenario_label",
            "mode",
            "mode_label",
            "repeats",
            "final_window_ler_mean",
            "final_window_ler_max",
            "mean_window_ler_mean",
            "mean_window_ler_max",
            "max_window_ler_mean",
            "max_window_ler_max",
            "mean_correction_utilization_mean",
            "mean_correction_utilization_max",
            "max_abs_teacher_b_mean",
            "max_abs_teacher_b_max",
            "final_abs_teacher_b_mean",
            "final_abs_teacher_b_max",
            "max_abs_delta_b_mean",
            "max_abs_delta_b_max",
            "final_abs_delta_b_mean",
            "final_abs_delta_b_max",
            "max_abs_committed_b_mean",
            "max_abs_committed_b_max",
            "final_abs_committed_b_mean",
            "final_abs_committed_b_max",
            "teacher_contribution_l2_mean_mean",
            "teacher_contribution_l2_mean_max",
            "delta_b_q_sign_flips_total",
            "delta_b_p_sign_flips_total",
            "committed_b_q_sign_flips_total",
            "committed_b_p_sign_flips_total",
        ],
    )
    _write_csv(
        output_dir / "paired_repeat_comparison.csv",
        paired_rows,
        [
            "scenario",
            "scenario_label",
            "repeat",
            "seed_full",
            "seed_gated_v5",
            "full_final_window_ler",
            "gated_v5_final_window_ler",
            "gated_minus_full_final_window_ler",
            "full_max_abs_teacher_b",
            "gated_v5_max_abs_teacher_b",
            "full_max_abs_delta_b",
            "gated_v5_max_abs_delta_b",
            "full_max_abs_committed_b",
            "gated_v5_max_abs_committed_b",
            "gated_minus_full_max_abs_delta_b",
            "gated_minus_full_max_abs_committed_b",
            "full_teacher_contribution_l2_mean",
            "gated_v5_teacher_contribution_l2_mean",
        ],
    )
    (output_dir / "field_availability.json").write_text(json.dumps(field_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "output_dir": str(output_dir),
                "trace_rows_csv": str(output_dir / "trace_rows.csv"),
                "repeat_summary_csv": str(output_dir / "repeat_summary.csv"),
                "scenario_mode_summary_csv": str(output_dir / "scenario_mode_summary.csv"),
                "paired_repeat_comparison_csv": str(output_dir / "paired_repeat_comparison.csv"),
                "field_availability_json": str(output_dir / "field_availability.json"),
                "trace_row_count": len(trace_rows),
                "repeat_summary_row_count": len(repeat_rows),
                "scenario_mode_summary_row_count": len(scenario_mode_rows),
                "paired_repeat_row_count": len(paired_rows),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(json.dumps(_stdout_summary(run_dir, output_dir, trace_rows, field_rows, paired_rows), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
