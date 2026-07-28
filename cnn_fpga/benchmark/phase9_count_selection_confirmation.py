"""Finite count selection and untouched confirmation for T04 preregistration."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import time
from typing import Any, Mapping, Sequence

from cnn_fpga.benchmark import phase9_highdim_joint_maxt_preflight as t05


TASK_ID = "T-RISK-20260728-06"
CONFIG_PATH = "configs/phase9/t_risk_20260728_06_count_selection_confirmation.json"
PASS = "PASS_COUNT_SELECTED_AND_UNTOUCHED_CONFIRMED"
FAIL = "TERMINAL_NO_GO_COUNT_SELECTION_OR_CONFIRMATION"


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _self_hash(value: Mapping[str, Any]) -> str:
    body = dict(value)
    body.pop("analysis_sha256", None)
    return _sha(body)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _binding(path: Path, root: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    return {
        "path": path.resolve().relative_to(root).as_posix(),
        "bytes": len(payload),
        "sha256": _sha_bytes(payload),
    }


def validate_config(config: Mapping[str, Any]) -> None:
    if config.get("task_id") != TASK_ID:
        raise ValueError("task ID drift")
    grid = config["linked_count_grid"]
    if [
        (row["scale"], row["state_clusters"], row["round_clusters"],
         row["aggregate_fault_clusters"]) for row in grid
    ] != [
        (1.5, 576, 1152, 3456),
        (2.0, 768, 1536, 4608),
        (2.5, 960, 1920, 5760),
        (3.0, 1152, 2304, 6912),
    ]:
        raise ValueError("linked finite count grid drift")
    density = config["density"]
    maxt = config["joint_maxt"]
    if (
        density["factor"] != 1.0
        or density["multiplier_replicates"] != 199
        or density["quantile"] != "higher"
        or maxt["factor"] != 1.0
        or maxt["multiplier_replicates"] != 199
        or maxt["quantile"] != "higher"
        or maxt["gate_deletion"] is not False
        or maxt["aggregate_rescue"] is not False
        or maxt["cross_state_averaging"] is not False
        or maxt["pointwise_z_substitution"] is not False
    ):
        raise ValueError("frozen statistical contract drift")
    firewall = config["outcome_firewall"]
    if (
        firewall["t04_run_exists"] is not False
        or firewall["t04_formal_outcomes_accessed"] is not False
        or firewall["t05_no_go_rewritten"] is not False
        or firewall["confirmation_may_choose_nothing"] is not True
    ):
        raise ValueError("outcome firewall drift")
    seeds = [
        value for key, value in density.items() if key.endswith("_seed_base")
    ] + [
        value for key, value in maxt.items() if key.endswith("_seed_base")
    ]
    if len(seeds) != len(set(seeds)):
        raise ValueError("seed namespace collision")
    for key, value in config["claim_boundary"].items():
        if key not in (
            "count_design_only", "t04_preregistration_released",
            "t04_scientific_execution_released",
        ) and value is not None:
            raise ValueError(f"prohibited claim populated: {key}")


def _density_worker(payload: Mapping[str, Any]) -> dict[str, Any]:
    spec = payload["spec"]
    family = payload["family"]
    count = int(spec["cluster_count"])
    records = []
    for trial in range(int(spec["trials"])):
        address = int(spec["cell_index"]) * int(spec["trials"]) + trial
        trial_seed = int(spec["trial_seed_base"]) + address
        multiplier_seed = int(spec["multiplier_seed_base"]) + address
        left, right, truth = t05._physical_density_trial(
            dimension=int(spec["dimension"]),
            count=count,
            true_distance=float(spec["effect"]),
            family=family,
            seed=trial_seed,
        )
        ucb = t05.paired_density_trace_ucb_physical(
            left, right, confidence=float(spec["confidence"]),
            multiplier_replicates=int(spec["B"]), seed=multiplier_seed,
            calibration_factor=float(spec["factor"]),
        )
        records.append({
            "row_type": "density_trial", "split": spec["split"],
            "candidate_scale": spec["scale"], "cluster_count": count,
            "cell_id": spec["cell_id"], "family": spec["family"],
            "dimension": spec["dimension"], "true_distance": truth,
            "trial": trial, "trial_seed": trial_seed,
            "multiplier_seed": multiplier_seed, "estimate": ucb.estimate,
            "raw_radius": ucb.raw_radius, "upper_bound": ucb.upper_bound,
            "covered": ucb.upper_bound + 1e-15 >= truth,
            "equivalence_pass": ucb.upper_bound
            <= float(spec["margin"]),
        })
    return {"spec": dict(spec), "records": records}


def _density_specs(
    config: Mapping[str, Any], t05_config: Mapping[str, Any],
    *, split: str, selected: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    d = config["density"]
    base = {
        "confidence": d["confidence"], "B": d["multiplier_replicates"],
        "factor": d["factor"], "margin": d["margin"], "split": split,
    }
    specs = []
    if split == "selection":
        index = 0
        for candidate in config["linked_count_grid"]:
            for dimension in d["dimensions"]:
                specs.append({
                    **base, **candidate, "cell_index": index,
                    "cell_id": (
                        f"selection__n{candidate['state_clusters']}__"
                        f"d{dimension}__heteroskedastic__effect_0.050"
                    ),
                    "family": d["selection_family"], "dimension": dimension,
                    "effect": d["selection_effect"],
                    "cluster_count": candidate["state_clusters"],
                    "trials": d["selection_trials_per_cell"],
                    "trial_seed_base": d["selection_trial_seed_base"],
                    "multiplier_seed_base": d["selection_multiplier_seed_base"],
                })
                index += 1
    else:
        if selected is None:
            raise ValueError("confirmation requires selected count")
        index = 0
        for family in sorted(d["confirmation_families"]):
            for dimension in d["dimensions"]:
                for effect in d["confirmation_effects"]:
                    specs.append({
                        **base, **selected, "cell_index": index,
                        "cell_id": (
                            f"confirmation__n{selected['state_clusters']}__"
                            f"d{dimension}__{family}__effect_{effect:.3f}"
                        ),
                        "family": family, "dimension": dimension,
                        "effect": effect,
                        "cluster_count": selected["state_clusters"],
                        "trials": d["confirmation_trials_per_cell"],
                        "trial_seed_base": d["confirmation_trial_seed_base"],
                        "multiplier_seed_base": d[
                            "confirmation_multiplier_seed_base"
                        ],
                    })
                    index += 1
    return specs


def _run_density(
    root: Path, config: Mapping[str, Any], t05_config: Mapping[str, Any],
    *, split: str, workers: int, selected: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    specs = _density_specs(config, t05_config, split=split, selected=selected)
    directory = root / config["artifact_paths"][f"{split}_chunks"]
    directory.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    pending = []
    config_sha = _sha(config)
    for spec in specs:
        path = directory / f"{spec['cell_id']}.json"
        if path.exists():
            chunk = _load(path)
            if (
                chunk.get("config_analysis_sha256") == config_sha
                and chunk.get("spec") == spec
                and chunk.get("analysis_sha256") == _self_hash(chunk)
                and len(chunk.get("records", [])) == spec["trials"]
            ):
                rows.extend(chunk["records"])
                continue
        pending.append(spec)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_density_worker, {
                "spec": spec,
                "family": t05_config["density_uq"]["families"][spec["family"]],
            }): spec for spec in pending
        }
        for future in as_completed(futures):
            result = future.result()
            chunk = {
                "task_id": TASK_ID, "schema_version": "PHASE9-T06-DENSITY-CHUNK-V1",
                "config_analysis_sha256": config_sha, "spec": result["spec"],
                "record_count": len(result["records"]), "records": result["records"],
            }
            chunk["analysis_sha256"] = _self_hash(chunk)
            _atomic(directory / f"{result['spec']['cell_id']}.json", chunk)
            rows.extend(result["records"])
            heartbeat = {
                "task_id": TASK_ID, "phase": f"density_{split}",
                "completed_rows": len(rows),
                "expected_rows": sum(item["trials"] for item in specs),
                "fresh_unix": time.time(),
            }
            heartbeat["analysis_sha256"] = _self_hash(heartbeat)
            _atomic(root / config["artifact_paths"]["heartbeat"], heartbeat)
    if len(rows) != sum(spec["trials"] for spec in specs):
        raise RuntimeError(f"{split} density denominator drift")
    if len({row["trial_seed"] for row in rows}) != len(rows):
        raise RuntimeError(f"{split} trial seed collision")
    return sorted(rows, key=lambda row: (row["cell_id"], row["trial"]))


def _summarize_selection(
    config: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], dict[float, bool]]:
    contract = config["density"]["selection_wilson"]
    output, decisions = [], {}
    for candidate in config["linked_count_grid"]:
        candidate_rows = [
            row for row in rows if row["candidate_scale"] == candidate["scale"]
        ]
        candidate_pass = True
        for dimension in config["density"]["dimensions"]:
            values = [row for row in candidate_rows if row["dimension"] == dimension]
            coverage = sum(row["covered"] for row in values)
            equivalent = sum(row["equivalence_pass"] for row in values)
            cov = t05._wilson(
                coverage, len(values), confidence=contract["confidence"],
                comparisons=contract["comparisons"],
            )
            eq = t05._wilson(
                equivalent, len(values), confidence=contract["confidence"],
                comparisons=contract["comparisons"],
            )
            passed = (
                cov[0] >= contract["coverage_lcb_minimum"]
                and eq[0] >= contract["local_equivalence_lcb_minimum"]
            )
            candidate_pass &= passed
            output.append({
                "split": "selection", "candidate_scale": candidate["scale"],
                "cluster_count": candidate["state_clusters"],
                "dimension": dimension, "trials": len(values),
                "coverage_successes": coverage, "coverage_lcb": cov[0],
                "equivalence_successes": equivalent,
                "equivalence_rate": equivalent / len(values),
                "equivalence_lcb": eq[0], "equivalence_ucb": eq[1],
                "gate_pass": passed,
            })
        decisions[float(candidate["scale"])] = candidate_pass
    return output, decisions


def _summarize_confirmation(
    config: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], bool]:
    c = config["density"]["confirmation_wilson"]
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["cell_id"]), []).append(row)
    output, all_pass = [], True
    for cell_id, values in sorted(grouped.items()):
        coverage = sum(row["covered"] for row in values)
        equivalent = sum(row["equivalence_pass"] for row in values)
        cov = t05._wilson(
            coverage, len(values), confidence=c["confidence"],
            comparisons=c["comparisons"],
        )
        eq = t05._wilson(
            equivalent, len(values), confidence=c["confidence"],
            comparisons=c["comparisons"],
        )
        effect = float(values[0]["true_distance"])
        coverage_pass = (
            coverage / len(values) >= c["coverage_rate_minimum"]
            and cov[0] >= c["coverage_lcb_minimum"]
        )
        power_pass = (
            eq[0] >= c["null_equivalence_lcb_minimum"] if effect == 0.0
            else eq[0] >= c["local_equivalence_lcb_minimum"] if effect == 0.05
            else eq[1] <= c["boundary_equivalence_ucb_maximum"] if effect == 0.1
            else eq[1] <= c["outside_equivalence_ucb_maximum"]
        )
        passed = coverage_pass and power_pass
        all_pass &= passed
        output.append({
            "cell_id": cell_id, "family": values[0]["family"],
            "dimension": values[0]["dimension"], "true_distance": effect,
            "cluster_count": values[0]["cluster_count"], "trials": len(values),
            "coverage_successes": coverage, "coverage_rate": coverage / len(values),
            "coverage_lcb": cov[0], "coverage_ucb": cov[1],
            "equivalence_successes": equivalent,
            "equivalence_rate": equivalent / len(values),
            "equivalence_lcb": eq[0], "equivalence_ucb": eq[1],
            "coverage_gate_pass": coverage_pass, "power_gate_pass": power_pass,
            "gate_pass": passed,
        })
    return output, all_pass


def _scaled_gates(
    blueprint: Sequence[Mapping[str, Any]], candidate: Mapping[str, Any]
) -> list[t05.Gate]:
    mapping = {
        384: int(candidate["state_clusters"]),
        768: int(candidate["round_clusters"]),
        2304: int(candidate["aggregate_fault_clusters"]),
        0: 0,
    }
    return [
        t05.Gate(**{**row, "cluster_count": mapping[int(row["cluster_count"])]})
        for row in blueprint
    ]


def _maxt_config(
    t05_config: Mapping[str, Any], config: Mapping[str, Any], *, split: str
) -> dict[str, Any]:
    value = json.loads(json.dumps(t05_config))
    source = config["joint_maxt"]
    target = value["joint_maxt"]
    target["correlation_model"] = source["correlation_model"]
    target["power"] = source["power"]
    target["multiplier_replicates"] = source["multiplier_replicates"]
    target["calibration_factor"] = source["factor"]
    target["quantile_method"] = source["quantile"]
    target["influence_seed_base"] = source[f"{split}_influence_seed_base"]
    target["rademacher_seed_base"] = source[f"{split}_rademacher_seed_base"]
    target["power_seed_base"] = source[f"{split}_power_seed_base"]
    return value


def _run_maxt_grid(
    config: Mapping[str, Any], t05_config: Mapping[str, Any],
    blueprint: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[float, bool]]:
    rows, decisions = [], {}
    maxt_config = _maxt_config(t05_config, config, split="selection")
    for candidate in config["linked_count_grid"]:
        q, replicate_rows, power_rows, gate_decisions = t05.build_joint_maxt(
            maxt_config, _scaled_gates(blueprint, candidate)
        )
        passed = all(gate_decisions.values())
        decisions[float(candidate["scale"])] = passed
        rows.append({
            "row_type": "maxt_selection_summary", "split": "selection",
            "candidate_scale": candidate["scale"],
            "cluster_count": candidate["state_clusters"], "critical": q,
            "gate_pass": passed,
            "failed_gate_count": sum(not value for value in gate_decisions.values()),
        })
        rows.extend({
            **row, "split": "selection",
            "candidate_scale": candidate["scale"],
            "cluster_count": candidate["state_clusters"],
        } for row in replicate_rows + power_rows)
    return rows, decisions


def _serialize(rows: Sequence[Mapping[str, Any]]) -> bytes:
    fields = sorted({key for row in rows for key in row})
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer, fieldnames=fields, extrasaction="ignore", lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def write_artifacts(root: Path | None = None, *, workers: int = 4) -> dict[str, Any]:
    base = (root or _root()).resolve()
    config = _load(base / CONFIG_PATH)
    validate_config(config)
    if workers != config["resource"]["workers"]:
        raise ValueError("worker count drift")
    paths = config["artifact_paths"]
    lock = base / paths["run_directory"] / "supervisor.owner.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        handle = lock.open("x", encoding="utf-8")
    except FileExistsError as exc:
        raise RuntimeError("another T06 supervisor may be active") from exc
    handle.write(json.dumps({"pid": os.getpid(), "time": time.time()}))
    handle.close()
    try:
        parents = {
            name: _load(base / path)
            for name, path in config["parent_artifacts"].items()
        }
        if (
            parents["t05_report"]["verdict"]
            != "FAIL_T04_STATISTICAL_PREREGISTRATION_BLOCKED"
            or parents["t05_verification"]["verdict"]
            != "PASS_INDEPENDENT_T04_STATISTICAL_NO_GO_VERIFICATION"
            or parents["t05_verification"]["t04_preregistration_released"] is not False
        ):
            raise ValueError("T05 verified NO-GO lineage drift")
        t05_config = parents["t05_config"]
        blueprint = parents["t05_blueprint"]["gates"]
        if len(blueprint) != 3043:
            raise ValueError("T05 blueprint denominator drift")

        selection_density = _run_density(
            base, config, t05_config, split="selection", workers=workers
        )
        selection_summary, density_decisions = _summarize_selection(
            config, selection_density
        )
        maxt_selection, maxt_decisions = _run_maxt_grid(
            config, t05_config, blueprint
        )
        selected = next((
            candidate for candidate in config["linked_count_grid"]
            if density_decisions[float(candidate["scale"])]
            and maxt_decisions[float(candidate["scale"])]
        ), None)
        if selected is None:
            confirmation_density, confirmation_summary = [], []
            confirmation_density_pass = False
            confirm_q, maxt_confirmation, maxt_confirmation_pass = None, [], False
            selected_blueprint = None
        else:
            confirmation_density = _run_density(
                base, config, t05_config, split="confirmation",
                workers=workers, selected=selected,
            )
            confirmation_summary, confirmation_density_pass = (
                _summarize_confirmation(config, confirmation_density)
            )
            selected_gates = _scaled_gates(blueprint, selected)
            confirmation_config = _maxt_config(
                t05_config, config, split="confirmation"
            )
            confirm_q, maxt_raw, maxt_power, maxt_gate_decisions = (
                t05.build_joint_maxt(confirmation_config, selected_gates)
            )
            maxt_confirmation_pass = all(maxt_gate_decisions.values())
            maxt_confirmation = [
                {**row, "split": "confirmation",
                 "candidate_scale": selected["scale"],
                 "cluster_count": selected["state_clusters"]}
                for row in maxt_raw + maxt_power
            ]
            selected_blueprint = {
                "task_id": TASK_ID,
                "schema_version": "PHASE9-T06-SELECTED-BLUEPRINT-V1",
                "selected_count": selected,
                "gate_count": len(selected_gates),
                "stochastic_gate_count": sum(
                    not gate.deterministic for gate in selected_gates
                ),
                "gates": [asdict(gate) for gate in selected_gates],
            }
            selected_blueprint["analysis_sha256"] = _self_hash(selected_blueprint)
            _atomic(base / paths["selected_blueprint"], selected_blueprint)

        scale = float(selected["scale"]) if selected else math.inf
        resource = config["resource"]
        exact_rows = (
            int(round(resource["base_t04_rows"] * scale)) if selected else None
        )
        resource_summary = {
            "selected_scale": selected["scale"] if selected else None,
            "exact_t04_rows": exact_rows,
            "estimated_wall_seconds": (
                resource["base_wall_seconds_4_workers"] * scale
                if selected else None
            ),
            "estimated_artifact_bytes": (
                math.ceil(resource["base_artifact_bytes"] * scale)
                if selected else None
            ),
            "estimated_rss_bytes": (
                math.ceil(resource["base_rss_bytes"] * scale)
                if selected else None
            ),
            "free_disk_bytes": shutil.disk_usage(base).free,
            "fresh_t04_layer_benchmark_required": True,
        }
        resource_pass = bool(
            selected
            and resource_summary["estimated_wall_seconds"]
            <= resource["maximum_wall_seconds"]
            and resource_summary["estimated_artifact_bytes"]
            <= resource["maximum_artifact_bytes"]
            and resource_summary["estimated_rss_bytes"]
            <= resource["maximum_rss_bytes"]
            and resource_summary["free_disk_bytes"]
            - resource_summary["estimated_artifact_bytes"]
            >= resource["minimum_free_disk_after_estimate_bytes"]
        )
        source_rows = (
            selection_density + selection_summary + maxt_selection
            + confirmation_density + confirmation_summary + maxt_confirmation
        )
        _atomic_bytes(base / paths["source_data"], _serialize(source_rows))
        gates = {
            "G01_t05_verified_no_go_live": True,
            "G02_t04_outcome_firewall_closed": True,
            "G03_finite_grid_complete": len(config["linked_count_grid"]) == 4,
            "G04_selection_seed_ranges_injective": (
                len({row["trial_seed"] for row in selection_density})
                == len(selection_density)
            ),
            "G05_all_candidates_density_evaluated": (
                len(selection_summary) == 8
            ),
            "G06_all_candidates_full_maxt_evaluated": (
                len([
                    row for row in maxt_selection
                    if row["row_type"] == "maxt_selection_summary"
                ]) == 4
            ),
            "G07_smallest_joint_passing_candidate_selected": selected is not None,
            "G08_confirmation_seed_ranges_disjoint": (
                selected is not None
                and {row["trial_seed"] for row in selection_density}.isdisjoint(
                    {row["trial_seed"] for row in confirmation_density}
                )
            ),
            "G09_full_24_cell_density_confirmation": (
                len(confirmation_summary) == 24
                and len(confirmation_density)
                == 24 * config["density"]["confirmation_trials_per_cell"]
            ),
            "G10_density_confirmation_all_pass": confirmation_density_pass,
            "G11_full_3043_gate_maxt_confirmation": (
                selected_blueprint is not None
                and selected_blueprint["gate_count"] == 3043
                and selected_blueprint["stochastic_gate_count"] == 3037
            ),
            "G12_maxt_confirmation_all_pass": maxt_confirmation_pass,
            "G13_resource_forecast_pass": resource_pass,
            "G14_t04_fresh_resource_gate_retained": (
                resource["t04_fresh_layer_benchmark_required"] is True
            ),
            "G15_claims_null": all(
                value is None for key, value in config["claim_boundary"].items()
                if key not in (
                    "count_design_only", "t04_preregistration_released",
                    "t04_scientific_execution_released",
                )
            ),
        }
        report = {
            "task_id": TASK_ID,
            "schema_version": "PHASE9-COUNT-SELECTION-CONFIRMATION-REPORT-V1",
            "formal_outcomes_accessed": False,
            "bindings": {
                "config": _binding(base / CONFIG_PATH, base),
                **{
                    name: _binding(base / path, base)
                    for name, path in config["parent_artifacts"].items()
                },
                "source_data": _binding(base / paths["source_data"], base),
                "selected_blueprint": (
                    _binding(base / paths["selected_blueprint"], base)
                    if selected else None
                ),
                "writer_source": _binding(Path(__file__), base),
            },
            "selection": {
                "density_summaries": selection_summary,
                "maxt_summaries": [
                    row for row in maxt_selection
                    if row["row_type"] == "maxt_selection_summary"
                ],
                "selected": selected,
                "rule": config["selection_rule"],
            },
            "confirmation": {
                "density_row_count": len(confirmation_density),
                "density_summaries": confirmation_summary,
                "maxt_critical": confirm_q,
                "maxt_power_rows": [
                    row for row in maxt_confirmation
                    if row["row_type"] == "maxt_power"
                ],
            },
            "resource": resource_summary,
            "gates": gates,
            "gate_summary": {
                "passed": sum(value is True for value in gates.values()),
                "total": len(gates),
            },
            "t04_preregistration_released": all(gates.values()),
            "t04_scientific_execution_released": False,
            "qualified_claim": None,
            "claim_boundary": {
                **config["claim_boundary"],
                "t04_preregistration_released": all(gates.values()),
            },
            "verdict": PASS if all(gates.values()) else FAIL,
        }
        report["analysis_sha256"] = _self_hash(report)
        _atomic(base / paths["report"], report)
        return report
    finally:
        try:
            lock.unlink()
        except FileNotFoundError:
            pass


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    report = write_artifacts(workers=args.workers)
    print(json.dumps({
        "verdict": report["verdict"], "selected": report["selection"]["selected"],
        "gate_summary": report["gate_summary"],
        "t04_preregistration_released": report["t04_preregistration_released"],
    }, ensure_ascii=False, indent=2))
    return 0 if report["verdict"] == PASS else 2


if __name__ == "__main__":
    raise SystemExit(main())
