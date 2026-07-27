"""Design-only power analysis for the fresh Phase-9 twin qualification.

No backend, historical formal cell, or formal seed is accessed here.  The
module expands the complete future gate blueprint and evaluates the frozen
intersection-union equivalence procedure under same-backend null replicas,
zero-effect A/B qualification, and predeclared outside-margin mutants.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from cnn_fpga.benchmark.phase9_fresh_twin_lineage import (
    PASS_VERDICT as LINEAGE_PASS,
    build_receipt as build_lineage_receipt,
)


TASK_ID = "T-RISK-20260726-01"
SCHEMA_VERSION = "1.0"
PASS_VERDICT = "PASS_FRESH_TWIN_DESIGN_POWER"
FAIL_VERDICT = "NO_GO_FRESH_TWIN_DESIGN_POWER"
CONFIG_PATH = "configs/phase9/t_risk_20260726_01_design_power.json"
DIAGNOSTIC_PATH = "docs/t_risk_20260726_01_iq_semantics_diagnostic.json"
READOUT_POWER_PATH = "docs/t_risk_20260726_01_readout_power.json"
Z_TOST = 1.6448536269514722
Z_WILSON = 1.959963984540054


@dataclass(frozen=True)
class GateBlueprint:
    gate_id: str
    family: str
    stage: str
    metric: str
    margin: float
    normalized_sd: float
    deterministic: bool = False


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha(value: object) -> str:
    return _sha_bytes(_canonical(value))


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _strict_int(value: object, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an exact integer")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _positive(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _interval(value: object, name: str) -> tuple[int, int]:
    if not isinstance(value, dict) or set(value) != {"start", "count"}:
        raise ValueError(f"{name} split schema mismatch")
    start = _strict_int(value["start"], f"{name}.start", minimum=0)
    count = _strict_int(value["count"], f"{name}.count")
    return start, start + count


def validate_config(config: Mapping[str, Any]) -> None:
    if config.get("task_id") != TASK_ID:
        raise ValueError("task_id mismatch")
    policy = config.get("historical_policy")
    if not isinstance(policy, dict):
        raise ValueError("historical_policy missing")
    expected_policy = {
        "historical_no_go_rewritten": False,
        "historical_formal_cell_data_access_allowed": False,
        "old_outcome_may_choose_margin": False,
        "old_outcome_may_choose_endpoint": False,
        "old_outcome_may_choose_family": False,
        "pilot_may_choose": ["round_sample_count", "trajectory_sample_count"],
    }
    if policy != expected_policy:
        raise ValueError("historical/pilot policy drift")

    readout = config.get("readout_convention")
    if not isinstance(readout, dict):
        raise ValueError("readout convention missing")
    if readout.get("raw_base_measure") != (
        "two_dimensional_lebesgue_per_complex_iq_sample"
    ):
        raise ValueError("raw IQ base measure drift")
    if readout.get("sigma") != "per_real_axis_standard_deviation":
        raise ValueError("sigma convention drift")
    if readout.get("integration") != "arithmetic_mean_over_window":
        raise ValueError("integration convention drift")
    if readout.get("proper_score_unit") != "nats_per_complex_sample":
        raise ValueError("proper-score unit drift")

    splits = config.get("splits")
    if not isinstance(splits, dict) or len(splits) != 16:
        raise ValueError("fresh split registry incomplete")
    intervals = {name: _interval(value, name) for name, value in splits.items()}
    names = sorted(intervals)
    for index, left_name in enumerate(names):
        left = intervals[left_name]
        for right_name in names[index + 1 :]:
            right = intervals[right_name]
            if max(left[0], right[0]) < min(left[1], right[1]):
                raise ValueError(f"seed overlap: {left_name}/{right_name}")
    # The old transaction occupied 920000--970000.  Fresh design must be
    # independently addressable and must not overlap it.
    if any(start < 1_000_000 for start, _ in intervals.values()):
        raise ValueError("fresh seed namespace overlaps historical range")

    candidates = config.get("candidate_sample_counts")
    if not isinstance(candidates, dict):
        raise ValueError("candidate sample counts missing")
    if candidates.get("round") != [128, 256, 512, 768]:
        raise ValueError("round candidate set drift")
    if candidates.get("trajectory") != [96, 192, 256]:
        raise ValueError("trajectory candidate set drift")

    matrix = config.get("formal_matrix")
    if not isinstance(matrix, dict) or matrix.get("all_cells_required") is not True:
        raise ValueError("full formal cell requirement missing")
    if matrix.get("cutoff_ladder") != [8, 12, 16, 20]:
        raise ValueError("cutoff ladder drift")
    if matrix.get("primary_cutoff_increments") != [[12, 16], [16, 20]]:
        raise ValueError("primary cutoff increments drift")
    if matrix.get("historical_cutoff_increment_diagnostic_only") != [8, 12]:
        raise ValueError("historical cutoff diagnostic must remain explicit")

    families = config.get("families")
    margins = config.get("margins")
    if not isinstance(families, dict) or len(families) != 7:
        raise ValueError("seven primary families required")
    if not isinstance(margins, dict):
        raise ValueError("margins missing")
    for family, spec in families.items():
        if not isinstance(spec, dict):
            raise ValueError(f"{family} family schema invalid")
        if spec.get("stage") not in {"round", "trajectory"}:
            raise ValueError(f"{family} stage invalid")
        _positive(spec.get("normalized_sd_envelope"), f"{family}.normalized_sd")
        metrics = spec.get("metrics")
        if not isinstance(metrics, list) or not metrics:
            raise ValueError(f"{family} metrics missing")
        for metric in metrics:
            if metric not in margins:
                raise ValueError(f"{family}/{metric} margin missing")
            _positive(margins[metric].get("value"), f"{metric}.margin")
            if "source" not in margins[metric]:
                raise ValueError(f"{metric} margin provenance missing")

    procedure = config.get("statistical_procedure")
    if not isinstance(procedure, dict):
        raise ValueError("statistical procedure missing")
    if procedure.get("global_test") != "intersection_union_equivalence":
        raise ValueError("global IUT procedure drift")
    if procedure.get("cell_test") != "two_one_sided_tests":
        raise ValueError("cell TOST procedure drift")
    if procedure.get("global_type_i_error") != 0.05:
        raise ValueError("global type-I error drift")
    if procedure.get("mixed_unit_vector_max") is not False:
        raise ValueError("mixed-unit vector max forbidden")
    if procedure.get("raw_log_evidence") != (
        "diagnostic only; never a cross-gain primary gate"
    ):
        raise ValueError("raw log-evidence must remain diagnostic")
    if procedure.get("cell_deletion") is not False:
        raise ValueError("cell deletion forbidden")
    if procedure.get("mean_only_rescue") is not False:
        raise ValueError("mean-only rescue forbidden")

    power = config.get("power_model")
    if not isinstance(power, dict):
        raise ValueError("power model missing")
    _strict_int(power.get("pseudoexperiments"), "pseudoexperiments", minimum=1000)
    correlation = _positive(
        power.get("family_shared_correlation"), "family_shared_correlation"
    )
    if correlation >= 1.0:
        raise ValueError("family correlation must be below one")
    if power.get("formal_pool_access") is not False:
        raise ValueError("design power cannot access formal pool")
    if power.get("model_selection_rule") != (
        "lexicographically smallest round/trajectory candidate satisfying every model power gate"
    ):
        raise ValueError("model sample selection rule drift")
    if power.get("final_selection_rule") != (
        "componentwise maximum of model selection, empirical readout selection, and the unmodeled-physics conservative floor"
    ):
        raise ValueError("final sample selection rule drift")
    floor = power.get("unmodeled_physics_conservative_floor")
    if not isinstance(floor, dict):
        raise ValueError("unmodeled-physics conservative floor missing")
    if floor.get("round_sample_count") != max(candidates["round"]):
        raise ValueError("round conservative floor must equal maximum candidate")
    if floor.get("trajectory_sample_count") != max(candidates["trajectory"]):
        raise ValueError("trajectory conservative floor must equal maximum candidate")
    if not isinstance(floor.get("reason"), str) or not floor["reason"]:
        raise ValueError("conservative floor provenance missing")
    outside = _positive(power.get("outside_margin_ratio"), "outside_margin_ratio")
    if outside < 1.25:
        raise ValueError("outside-margin ratio is too small")

    mutants = config.get("alternative_mutants")
    if not isinstance(mutants, dict) or len(mutants) < 10:
        raise ValueError("alternative mutation registry incomplete")
    for name, mutant in mutants.items():
        if mutant.get("family") not in families:
            raise ValueError(f"{name} family invalid")
        if _positive(mutant.get("effect_ratio"), f"{name}.effect_ratio") < outside:
            raise ValueError(f"{name} alternative is inside required margin")


def _metric_gate(
    config: Mapping[str, Any],
    *,
    family: str,
    metric: str,
    gate_id: str,
    deterministic: bool = False,
) -> GateBlueprint:
    family_spec = config["families"][family]
    return GateBlueprint(
        gate_id=gate_id,
        family=family,
        stage=family_spec["stage"],
        metric=metric,
        margin=float(config["margins"][metric]["value"]),
        normalized_sd=float(family_spec["normalized_sd_envelope"]),
        deterministic=deterministic,
    )


def build_blueprint(config: Mapping[str, Any]) -> list[GateBlueprint]:
    """Expand every declared same-cutoff, fault, and cutoff-tail cell."""

    validate_config(config)
    matrix = config["formal_matrix"]
    gates: list[GateBlueprint] = []
    shared = matrix["shared_fock_states"]
    actions = matrix["nominal_actions"]
    probes = matrix["representative_probes"]
    labels = matrix["logical_labels"]
    faults = matrix["fault_scenarios"]
    cutoffs = matrix["same_cutoff_ab"]

    physical_metrics = config["families"]["physical_state_channel"]["metrics"]
    iq_metrics = config["families"]["iq_conditional_distribution"]["metrics"]
    likelihood_metrics = config["families"]["likelihood_score_posterior"]["metrics"]
    for cutoff in cutoffs:
        for state in shared:
            for action in actions:
                scope = f"ab/c{cutoff}/shared/{state}/{action}"
                for metric in physical_metrics:
                    gates.append(_metric_gate(
                        config, family="physical_state_channel", metric=metric,
                        gate_id=f"{scope}/{metric}",
                    ))
                for metric in iq_metrics:
                    gates.append(_metric_gate(
                        config, family="iq_conditional_distribution", metric=metric,
                        gate_id=f"{scope}/{metric}",
                    ))
                for metric in likelihood_metrics:
                    gates.append(_metric_gate(
                        config, family="likelihood_score_posterior", metric=metric,
                        gate_id=f"{scope}/{metric}",
                    ))
        for label in labels:
            for action in actions:
                gates.append(_metric_gate(
                    config,
                    family="logical_ptm_survival",
                    metric="logical_survival",
                    gate_id=f"ab/c{cutoff}/logical/{label}/{action}/logical_survival",
                ))
        for action in actions:
            gates.append(_metric_gate(
                config,
                family="logical_ptm_survival",
                metric="logical_ptm_entry",
                gate_id=f"ab/c{cutoff}/logical/PTM/{action}/logical_ptm_entry",
            ))
        for state in shared:
            gates.append(_metric_gate(
                config,
                family="reset_leakage",
                metric="rao_blackwell_reset_success",
                gate_id=f"ab/c{cutoff}/reset/{state}/rao_blackwell_reset_success",
            ))
            gates.append(_metric_gate(
                config,
                family="reset_leakage",
                metric="leakage_residence_rate",
                gate_id=f"ab/c{cutoff}/reset/{state}/leakage_residence_rate",
            ))
        for scenario in faults:
            for metric in config["families"]["fault_trajectory_tail"]["metrics"]:
                gates.append(_metric_gate(
                    config, family="fault_trajectory_tail", metric=metric,
                    gate_id=f"ab/c{cutoff}/fault/{scenario}/{metric}",
                ))
        gates.append(_metric_gate(
            config, family="cutoff_mapping", metric="principal_singular",
            gate_id=f"mapping/c{cutoff}/principal_singular", deterministic=True,
        ))
        gates.append(_metric_gate(
            config, family="cutoff_mapping", metric="projector_frobenius",
            gate_id=f"mapping/c{cutoff}/projector_frobenius", deterministic=True,
        ))

    # Representative action probes remain all-required, but only at the base
    # cutoff because they test the fixed action contract rather than cutoff.
    for probe in probes:
        scope = f"ab/c8/probe/{probe}"
        for metric in physical_metrics:
            gates.append(_metric_gate(
                config, family="physical_state_channel", metric=metric,
                gate_id=f"{scope}/{metric}",
            ))
        for metric in iq_metrics:
            gates.append(_metric_gate(
                config, family="iq_conditional_distribution", metric=metric,
                gate_id=f"{scope}/{metric}",
            ))
        for metric in likelihood_metrics:
            gates.append(_metric_gate(
                config, family="likelihood_score_posterior", metric=metric,
                gate_id=f"{scope}/{metric}",
            ))
    for probe in ("P09_LEAK_RESET", "P10_RESET_OK", "P11_RESET_FAIL"):
        for metric in ("rao_blackwell_reset_success", "leakage_residence_rate"):
            gates.append(_metric_gate(
                config, family="reset_leakage", metric=metric,
                gate_id=f"ab/c8/probe/{probe}/{metric}",
            ))

    cutoff_actions = ("IDLE", "XZ", "RESET")
    for lower, upper in matrix["primary_cutoff_increments"]:
        for state in shared:
            for action in cutoff_actions:
                gates.append(_metric_gate(
                    config, family="cutoff_mapping",
                    metric="cutoff_density_trace_distance",
                    gate_id=f"cutoff/{lower}-{upper}/shared/{state}/{action}/density",
                ))
        for label in labels:
            for action in cutoff_actions:
                gates.append(_metric_gate(
                    config, family="cutoff_mapping",
                    metric="cutoff_logical_survival",
                    gate_id=f"cutoff/{lower}-{upper}/logical/{label}/{action}/survival",
                ))
        for scenario in faults:
            gates.append(_metric_gate(
                config, family="cutoff_mapping",
                metric="cutoff_density_trace_distance",
                gate_id=f"cutoff/{lower}-{upper}/fault/{scenario}/terminal_density",
            ))
            gates.append(_metric_gate(
                config, family="cutoff_mapping",
                metric="cutoff_logical_survival",
                gate_id=f"cutoff/{lower}-{upper}/fault/{scenario}/survival",
            ))

    identifiers = [gate.gate_id for gate in gates]
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError("duplicate gate blueprint identifier")
    return gates


def _wilson(successes: int, trials: int) -> tuple[float, float]:
    if not 0 <= successes <= trials or trials <= 0:
        raise ValueError("invalid Wilson inputs")
    proportion = successes / trials
    denominator = 1.0 + Z_WILSON * Z_WILSON / trials
    center = (
        proportion + Z_WILSON * Z_WILSON / (2.0 * trials)
    ) / denominator
    radius = (
        Z_WILSON
        * math.sqrt(
            proportion * (1.0 - proportion) / trials
            + Z_WILSON * Z_WILSON / (4.0 * trials * trials)
        )
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def _global_equivalence_trials(
    blueprint: Sequence[GateBlueprint],
    *,
    round_count: int,
    trajectory_count: int,
    pseudoexperiments: int,
    correlation: float,
    seed: int,
    injected_family: str | None = None,
    effect_ratio: float = 0.0,
) -> int:
    stochastic = [gate for gate in blueprint if not gate.deterministic]
    families = sorted({gate.family for gate in stochastic})
    family_index = {name: index for index, name in enumerate(families)}
    rng = np.random.default_rng(seed)
    successes = 0
    batch_size = 100
    for start in range(0, pseudoexperiments, batch_size):
        batch = min(batch_size, pseudoexperiments - start)
        family_noise = rng.standard_normal((batch, len(families)))
        passed = np.ones(batch, dtype=np.bool_)
        injected = False
        for gate in stochastic:
            count = round_count if gate.stage == "round" else trajectory_count
            standard_error = gate.normalized_sd * math.sqrt(2.0 / count)
            cell_noise = rng.standard_normal(batch)
            noise = (
                math.sqrt(correlation)
                * family_noise[:, family_index[gate.family]]
                + math.sqrt(1.0 - correlation) * cell_noise
            )
            effect = 0.0
            if (
                injected_family is not None
                and gate.family == injected_family
                and not injected
            ):
                effect = effect_ratio
                injected = True
            estimate = effect + standard_error * noise
            gate_pass = np.abs(estimate) + Z_TOST * standard_error <= 1.0
            passed &= gate_pass
        if injected_family is not None and not injected:
            raise RuntimeError(f"mutant family absent: {injected_family}")
        successes += int(np.count_nonzero(passed))
    return successes


def _power_rows(
    config: Mapping[str, Any],
    blueprint: Sequence[GateBlueprint],
) -> tuple[list[dict[str, object]], dict[str, int] | None]:
    power = config["power_model"]
    trials = int(power["pseudoexperiments"])
    correlation = float(power["family_shared_correlation"])
    root_seed = int(config["splits"]["power_rng"]["start"])
    rows: list[dict[str, object]] = []
    selected: dict[str, int] | None = None
    pair_index = 0
    for round_count in config["candidate_sample_counts"]["round"]:
        for trajectory_count in config["candidate_sample_counts"]["trajectory"]:
            outcome_bounds: dict[str, tuple[float, float]] = {}
            for replica_index, label in enumerate(
                ("same_backend_a", "same_backend_b", "ab_zero_effect")
            ):
                successes = _global_equivalence_trials(
                    blueprint,
                    round_count=round_count,
                    trajectory_count=trajectory_count,
                    pseudoexperiments=trials,
                    correlation=correlation,
                    seed=root_seed + pair_index * 100 + replica_index,
                )
                lower, upper = _wilson(successes, trials)
                outcome_bounds[label] = (lower, upper)
                rows.append({
                    "row_type": "null_power",
                    "round_count": round_count,
                    "trajectory_count": trajectory_count,
                    "case": label,
                    "family": "all",
                    "effect_ratio": 0.0,
                    "successes": successes,
                    "trials": trials,
                    "point": successes / trials,
                    "wilson_lcb": lower,
                    "wilson_ucb": upper,
                })

            mutant_pass = True
            for mutation_index, (name, mutant) in enumerate(
                sorted(config["alternative_mutants"].items())
            ):
                false_passes = _global_equivalence_trials(
                    blueprint,
                    round_count=round_count,
                    trajectory_count=trajectory_count,
                    pseudoexperiments=trials,
                    correlation=correlation,
                    seed=root_seed + pair_index * 100 + 10 + mutation_index,
                    injected_family=mutant["family"],
                    effect_ratio=float(mutant["effect_ratio"]),
                )
                false_lcb, false_ucb = _wilson(false_passes, trials)
                detections = trials - false_passes
                detect_lcb, detect_ucb = _wilson(detections, trials)
                this_pass = (
                    false_ucb
                    <= float(power["outside_margin_false_pass_ucb_maximum"])
                    and detect_lcb
                    >= float(power["outside_margin_detection_power_lcb_minimum"])
                )
                mutant_pass &= this_pass
                rows.append({
                    "row_type": "alternative_power",
                    "round_count": round_count,
                    "trajectory_count": trajectory_count,
                    "case": name,
                    "family": mutant["family"],
                    "effect_ratio": float(mutant["effect_ratio"]),
                    "successes": false_passes,
                    "trials": trials,
                    "point": false_passes / trials,
                    "wilson_lcb": false_lcb,
                    "wilson_ucb": false_ucb,
                    "detection_wilson_lcb": detect_lcb,
                    "detection_wilson_ucb": detect_ucb,
                    "power_gate_pass": this_pass,
                })

            null_pass = (
                outcome_bounds["same_backend_a"][0]
                >= float(power["same_backend_global_equivalence_power_lcb_minimum"])
                and outcome_bounds["same_backend_b"][0]
                >= float(power["same_backend_global_equivalence_power_lcb_minimum"])
                and outcome_bounds["ab_zero_effect"][0]
                >= float(power["ab_zero_effect_global_equivalence_power_lcb_minimum"])
            )
            pair_pass = null_pass and mutant_pass
            rows.append({
                "row_type": "candidate_decision",
                "round_count": round_count,
                "trajectory_count": trajectory_count,
                "case": "all_power_gates",
                "family": "all",
                "effect_ratio": "",
                "successes": "",
                "trials": trials,
                "point": "",
                "wilson_lcb": "",
                "wilson_ucb": "",
                "power_gate_pass": pair_pass,
            })
            if selected is None and pair_pass:
                selected = {
                    "round_sample_count": round_count,
                    "trajectory_sample_count": trajectory_count,
                }
            pair_index += 1
    return rows, selected


def build_report(
    root: Path | None = None,
    *,
    config_override: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, object]], list[GateBlueprint]]:
    base = (root or _root()).resolve()
    config_path = base / CONFIG_PATH
    config = (
        dict(config_override)
        if config_override is not None
        else _load_json(config_path)
    )
    validate_config(config)
    lineage = build_lineage_receipt(base)
    diagnostic = _load_json(base / DIAGNOSTIC_PATH)
    if diagnostic.get("verdict") != "PASS_FRESH_IQ_SEMANTICS_DIAGNOSTIC":
        raise ValueError("fresh IQ semantics diagnostic is not live PASS")
    blueprint = build_blueprint(config)
    rows, model_selected = _power_rows(config, blueprint)
    readout_power = _load_json(base / READOUT_POWER_PATH)
    if readout_power.get("verdict") != "PASS_FRESH_READOUT_EMPIRICAL_POWER":
        raise ValueError("fresh empirical readout power is not live PASS")
    if readout_power.get("config_sha256") != _sha_bytes(config_path.read_bytes()):
        raise ValueError("empirical readout power config binding is stale")
    empirical_round = int(readout_power["selected_round_sample_count"])
    floor = config["power_model"]["unmodeled_physics_conservative_floor"]
    selected = (
        {
            "round_sample_count": max(
                int(model_selected["round_sample_count"]),
                empirical_round,
                int(floor["round_sample_count"]),
            ),
            "trajectory_sample_count": max(
                int(model_selected["trajectory_sample_count"]),
                int(floor["trajectory_sample_count"]),
            ),
        }
        if model_selected is not None
        else None
    )
    family_counts = {
        family: sum(gate.family == family for gate in blueprint)
        for family in config["families"]
    }
    deterministic_count = sum(gate.deterministic for gate in blueprint)
    decision_rows = [row for row in rows if row["row_type"] == "candidate_decision"]
    selected_row = next(
        (
            row for row in decision_rows
            if model_selected is not None
            and row["round_count"] == model_selected["round_sample_count"]
            and row["trajectory_count"] == model_selected["trajectory_sample_count"]
        ),
        None,
    )
    source = Path(__file__).read_text(encoding="utf-8")
    prohibited = (
        "t9_2_4_dual_backend_" + "cell_ledger.csv",
        "t9_2_4_dual_backend_" + "qualification_source_data.csv",
        "t9_2_4_dual_backend_" + "state_archive.npz",
    )
    gates = {
        "G01_historical_lineage_live": lineage["verdict"] == LINEAGE_PASS,
        "G02_iq_semantics_diagnostic_live": (
            diagnostic.get("verdict") == "PASS_FRESH_IQ_SEMANTICS_DIAGNOSTIC"
        ),
        "G03_old_formal_cell_data_not_referenced": not any(
            value in source for value in prohibited
        ),
        "G04_seed_splits_pairwise_disjoint": True,
        "G05_formal_pool_not_accessed": (
            config["power_model"]["formal_pool_access"] is False
        ),
        "G06_pilot_only_selects_sample_count": (
            config["historical_policy"]["pilot_may_choose"]
            == ["round_sample_count", "trajectory_sample_count"]
        ),
        "G07_complete_seven_family_blueprint": (
            set(family_counts) == set(config["families"])
            and all(value > 0 for value in family_counts.values())
        ),
        "G08_every_blueprint_id_unique": (
            len(blueprint) == len({gate.gate_id for gate in blueprint})
        ),
        "G09_all_blueprint_metrics_have_margin": all(
            gate.metric in config["margins"] for gate in blueprint
        ),
        "G10_iut_tost_alpha_frozen": (
            config["statistical_procedure"]["global_test"]
            == "intersection_union_equivalence"
            and config["statistical_procedure"]["cell_test"]
            == "two_one_sided_tests"
            and config["statistical_procedure"]["global_type_i_error"] == 0.05
        ),
        "G11_raw_log_evidence_diagnostic_only": (
            config["statistical_procedure"]["raw_log_evidence"]
            == "diagnostic only; never a cross-gain primary gate"
        ),
        "G12_mixed_unit_composite_removed": (
            config["statistical_procedure"]["mixed_unit_vector_max"] is False
        ),
        "G13_two_tail_cutoff_increments_primary": (
            config["formal_matrix"]["primary_cutoff_increments"]
            == [[12, 16], [16, 20]]
        ),
        "G14_historical_cutoff_increment_retained_diagnostic": (
            config["formal_matrix"]["historical_cutoff_increment_diagnostic_only"]
            == [8, 12]
        ),
        "G15_same_backend_a_power_covered": any(
            row["case"] == "same_backend_a" for row in rows
        ),
        "G16_same_backend_b_power_covered": any(
            row["case"] == "same_backend_b" for row in rows
        ),
        "G17_ab_zero_effect_power_covered": any(
            row["case"] == "ab_zero_effect" for row in rows
        ),
        "G18_all_alternative_mutants_covered": (
            {
                row["case"] for row in rows
                if row["row_type"] == "alternative_power"
            }
            == set(config["alternative_mutants"])
        ),
        "G19_smallest_passing_candidate_selected": (
            model_selected is not None
            and selected_row is not None
            and selected_row.get("power_gate_pass") is True
            and all(
                row.get("power_gate_pass") is not True
                for row in decision_rows[: decision_rows.index(selected_row)]
            )
        ),
        "G20_selected_counts_fit_frozen_formal_pools": (
            selected is not None
            and selected["round_sample_count"]
            <= config["splits"]["formal_round_backend_a"]["count"]
            and selected["round_sample_count"]
            <= config["splits"]["formal_round_backend_b"]["count"]
            and selected["trajectory_sample_count"]
            <= config["splits"]["formal_trajectory_backend_a"]["count"]
            and selected["trajectory_sample_count"]
            <= config["splits"]["formal_trajectory_backend_b"]["count"]
        ),
        "G21_empirical_readout_power_live_and_bound": (
            readout_power.get("verdict") == "PASS_FRESH_READOUT_EMPIRICAL_POWER"
            and readout_power.get("config_sha256")
            == _sha_bytes(config_path.read_bytes())
            and empirical_round
            in config["candidate_sample_counts"]["round"]
        ),
        "G22_unmodeled_physics_floor_applied": (
            selected is not None
            and selected["round_sample_count"] >= floor["round_sample_count"]
            and selected["trajectory_sample_count"]
            >= floor["trajectory_sample_count"]
        ),
    }
    report: dict[str, Any] = {
        "task_id": TASK_ID,
        "schema_version": SCHEMA_VERSION,
        "purpose": "design_only_power_not_scientific_outcome",
        "formal_seed_values_accessed": False,
        "historical_formal_cell_data_accessed": False,
        "old_outcome_used_to_choose_design": False,
        "bindings": {
            "config": {
                "path": CONFIG_PATH,
                "sha256": _sha_bytes(config_path.read_bytes()),
            },
            "lineage_analysis_sha256": lineage["analysis_sha256"],
            "iq_diagnostic_analysis_sha256": diagnostic["analysis_sha256"],
            "readout_power": {
                "path": READOUT_POWER_PATH,
                "analysis_sha256": readout_power["analysis_sha256"],
                "config_sha256": readout_power["config_sha256"],
            },
        },
        "procedure": config["statistical_procedure"],
        "power_model": config["power_model"],
        "blueprint": {
            "gate_count": len(blueprint),
            "stochastic_gate_count": len(blueprint) - deterministic_count,
            "deterministic_gate_count": deterministic_count,
            "family_counts": family_counts,
            "sha256": _sha([
                {
                    "gate_id": gate.gate_id,
                    "family": gate.family,
                    "stage": gate.stage,
                    "metric": gate.metric,
                    "margin": gate.margin,
                    "normalized_sd": gate.normalized_sd,
                    "deterministic": gate.deterministic,
                }
                for gate in blueprint
            ]),
        },
        "candidate_pair_count": len(decision_rows),
        "model_selected_sample_counts": model_selected,
        "empirical_readout_selected_round_count": empirical_round,
        "selected_sample_counts": selected,
        "model_selection_rule": config["power_model"]["model_selection_rule"],
        "final_selection_rule": config["power_model"]["final_selection_rule"],
        "unmodeled_physics_conservative_floor": floor,
        "gates": gates,
        "gate_summary": {
            "passed": sum(value is True for value in gates.values()),
            "total": len(gates),
        },
    }
    report["verdict"] = PASS_VERDICT if all(gates.values()) else FAIL_VERDICT
    report["analysis_sha256"] = _sha(report)
    return report, rows, blueprint


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def write_artifacts(root: Path | None = None) -> dict[str, Any]:
    base = (root or _root()).resolve()
    report, rows, blueprint = build_report(base)
    if report["verdict"] != PASS_VERDICT:
        raise RuntimeError(f"design power failed: {report['gates']}")
    report_path = base / "docs/t_risk_20260726_01_design_power.json"
    source_path = base / "docs/t_risk_20260726_01_design_power_source_data.csv"
    _atomic_text(
        report_path,
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    combined: list[dict[str, object]] = list(rows)
    for gate in blueprint:
        combined.append({
            "row_type": "gate_blueprint",
            "round_count": "",
            "trajectory_count": "",
            "case": gate.gate_id,
            "family": gate.family,
            "effect_ratio": "",
            "successes": "",
            "trials": "",
            "point": gate.margin,
            "wilson_lcb": gate.normalized_sd,
            "wilson_ucb": "",
            "power_gate_pass": not gate.deterministic,
            "metric": gate.metric,
            "stage": gate.stage,
            "deterministic": gate.deterministic,
        })
    fieldnames = sorted({key for row in combined for key in row})
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(combined)
        stream.seek(0)
        _atomic_text(source_path, stream.read())
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Write the preregistered fresh-twin design-power artifacts."
    )
    parser.parse_args(argv)
    report = write_artifacts()
    print(json.dumps({
        "verdict": report["verdict"],
        "analysis_sha256": report["analysis_sha256"],
        "gate_summary": report["gate_summary"],
        "selected_sample_counts": report["selected_sample_counts"],
        "blueprint_gate_count": report["blueprint"]["gate_count"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
