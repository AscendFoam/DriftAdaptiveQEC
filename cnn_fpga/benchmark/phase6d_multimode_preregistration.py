"""T6.20.3 prospective Phase-6D multimode split and statistics seal.

This builder is intentionally usable only before any registered pilot/formal
outcome exists.  It expands the compact configuration into an immutable cell
manifest, independently recomputes power and compute arithmetic, and rejects
opened T6.18.3 seeds/factors or any cross-split reuse.
"""

from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "T6.20.3"
SCHEMA_VERSION = "t6.20.3-phase6d-multimode-preregistration-report-v1"
VERDICT = "PASS_PHASE6D_MULTIMODE_PREREGISTRATION_SEALED"

CONFIG = ROOT / "configs" / "phase6d" / "t6_20_3_multimode_preregistration.json"
DEFAULT_REPORT = ROOT / "docs" / "t6_20_3_phase6d_multimode_preregistration.json"
DEFAULT_SOURCE_DATA = ROOT / "docs" / "t6_20_3_phase6d_multimode_preregistration_source_data.csv"
DEFAULT_MARKDOWN = ROOT / "docs" / "phase6d_multimode_preregistration.md"

SPLIT_IDS = ("train", "calibration", "pilot", "formal")
EXPECTED_ROLES = {
    "train": "fit posterior/approximation parameters only; no claim, threshold or candidate selection",
    "calibration": "freeze posterior calibration, likelihood truncation and finite candidate grid without choosing a winner",
    "pilot": "select exactly one candidate and strongest eligible deployable baseline once",
    "formal": "untouched confirmatory evaluation; no fit, calibration, threshold change, candidate replacement or denominator change",
}
EXPECTED_SCENARIOS = {
    "stationary_control", "mean_drift", "variance_drift", "correlation_drift",
    "periodic_drift", "ou_drift", "random_walk", "step_calibration_shift",
    "telegraph_drift", "burst_outlier", "heavy_tail", "compound_ood",
    "likelihood_mismatch",
}

ARTIFACT_PATHS = {
    "config": CONFIG,
    "dual_lane_contract": ROOT / "docs" / "t6_20_2_dual_evidence_lane_contract.json",
    "baseline_registry": ROOT / "docs" / "multimode_strong_baseline_registry.md",
    "opened_t6_18_3_config": ROOT / "configs" / "literature" / "t6_18_3_multimode_drift.json",
    "opened_t6_18_3_report": ROOT / "docs" / "t6_18_3_multimode_posterior_weighted_cpd.json",
    "official_julia_project": ROOT / "configs" / "literature" / "t6_18_2_julia_env" / "Project.toml",
    "official_julia_manifest": ROOT / "configs" / "literature" / "t6_18_2_julia_env" / "Manifest.toml",
    "implementation": Path(__file__).resolve(),
}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _binding(path: Path) -> dict[str, Any]:
    return {"path": _relative(path), "sha256": _sha256(path), "bytes": path.stat().st_size}


def _live(binding: Mapping[str, Any]) -> bool:
    path = ROOT / str(binding["path"])
    return path.is_file() and path.stat().st_size == binding.get("bytes") and _sha256(path) == binding.get("sha256")


def _atomic_text(value: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _atomic_json(value: Mapping[str, Any], path: Path) -> None:
    _atomic_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", path)


def _split_seeds(split: Mapping[str, Any]) -> list[int]:
    start = split["seed_start"]
    count = split["seed_count"]
    if isinstance(start, bool) or not isinstance(start, int) or start <= 0:
        raise ValueError("seed_start must be a positive integer")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("seed_count must be a positive integer")
    return list(range(start, start + count))


def _hash_order(length: int, token: str) -> list[int]:
    return sorted(
        range(length),
        key=lambda index: hashlib.sha256(f"{token}:{index}".encode("ascii")).digest(),
    )


def _spatial_pattern(distance: int, key: int) -> dict[str, Any]:
    coordinates = 2 * distance * distance
    order = _hash_order(coordinates, f"pattern:{distance}:{key}")
    signs = [-1] * coordinates
    for index in order[: coordinates // 2]:
        signs[index] = 1
    permutation = _hash_order(coordinates, f"permutation:{distance}:{key}")
    payload = {"distance": distance, "key": key, "signs": signs, "permutation": permutation}
    return {**payload, "pattern_sha256": _canonical_sha256(payload)}


def _expanded_split(split: Mapping[str, Any]) -> dict[str, Any]:
    return {
        **dict(split),
        "seeds": _split_seeds(split),
        "spatial_patterns": [
            _spatial_pattern(int(distance), int(key))
            for distance in split["distances"]
            for key in split["spatial_pattern_keys"]
        ],
    }


def _choice(values: Sequence[Any], token: str) -> Any:
    if not values:
        raise ValueError("cannot choose from an empty registered factor")
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    return values[int.from_bytes(digest[:8], "big") % len(values)]


def _execution_cells(splits: Sequence[Mapping[str, Any]], scenarios: Sequence[str]) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for split in splits:
        sigma_strata = [
            (distance, sigma)
            for distance in split["distances"]
            for sigma in split["base_sigmas"]
        ]
        patterns_by_distance = {
            distance: [row for row in split["spatial_patterns"] if row["distance"] == distance]
            for distance in split["distances"]
        }
        for seed_index, seed in enumerate(split["seeds"]):
            distance, sigma = sigma_strata[seed_index % len(sigma_strata)]
            pattern = patterns_by_distance[distance][seed_index % len(patterns_by_distance[distance])]
            for family in scenarios:
                token = f"{split['split_id']}:{seed}:{family}"
                covariance = _choice(split["covariance_designs"], token + ":cov")
                aux_noise = _choice(split["aux_noise"], token + ":aux")
                cell = {
                    "cell_id": f"{split['split_id']}:{seed}:{family}",
                    "split_id": split["split_id"],
                    "seed": seed,
                    "scenario_family": family,
                    "distance": distance,
                    "base_sigma": sigma,
                    "spatial_pattern_sha256": pattern["pattern_sha256"],
                    "variance_law_id": _choice(split["variance_law_ids"], token + ":variance"),
                    "covariance_family": covariance["family"],
                    "covariance_rho": covariance["rho"],
                    "transition_rate_per_1000_rounds": _choice(split["transition_rates_per_1000_rounds"], token + ":rate"),
                    "amplitude": _choice(split["amplitudes"], token + ":amplitude"),
                    "duration_rounds": _choice(split["durations_rounds"], token + ":duration"),
                    "readout_flip": aux_noise["readout_flip"],
                    "reset_failure": aux_noise["reset_failure"],
                    "missing_syndrome": aux_noise["missing_syndrome"],
                    "rounds": split["rounds_per_scenario_cell"],
                    "stratum_id": f"d{distance}:sigma{sigma:.3f}",
                }
                cells.append({**cell, "cell_sha256": _canonical_sha256(cell)})
    return cells


def _power_analysis(config: Mapping[str, Any]) -> dict[str, Any]:
    plan = config["power_plan"]
    alpha_per = plan["familywise_alpha"] / plan["bonferroni_comparators"]
    z_critical = NormalDist().inv_cdf(1.0 - alpha_per)
    z_power = NormalDist().inv_cdf(plan["target_power"])
    delta = plan["target_absolute_difference"]
    sd = plan["paired_cluster_difference_sd_ceiling"]
    required = math.ceil(((z_critical + z_power) * sd / delta) ** 2)
    actual = plan["formal_cluster_count"]
    achieved = NormalDist().cdf(math.sqrt(actual) * delta / sd - z_critical)
    sensitivity = []
    for candidate_sd in (0.020, 0.024, 0.028, 0.032, 0.036):
        sensitivity.append(
            {
                "paired_cluster_sd": candidate_sd,
                "planned_clusters": actual,
                "approximate_power": NormalDist().cdf(
                    math.sqrt(actual) * delta / candidate_sd - z_critical
                ),
            }
        )
    return {
        "alpha_per_comparator": alpha_per,
        "one_sided_z_critical": z_critical,
        "z_target_power": z_power,
        "required_clusters": required,
        "planned_clusters": actual,
        "achieved_power_at_registered_sd": achieved,
        "sensitivity": sensitivity,
        "design_only_not_final_inference": True,
    }


def _opened_audit(config: Mapping[str, Any], splits: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    opened_config = _load(ROOT / config["opened_development_reference"]["config"])
    opened_seeds = set(opened_config["formal"]["seeds"]) | set(opened_config["pilot"]["seeds"])
    registered_seeds = {seed for split in splits for seed in split["seeds"]}
    forbidden = config["opened_development_reference"]["forbidden_exact_values"]
    registered_sigmas = {value for split in splits for value in split["base_sigmas"]}
    registered_amplitudes = {value for split in splits for value in split["amplitudes"]}
    registered_durations = {value for split in splits for value in split["durations_rounds"]}
    opened_fixed_patterns = {
        distance: [1] * (distance * distance) + [-1] * (distance * distance)
        for distance in (3, 5)
    }
    patterns = [row for split in splits for row in split["spatial_patterns"]]
    return {
        "opened_config_binding": _binding(ROOT / config["opened_development_reference"]["config"]),
        "opened_report_binding": _binding(ROOT / config["opened_development_reference"]["report"]),
        "seed_overlap": sorted(opened_seeds & registered_seeds),
        "base_sigma_overlap": sorted(set(forbidden["base_sigma"]) & registered_sigmas),
        "amplitude_overlap": sorted(
            ({abs(value) for value in forbidden["smooth_amplitude"] + forbidden["calibration_after"] + forbidden["telegraph_levels"]})
            & registered_amplitudes
        ),
        "duration_overlap": sorted(set(forbidden["smooth_period_rounds"] + forbidden["telegraph_dwell_rounds"]) & registered_durations),
        "opened_fixed_spatial_pattern_matches": [
            row["pattern_sha256"]
            for row in patterns
            if row["signs"] == opened_fixed_patterns[row["distance"]]
        ],
        "opened_distance_three_reuse_is_allowed_only_with_new_factors": 3 in {value for split in splits for value in split["distances"]},
        "t6_18_3_status": "OPENED_DEVELOPMENT_ONLY_NEVER_FORMAL",
        "opened_report_verdict": _load(ROOT / config["opened_development_reference"]["report"])["verdict"],
    }


def _factor_disjointness(splits: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    fields: dict[str, list[tuple[str, Any]]] = {
        "seeds": [],
        "base_sigmas": [],
        "spatial_pattern_keys": [],
        "spatial_pattern_sha256": [],
        "variance_law_ids": [],
        "covariance_designs": [],
        "transition_rates_per_1000_rounds": [],
        "amplitudes": [],
        "durations_rounds": [],
        "aux_noise": [],
    }
    for split in splits:
        split_id = split["split_id"]
        for value in split["seeds"]:
            fields["seeds"].append((split_id, value))
        for name in ("base_sigmas", "spatial_pattern_keys", "variance_law_ids", "transition_rates_per_1000_rounds", "amplitudes", "durations_rounds"):
            fields[name].extend((split_id, value) for value in split[name])
        fields["spatial_pattern_sha256"].extend((split_id, row["pattern_sha256"]) for row in split["spatial_patterns"])
        fields["covariance_designs"].extend((split_id, (row["family"], row["rho"])) for row in split["covariance_designs"])
        fields["aux_noise"].extend(
            (split_id, (row["readout_flip"], row["reset_failure"], row["missing_syndrome"]))
            for row in split["aux_noise"]
        )
    result: dict[str, Any] = {}
    for name, rows in fields.items():
        owners: dict[str, set[str]] = {}
        values: dict[str, Any] = {}
        for split_id, value in rows:
            key = json.dumps(value, sort_keys=True)
            owners.setdefault(key, set()).add(split_id)
            values[key] = value
        overlaps = [values[key] for key, split_ids in owners.items() if len(split_ids) > 1]
        result[name] = {"unique_values": len(owners), "cross_split_overlap": overlaps, "passed": not overlaps}
    return result


def _formal_balance(cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    formal = [row for row in cells if row["split_id"] == "formal"]
    seed_strata: dict[int, str] = {}
    for row in formal:
        seed_strata.setdefault(int(row["seed"]), str(row["stratum_id"]))
        if seed_strata[int(row["seed"])] != row["stratum_id"]:
            raise ValueError("one formal seed must retain one distance/sigma stratum")
    counts: dict[str, int] = {}
    for stratum in seed_strata.values():
        counts[stratum] = counts.get(stratum, 0) + 1
    family_clusters = {
        family: len({row["seed"] for row in formal if row["scenario_family"] == family})
        for family in EXPECTED_SCENARIOS
    }
    return {
        "seed_stratum_counts": dict(sorted(counts.items())),
        "clusters_per_scenario_family": dict(sorted(family_clusters.items())),
        "all_strata_equal": len(set(counts.values())) == 1,
        "all_families_have_all_formal_clusters": set(family_clusters.values()) == {len(seed_strata)},
    }


def _seal_absence(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for relative in config["future_outcome_paths"]:
        path = ROOT / relative
        rows.append({"path": relative, "exists_at_seal": path.exists()})
    return rows


def _source_rows(report: Mapping[str, Any]) -> list[dict[str, str]]:
    records: list[tuple[str, str, Mapping[str, Any]]] = []
    for split in report["splits"]:
        records.append(("split", split["split_id"], split))
    for cell in report["execution_cells"]:
        records.append(("execution_cell", cell["cell_id"], cell))
    for row in report["power_analysis"]["sensitivity"]:
        records.append(("power_sensitivity", f"sd={row['paired_cluster_sd']:.3f}", row))
    for artifact_id, binding in report["artifact_registry"].items():
        records.append(("artifact_binding", artifact_id, binding))
    for row in report["seal_absence_proof"]:
        records.append(("outcome_absence", row["path"], row))
    output: list[dict[str, str]] = []
    for record_type, record_id, payload in records:
        payload_json = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        output.append(
            {
                "record_type": record_type,
                "record_id": record_id,
                "payload_json": payload_json,
                "canonical_sha256": hashlib.sha256(payload_json.encode("utf-8")).hexdigest(),
            }
        )
    return output


def _write_source_data(report: Mapping[str, Any], path: Path) -> int:
    rows = _source_rows(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("record_type", "record_id", "payload_json", "canonical_sha256"))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
    return len(rows)


def _csv_lossless(report: Mapping[str, Any], path: Path) -> bool:
    if not path.is_file():
        return False
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return rows == _source_rows(report) and all(
        row["canonical_sha256"] == hashlib.sha256(row["payload_json"].encode("utf-8")).hexdigest()
        for row in rows
    )


def _render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Phase 6D multimode 确认性预注册",
        "",
        f"> seal：`{report['seal_id']}`；verdict：`{report.get('verdict', 'PENDING_BUILD')}`。T6.18.3 仅作 opened development，不进入本 formal。",
        "",
        "## 四分割",
        "",
        "| Split | 独立 clusters | d | sigma | 每 family rounds | 角色 |",
        "| --- | ---: | --- | --- | ---: | --- |",
    ]
    for split in report["splits"]:
        lines.append(
            f"| `{split['split_id']}` | {len(split['seeds'])} | {split['distances']} | {split['base_sigmas']} | "
            f"{split['rounds_per_scenario_cell']} | {split['role']} |"
        )
    power = report["power_analysis"]
    lines += [
        "",
        "## 功效与 formal 规模",
        "",
        f"- 60 个 formal seed-cluster；13 个 family；每方法 {report['compute_arithmetic']['formal_physical_rounds_per_method']:,} physical rounds。",
        f"- 12-comparator Bonferroni 一侧设计：required clusters={power['required_clusters']}，planned={power['planned_clusters']}，approximate power={power['achieved_power_at_registered_sd']:.4f}。",
        "- 功效计算只用于冻结 N；正式结论只由 paired simultaneous bootstrap CI 决定，pilot 后不扩样。",
        "",
        "## 统计、tail 与缺失处理",
        "",
        f"- `{report['statistics']['independent_cluster']}`；bootstrap={report['statistics']['paired_bootstrap_resamples']:,}。",
        f"- SOTA 门：relative LER simultaneous 95% LCB `>{report['statistics']['formal_sota_relative_ler_95_lcb_min_exclusive']:.0%}`，absolute LCB `>0`。",
        "- calibration/telegraph worst-window 与 CVaR95 独立成 family；stationary/OOD margin 不得由 aggregate 掩盖。",
        "- proposed failure 保守计错并使 integrity gate 失败；required baseline failure 关闭 SOTA，不得删除对手；零填补禁止。",
        "",
        "## 不可变与访问规则",
        "",
        "- train→calibration→pilot→formal 单向；pilot 只选一次，锁 candidate/baseline/checkpoint/config hash 后才可打开 formal。",
        "- outcome/significance/precision 不得触发 early stop 或扩样；formal 只允许资源、完整性、数值或工具失败终止。",
        "- seal 后 amendment 只能修正文案或 locator，`affects_analysis=false`；分析字段变化必须新建前瞻协议，v1 结果不能救援。",
        "",
        "## Opened-data 隔离",
        "",
        f"- seed overlap={report['opened_development_audit']['seed_overlap']}；sigma overlap={report['opened_development_audit']['base_sigma_overlap']}；spatial fixed-pattern matches={report['opened_development_audit']['opened_fixed_spatial_pattern_matches']}。",
        f"- 全部 cross-split factor overlap 均为空：{all(row['passed'] for row in report['factor_disjointness'].values())}。",
        "",
    ]
    return "\n".join(lines)


def evaluate_gates(report: Mapping[str, Any], *, check_live_files: bool = True) -> dict[str, bool]:
    config = report["config_snapshot"]
    splits = {row["split_id"]: row for row in report["splits"]}
    cells = report["execution_cells"]
    expected_config = _load(CONFIG)
    expected_splits = [_expanded_split(row) for row in config["splits"]]
    recomputed_opened = _opened_audit(config, list(splits.values()))
    recomputed_disjoint = _factor_disjointness(list(splits.values()))
    recomputed_cells = _execution_cells(list(splits.values()), list(config["scenario_families"]))
    recomputed_balance = _formal_balance(cells)
    recomputed_power = _power_analysis(config)
    formal_cells = [row for row in cells if row["split_id"] == "formal"]
    expected_cells = sum(len(row["seeds"]) * len(EXPECTED_SCENARIOS) for row in splits.values())
    cell_ids = {row["cell_id"] for row in cells}
    opened = report["opened_development_audit"]
    disjoint = report["factor_disjointness"]
    power = report["power_analysis"]
    statistics = report["statistics"]
    compute = report["compute_arithmetic"]
    source_path = ROOT / report["source_data"]["path"]
    markdown_path = ROOT / report["markdown"]["path"]
    expected_formal_rounds = len(splits["formal"]["seeds"]) * len(EXPECTED_SCENARIOS) * splits["formal"]["rounds_per_scenario_cell"]
    return {
        "G01_identity_and_config_schema_are_frozen": report.get("task_id") == TASK_ID and report.get("schema_version") == SCHEMA_VERSION and config == expected_config and config.get("protocol_id") == "PHASE6D-MULTIMODE-UNTOUCHED-FORMAL-V1" and config.get("task_signature") == "MULTIMODE_SOFTWARE_ALGORITHM",
        "G02_exact_four_split_roles_and_one_way_access": tuple(splits) == SPLIT_IDS and report["splits"] == expected_splits and all(splits[key]["role"] == EXPECTED_ROLES[key] for key in SPLIT_IDS) and "never read pilot or formal" in splits["train"]["access_policy"] and "one deterministic selection pass" in splits["pilot"]["access_policy"] and "read once" in splits["formal"]["access_policy"],
        "G03_all_seed_namespaces_are_unique_and_opened_disjoint": report["opened_development_audit"] == recomputed_opened and report["factor_disjointness"] == recomputed_disjoint and disjoint["seeds"]["passed"] and opened["seed_overlap"] == [] and len({seed for row in splits.values() for seed in row["seeds"]}) == sum(len(row["seeds"]) for row in splits.values()),
        "G04_spatial_signs_permutations_are_explicit_unique_and_not_opened": report["opened_development_audit"] == recomputed_opened and report["factor_disjointness"] == recomputed_disjoint and disjoint["spatial_pattern_keys"]["passed"] and disjoint["spatial_pattern_sha256"]["passed"] and opened["opened_fixed_spatial_pattern_matches"] == [] and all(sum(pattern["signs"]) == 0 and sorted(pattern["permutation"]) == list(range(2 * pattern["distance"] * pattern["distance"])) for row in splits.values() for pattern in row["spatial_patterns"]),
        "G05_variance_and_covariance_designs_are_cross_split_disjoint": report["factor_disjointness"] == recomputed_disjoint and disjoint["variance_law_ids"]["passed"] and disjoint["covariance_designs"]["passed"] and all(row["rho"] != 0.0 for split in splits.values() for row in split["covariance_designs"]),
        "G06_sigma_transition_amplitude_duration_and_aux_are_disjoint": report["opened_development_audit"] == recomputed_opened and report["factor_disjointness"] == recomputed_disjoint and all(disjoint[key]["passed"] for key in ("base_sigmas", "transition_rates_per_1000_rounds", "amplitudes", "durations_rounds", "aux_noise")) and opened["base_sigma_overlap"] == opened["amplitude_overlap"] == opened["duration_overlap"] == [],
        "G07_execution_manifest_is_complete_unique_and_hash_bound": cells == recomputed_cells and len(cells) == len(cell_ids) == expected_cells and all(row["scenario_family"] in EXPECTED_SCENARIOS and row["cell_sha256"] == _canonical_sha256({key: value for key, value in row.items() if key != "cell_sha256"}) for row in cells),
        "G08_formal_strata_and_families_are_balanced": report["formal_balance"] == recomputed_balance and report["formal_balance"]["all_strata_equal"] is True and report["formal_balance"]["all_families_have_all_formal_clusters"] is True and set(report["formal_balance"]["seed_stratum_counts"].values()) == {10} and set(report["formal_balance"]["clusters_per_scenario_family"].values()) == {60} and len(formal_cells) == 60 * len(EXPECTED_SCENARIOS),
        "G09_seal_precedes_every_registered_outcome": report["seal_state"] == "SEALED_PRE_OUTCOME" and all(row["exists_at_seal"] is False for row in report["seal_absence_proof"]) and config["selection_contract"]["failed_pilot_verdict"] == "NO_GO_MULTIMODE_PILOT_NO_FORMAL_ACCESS",
        "G10_power_is_recomputed_and_fixed_n_meets_target": power == recomputed_power and power["required_clusters"] <= power["planned_clusters"] == 60 and power["achieved_power_at_registered_sd"] >= config["power_plan"]["target_power"] and config["power_plan"]["no_variance_based_resize_after_pilot"] is True,
        "G11_statistics_use_paired_clusters_and_fixed_simultaneous_divisor": statistics["independent_cluster"].startswith("seed;") and statistics["paired_bootstrap_resamples"] == 50000 and statistics["maximum_ranked_deployable_comparators"] == 12 and "fixed Bonferroni divisor 12" in statistics["simultaneous_method"] and "undefined" in statistics["relative_improvement"],
        "G12_selection_is_single_pass_and_formal_reselection_is_prohibited": config["selection_contract"]["pilot_selection_passes"] == 1 and config["selection_contract"]["formal_reselection_prohibited"] is True and config["selection_contract"]["pilot_min_relative_ler_point_improvement"] == 0.10 and config["selection_contract"]["pilot_absolute_improvement_95_lcb_exclusive"] == 0.0,
        "G13_sota_tail_stationary_ood_and_directional_gates_are_fixed": statistics["formal_sota_relative_ler_95_lcb_min_exclusive"] == 0.10 and statistics["formal_absolute_ler_95_lcb_min_exclusive"] == 0.0 and len(statistics["tail_endpoints"]) == 4 and statistics["stationary_relative_degradation_95_ucb_max"] == 0.02 and statistics["each_ood_relative_degradation_95_ucb_max"] == 0.05 and "no stratum may reverse" in statistics["directional_strata"],
        "G14_missingness_is_zero_tolerance_and_fail_closed": config["missingness"]["missing_cell_fraction_max"] == 0.0 and config["missingness"]["zero_imputation_prohibited"] is True and "count remaining" in config["missingness"]["proposed_failure"] and "SOTA wording closes" in config["missingness"]["baseline_failure"] and "same physical trace hash" in config["missingness"]["paired_trace_rule"],
        "G15_stopping_has_no_outcome_or_precision_extension": config["stopping_rules"]["outcome_based_early_stop"] is False and config["stopping_rules"]["precision_or_significance_based_extension"] is False and config["stopping_rules"]["formal_open_once"] is True and "favorable subset" in config["stopping_rules"]["partial_results_policy"],
        "G16_compute_arithmetic_and_caps_are_exact": compute["formal_physical_rounds_per_method"] == expected_formal_rounds == config["compute_caps"]["formal_max_physical_rounds_per_method"] and compute["formal_max_method_decodes"] == expected_formal_rounds * splits["formal"]["max_method_count"] == config["compute_caps"]["formal_max_method_decodes"] and config["compute_caps"]["formal_core_hours"] == 12000 and config["compute_caps"]["host_memory_gib"] == 64,
        "G17_amendments_cannot_change_analysis_or_rescue_v1": config["amendment_policy"]["affects_analysis_must_be_false"] is True and set(config["amendment_policy"]["required_fields"]) == {"timestamp_utc", "author", "reason", "old_sha256", "new_sha256", "affects_analysis", "affected_fields"} and "invalidate protocol v1" in config["amendment_policy"]["analysis_change_rule"],
        "G18_artifact_bindings_are_complete_and_live": set(report["artifact_registry"]) == set(ARTIFACT_PATHS) and all(len(row.get("sha256", "")) == 64 and row.get("bytes", 0) > 0 for row in report["artifact_registry"].values()) and (not check_live_files or all(_live(row) for row in report["artifact_registry"].values())),
        "G19_source_data_and_human_report_are_live": report["source_data"]["rows"] == len(_source_rows(report)) and source_path.is_file() and markdown_path.is_file() and "60 个 formal seed-cluster" in markdown_path.read_text(encoding="utf-8") and (not check_live_files or (_live(report["source_data"]) and _live(report["markdown"]) and _csv_lossless(report, source_path))),
        "G20_one_substantive_mutation_per_gate_fails_closed": report["semantic_mutation_audit"]["count"] == report["semantic_mutation_audit"]["detected"] == 20 and len(report["semantic_mutation_audit"]["cases"]) == 20,
    }


def _mutations(report: Mapping[str, Any]) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []

    def split(value: dict[str, Any], split_id: str) -> dict[str, Any]:
        return next(row for row in value["splits"] if row["split_id"] == split_id)

    def attempt(name: str, target: str, change: Any) -> None:
        candidate = copy.deepcopy(report)
        candidate["semantic_mutation_audit"] = {"count": 20, "detected": 20, "cases": []}
        change(candidate)
        try:
            rejected = not evaluate_gates(candidate, check_live_files=False)[target]
        except Exception:
            rejected = True
        cases.append({"case": name, "target_gate": target, "rejected": rejected})

    attempt("change_protocol_id", "G01_identity_and_config_schema_are_frozen", lambda x: x["config_snapshot"].update(protocol_id="POSTHOC"))
    attempt("calibration_reads_formal", "G02_exact_four_split_roles_and_one_way_access", lambda x: split(x, "train").update(access_policy="read pilot and formal"))
    attempt("reuse_opened_seed", "G03_all_seed_namespaces_are_unique_and_opened_disjoint", lambda x: x["opened_development_audit"].update(seed_overlap=[61830001]))
    attempt("opened_spatial_pattern", "G04_spatial_signs_permutations_are_explicit_unique_and_not_opened", lambda x: x["opened_development_audit"].update(opened_fixed_spatial_pattern_matches=["forged"]))
    attempt("diagonal_covariance_reuse", "G05_variance_and_covariance_designs_are_cross_split_disjoint", lambda x: split(x, "formal")["covariance_designs"][0].update(rho=0.0))
    attempt("reuse_sigma_between_splits", "G06_sigma_transition_amplitude_duration_and_aux_are_disjoint", lambda x: x["factor_disjointness"]["base_sigmas"].update(passed=False, cross_split_overlap=[0.52]))
    attempt("drop_execution_cell", "G07_execution_manifest_is_complete_unique_and_hash_bound", lambda x: x["execution_cells"].pop())
    attempt("unbalance_formal_strata", "G08_formal_strata_and_families_are_balanced", lambda x: x["formal_balance"].update(all_strata_equal=False))
    attempt("formal_outcome_preexists", "G09_seal_precedes_every_registered_outcome", lambda x: x["seal_absence_proof"][0].update(exists_at_seal=True))
    attempt("forge_power", "G10_power_is_recomputed_and_fixed_n_meets_target", lambda x: x["power_analysis"].update(achieved_power_at_registered_sd=0.5))
    attempt("reduce_bootstrap", "G11_statistics_use_paired_clusters_and_fixed_simultaneous_divisor", lambda x: x["statistics"].update(paired_bootstrap_resamples=1000))
    attempt("allow_formal_reselection", "G12_selection_is_single_pass_and_formal_reselection_is_prohibited", lambda x: x["config_snapshot"]["selection_contract"].update(formal_reselection_prohibited=False))
    attempt("weaken_sota_gate", "G13_sota_tail_stationary_ood_and_directional_gates_are_fixed", lambda x: x["statistics"].update(formal_sota_relative_ler_95_lcb_min_exclusive=0.0))
    attempt("zero_impute_missing", "G14_missingness_is_zero_tolerance_and_fail_closed", lambda x: x["config_snapshot"]["missingness"].update(zero_imputation_prohibited=False))
    attempt("significance_extension", "G15_stopping_has_no_outcome_or_precision_extension", lambda x: x["config_snapshot"]["stopping_rules"].update(precision_or_significance_based_extension=True))
    attempt("halve_formal_rounds", "G16_compute_arithmetic_and_caps_are_exact", lambda x: x["compute_arithmetic"].update(formal_physical_rounds_per_method=x["compute_arithmetic"]["formal_physical_rounds_per_method"] // 2))
    attempt("analysis_amendment", "G17_amendments_cannot_change_analysis_or_rescue_v1", lambda x: x["config_snapshot"]["amendment_policy"].update(affects_analysis_must_be_false=False))
    attempt("forge_config_hash", "G18_artifact_bindings_are_complete_and_live", lambda x: x["artifact_registry"]["config"].update(sha256="0"))
    attempt("forge_source_rows", "G19_source_data_and_human_report_are_live", lambda x: x["source_data"].update(rows=x["source_data"]["rows"] - 1))
    attempt("forge_mutation_count", "G20_one_substantive_mutation_per_gate_fails_closed", lambda x: x.update(semantic_mutation_audit={"count": 20, "detected": 19, "cases": []}))
    return {"count": len(cases), "detected": sum(row["rejected"] for row in cases), "cases": cases}


def _analysis_payload(report: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "seal_id", "seal_state", "config_snapshot", "artifact_registry", "splits",
        "execution_cells", "opened_development_audit", "factor_disjointness",
        "formal_balance", "power_analysis", "statistics", "missingness",
        "stopping_rules", "compute_arithmetic", "amendment_policy",
        "seal_absence_proof", "source_data", "markdown", "semantic_mutation_audit",
        "gates", "verdict",
    )
    return {field: report[field] for field in fields}


def build_report(
    source_data: Path = DEFAULT_SOURCE_DATA,
    markdown: Path = DEFAULT_MARKDOWN,
) -> dict[str, Any]:
    config = _load(CONFIG)
    absence = _seal_absence(config)
    existing = [row["path"] for row in absence if row["exists_at_seal"]]
    if existing:
        raise ValueError(f"cannot create preregistration after outcome access: {existing}")
    if [row["split_id"] for row in config["splits"]] != list(SPLIT_IDS):
        raise ValueError("split order must be train/calibration/pilot/formal")
    splits = [_expanded_split(row) for row in config["splits"]]
    scenarios = list(config["scenario_families"])
    cells = _execution_cells(splits, scenarios)
    power = _power_analysis(config)
    formal_rounds = sum(row["rounds"] for row in cells if row["split_id"] == "formal")
    report: dict[str, Any] = {
        "task_id": TASK_ID,
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seal_id": _canonical_sha256({"config": config, "implementation": _sha256(Path(__file__))}),
        "seal_state": "SEALED_PRE_OUTCOME",
        "config_snapshot": config,
        "artifact_registry": {key: _binding(path) for key, path in ARTIFACT_PATHS.items()},
        "splits": splits,
        "execution_cells": cells,
        "opened_development_audit": _opened_audit(config, splits),
        "factor_disjointness": _factor_disjointness(splits),
        "formal_balance": _formal_balance(cells),
        "power_analysis": power,
        "statistics": config["statistics"],
        "missingness": config["missingness"],
        "stopping_rules": config["stopping_rules"],
        "compute_arithmetic": {
            "cells_total": len(cells),
            "formal_cells": sum(row["split_id"] == "formal" for row in cells),
            "formal_physical_rounds_per_method": formal_rounds,
            "formal_max_method_decodes": formal_rounds * next(row for row in splits if row["split_id"] == "formal")["max_method_count"],
            "all_split_physical_rounds_per_method": {
                split_id: sum(row["rounds"] for row in cells if row["split_id"] == split_id)
                for split_id in SPLIT_IDS
            },
        },
        "amendment_policy": config["amendment_policy"],
        "seal_absence_proof": absence,
    }
    _atomic_text(_render_markdown(report), markdown)
    rows = _write_source_data(report, source_data)
    report["source_data"] = {**_binding(source_data), "rows": rows}
    report["markdown"] = _binding(markdown)
    report["semantic_mutation_audit"] = {"count": 20, "detected": 20, "cases": []}
    report["semantic_mutation_audit"] = _mutations(report)
    report["gates"] = evaluate_gates(report)
    report["gate_summary"] = {
        "passed": sum(report["gates"].values()),
        "failed": [key for key, passed in report["gates"].items() if not passed],
    }
    report["verdict"] = VERDICT if not report["gate_summary"]["failed"] else "FAIL_PHASE6D_MULTIMODE_PREREGISTRATION"
    report["analysis_sha256"] = _canonical_sha256(_analysis_payload(report))
    # Re-render once so the human report contains the final verdict.  The
    # scientific source rows are unchanged because markdown is not a source row.
    _atomic_text(_render_markdown(report), markdown)
    report["markdown"] = _binding(markdown)
    report["gates"] = evaluate_gates(report)
    report["gate_summary"] = {
        "passed": sum(report["gates"].values()),
        "failed": [key for key, passed in report["gates"].items() if not passed],
    }
    report["verdict"] = VERDICT if not report["gate_summary"]["failed"] else "FAIL_PHASE6D_MULTIMODE_PREREGISTRATION"
    report["analysis_sha256"] = _canonical_sha256(_analysis_payload(report))
    return report


def verify_report(
    report: Mapping[str, Any] | None = None,
    path: Path = DEFAULT_REPORT,
) -> dict[str, bool]:
    value = dict(report) if report is not None else _load(path)
    gates = evaluate_gates(value)
    checks = {
        "identity": value.get("task_id") == TASK_ID and value.get("schema_version") == SCHEMA_VERSION,
        "gates": value.get("gates") == gates and all(gates.values()),
        "verdict": value.get("verdict") == VERDICT,
        "analysis_hash": value.get("analysis_sha256") == _canonical_sha256(_analysis_payload(value)),
    }
    if not all(checks.values()):
        raise ValueError(f"T6.20.3 verification failed: {[key for key, passed in checks.items() if not passed]}")
    return checks


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--source-data", type=Path, default=DEFAULT_SOURCE_DATA)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)
    if args.verify:
        verify_report(path=args.report)
        print(json.dumps({"verified": _relative(args.report), "verdict": VERDICT}, ensure_ascii=False))
        return 0
    report = build_report(args.source_data, args.markdown)
    _atomic_json(report, args.report)
    verify_report(report, args.report)
    print(
        json.dumps(
            {
                "output": _relative(args.report),
                "seal_id": report["seal_id"],
                "execution_cells": len(report["execution_cells"]),
                "formal_rounds_per_method": report["compute_arithmetic"]["formal_physical_rounds_per_method"],
                "power": report["power_analysis"]["achieved_power_at_registered_sd"],
                "gates": report["gate_summary"],
                "verdict": report["verdict"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
