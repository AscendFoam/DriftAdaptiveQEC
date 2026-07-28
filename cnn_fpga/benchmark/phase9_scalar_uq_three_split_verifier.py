"""Independent verifier for the three-split scalar UQ calibration.

This module deliberately does not import the production calibration runner.
It reconstructs every cell, seed, factor gate, selection decision, and
confirmation decision from the published CSV files.
"""

from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
import math
from pathlib import Path
import tempfile
import os
from statistics import NormalDist
from typing import Any, Mapping, Sequence


TASK_ID = "T-RISK-20260728-02"
CONFIG_PATH = "configs/phase9/t_risk_20260728_02_scalar_uq_calibration.json"
REPORT_PATH = "docs/t_risk_20260728_02_scalar_uq_calibration.json"
OUTPUT_PATH = "docs/t_risk_20260728_02_scalar_uq_calibration_verification.json"
CONFIG_SCHEMA = "PHASE9-SCALAR-UQ-THREE-SPLIT-CALIBRATION-CONFIG-V1"
REPORT_SCHEMA = "PHASE9-SCALAR-UQ-THREE-SPLIT-CALIBRATION-REPORT-V1"
VERIFIER_SCHEMA = "PHASE9-SCALAR-UQ-THREE-SPLIT-INDEPENDENT-VERIFIER-V1"
PASS_VERDICT = "PASS_SCALAR_UQ_THREE_SPLIT_CALIBRATION"
VERIFIED_PASS = "VERIFIED_PASS_SCALAR_UQ_THREE_SPLIT_CALIBRATION"
VERIFIED_NO_GO = "VERIFIED_NO_GO_SCALAR_UQ_THREE_SPLIT_CALIBRATION"
FACTOR_GRID = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
MARGINS = [0.005, 0.01, 0.02, 0.05, 0.08, 0.1, 0.15, 0.25]
COUNTS = [12, 384]
RATIOS = [0.0, 0.5, 1.0, 1.2]
FAMILY_ORDER = ["gaussian_balanced", "rare_heavy_tail", "heteroskedastic"]
PRIMARY = {
    "gaussian_balanced": True,
    "rare_heavy_tail": True,
    "heteroskedastic": False,
}
GATES = {
    "minimum_cell_coverage_rate": 0.94,
    "minimum_cell_coverage_wilson_lcb": 0.9,
    "null_equivalence_wilson_lcb": 0.8,
    "local_half_margin_equivalence_wilson_lcb": 0.65,
    "boundary_equivalence_wilson_ucb": 0.1,
    "outside_equivalence_wilson_ucb": 0.05,
}
CLAIM_BOUNDARY = {
    "scalar_uq_calibration_only": True,
    "twin_qualification": None,
    "ler": None,
    "lifetime": None,
    "physical_break_even": None,
    "official_puviani_exact": None,
    "puviani_nmf_surpass": None,
    "external_sota": None,
    "hardware_measured": None,
}
RAW_FIELDS = [
    "split",
    "cell_id",
    "family",
    "margin",
    "cluster_count",
    "effect_ratio",
    "true_difference",
    "trial_index",
    "trial_seed",
    "multiplier_seed",
    "estimate",
    "raw_radius",
]


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha(value: object) -> str:
    return sha256(_canonical(value)).hexdigest()


def _self_hash(value: Mapping[str, Any], label: str) -> None:
    unsigned = dict(value)
    observed = unsigned.pop("analysis_sha256", None)
    if not isinstance(observed, str) or observed != _sha(unsigned):
        raise RuntimeError(f"{label} self-hash drift")


def _binding(path: Path, root: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {
        "path": path.resolve().relative_to(root.resolve()).as_posix(),
        "bytes": len(payload),
        "sha256": sha256(payload).hexdigest(),
    }


def _verify_binding(root: Path, value: Mapping[str, Any], label: str) -> None:
    if set(value) != {"path", "bytes", "sha256"}:
        raise RuntimeError(f"{label} binding schema drift")
    path = (root / str(value["path"])).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise RuntimeError(f"{label} path invalid")
    if _binding(path, root) != dict(value):
        raise RuntimeError(f"{label} live binding drift")


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} is not a JSON object")
    return value


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _cell_id(family: str, margin: float, count: int, ratio: float) -> str:
    return (
        f"{family}__m{margin:.6f}__n{count}__r{ratio:.3f}"
    ).replace(".", "p")


def _cells() -> list[tuple[str, float, int, float, str]]:
    cells = [
        (family, margin, count, ratio, _cell_id(family, margin, count, ratio))
        for family in FAMILY_ORDER
        for margin in MARGINS
        for count in COUNTS
        for ratio in RATIOS
    ]
    if len(cells) != 192 or len({item[4] for item in cells}) != 192:
        raise RuntimeError("independent verifier cell denominator drift")
    return cells


def _address(base: int, *parts: object) -> int:
    digest = sha256("|".join(map(str, parts)).encode("utf-8")).digest()
    return int(base) + int.from_bytes(digest[:8], "big") % 1_000_000_000


def _wilson(successes: int, total: int) -> tuple[float, float]:
    alpha = (1.0 - 0.95) / 256
    z = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def _factor_gate(
    factor: float, summaries: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    if len(summaries) != 192:
        raise RuntimeError("independent verifier summary denominator drift")
    coverage_failed = [
        str(row["cell_id"])
        for row in summaries
        if (
            float(row["coverage_rate"]) < GATES["minimum_cell_coverage_rate"]
            or float(row["coverage_wilson_lcb"])
            < GATES["minimum_cell_coverage_wilson_lcb"]
        )
    ]
    rules = {
        0.0: ("lcb", GATES["null_equivalence_wilson_lcb"]),
        0.5: ("lcb", GATES["local_half_margin_equivalence_wilson_lcb"]),
        1.0: ("ucb", GATES["boundary_equivalence_wilson_ucb"]),
        1.2: ("ucb", GATES["outside_equivalence_wilson_ucb"]),
    }
    power = []
    for ratio, (bound, threshold) in rules.items():
        strata = [
            row
            for row in summaries
            if int(row["cluster_count"]) == 384
            and bool(row["power_primary"])
            and float(row["effect_ratio"]) == ratio
        ]
        if len(strata) != 16:
            raise RuntimeError("independent verifier power denominator drift")
        failed = [
            str(row["cell_id"])
            for row in strata
            if (
                float(row["equivalence_wilson_lcb"]) < threshold
                if bound == "lcb"
                else float(row["equivalence_wilson_ucb"]) > threshold
            )
        ]
        power.append(
            {
                "effect_ratio": ratio,
                "bound": bound,
                "threshold": threshold,
                "stratum_count": 16,
                "failed_strata": failed,
                "global_iut_pass": not failed,
            }
        )
    coverage_pass = not coverage_failed
    power_pass = all(item["global_iut_pass"] for item in power)
    return {
        "factor": factor,
        "coverage_pass": coverage_pass,
        "coverage_failed_cells": coverage_failed,
        "power_pass": power_pass,
        "power_ledger": power,
        "global_pass": coverage_pass and power_pass,
        "minimum_coverage_rate": min(float(row["coverage_rate"]) for row in summaries),
        "minimum_coverage_wilson_lcb": min(
            float(row["coverage_wilson_lcb"]) for row in summaries
        ),
        "summaries_sha256": _sha(list(summaries)),
    }


def _recompute_split(
    root: Path,
    config: Mapping[str, Any],
    split: str,
    factors: Sequence[float],
    expected_binding: Mapping[str, Any],
) -> dict[str, Any]:
    path = root / str(expected_binding["path"])
    _verify_binding(root, expected_binding, f"{split} source")
    trial_count = int(config["trial_count_per_cell"])
    split_spec = config["splits"][split]
    summaries: dict[float, list[dict[str, Any]]] = {
        factor: [] for factor in factors
    }
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != RAW_FIELDS:
            raise RuntimeError(f"{split} raw field drift")
        for family, margin, count, ratio, cell_id in _cells():
            counters = {
                factor: {"coverage": 0, "equivalence": 0} for factor in factors
            }
            for trial in range(trial_count):
                try:
                    row = next(reader)
                except StopIteration as exc:
                    raise RuntimeError(f"{split} raw denominator underflow") from exc
                if set(row) != set(RAW_FIELDS):
                    raise RuntimeError(f"{split} raw row schema drift")
                expected_trial_seed = _address(
                    int(split_spec["trial_seed_base"]), split, cell_id, trial
                )
                expected_multiplier_seed = _address(
                    int(split_spec["multiplier_seed_base"]), split, cell_id, trial
                )
                estimate = float(row["estimate"])
                radius = float(row["raw_radius"])
                truth = ratio * margin
                if (
                    row["split"] != split
                    or row["cell_id"] != cell_id
                    or row["family"] != family
                    or float(row["margin"]) != margin
                    or int(row["cluster_count"]) != count
                    or float(row["effect_ratio"]) != ratio
                    or float(row["true_difference"]) != truth
                    or int(row["trial_index"]) != trial
                    or int(row["trial_seed"]) != expected_trial_seed
                    or int(row["multiplier_seed"]) != expected_multiplier_seed
                    or not math.isfinite(estimate)
                    or not math.isfinite(radius)
                    or estimate < 0.0
                    or radius < 0.0
                ):
                    raise RuntimeError(f"{split} raw row semantic drift")
                for factor in factors:
                    upper = estimate + factor * radius
                    counters[factor]["coverage"] += upper + 1e-15 >= truth
                    counters[factor]["equivalence"] += upper <= margin
            for factor in factors:
                coverage = counters[factor]["coverage"]
                equivalence = counters[factor]["equivalence"]
                coverage_lcb, coverage_ucb = _wilson(coverage, trial_count)
                equivalence_lcb, equivalence_ucb = _wilson(
                    equivalence, trial_count
                )
                summaries[factor].append(
                    {
                        "cell_id": cell_id,
                        "family": family,
                        "margin": margin,
                        "cluster_count": count,
                        "effect_ratio": ratio,
                        "power_primary": PRIMARY[family],
                        "coverage_successes": coverage,
                        "coverage_rate": coverage / trial_count,
                        "coverage_wilson_lcb": coverage_lcb,
                        "coverage_wilson_ucb": coverage_ucb,
                        "equivalence_successes": equivalence,
                        "equivalence_rate": equivalence / trial_count,
                        "equivalence_wilson_lcb": equivalence_lcb,
                        "equivalence_wilson_ucb": equivalence_ucb,
                    }
                )
        try:
            next(reader)
        except StopIteration:
            pass
        else:
            raise RuntimeError(f"{split} raw denominator overflow")
    return {
        "split": split,
        "role": config["splits"][split]["role"],
        "raw_trial_count": 192 * trial_count,
        "cell_count": 192,
        "trial_count_per_cell": trial_count,
        "evaluated_factors": list(factors),
        "factor_gates": {
            f"{factor:.1f}": _factor_gate(factor, summaries[factor])
            for factor in factors
        },
        "source_data_binding": dict(expected_binding),
    }


def verify(root: Path | None = None) -> dict[str, Any]:
    base = (root or _root()).resolve()
    config = _read(base / CONFIG_PATH)
    report = _read(base / REPORT_PATH)
    _self_hash(report, "target calibration report")
    if (
        config.get("schema_version") != CONFIG_SCHEMA
        or config.get("factor_grid") != FACTOR_GRID
        or config.get("margins") != MARGINS
        or config.get("cluster_counts") != COUNTS
        or config.get("effect_ratios") != RATIOS
        or list(config.get("families", {})) != FAMILY_ORDER
        or config.get("trial_count_per_cell") != 2048
        or config.get("gates") != GATES
        or config.get("claim_boundary") != CLAIM_BOUNDARY
        or report.get("schema_version") != REPORT_SCHEMA
        or report.get("claim_state") != CLAIM_BOUNDARY
        or report.get("design_outcomes_accessed") is not False
        or report.get("diagnostic_parent_used_as_selection_or_confirmation_evidence")
        is not False
    ):
        raise RuntimeError("independent verifier contract drift")
    bindings = report.get("bindings")
    if not isinstance(bindings, Mapping):
        raise RuntimeError("target report bindings missing")
    for name, binding in bindings.items():
        if not isinstance(binding, Mapping):
            raise RuntimeError(f"target binding type drift: {name}")
        _verify_binding(base, binding, f"target/{name}")
    selection_a = _recompute_split(
        base,
        config,
        "selection_a",
        FACTOR_GRID,
        bindings["selection_a_source_data"],
    )
    selection_b = _recompute_split(
        base,
        config,
        "selection_b",
        FACTOR_GRID,
        bindings["selection_b_source_data"],
    )
    selected = None
    for factor in FACTOR_GRID:
        key = f"{factor:.1f}"
        if (
            selection_a["factor_gates"][key]["global_pass"]
            and selection_b["factor_gates"][key]["global_pass"]
        ):
            selected = factor
            break
    receipt = _read(base / str(bindings["selection_receipt"]["path"]))
    _self_hash(receipt, "selection receipt")
    if (
        receipt.get("selected_factor") != selected
        or receipt.get("selection_passed") is not (selected is not None)
        or receipt.get("confirmation_outcomes_accessed") is not False
    ):
        raise RuntimeError("independent verifier selection decision mismatch")
    for reconstructed, published, name in (
        (selection_a, report.get("selection_a"), "selection_a"),
        (selection_b, report.get("selection_b"), "selection_b"),
    ):
        if not isinstance(published, Mapping):
            raise RuntimeError(f"published {name} missing")
        for key in (
            "split",
            "role",
            "raw_trial_count",
            "cell_count",
            "trial_count_per_cell",
            "evaluated_factors",
            "factor_gates",
            "source_data_binding",
        ):
            if reconstructed[key] != published.get(key):
                raise RuntimeError(f"independent verifier {name}/{key} mismatch")
        if receipt.get(name) != published:
            raise RuntimeError(f"selection receipt {name} mismatch")
    confirmation = None
    if selected is not None:
        confirmation = _recompute_split(
            base,
            config,
            "confirmation",
            [selected],
            bindings["confirmation_source_data"],
        )
        published_confirmation = report.get("confirmation")
        if not isinstance(published_confirmation, Mapping):
            raise RuntimeError("published confirmation missing")
        for key in (
            "split",
            "role",
            "raw_trial_count",
            "cell_count",
            "trial_count_per_cell",
            "evaluated_factors",
            "factor_gates",
            "source_data_binding",
        ):
            if confirmation[key] != published_confirmation.get(key):
                raise RuntimeError(f"independent verifier confirmation/{key} mismatch")
    confirmation_passed = (
        None
        if confirmation is None
        else confirmation["factor_gates"][f"{selected:.1f}"]["global_pass"]
    )
    expected_verdict = (
        "NO_GO_SCALAR_UQ_FACTOR_SELECTION"
        if selected is None
        else (
            PASS_VERDICT
            if confirmation_passed
            else "NO_GO_SCALAR_UQ_UNTOUCHED_CONFIRMATION"
        )
    )
    if (
        report.get("selected_factor") != selected
        or report.get("selection_passed") is not (selected is not None)
        or report.get("confirmation_passed") is not confirmation_passed
        or report.get("verdict") != expected_verdict
        or (
            expected_verdict == PASS_VERDICT
            and report.get("qualified_claim")
            != "COVERAGE_CALIBRATED_SCALAR_UQ_FACTOR_FOR_T_RISK_20260728_01"
        )
        or (
            expected_verdict != PASS_VERDICT
            and report.get("qualified_claim") is not None
        )
    ):
        raise RuntimeError("independent verifier final decision mismatch")
    output = {
        "task_id": TASK_ID,
        "schema_version": VERIFIER_SCHEMA,
        "verdict": (
            VERIFIED_PASS if expected_verdict == PASS_VERDICT else VERIFIED_NO_GO
        ),
        "target_verdict": expected_verdict,
        "target_analysis_sha256": report["analysis_sha256"],
        "selected_factor": selected,
        "selection_recomputed": True,
        "confirmation_recomputed": confirmation is not None,
        "raw_rows_recomputed": (
            2 * 192 * 2048
            + (0 if confirmation is None else 192 * 2048)
        ),
        "seed_rows_exact": True,
        "factor_gate_rows_exact": True,
        "design_outcomes_accessed": False,
        "claim_state": dict(CLAIM_BOUNDARY),
        "bindings": {
            "config": _binding(base / CONFIG_PATH, base),
            "target_report": _binding(base / REPORT_PATH, base),
            "verifier_implementation": _binding(Path(__file__).resolve(), base),
            "selection_a_source_data": bindings["selection_a_source_data"],
            "selection_b_source_data": bindings["selection_b_source_data"],
            **(
                {"confirmation_source_data": bindings["confirmation_source_data"]}
                if confirmation is not None
                else {}
            ),
        },
    }
    output["analysis_sha256"] = _sha(output)
    return output


def write_verification(root: Path | None = None) -> dict[str, Any]:
    base = (root or _root()).resolve()
    path = base / OUTPUT_PATH
    if path.exists():
        raise RuntimeError("independent verifier canonical output already exists")
    report = verify(base)
    _atomic_write(
        path,
        (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    live = _read(path)
    _self_hash(live, "independent verifier report")
    if live != report:
        raise RuntimeError("independent verifier publication drift")
    return live


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    result = write_verification()
    print(
        json.dumps(
            {
                "analysis_sha256": result["analysis_sha256"],
                "verdict": result["verdict"],
                "target_verdict": result["target_verdict"],
                "selected_factor": result["selected_factor"],
                "raw_rows_recomputed": result["raw_rows_recomputed"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["verify", "write_verification"]
