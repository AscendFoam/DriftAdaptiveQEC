"""Build and verify the T9.1.3 post-outcome governance addendum.

The production T9.1.3 report and its 42-entry deviation ledger are immutable
sealed evidence.  This module therefore records four omissions discovered
during the post-outcome anti-simplification review in a separate addendum.  It
also freezes the exact input contract handed to T9.1.4.

The addendum is intentionally non-promotional:

* it never rewrites or re-seals the parent report or deviation ledger;
* official-exact, Puviani-surpass, and paper-scale lifetime values stay null;
* both qualified and typed no-go terminal states release T9.1.4; and
* the short-horizon, binary-observation parent cannot enter matched Phase-9
  ranking.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "T9.1.3"
SCHEMA_VERSION = "t9.1.3-post-outcome-governance-addendum-v2"
STATUS = "POST_OUTCOME_GOVERNANCE_ADDENDUM_NO_RETROACTIVE_RESEAL"
VERDICT_PASS = "PASS_POST_OUTCOME_GOVERNANCE_ADDENDUM"
VERDICT_FAIL = "FAIL_POST_OUTCOME_GOVERNANCE_ADDENDUM"

DEFAULT_REPORT = ROOT / "docs/t9_1_3_post_outcome_governance_addendum.json"
DEFAULT_SOURCE_DATA = (
    ROOT / "docs/t9_1_3_post_outcome_governance_source_data.csv"
)
DEFAULT_INPUT_CONTRACT = ROOT / "configs/phase9/t9_1_4_input_contract.json"
IMPLEMENTATION = Path(__file__).resolve()

PARENT_REPORT_STATUS = "PASS_ARTIFACT_LANE_AND_EXECUTABLE_REIMPLEMENTATION"
QUALIFIED_TERMINAL_STATE = "QUALIFIED_PAPER_CONSTRAINED_BASELINE"
NO_GO_TERMINAL_STATE = "NO_GO_PAPER_CONSTRAINED_REIMPLEMENTATION"
PARENT_REPORT_ANALYSIS_SHA256 = (
    "35d52c4add5aae124da4127223fb14d08c59a375baaf8bf26146a60f7fc3a152"
)
PARENT_REPORT_RAW_SHA256 = (
    "f3e458765ed684d69be3011e532a4334de872d2b39d1f07da7c5aaec61816ffd"
)
PRODUCTION_CONFIG_CANONICAL_SHA256 = (
    "2b708da81a16741b208e47ff3c55da646f7876ee4c743962b18945e66c5e76f0"
)
GRADIENT_CLIP_NORM = 10.0
PYTORCH_CLIP_EPSILON = 1.0e-6
SEMANTIC_SEAL_VERSION = "t9.1.3-to-t9.1.4-semantic-seal-v1"
QUALIFIED_BRANCH = "QUALIFIED"
NO_GO_BRANCH = "NO_GO"
FAILURE_MANIFEST_SCHEMA_VERSION = (
    "t9.1.3-paper-constrained-terminal-failure-manifest-v1"
)
EXPECTED_MUTATION_COUNT = 14
NO_GO_FIXTURE_POLICY = {
    "role": "SCHEMA_EXAMPLE_ONLY",
    "live_evidence_required_for_embedded_example": False,
    "clean_checkout_dependency": "NONE",
    "executable_no_go_source": "EXTERNAL_LIVE_FAILURE_MANIFEST_REQUIRED",
    "test_fixture_requirement": (
        "TEMPORARY_LIVE_ARCHIVE_SEAL_INVENTORY_AND_MANIFEST"
    ),
}
DOWNSTREAM_CONSUMER_RULE = {
    "ordered_steps": [
        {
            "step": 1,
            "action": "validate_report",
            "verify_live_files": True,
            "purpose": "VERIFY_BRANCH_LOCAL_EVIDENCE_AND_REPORT",
        },
        {
            "step": 2,
            "action": "reconstruct_downstream_semantic_seal",
            "purpose": "RECOMPUTE_CANONICAL_PAYLOAD_AND_SHA256",
        },
        {
            "step": 3,
            "action": "compare_semantic_seal",
            "purpose": (
                "REQUIRE_EXACT_MATCH_BEFORE_T9_1_4_CONSUMPTION"
            ),
        },
    ],
    "acceptance": "ALL_STEPS_MUST_PASS",
    "on_failure": "FAIL_CLOSED_DO_NOT_CONSUME",
    "seal_only_acceptance": "PROHIBITED",
}

EXPECTED_PARENT_ARTIFACTS: dict[str, dict[str, Any]] = {
    "parent_report": {
        "path": "docs/t9_1_3_puviani_paper_constrained.json",
        "bytes": 187866,
        "sha256": PARENT_REPORT_RAW_SHA256,
    },
    "production_config": {
        "path": "configs/phase9/t9_1_3_puviani_paper_constrained.json",
        "bytes": 15145,
        "sha256": "a93edf5cfbdfe22490a1d7cff967f61b0b0add38662eeaf546390dae32c3cbf9",
        "canonical_sha256": PRODUCTION_CONFIG_CANONICAL_SHA256,
    },
    "deviation_ledger": {
        "path": "configs/phase9/t9_1_3_deviation_ledger.json",
        "bytes": 48578,
        "sha256": "69a75c07452b9dea3e445cf683394663e165a72db47986276cc65dd44078e8e0",
        "rows": 42,
    },
    "training_ledger": {
        "path": "docs/t9_1_3_puviani_training_ledger.parquet",
        "bytes": 12511484,
        "sha256": "cdf4d72816a54f4a32e59cbbbe704532a39462cdbbd6917025a6ec8101b3e10a",
        "rows": 40000,
    },
    "selection_ledger": {
        "path": "docs/t9_1_3_puviani_selection_ledger.csv",
        "bytes": 224569,
        "sha256": "f346a188613ffc47a1708382a75c163f08e51ceeb503400a0b7d448058ab5171",
        "rows": 1040,
    },
    "agent_registry": {
        "path": "docs/t9_1_3_puviani_agent_registry.csv",
        "bytes": 57263,
        "sha256": "5290d62c462ed71ae003f3f75a80da9dddd9951c7ed1b0dd587ace96a4a6c3cc",
        "rows": 40,
    },
    "six_state_trajectories": {
        "path": "docs/t9_1_3_puviani_six_state_trajectories.parquet",
        "bytes": 136081,
        "sha256": "227ba0efbafea6d7abb9a2de3048472a8f907ee28440c5ec4ec84c85a7072a6b",
        "rows": 1008,
    },
    "six_state_events": {
        "path": "docs/t9_1_3_puviani_six_state_events.parquet",
        "bytes": 874687,
        "sha256": "d5478dab1b938f3953ed8929de4d36adc199b99c152968e14df627bdf41622ca",
        "rows": 20160,
    },
}

EXPECTED_PARENT_SEAL: dict[str, Any] = {
    "schema_version": "t9.1.3-puviani-paper-constrained-artifacts-v1",
    "status": PARENT_REPORT_STATUS,
    "evidence_grade": "PAPER_CONSTRAINED_REIMPLEMENTATION",
    "analysis_sha256": PARENT_REPORT_ANALYSIS_SHA256,
    "required_gate_count": 52,
    "required_mutation_count": 81,
    "official_exact": None,
    "puviani_surpass": None,
    "paper_scale_lifetime": None,
}

EXPECTED_SEVERITY_COUNTS = {
    "Critical": 24,
    "High": 19,
    "Medium": 3,
    "Low": 0,
}
EXPECTED_ADDENDUM_SEVERITIES = {
    "D43": "High",
    "D44": "High",
    "D45": "Critical",
    "D46": "High",
}
CLAIM_NULLS = {
    "official_exact": None,
    "puviani_surpass": None,
    "paper_scale_lifetime": None,
}
SOURCE_COLUMNS = (
    "section",
    "record_id",
    "payload_json",
    "payload_sha256",
)
REPORT_KEYS = frozenset(
    {
        "schema_version",
        "task_id",
        "status",
        "terminal_branch",
        "generated_from_parent_at_utc",
        "input_contract",
        "implementation",
        "parent_bindings",
        "parent_snapshot",
        "failure_manifest",
        "governance_policy",
        "gradient_clipping_reanalysis",
        "addendum_deviations",
        "deviation_summary",
        "terminal_state_mapping",
        "current_terminal_resolution",
        "claim_slots",
        "ranking_boundary",
        "downstream_semantic_seal",
        "source_data",
        "semantic_mutation_audit",
        "gates",
        "gate_summary",
        "verdict",
        "analysis_sha256",
    }
)
ANALYSIS_FIELDS = tuple(sorted(REPORT_KEYS - {"analysis_sha256"}))


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path_label(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _resolve_path(label: str) -> Path:
    path = Path(label)
    return path if path.is_absolute() else ROOT / path


def _binding(path: Path) -> dict[str, Any]:
    return {
        "path": _path_label(path),
        "bytes": path.stat().st_size,
        "sha256": _file_sha256(path),
    }


def _binding_is_live(value: Mapping[str, Any]) -> bool:
    path = _resolve_path(str(value["path"]))
    return (
        path.is_file()
        and path.stat().st_size == int(value["bytes"])
        and _file_sha256(path) == str(value["sha256"])
    )


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        stream.write(text)
    temporary.replace(path)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_text(
        path,
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
    )


def _failure_manifest_semantic_sha256(value: Mapping[str, Any]) -> str:
    payload = copy.deepcopy(dict(value))
    payload.pop("semantic_sha256", None)
    return _canonical_sha256(payload)


def _validate_failure_manifest_structure(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    manifest = copy.deepcopy(dict(value))
    if set(manifest) != {
        "schema_version",
        "task_id",
        "terminal_state",
        "terminal_result",
        "observed_at_utc",
        "transaction_id",
        "reason",
        "failure_evidence",
        "typed_null_payload",
        "semantic_sha256",
    }:
        raise ValueError("NO_GO failure-manifest schema drifted")
    if (
        manifest["schema_version"] != FAILURE_MANIFEST_SCHEMA_VERSION
        or manifest["task_id"] != TASK_ID
        or manifest["terminal_state"] != "FAILED_FAIL_CLOSED"
        or manifest["terminal_result"] != NO_GO_TERMINAL_STATE
        or not isinstance(manifest["observed_at_utc"], str)
        or not manifest["observed_at_utc"]
        or not isinstance(manifest["transaction_id"], str)
        or not manifest["transaction_id"]
    ):
        raise ValueError("NO_GO failure-manifest identity drifted")
    reason = manifest["reason"]
    if (
        not isinstance(reason, Mapping)
        or set(reason) != {"code", "stage", "message"}
        or any(not isinstance(reason[key], str) or not reason[key] for key in reason)
    ):
        raise ValueError("NO_GO failure reason is incomplete")
    evidence = manifest["failure_evidence"]
    if (
        not isinstance(evidence, Mapping)
        or set(evidence)
        != {
            "archive",
            "archive_seal",
            "inventory",
            "archive_hash_verification",
        }
        or not isinstance(evidence["archive_hash_verification"], str)
        or not evidence["archive_hash_verification"]
    ):
        raise ValueError("NO_GO failure evidence is incomplete")
    for key in ("archive", "archive_seal", "inventory"):
        binding = evidence[key]
        if (
            not isinstance(binding, Mapping)
            or set(binding) != {"path", "bytes", "sha256"}
            or not isinstance(binding["path"], str)
            or not binding["path"]
            or not isinstance(binding["bytes"], int)
            or binding["bytes"] <= 0
            or not isinstance(binding["sha256"], str)
            or len(binding["sha256"]) != 64
            or any(character not in "0123456789abcdef" for character in binding["sha256"])
        ):
            raise ValueError(f"NO_GO {key} binding is invalid")
    typed_null = manifest["typed_null_payload"]
    if (
        not isinstance(typed_null, Mapping)
        or typed_null
        != {
            "selected_controller": None,
            "numeric_metrics": None,
            "rank": None,
            "official_exact": None,
            "puviani_surpass": None,
            "paper_scale_lifetime": None,
        }
    ):
        raise ValueError("NO_GO failure manifest is not typed-null")
    if manifest["semantic_sha256"] != _failure_manifest_semantic_sha256(manifest):
        raise ValueError("NO_GO failure-manifest semantic seal drifted")
    return manifest


def _validate_failure_manifest_live(
    value: Mapping[str, Any],
    *,
    verify_live_files: bool,
) -> dict[str, Any]:
    manifest = _validate_failure_manifest_structure(value)
    if not verify_live_files:
        return manifest
    evidence = manifest["failure_evidence"]
    if not all(
        _binding_is_live(evidence[key])
        for key in ("archive", "archive_seal", "inventory")
    ):
        raise ValueError("NO_GO failure evidence raw binding drifted")
    archive_seal = _load_json(
        _resolve_path(str(evidence["archive_seal"]["path"]))
    )
    inventory = _load_json(
        _resolve_path(str(evidence["inventory"]["path"]))
    )
    archive = evidence["archive"]
    if (
        archive_seal.get("schema_version")
        != "t9.1.3-failed-finalization-tar-seal-v1"
        or archive_seal.get("archive_path") != archive["path"]
        or archive_seal.get("archive_bytes") != archive["bytes"]
        or archive_seal.get("archive_sha256") != archive["sha256"]
        or archive_seal.get("inventory_path") != evidence["inventory"]["path"]
        or archive_seal.get("inventory_sha256")
        != evidence["inventory"]["sha256"]
    ):
        raise ValueError("NO_GO archive seal does not bind archive/inventory")
    if (
        inventory.get("schema_version")
        != "t9.1.3-failed-finalization-archive-v1"
        or inventory.get("source_file_count") is None
        or int(inventory["source_file_count"]) <= 0
        or inventory.get("transaction_id") != manifest["transaction_id"]
        or inventory.get("terminal_state") != manifest["terminal_state"]
        or inventory.get("failure_marker_status") != manifest["reason"]["code"]
        or inventory.get("valid_pass_seal") is not False
    ):
        raise ValueError("NO_GO failure inventory semantics drifted")
    return manifest


def _validate_input_contract(path: Path = DEFAULT_INPUT_CONTRACT) -> dict[str, Any]:
    contract = _load_json(path)
    expected_top_keys = {
        "schema_version",
        "task_id",
        "source_task_id",
        "contract_status",
        "parent_artifacts",
        "parent_semantic_seal",
        "no_go_failure_manifest_fixture",
        "no_go_failure_manifest_fixture_policy",
        "post_outcome_deviation_contract",
        "terminal_state_mapping",
        "claim_null_contract",
        "ranking_boundary",
        "downstream_addendum_semantic_contract",
    }
    if set(contract) != expected_top_keys:
        raise ValueError("T9.1.4 input-contract schema drifted")
    if (
        contract["schema_version"]
        != "t9.1.4-paper-constrained-input-contract-v2"
        or contract["task_id"] != "T9.1.4"
        or contract["source_task_id"] != TASK_ID
        or contract["contract_status"] != "FROZEN_POST_OUTCOME_INPUT_CONTRACT"
        or contract["parent_artifacts"] != EXPECTED_PARENT_ARTIFACTS
        or contract["parent_semantic_seal"] != EXPECTED_PARENT_SEAL
        or contract["claim_null_contract"] != CLAIM_NULLS
    ):
        raise ValueError("T9.1.4 input-contract identity or parent seal drifted")
    deviation = contract["post_outcome_deviation_contract"]
    if deviation != {
        "parent_entry_count": 42,
        "addendum_ids": ["D43", "D44", "D45", "D46"],
        "combined_entry_count": 46,
        "combined_severity_counts": EXPECTED_SEVERITY_COUNTS,
        "gradient_clip_norm": GRADIENT_CLIP_NORM,
        "gradient_rows_expected": 40000,
        "parent_reseal": "PROHIBITED",
    }:
        raise ValueError("post-outcome deviation contract drifted")
    mappings = contract["terminal_state_mapping"]
    if (
        not isinstance(mappings, list)
        or len(mappings) != 2
        or mappings[0].get("match") != PARENT_REPORT_STATUS
        or mappings[0].get("terminal_state") != QUALIFIED_TERMINAL_STATE
        or mappings[1].get("match")
        != "TERMINAL_NONPASS_WITH_VALID_FAILURE_MANIFEST"
        or mappings[1].get("terminal_state") != NO_GO_TERMINAL_STATE
        or mappings[1].get("supported_terminal_states")
        != ["FAILED_FAIL_CLOSED"]
        or any(row.get("releases_t9_1_4") is not True for row in mappings)
        or any(
            row.get("matched_phase9_ranking_eligible") is not False
            for row in mappings
        )
    ):
        raise ValueError("terminal-state mapping drifted")
    no_go_payload = mappings[1].get("typed_payload")
    if (
        not isinstance(no_go_payload, Mapping)
        or no_go_payload
        != {
            "parent_report_sha256": None,
            "evidence_grade": "PAPER_CONSTRAINED_REIMPLEMENTATION",
            "numeric_scope": None,
            "selected_controller": None,
            "numeric_metrics": None,
            "rank": None,
            "official_exact": None,
            "puviani_surpass": None,
            "paper_scale_lifetime": None,
        }
    ):
        raise ValueError("no-go branch is not an exact typed-null payload")
    ranking = contract["ranking_boundary"]
    if (
        ranking.get("matched_phase9_ranking_eligible") is not False
        or ranking.get("sota_claim_eligible") is not False
        or ranking.get("t9_1_3_registry_role")
        != "PROVENANCE_BOUND_PAPER_CONSTRAINED_CONTEXT_BASELINE"
    ):
        raise ValueError("T9.1.3 ranking boundary drifted")
    fixture = contract["no_go_failure_manifest_fixture"]
    if not isinstance(fixture, Mapping):
        raise ValueError("NO_GO failure-manifest fixture is missing")
    _validate_failure_manifest_structure(fixture)
    if (
        contract["no_go_failure_manifest_fixture_policy"]
        != NO_GO_FIXTURE_POLICY
    ):
        raise ValueError("NO_GO schema-example fixture policy drifted")
    downstream = contract["downstream_addendum_semantic_contract"]
    if downstream != {
        "schema_version": "t9.1.4-addendum-semantic-consumption-v1",
        "accepted_addendum_schema_versions": [SCHEMA_VERSION],
        "semantic_seal_version": SEMANTIC_SEAL_VERSION,
        "algorithm": "sha256(canonical-json-utf8(payload))",
        "payload_fields": [
            "addendum_schema_version",
            "task_id",
            "terminal_branch",
            "terminal_evidence_semantic_sha256",
            "governance_status",
            "deviation_state",
            "claim_slots",
            "terminal_resolution",
            "ranking_boundary",
        ],
        "accepted_terminal_branches": [QUALIFIED_BRANCH, NO_GO_BRANCH],
        "raw_hash_dependency": (
            "PROHIBITED_TO_AVOID_CONTRACT_REPORT_CYCLE"
        ),
        "consumer_rule": DOWNSTREAM_CONSUMER_RULE,
    }:
        raise ValueError("downstream addendum semantic contract drifted")
    return contract


def _parent_snapshot(
    parent: Mapping[str, Any], contract: Mapping[str, Any]
) -> dict[str, Any]:
    mutation = parent.get("mutation_audit", {})
    gates = parent.get("gates", {})
    external = parent.get("external_claim_slots", {})
    paper_result = parent.get("paper_scale_numerical_result", {})
    return {
        "schema_version": parent.get("schema_version"),
        "status": parent.get("status"),
        "evidence_grade": parent.get("evidence_grade"),
        "analysis_sha256": parent.get("analysis_sha256"),
        "production_contract": parent.get("production_contract"),
        "config_sha256": parent.get("config_sha256"),
        "deviation_ledger_sha256": parent.get("deviation_ledger", {}).get(
            "sha256"
        ),
        "required_gate_count": len(parent.get("required_gates", [])),
        "published_gate_count": len(gates) if isinstance(gates, Mapping) else -1,
        "all_published_gates_pass": bool(gates) and all(gates.values()),
        "required_mutation_count": mutation.get("mutation_count"),
        "detected_mutation_count": mutation.get("detected_count"),
        "all_parent_mutations_detected": mutation.get("all_detected"),
        "official_exact": external.get("official_exact"),
        "puviani_surpass": external.get("puviani_surpass"),
        "paper_scale_lifetime": external.get("paper_T_X_T_Y_T_Z_T_ch"),
        "paper_scale_result_state": paper_result.get("state"),
        "paper_scale_result_value": paper_result.get("value"),
        "matches_frozen_parent_semantic_seal": {
            "schema_version": parent.get("schema_version"),
            "status": parent.get("status"),
            "evidence_grade": parent.get("evidence_grade"),
            "analysis_sha256": parent.get("analysis_sha256"),
            "required_gate_count": len(parent.get("required_gates", [])),
            "required_mutation_count": mutation.get("mutation_count"),
            "official_exact": external.get("official_exact"),
            "puviani_surpass": external.get("puviani_surpass"),
            "paper_scale_lifetime": external.get(
                "paper_T_X_T_Y_T_Z_T_ch"
            ),
        }
        == contract["parent_semantic_seal"],
    }


def _linear_quantile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("quantile requires non-empty values")
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return float(
        sorted_values[lower]
        + fraction * (sorted_values[upper] - sorted_values[lower])
    )


def _gradient_scope(values: Sequence[float], threshold: float) -> dict[str, Any]:
    ordered = sorted(float(value) for value in values)
    strictly_above = sum(value > threshold for value in ordered)
    would_scale = sum(
        threshold / (value + PYTORCH_CLIP_EPSILON) < 1.0
        for value in ordered
    )
    return {
        "rows": len(ordered),
        "nonfinite_rows": sum(not math.isfinite(value) for value in ordered),
        "nonpositive_rows": sum(value <= 0.0 for value in ordered),
        "rows_strictly_gt_max_norm": strictly_above,
        "rows_scaled_by_pytorch_clip_rule": would_scale,
        "scaled_fraction": would_scale / len(ordered),
        "minimum_unclamped_scale_coefficient": (
            threshold / (ordered[-1] + PYTORCH_CLIP_EPSILON)
        ),
        "minimum": ordered[0],
        "maximum": ordered[-1],
        "mean": math.fsum(ordered) / len(ordered),
        "p50_linear": _linear_quantile(ordered, 0.50),
        "p95_linear": _linear_quantile(ordered, 0.95),
        "p99_linear": _linear_quantile(ordered, 0.99),
    }


def _gradient_clipping_reanalysis(
    training_path: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as error:  # pragma: no cover - production env.
        raise RuntimeError(
            "pyarrow is required for the lossless T9.1.3 ledger audit"
        ) from error

    columns = ["strategy", "root_seed", "epoch", "gradient_norm_before_clip"]
    table = pq.read_table(training_path, columns=columns)
    rows = table.to_pylist()
    if table.num_rows != 40000:
        raise ValueError("training ledger must contain exactly 40,000 epochs")
    groups: dict[tuple[str, int], set[int]] = {}
    gradients: dict[str, list[float]] = {"mf": [], "nmf": []}
    for row in rows:
        strategy = str(row["strategy"])
        if strategy not in gradients:
            raise ValueError(f"unexpected strategy in training ledger: {strategy}")
        seed = int(row["root_seed"])
        epoch = int(row["epoch"])
        groups.setdefault((strategy, seed), set()).add(epoch)
        gradients[strategy].append(float(row["gradient_norm_before_clip"]))
    expected_epochs = set(range(1, 1001))
    if (
        len(groups) != 40
        or any(epochs != expected_epochs for epochs in groups.values())
        or any(len(values) != 20000 for values in gradients.values())
    ):
        raise ValueError("training ledger agent/epoch census is incomplete")
    threshold = float(config["training"]["gradient_clip_norm"])
    if threshold != GRADIENT_CLIP_NORM:
        raise ValueError("gradient clipping threshold drifted")
    all_values = gradients["mf"] + gradients["nmf"]
    return {
        "source_path": _path_label(training_path),
        "column": "gradient_norm_before_clip",
        "clip_rule": (
            "torch.nn.utils.clip_grad_norm_; scale coefficient is "
            "max_norm/(total_norm+1e-6), clamped above at 1.0"
        ),
        "pytorch_clip_epsilon": PYTORCH_CLIP_EPSILON,
        "max_norm": threshold,
        "row_census": {
            "total_rows": len(rows),
            "agent_count": len(groups),
            "agents_per_family": {"mf": 20, "nmf": 20},
            "epochs_per_agent": 1000,
            "all_agent_epoch_sets_exact": True,
        },
        "scopes": {
            "all": _gradient_scope(all_values, threshold),
            "mf": _gradient_scope(gradients["mf"], threshold),
            "nmf": _gradient_scope(gradients["nmf"], threshold),
        },
        "observed_run_conclusion": (
            "The max_norm=10 rule was present but inactive on all 40,000 "
            "recorded updates; this does not establish counterfactual "
            "equivalence on other seeds or configurations."
        ),
    }


def _addendum_deviations(
    gradient: Mapping[str, Any], config: Mapping[str, Any]
) -> list[dict[str, Any]]:
    training = config["training"]
    epochs = int(training["epochs"])
    interval = int(training["validation_interval"])
    candidates = 1 + epochs // interval
    validation_trajectories = (
        len(training["validation_seeds"])
        * int(training["validation_batch_size"])
    )
    roots = len(training["paired_root_seeds"])
    warmup_total = roots * (
        int(training["mf_batch_size"]) + int(training["nmf_batch_size"])
    )
    return [
        {
            "id": "D43",
            "topic": "gradient clipping max_norm=10",
            "discovery_stage": "POST_OUTCOME_ANTI_SIMPLIFICATION_AUDIT",
            "locator": {
                "paper": "Supplement S3C specifies gradient optimization but no clipping rule",
                "official": "feedback_GRAPE.py contains no frozen max-norm clipping contract",
                "project": (
                    "training.gradient_clip_norm and "
                    "docs/t9_1_3_puviani_training_ledger.parquet:"
                    "gradient_norm_before_clip"
                ),
            },
            "value": {
                "paper": None,
                "official": "no qualified clipping threshold",
                "project": {
                        "max_norm": gradient["max_norm"],
                        "total_updates": gradient["scopes"]["all"]["rows"],
                        "scaled_updates_under_pytorch_rule": gradient[
                            "scopes"
                        ]["all"][
                            "rows_scaled_by_pytorch_clip_rule"
                        ],
                        "updates_strictly_above_max_norm": gradient[
                            "scopes"
                        ]["all"]["rows_strictly_gt_max_norm"],
                    "maximum_gradient_norm_before_clip": gradient["scopes"][
                        "all"
                    ]["maximum"],
                    "family_breakdown": {
                        key: {
                            "rows": gradient["scopes"][key]["rows"],
                            "scaled_updates_under_pytorch_rule": gradient[
                                "scopes"
                            ][key][
                                "rows_scaled_by_pytorch_clip_rule"
                            ],
                            "maximum": gradient["scopes"][key]["maximum"],
                        }
                        for key in ("mf", "nmf")
                    },
                },
            },
            "chosen": (
                "Retain the frozen safety rule and disclose its empirical "
                "inactivity; do not infer equivalence outside this sealed run."
            ),
            "rationale": (
                "No recorded update was altered by clipping, but the algorithm "
                "contains a counterfactual branch absent from the paper contract."
            ),
            "severity": "High",
            "comparability": (
                "TRAINING_STABILIZATION_RULE_DIFFERENT_"
                "EMPIRICALLY_INACTIVE_ON_THIS_RUN"
            ),
            "paper_numeric_anchor_eligible": False,
            "claim_effect": (
                "The run remains paper-constrained; clipping inactivity cannot "
                "upgrade it to official exact or support a Puviani-surpass claim."
            ),
        },
        {
            "id": "D44",
            "topic": "persistent PyTorch Adam versus per-epoch TensorFlow Adam",
            "discovery_stage": "POST_OUTCOME_ANTI_SIMPLIFICATION_AUDIT",
            "locator": {
                "paper": "Supplement S3B-S3C describes TensorFlow GPU optimization but does not freeze the complete Adam state/default contract",
                "official": "feedback_GRAPE.py:196-212 constructs TensorFlow Adam inside every epoch",
                "project": "puviani_paper_constrained_artifacts.py:_train_paper_agent constructs torch.optim.Adam once per agent",
            },
            "value": {
                "paper": "learning rate 1e-4; optimizer-state lifetime and complete framework defaults unspecified",
                "official": "unversioned TensorFlow Adam defaults with state recreated per epoch",
                "project": (
                    "persistent PyTorch Adam state for 1000 epochs using the "
                    "sealed runtime's framework defaults"
                ),
            },
            "chosen": (
                "Keep persistent PyTorch Adam for restartable project training, "
                "while separating both framework-default and state-lifetime "
                "semantics from the paper/official path."
            ),
            "rationale": (
                "Even when learning rate and nominal beta values agree, "
                "framework epsilon/kernel details and moment lifetime can change "
                "the optimization trajectory."
            ),
            "severity": "High",
            "comparability": "FRAMEWORK_DEFAULTS_AND_OPTIMIZER_STATE_LIFETIME_DIFFER",
            "paper_numeric_anchor_eligible": False,
            "claim_effect": (
                "Optimizer convergence and selected-agent quality are project "
                "results, not paper-exact numerical reproduction."
            ),
        },
        {
            "id": "D45",
            "topic": "validation cadence, budget, and selection endpoint",
            "discovery_stage": "POST_OUTCOME_ANTI_SIMPLIFICATION_AUDIT",
            "locator": {
                "paper": "Supplement S3C selects the best-performing agent by longer lifetime",
                "official": "no disjoint six-state validation evaluator or frozen candidate budget is published",
                "project": "training.validation_interval, validation_seeds, validation_batch_size, and validation-only selection ledger",
            },
            "value": {
                "paper": "best-agent selection by long-lifetime performance",
                "official": None,
                "project": {
                    "validation_interval_epochs": interval,
                    "candidate_count_per_agent_including_epoch_zero": candidates,
                    "validation_trajectories_per_candidate": validation_trajectories,
                    "selection_metric": training["selection"]["metric"],
                    "selection_horizon_cycles": int(training["full_cycles"]),
                    "selection_ledger_rows": roots * 2 * candidates,
                },
            },
            "chosen": (
                "Use disjoint validation-only short-horizon checkpoint and agent "
                "selection, retain all candidates, and prohibit lifetime/test "
                "selection."
            ),
            "rationale": (
                "This controls leakage but changes the selected estimand and "
                "selection budget relative to the paper's long-lifetime rule."
            ),
            "severity": "Critical",
            "comparability": "SELECTION_ENDPOINT_AND_BUDGET_MATERIALLY_DIFFER",
            "paper_numeric_anchor_eligible": False,
            "claim_effect": (
                "The selected controllers cannot validate or refute the paper's "
                "best-agent lifetime and cannot establish Puviani surpass."
            ),
        },
        {
            "id": "D46",
            "topic": "train-only score-baseline warmup and EMA decay",
            "discovery_stage": "POST_OUTCOME_ANTI_SIMPLIFICATION_AUDIT",
            "locator": {
                "paper": "Eq. 1 and Supplement S3C do not specify a warmup rollout or moving score baseline",
                "official": "feedback_GRAPE.py uses the raw score term without a frozen train-only EMA baseline",
                "project": "puviani_paper_constrained_artifacts.py:_train_paper_agent warmup_seed and score_baseline update",
            },
            "value": {
                "paper": None,
                "official": "no qualified warmup/EMA contract",
                "project": {
                    "warmup_seed_domain": "training_trajectory_seed(root_seed, epoch=0)",
                    "warmup_trajectories_per_mf_agent": int(
                        training["mf_batch_size"]
                    ),
                    "warmup_trajectories_per_nmf_agent": int(
                        training["nmf_batch_size"]
                    ),
                    "warmup_trajectories_population_total": warmup_total,
                    "score_baseline_decay": float(
                        training["score_baseline_decay"]
                    ),
                    "validation_or_qualification_data_used": False,
                },
            },
            "chosen": (
                "Retain the train-only warmup and EMA variance-reduction rule, "
                "publish its seed/budget, and forbid treating it as paper-exact."
            ),
            "rationale": (
                "The rule is causally train-only and leakage-safe, but it changes "
                "the score-function gradient estimator and its early trajectory."
            ),
            "severity": "High",
            "comparability": "SCORE_GRADIENT_ESTIMATOR_AND_WARMUP_DIFFER",
            "paper_numeric_anchor_eligible": False,
            "claim_effect": (
                "Training stability is attributable to a disclosed project "
                "heuristic; official-exact and paper-lifetime claims stay null."
            ),
        },
    ]


def _severity_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {
        severity: sum(row.get("severity") == severity for row in rows)
        for severity in ("Critical", "High", "Medium", "Low")
    }


def _deviation_summary(
    parent_rows: Sequence[Mapping[str, Any]],
    addendum_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    combined = [*parent_rows, *addendum_rows]
    return {
        "parent_entry_count": len(parent_rows),
        "parent_severity_counts": _severity_counts(parent_rows),
        "addendum_entry_count": len(addendum_rows),
        "addendum_severity_counts": _severity_counts(addendum_rows),
        "combined_entry_count": len(combined),
        "combined_severity_counts": _severity_counts(combined),
        "combined_ids": [str(row["id"]) for row in combined],
        "source_of_truth": (
            "original D01-D42 plus this non-retroactive D43-D46 addendum"
        ),
    }


def _no_go_parent_snapshot(
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "terminal_branch": NO_GO_BRANCH,
        "schema_version": manifest["schema_version"],
        "status": manifest["terminal_state"],
        "terminal_result": manifest["terminal_result"],
        "transaction_id": manifest["transaction_id"],
        "failure_reason": copy.deepcopy(manifest["reason"]),
        "failure_manifest_semantic_sha256": manifest["semantic_sha256"],
        "archive_sha256": manifest["failure_evidence"]["archive"]["sha256"],
        "valid_parent_pass_seal": False,
        "official_exact": None,
        "puviani_surpass": None,
        "paper_scale_lifetime": None,
    }


def _no_go_gradient_state(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "state": "NOT_EVALUATED_TYPED_NULL",
        "value": None,
        "reason_code": manifest["reason"]["code"],
        "source_path": None,
        "row_census": None,
        "scopes": None,
    }


def _no_go_deviation_state(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "state": "NOT_EVALUATED_TYPED_NULL",
        "value": None,
        "reason_code": manifest["reason"]["code"],
        "parent_entry_count": None,
        "addendum_entry_count": None,
        "combined_entry_count": None,
        "combined_severity_counts": None,
        "combined_ids": None,
    }


def _current_terminal_resolution(
    parent: Mapping[str, Any], mappings: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    status = parent.get("status")
    if status == PARENT_REPORT_STATUS:
        selected = mappings[0]
    elif status == "FAILED_FAIL_CLOSED":
        selected = mappings[1]
    else:
        raise ValueError(
            "parent status is not a recognized qualified/no-go terminal state"
        )
    return {
        "parent_status": status,
        "resolved_terminal_state": selected["terminal_state"],
        "releases_t9_1_4": selected["releases_t9_1_4"],
        "registry_entry_eligible": selected["registry_entry_eligible"],
        "matched_phase9_ranking_eligible": selected[
            "matched_phase9_ranking_eligible"
        ],
        "typed_payload": copy.deepcopy(selected["typed_payload"]),
    }


def _semantic_seal_payload(report: Mapping[str, Any]) -> dict[str, Any]:
    branch = str(report["terminal_branch"])
    if branch == QUALIFIED_BRANCH:
        terminal_evidence_sha256 = report["parent_snapshot"][
            "analysis_sha256"
        ]
        deviation_state: Any = {
            "state": "EVALUATED_ADDENDUM",
            "combined_entry_count": report["deviation_summary"][
                "combined_entry_count"
            ],
            "combined_severity_counts": report["deviation_summary"][
                "combined_severity_counts"
            ],
            "addendum_ids": [
                row["id"] for row in report["addendum_deviations"]
            ],
        }
    elif branch == NO_GO_BRANCH:
        terminal_evidence_sha256 = report["failure_manifest"][
            "semantic_sha256"
        ]
        deviation_state = copy.deepcopy(report["deviation_summary"])
    else:
        raise ValueError("unknown terminal branch for downstream semantic seal")
    return {
        "addendum_schema_version": report["schema_version"],
        "task_id": report["task_id"],
        "terminal_branch": branch,
        "terminal_evidence_semantic_sha256": terminal_evidence_sha256,
        "governance_status": report["status"],
        "deviation_state": deviation_state,
        "claim_slots": copy.deepcopy(report["claim_slots"]),
        "terminal_resolution": copy.deepcopy(
            report["current_terminal_resolution"]
        ),
        "ranking_boundary": copy.deepcopy(report["ranking_boundary"]),
    }


def _downstream_semantic_seal(
    report: Mapping[str, Any],
) -> dict[str, Any]:
    payload = _semantic_seal_payload(report)
    return {
        "schema_version": SEMANTIC_SEAL_VERSION,
        "payload": payload,
        "semantic_sha256": _canonical_sha256(payload),
        "raw_addendum_hash_required_by_T9_1_4": False,
        "consumer_rule": copy.deepcopy(DOWNSTREAM_CONSUMER_RULE),
    }


def _source_records(report: Mapping[str, Any]) -> list[dict[str, str]]:
    payloads: list[tuple[str, str, Any]] = []
    payloads.extend(
        ("parent_binding", key, value)
        for key, value in report["parent_bindings"].items()
    )
    payloads.extend(
        [
            (
                "terminal_branch",
                str(report["terminal_branch"]),
                {"terminal_branch": report["terminal_branch"]},
            ),
            ("parent_snapshot", "sealed_parent", report["parent_snapshot"]),
            ("governance", "policy", report["governance_policy"]),
            (
                "gradient_clipping_reanalysis",
                "all_40000_updates",
                report["gradient_clipping_reanalysis"],
            ),
        ]
    )
    if report["terminal_branch"] == NO_GO_BRANCH:
        payloads.append(
            (
                "failure_manifest",
                str(report["failure_manifest"]["transaction_id"]),
                report["failure_manifest"],
            )
        )
    else:
        parent_ledger = _load_json(
            _resolve_path(
                str(report["parent_bindings"]["deviation_ledger"]["path"])
            )
        )
        payloads.extend(
            ("deviation", str(row["id"]), row)
            for row in parent_ledger["deviations"]
        )
        payloads.extend(
            ("deviation", str(row["id"]), row)
            for row in report["addendum_deviations"]
        )
    payloads.append(
        ("deviation_summary", "D01-D46", report["deviation_summary"])
    )
    payloads.extend(
        ("terminal_mapping", str(index), row)
        for index, row in enumerate(report["terminal_state_mapping"])
    )
    payloads.append(
        (
            "terminal_resolution",
            "current_parent",
            report["current_terminal_resolution"],
        )
    )
    payloads.extend(
        ("claim_null", key, {"claim_id": key, "value": value})
        for key, value in report["claim_slots"].items()
    )
    payloads.append(
        ("ranking_boundary", "T9.1.3", report["ranking_boundary"])
    )
    payloads.append(
        (
            "downstream_semantic_seal",
            SEMANTIC_SEAL_VERSION,
            report["downstream_semantic_seal"],
        )
    )
    rows: list[dict[str, str]] = []
    for section, record_id, payload in payloads:
        payload_json = _canonical_json(payload)
        rows.append(
            {
                "section": section,
                "record_id": record_id,
                "payload_json": payload_json,
                "payload_sha256": hashlib.sha256(
                    payload_json.encode("utf-8")
                ).hexdigest(),
            }
        )
    return rows


def _write_source_data(
    report: Mapping[str, Any], path: Path
) -> list[dict[str, str]]:
    rows = _source_records(report)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=list(SOURCE_COLUMNS),
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    _atomic_text(path, buffer.getvalue())
    return rows


def _read_source_data(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != SOURCE_COLUMNS:
            raise ValueError("post-outcome source-data columns drifted")
        return [dict(row) for row in reader]


def _expected_parent_snapshot(
    parent: Mapping[str, Any], contract: Mapping[str, Any]
) -> dict[str, Any]:
    return _parent_snapshot(parent, contract)


def evaluate_gates(
    report: Mapping[str, Any],
    *,
    verify_live_files: bool = True,
    input_contract_path: Path | None = None,
) -> dict[str, bool]:
    contract_label = str(report["input_contract"]["path"])
    contract_path = (
        input_contract_path
        if input_contract_path is not None
        else _resolve_path(contract_label)
    )
    contract = _validate_input_contract(contract_path)
    branch = str(report["terminal_branch"])
    mappings = contract["terminal_state_mapping"]
    qualified = branch == QUALIFIED_BRANCH
    no_go = branch == NO_GO_BRANCH

    parent_binding_contract = False
    parent_bindings_live = False
    parent_semantics_pass = False
    nonretroactive_evidence = False
    gradient_pass = False
    deviations_pass = False
    deviation_census_pass = False
    current_resolution_pass = False

    if qualified:
        parent_path = _resolve_path(
            str(contract["parent_artifacts"]["parent_report"]["path"])
        )
        config_path = _resolve_path(
            str(contract["parent_artifacts"]["production_config"]["path"])
        )
        deviation_path = _resolve_path(
            str(contract["parent_artifacts"]["deviation_ledger"]["path"])
        )
        training_path = _resolve_path(
            str(contract["parent_artifacts"]["training_ledger"]["path"])
        )
        parent = _load_json(parent_path)
        config = _load_json(config_path)
        parent_deviations = _load_json(deviation_path)["deviations"]
        expected_gradient = _gradient_clipping_reanalysis(
            training_path, config
        )
        expected_addendum = _addendum_deviations(
            expected_gradient, config
        )
        expected_summary = _deviation_summary(
            parent_deviations, expected_addendum
        )
        expected_snapshot = _expected_parent_snapshot(parent, contract)
        expected_resolution = _current_terminal_resolution(parent, mappings)
        parent_binding_contract = (
            report["parent_bindings"] == contract["parent_artifacts"]
            and report["failure_manifest"] is None
            and _canonical_sha256(config)
            == PRODUCTION_CONFIG_CANONICAL_SHA256
        )
        parent_bindings_live = all(
            _binding_is_live(value)
            for value in report["parent_bindings"].values()
        )
        parent_semantics_pass = (
            report["parent_snapshot"] == expected_snapshot
            and expected_snapshot["matches_frozen_parent_semantic_seal"]
            is True
            and expected_snapshot["production_contract"] is True
            and expected_snapshot["published_gate_count"] == 52
            and expected_snapshot["all_published_gates_pass"] is True
            and expected_snapshot["detected_mutation_count"] == 81
            and expected_snapshot["all_parent_mutations_detected"] is True
            and expected_snapshot["paper_scale_result_state"]
            == "INCOMPLETE_NULL"
            and expected_snapshot["paper_scale_result_value"] is None
        )
        original_ids = [str(row["id"]) for row in parent_deviations]
        addendum_ids = [
            str(row["id"]) for row in report["addendum_deviations"]
        ]
        nonretroactive_evidence = (
            len(parent_deviations) == 42
            and original_ids == [
                f"D{index:02d}" for index in range(1, 43)
            ]
            and not (set(original_ids) & set(addendum_ids))
        )
        gradient_pass = (
            report["gradient_clipping_reanalysis"] == expected_gradient
            and expected_gradient["row_census"]["total_rows"] == 40000
            and expected_gradient["row_census"][
                "all_agent_epoch_sets_exact"
            ]
            is True
            and expected_gradient["scopes"]["all"][
                "rows_scaled_by_pytorch_clip_rule"
            ]
            == 0
            and expected_gradient["scopes"]["all"]["maximum"]
            == 6.72778063230222
        )
        deviations_pass = (
            report["addendum_deviations"] == expected_addendum
            and addendum_ids == ["D43", "D44", "D45", "D46"]
            and {
                str(row["id"]): str(row["severity"])
                for row in report["addendum_deviations"]
            }
            == EXPECTED_ADDENDUM_SEVERITIES
            and all(
                row["paper_numeric_anchor_eligible"] is False
                and row["discovery_stage"]
                == "POST_OUTCOME_ANTI_SIMPLIFICATION_AUDIT"
                for row in report["addendum_deviations"]
            )
        )
        deviation_census_pass = (
            report["deviation_summary"] == expected_summary
            and expected_summary["combined_entry_count"] == 46
            and expected_summary["combined_severity_counts"]
            == EXPECTED_SEVERITY_COUNTS
            and expected_summary["combined_ids"]
            == [f"D{index:02d}" for index in range(1, 47)]
        )
        current_resolution_pass = (
            report["current_terminal_resolution"] == expected_resolution
            and expected_resolution["resolved_terminal_state"]
            == QUALIFIED_TERMINAL_STATE
            and expected_resolution["releases_t9_1_4"] is True
            and expected_resolution["matched_phase9_ranking_eligible"]
            is False
        )
    elif no_go:
        manifest = _validate_failure_manifest_live(
            report["failure_manifest"],
            verify_live_files=verify_live_files,
        )
        expected_bindings = {
            key: copy.deepcopy(manifest["failure_evidence"][key])
            for key in ("archive", "archive_seal", "inventory")
        }
        expected_snapshot = _no_go_parent_snapshot(manifest)
        expected_gradient = _no_go_gradient_state(manifest)
        expected_summary = _no_go_deviation_state(manifest)
        expected_resolution = _current_terminal_resolution(
            {"status": manifest["terminal_state"]}, mappings
        )
        parent_binding_contract = report["parent_bindings"] == expected_bindings
        parent_bindings_live = all(
            _binding_is_live(value)
            for value in report["parent_bindings"].values()
        )
        parent_semantics_pass = (
            report["parent_snapshot"] == expected_snapshot
            and expected_snapshot["valid_parent_pass_seal"] is False
            and expected_snapshot["terminal_result"]
            == NO_GO_TERMINAL_STATE
            and expected_snapshot["official_exact"] is None
            and expected_snapshot["puviani_surpass"] is None
            and expected_snapshot["paper_scale_lifetime"] is None
        )
        nonretroactive_evidence = (
            report["addendum_deviations"] == []
            and report["failure_manifest"]["semantic_sha256"]
            == _failure_manifest_semantic_sha256(
                report["failure_manifest"]
            )
        )
        gradient_pass = (
            report["gradient_clipping_reanalysis"] == expected_gradient
            and expected_gradient["state"] == "NOT_EVALUATED_TYPED_NULL"
            and expected_gradient["value"] is None
            and expected_gradient["row_census"] is None
            and expected_gradient["scopes"] is None
        )
        deviations_pass = report["addendum_deviations"] == []
        deviation_census_pass = (
            report["deviation_summary"] == expected_summary
            and expected_summary["state"] == "NOT_EVALUATED_TYPED_NULL"
            and expected_summary["value"] is None
            and expected_summary["combined_entry_count"] is None
            and expected_summary["combined_severity_counts"] is None
            and expected_summary["combined_ids"] is None
        )
        current_resolution_pass = (
            report["current_terminal_resolution"] == expected_resolution
            and expected_resolution["resolved_terminal_state"]
            == NO_GO_TERMINAL_STATE
            and expected_resolution["releases_t9_1_4"] is True
            and expected_resolution["matched_phase9_ranking_eligible"]
            is False
            and all(
                expected_resolution["typed_payload"][key] is None
                for key in (
                    "parent_report_sha256",
                    "numeric_scope",
                    "selected_controller",
                    "numeric_metrics",
                    "rank",
                    "official_exact",
                    "puviani_surpass",
                    "paper_scale_lifetime",
                )
            )
        )
    else:
        raise ValueError("terminal_branch must be QUALIFIED or NO_GO")

    source_path = _resolve_path(str(report["source_data"]["path"]))
    expected_source_rows = _source_records(report)
    actual_source_rows = (
        _read_source_data(source_path) if source_path.is_file() else []
    )
    expected_source_binding = (
        {
            **_binding(source_path),
            "rows": len(expected_source_rows),
            "columns": list(SOURCE_COLUMNS),
            "payload_encoding": (
                "canonical-json-utf8-sha256; every nested field retained"
            ),
        }
        if source_path.is_file()
        else {}
    )
    no_go_payload = report["terminal_state_mapping"][1]["typed_payload"]
    no_go_nullable_keys = {
        "parent_report_sha256",
        "numeric_scope",
        "selected_controller",
        "numeric_metrics",
        "rank",
        "official_exact",
        "puviani_surpass",
        "paper_scale_lifetime",
    }
    mutation = report["semantic_mutation_audit"]
    expected_downstream_seal = _downstream_semantic_seal(report)
    return {
        "G01_identity_status_and_closed_schema": (
            set(report) == REPORT_KEYS
            and report["schema_version"] == SCHEMA_VERSION
            and report["task_id"] == TASK_ID
            and report["status"] == STATUS
            and branch in {QUALIFIED_BRANCH, NO_GO_BRANCH}
        ),
        "G02_input_contract_is_frozen_hash_bound_and_live": (
            report["input_contract"] == _binding(contract_path)
            and (
                not verify_live_files
                or _binding_is_live(report["input_contract"])
            )
        ),
        "G03_terminal_evidence_is_branch_local_hash_bound_and_live": (
            parent_binding_contract
            and (
                not verify_live_files
                or parent_bindings_live
            )
        ),
        "G04_terminal_semantic_evidence_matches_selected_branch": (
            parent_semantics_pass
        ),
        "G05_addendum_is_nonretroactive_and_parent_reseal_is_prohibited": (
            report["governance_policy"]
            == {
                "mode": STATUS,
                "parent_report_reseal": False,
                "parent_deviation_ledger_edit": False,
                "parent_artifact_tree_edit": False,
                "addendum_only_ids": ["D43", "D44", "D45", "D46"],
                "effect_on_parent_analysis_sha256": "NONE",
                "effect_on_parent_verdict": "NONE",
            }
            and nonretroactive_evidence
        ),
        "G06_gradient_evidence_is_complete_or_typed_null_by_branch": gradient_pass,
        "G07_D43_D46_are_exact_or_typed_null_by_branch": deviations_pass,
        "G08_deviation_census_is_exact_or_typed_null_by_branch": (
            deviation_census_pass
        ),
        "G09_official_surpass_and_paper_lifetime_claims_stay_null": (
            report["claim_slots"] == CLAIM_NULLS
            and contract["claim_null_contract"] == CLAIM_NULLS
            and all(value is None for value in report["claim_slots"].values())
        ),
        "G10_qualified_and_typed_no_go_both_release_T9_1_4": (
            report["terminal_state_mapping"] == mappings
            and len(report["terminal_state_mapping"]) == 2
            and all(
                row["releases_t9_1_4"] is True
                for row in report["terminal_state_mapping"]
            )
            and report["terminal_state_mapping"][0]["terminal_state"]
            == QUALIFIED_TERMINAL_STATE
            and report["terminal_state_mapping"][1]["terminal_state"]
            == NO_GO_TERMINAL_STATE
            and all(no_go_payload[key] is None for key in no_go_nullable_keys)
            and no_go_payload["evidence_grade"]
            == "PAPER_CONSTRAINED_REIMPLEMENTATION"
        ),
        "G11_current_terminal_resolution_matches_executable_branch": (
            current_resolution_pass
            and report["current_terminal_resolution"]["typed_payload"][
                "official_exact"
            ]
            is None
            and report["current_terminal_resolution"]["typed_payload"][
                "puviani_surpass"
            ]
            is None
            and report["current_terminal_resolution"]["typed_payload"][
                "paper_scale_lifetime"
            ]
            is None
        ),
        "G12_registry_context_cannot_be_promoted_to_matched_or_SOTA_rank": (
            report["ranking_boundary"] == contract["ranking_boundary"]
            and report["ranking_boundary"][
                "matched_phase9_ranking_eligible"
            ]
            is False
            and report["ranking_boundary"]["sota_claim_eligible"] is False
            and all(
                row["matched_phase9_ranking_eligible"] is False
                for row in report["terminal_state_mapping"]
            )
        ),
        "G13_source_csv_is_lossless_unique_and_hash_bound": (
            report["source_data"] == expected_source_binding
            and actual_source_rows == expected_source_rows
            and len(
                {
                    (row["section"], row["record_id"])
                    for row in actual_source_rows
                }
            )
            == len(actual_source_rows)
            and all(
                hashlib.sha256(row["payload_json"].encode("utf-8")).hexdigest()
                == row["payload_sha256"]
                and _canonical_json(json.loads(row["payload_json"]))
                == row["payload_json"]
                for row in actual_source_rows
            )
        ),
        "G14_generator_implementation_is_hash_bound": (
            report["implementation"] == _binding(IMPLEMENTATION)
            and (
                not verify_live_files
                or _binding_is_live(report["implementation"])
            )
        ),
        "G15_targeted_semantic_mutations_all_fail_closed": (
            mutation.get("count") == EXPECTED_MUTATION_COUNT
            and mutation.get("detected") == EXPECTED_MUTATION_COUNT
            and mutation.get("all_detected") is True
            and mutation.get("terminal_branch") == branch
            and mutation.get("branch_case_count") == EXPECTED_MUTATION_COUNT
            and len(mutation.get("cases", [])) == EXPECTED_MUTATION_COUNT
            and all(
                row.get("terminal_branch") == branch
                and row.get("mutation_applied") is True
                and row.get("rejected") is True
                and row.get("rejection_mode") == "TARGET_GATE_FALSE"
                for row in mutation["cases"]
            )
        ),
        "G16_T9_1_4_consumes_one_way_versioned_semantic_seal": (
            report["downstream_semantic_seal"] == expected_downstream_seal
            and expected_downstream_seal["schema_version"]
            == SEMANTIC_SEAL_VERSION
            and expected_downstream_seal[
                "raw_addendum_hash_required_by_T9_1_4"
            ]
            is False
            and contract["downstream_addendum_semantic_contract"][
                "accepted_addendum_schema_versions"
            ]
            == [SCHEMA_VERSION]
            and contract["downstream_addendum_semantic_contract"][
                "raw_hash_dependency"
            ]
            == "PROHIBITED_TO_AVOID_CONTRACT_REPORT_CYCLE"
            and expected_downstream_seal["consumer_rule"]
            == DOWNSTREAM_CONSUMER_RULE
            and contract["downstream_addendum_semantic_contract"][
                "consumer_rule"
            ]
            == DOWNSTREAM_CONSUMER_RULE
        ),
    }


def _semantic_mutation_audit(report: Mapping[str, Any]) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    branch = str(report["terminal_branch"])
    baseline_gate_names = set(
        evaluate_gates(report, verify_live_files=False)
    )

    def attempt(
        mutation_id: str,
        target_gate: str,
        change: Any,
    ) -> None:
        if target_gate not in baseline_gate_names:
            raise AssertionError(
                f"mutation {mutation_id} targets unknown gate {target_gate}"
            )
        candidate = copy.deepcopy(dict(report))
        candidate["semantic_mutation_audit"] = {
            "count": EXPECTED_MUTATION_COUNT,
            "detected": EXPECTED_MUTATION_COUNT,
            "all_detected": True,
            "terminal_branch": branch,
            "branch_case_count": EXPECTED_MUTATION_COUNT,
            "cases": [],
        }
        try:
            change(candidate)
        except Exception as error:
            raise AssertionError(
                f"mutation {mutation_id} could not be applied"
            ) from error
        mutated_gates = evaluate_gates(
            candidate, verify_live_files=False
        )
        if target_gate not in mutated_gates:
            raise AssertionError(
                f"mutation {mutation_id} lost target gate {target_gate}"
            )
        rejected = mutated_gates[target_gate] is False
        rejection_mode = "TARGET_GATE_FALSE"
        cases.append(
            {
                "mutation_id": mutation_id,
                "terminal_branch": branch,
                "target_gate": target_gate,
                "mutation_applied": True,
                "rejection_mode": rejection_mode,
                "rejected": rejected,
            }
        )

    attempt(
        "fill_official_exact_claim",
        "G09_official_surpass_and_paper_lifetime_claims_stay_null",
        lambda value: value["claim_slots"].update(
            official_exact={"claimed": True}
        ),
    )
    attempt(
        "fill_paper_scale_lifetime_claim",
        "G09_official_surpass_and_paper_lifetime_claims_stay_null",
        lambda value: value["claim_slots"].update(
            paper_scale_lifetime={"T_ch": 1000.0}
        ),
    )

    def change_terminal_binding(value: dict[str, Any]) -> None:
        first_key = sorted(value["parent_bindings"])[0]
        value["parent_bindings"][first_key]["sha256"] = "0" * 64

    attempt(
        "change_terminal_evidence_hash",
        "G03_terminal_evidence_is_branch_local_hash_bound_and_live",
        change_terminal_binding,
    )

    def change_terminal_semantics(value: dict[str, Any]) -> None:
        if branch == QUALIFIED_BRANCH:
            value["parent_snapshot"]["analysis_sha256"] = "0" * 64
        else:
            value["failure_manifest"]["reason"]["message"] += " tampered"
            value["failure_manifest"][
                "semantic_sha256"
            ] = _failure_manifest_semantic_sha256(
                value["failure_manifest"]
            )

    attempt(
        "change_terminal_semantic_evidence",
        "G04_terminal_semantic_evidence_matches_selected_branch",
        change_terminal_semantics,
    )
    attempt(
        "make_no_go_block_T9_1_4",
        "G10_qualified_and_typed_no_go_both_release_T9_1_4",
        lambda value: value["terminal_state_mapping"][1].update(
            releases_t9_1_4=False
        ),
    )
    attempt(
        "fill_no_go_numeric_payload",
        "G10_qualified_and_typed_no_go_both_release_T9_1_4",
        lambda value: value["terminal_state_mapping"][1][
            "typed_payload"
        ].update(numeric_metrics={"lifetime": 1.0}),
    )
    attempt(
        "promote_to_matched_rank",
        "G12_registry_context_cannot_be_promoted_to_matched_or_SOTA_rank",
        lambda value: value["ranking_boundary"].update(
            matched_phase9_ranking_eligible=True
        ),
    )

    def drift_deviation_count(value: dict[str, Any]) -> None:
        value["deviation_summary"]["combined_entry_count"] = (
            45 if branch == QUALIFIED_BRANCH else 0
        )

    attempt(
        "drift_combined_deviation_count",
        "G08_deviation_census_is_exact_or_typed_null_by_branch",
        drift_deviation_count,
    )

    def forge_gradient(value: dict[str, Any]) -> None:
        if branch == QUALIFIED_BRANCH:
            value["gradient_clipping_reanalysis"]["scopes"]["all"][
                "rows_scaled_by_pytorch_clip_rule"
            ] = 1
        else:
            value["gradient_clipping_reanalysis"]["value"] = 0.0

    attempt(
        "forge_clipped_update",
        "G06_gradient_evidence_is_complete_or_typed_null_by_branch",
        forge_gradient,
    )

    def mutate_addendum_deviation(value: dict[str, Any]) -> None:
        if branch == QUALIFIED_BRANCH:
            value["addendum_deviations"].pop()
        else:
            value["addendum_deviations"].append(
                {"id": "D43", "paper_numeric_anchor_eligible": False}
            )

    attempt(
        "mutate_branch_deviation_payload",
        "G07_D43_D46_are_exact_or_typed_null_by_branch",
        mutate_addendum_deviation,
    )
    attempt(
        "retroactively_reseal_parent",
        "G05_addendum_is_nonretroactive_and_parent_reseal_is_prohibited",
        lambda value: value["governance_policy"].update(
            parent_report_reseal=True
        ),
    )
    attempt(
        "forge_source_row_count",
        "G13_source_csv_is_lossless_unique_and_hash_bound",
        lambda value: value["source_data"].update(rows=0),
    )
    attempt(
        "forge_downstream_semantic_seal",
        "G16_T9_1_4_consumes_one_way_versioned_semantic_seal",
        lambda value: value["downstream_semantic_seal"].update(
            semantic_sha256="0" * 64
        ),
    )
    attempt(
        "forge_mutation_count",
        "G15_targeted_semantic_mutations_all_fail_closed",
        lambda value: value.update(
            semantic_mutation_audit={
                "count": EXPECTED_MUTATION_COUNT,
                "detected": EXPECTED_MUTATION_COUNT - 1,
                "all_detected": False,
                "terminal_branch": branch,
                "branch_case_count": EXPECTED_MUTATION_COUNT,
                "cases": [],
            }
        ),
    )
    return {
        "count": len(cases),
        "detected": sum(row["rejected"] for row in cases),
        "all_detected": all(row["rejected"] for row in cases),
        "terminal_branch": branch,
        "branch_case_count": len(cases),
        "cases": cases,
    }


def _analysis_sha256(report: Mapping[str, Any]) -> str:
    return _canonical_sha256({key: report[key] for key in ANALYSIS_FIELDS})


def build_report(
    *,
    source_data_path: Path = DEFAULT_SOURCE_DATA,
    input_contract_path: Path = DEFAULT_INPUT_CONTRACT,
    terminal_branch: str = QUALIFIED_BRANCH,
    failure_manifest: Mapping[str, Any] | str | Path | None = None,
) -> dict[str, Any]:
    contract = _validate_input_contract(input_contract_path)
    branch = terminal_branch.upper()
    if branch not in {QUALIFIED_BRANCH, NO_GO_BRANCH}:
        raise ValueError("terminal_branch must be QUALIFIED or NO_GO")
    mappings = copy.deepcopy(contract["terminal_state_mapping"])
    if branch == QUALIFIED_BRANCH:
        if failure_manifest is not None:
            raise ValueError(
                "QUALIFIED branch must not receive a failure manifest"
            )
        parent_path = _resolve_path(
            str(contract["parent_artifacts"]["parent_report"]["path"])
        )
        config_path = _resolve_path(
            str(contract["parent_artifacts"]["production_config"]["path"])
        )
        deviation_path = _resolve_path(
            str(contract["parent_artifacts"]["deviation_ledger"]["path"])
        )
        training_path = _resolve_path(
            str(contract["parent_artifacts"]["training_ledger"]["path"])
        )
        parent = _load_json(parent_path)
        config = _load_json(config_path)
        parent_deviations = _load_json(deviation_path)["deviations"]
        gradient = _gradient_clipping_reanalysis(training_path, config)
        addendum = _addendum_deviations(gradient, config)
        deviation_summary = _deviation_summary(
            parent_deviations, addendum
        )
        parent_bindings = copy.deepcopy(contract["parent_artifacts"])
        parent_snapshot = _parent_snapshot(parent, contract)
        manifest_value: dict[str, Any] | None = None
        generated_at = str(parent["generated_at_utc"])
        resolution_parent: Mapping[str, Any] = parent
    else:
        if failure_manifest is None:
            raise ValueError(
                "NO_GO branch requires an external failure manifest"
            )
        raw_manifest = (
            copy.deepcopy(dict(failure_manifest))
            if isinstance(failure_manifest, Mapping)
            else _load_json(Path(failure_manifest))
        )
        manifest_value = _validate_failure_manifest_live(
            raw_manifest,
            verify_live_files=True,
        )
        parent_bindings = {
            key: copy.deepcopy(
                manifest_value["failure_evidence"][key]
            )
            for key in ("archive", "archive_seal", "inventory")
        }
        parent_snapshot = _no_go_parent_snapshot(manifest_value)
        gradient = _no_go_gradient_state(manifest_value)
        addendum = []
        deviation_summary = _no_go_deviation_state(manifest_value)
        generated_at = str(manifest_value["observed_at_utc"])
        resolution_parent = {"status": manifest_value["terminal_state"]}
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "status": STATUS,
        "terminal_branch": branch,
        "generated_from_parent_at_utc": generated_at,
        "input_contract": _binding(input_contract_path),
        "implementation": _binding(IMPLEMENTATION),
        "parent_bindings": parent_bindings,
        "parent_snapshot": parent_snapshot,
        "failure_manifest": manifest_value,
        "governance_policy": {
            "mode": STATUS,
            "parent_report_reseal": False,
            "parent_deviation_ledger_edit": False,
            "parent_artifact_tree_edit": False,
            "addendum_only_ids": ["D43", "D44", "D45", "D46"],
            "effect_on_parent_analysis_sha256": "NONE",
            "effect_on_parent_verdict": "NONE",
        },
        "gradient_clipping_reanalysis": gradient,
        "addendum_deviations": addendum,
        "deviation_summary": deviation_summary,
        "terminal_state_mapping": mappings,
        "current_terminal_resolution": _current_terminal_resolution(
            resolution_parent, mappings
        ),
        "claim_slots": copy.deepcopy(CLAIM_NULLS),
        "ranking_boundary": copy.deepcopy(contract["ranking_boundary"]),
        "downstream_semantic_seal": {},
        "source_data": {},
        "semantic_mutation_audit": {
            "count": EXPECTED_MUTATION_COUNT,
            "detected": EXPECTED_MUTATION_COUNT,
            "all_detected": True,
            "terminal_branch": branch,
            "branch_case_count": EXPECTED_MUTATION_COUNT,
            "cases": [],
        },
        "gates": {},
        "gate_summary": {},
        "verdict": VERDICT_PASS,
        "analysis_sha256": "",
    }
    report["downstream_semantic_seal"] = _downstream_semantic_seal(
        report
    )
    source_rows = _write_source_data(report, source_data_path)
    report["source_data"] = {
        **_binding(source_data_path),
        "rows": len(source_rows),
        "columns": list(SOURCE_COLUMNS),
        "payload_encoding": (
            "canonical-json-utf8-sha256; every nested field retained"
        ),
    }
    report["semantic_mutation_audit"] = _semantic_mutation_audit(report)
    report["gates"] = evaluate_gates(
        report,
        verify_live_files=True,
        input_contract_path=input_contract_path,
    )
    failed = [key for key, passed in report["gates"].items() if not passed]
    report["gate_summary"] = {
        "passed": sum(report["gates"].values()),
        "total": len(report["gates"]),
        "failed": failed,
    }
    report["verdict"] = VERDICT_PASS if not failed else VERDICT_FAIL
    report["analysis_sha256"] = _analysis_sha256(report)
    return report


def validate_report(
    payload_or_path: Mapping[str, Any] | str | Path = DEFAULT_REPORT,
    *,
    verify_live_files: bool = True,
    input_contract_path: Path | None = None,
) -> dict[str, bool]:
    report = (
        copy.deepcopy(dict(payload_or_path))
        if isinstance(payload_or_path, Mapping)
        else _load_json(Path(payload_or_path))
    )
    if set(report) != REPORT_KEYS:
        raise ValueError("post-outcome report schema drifted")
    expected_mutations = _semantic_mutation_audit(report)
    gates = evaluate_gates(
        report,
        verify_live_files=verify_live_files,
        input_contract_path=input_contract_path,
    )
    expected_summary = {
        "passed": sum(gates.values()),
        "total": len(gates),
        "failed": [key for key, passed in gates.items() if not passed],
    }
    expected_verdict = (
        VERDICT_PASS if not expected_summary["failed"] else VERDICT_FAIL
    )
    checks = {
        "identity": (
            report["schema_version"] == SCHEMA_VERSION
            and report["task_id"] == TASK_ID
            and report["status"] == STATUS
        ),
        "mutation_replay": (
            report["semantic_mutation_audit"] == expected_mutations
        ),
        "gates": report["gates"] == gates and all(gates.values()),
        "gate_summary": report["gate_summary"] == expected_summary,
        "verdict": report["verdict"] == expected_verdict == VERDICT_PASS,
        "analysis_sha256": (
            report["analysis_sha256"] == _analysis_sha256(report)
        ),
    }
    if not all(checks.values()):
        raise ValueError(
            "T9.1.3 post-outcome addendum verification failed: "
            + ", ".join(key for key, passed in checks.items() if not passed)
        )
    return checks


def validate_downstream_handoff(
    payload_or_path: Mapping[str, Any] | str | Path = DEFAULT_REPORT,
    *,
    input_contract_path: Path | None = None,
) -> dict[str, bool]:
    """Apply the ordered T9.1.4 consumer rule with live evidence enabled."""

    report = (
        copy.deepcopy(dict(payload_or_path))
        if isinstance(payload_or_path, Mapping)
        else _load_json(Path(payload_or_path))
    )
    checks = validate_report(
        report,
        verify_live_files=True,
        input_contract_path=input_contract_path,
    )
    reconstructed = _downstream_semantic_seal(report)
    semantic_match = report["downstream_semantic_seal"] == reconstructed
    if not semantic_match:
        raise ValueError(
            "T9.1.4 downstream semantic seal reconstruction failed"
        )
    return {
        **checks,
        "downstream_semantic_seal_reconstructed": semantic_match,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--source-data", type=Path, default=DEFAULT_SOURCE_DATA
    )
    parser.add_argument(
        "--input-contract", type=Path, default=DEFAULT_INPUT_CONTRACT
    )
    parser.add_argument(
        "--terminal-branch",
        choices=(QUALIFIED_BRANCH, NO_GO_BRANCH),
        default=QUALIFIED_BRANCH,
    )
    parser.add_argument(
        "--failure-manifest",
        type=Path,
        default=None,
        help="Required external manifest when --terminal-branch=NO_GO.",
    )
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)
    if args.verify:
        checks = validate_downstream_handoff(
            args.report,
            input_contract_path=args.input_contract,
        )
        print(
            _canonical_json(
                {
                    "verified": _path_label(args.report),
                    "checks": checks,
                    "verdict": VERDICT_PASS,
                }
            )
        )
        return 0
    report = build_report(
        source_data_path=args.source_data,
        input_contract_path=args.input_contract,
        terminal_branch=args.terminal_branch,
        failure_manifest=args.failure_manifest,
    )
    _atomic_json(args.report, report)
    validate_report(
        args.report,
        verify_live_files=True,
        input_contract_path=args.input_contract,
    )
    print(
        _canonical_json(
            {
                "output": _path_label(args.report),
                "source_data": _path_label(args.source_data),
                "deviations": report["deviation_summary"][
                    "combined_severity_counts"
                ],
                "terminal_branch": report["terminal_branch"],
                "gradient_clipped_updates": (
                    report["gradient_clipping_reanalysis"]["scopes"]["all"][
                        "rows_scaled_by_pytorch_clip_rule"
                    ]
                    if report["terminal_branch"] == QUALIFIED_BRANCH
                    else None
                ),
                "gates": report["gate_summary"],
                "mutations": report["semantic_mutation_audit"],
                "verdict": report["verdict"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
