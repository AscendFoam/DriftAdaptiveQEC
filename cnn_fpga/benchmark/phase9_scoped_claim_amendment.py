"""Compile and verify the parent-bound T9.1.5 scoped-claim amendment.

The compiler does not run a decoder and does not emit a performance result.
It binds the immutable T9.1.1 protocol and T9.1.4 registry, splits ambiguous
legacy candidate labels into separately auditable claim states, and keeps all
currently unsupported result fields as typed nulls.
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
import re
from typing import Any, Callable, Mapping, Sequence

from cnn_fpga.benchmark import phase9_baseline_search_power_registry
from cnn_fpga.benchmark import phase9_three_lane_protocol


ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "T9.1.5"
CONFIG_SCHEMA_VERSION = "t9.1.5-scoped-claim-amendment-config-v1"
REPORT_SCHEMA_VERSION = "t9.1.5-scoped-claim-amendment-report-v1"
PROTOCOL_ID = "PHASE9-SCOPED-CLAIM-AMENDMENT-V2"
VERDICT = "PASS_T9_1_5_PARENT_BOUND_SCOPED_CLAIM_AMENDMENT"
REVOCATION_LEDGER_SCHEMA_VERSION = (
    "phase9-scoped-claim-revocation-ledger-v2"
)
REVOCATION_LEDGER_GENESIS_SHA256 = hashlib.sha256(
    f"{PROTOCOL_ID}:{REVOCATION_LEDGER_SCHEMA_VERSION}:GENESIS".encode(
        "utf-8"
    )
).hexdigest()

DEFAULT_CONFIG = (
    ROOT / "configs/phase9/t9_1_5_scoped_claim_amendment.json"
)
DEFAULT_REPORT = ROOT / "docs/t9_1_5_scoped_claim_amendment.json"
DEFAULT_SOURCE_DATA = (
    ROOT / "docs/t9_1_5_scoped_claim_amendment_source_data.csv"
)
DEFAULT_MARKDOWN = ROOT / "docs/phase9_scoped_claim_amendment.md"
DEFAULT_RELEASE_PIN = ROOT / "configs/phase9/t9_1_5_release_pin.json"
IMPLEMENTATION = Path(__file__).resolve()

STATE_IDS = (
    "GO_LER_REGISTERED_BEST",
    "GO_LER_EXTERNAL_SOTA",
    "GO_LIFETIME_PROJECT_NATIVE",
    "GO_LIFETIME_EXTERNAL_SOTA",
    "OFFICIAL_PUVIANI_EXACT",
    "GO_PUVIANI_NMF_SURPASS",
    "GO_PHYSICAL_LIFETIME",
    "GO_HIL_INTEGRATED",
    "GO_HIL_EXTERNAL_SPEED",
)

GATE_IDS = (
    "G01_identity_and_parent_bound_preoutcome_status",
    "G02_t9_1_1_parent_is_live_verified_and_byte_immutable",
    "G03_parent_legacy_outputs_are_dynamically_and_completely_covered",
    "G04_t9_1_4_registry_is_live_verified_and_exactly_bound",
    "G05_literature_cutoff_dedup_ledger_and_current_eligibility_are_bound",
    "G06_config_implementation_and_parent_artifacts_are_live",
    "G07_downstream_consumers_validate_then_reconstruct_then_compare",
    "G08_scoped_state_ids_are_exact_unique_and_lane_bound",
    "G09_current_state_objects_have_a_strict_typed_null_schema",
    "G10_all_current_claim_performance_rank_and_vote_values_are_null",
    "G11_official_board_external_speed_and_qpu_inventory_is_typed_null",
    "G12_protocol_pass_does_not_open_any_performance_claim",
    "G13_migration_table_has_exactly_29_lane_qualified_parent_outputs",
    "G14_no_legacy_output_automatically_promotes_a_v2_state",
    "G15_legacy_ler_go_is_registered_candidate_only",
    "G16_legacy_lifetime_go_is_project_native_candidate_only",
    "G17_legacy_hil_go_is_integrated_candidate_only",
    "G18_null_incomplete_negative_and_supporting_outputs_open_no_go_state",
    "G19_registered_ler_predicate_requires_all_matched_formal_evidence",
    "G20_external_ler_requires_registered_go_complete_ledger_and_audit",
    "G21_project_native_lifetime_requires_six_state_cost_and_two_backends",
    "G22_external_lifetime_requires_project_native_and_external_evidence",
    "G23_physical_lifetime_requires_qpu_real_gkp_raw_measurements",
    "G24_integrated_hil_requires_real_board_full_chain_measurement",
    "G25_external_speed_requires_integrated_hil_and_same_boundary_comparators",
    "G26_official_puviani_exact_and_surpass_are_separate_and_locally_blocked",
    "G27_external_eligibility_inherits_all_nine_t9_1_4_signature_checks",
    "G28_cutoff_or_ledger_change_requires_reseal_and_external_revocation",
    "G29_every_state_has_fail_closed_append_only_revocation_semantics",
    "G30_revocation_propagates_to_broader_states_as_null_not_no_go",
    "G31_forbidden_transfer_matrix_is_exact_and_complete",
    "G32_preboard_six_cycle_recorded_live_and_external_hil_levels_do_not_mix",
    "G33_registered_project_integrated_states_never_auto_promote_broader_states",
    "G34_weighted_score_winner_count_and_paper_scope_cannot_derive_sota",
    "G35_source_data_and_markdown_are_lossless_live_and_complete",
    "G36_one_semantic_mutation_per_gate_is_replayed_and_rejected",
)

PARENT_V1_REPORT = ROOT / "docs/t9_1_1_three_lane_protocol.json"
PARENT_REGISTRY_REPORT = (
    ROOT / "docs/t9_1_4_baseline_search_power_registry.json"
)

PREREQUISITE_EDGES = (
    ("GO_LER_REGISTERED_BEST", "GO_LER_EXTERNAL_SOTA"),
    ("GO_LIFETIME_PROJECT_NATIVE", "GO_LIFETIME_EXTERNAL_SOTA"),
    ("GO_LIFETIME_PROJECT_NATIVE", "GO_PUVIANI_NMF_SURPASS"),
    ("OFFICIAL_PUVIANI_EXACT", "GO_PUVIANI_NMF_SURPASS"),
    ("GO_HIL_INTEGRATED", "GO_HIL_EXTERNAL_SPEED"),
)

ASSET_INVENTORY_KEYS = (
    "official_puviani_exact",
    "puviani_nmf_surpass",
    "board_recorded_replay",
    "board_live_raw_iq_hil",
    "external_measured_speed",
    "qpu_physical_lifetime",
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _binding(path: Path) -> dict[str, Any]:
    return {
        "path": _relative(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _binding_live(binding: Mapping[str, Any]) -> bool:
    path = ROOT / str(binding["path"])
    return (
        path.is_file()
        and path.stat().st_size == binding["bytes"]
        and _sha256(path) == binding["sha256"]
    )


def _atomic_text(value: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8", newline="\n")
    temporary.replace(path)


def _atomic_json(value: Mapping[str, Any], path: Path) -> None:
    _atomic_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", path
    )


def _safe(check: Callable[[], Any]) -> bool:
    try:
        return bool(check())
    except (AssertionError, KeyError, TypeError, ValueError, IndexError):
        return False


def _parent_legacy_outputs(
    parent: Mapping[str, Any],
) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for lane in parent["lanes"]:
        gate = lane["result_gate"]
        labels = [
            gate["go_verdict"],
            gate["no_go_verdict"],
            *gate["failure_branches"],
        ]
        rows.extend((lane["lane_id"], label) for label in labels)
    return rows


def _parent_summary(
    config: Mapping[str, Any], *, verify_live: bool
) -> dict[str, Any]:
    parent = _load(PARENT_V1_REPORT)
    registry = _load(PARENT_REGISTRY_REPORT)
    if verify_live:
        phase9_three_lane_protocol.verify_report(parent)
        phase9_baseline_search_power_registry.verify_report(registry)
    return {
        "v1": {
            "task_id": parent["task_id"],
            "schema_version": parent["schema_version"],
            "protocol_id": parent["protocol_id"],
            "verdict": parent["verdict"],
            "analysis_sha256": parent["analysis_sha256"],
            "raw_report": _binding(PARENT_V1_REPORT),
            "legacy_lane_outputs": [
                {"lane": lane, "output": output}
                for lane, output in _parent_legacy_outputs(parent)
            ],
            "current_result_verdicts": {
                lane["lane_id"]: lane["current_result"]["result_verdict"]
                for lane in parent["lanes"]
            },
        },
        "registry": {
            "task_id": registry["task_id"],
            "schema_version": registry["schema_version"],
            "protocol_id": registry["protocol_id"],
            "verdict": registry["verdict"],
            "analysis_sha256": registry["analysis_sha256"],
            "raw_report": _binding(PARENT_REGISTRY_REPORT),
            "literature_cutoff_inclusive": registry[
                "literature_cutoff_inclusive"
            ],
            "canonical_work_count": len(registry["literature_records"]),
            "external_same_task_eligible_count": registry[
                "external_claim_contract"
            ]["current_external_same_task_eligible_count"],
            "external_claim_contract": copy.deepcopy(
                registry["external_claim_contract"]
            ),
            "performance_state": copy.deepcopy(
                registry["performance_state"]
            ),
        },
        "ordered_consumer_checks": {
            "validate_t9_1_1_live": verify_live,
            "validate_t9_1_4_live": verify_live,
            "reconstruct_parent_analysis_hashes": (
                parent["analysis_sha256"]
                == config["parent_contract"]["parent_analysis_sha256"]
                and registry["analysis_sha256"]
                == config["registry_parent_contract"][
                    "parent_analysis_sha256"
                ]
            ),
            "compare_parent_contracts": (
                _binding(PARENT_V1_REPORT)
                == config["parent_contract"]["parent_report"]
                and _binding(PARENT_REGISTRY_REPORT)
                == config["registry_parent_contract"]["parent_report"]
            ),
        },
    }


def _current_states(
    definitions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for definition in definitions:
        reason = str(definition["current_reason"])
        state_id = str(definition["state_id"])
        if state_id in {"GO_HIL_INTEGRATED", "GO_HIL_EXTERNAL_SPEED"}:
            status = "MISSING_BOARD_NULL"
        elif state_id in {
            "OFFICIAL_PUVIANI_EXACT",
            "GO_PUVIANI_NMF_SURPASS",
        }:
            status = "MISSING_EXTERNAL_ASSET_NULL"
        elif state_id == "GO_PHYSICAL_LIFETIME":
            status = "MISSING_QPU_NULL"
        elif state_id in {
            "GO_LER_EXTERNAL_SOTA",
            "GO_LIFETIME_EXTERNAL_SOTA",
        }:
            status = "MISSING_EXTERNAL_COMPARATOR_NULL"
        else:
            status = "NOT_EVALUATED_NULL"
        rows.append(
            {
                "state_id": definition["state_id"],
                "lane": definition["lane"],
                "evidence_scope": definition["evidence_scope"],
                "status": status,
                "value": None,
                "verdict": None,
                "evidence_grade": None,
                "evidence_refs": [],
                "numeric_metrics": None,
                "rank": None,
                "vote": None,
                "opened": False,
                "null_reason": reason,
                "revocation": {
                    "status": "NOT_REVOKED",
                    "reason": None,
                    "evidence_hashes": [],
                },
            }
        )
    return rows


def _asset_inventory() -> dict[str, dict[str, Any]]:
    reasons = {
        "official_puviani_exact": "MISSING_EXTERNAL_ASSET",
        "puviani_nmf_surpass": "OFFICIAL_EXACT_PREREQUISITE_NULL",
        "board_recorded_replay": "MISSING_BOARD",
        "board_live_raw_iq_hil": "MISSING_BOARD",
        "external_measured_speed": "MISSING_BOARD_AND_COMPARATOR",
        "qpu_physical_lifetime": "MISSING_QPU_REAL_GKP",
    }
    return {
        key: {
            "value": None,
            "verdict": None,
            "evidence_grade": None,
            "evidence_refs": [],
            "status": reasons[key],
        }
        for key in ASSET_INVENTORY_KEYS
    }


def _predicate_missing(
    definition: Mapping[str, Any], evidence: Mapping[str, Any]
) -> list[str]:
    predicate = definition["predicate"]
    missing: list[str] = []
    for key in predicate.get("true", []):
        if evidence.get(key) is not True:
            missing.append(key)
    for key in predicate.get("non_null", []):
        value = evidence.get(key)
        if not isinstance(value, str) or re.fullmatch(
            r"[0-9a-f]{64}", value
        ) is None:
            missing.append(key)
    for key in predicate.get("positive", []):
        value = evidence.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not math.isfinite(value)
        ):
            missing.append(key)
        elif value <= 0:
            missing.append(key)
    for key in predicate.get("zero", []):
        value = evidence.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not math.isfinite(value)
            or value != 0
        ):
            missing.append(key)
    for key, threshold in predicate.get("minimum", {}).items():
        value = evidence.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not math.isfinite(value)
            or value < threshold
        ):
            missing.append(key)
    return missing


def evaluate_claim_candidate(
    definition: Mapping[str, Any],
    evidence: Mapping[str, Any],
    *,
    revoked: bool = False,
) -> dict[str, Any]:
    """Validate schema completeness without granting a scientific GO claim.

    Actual promotion belongs to a downstream task-specific verifier that
    checks live source/config/data/result bindings and the canonical release
    pin.  This generic mapping evaluator therefore always returns null
    value/verdict, even when every predicate is structurally complete.
    """

    fixture_only = evidence.get("_fixture_only") is True
    if fixture_only:
        return {
            "state_id": definition["state_id"],
            "status": "FIXTURE_ONLY_NULL",
            "value": None,
            "verdict": None,
            "missing_conditions": [],
            "fixture_only": True,
        }
    missing = _predicate_missing(definition, evidence)
    if revoked:
        return {
            "state_id": definition["state_id"],
            "status": "REVOKED_NULL",
            "value": None,
            "verdict": None,
            "missing_conditions": [],
            "fixture_only": False,
        }
    if missing:
        return {
            "state_id": definition["state_id"],
            "status": "INCOMPLETE_NULL",
            "value": None,
            "verdict": None,
            "missing_conditions": missing,
            "fixture_only": False,
        }
    return {
        "state_id": definition["state_id"],
        "status": "SCHEMA_COMPLETE_NONPROMOTIONAL_NULL",
        "value": None,
        "verdict": None,
        "missing_conditions": [],
        "fixture_only": False,
    }


def _passing_evidence(
    definition: Mapping[str, Any], *, fixture_only: bool = True
) -> dict[str, Any]:
    predicate = definition["predicate"]
    evidence: dict[str, Any] = {}
    if fixture_only:
        evidence["_fixture_only"] = True
    evidence.update({key: True for key in predicate.get("true", [])})
    evidence.update(
        {
            key: hashlib.sha256(
                f"{definition['state_id']}:{key}".encode("utf-8")
            ).hexdigest()
            for key in predicate.get("non_null", [])
        }
    )
    evidence.update({key: 1 for key in predicate.get("positive", [])})
    evidence.update({key: 0 for key in predicate.get("zero", [])})
    evidence.update(
        {
            key: threshold
            for key, threshold in predicate.get("minimum", {}).items()
        }
    )
    return evidence


def _evaluate_fixture(
    definition: Mapping[str, Any],
    evidence: Mapping[str, Any],
    *,
    revoked: bool = False,
) -> dict[str, Any]:
    if evidence.get("_fixture_only") is not True:
        raise ValueError("schema fixture must be explicitly fixture-only")
    missing = _predicate_missing(definition, evidence)
    if revoked:
        status = "SYNTHETIC_REVOKED_NULL"
        missing = []
    elif missing:
        status = "SYNTHETIC_INCOMPLETE_NULL"
    else:
        status = "SYNTHETIC_SCHEMA_COMPLETE_NULL"
    return {
        "state_id": definition["state_id"],
        "status": status,
        "value": None,
        "verdict": None,
        "missing_conditions": missing,
        "fixture_only": True,
    }


def _predicate_fixtures(
    definitions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    fixtures: list[dict[str, Any]] = []
    for definition in definitions:
        passing = _passing_evidence(definition)
        required = [
            *definition["predicate"].get("true", []),
            *definition["predicate"].get("non_null", []),
            *definition["predicate"].get("positive", []),
            *definition["predicate"].get("zero", []),
            *definition["predicate"].get("minimum", {}).keys(),
        ]
        failing = copy.deepcopy(passing)
        first = required[0]
        if first in definition["predicate"].get("zero", []):
            failing[first] = 1
        elif first in definition["predicate"].get("positive", []):
            failing[first] = 0
        elif first in definition["predicate"].get("non_null", []):
            failing[first] = None
        elif first in definition["predicate"].get("minimum", {}):
            failing[first] = (
                definition["predicate"]["minimum"][first] - 1
            )
        else:
            failing[first] = False
        fixtures.extend(
            [
                {
                    "fixture_id": f"FIX-{definition['state_id']}-PASS",
                    "state_id": definition["state_id"],
                    "kind": "SYNTHETIC_SCHEMA_PASS_NOT_SCIENTIFIC_RESULT",
                    "evidence": passing,
                    "outcome": _evaluate_fixture(definition, passing),
                },
                {
                    "fixture_id": f"FIX-{definition['state_id']}-INCOMPLETE",
                    "state_id": definition["state_id"],
                    "kind": "SYNTHETIC_SINGLE_PREDICATE_FAILURE",
                    "evidence": failing,
                    "outcome": _evaluate_fixture(definition, failing),
                },
                {
                    "fixture_id": f"FIX-{definition['state_id']}-REVOKED",
                    "state_id": definition["state_id"],
                    "kind": "SYNTHETIC_REVOCATION_PROPAGATION",
                    "evidence": passing,
                    "outcome": _evaluate_fixture(
                        definition, passing, revoked=True
                    ),
                },
            ]
        )
    return fixtures


def _revocation_contract(
    definitions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "mode": "FAIL_CLOSED_APPEND_ONLY",
        "on_revocation": {
            "value": None,
            "verdict": None,
            "status": "REVOKED_NULL",
            "prior_evidence": "RETAIN_BY_HASH_IN_APPEND_ONLY_LEDGER",
        },
        "global_triggers": [
            "parent report/config/implementation/analysis binding drift",
            "task signature, denominator, observation, action, frontend, precision, compute or missingness contract drift",
            "formal reselection, hidden best-of-N, omitted timeout/failure/control cost or postselection",
            "external cutoff/query/dedup/ledger/eligibility or credible comparator change",
            "board, bitstream, source, clock or latency-boundary change",
            "QPU, real-GKP, physical baseline or raw-measurement provenance change",
            "cross-lane, estimate-as-measured, project-native-as-official or null-as-zero transfer"
        ],
        "prerequisite_edges": [
            {"narrower": narrower, "broader": broader}
            for narrower, broader in PREREQUISITE_EDGES
        ],
        "propagation": "REVOKING_A_NARROWER_PREREQUISITE_REVOKES_THE_BROADER_STATE_TO_NULL_NOT_NO_GO",
        "current_ledger": [],
        "current_ledger_anchor": revocation_ledger_anchor([]),
        "ledger_schema_version": REVOCATION_LEDGER_SCHEMA_VERSION,
        "genesis_sha256": REVOCATION_LEDGER_GENESIS_SHA256,
        "trusted_anchor_rule": "EVERY_STORED_LEDGER_OR_APPEND_MUST_BE_VERIFIED_AGAINST_AN_EXTERNALLY_PINNED_PRIOR_SNAPSHOT_ANCHOR",
        "amendment_on_external_cutoff_or_ledger_change": "NEW_VERSION_AND_REAUDIT_REQUIRED",
        "executor": "apply_revocations",
        "ledger_verifier": "verify_revocation_ledger",
    }


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(
        r"[0-9a-f]{64}", value
    ) is not None


def _parse_utc_timestamp(value: Any) -> datetime:
    if not isinstance(value, str):
        raise ValueError("revocation timestamp must be a string")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("revocation timestamp must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def revocation_ledger_anchor(
    ledger: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return the externally pinnable identity of one immutable snapshot."""

    rows = copy.deepcopy(list(ledger))
    head_sha256 = (
        rows[-1].get("ledger_entry_sha256")
        if rows
        else REVOCATION_LEDGER_GENESIS_SHA256
    )
    if not _is_sha256(head_sha256):
        raise ValueError("revocation ledger head is not a SHA-256")
    return {
        "schema_version": REVOCATION_LEDGER_SCHEMA_VERSION,
        "entry_count": len(rows),
        "head_sha256": head_sha256,
        "ledger_sha256": _canonical_sha256(rows),
    }


def _valid_revocation_anchor(anchor: Mapping[str, Any]) -> bool:
    required = {
        "schema_version",
        "entry_count",
        "head_sha256",
        "ledger_sha256",
    }
    return (
        isinstance(anchor, Mapping)
        and set(anchor) == required
        and anchor["schema_version"] == REVOCATION_LEDGER_SCHEMA_VERSION
        and isinstance(anchor["entry_count"], int)
        and not isinstance(anchor["entry_count"], bool)
        and anchor["entry_count"] >= 0
        and _is_sha256(anchor["head_sha256"])
        and _is_sha256(anchor["ledger_sha256"])
    )


def verify_revocation_ledger(
    ledger: Sequence[Mapping[str, Any]],
    *,
    trusted_anchor: Mapping[str, Any],
    expected_prefix: Sequence[Mapping[str, Any]] | None = None,
    allow_appends: bool = False,
) -> bool:
    """Verify exact closure, hash chain and an externally pinned snapshot.

    ``trusted_anchor`` is deliberately mandatory.  In full-snapshot mode it
    must bind the complete ledger.  In append mode it must bind the immutable
    ``expected_prefix`` supplied by the caller; the candidate ledger may only
    extend that prefix.
    """

    if not _valid_revocation_anchor(trusted_anchor):
        return False
    rows = copy.deepcopy(list(ledger))
    prefix = (
        None
        if expected_prefix is None
        else copy.deepcopy(list(expected_prefix))
    )
    if allow_appends:
        if (
            prefix is None
            or rows[: len(prefix)] != prefix
            or trusted_anchor["entry_count"] != len(prefix)
        ):
            return False
        try:
            if revocation_ledger_anchor(prefix) != dict(trusted_anchor):
                return False
        except (TypeError, ValueError):
            return False
    elif (
        prefix is not None
        or trusted_anchor["entry_count"] != len(rows)
    ):
        return False
    event_ids: set[str] = set()
    previous_time: datetime | None = None
    previous_entry_sha256 = REVOCATION_LEDGER_GENESIS_SHA256
    required = {
        "sequence_number",
        "previous_entry_sha256",
        "event_id",
        "revoked_state_id",
        "observed_at_utc",
        "prior_evidence_sha256",
        "reason_code",
        "prior_state_semantic_sha256",
        "propagated_states",
        "ledger_entry_sha256",
    }
    try:
        for sequence_number, row in enumerate(rows, start=1):
            if set(row) != required:
                return False
            if (
                isinstance(row["sequence_number"], bool)
                or row["sequence_number"] != sequence_number
                or row["previous_entry_sha256"]
                != previous_entry_sha256
            ):
                return False
            event_id = row["event_id"]
            if (
                not isinstance(event_id, str)
                or not event_id
                or event_id in event_ids
            ):
                return False
            event_ids.add(event_id)
            observed = _parse_utc_timestamp(row["observed_at_utc"])
            if previous_time is not None and observed <= previous_time:
                return False
            previous_time = observed
            if (
                row["revoked_state_id"] not in STATE_IDS
                or not _is_sha256(row["prior_evidence_sha256"])
                or not _is_sha256(row["prior_state_semantic_sha256"])
                or not isinstance(row["reason_code"], str)
                or not row["reason_code"]
                or not isinstance(row["propagated_states"], list)
                or row["propagated_states"]
                != sorted(
                    _revocation_closure(row["revoked_state_id"])
                )
            ):
                return False
            payload = {
                key: copy.deepcopy(value)
                for key, value in row.items()
                if key != "ledger_entry_sha256"
            }
            if row["ledger_entry_sha256"] != _canonical_sha256(payload):
                return False
            previous_entry_sha256 = row["ledger_entry_sha256"]
    except (TypeError, ValueError):
        return False
    try:
        if not allow_appends and revocation_ledger_anchor(rows) != dict(
            trusted_anchor
        ):
            return False
    except (TypeError, ValueError):
        return False
    return True


def _revocation_closure(state_id: str) -> set[str]:
    affected = {state_id}
    changed = True
    while changed:
        changed = False
        for narrower, broader in PREREQUISITE_EDGES:
            if narrower in affected and broader not in affected:
                affected.add(broader)
                changed = True
    return affected


def apply_revocations(
    states: Sequence[Mapping[str, Any]],
    *,
    prior_ledger: Sequence[Mapping[str, Any]],
    trusted_prior_anchor: Mapping[str, Any],
    new_events: Sequence[Mapping[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    """Append verified revocations and propagate them to broader states."""

    if not verify_revocation_ledger(
        prior_ledger, trusted_anchor=trusted_prior_anchor
    ):
        raise ValueError("prior revocation ledger or trusted anchor is invalid")
    result_states = copy.deepcopy(list(states))
    by_state = {row["state_id"]: row for row in result_states}
    if set(by_state) != set(STATE_IDS) or len(by_state) != len(
        result_states
    ):
        raise ValueError("revocation state inventory mismatch")
    ledger = copy.deepcopy(list(prior_ledger))
    seen_ids = {row["event_id"] for row in ledger}
    previous_time = (
        _parse_utc_timestamp(ledger[-1]["observed_at_utc"])
        if ledger
        else None
    )
    event_keys = {
        "event_id",
        "revoked_state_id",
        "observed_at_utc",
        "prior_evidence_sha256",
        "reason_code",
    }
    for raw_event in new_events:
        event = copy.deepcopy(dict(raw_event))
        if set(event) != event_keys:
            raise ValueError("revocation event schema mismatch")
        event_id = event["event_id"]
        state_id = event["revoked_state_id"]
        observed = _parse_utc_timestamp(event["observed_at_utc"])
        prior_hash = event["prior_evidence_sha256"]
        if (
            not isinstance(event_id, str)
            or not event_id
            or event_id in seen_ids
            or state_id not in by_state
            or not _is_sha256(prior_hash)
            or not isinstance(event["reason_code"], str)
            or not event["reason_code"]
            or (previous_time is not None and observed <= previous_time)
        ):
            raise ValueError("invalid or non-append-only revocation event")
        if prior_hash not in by_state[state_id]["evidence_refs"]:
            raise ValueError("revocation prior evidence hash is not bound")
        seen_ids.add(event_id)
        previous_time = observed
        affected = _revocation_closure(state_id)
        prior_state_semantic_sha256 = _canonical_sha256(
            by_state[state_id]
        )
        for affected_id in affected:
            row = by_state[affected_id]
            row["status"] = "REVOKED_NULL"
            row["value"] = None
            row["verdict"] = None
            row["numeric_metrics"] = None
            row["rank"] = None
            row["vote"] = None
            row["opened"] = False
            row["revocation"] = {
                "status": "REVOKED",
                "reason": (
                    event["reason_code"]
                    if affected_id == state_id
                    else f"PROPAGATED_FROM:{state_id}"
                ),
                "evidence_hashes": sorted(
                    set(
                        [
                            *row["revocation"].get(
                                "evidence_hashes", []
                            ),
                            prior_hash,
                        ]
                    )
                ),
            }
        ledger_payload = {
            "sequence_number": len(ledger) + 1,
            "previous_entry_sha256": (
                ledger[-1]["ledger_entry_sha256"]
                if ledger
                else REVOCATION_LEDGER_GENESIS_SHA256
            ),
            **event,
            "prior_state_semantic_sha256": prior_state_semantic_sha256,
            "propagated_states": sorted(affected),
        }
        ledger.append(
            {
                **ledger_payload,
                "ledger_entry_sha256": _canonical_sha256(
                    ledger_payload
                ),
            }
        )
    if not verify_revocation_ledger(
        ledger,
        trusted_anchor=trusted_prior_anchor,
        expected_prefix=prior_ledger,
        allow_appends=True,
    ):
        raise ValueError("new revocation ledger failed verification")
    new_anchor = revocation_ledger_anchor(ledger)
    if not verify_revocation_ledger(
        ledger, trusted_anchor=new_anchor
    ):
        raise ValueError("new revocation ledger snapshot anchor is invalid")
    return result_states, ledger, new_anchor


def _revocation_fixtures(
    current_states: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    fixtures: list[dict[str, Any]] = []
    for index, state_id in enumerate(
        (
            "GO_LER_REGISTERED_BEST",
            "GO_LIFETIME_PROJECT_NATIVE",
            "OFFICIAL_PUVIANI_EXACT",
            "GO_HIL_INTEGRATED",
        ),
        start=1,
    ):
        states = copy.deepcopy(list(current_states))
        prior_hash = hashlib.sha256(
            f"revocation-fixture:{state_id}".encode("utf-8")
        ).hexdigest()
        for row in states:
            row["status"] = "GO"
            row["value"] = row["state_id"]
            row["verdict"] = row["state_id"]
            row["opened"] = True
            row["evidence_grade"] = "SYNTHETIC_FIXTURE_ONLY"
            row["evidence_refs"] = [
                hashlib.sha256(
                    f"revocation-fixture:{row['state_id']}".encode(
                        "utf-8"
                    )
                ).hexdigest()
            ]
        event = {
            "event_id": f"REV-FIX-{index:02d}",
            "revoked_state_id": state_id,
            "observed_at_utc": f"2026-07-25T00:0{index}:00+00:00",
            "prior_evidence_sha256": prior_hash,
            "reason_code": "SYNTHETIC_REVOCATION_FIXTURE_ONLY",
        }
        genesis_anchor = revocation_ledger_anchor([])
        updated, ledger, ledger_anchor = apply_revocations(
            states,
            prior_ledger=[],
            trusted_prior_anchor=genesis_anchor,
            new_events=[event],
        )
        affected = sorted(_revocation_closure(state_id))
        by_state = {row["state_id"]: row for row in updated}
        fixtures.append(
            {
                "fixture_id": f"REV-FIXTURE-{state_id}",
                "kind": "SYNTHETIC_REVOCATION_EXECUTION_NOT_SCIENTIFIC_RESULT",
                "revoked_state_id": state_id,
                "expected_affected_states": affected,
                "actual_affected_states": sorted(
                    row["state_id"]
                    for row in updated
                    if row["status"] == "REVOKED_NULL"
                ),
                "all_affected_null": all(
                    by_state[affected_id]["value"] is None
                    and by_state[affected_id]["verdict"] is None
                    for affected_id in affected
                ),
                "prior_evidence_retained": all(
                    prior_hash
                    in by_state[affected_id]["revocation"][
                        "evidence_hashes"
                    ]
                    for affected_id in affected
                ),
                "ledger": ledger,
                "trusted_prior_anchor": genesis_anchor,
                "ledger_anchor": ledger_anchor,
            }
        )
    return fixtures


def _aggregation_contract() -> dict[str, Any]:
    return {
        "weighted_global_score": "PROHIBITED",
        "winner_count": "PROHIBITED",
        "paper_scope_implies_sota": False,
        "go_single_paper_implies_three_sota": False,
        "lane_verdicts_independent": True,
        "null_lane_rescue": "PROHIBITED",
    }


def _artifact_registry(config_path: Path) -> dict[str, Any]:
    return {
        "config": _binding(config_path),
        "implementation": _binding(IMPLEMENTATION),
    }


def _atomic_ids(report: Mapping[str, Any]) -> list[str]:
    return [
        *[row["state_id"] for row in report["state_definitions"]],
        *[row["migration_id"] for row in report["legacy_migration_table"]],
        *[row["transfer_id"] for row in report["forbidden_transfers"]],
        *[row["fixture_id"] for row in report["predicate_fixtures"]],
    ]


def _source_rows(report: Mapping[str, Any]) -> list[dict[str, str]]:
    payloads: list[tuple[str, str, Any]] = [
        ("analysis_field", field, value)
        for field, value in _analysis_payload(report).items()
    ]
    payloads.extend(
        [
        ("parent_summary", "PARENT-SUMMARY", report["parent_summary"]),
        (
            "external_contract",
            "EXTERNAL-SAME-TASK-CONTRACT",
            report["external_same_task_contract"],
        ),
        (
            "revocation_contract",
            "REVOCATION-CONTRACT",
            report["revocation_contract"],
        ),
        (
            "downstream_contract",
            "DOWNSTREAM-CONSUMPTION-CONTRACT",
            report["downstream_consumption_contract"],
        ),
        (
            "aggregation_contract",
            "AGGREGATION-CONTRACT",
            report["aggregation_contract"],
        ),
        (
            "asset_inventory",
            "CURRENT-ASSET-INVENTORY",
            report["current_asset_inventory"],
        ),
        ]
    )
    payloads.extend(
        ("state_definition", row["state_id"], row)
        for row in report["state_definitions"]
    )
    payloads.extend(
        ("current_state", row["state_id"], row)
        for row in report["current_states"]
    )
    payloads.extend(
        ("legacy_migration", row["migration_id"], row)
        for row in report["legacy_migration_table"]
    )
    payloads.extend(
        ("forbidden_transfer", row["transfer_id"], row)
        for row in report["forbidden_transfers"]
    )
    payloads.extend(
        ("predicate_fixture", row["fixture_id"], row)
        for row in report["predicate_fixtures"]
    )
    payloads.extend(
        ("revocation_fixture", row["fixture_id"], row)
        for row in report["revocation_fixtures"]
    )
    rows: list[dict[str, str]] = []
    for record_type, record_id, payload in payloads:
        canonical = _canonical_json(payload)
        rows.append(
            {
                "record_type": record_type,
                "record_id": record_id,
                "canonical_json": canonical,
                "canonical_sha256": hashlib.sha256(
                    canonical.encode("utf-8")
                ).hexdigest(),
            }
        )
    return rows


def _write_source_data(report: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "record_type",
                "record_id",
                "canonical_json",
                "canonical_sha256",
            ],
        )
        writer.writeheader()
        writer.writerows(_source_rows(report))
    temporary.replace(path)


def _csv_lossless(report: Mapping[str, Any], path: Path) -> bool:
    if not path.is_file():
        return False
    with path.open("r", encoding="utf-8", newline="") as handle:
        actual = list(csv.DictReader(handle))
    expected = _source_rows(report)
    analysis_rows = {
        row["record_id"]: json.loads(row["canonical_json"])
        for row in actual
        if row["record_type"] == "analysis_field"
    }
    return (
        actual == expected
        and len(actual) == len(
            {(row["record_type"], row["record_id"]) for row in actual}
        )
        and all(
            row["canonical_sha256"]
            == hashlib.sha256(
                row["canonical_json"].encode("utf-8")
            ).hexdigest()
            for row in actual
        )
        and analysis_rows == _analysis_payload(report)
    )


def _render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Phase 9 scoped claim amendment（T9.1.5）",
        "",
        "> 本文档是 parent-bound、pre-outcome 的 claim 协议 child seal，"
        "不是解码性能、外部 SOTA、真实板延迟或物理寿命结果。",
        "",
        "## 1. 不可变 parent",
        "",
        f"- T9.1.1 `analysis_sha256`：`{report['parent_summary']['v1']['analysis_sha256']}`；",
        f"- T9.1.4 `analysis_sha256`：`{report['parent_summary']['registry']['analysis_sha256']}`；",
        f"- 文献截止日：`{report['external_same_task_contract']['cutoff_inclusive']}`；",
        f"- 当前 external same-task eligible：`{report['external_same_task_contract']['current_external_same_task_eligible_count']}`。",
        "",
        "## 2. 当前 scoped states",
        "",
    ]
    for row in report["current_states"]:
        lines.append(
            f"- `{row['state_id']}`：`{row['status']}`，value/verdict=`null`；"
            f"原因 `{row['null_reason']}`。"
        )
    lines.extend(["", "## 3. state 证据与撤销边界", ""])
    for definition in report["state_definitions"]:
        predicate = definition["predicate"]
        requirements = sum(
            len(predicate.get(key, []))
            for key in ("true", "non_null", "positive", "zero")
        )
        lines.append(
            f"- `{definition['state_id']}`：scope=`{definition['evidence_scope']}`，"
            f"{requirements} 个原子 predicate，"
            f"{len(definition['revocation_triggers'])} 个 state-specific revocation triggers；"
            f"允许措辞：{definition['allowed_wording']}。"
        )
    lines.extend(["", "## 4. legacy migration（不自动迁移）", ""])
    for row in report["legacy_migration_table"]:
        destinations = ",".join(row["candidate_destinations"]) or "null"
        lines.append(
            f"- `{row['migration_id']}`：`{row['legacy_lane']}/{row['legacy_label']}`"
            f" → candidate `{destinations}`；auto=`false`；当前 mapped value=`null`。"
        )
    lines.extend(["", "## 5. 禁止证据迁移", ""])
    for row in report["forbidden_transfers"]:
        lines.append(
            f"- `{row['transfer_id']}`：`{row['from']}` ↛ `{row['to']}`"
            f"（`{row['rejection']}`）。"
        )
    lines.extend(
        [
            "",
            "## 6. nullable terminal",
            "",
            "- T9.1.2 缺 official assets 只保持 official exact / Puviani surpass 为 null；",
            "- T9.7.4 可为 `Done` 或 terminal `Blocked/null`，无板只保持 HIL states 为 null；",
            "- 无 Phase-8 QPU/real-GKP 证据只保持 physical lifetime 为 null；",
            "- 上述局部 null 均不得阻塞 algorithm-only、split 或诚实 NO-GO 决策。",
            "",
            "## 7. 反简化证据",
            "",
            f"- 9 个 state 各有 complete/incomplete/revoked 三类 synthetic schema fixtures，"
            f"共 `{len(report['predicate_fixtures'])}` 条；value/verdict 均为 null，"
            "generic evaluator 也永不签发 GO；",
            f"- `{len(report['revocation_fixtures'])}` 条 executable revocation fixtures"
            "验证 exact prerequisite closure、sequence/previous-entry hash chain、"
            "externally pinned snapshot anchor 和证据 hash 保留；",
            f"- legacy migration `{len(report['legacy_migration_table'])}` 条，"
            f"forbidden transfers `{len(report['forbidden_transfers'])}` 条；",
            f"- gates `{report['gate_summary']['passed']}/{report['gate_summary']['total']}`，"
            f"mutations `{report['semantic_mutation_audit']['detected']}/"
            f"{report['semantic_mutation_audit']['count']}`。",
            "",
            "## 8. 复现",
            "",
            "```powershell",
            "python -m cnn_fpga.benchmark.phase9_scoped_claim_amendment",
            "python -m cnn_fpga.benchmark.phase9_scoped_claim_amendment --verify",
            "python -m pytest -q tests/test_phase9_scoped_claim_amendment.py",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def _definition(
    report: Mapping[str, Any], state_id: str
) -> Mapping[str, Any]:
    return next(
        row
        for row in report["state_definitions"]
        if row["state_id"] == state_id
    )


def _migration(
    report: Mapping[str, Any], lane: str, label: str
) -> Mapping[str, Any]:
    return next(
        row
        for row in report["legacy_migration_table"]
        if row["legacy_lane"] == lane and row["legacy_label"] == label
    )


def _requirements(definition: Mapping[str, Any]) -> set[str]:
    predicate = definition["predicate"]
    return set().union(
        predicate.get("true", []),
        predicate.get("non_null", []),
        predicate.get("positive", []),
        predicate.get("zero", []),
        predicate.get("minimum", {}).keys(),
    )


def _valid_mutation_placeholder() -> dict[str, Any]:
    return {
        "count": len(GATE_IDS),
        "detected": len(GATE_IDS),
        "all_detected": True,
        "one_per_gate": True,
        "records": [
            {
                "mutation_id": f"M{index:02d}_placeholder",
                "target_gate": gate,
                "detected": True,
            }
            for index, gate in enumerate(GATE_IDS, start=1)
        ],
    }


def evaluate_gates(
    report: Mapping[str, Any],
    *,
    config_path: Path = DEFAULT_CONFIG,
    check_live_files: bool = True,
    expected_parent_summary: Mapping[str, Any] | None = None,
    expected_artifact_registry: Mapping[str, Any] | None = None,
) -> dict[str, bool]:
    config = _load(config_path)
    definitions = report["state_definitions"]
    current_states = report["current_states"]
    migrations = report["legacy_migration_table"]
    transfers = report["forbidden_transfers"]
    parent_summary = (
        copy.deepcopy(dict(expected_parent_summary))
        if expected_parent_summary is not None
        else _parent_summary(config, verify_live=check_live_files)
    )
    artifacts = (
        copy.deepcopy(dict(expected_artifact_registry))
        if expected_artifact_registry is not None
        else _artifact_registry(config_path)
    )

    expected_current = _current_states(config["state_definitions"])
    expected_assets = _asset_inventory()
    expected_legacy = {
        (row["lane"], row["output"])
        for row in parent_summary["v1"]["legacy_lane_outputs"]
    }
    migration_keys = {
        (row["legacy_lane"], row["legacy_label"]) for row in migrations
    }
    by_state = {row["state_id"]: row for row in definitions}

    registered_ler_required = {
        "formal_phase9_complete",
        "all_mandatory_baselines_complete",
        "every_eligible_matched_baseline_beaten_with_registered_thresholds",
        "same_task_signature",
        "simultaneous_ci_gate_passed",
        "both_independent_backends_passed",
        "zero_postselection",
        "complete_missingness_ledger",
        "independent_recompute_passed",
        "formal_manifest_sha256",
        "source_data_sha256",
        "analysis_sha256",
    }
    external_common = {
        "literature_cutoff_bound",
        "complete_deduplicated_comparator_ledger",
        "all_credible_in_window_records_screened",
        "all_external_same_task_results_complete",
        "every_external_same_task_comparator_beaten_with_simultaneous_ci",
        "independent_external_audit_passed",
        "external_same_task_eligible_count",
        "unresolved_eligible_stronger_comparator_count",
        "comparator_ledger_sha256",
        "independent_audit_sha256",
    }

    def current_schema_ok() -> bool:
        expected_keys = {
            "state_id",
            "lane",
            "evidence_scope",
            "status",
            "value",
            "verdict",
            "evidence_grade",
            "evidence_refs",
            "numeric_metrics",
            "rank",
            "vote",
            "opened",
            "null_reason",
            "revocation",
        }
        revocation_keys = {"status", "reason", "evidence_hashes"}
        return all(
            set(row) == expected_keys
            and set(row["revocation"]) == revocation_keys
            and row["status"]
            in {
                "NOT_EVALUATED_NULL",
                "MISSING_EXTERNAL_ASSET_NULL",
                "MISSING_EXTERNAL_COMPARATOR_NULL",
                "MISSING_BOARD_NULL",
                "MISSING_QPU_NULL",
            }
            for row in current_states
        )

    def fixtures_ok() -> bool:
        fixtures = report["predicate_fixtures"]
        if len(fixtures) != len(definitions) * 3:
            return False
        for definition in definitions:
            rows = [
                row
                for row in fixtures
                if row["state_id"] == definition["state_id"]
            ]
            if len(rows) != 3:
                return False
            outcomes = {row["outcome"]["status"] for row in rows}
            if outcomes != {
                "SYNTHETIC_SCHEMA_COMPLETE_NULL",
                "SYNTHETIC_INCOMPLETE_NULL",
                "SYNTHETIC_REVOKED_NULL",
            }:
                return False
            if not all(
                row["outcome"]["fixture_only"] is True
                and row["outcome"]["value"] is None
                and row["outcome"]["verdict"] is None
                and evaluate_claim_candidate(
                    definition, row["evidence"]
                )["status"]
                == "FIXTURE_ONLY_NULL"
                for row in rows
            ):
                return False
        return True

    def mutation_audit_ok() -> bool:
        audit = report["semantic_mutation_audit"]
        records = audit["records"]
        return (
            audit["count"] == len(GATE_IDS)
            and audit["detected"] == len(GATE_IDS)
            and audit["all_detected"] is True
            and audit["one_per_gate"] is True
            and len(records) == len(GATE_IDS)
            and len({row["mutation_id"] for row in records})
            == len(GATE_IDS)
            and [row["target_gate"] for row in records]
            == list(GATE_IDS)
            and all(row["detected"] is True for row in records)
        )

    def outputs_ok() -> bool:
        basic = (
            report["source_data"]["rows"] == len(_source_rows(report))
            and report["source_data"]["path"]
            == _relative(DEFAULT_SOURCE_DATA)
            and report["markdown"]["path"] == _relative(DEFAULT_MARKDOWN)
        )
        if not basic or not check_live_files:
            return basic
        source_path = ROOT / report["source_data"]["path"]
        markdown_path = ROOT / report["markdown"]["path"]
        markdown_text = markdown_path.read_text(encoding="utf-8")
        return (
            _binding_live(report["source_data"])
            and _binding_live(report["markdown"])
            and _csv_lossless(report, source_path)
            and markdown_text == _render_markdown(report)
            and all(
                f"`{identifier}`" in markdown_text
                for identifier in _atomic_ids(report)
                if not identifier.startswith("FIX-")
            )
        )

    gates = {
        "G01_identity_and_parent_bound_preoutcome_status": _safe(
            lambda: report["task_id"] == TASK_ID
            and report["schema_version"] == REPORT_SCHEMA_VERSION
            and report["config_schema_version"] == CONFIG_SCHEMA_VERSION
            and report["protocol_id"] == PROTOCOL_ID
            and report["amendment_status"]
            == "SEALED_PARENT_BOUND_PRE_OUTCOME_V2"
            and report["frozen_at"] == config["frozen_at"]
        ),
        "G02_t9_1_1_parent_is_live_verified_and_byte_immutable": _safe(
            lambda: report["parent_contract"] == config["parent_contract"]
            and report["parent_summary"]["v1"] == parent_summary["v1"]
            and report["parent_summary"]["v1"]["analysis_sha256"]
            == "c88110375c358794339e72d672e4624871425fe480e5da091ddd1d6595255e18"
            and all(
                _binding_live(binding)
                for key, binding in config["parent_contract"].items()
                if isinstance(binding, dict) and "path" in binding
            )
            if check_live_files
            else report["parent_contract"] == config["parent_contract"]
            and report["parent_summary"]["v1"] == parent_summary["v1"]
            and report["parent_summary"]["v1"]["analysis_sha256"]
            == "c88110375c358794339e72d672e4624871425fe480e5da091ddd1d6595255e18"
        ),
        "G03_parent_legacy_outputs_are_dynamically_and_completely_covered": _safe(
            lambda: expected_legacy == migration_keys
            and len(expected_legacy) == len(migrations) == 29
            and len(migration_keys) == 29
        ),
        "G04_t9_1_4_registry_is_live_verified_and_exactly_bound": _safe(
            lambda: report["registry_parent_contract"]
            == config["registry_parent_contract"]
            and report["parent_summary"]["registry"]
            == parent_summary["registry"]
            and report["parent_summary"]["registry"]["analysis_sha256"]
            == "d6c5ac4fd9587854cd6fec7d390c1fd2ddd5300bbe069ff2db1e16950aa21b7d"
            and all(
                _binding_live(binding)
                for key, binding in config[
                    "registry_parent_contract"
                ].items()
                if isinstance(binding, dict) and "path" in binding
            )
            if check_live_files
            else report["registry_parent_contract"]
            == config["registry_parent_contract"]
            and report["parent_summary"]["registry"]
            == parent_summary["registry"]
            and report["parent_summary"]["registry"]["analysis_sha256"]
            == "d6c5ac4fd9587854cd6fec7d390c1fd2ddd5300bbe069ff2db1e16950aa21b7d"
        ),
        "G05_literature_cutoff_dedup_ledger_and_current_eligibility_are_bound": _safe(
            lambda: report["external_same_task_contract"]
            == config["external_same_task_contract"]
            and report["external_same_task_contract"]["cutoff_inclusive"]
            == parent_summary["registry"]["literature_cutoff_inclusive"]
            and report["external_same_task_contract"][
                "canonical_work_count"
            ]
            == parent_summary["registry"]["canonical_work_count"]
            == 12
            and report["external_same_task_contract"][
                "current_external_same_task_eligible_count"
            ]
            == parent_summary["registry"][
                "external_same_task_eligible_count"
            ]
            == 0
        ),
        "G06_config_implementation_and_parent_artifacts_are_live": _safe(
            lambda: report["artifact_registry"] == artifacts
            and (
                not check_live_files
                or all(_binding_live(row) for row in artifacts.values())
            )
        ),
        "G07_downstream_consumers_validate_then_reconstruct_then_compare": _safe(
            lambda: report["downstream_consumption_contract"]
            == config["downstream_consumption_contract"]
            and report["downstream_consumption_contract"][
                "direct_consumers"
            ]
            == ["T9.4.5", "T9.6.1", "T9.6.5", "T9.8.1"]
            and report["downstream_consumption_contract"][
                "required_checks"
            ][:5]
            == [
                "verify child report live",
                "reconstruct child analysis_sha256",
                "match immutable T9.1.1 parent analysis_sha256",
                "match T9.1.4 registry analysis_sha256 and literature cutoff",
                "match canonical T9.1.5 release-pin analysis/config/implementation hashes",
            ]
            and report["downstream_consumption_contract"][
                "release_pin_path"
            ]
            == _relative(DEFAULT_RELEASE_PIN)
            and "report-selected config/output paths"
            in report["downstream_consumption_contract"][
                "release_pin_rule"
            ]
            and "Done or terminal Blocked/null"
            in report["downstream_consumption_contract"][
                "t9_7_4_nullable_terminal_rule"
            ]
        ),
        "G08_scoped_state_ids_are_exact_unique_and_lane_bound": _safe(
            lambda: definitions == config["state_definitions"]
            and tuple(row["state_id"] for row in definitions)
            == STATE_IDS
            and len(definitions)
            == len({row["state_id"] for row in definitions})
            == 9
            and all(row["lane"] for row in definitions)
        ),
        "G09_current_state_objects_have_a_strict_typed_null_schema": _safe(
            lambda: current_schema_ok()
            and report["predicate_type_contract"]
            == config["predicate_type_contract"]
            and report["predicate_type_contract"][
                "generic_mapping_evaluator_may_open_claim"
            ]
            is False
        ),
        "G10_all_current_claim_performance_rank_and_vote_values_are_null": _safe(
            lambda: current_states == expected_current
            and report["current_claim_state"]
            == config["current_claim_state"]
            and all(value is None for value in report["current_claim_state"].values())
            and report["performance_state"] == config["performance_state"]
            and all(
                value is None
                for key, value in report["performance_state"].items()
                if key != "protocol_only"
            )
            and report["performance_state"]["protocol_only"] is True
        ),
        "G11_official_board_external_speed_and_qpu_inventory_is_typed_null": _safe(
            lambda: report["current_asset_inventory"] == expected_assets
            and tuple(report["current_asset_inventory"])
            == ASSET_INVENTORY_KEYS
            and all(
                row["value"] is None
                and row["verdict"] is None
                and row["evidence_grade"] is None
                and row["evidence_refs"] == []
                for row in report["current_asset_inventory"].values()
            )
        ),
        "G12_protocol_pass_does_not_open_any_performance_claim": _safe(
            lambda: report["verdict"] in {None, VERDICT}
            and all(row["opened"] is False for row in current_states)
            and all(row["value"] is None for row in current_states)
            and report["performance_state"]["protocol_only"] is True
            and fixtures_ok()
            and all(
                (
                    outcome := evaluate_claim_candidate(
                        definition,
                        _passing_evidence(
                            definition, fixture_only=False
                        ),
                    )
                )["status"]
                == "SCHEMA_COMPLETE_NONPROMOTIONAL_NULL"
                and outcome["value"] is None
                and outcome["verdict"] is None
                for definition in definitions
            )
        ),
        "G13_migration_table_has_exactly_29_lane_qualified_parent_outputs": _safe(
            lambda: migrations == config["legacy_migration_table"]
            and len(migrations) == 29
            and len({row["migration_id"] for row in migrations}) == 29
            and migration_keys == expected_legacy
        ),
        "G14_no_legacy_output_automatically_promotes_a_v2_state": _safe(
            lambda: all(
                row["automatic_mapping"] is False
                and row["current_mapped_value"] is None
                for row in migrations
            )
        ),
        "G15_legacy_ler_go_is_registered_candidate_only": _safe(
            lambda: _migration(
                report, "ROUND_LER_SINGLE_MODE", "GO_LER_SOTA"
            )["candidate_destinations"]
            == ["GO_LER_REGISTERED_BEST"]
            and "GO_LER_EXTERNAL_SOTA"
            in _migration(
                report, "ROUND_LER_SINGLE_MODE", "GO_LER_SOTA"
            )["prohibited_destinations"]
        ),
        "G16_legacy_lifetime_go_is_project_native_candidate_only": _safe(
            lambda: _migration(
                report, "SIX_STATE_LOGICAL_LIFETIME", "GO_LIFETIME"
            )["candidate_destinations"]
            == ["GO_LIFETIME_PROJECT_NATIVE"]
            and {
                "GO_LIFETIME_EXTERNAL_SOTA",
                "GO_PHYSICAL_LIFETIME",
                "OFFICIAL_PUVIANI_EXACT",
                "GO_PUVIANI_NMF_SURPASS",
            }
            <= set(
                _migration(
                    report,
                    "SIX_STATE_LOGICAL_LIFETIME",
                    "GO_LIFETIME",
                )["prohibited_destinations"]
            )
        ),
        "G17_legacy_hil_go_is_integrated_candidate_only": _safe(
            lambda: _migration(
                report, "RAW_IQ_DIGITAL_HIL", "GO_HIL_SPEED"
            )["candidate_destinations"]
            == ["GO_HIL_INTEGRATED"]
            and _migration(
                report, "RAW_IQ_DIGITAL_HIL", "GO_HIL_SPEED"
            )["prohibited_destinations"]
            == ["GO_HIL_EXTERNAL_SPEED"]
        ),
        "G18_null_incomplete_negative_and_supporting_outputs_open_no_go_state": _safe(
            lambda: all(
                row["candidate_destinations"] == []
                for row in migrations
                if row["legacy_label"]
                not in {"GO_LER_SOTA", "GO_LIFETIME", "GO_HIL_SPEED"}
            )
        ),
        "G19_registered_ler_predicate_requires_all_matched_formal_evidence": _safe(
            lambda: registered_ler_required
            <= _requirements(by_state["GO_LER_REGISTERED_BEST"])
        ),
        "G20_external_ler_requires_registered_go_complete_ledger_and_audit": _safe(
            lambda: external_common
            | {
                "go_ler_registered_best",
                "external_evaluation_manifest_sha256",
            }
            <= _requirements(by_state["GO_LER_EXTERNAL_SOTA"])
        ),
        "G21_project_native_lifetime_requires_six_state_cost_and_two_backends": _safe(
            lambda: {
                "lifetime_power_and_estimand_frozen_preoutcome",
                "six_state_full_denominator",
                "no_postselection",
                "control_reset_fallback_cost_included",
                "survival_and_logical_channel_fit_qualified",
                "minimum_over_six_states_gain_gate_passed",
                "simultaneous_ci_gate_passed",
                "both_independent_backends_passed",
                "independent_recompute_passed",
            }
            <= _requirements(
                by_state["GO_LIFETIME_PROJECT_NATIVE"]
            )
        ),
        "G22_external_lifetime_requires_project_native_and_external_evidence": _safe(
            lambda: external_common
            | {
                "go_lifetime_project_native",
                "external_lifetime_manifest_sha256",
            }
            <= _requirements(
                by_state["GO_LIFETIME_EXTERNAL_SOTA"]
            )
        ),
        "G23_physical_lifetime_requires_qpu_real_gkp_raw_measurements": _safe(
            lambda: {
                "qpu_measured",
                "real_gkp_state",
                "physical_break_even_definition_frozen",
                "physical_time_axis_measured",
                "matched_idle_memory_baseline_measured",
                "physical_break_even_gate_passed",
                "six_state_full_denominator",
                "no_postselection",
                "control_reset_fallback_cost_included",
                "measurement_calibration_and_uncertainty_complete",
                "independent_physical_audit_passed",
                "independent_qpu_run_count",
                "measured_physical_event_count",
                "qpu_run_manifest_sha256",
                "raw_measurement_data_sha256",
            }
            <= _requirements(by_state["GO_PHYSICAL_LIFETIME"])
        ),
        "G24_integrated_hil_requires_real_board_full_chain_measurement": _safe(
            lambda: {
                "real_board_present",
                "board_recorded_iq_replay_qualified",
                "board_live_raw_iq_hil_qualified",
                "recorded_replay_to_discriminator_measured",
                "live_adc_to_discriminator_measured",
                "discriminator_to_action_measured",
                "action_to_trigger_measured",
                "transport_and_backpressure_measured",
                "bitstream_platform_clock_provenance_complete",
                "deadline_missingness_failure_ledger_complete",
                "independent_hil_audit_passed",
                "measured_event_count",
                "independent_build_seed_count",
                "measured_transaction_count",
                "bit_mismatch_count",
                "undefined_action_count",
                "silent_overflow_count",
                "deadline_miss_count",
                "board_manifest_sha256",
                "bitstream_sha256",
                "raw_trace_sha256",
                "latency_source_data_sha256",
            }
            <= _requirements(by_state["GO_HIL_INTEGRATED"])
            and by_state["GO_HIL_INTEGRATED"]["predicate"][
                "minimum"
            ]
            == {
                "independent_build_seed_count": 3,
                "measured_transaction_count": 1_000_000,
            }
            and by_state["GO_HIL_INTEGRATED"]["predicate"]["zero"]
            == [
                "bit_mismatch_count",
                "undefined_action_count",
                "silent_overflow_count",
                "deadline_miss_count",
            ]
        ),
        "G25_external_speed_requires_integrated_hil_and_same_boundary_comparators": _safe(
            lambda: {
                "go_hil_integrated",
                "literature_cutoff_bound",
                "complete_external_hardware_comparator_ledger",
                "same_code_problem_input_precision_and_latency_boundary",
                "all_external_same_task_results_complete",
                "resource_power_deadline_context_complete",
                "paired_speed_superiority_simultaneous_ci_passed",
                "independent_external_audit_passed",
                "external_same_boundary_eligible_count",
                "unresolved_eligible_stronger_comparator_count",
                "hardware_comparator_ledger_sha256",
                "external_speed_manifest_sha256",
                "independent_audit_sha256",
            }
            <= _requirements(by_state["GO_HIL_EXTERNAL_SPEED"])
        ),
        "G26_official_puviani_exact_and_surpass_are_separate_and_locally_blocked": _safe(
            lambda: {
                "official_checkpoint_available",
                "official_twenty_agent_seed_set_available",
                "official_selection_ledger_available",
                "official_six_state_evaluator_available",
                "official_asset_manifest_sha256",
            }
            <= _requirements(by_state["OFFICIAL_PUVIANI_EXACT"])
            and {
                "official_puviani_exact",
                "go_lifetime_project_native",
                "same_task_signature",
                "paired_comparison_manifest_sha256",
            }
            <= _requirements(by_state["GO_PUVIANI_NMF_SURPASS"])
            and next(
                row
                for row in current_states
                if row["state_id"] == "OFFICIAL_PUVIANI_EXACT"
            )["status"]
            == "MISSING_EXTERNAL_ASSET_NULL"
            and report["downstream_consumption_contract"][
                "t9_1_2_local_block_rule"
            ].endswith(
                "OFFICIAL_PUVIANI_EXACT and GO_PUVIANI_NMF_SURPASS"
            )
        ),
        "G27_external_eligibility_inherits_all_nine_t9_1_4_signature_checks": _safe(
            lambda: report["external_same_task_contract"][
                "required_signature_fields"
            ]
            == [
                "input",
                "history",
                "action",
                "physics",
                "online_timing",
                "postselection",
                "denominator",
                "metric",
                "compute",
            ]
            and len(
                report["external_same_task_contract"][
                    "required_signature_fields"
                ]
            )
            == 9
        ),
        "G28_cutoff_or_ledger_change_requires_reseal_and_external_revocation": _safe(
            lambda: report["revocation_contract"][
                "amendment_on_external_cutoff_or_ledger_change"
            ]
            == "NEW_VERSION_AND_REAUDIT_REQUIRED"
            and any(
                "external cutoff/query/dedup/ledger/eligibility"
                in trigger
                for trigger in report["revocation_contract"][
                    "global_triggers"
                ]
            )
            and report["external_same_task_contract"][
                "registry_best_auto_promotes_external_sota"
            ]
            is False
        ),
        "G29_every_state_has_fail_closed_append_only_revocation_semantics": _safe(
            lambda: report["revocation_contract"]
            == _revocation_contract(config["state_definitions"])
            and report["revocation_contract"]["mode"]
            == "FAIL_CLOSED_APPEND_ONLY"
            and report["revocation_contract"]["on_revocation"]
            == {
                "value": None,
                "verdict": None,
                "status": "REVOKED_NULL",
                "prior_evidence": "RETAIN_BY_HASH_IN_APPEND_ONLY_LEDGER",
            }
            and all(row["revocation_triggers"] for row in definitions)
        ),
        "G30_revocation_propagates_to_broader_states_as_null_not_no_go": _safe(
            lambda: {
                (row["narrower"], row["broader"])
                for row in report["revocation_contract"][
                    "prerequisite_edges"
                ]
            }
            == set(PREREQUISITE_EDGES)
            and report["revocation_contract"]["propagation"]
            == "REVOKING_A_NARROWER_PREREQUISITE_REVOKES_THE_BROADER_STATE_TO_NULL_NOT_NO_GO"
            and report["revocation_fixtures"]
            == _revocation_fixtures(expected_current)
            and len(report["revocation_fixtures"]) == 4
            and all(
                row["actual_affected_states"]
                == row["expected_affected_states"]
                and row["all_affected_null"] is True
                and row["prior_evidence_retained"] is True
                and verify_revocation_ledger(
                    row["ledger"],
                    trusted_anchor=row["ledger_anchor"],
                )
                for row in report["revocation_fixtures"]
            )
        ),
        "G31_forbidden_transfer_matrix_is_exact_and_complete": _safe(
            lambda: transfers == config["forbidden_transfers"]
            and len(transfers)
            == len({row["transfer_id"] for row in transfers})
            == 17
        ),
        "G32_preboard_six_cycle_recorded_live_and_external_hil_levels_do_not_mix": _safe(
            lambda: {
                (
                    row["from"],
                    row["to"],
                    row["rejection"],
                )
                for row in transfers
            }
            >= {
                (
                    "CXXRTL_OR_SYNTHESIS",
                    "GO_HIL_INTEGRATED",
                    "ESTIMATE_IS_NOT_REAL_BOARD_HIL",
                ),
                (
                    "BOARD_RECORDED_IQ_REPLAY",
                    "BOARD_LIVE_RAW_IQ_HIL",
                    "RECORDED_REPLAY_IS_NOT_LIVE_ADC_RAW_IQ",
                ),
                (
                    "DISCRIMINATOR_OUT_TO_ACTION_6_CYCLES",
                    "ADC_OR_REPLAY_TO_TRIGGER_LATENCY",
                    "LATENCY_BOUNDARY_MISMATCH",
                ),
                (
                    "GO_HIL_INTEGRATED",
                    "GO_HIL_EXTERNAL_SPEED",
                    "MISSING_EXTERNAL_SAME_BOUNDARY_COMPARATOR",
                ),
            }
        ),
        "G33_registered_project_integrated_states_never_auto_promote_broader_states": _safe(
            lambda: report["hierarchy_contract"]
            == {
                "automatic_transfer": False,
                "edges": [
                    {"narrower": narrower, "broader": broader}
                    for narrower, broader in PREREQUISITE_EDGES
                ],
                "broader_requires_fresh_evidence": True,
            }
        ),
        "G34_weighted_score_winner_count_and_paper_scope_cannot_derive_sota": _safe(
            lambda: report["aggregation_contract"]
            == _aggregation_contract()
            and report["aggregation_contract"]["weighted_global_score"]
            == "PROHIBITED"
            and report["aggregation_contract"]["winner_count"]
            == "PROHIBITED"
            and report["aggregation_contract"][
                "go_single_paper_implies_three_sota"
            ]
            is False
        ),
        "G35_source_data_and_markdown_are_lossless_live_and_complete": _safe(
            outputs_ok
        ),
        "G36_one_semantic_mutation_per_gate_is_replayed_and_rejected": _safe(
            mutation_audit_ok
        ),
    }
    if tuple(gates) != GATE_IDS:
        raise AssertionError("gate order drifted")
    return gates


def _semantic_mutation_audit(
    report: Mapping[str, Any], *, config_path: Path
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    parent = copy.deepcopy(report["parent_summary"])
    artifacts = copy.deepcopy(report["artifact_registry"])

    def attempt(
        mutation_id: str,
        target_gate: str,
        mutate: Callable[[dict[str, Any]], None],
    ) -> None:
        candidate = copy.deepcopy(dict(report))
        mutate(candidate)
        gates = evaluate_gates(
            candidate,
            config_path=config_path,
            check_live_files=False,
            expected_parent_summary=parent,
            expected_artifact_registry=artifacts,
        )
        records.append(
            {
                "mutation_id": mutation_id,
                "target_gate": target_gate,
                "detected": gates[target_gate] is False,
                "failed_gates": [
                    key for key, passed in gates.items() if not passed
                ],
            }
        )

    attempt("M01_change_task_id", GATE_IDS[0], lambda x: x.update(task_id="T9.X"))
    attempt(
        "M02_change_parent_analysis",
        GATE_IDS[1],
        lambda x: x["parent_summary"]["v1"].update(analysis_sha256="0" * 64),
    )
    attempt(
        "M03_remove_parent_output_migration",
        GATE_IDS[2],
        lambda x: x["legacy_migration_table"].pop(),
    )
    attempt(
        "M04_change_registry_analysis",
        GATE_IDS[3],
        lambda x: x["parent_summary"]["registry"].update(
            analysis_sha256="0" * 64
        ),
    )
    attempt(
        "M05_change_literature_cutoff",
        GATE_IDS[4],
        lambda x: x["external_same_task_contract"].update(
            cutoff_inclusive="2026-07-26T00:00:00+08:00"
        ),
    )
    attempt(
        "M06_change_implementation_hash",
        GATE_IDS[5],
        lambda x: x["artifact_registry"]["implementation"].update(
            sha256="0" * 64
        ),
    )
    attempt(
        "M07_remove_direct_consumer",
        GATE_IDS[6],
        lambda x: x["downstream_consumption_contract"][
            "direct_consumers"
        ].pop(),
    )
    attempt(
        "M08_remove_scoped_state",
        GATE_IDS[7],
        lambda x: x["state_definitions"].pop(),
    )
    attempt(
        "M09_remove_typed_state_field",
        GATE_IDS[8],
        lambda x: x["current_states"][0].pop("evidence_grade"),
    )
    attempt(
        "M10_fill_current_claim_value",
        GATE_IDS[9],
        lambda x: x["current_states"][0].update(
            value="GO_LER_REGISTERED_BEST"
        ),
    )
    attempt(
        "M11_fill_missing_official_with_false",
        GATE_IDS[10],
        lambda x: x["current_asset_inventory"][
            "official_puviani_exact"
        ].update(value=False),
    )
    attempt(
        "M12_make_protocol_pass_a_performance_result",
        GATE_IDS[11],
        lambda x: x["performance_state"].update(protocol_only=False),
    )
    attempt(
        "M13_duplicate_migration",
        GATE_IDS[12],
        lambda x: x["legacy_migration_table"].append(
            copy.deepcopy(x["legacy_migration_table"][0])
        ),
    )
    attempt(
        "M14_enable_legacy_auto_promotion",
        GATE_IDS[13],
        lambda x: x["legacy_migration_table"][0].update(
            automatic_mapping=True
        ),
    )
    attempt(
        "M15_map_legacy_ler_to_external",
        GATE_IDS[14],
        lambda x: _migration(
            x, "ROUND_LER_SINGLE_MODE", "GO_LER_SOTA"
        ).update(candidate_destinations=["GO_LER_EXTERNAL_SOTA"]),
    )
    attempt(
        "M16_map_legacy_lifetime_to_physical",
        GATE_IDS[15],
        lambda x: _migration(
            x, "SIX_STATE_LOGICAL_LIFETIME", "GO_LIFETIME"
        ).update(candidate_destinations=["GO_PHYSICAL_LIFETIME"]),
    )
    attempt(
        "M17_map_legacy_hil_to_external_speed",
        GATE_IDS[16],
        lambda x: _migration(
            x, "RAW_IQ_DIGITAL_HIL", "GO_HIL_SPEED"
        ).update(candidate_destinations=["GO_HIL_EXTERNAL_SPEED"]),
    )
    attempt(
        "M18_map_tail_only_to_go",
        GATE_IDS[17],
        lambda x: _migration(
            x,
            "ROUND_LER_SINGLE_MODE",
            "NO_GO_LER_SOTA_TAIL_ONLY",
        ).update(candidate_destinations=["GO_LER_REGISTERED_BEST"]),
    )
    attempt(
        "M19_remove_all_baselines_requirement",
        GATE_IDS[18],
        lambda x: _definition(
            x, "GO_LER_REGISTERED_BEST"
        )["predicate"]["true"].remove("all_mandatory_baselines_complete"),
    )
    attempt(
        "M20_allow_zero_external_comparators",
        GATE_IDS[19],
        lambda x: _definition(
            x, "GO_LER_EXTERNAL_SOTA"
        )["predicate"]["positive"].remove(
            "external_same_task_eligible_count"
        ),
    )
    attempt(
        "M21_remove_six_state_lifetime_denominator",
        GATE_IDS[20],
        lambda x: _definition(
            x, "GO_LIFETIME_PROJECT_NATIVE"
        )["predicate"]["true"].remove("six_state_full_denominator"),
    )
    attempt(
        "M22_remove_project_native_prerequisite",
        GATE_IDS[21],
        lambda x: _definition(
            x, "GO_LIFETIME_EXTERNAL_SOTA"
        )["predicate"]["true"].remove("go_lifetime_project_native"),
    )
    attempt(
        "M23_remove_qpu_measurement_requirement",
        GATE_IDS[22],
        lambda x: _definition(
            x, "GO_PHYSICAL_LIFETIME"
        )["predicate"]["true"].remove("qpu_measured"),
    )
    attempt(
        "M24_remove_real_board_requirement",
        GATE_IDS[23],
        lambda x: _definition(
            x, "GO_HIL_INTEGRATED"
        )["predicate"]["true"].remove("real_board_present"),
    )
    attempt(
        "M25_remove_same_boundary_requirement",
        GATE_IDS[24],
        lambda x: _definition(
            x, "GO_HIL_EXTERNAL_SPEED"
        )["predicate"]["true"].remove(
            "same_code_problem_input_precision_and_latency_boundary"
        ),
    )
    attempt(
        "M26_remove_official_exact_prerequisite",
        GATE_IDS[25],
        lambda x: _definition(
            x, "GO_PUVIANI_NMF_SURPASS"
        )["predicate"]["true"].remove("official_puviani_exact"),
    )
    attempt(
        "M27_remove_external_signature_field",
        GATE_IDS[26],
        lambda x: x["external_same_task_contract"][
            "required_signature_fields"
        ].pop(),
    )
    attempt(
        "M28_remove_cutoff_revocation",
        GATE_IDS[27],
        lambda x: x["revocation_contract"]["global_triggers"].pop(3),
    )
    attempt(
        "M29_remove_state_revocation_trigger",
        GATE_IDS[28],
        lambda x: x["state_definitions"][0].update(
            revocation_triggers=[]
        ),
    )
    attempt(
        "M30_remove_revocation_edge",
        GATE_IDS[29],
        lambda x: x["revocation_contract"][
            "prerequisite_edges"
        ].pop(),
    )
    attempt(
        "M31_remove_forbidden_transfer",
        GATE_IDS[30],
        lambda x: x["forbidden_transfers"].pop(),
    )
    attempt(
        "M32_relabel_six_cycle_as_end_to_end",
        GATE_IDS[31],
        lambda x: next(
            row
            for row in x["forbidden_transfers"]
            if row["transfer_id"] == "FT-V2-SIX-CYCLE-TO-END-TO-END"
        ).update(rejection="ALLOWED"),
    )
    attempt(
        "M33_enable_hierarchy_auto_transfer",
        GATE_IDS[32],
        lambda x: x["hierarchy_contract"].update(
            automatic_transfer=True
        ),
    )
    attempt(
        "M34_enable_weighted_global_score",
        GATE_IDS[33],
        lambda x: x["aggregation_contract"].update(
            weighted_global_score="ALLOWED"
        ),
    )
    attempt(
        "M35_change_source_row_count",
        GATE_IDS[34],
        lambda x: x["source_data"].update(
            rows=x["source_data"]["rows"] - 1
        ),
    )
    attempt(
        "M36_forge_mutation_count",
        GATE_IDS[35],
        lambda x: x["semantic_mutation_audit"].update(
            detected=len(GATE_IDS) - 1
        ),
    )
    return {
        "count": len(records),
        "detected": sum(row["detected"] for row in records),
        "all_detected": all(row["detected"] for row in records),
        "one_per_gate": [row["target_gate"] for row in records]
        == list(GATE_IDS),
        "records": records,
    }


def _analysis_payload(report: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "task_id",
        "schema_version",
        "config_schema_version",
        "protocol_id",
        "amendment_status",
        "frozen_at",
        "parent_contract",
        "registry_parent_contract",
        "parent_summary",
        "legacy_migration_table",
        "external_same_task_contract",
        "predicate_type_contract",
        "state_definitions",
        "current_states",
        "current_asset_inventory",
        "current_claim_state",
        "performance_state",
        "forbidden_transfers",
        "revocation_contract",
        "hierarchy_contract",
        "aggregation_contract",
        "downstream_consumption_contract",
        "predicate_fixtures",
        "revocation_fixtures",
        "artifact_registry",
        "semantic_mutation_audit",
    )
    return {field: copy.deepcopy(report[field]) for field in fields}


def build_report(
    *,
    config_path: Path = DEFAULT_CONFIG,
    source_data_path: Path = DEFAULT_SOURCE_DATA,
    markdown_path: Path = DEFAULT_MARKDOWN,
) -> dict[str, Any]:
    config = _load(config_path)
    if (
        config.get("schema_version") != CONFIG_SCHEMA_VERSION
        or config.get("task_id") != TASK_ID
        or config.get("protocol_id") != PROTOCOL_ID
    ):
        raise ValueError("T9.1.5 config identity mismatch")
    if tuple(
        row["state_id"] for row in config["state_definitions"]
    ) != STATE_IDS:
        raise ValueError("T9.1.5 state definition order mismatch")

    parent_summary = _parent_summary(config, verify_live=True)
    definitions = copy.deepcopy(config["state_definitions"])
    report: dict[str, Any] = {
        "task_id": TASK_ID,
        "schema_version": REPORT_SCHEMA_VERSION,
        "config_schema_version": config["schema_version"],
        "protocol_id": PROTOCOL_ID,
        "amendment_status": config["amendment_status"],
        "frozen_at": config["frozen_at"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "parent_contract": copy.deepcopy(config["parent_contract"]),
        "registry_parent_contract": copy.deepcopy(
            config["registry_parent_contract"]
        ),
        "parent_summary": parent_summary,
        "legacy_migration_table": copy.deepcopy(
            config["legacy_migration_table"]
        ),
        "external_same_task_contract": copy.deepcopy(
            config["external_same_task_contract"]
        ),
        "predicate_type_contract": copy.deepcopy(
            config["predicate_type_contract"]
        ),
        "state_definitions": definitions,
        "current_states": _current_states(definitions),
        "current_asset_inventory": _asset_inventory(),
        "current_claim_state": copy.deepcopy(
            config["current_claim_state"]
        ),
        "performance_state": copy.deepcopy(config["performance_state"]),
        "forbidden_transfers": copy.deepcopy(
            config["forbidden_transfers"]
        ),
        "revocation_contract": _revocation_contract(definitions),
        "hierarchy_contract": {
            "automatic_transfer": False,
            "edges": [
                {"narrower": narrower, "broader": broader}
                for narrower, broader in PREREQUISITE_EDGES
            ],
            "broader_requires_fresh_evidence": True,
        },
        "aggregation_contract": _aggregation_contract(),
        "downstream_consumption_contract": copy.deepcopy(
            config["downstream_consumption_contract"]
        ),
        "predicate_fixtures": _predicate_fixtures(definitions),
        "revocation_fixtures": _revocation_fixtures(
            _current_states(definitions)
        ),
        "artifact_registry": _artifact_registry(config_path),
        "source_data": {
            "path": _relative(source_data_path),
            "rows": 0,
        },
        "markdown": {"path": _relative(markdown_path)},
        "semantic_mutation_audit": _valid_mutation_placeholder(),
        "gates": {},
        "gate_summary": {},
        "verdict": None,
        "analysis_sha256": "",
    }
    report["source_data"]["rows"] = len(_source_rows(report))
    report["semantic_mutation_audit"] = _semantic_mutation_audit(
        report, config_path=config_path
    )
    offline_gates = evaluate_gates(
        report,
        config_path=config_path,
        check_live_files=False,
        expected_parent_summary=parent_summary,
        expected_artifact_registry=report["artifact_registry"],
    )
    if not all(offline_gates.values()):
        raise ValueError(
            "T9.1.5 prepublication gates failed: "
            + ", ".join(
                key for key, passed in offline_gates.items() if not passed
            )
        )
    report["gates"] = dict(offline_gates)
    report["gate_summary"] = {
        "passed": len(GATE_IDS),
        "total": len(GATE_IDS),
        "failed": [],
    }
    report["verdict"] = VERDICT

    _write_source_data(report, source_data_path)
    report["source_data"] = {
        **_binding(source_data_path),
        "rows": len(_source_rows(report)),
    }
    _atomic_text(_render_markdown(report), markdown_path)
    report["markdown"] = _binding(markdown_path)
    live_gates = evaluate_gates(
        report, config_path=config_path, check_live_files=True
    )
    failed = [key for key, passed in live_gates.items() if not passed]
    report["gates"] = live_gates
    report["gate_summary"] = {
        "passed": len(GATE_IDS) - len(failed),
        "total": len(GATE_IDS),
        "failed": failed,
    }
    report["verdict"] = (
        VERDICT
        if not failed
        else "FAIL_T9_1_5_SCOPED_CLAIM_AMENDMENT"
    )
    if failed:
        _atomic_text(
            "# T9.1.5 生成失败\n\n"
            "最终 live gate 未通过，本文件不构成 child seal。失败 gates："
            + ", ".join(failed)
            + "\n",
            markdown_path,
        )
        raise ValueError(
            "T9.1.5 final live gates failed: " + ", ".join(failed)
        )
    report["analysis_sha256"] = _canonical_sha256(
        _analysis_payload(report)
    )
    return report


def verify_report(
    report_or_path: Mapping[str, Any] | str | Path = DEFAULT_REPORT,
    *,
    expected_analysis_sha256: str | None = None,
) -> dict[str, bool]:
    if not isinstance(report_or_path, Mapping) and (
        Path(report_or_path).resolve() != DEFAULT_REPORT.resolve()
    ):
        raise ValueError(
            "T9.1.5 verifier accepts only the canonical report path"
        )
    report = (
        copy.deepcopy(dict(report_or_path))
        if isinstance(report_or_path, Mapping)
        else _load(Path(report_or_path))
    )
    config_path = DEFAULT_CONFIG
    release_pin = _load(DEFAULT_RELEASE_PIN)
    expected_mutations = _semantic_mutation_audit(
        report, config_path=config_path
    )
    gates = evaluate_gates(
        report, config_path=config_path, check_live_files=True
    )
    expected_summary = {
        "passed": sum(gates.values()),
        "total": len(gates),
        "failed": [key for key, passed in gates.items() if not passed],
    }
    checks = {
        "identity": report.get("task_id") == TASK_ID
        and report.get("schema_version") == REPORT_SCHEMA_VERSION,
        "parents_live": all(
            report["parent_summary"]["ordered_consumer_checks"].values()
        ),
        "mutation_replay": report["semantic_mutation_audit"]
        == expected_mutations,
        "all_gates": all(gates.values()),
        "gate_cache": report["gates"] == gates
        and report["gate_summary"] == expected_summary,
        "verdict": report["verdict"] == VERDICT,
        "analysis_sha256": report["analysis_sha256"]
        == _canonical_sha256(_analysis_payload(report)),
        "canonical_release_pin": release_pin
        == {
            "schema_version": "t9.1.5-release-pin-v1",
            "task_id": TASK_ID,
            "protocol_id": PROTOCOL_ID,
            "analysis_sha256": report["analysis_sha256"],
            "config": _binding(DEFAULT_CONFIG),
            "implementation": _binding(IMPLEMENTATION),
            "report": _binding(DEFAULT_REPORT),
            "source_data": _binding(DEFAULT_SOURCE_DATA),
            "markdown": _binding(DEFAULT_MARKDOWN),
        },
        "caller_expected_analysis": expected_analysis_sha256 is None
        or report["analysis_sha256"] == expected_analysis_sha256,
        "source_data": _csv_lossless(
            report, DEFAULT_SOURCE_DATA
        ),
        "markdown": report["markdown"] == _binding(DEFAULT_MARKDOWN)
        and DEFAULT_MARKDOWN.read_text(encoding="utf-8")
        == _render_markdown(report),
        "all_current_claims_null": all(
            value is None
            for value in report["current_claim_state"].values()
        ),
        "all_performance_fields_null": all(
            value is None
            for key, value in report["performance_state"].items()
            if key != "protocol_only"
        )
        and report["performance_state"]["protocol_only"] is True,
    }
    if not all(checks.values()):
        raise ValueError(
            "T9.1.5 verification failed: "
            + ", ".join(
                key for key, passed in checks.items() if not passed
            )
            + "; failed_gates="
            + repr([key for key, passed in gates.items() if not passed])
        )
    return checks


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--source-data", type=Path, default=DEFAULT_SOURCE_DATA
    )
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--expected-analysis-sha256")
    arguments = parser.parse_args(argv)
    if arguments.verify:
        print(
            json.dumps(
                verify_report(
                    arguments.report,
                    expected_analysis_sha256=arguments.expected_analysis_sha256,
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    report = build_report(
        config_path=arguments.config,
        source_data_path=arguments.source_data,
        markdown_path=arguments.markdown,
    )
    _atomic_json(report, arguments.report)
    print(
        json.dumps(
            {
                "verdict": report["verdict"],
                "gate_summary": report["gate_summary"],
                "mutation_summary": {
                    key: report["semantic_mutation_audit"][key]
                    for key in ("count", "detected", "all_detected")
                },
                "analysis_sha256": report["analysis_sha256"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
