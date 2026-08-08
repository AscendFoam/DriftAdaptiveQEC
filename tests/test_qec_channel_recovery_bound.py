from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from cnn_fpga.benchmark import qec_channel_recovery_bound as report


ARTIFACT = Path("docs/t5_3_5_qec_channel_recovery_bound.json")
SOURCE = Path("docs/t5_3_5_qec_channel_recovery_bound_source_data.csv")


@pytest.fixture(scope="module")
def payload() -> dict:
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def test_checked_in_artifact_is_live_pass(payload: dict) -> None:
    gates = report.validate_artifact_payload(payload)
    assert payload["status"] == "PASS"
    assert len(gates) == 21
    assert all(gates.values())
    assert payload["implementation_sha256"] == report.implementation_sha256()


def test_report_has_full_sdp_cutoff_energy_and_gap_coverage(payload: dict) -> None:
    assert len(payload["small_sdp_validation"]) == 15
    assert len(payload["extended_cutoff_scan"]) == 15
    assert len(payload["energy_extension_scan"]) == 27
    assert len(payload["cutoff_diagnostics"]) == 21
    assert len(payload["actual_sbs_gap_diagnostics"]) == 12
    assert len(payload["teacher_student_gap_audit"]) == 4
    assert payload["source_data"]["row_count"] == 119


def test_every_small_sdp_has_repaired_primal_and_dual_certificate(payload: dict) -> None:
    for row in payload["small_sdp_validation"]:
        certificate = row["sdp"]
        assert certificate["primal_status"] in {"optimal", "optimal_inaccurate"}
        assert certificate["dual_status"] in {"optimal", "optimal_inaccurate"}
        assert certificate["repaired_primal_tp_residual"] <= 2.0e-8
        assert certificate["repaired_primal_minimum_eigenvalue"] >= -2.0e-8
        assert certificate["repaired_dual_minimum_slack_eigenvalue"] >= 0.9e-10
        assert certificate["intersection_certified_lower"] <= certificate[
            "intersection_certified_upper"
        ] + 2.0e-7
        assert certificate["repaired_certificate_width"] <= 2.0e-6


def test_cutoff_and_energy_extensions_do_not_claim_infinite_convergence(payload: dict) -> None:
    cutoff48 = [
        row for row in payload["extended_cutoff_scan"] if row["cutoff"] == 48
    ]
    assert len(cutoff48) == 3
    assert all(
        row["infinite_cutoff_convergence_claimed"] is False
        for row in payload["cutoff_diagnostics"]
    )
    high_noise_energy = {
        row["projector_delta"]: row["mean_code_photon_number"]
        for row in payload["energy_extension_scan"]
        if row["cutoff"] == 48 and row["noise_profile"] == "high"
    }
    assert high_noise_energy[0.28] > high_noise_energy[0.34] > high_noise_energy[0.44]
    assert payload["verdict"]["infinite_cutoff_or_energy_convergence"] == "NOT_ESTABLISHED"


def test_actual_sbs_completion_and_schedule_mismatch_are_explicit(payload: dict) -> None:
    for row in payload["actual_sbs_gap_diagnostics"]:
        expected = row["actual_sbs_cptni_entanglement_fidelity"] + (
            1.0 - row["actual_sbs_mean_code_survival"]
        ) / 4.0
        assert row["actual_sbs_completed_entanglement_fidelity"] == pytest.approx(
            expected, abs=2.0e-12
        )
        assert row["comparison_status"] == "SCHEDULE_MISMATCHED_DIAGNOSTIC_ONLY"
        assert row["certified_ordering_claimed"] is False
        assert row["deployable_decoder_gap_claimed"] is False


def test_teacher_student_metric_mismatch_stays_null(payload: dict) -> None:
    for row in payload["teacher_student_gap_audit"]:
        assert row["recovery_bound_gap"] is None
        assert row["status"] == "INCOMPARABLE"
        assert row["heterogeneous_metric_subtraction_performed"] is False


def test_source_data_hash_and_row_count_are_bound(payload: dict) -> None:
    digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    assert payload["source_data"] == {
        "path": SOURCE.as_posix(),
        "sha256": digest,
        "row_count": 119,
    }
    assert len(SOURCE.read_text(encoding="utf-8").splitlines()) == 120


def test_runner_rebuilds_all_machine_gates() -> None:
    rebuilt = report.run_report()
    assert rebuilt["status"] == "PASS"
    assert len(rebuilt["gates"]) == 21
    assert all(rebuilt["gates"].values())


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value["teacher_student_gap_audit"][0].update(
            recovery_bound_gap=0.1
        ),
        lambda value: value["actual_sbs_gap_diagnostics"][0].update(
            certified_ordering_claimed=True
        ),
        lambda value: value["small_sdp_validation"][0]["sdp"].update(
            repaired_dual_minimum_slack_eigenvalue=-1.0e-3
        ),
        lambda value: value["small_sdp_validation"][0]["petz"].update(
            theorem_optimal_upper=0.1
        ),
        lambda value: value["parent_audits"]["T5.3.2"].update(sha256="stale"),
        lambda value: value["claim_boundary"].update(
            deployable_decoder_or_controller=True
        ),
    ],
)
def test_semantic_mutations_are_rejected(payload: dict, mutate) -> None:
    changed = deepcopy(payload)
    mutate(changed)
    with pytest.raises(ValueError):
        report.validate_artifact_payload(changed)
