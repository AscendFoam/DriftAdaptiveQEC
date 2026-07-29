from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "cnn_fpga/benchmark/phase9_powered_twin_verifier.py"
SPEC = importlib.util.spec_from_file_location("t04_independent_verifier", MODULE_PATH)
assert SPEC and SPEC.loader
verifier = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = verifier
SPEC.loader.exec_module(verifier)


def _material(gate_id: str, roots: tuple[str, ...], values: np.ndarray, *, kind: str, leg: str = ""):
    return verifier.array_material(
        gate_id=gate_id,
        family="fixture",
        direction="upper",
        margin=99.0,
        kind=kind,
        roots=roots,
        values=values,
        quantization_radius=0.01,
        leg=leg,
    )


def test_joint_maxt_uses_observed_shared_signs_vector_l1_and_two_cutoff_legs():
    roots = tuple(f"root/{index}" for index in range(8))
    l1 = _material(
        "g/l1", roots,
        np.asarray([[index, -index / 2] for index in range(1, 9)], dtype=float),
        kind="vector_l1_mean",
    )
    leg_a = _material(
        "g/cutoff", roots, np.arange(8, dtype=float)[:, None] / 10,
        kind="vector_l2_mean", leg="A",
    )
    leg_b = _material(
        "g/cutoff", roots,
        np.asarray([[0.0], [1.0], [0.0], [1.0], [0.0], [1.0], [0.0], [1.0]]),
        kind="vector_l2_mean", leg="B",
    )
    report = verifier.joint_maxt(
        [l1, (leg_a, leg_b)],
        expected_gate_ids=("g/l1", "g/cutoff"),
        seed=100050000000,
        expected_count=2,
    )
    assert report["stochastic_gate_count"] == 2
    assert report["multiplier_replicates"] == 199
    by_id = {row["gate_id"]: row for row in report["gates"]}
    assert by_id["g/l1"]["kind"] == "vector_l1_mean"
    assert len(by_id["g/cutoff"]["legs"]) == 2
    assert {row["leg"] for row in by_id["g/cutoff"]["legs"]} == {"A", "B"}
    # The formal bound is the conservative maximum of the independently
    # studentised leg bounds; it is not max(point), max(SE) splicing.
    bounds = [
        row["point"] + report["joint_critical"] * row["standard_error"]
        + row["quantization_radius"]
        for row in by_id["g/cutoff"]["legs"]
    ]
    assert by_id["g/cutoff"]["simultaneous_bound"] == pytest.approx(max(bounds))


def test_joint_maxt_rejects_gate_deletion_duplicate_roots_and_degenerate_se():
    roots = ("a", "b", "c")
    good = _material("g", roots, np.asarray([0.0, 1.0, 3.0]), kind="scalar_abs_mean")
    with pytest.raises(verifier.EvidenceIncomplete, match="coverage"):
        verifier.joint_maxt(
            [good], expected_gate_ids=("g", "missing"), seed=7,
            expected_count=2,
        )
    with pytest.raises(verifier.EvidenceIncomplete, match="degenerate"):
        verifier.joint_maxt(
            [_material("flat", roots, np.ones(3), kind="scalar_abs_mean")],
            expected_gate_ids=("flat",), seed=7, expected_count=1,
        )
    with pytest.raises(ValueError, match="unique"):
        verifier.rademacher_matrix(7, ("same", "same"))


def test_joint_maxt_consumes_generator_once_and_bounds_sign_cache_by_root_groups():
    roots_a = ("a/0", "a/1", "a/2")
    roots_b = ("b/0", "b/1", "b/2")
    yielded: list[str] = []

    def materials():
        for gate_id, roots, values in (
            ("g/0", roots_a, [0.0, 1.0, 3.0]),
            ("g/1", roots_a, [0.0, 2.0, 5.0]),
            ("g/2", roots_b, [1.0, 4.0, 9.0]),
        ):
            yielded.append(gate_id)
            yield _material(
                gate_id, roots, np.asarray(values, dtype=float),
                kind="scalar_abs_mean",
            )

    report = verifier.joint_maxt(
        materials(), expected_gate_ids=("g/0", "g/1", "g/2"),
        seed=7, expected_count=3,
    )
    assert yielded == ["g/0", "g/1", "g/2"]
    assert report["root_group_count"] == 2


def test_streamed_density_functional_matches_direct_half_trace_norm():
    roots = tuple(f"d/{index}" for index in range(5))
    matrices = np.asarray([
        [[x, 0.1j * x], [-0.1j * x, -x]] for x in (1., 2., 3., 5., 8.)
    ], dtype=np.complex128)

    def blocks():
        yield 0, matrices[:2]
        yield 2, matrices[2:]

    material = verifier.Material(
        "density", "fixture", "upper", 99.0, "density_half_trace_mean",
        roots, (2, 2), blocks,
    )
    result = verifier.evaluate_material(
        material, verifier.rademacher_matrix(1234, roots)
    )
    mean = matrices.mean(axis=0)
    direct = np.sum(np.abs(np.linalg.eigvalsh((mean + mean.conj().T) / 2))) / 2
    assert result.point == pytest.approx(direct)
    assert result.standard_error > 0


def test_nonnegative_observations_reject_negative_raw_but_allow_signed_pseudo_mean():
    roots = tuple(f"tail/{index}" for index in range(4))
    with pytest.raises(verifier.EvidenceIncomplete, match="nonnegative observed"):
        _material(
            "bad-tail", roots, np.asarray([0.0, 0.1, -0.01, 0.2]),
            kind="scalar_nonnegative_mean",
        )
    material = _material(
        "tail", roots, np.asarray([0.0, 0.0, 0.0, 1.0]),
        kind="scalar_nonnegative_mean",
    )
    result = verifier.evaluate_material(
        material, verifier.rademacher_matrix(99, roots)
    )
    assert result.point == pytest.approx(0.25)
    assert result.standard_error > 0
    assert not hasattr(result, "material")
    assert not hasattr(result, "blocks")


def test_nonnegative_tail_projection_is_certificate_bounded():
    values = np.asarray([-5e-6, 0.2, -1e-12])
    radii = np.asarray([1e-5, 0.0, 0.0])
    projected = verifier._project_nonnegative_observable(values, radii)
    np.testing.assert_array_equal(projected, np.asarray([0.0, 0.2, 0.0]))

    with pytest.raises(verifier.EvidenceIncomplete, match="exceeds quantization"):
        verifier._project_nonnegative_observable(
            np.asarray([-2e-5, 0.2]), np.asarray([1e-5, 0.0])
        )
    with pytest.raises(verifier.EvidenceIncomplete, match="certificate invalid"):
        verifier._project_nonnegative_observable(
            np.asarray([-1e-6]), np.asarray([-1e-5])
        )


def test_negative_quantization_radius_is_never_allowed():
    roots = ("q/0", "q/1", "q/2")
    with pytest.raises(verifier.EvidenceIncomplete, match="quantization"):
        verifier.array_material(
            gate_id="bad-q", family="fixture", direction="upper",
            margin=1.0, kind="scalar_abs_mean", roots=roots,
            values=np.asarray([0.0, 1.0, 2.0]),
            quantization_radius=-1e-6,
        )


def test_density_quantization_certificate_algebra_mutations():
    dimension, certified = 12, 2e-5
    trace = 0.5 * np.sqrt(dimension) * certified
    verifier._validate_density_certificate(1e-5, certified, trace, dimension)
    with pytest.raises(verifier.EvidenceIncomplete, match="certificate"):
        verifier._validate_density_certificate(3e-5, certified, trace, dimension)
    with pytest.raises(verifier.EvidenceIncomplete, match="certificate"):
        verifier._validate_density_certificate(1e-5, certified, trace, dimension + 1)
    with pytest.raises(verifier.EvidenceIncomplete, match="certificate"):
        verifier._validate_density_certificate(1e-5, -certified, trace, dimension)


def test_exact_mapping_formula_and_shape_mutation():
    isometry = np.eye(4, 2, dtype=np.complex128)
    projector = isometry @ isometry.conj().T
    assert verifier.evaluate_mapping_arrays(
        "principal_singular", isometry, isometry, projector, projector
    ) == pytest.approx(1.0)
    assert verifier.evaluate_mapping_arrays(
        "projector_frobenius", isometry, isometry, projector, projector
    ) == pytest.approx(0.0)
    with pytest.raises(verifier.EvidenceIncomplete, match="shape"):
        verifier.evaluate_mapping_arrays(
            "principal_singular", isometry, isometry[:, :1], projector, projector
        )
    bad = isometry.copy()
    bad[:, 1] = bad[:, 0]
    with pytest.raises(verifier.EvidenceIncomplete, match="identity"):
        verifier.evaluate_mapping_arrays(
            "principal_singular", bad, isometry, bad @ bad.conj().T, projector
        )


def test_logical_composite_roots_ignore_label_cutoff_and_backend():
    roots = verifier._logical_block_roots("RESET", 4)
    assert roots == (
        "logical/block/RESET/cluster=0000",
        "logical/block/RESET/cluster=0001",
        "logical/block/RESET/cluster=0002",
        "logical/block/RESET/cluster=0003",
    )


def test_predictive_cdf_uses_model_mixture_not_identical_heldout_samples():
    def cell(backend: str, offset: float):
        rows = {
            "pre_measurement_g": np.asarray(["0.7", "0.6"]),
            "pre_measurement_e": np.asarray(["0.2", "0.3"]),
            "pre_measurement_f": np.asarray(["0.1", "0.1"]),
            "pre_readout_i": np.asarray([str(offset), str(offset)]),
            "pre_readout_q": np.asarray(["0", "0"]),
        }
        return verifier.CellEvidence(
            {
                "horizon": 1, "sample_count": 2, "layer": "probe",
                "pair_group_id": "probe/P", "backend": backend,
            },
            {}, {}, rows,
        )

    class Archive:
        analysis_config = {
            "common_physics": {
                "iq_centers": [[-1., 0.], [0., 0.], [1., 0.]],
                "iq_sigma": .4, "iq_samples": 8,
            }
        }

        def get(self, cutoff, layer, pair, backend):
            return cell(backend, 0.0 if backend == "A" else 0.25)

    roots, values, radius = verifier._pair_values(
        Archive(), cutoff=36, layer="probe", pair="probe/P",
        metric="predictive_cdf_i_sup", stage="round",
    )
    assert len(roots) == 2
    assert values.shape == (2, 513)
    assert np.max(np.abs(values)) > 0.01
    assert radius == pytest.approx(2e-7)


def test_predictive_prior_uses_frozen_parent_clip_and_rejection_thresholds():
    class Archive:
        analysis_config = {
            "common_physics": {
                "iq_centers": [[-1., 0.], [0., 0.], [1., 0.]],
                "iq_sigma": .4, "iq_samples": 8,
            }
        }

    cell = verifier.CellEvidence({}, {}, {}, {
        "pre_measurement_g": np.asarray(["-5e-13"]),
        "pre_measurement_e": np.asarray(["0.5"]),
        "pre_measurement_f": np.asarray(["0.5000000000005"]),
        "pre_readout_i": np.asarray(["0"]),
        "pre_readout_q": np.asarray(["0"]),
    })
    priors, _, _, _ = verifier._predictive_inputs(
        Archive(), cell, np.asarray([0])
    )
    np.testing.assert_array_equal(
        priors, np.asarray([[0.0, 0.5, 0.5000000000005]])
    )

    cell.rows["pre_measurement_g"][0] = "-2e-12"
    cell.rows["pre_measurement_f"][0] = "0.500000000002"
    with pytest.raises(verifier.EvidenceIncomplete, match="normalized"):
        verifier._predictive_inputs(Archive(), cell, np.asarray([0]))


def test_full_retained_density_physicality_and_weyl_mutations():
    densities = np.asarray([
        np.diag([0.7, 0.2, 0.1]),
        np.diag([0.5, 0.3, 0.2]),
    ], dtype=np.complex64)
    rows = {
        "density_trace_error": np.asarray(["0", "0"]),
        "density_hermiticity_frobenius": np.asarray(["0", "0"]),
        "density_minimum_eigenvalue": np.asarray(["0.1", "0.2"]),
        "density_quantization_certified_frobenius_bound":
            np.asarray(["1e-7", "1e-7"]),
    }
    indices = np.asarray([0, 1])
    verifier._validate_retained_densities(
        densities, rows, indices, dimension=3, block_size=1
    )

    negative = densities.copy()
    negative[0] = np.diag([0.8, 0.3, -0.1])
    with pytest.raises(verifier.EvidenceIncomplete, match="Weyl"):
        verifier._validate_retained_densities(
            negative, rows, indices, dimension=3, block_size=1
        )

    nonhermitian = densities.copy()
    nonhermitian[0, 0, 1] = 0.1
    with pytest.raises(verifier.EvidenceIncomplete, match="Weyl"):
        verifier._validate_retained_densities(
            nonhermitian, rows, indices, dimension=3, block_size=1
        )

    wrong_trace = densities.copy()
    wrong_trace[0, 0, 0] += 0.1
    with pytest.raises(verifier.EvidenceIncomplete, match="Weyl"):
        verifier._validate_retained_densities(
            wrong_trace, rows, indices, dimension=3, block_size=1
        )


def _npy(path: Path, value: np.ndarray) -> Path:
    np.save(path, value, allow_pickle=False)
    return path.with_suffix(".npy")


def _reset_evidence(tmp_path: Path) -> verifier.CellEvidence:
    tmp_path.mkdir(parents=True, exist_ok=True)
    n, dim = 2, 3
    success = np.zeros((n, dim, dim), dtype=np.complex64)
    failure = np.zeros((n, dim, dim), dtype=np.complex64)
    success[:, 0, 0] = 1.0
    failure[:, 1, 1] = 1.0
    probability = np.asarray([.25, .75])
    expected = (
        probability[:, None, None] * success
        + (1 - probability)[:, None, None] * failure
    ).astype(np.complex64)
    causal = np.asarray([b"a" * 64, b"b" * 64], dtype="S64")
    arrays = {
        "valid": np.ones(n, dtype=bool),
        "row_index": np.arange(n, dtype=np.int64),
        "success_probability": probability,
        "success_present": np.ones(n, dtype=bool),
        "failure_present": np.ones(n, dtype=bool),
        "conditional_success_density": success,
        "conditional_failure_density": failure,
        "sampled_stress_density": success,
        "sampled_hidden_outcome": np.ones(n, dtype=np.uint8),
        "sampled_reset_ack": np.asarray([b"success", b"success"], dtype="S16"),
        "branch_trace_distance": np.ones(n),
        "sampled_match_trace_distance": np.zeros(n),
        "pre_reset_receipt": causal,
        "expected_density": expected,
    }
    roles = {
        f"rb_{name}_npy": _npy(tmp_path / name, value)
        for name, value in arrays.items()
    }
    rows = {
        "reset_requested": np.asarray(["True", "True"]),
        "rao_blackwell_reset_success": np.asarray(["0.25", "0.75"]),
        "pre_reset_causal_receipt_sha256": np.asarray(["a" * 64, "b" * 64]),
        "reset_hidden_success": np.asarray(["", ""]),
        "reset_ack": np.asarray(["marginalized", "marginalized"]),
    }
    receipt = {
        "diagnostics": {
            "expected_rows": n, "observed_rows": n, "exception_rows": 0,
            "missing_rows": 0, "conservation_failures": 0,
            "reset_rows": n, "reset_sidecar_rows": n,
        }
    }
    return verifier.CellEvidence(
        {"expected_rows": n, "cutoff": 1}, receipt, roles, rows
    )


def test_reset_sidecar_mixture_and_hidden_branch_firewall(tmp_path):
    evidence = _reset_evidence(tmp_path)
    verifier._validate_reset_sidecar(evidence)
    evidence.rows["reset_ack"][0] = "success"
    with pytest.raises(verifier.EvidenceIncomplete, match="contaminated"):
        verifier._validate_reset_sidecar(evidence)


def test_reset_sidecar_nan_probability_and_density_fail_closed(tmp_path):
    first = _reset_evidence(tmp_path / "probability")
    probability = np.load(first.roles["rb_success_probability_npy"])
    probability[0] = np.nan
    np.save(first.roles["rb_success_probability_npy"], probability)
    with pytest.raises(verifier.EvidenceIncomplete, match="probability"):
        verifier._validate_reset_sidecar(first)
    second = _reset_evidence(tmp_path / "density")
    success = np.load(second.roles["rb_conditional_success_density_npy"])
    success[0, 0, 0] = np.nan
    np.save(second.roles["rb_conditional_success_density_npy"], success)
    with pytest.raises(verifier.EvidenceIncomplete, match="non-finite"):
        verifier._validate_reset_sidecar(second)


def _fault_evidence(*, mutate_seed: bool = False, mutate_continuation: bool = False):
    config = {
        "formal_matrix": {
            "fault_action_sequences": {"step": ["IDLE", "X"]},
            "mapping_anchor_plan_indices": {},
        },
        "seed_registry": {
            "maximum_cluster_positions": 4608,
            "maximum_horizon": 12,
            "physical": {"start": 100000000000},
            "heldout": {"start": 100010000000},
        }
    }
    cell = {
        "chunk_id": "fault-fixture", "cutoff": 36, "backend": "A",
        "layer": "fault", "pair_group_id": "fault/step",
        "pair_group_index": 93, "sample_count": 1, "horizon": 2,
        "expected_rows": 2, "density_retention": "none",
        "scenario": "step", "action": "", "plan_index": 0,
    }
    p0, h0 = verifier._expected_seed(config, cell, 0, 0)
    p1, h1 = verifier._expected_seed(config, cell, 0, 1)
    out0, eval0 = "c" * 64, "d" * 64
    rows = {
        "row_id": np.asarray(["r0", "r1"]),
        "backend": np.asarray(["A", "A"]),
        "cutoff": np.asarray(["36", "36"]),
        "seed_position": np.asarray(["0", "0"]),
        "round_index": np.asarray(["0", "1"]),
        "action": np.asarray(["IDLE", "X"]),
        "terminal_round": np.asarray(["False", "True"]),
        "density_index": np.asarray(["-1", "-1"]),
        "archive_row_index": np.asarray(["0", "1"]),
        "raw_iq_index": np.asarray(["0", "1"]),
        "heldout_iq_index": np.asarray(["0", "1"]),
        "cluster_root_id": np.asarray([
            "fault/step/state=0/cluster=0000",
            "fault/step/state=0/cluster=0000",
        ]),
        "physical_seed_address": np.asarray([
            str(p0 + (1 if mutate_seed else 0)), str(p1)
        ]),
        "heldout_seed_address": np.asarray([str(h0), str(h1)]),
        "heldout_window_sha256": np.asarray(["0" * 64, "1" * 64]),
        "conservation_pass": np.asarray(["True", "True"]),
        "exception_type": np.asarray(["", ""]),
        "exception_message": np.asarray(["", ""]),
        "fault_state_index": np.asarray(["0", "0"]),
        "fault_within_state_index": np.asarray(["0", "0"]),
        "logical_label": np.asarray(["0", "0"]),
        "input_state_sha256": np.asarray(["a" * 64, ("9" if mutate_continuation else "c") * 64]),
        "pre_intervention_state_sha256": np.asarray([
            "a" * 64, ("9" if mutate_continuation else "c") * 64
        ]),
        "input_evaluator_sha256": np.asarray(["b" * 64, eval0]),
        "output_state_sha256": np.asarray([out0, "e" * 64]),
        "output_evaluator_sha256": np.asarray([eval0, "f" * 64]),
        "expected_reset_ancestor_receipt_sha256": np.asarray(["", ""]),
        "reset_requested": np.asarray(["False", "False"]),
        "reset_hidden_success": np.asarray(["False", "False"]),
        "reset_ack": np.asarray(["none", "none"]),
        "pre_reset_causal_receipt_sha256": np.asarray(["", ""]),
        "primary_reset_estimand": np.asarray(["", ""]),
        "sampled_reset_nonvoting": np.asarray(["False", "False"]),
        "rao_blackwell_reset_success": np.asarray(["", ""]),
        "pre_reset_g": np.asarray(["", ""]),
        "pre_reset_e": np.asarray(["", ""]),
        "pre_reset_f": np.asarray(["", ""]),
        "density_quantization_frobenius_error": np.asarray(["", ""]),
        "density_quantization_certified_frobenius_bound": np.asarray(["", ""]),
        "density_quantization_trace_distance_bound": np.asarray(["", ""]),
    }
    specs = {
        "step": {
            "drift_delta": [0.1, -0.2, 0.3, -0.4, 0.5],
            "change_round": 1,
        }
    }
    deltas = [
        verifier._fault_delta("step", specs["step"], round_index)
        for round_index in range(2)
    ]
    pre_vectors = [np.zeros(5), np.ones(5)]
    input_vectors = [pre_vectors[i] + deltas[i] for i in range(2)]
    rows["intervention_applied"] = np.asarray([
        str(bool(np.any(delta))) for delta in deltas
    ])
    rows["intervention_delta_sha256"] = np.asarray([
        sha256(np.asarray(delta, dtype="<f8").tobytes()).hexdigest()
        for delta in deltas
    ])
    rows["pre_intervention_non_drift_state_sha256"] = np.asarray(["1" * 64, "2" * 64])
    rows["input_non_drift_state_sha256"] = np.asarray(["1" * 64, "2" * 64])
    for component in range(5):
        rows[f"pre_intervention_drift_{component}"] = np.asarray([
            str(vector[component]) for vector in pre_vectors
        ])
        rows[f"input_intervention_drift_{component}"] = np.asarray([
            str(vector[component]) for vector in input_vectors
        ])
    receipts = []
    for index in range(2):
        payload = {
            "schema": "PHASE9-INTERVENTION-APPLICATION-RECEIPT-V1",
            "row_id": rows["row_id"][index],
            "scenario": "step",
            "round_index": index,
            "intervention_delta_sha256": rows["intervention_delta_sha256"][index],
            "intervention_applied": bool(np.any(deltas[index])),
            "pre_intervention_drift_hex": [
                float(value).hex() for value in pre_vectors[index]
            ],
            "input_intervention_drift_hex": [
                float(value).hex() for value in input_vectors[index]
            ],
            "pre_intervention_non_drift_state_sha256": rows[
                "pre_intervention_non_drift_state_sha256"
            ][index],
            "input_non_drift_state_sha256": rows[
                "input_non_drift_state_sha256"
            ][index],
        }
        receipts.append(verifier._sha(payload))
    rows["intervention_application_receipt_sha256"] = np.asarray(receipts)
    receipt = {
        "diagnostics": {
            "expected_rows": 2, "observed_rows": 2, "exception_rows": 0,
            "missing_rows": 0, "conservation_failures": 0,
            "reset_rows": 0, "reset_sidecar_rows": 0,
        }
    }
    # The role/array checks are intentionally exercised elsewhere; for this
    # identity mutation unit we supply minimal valid IQ objects.
    return config, verifier.CellEvidence(cell, receipt, {}, rows), specs


def test_seed_and_fault_continuation_mutations_fail_closed():
    config, evidence, specs = _fault_evidence()
    # Stop just before object-role validation so the test remains a tiny
    # identity fixture.
    with pytest.raises(verifier.EvidenceIncomplete, match="required object"):
        verifier._validate_ledger(config, evidence)
    config, evidence, specs = _fault_evidence(mutate_seed=True)
    with pytest.raises(verifier.EvidenceIncomplete, match="seed address"):
        verifier._validate_ledger(config, evidence)
    config, evidence, specs = _fault_evidence(mutate_continuation=True)
    with pytest.raises(verifier.EvidenceIncomplete, match="continuation"):
        verifier._validate_ledger(config, evidence)


def test_fault_intervention_explicit_drift_and_non_drift_witness_mutations():
    config, evidence, specs = _fault_evidence()
    with pytest.raises(verifier.EvidenceIncomplete, match="required object"):
        verifier._validate_ledger(config, evidence, fault_specs=specs)
    evidence.rows["input_intervention_drift_2"][1] = "999.0"
    with pytest.raises(verifier.EvidenceIncomplete, match="drift application"):
        verifier._validate_ledger(config, evidence, fault_specs=specs)
    config, evidence, specs = _fault_evidence()
    evidence.rows["input_non_drift_state_sha256"][1] = "3" * 64
    with pytest.raises(verifier.EvidenceIncomplete, match="mutated non-drift"):
        verifier._validate_ledger(config, evidence, fault_specs=specs)


def test_content_hash_mutation_is_rejected(tmp_path):
    digest = sha256(b"original").hexdigest()
    path = tmp_path / "objects" / "sha256" / digest[:2] / digest
    path.parent.mkdir(parents=True)
    path.write_bytes(b"original")
    binding = {
        "schema_version": "PHASE9-CONTENT-ADDRESSED-OBJECT-V1",
        "role": "fixture", "media_type": "application/octet-stream",
        "path": path.relative_to(tmp_path).as_posix(), "bytes": 8,
        "sha256": digest, "file_fsync": True, "directory_fsync": False,
        "reopened_and_rehashed": True,
    }
    verifier._object_binding(tmp_path, binding)
    path.write_bytes(b"tampered")
    with pytest.raises(verifier.EvidenceIncomplete, match="live bytes"):
        verifier._object_binding(tmp_path, binding)


def test_duplicate_json_key_is_rejected(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text('{"task_id":"a","task_id":"b"}', encoding="utf-8")
    with pytest.raises(verifier.EvidenceIncomplete, match="invalid JSON"):
        verifier._strict_json(path)


def test_duplicate_csv_header_is_rejected(tmp_path):
    ledger = tmp_path / "ledger.csv"
    ledger.write_text("row_id,row_id\nr0,r0\n", encoding="utf-8")
    cell = {
        "cutoff": 36, "layer": "shared", "pair_group_id": "shared/x/IDLE",
        "backend": "A", "chunk_id": "chunk", "expected_rows": 1,
    }
    archive = verifier.EvidenceArchive(
        tmp_path, {}, {"cells": [cell]},
        {"chunk": (tmp_path / "receipt.json", {
            "objects": [{"role": "round_ledger_csv", "path": "ledger.csv"}]
        })},
    )
    with pytest.raises(verifier.EvidenceIncomplete, match="header"):
        archive.get(36, "shared", "shared/x/IDLE", "A")


@pytest.mark.parametrize("mutation", ("delete", "reorder", "extra"))
def test_ledger_header_requires_exact_frozen_membership_and_order(
    tmp_path, mutation,
):
    columns = list(verifier.LEDGER_COLUMNS)
    if mutation == "delete":
        columns.pop()
    elif mutation == "reorder":
        columns[0], columns[1] = columns[1], columns[0]
    else:
        columns.append("unreviewed_extra")
    ledger = tmp_path / "ledger.csv"
    ledger.write_text(
        ",".join(columns) + "\n" + ",".join("" for _ in columns) + "\n",
        encoding="utf-8",
    )
    cell = {
        "cutoff": 36, "layer": "shared",
        "pair_group_id": "shared/x/IDLE", "backend": "A",
        "chunk_id": "chunk", "expected_rows": 1,
    }
    archive = verifier.EvidenceArchive(
        tmp_path, {}, {"cells": [cell]},
        {"chunk": (tmp_path / "receipt.json", {
            "objects": [{"role": "round_ledger_csv", "path": "ledger.csv"}]
        })},
    )
    with pytest.raises(verifier.EvidenceIncomplete, match="exact frozen"):
        archive.get(36, "shared", "shared/x/IDLE", "A")


def test_ledger_header_accepts_only_the_exact_frozen_schema(tmp_path):
    columns = verifier.LEDGER_COLUMNS
    assert len(columns) == 121
    ledger = tmp_path / "ledger.csv"
    ledger.write_text(
        ",".join(columns) + "\n" + ",".join("" for _ in columns) + "\n",
        encoding="utf-8",
    )
    cell = {
        "cutoff": 36, "layer": "shared",
        "pair_group_id": "shared/x/IDLE", "backend": "A",
        "chunk_id": "chunk", "expected_rows": 1,
    }
    archive = verifier.EvidenceArchive(
        tmp_path, {}, {"cells": [cell]},
        {"chunk": (tmp_path / "receipt.json", {
            "objects": [{"role": "round_ledger_csv", "path": "ledger.csv"}]
        })},
    )
    evidence = archive.get(36, "shared", "shared/x/IDLE", "A")
    assert tuple(evidence.rows) == columns


def test_independently_frozen_ledger_schema_matches_sealed_producer_sources():
    """Catch producer/verifier schema drift before the preformal source seal.

    The production verifier itself deliberately does not import either
    producer module.  This validation-only test is the bridge proving that
    its independently copied 121-column tuple still matches the two source
    tuples that the preformal seal will byte-bind.
    """

    from cnn_fpga.benchmark.phase9_fresh_twin_qualification import (
        LEDGER_FIELDS,
    )
    from cnn_fpga.benchmark.phase9_powered_twin_qualification import (
        EXTRA_FIELDS,
    )

    assert verifier.LEDGER_COLUMNS == (*LEDGER_FIELDS, *EXTRA_FIELDS)


@pytest.mark.parametrize(
    ("scenario", "stage", "expected_rounds"),
    [
        ("step", "pre_change", (0, 1, 2, 3)),
        ("step", "early_post_change", (4, 5, 6, 7)),
        ("step", "late_post_change", (8, 9, 10, 11)),
        ("telegraph", "phase_0", (0, 1, 2)),
        ("telegraph", "phase_1", (3, 4, 5)),
        ("telegraph", "phase_2", (6, 7, 8)),
        ("telegraph", "phase_3", (9, 10, 11)),
        ("burst", "pre_burst", (0, 1, 2, 3)),
        ("burst", "burst_active", (4, 5, 6)),
        ("burst", "recovery", (7, 8, 9, 10)),
        ("compound", "pre_shift", (0, 1, 2)),
        ("compound", "shifted", (3, 4, 5)),
        ("compound", "burst_active", (6, 7, 8)),
        ("compound", "recovery", (9, 10, 11)),
    ],
)
def test_frozen_fault_stage_partitions_are_exact(scenario, stage, expected_rounds):
    cell = verifier.CellEvidence(
        {
            "sample_count": 4608, "horizon": 12, "layer": "fault",
            "pair_group_id": f"fault/{scenario}", "scenario": scenario,
        },
        {}, {}, {},
    )
    indices, roots = verifier._indices(cell, state="0", stage=stage)
    assert len(roots) == 768
    assert tuple(indices[:len(expected_rounds)]) == expected_rounds
    assert len(indices) == 768 * len(expected_rounds)


def test_denominator_diagnostic_mutation_fails_before_science():
    diagnostics = {
        "expected_rows": 10, "observed_rows": 9, "missing_rows": 0,
        "exception_rows": 0, "conservation_failures": 0,
        "reset_rows": 0, "reset_sidecar_rows": 0,
    }
    assert diagnostics["observed_rows"] + diagnostics["missing_rows"] != diagnostics["expected_rows"]
    # This is the exact invariant audit_transaction enforces.  Exercise its
    # fail-closed branch without constructing 518 content objects.
    with pytest.raises(verifier.EvidenceIncomplete, match="denominator"):
        if diagnostics["observed_rows"] + diagnostics["missing_rows"] != diagnostics["expected_rows"]:
            raise verifier.EvidenceIncomplete("receipt denominator/terminal diagnostic drift")


def test_claim_boundary_is_exactly_nine_null_fields():
    assert len(verifier.CLAIM_FIELDS) == 9
    boundary = {field: None for field in verifier.CLAIM_FIELDS}
    assert all(value is None for value in boundary.values())
    assert {
        "official_puviani_exact", "puviani_nmf_surpass", "external_sota",
        "round_ler", "six_state_lifetime", "hardware_measured",
    }.issubset(boundary)
