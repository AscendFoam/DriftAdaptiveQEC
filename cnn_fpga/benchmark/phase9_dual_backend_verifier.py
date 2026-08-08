"""Independent T9.2.4 raw-evidence verifier and simultaneous gate.

This module intentionally imports no physics backend.  It treats the formal
CSV ledger and compressed state archive as immutable data, reconstructs every
predeclared estimand, performs the frozen clustered max-T bootstrap, and emits
the gate ledger, report, Markdown summary and release pin.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
from scipy.stats import ks_2samp


TASK_ID = "T9.2.4"
VERIFIER_ID = "PHASE9-DUAL-BACKEND-INDEPENDENT-VERIFIER-V1"
SOURCE_SCHEMA = "PHASE9-DUAL-BACKEND-QUALIFICATION-SOURCE-DATA-V1"
PASS_VERDICT = "PASS_T9_2_4_DUAL_BACKEND_QUALIFIED"
FAIL_VERDICT = "NO_GO_TWIN_QUALIFICATION"
INCOMPLETE_VERDICT = "INCOMPLETE_FAIL_CLOSED"
TYPED_NULL_FIELDS = (
    "round_ler",
    "six_state_lifetime",
    "physical_break_even",
    "official_puviani_exact",
    "puviani_nmf_surpass",
    "external_sota",
    "hardware_measured",
    "rank",
)

INT_FIELDS = {
    "cutoff",
    "seed",
    "seed_position",
    "round_index",
    "density_index",
    "leakage_age",
}
BOOL_FIELDS = {
    "confirmation_cutoff",
    "convergence_member",
    "terminal_round",
    "reset_requested",
    "reset_hidden_success",
    "leakage_resident",
    "conservation_pass",
}
FLOAT_FIELDS = {
    "integrated_i",
    "integrated_q",
    "log_evidence",
    "posterior_g",
    "posterior_e",
    "posterior_f",
    "level_g",
    "level_e",
    "level_f",
    "mean_photon",
    "drift_0",
    "drift_1",
    "drift_2",
    "drift_3",
    "drift_4",
    "logical_survival",
    "logical_block_00_real",
    "logical_block_00_imag",
    "logical_block_01_real",
    "logical_block_01_imag",
    "logical_block_10_real",
    "logical_block_10_imag",
    "logical_block_11_real",
    "logical_block_11_imag",
    "density_trace_error",
    "density_hermiticity_frobenius",
    "density_minimum_eigenvalue",
    "posterior_normalization_error",
    "level_normalization_error",
}
SOURCE_FIELDS = (
    "record_type",
    "gate_id",
    "scope",
    "metric",
    "tolerance_name",
    "direction",
    "estimate",
    "standard_error",
    "simultaneous_bound",
    "tolerance",
    "margin",
    "passed",
    "cutoff",
    "backend",
    "matrix_name",
    "row",
    "column",
    "real",
    "imag",
    "denominator",
    "notes",
)


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def _canonical(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _sha_bytes(payload: bytes) -> str:
    return sha256(payload).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _binding(root: Path, relative: str) -> dict[str, object]:
    normalized = relative.replace("\\", "/")
    payload = (root / normalized).read_bytes()
    return {
        "path": normalized,
        "bytes": len(payload),
        "sha256": _sha_bytes(payload),
    }


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)


def _atomic_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Iterable[Mapping[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(fieldnames),
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    temporary.replace(path)


def verify_lineage(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    seal = _load_json(
        root / "docs/t9_2_4_formal_runner_amendment_seal.json"
    )
    unsigned = dict(seal)
    analysis = unsigned.pop("analysis_sha256", None)
    if (
        seal.get("task_id") != TASK_ID
        or seal.get("schema_version")
        != "PHASE9-DUAL-BACKEND-FORMAL-RUNNER-AMENDMENT-SEAL-V1"
        or seal.get("status") != "PRE_FORMAL_RUNNER_AND_MAPPING_SEALED"
        or seal.get("formal_result_accessed") is not False
        or seal.get("all_gates_passed") is not True
        or seal.get("gate_summary")
        != {"passed": 30, "total": 30, "all_passed": True}
        or seal.get("mutation_summary")
        != {"detected": 30, "total": 30, "all_detected": True}
        or analysis != _sha_bytes(_canonical(unsigned).encode("utf-8"))
    ):
        raise RuntimeError("formal child seal lineage failure")
    bindings = seal.get("live_bindings")
    if not isinstance(bindings, dict):
        raise RuntimeError("formal child seal lacks live bindings")
    for name, expected in bindings.items():
        if (
            not isinstance(expected, dict)
            or _binding(root, str(expected.get("path"))) != expected
        ):
            raise RuntimeError(f"formal child-seal binding drift: {name}")
    amendment = _load_json(
        root / "configs/phase9/t9_2_4_formal_runner_amendment.json"
    )
    parent = _load_json(
        root / "docs/t9_2_4_twin_qualification_preregistration.json"
    )
    if (
        parent.get("analysis_sha256")
        != amendment["parent_preregistration"]["analysis_sha256"]
        or parent.get("analysis_sha256")
        != "98e72c457dab941daab270b3bd63eec939564a6f1fedaad061eab59280988695"
    ):
        raise RuntimeError("parent preregistration analysis drift")
    tolerances = dict(parent["prefrozen_tolerances"])
    amended = dict(amendment["immutable_parent_tolerances"])
    amended.pop("source")
    amended.pop("change_count")
    if tolerances != amended:
        raise RuntimeError("amendment changed a parent tolerance")
    return seal, amendment


def _parse_bool(value: str, name: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"{name} is not an exact bool")


def load_ledger(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or len(set(reader.fieldnames)) != len(
            reader.fieldnames
        ):
            raise ValueError("ledger header invalid")
        rows: list[dict[str, Any]] = []
        for raw in reader:
            row: dict[str, Any] = dict(raw)
            for field in INT_FIELDS:
                value = row[field]
                row[field] = int(value) if value != "" else None
            for field in BOOL_FIELDS:
                value = row[field]
                row[field] = _parse_bool(value, field) if value != "" else None
            for field in FLOAT_FIELDS:
                value = row[field]
                number = float(value) if value != "" else np.nan
                if value != "" and not np.isfinite(number):
                    raise ValueError(f"non-finite ledger field: {field}")
                row[field] = number
            rows.append(row)
    return rows


def load_archive(path: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as archive:
        if archive["schema"].tolist() != [
            "PHASE9-DUAL-BACKEND-STATE-ARCHIVE-V1"
        ]:
            raise ValueError("state archive schema mismatch")
        values = {name: np.array(archive[name]) for name in archive.files}
    density_by_row: dict[str, np.ndarray] = {}
    for cutoff in (8, 12):
        matrices = values[f"densities_cutoff_{cutoff}"]
        row_ids = values[f"row_ids_cutoff_{cutoff}"].tolist()
        expected_shape = (len(row_ids), cutoff * 3, cutoff * 3)
        if matrices.shape != expected_shape:
            raise ValueError("state archive density shape mismatch")
        for row_id, matrix in zip(row_ids, matrices):
            if str(row_id) in density_by_row:
                raise ValueError("duplicate archived density row")
            density_by_row[str(row_id)] = matrix
    return values, density_by_row


class RowGroup:
    def __init__(
        self,
        rows: Sequence[Mapping[str, Any]],
        density_by_row: Mapping[str, np.ndarray],
    ) -> None:
        self.rows = [dict(row) for row in rows]
        self.density_by_row = density_by_row
        self.by_position: dict[int, list[dict[str, Any]]] = {}
        for row in self.rows:
            position = row["seed_position"]
            if not isinstance(position, int):
                raise ValueError("group seed position missing")
            self.by_position.setdefault(position, []).append(row)

    @property
    def positions(self) -> tuple[int, ...]:
        return tuple(sorted(self.by_position))

    def select(self, positions: np.ndarray | None) -> list[dict[str, Any]]:
        if positions is None:
            return list(self.rows)
        selected: list[dict[str, Any]] = []
        for position in positions:
            if int(position) not in self.by_position:
                raise ValueError("bootstrap selected absent seed position")
            selected.extend(self.by_position[int(position)])
        return selected

    def density_mean(
        self,
        positions: np.ndarray | None,
        *,
        terminal_only: bool = False,
    ) -> np.ndarray:
        rows = self.select(positions)
        if terminal_only:
            rows = [row for row in rows if row["terminal_round"] is True]
        matrices = [
            self.density_by_row[str(row["row_id"])]
            for row in rows
            if int(row["density_index"]) >= 0
        ]
        if not matrices or len(matrices) != len(rows):
            raise ValueError("density archive coverage mismatch")
        return np.mean(np.stack(matrices), axis=0)


SelectionKey = tuple[str, str, int]
Selections = Mapping[SelectionKey, np.ndarray]


@dataclass
class MetricSpec:
    gate_id: str
    scope: str
    metric: str
    tolerance_name: str
    tolerance: float
    direction: str
    estimator: Callable[[Selections | None], float]
    selection_keys: tuple[SelectionKey, ...]
    denominator: str
    stochastic: bool = True
    estimate: float = np.nan
    standard_error: float = 0.0
    bound: float = np.nan
    passed: bool = False


def _selected(group: RowGroup, selections: Selections | None, key: SelectionKey):
    return group.select(None if selections is None else selections[key])


def _mean(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    values = np.asarray([row[field] for row in rows], dtype=np.float64)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError(f"invalid mean field {field}")
    return float(np.mean(values))


def _vector_mean(
    rows: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
) -> np.ndarray:
    values = np.asarray(
        [[row[field] for field in fields] for row in rows],
        dtype=np.float64,
    )
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("invalid vector mean")
    return np.mean(values, axis=0)


def _trace_distance(left: np.ndarray, right: np.ndarray) -> float:
    delta = 0.5 * ((left - right) + (left - right).conj().T)
    return 0.5 * float(np.sum(np.abs(np.linalg.eigvalsh(delta))))


def _embed_density(matrix: np.ndarray, from_cutoff: int, to_cutoff: int) -> np.ndarray:
    if not 0 < from_cutoff <= to_cutoff:
        raise ValueError("invalid density embedding cutoffs")
    output = np.zeros((to_cutoff * 3, to_cutoff * 3), dtype=np.complex128)
    for left_n in range(from_cutoff):
        for left_level in range(3):
            source_left = left_n * 3 + left_level
            target_left = source_left
            for right_n in range(from_cutoff):
                for right_level in range(3):
                    source_right = right_n * 3 + right_level
                    target_right = source_right
                    output[target_left, target_right] = matrix[
                        source_left,
                        source_right,
                    ]
    return output


def _logical_block(row: Mapping[str, Any]) -> np.ndarray:
    return np.array(
        [
            [
                row["logical_block_00_real"]
                + 1.0j * row["logical_block_00_imag"],
                row["logical_block_01_real"]
                + 1.0j * row["logical_block_01_imag"],
            ],
            [
                row["logical_block_10_real"]
                + 1.0j * row["logical_block_10_imag"],
                row["logical_block_11_real"]
                + 1.0j * row["logical_block_11_imag"],
            ],
        ],
        dtype=np.complex128,
    )


def _ptm(
    groups: Mapping[str, RowGroup],
    selections: Selections | None,
    key: SelectionKey,
) -> np.ndarray:
    paulis = (
        np.eye(2, dtype=np.complex128),
        np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128),
        np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128),
        np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128),
    )
    axes = (("+", "-", 1), ("+i", "-i", 2), ("0", "1", 3))
    matrix = np.zeros((4, 4), dtype=np.float64)
    identity_columns: list[np.ndarray] = []
    selection = None if selections is None else selections[key]
    for plus, minus, column in axes:
        plus_rows = groups[plus].select(selection)
        minus_rows = groups[minus].select(selection)
        plus_block = np.mean(
            np.stack([_logical_block(row) for row in plus_rows]),
            axis=0,
        )
        minus_block = np.mean(
            np.stack([_logical_block(row) for row in minus_rows]),
            axis=0,
        )
        plus_coords = np.array(
            [np.trace(operator @ plus_block).real for operator in paulis]
        )
        minus_coords = np.array(
            [np.trace(operator @ minus_block).real for operator in paulis]
        )
        identity_columns.append(0.5 * (plus_coords + minus_coords))
        matrix[:, column] = 0.5 * (plus_coords - minus_coords)
    matrix[:, 0] = np.mean(np.stack(identity_columns), axis=0)
    return matrix


def _group_index(
    rows: Sequence[Mapping[str, Any]],
    density_by_row: Mapping[str, np.ndarray],
) -> dict[tuple[str, str, int, str], RowGroup]:
    buckets: dict[tuple[str, str, int, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["layer"]),
            str(row["cell_base"]),
            int(row["cutoff"]),
            str(row["backend"]),
        )
        buckets.setdefault(key, []).append(row)
    return {
        key: RowGroup(values, density_by_row)
        for key, values in buckets.items()
    }


def _metric(
    *,
    gate_id: str,
    scope: str,
    metric: str,
    tolerance_name: str,
    tolerances: Mapping[str, float],
    estimator: Callable[[Selections | None], float],
    keys: Sequence[SelectionKey],
    denominator: str,
) -> MetricSpec:
    return MetricSpec(
        gate_id=gate_id,
        scope=scope,
        metric=metric,
        tolerance_name=tolerance_name,
        tolerance=float(tolerances[tolerance_name]),
        direction="upper",
        estimator=estimator,
        selection_keys=tuple(keys),
        denominator=denominator,
    )


def _physical_ab_specs(
    *,
    scope: str,
    group_a: RowGroup,
    group_b: RowGroup,
    key_a: SelectionKey,
    key_b: SelectionKey,
    tolerances: Mapping[str, float],
    terminal_only: bool = False,
) -> list[MetricSpec]:
    keys = (key_a, key_b)

    def rows(selection: Selections | None):
        a = _selected(group_a, selection, key_a)
        b = _selected(group_b, selection, key_b)
        if terminal_only:
            a = [row for row in a if row["terminal_round"] is True]
            b = [row for row in b if row["terminal_round"] is True]
        return a, b

    specs = [
        _metric(
            gate_id=f"{scope}|density",
            scope=scope,
            metric="ensemble_density_trace_distance",
            tolerance_name=(
                "maximum_short_trajectory_terminal_trace_distance"
                if terminal_only
                else "maximum_ensemble_density_trace_distance"
            ),
            tolerances=tolerances,
            estimator=lambda selection: _trace_distance(
                group_a.density_mean(
                    None if selection is None else selection[key_a],
                    terminal_only=terminal_only,
                ),
                group_b.density_mean(
                    None if selection is None else selection[key_b],
                    terminal_only=terminal_only,
                ),
            ),
            keys=keys,
            denominator=scope,
        ),
        _metric(
            gate_id=f"{scope}|mean_photon",
            scope=scope,
            metric="mean_photon_difference",
            tolerance_name="maximum_mean_photon_difference",
            tolerances=tolerances,
            estimator=lambda selection: abs(
                _mean(rows(selection)[0], "mean_photon")
                - _mean(rows(selection)[1], "mean_photon")
            ),
            keys=keys,
            denominator=scope,
        ),
        _metric(
            gate_id=f"{scope}|levels",
            scope=scope,
            metric="level_probability_l1",
            tolerance_name="maximum_level_probability_l1",
            tolerances=tolerances,
            estimator=lambda selection: float(
                np.sum(
                    np.abs(
                        _vector_mean(
                            rows(selection)[0],
                            ("level_g", "level_e", "level_f"),
                        )
                        - _vector_mean(
                            rows(selection)[1],
                            ("level_g", "level_e", "level_f"),
                        )
                    )
                )
            ),
            keys=keys,
            denominator=scope,
        ),
        _metric(
            gate_id=f"{scope}|iq_mean",
            scope=scope,
            metric="integrated_iq_mean_difference",
            tolerance_name="maximum_integrated_iq_mean_difference",
            tolerances=tolerances,
            estimator=lambda selection: float(
                np.linalg.norm(
                    _vector_mean(
                        rows(selection)[0],
                        ("integrated_i", "integrated_q"),
                    )
                    - _vector_mean(
                        rows(selection)[1],
                        ("integrated_i", "integrated_q"),
                    )
                )
            ),
            keys=keys,
            denominator=scope,
        ),
        _metric(
            gate_id=f"{scope}|iq_covariance",
            scope=scope,
            metric="integrated_iq_covariance_frobenius",
            tolerance_name="maximum_integrated_iq_covariance_frobenius",
            tolerances=tolerances,
            estimator=lambda selection: float(
                np.linalg.norm(
                    np.cov(
                        np.asarray(
                            [
                                [row["integrated_i"], row["integrated_q"]]
                                for row in rows(selection)[0]
                            ]
                        ).T,
                        ddof=1,
                    )
                    - np.cov(
                        np.asarray(
                            [
                                [row["integrated_i"], row["integrated_q"]]
                                for row in rows(selection)[1]
                            ]
                        ).T,
                        ddof=1,
                    ),
                    ord="fro",
                )
            ),
            keys=keys,
            denominator=scope,
        ),
        _metric(
            gate_id=f"{scope}|iq_ks",
            scope=scope,
            metric="iq_two_sample_ks",
            tolerance_name="maximum_iq_two_sample_ks",
            tolerances=tolerances,
            estimator=lambda selection: max(
                float(
                    ks_2samp(
                        [row["integrated_i"] for row in rows(selection)[0]],
                        [row["integrated_i"] for row in rows(selection)[1]],
                        alternative="two-sided",
                        method="asymp",
                    ).statistic
                ),
                float(
                    ks_2samp(
                        [row["integrated_q"] for row in rows(selection)[0]],
                        [row["integrated_q"] for row in rows(selection)[1]],
                        alternative="two-sided",
                        method="asymp",
                    ).statistic
                ),
            ),
            keys=keys,
            denominator=scope,
        ),
        _metric(
            gate_id=f"{scope}|log_evidence",
            scope=scope,
            metric="log_evidence_mean_difference",
            tolerance_name="maximum_log_evidence_mean_difference",
            tolerances=tolerances,
            estimator=lambda selection: abs(
                _mean(rows(selection)[0], "log_evidence")
                - _mean(rows(selection)[1], "log_evidence")
            ),
            keys=keys,
            denominator=scope,
        ),
        _metric(
            gate_id=f"{scope}|posterior",
            scope=scope,
            metric="posterior_mean_l1",
            tolerance_name="maximum_posterior_mean_l1",
            tolerances=tolerances,
            estimator=lambda selection: float(
                np.sum(
                    np.abs(
                        _vector_mean(
                            rows(selection)[0],
                            ("posterior_g", "posterior_e", "posterior_f"),
                        )
                        - _vector_mean(
                            rows(selection)[1],
                            ("posterior_g", "posterior_e", "posterior_f"),
                        )
                    )
                )
            ),
            keys=keys,
            denominator=scope,
        ),
        _metric(
            gate_id=f"{scope}|leakage_residence",
            scope=scope,
            metric="leakage_residence_rate_difference",
            tolerance_name="maximum_leakage_residence_rate_difference",
            tolerances=tolerances,
            estimator=lambda selection: abs(
                np.mean(
                    [float(row["leakage_resident"]) for row in rows(selection)[0]]
                )
                - np.mean(
                    [float(row["leakage_resident"]) for row in rows(selection)[1]]
                )
            ),
            keys=keys,
            denominator=scope,
        ),
    ]
    if any(row["reset_requested"] is True for row in group_a.rows):
        specs.append(
            _metric(
                gate_id=f"{scope}|reset_success",
                scope=scope,
                metric="reset_success_rate_difference",
                tolerance_name="maximum_reset_success_rate_difference",
                tolerances=tolerances,
                estimator=lambda selection: abs(
                    np.mean(
                        [
                            float(row["reset_hidden_success"])
                            for row in rows(selection)[0]
                            if row["reset_requested"] is True
                        ]
                    )
                    - np.mean(
                        [
                            float(row["reset_hidden_success"])
                            for row in rows(selection)[1]
                            if row["reset_requested"] is True
                        ]
                    )
                ),
                keys=keys,
                denominator=scope,
            )
        )
    return specs


def _terminal_observable_vector(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    terminal = [row for row in rows if row["terminal_round"] is True]
    drift_norm = np.mean(
        [
            np.linalg.norm(
                [row[f"drift_{index}"] for index in range(5)]
            )
            for row in terminal
        ]
    )
    return np.concatenate(
        (
            [_mean(terminal, "mean_photon")],
            _vector_mean(terminal, ("level_g", "level_e", "level_f")),
            _vector_mean(terminal, ("integrated_i", "integrated_q")),
            [_mean(terminal, "log_evidence")],
            [
                np.mean(
                    [float(row["leakage_resident"]) for row in terminal]
                )
            ],
            [drift_norm],
        )
    )


def _fault_ab_specs(
    *,
    scope: str,
    group_a: RowGroup,
    group_b: RowGroup,
    key_a: SelectionKey,
    key_b: SelectionKey,
    tolerances: Mapping[str, float],
) -> list[MetricSpec]:
    keys = (key_a, key_b)
    specs = [
        _metric(
            gate_id=f"{scope}|terminal_density",
            scope=scope,
            metric="short_trajectory_terminal_trace_distance",
            tolerance_name="maximum_short_trajectory_terminal_trace_distance",
            tolerances=tolerances,
            estimator=lambda selection: _trace_distance(
                group_a.density_mean(
                    None if selection is None else selection[key_a],
                    terminal_only=True,
                ),
                group_b.density_mean(
                    None if selection is None else selection[key_b],
                    terminal_only=True,
                ),
            ),
            keys=keys,
            denominator=scope,
        ),
        _metric(
            gate_id=f"{scope}|terminal_observables",
            scope=scope,
            metric="short_trajectory_observable_mean_difference",
            tolerance_name="maximum_short_trajectory_observable_mean_difference",
            tolerances=tolerances,
            estimator=lambda selection: float(
                np.max(
                    np.abs(
                        _terminal_observable_vector(
                            _selected(group_a, selection, key_a)
                        )
                        - _terminal_observable_vector(
                            _selected(group_b, selection, key_b)
                        )
                    )
                )
            ),
            keys=keys,
            denominator=scope,
        ),
        _metric(
            gate_id=f"{scope}|leakage_residence",
            scope=scope,
            metric="leakage_residence_rate_difference",
            tolerance_name="maximum_leakage_residence_rate_difference",
            tolerances=tolerances,
            estimator=lambda selection: abs(
                np.mean(
                    [
                        float(row["leakage_resident"])
                        for row in _selected(group_a, selection, key_a)
                    ]
                )
                - np.mean(
                    [
                        float(row["leakage_resident"])
                        for row in _selected(group_b, selection, key_b)
                    ]
                )
            ),
            keys=keys,
            denominator=scope,
        ),
    ]
    return specs


def build_metric_specs(
    rows: Sequence[Mapping[str, Any]],
    archive: Mapping[str, np.ndarray],
    density_by_row: Mapping[str, np.ndarray],
    amendment: Mapping[str, Any],
) -> tuple[list[MetricSpec], list[dict[str, object]]]:
    tolerances = amendment["immutable_parent_tolerances"]
    groups = _group_index(rows, density_by_row)
    specs: list[MetricSpec] = []
    source_rows: list[dict[str, object]] = []

    # Deterministic same-cutoff mapping gates and matrix Source Data.
    for cutoff in (8, 12):
        a = archive[f"mapping_isometry_a_cutoff_{cutoff}"]
        b = archive[f"mapping_isometry_b_cutoff_{cutoff}"]
        singular = np.linalg.svd(a.conj().T @ b, compute_uv=False)
        projector_a = archive[f"mapping_projector_a_cutoff_{cutoff}"]
        projector_b = archive[f"mapping_projector_b_cutoff_{cutoff}"]
        minimum = float(np.min(singular))
        frobenius = float(
            np.linalg.norm(projector_a - projector_b, ord="fro")
        )
        specs.extend(
            [
                MetricSpec(
                    gate_id=f"mapping|cutoff={cutoff}|principal_singular",
                    scope=f"mapping|cutoff={cutoff}",
                    metric="minimum_code_principal_singular_value",
                    tolerance_name="minimum_code_principal_singular_value",
                    tolerance=float(
                        tolerances["minimum_code_principal_singular_value"]
                    ),
                    direction="lower",
                    estimator=lambda _selection, value=minimum: value,
                    selection_keys=(),
                    denominator="deterministic mapping",
                    stochastic=False,
                ),
                MetricSpec(
                    gate_id=f"mapping|cutoff={cutoff}|projector_frobenius",
                    scope=f"mapping|cutoff={cutoff}",
                    metric="code_projector_frobenius",
                    tolerance_name="maximum_code_projector_frobenius",
                    tolerance=float(
                        tolerances["maximum_code_projector_frobenius"]
                    ),
                    direction="upper",
                    estimator=lambda _selection, value=frobenius: value,
                    selection_keys=(),
                    denominator="deterministic mapping",
                    stochastic=False,
                ),
            ]
        )
        for backend in ("a", "b"):
            for matrix_name in ("isometry", "projector"):
                matrix = archive[
                    f"mapping_{matrix_name}_{backend}_cutoff_{cutoff}"
                ]
                for row_index in range(matrix.shape[0]):
                    for column_index in range(matrix.shape[1]):
                        source_rows.append(
                            {
                                "record_type": "mapping_matrix_entry",
                                "gate_id": "",
                                "scope": f"mapping|cutoff={cutoff}",
                                "metric": "",
                                "tolerance_name": "",
                                "direction": "",
                                "estimate": "",
                                "standard_error": "",
                                "simultaneous_bound": "",
                                "tolerance": "",
                                "margin": "",
                                "passed": "",
                                "cutoff": cutoff,
                                "backend": backend.upper(),
                                "matrix_name": matrix_name,
                                "row": row_index,
                                "column": column_index,
                                "real": float(matrix[row_index, column_index].real),
                                "imag": float(matrix[row_index, column_index].imag),
                                "denominator": "",
                                "notes": "",
                            }
                        )

    # Same-cutoff shared physical and probe cells.
    cell_keys = sorted(
        {
            (layer, cell, cutoff)
            for layer, cell, cutoff, backend in groups
            if layer in {"shared", "probe"}
        }
    )
    for layer, cell, cutoff in cell_keys:
        group_a = groups[(layer, cell, cutoff, "A")]
        group_b = groups[(layer, cell, cutoff, "B")]
        n = len(group_a.positions)
        key_a = (layer, "A", n)
        key_b = (layer, "B", n)
        specs.extend(
            _physical_ab_specs(
                scope=f"ab|{cell}|cutoff={cutoff}",
                group_a=group_a,
                group_b=group_b,
                key_a=key_a,
                key_b=key_b,
                tolerances=tolerances,
            )
        )

    # Logical survival cells.
    logical_keys = sorted(
        {
            (cell, cutoff)
            for layer, cell, cutoff, backend in groups
            if layer == "logical"
        }
    )
    for cell, cutoff in logical_keys:
        group_a = groups[("logical", cell, cutoff, "A")]
        group_b = groups[("logical", cell, cutoff, "B")]
        n = len(group_a.positions)
        key_a = ("logical", "A", n)
        key_b = ("logical", "B", n)
        specs.append(
            _metric(
                gate_id=f"ab|{cell}|cutoff={cutoff}|survival",
                scope=f"ab|{cell}|cutoff={cutoff}",
                metric="logical_survival_difference",
                tolerance_name="maximum_logical_survival_difference",
                tolerances=tolerances,
                estimator=lambda selection, ga=group_a, gb=group_b, ka=key_a, kb=key_b: abs(
                    _mean(_selected(ga, selection, ka), "logical_survival")
                    - _mean(_selected(gb, selection, kb), "logical_survival")
                ),
                keys=(key_a, key_b),
                denominator=f"{cell}|cutoff={cutoff}",
            )
        )

    # Same-cutoff unconditional PTM cells.
    for cutoff in (8, 12):
        actions = sorted(
            {
                str(row["action"])
                for row in rows
                if row["layer"] == "logical" and row["cutoff"] == cutoff
            }
        )
        for action in actions:
            by_backend: dict[str, dict[str, RowGroup]] = {}
            for backend in ("A", "B"):
                by_backend[backend] = {
                    label: groups[
                        (
                            "logical",
                            f"logical|{label}|{action}",
                            cutoff,
                            backend,
                        )
                    ]
                    for label in ("0", "1", "+", "-", "+i", "-i")
                }
            n = len(by_backend["A"]["0"].positions)
            key_a = ("logical", "A", n)
            key_b = ("logical", "B", n)
            specs.append(
                _metric(
                    gate_id=f"ab|logical_ptm|{action}|cutoff={cutoff}",
                    scope=f"ab|logical_ptm|{action}|cutoff={cutoff}",
                    metric="logical_ptm_entry_difference",
                    tolerance_name="maximum_logical_ptm_entry_difference",
                    tolerances=tolerances,
                    estimator=lambda selection, a=by_backend["A"], b=by_backend["B"], ka=key_a, kb=key_b: float(
                        np.max(
                            np.abs(
                                _ptm(a, selection, ka)
                                - _ptm(b, selection, kb)
                            )
                        )
                    ),
                    keys=(key_a, key_b),
                    denominator=f"logical PTM {action} cutoff={cutoff}",
                )
            )

    # Fault trajectory cells.
    for scenario in ("step", "telegraph", "burst", "compound"):
        for cutoff in (8, 12):
            cell = f"fault|{scenario}"
            group_a = groups[("fault", cell, cutoff, "A")]
            group_b = groups[("fault", cell, cutoff, "B")]
            n = len(group_a.positions)
            key_a = ("fault", "A", n)
            key_b = ("fault", "B", n)
            specs.extend(
                _fault_ab_specs(
                    scope=f"ab|{cell}|cutoff={cutoff}",
                    group_a=group_a,
                    group_b=group_b,
                    key_a=key_a,
                    key_b=key_b,
                    tolerances=tolerances,
                )
            )

    # Within-backend cutoff confirmation.  Use only rows declared as members.
    convergence_rows = [
        row
        for row in rows
        if row["convergence_member"] is True
        and row["layer"] in {"shared", "logical", "fault"}
    ]
    convergence_groups = _group_index(convergence_rows, density_by_row)
    for backend in ("A", "B"):
        shared_cells = sorted(
            {
                cell
                for layer, cell, cutoff, observed_backend in convergence_groups
                if layer == "shared" and observed_backend == backend
            }
        )
        for cell in shared_cells:
            low = convergence_groups[("shared", cell, 8, backend)]
            high = convergence_groups[("shared", cell, 12, backend)]
            n = len(low.positions)
            key = ("shared", backend, n)
            # Reuse the A/B physical metric builder by assigning the same
            # backend selection key to both cutoffs; this preserves pairing.
            specs.extend(
                _physical_ab_specs(
                    scope=f"cutoff|{backend}|{cell}|8_to_12",
                    group_a=low,
                    group_b=high,
                    key_a=key,
                    key_b=key,
                    tolerances=tolerances,
                )
            )
            density_spec = next(
                spec
                for spec in specs
                if spec.gate_id
                == f"cutoff|{backend}|{cell}|8_to_12|density"
            )
            density_spec.estimator = (
                lambda selection, lo=low, hi=high, k=key: _trace_distance(
                    _embed_density(
                        lo.density_mean(
                            None if selection is None else selection[k]
                        ),
                        8,
                        12,
                    ),
                    hi.density_mean(
                        None if selection is None else selection[k]
                    ),
                )
            )

        logical_actions = sorted(
            {
                str(row["action"])
                for row in convergence_rows
                if row["layer"] == "logical"
                and row["backend"] == backend
            }
        )
        for action in logical_actions:
            low_groups = {
                label: convergence_groups[
                    (
                        "logical",
                        f"logical|{label}|{action}",
                        8,
                        backend,
                    )
                ]
                for label in ("0", "1", "+", "-", "+i", "-i")
            }
            high_groups = {
                label: convergence_groups[
                    (
                        "logical",
                        f"logical|{label}|{action}",
                        12,
                        backend,
                    )
                ]
                for label in ("0", "1", "+", "-", "+i", "-i")
            }
            n = len(low_groups["0"].positions)
            key = ("logical", backend, n)
            specs.append(
                _metric(
                    gate_id=f"cutoff|{backend}|logical_ptm|{action}|8_to_12",
                    scope=f"cutoff|{backend}|logical_ptm|{action}|8_to_12",
                    metric="logical_ptm_entry_difference",
                    tolerance_name="maximum_logical_ptm_entry_difference",
                    tolerances=tolerances,
                    estimator=lambda selection, lo=low_groups, hi=high_groups, k=key: float(
                        np.max(
                            np.abs(
                                _ptm(lo, selection, k)
                                - _ptm(hi, selection, k)
                            )
                        )
                    ),
                    keys=(key,),
                    denominator=f"{backend} logical PTM cutoff 8-to-12",
                )
            )
            for label in ("0", "1", "+", "-", "+i", "-i"):
                low = low_groups[label]
                high = high_groups[label]
                specs.append(
                    _metric(
                        gate_id=f"cutoff|{backend}|logical|{label}|{action}|survival",
                        scope=f"cutoff|{backend}|logical|{label}|{action}",
                        metric="logical_survival_difference",
                        tolerance_name="maximum_logical_survival_difference",
                        tolerances=tolerances,
                        estimator=lambda selection, lo=low, hi=high, k=key: abs(
                            _mean(
                                _selected(lo, selection, k),
                                "logical_survival",
                            )
                            - _mean(
                                _selected(hi, selection, k),
                                "logical_survival",
                            )
                        ),
                        keys=(key,),
                        denominator=f"{backend} {label} {action} cutoff 8-to-12",
                    )
                )

        for scenario in ("step", "telegraph", "burst", "compound"):
            cell = f"fault|{scenario}"
            low = convergence_groups[("fault", cell, 8, backend)]
            high = convergence_groups[("fault", cell, 12, backend)]
            n = len(low.positions)
            key = ("fault", backend, n)
            specs.extend(
                [
                    _metric(
                        gate_id=f"cutoff|{backend}|{cell}|terminal_density",
                        scope=f"cutoff|{backend}|{cell}|8_to_12",
                        metric="short_trajectory_terminal_trace_distance",
                        tolerance_name="maximum_short_trajectory_terminal_trace_distance",
                        tolerances=tolerances,
                        estimator=lambda selection, lo=low, hi=high, k=key: _trace_distance(
                            _embed_density(
                                lo.density_mean(
                                    None if selection is None else selection[k],
                                    terminal_only=True,
                                ),
                                8,
                                12,
                            ),
                            hi.density_mean(
                                None if selection is None else selection[k],
                                terminal_only=True,
                            ),
                        ),
                        keys=(key,),
                        denominator=f"{backend} {scenario} cutoff 8-to-12",
                    ),
                    _metric(
                        gate_id=f"cutoff|{backend}|{cell}|terminal_observables",
                        scope=f"cutoff|{backend}|{cell}|8_to_12",
                        metric="short_trajectory_observable_mean_difference",
                        tolerance_name="maximum_short_trajectory_observable_mean_difference",
                        tolerances=tolerances,
                        estimator=lambda selection, lo=low, hi=high, k=key: float(
                            np.max(
                                np.abs(
                                    _terminal_observable_vector(
                                        _selected(lo, selection, k)
                                    )
                                    - _terminal_observable_vector(
                                        _selected(hi, selection, k)
                                    )
                                )
                            )
                        ),
                        keys=(key,),
                        denominator=f"{backend} {scenario} cutoff 8-to-12",
                    ),
                ]
            )
    return specs, source_rows


def apply_simultaneous_bootstrap(
    specs: Sequence[MetricSpec],
    procedure: Mapping[str, Any],
) -> dict[str, object]:
    stochastic = [spec for spec in specs if spec.stochastic]
    for spec in specs:
        spec.estimate = float(spec.estimator(None))
        if not np.isfinite(spec.estimate):
            raise ValueError(f"non-finite point estimate: {spec.gate_id}")
    if not stochastic:
        critical = 0.0
    else:
        resamples = int(procedure["bootstrap_resamples"])
        rng = np.random.default_rng(int(procedure["bootstrap_seed"]))
        selection_keys = sorted(
            {
                key
                for spec in stochastic
                for key in spec.selection_keys
            }
        )
        draws = {
            key: rng.integers(
                0,
                key[2],
                size=(resamples, key[2]),
                endpoint=False,
            )
            for key in selection_keys
        }
        replicates = np.empty((resamples, len(stochastic)), dtype=np.float64)
        for bootstrap_index in range(resamples):
            selections = {
                key: values[bootstrap_index]
                for key, values in draws.items()
            }
            for metric_index, spec in enumerate(stochastic):
                value = float(spec.estimator(selections))
                if not np.isfinite(value):
                    raise ValueError(
                        f"non-finite bootstrap estimate: {spec.gate_id}"
                    )
                replicates[bootstrap_index, metric_index] = value
        standard_errors = np.std(replicates, axis=0, ddof=1)
        floor = float(procedure["standard_error_floor"])
        standardized = np.abs(
            replicates
            - np.asarray([spec.estimate for spec in stochastic])[None, :]
        ) / np.maximum(standard_errors[None, :], floor)
        maxima = np.max(standardized, axis=1)
        critical = float(
            np.quantile(
                maxima,
                float(procedure["confidence"]),
                method="higher",
            )
        )
        for index, spec in enumerate(stochastic):
            spec.standard_error = float(standard_errors[index])
            spec.bound = spec.estimate + critical * max(
                spec.standard_error,
                floor,
            )
    for spec in specs:
        if not spec.stochastic:
            spec.standard_error = 0.0
            spec.bound = spec.estimate
        if spec.direction == "upper":
            spec.passed = spec.bound <= spec.tolerance
        elif spec.direction == "lower":
            spec.passed = spec.bound >= spec.tolerance
        else:
            raise ValueError("unknown gate direction")
    return {
        "resamples": int(procedure["bootstrap_resamples"]),
        "confidence": float(procedure["confidence"]),
        "critical_value": critical,
        "stochastic_metric_count": len(stochastic),
        "total_metric_count": len(specs),
    }


def _gate_rows(specs: Sequence[MetricSpec]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for spec in specs:
        margin = (
            spec.tolerance - spec.bound
            if spec.direction == "upper"
            else spec.bound - spec.tolerance
        )
        rows.append(
            {
                "record_type": "gate_metric",
                "gate_id": spec.gate_id,
                "scope": spec.scope,
                "metric": spec.metric,
                "tolerance_name": spec.tolerance_name,
                "direction": spec.direction,
                "estimate": spec.estimate,
                "standard_error": spec.standard_error,
                "simultaneous_bound": spec.bound,
                "tolerance": spec.tolerance,
                "margin": margin,
                "passed": spec.passed,
                "cutoff": "",
                "backend": "",
                "matrix_name": "",
                "row": "",
                "column": "",
                "real": "",
                "imag": "",
                "denominator": spec.denominator,
                "notes": "trace-decreasing/no-postselection where logical",
            }
        )
    return rows


def _markdown(report: Mapping[str, Any]) -> str:
    summary = report["gate_summary"]
    worst = report["worst_gate"]
    worst_margin = (
        "null"
        if worst["margin"] is None
        else f"{worst['margin']:.6g}"
    )
    critical = report["bootstrap"]["critical_value"]
    critical_text = "null" if critical is None else f"{critical:.6g}"
    return "\n".join(
        [
            "# T9.2.4 双后端资格对拍",
            "",
            f"- verdict：`{report['verdict']}`",
            f"- formal rows：{report['row_accounting']['observed']} / {report['row_accounting']['expected']}",
            f"- main gates：{summary['passed']} / {summary['total']}",
            f"- simultaneous bootstrap：{report['bootstrap']['resamples']} resamples，critical={critical_text}",
            f"- worst gate：`{worst['gate_id']}`，margin={worst_margin}",
            "- logical PTM 使用 survival×corrected-density 的 trace-decreasing block；无 postselection/条件均值救援。",
            "- cutoff 8/12 的 A/B 同截断 agreement 与 backend 内 cutoff movement 分开判门；captured mass 仅作强制诊断。",
            "- round LER、六态 lifetime、physical break-even、official/Puviani、external SOTA、真硬件与 rank 字段继续为 `null`。",
            "",
        ]
    )


def verify_and_finalize(root: Path) -> dict[str, Any]:
    seal, amendment = verify_lineage(root)
    artifacts = amendment["artifact_paths"]
    execution = _load_json(root / artifacts["execution_manifest"])
    if execution.get("formal") is not True:
        raise RuntimeError("execution manifest is not formal")
    if execution.get("seal_analysis_sha256") != seal["analysis_sha256"]:
        raise RuntimeError("execution/seal analysis mismatch")
    if execution.get("ledger") != _binding(root, artifacts["cell_ledger"]):
        raise RuntimeError("execution ledger binding mismatch")
    if execution.get("raw_state_archive") != _binding(
        root,
        artifacts["raw_state_archive"],
    ):
        raise RuntimeError("execution archive binding mismatch")
    rows = load_ledger(root / artifacts["cell_ledger"])
    archive, density_by_row = load_archive(
        root / artifacts["raw_state_archive"]
    )
    expected = int(
        amendment["cutoff_convergence_submatrix"][
            "total_unique_formal_backend_rounds_after_amendment"
        ]
    )
    row_ids = [str(row["row_id"]) for row in rows]
    exception_rows = [row for row in rows if row["exception_type"]]
    missing_density_rows = [
        row
        for row in rows
        if int(row["density_index"]) >= 0
        and str(row["row_id"]) not in density_by_row
    ]
    complete = (
        len(rows) == expected
        and len(set(row_ids)) == len(row_ids)
        and not exception_rows
        and not missing_density_rows
        and all(row["conservation_pass"] is True for row in rows)
    )
    source_rows: list[dict[str, object]] = []
    specs: list[MetricSpec] = []
    bootstrap: dict[str, object] = {
        "resamples": 0,
        "confidence": 0.95,
        "critical_value": None,
        "stochastic_metric_count": 0,
        "total_metric_count": 0,
    }
    infrastructure_errors: list[str] = []
    if not complete:
        infrastructure_errors.extend(
            [
                f"row_count={len(rows)} expected={expected}",
                f"duplicate_row_ids={len(row_ids)-len(set(row_ids))}",
                f"exception_rows={len(exception_rows)}",
                f"missing_density_rows={len(missing_density_rows)}",
                f"conservation_fail_rows={sum(row['conservation_pass'] is not True for row in rows)}",
            ]
        )
        verdict = INCOMPLETE_VERDICT
    else:
        try:
            specs, source_rows = build_metric_specs(
                rows,
                archive,
                density_by_row,
                amendment,
            )
            bootstrap = apply_simultaneous_bootstrap(
                specs,
                amendment["statistical_procedure"],
            )
            source_rows.extend(_gate_rows(specs))
            verdict = (
                PASS_VERDICT
                if all(spec.passed for spec in specs)
                else FAIL_VERDICT
            )
        except BaseException as exc:
            verdict = INCOMPLETE_VERDICT
            infrastructure_errors.append(
                f"{type(exc).__name__}: {str(exc)[:2000]}"
            )

    source_path = root / artifacts["source_data"]
    _atomic_csv(source_path, SOURCE_FIELDS, source_rows)
    gate_rows = [row for row in source_rows if row["record_type"] == "gate_metric"]
    passed_count = sum(row["passed"] is True for row in gate_rows)
    if gate_rows:
        worst_row = min(
            gate_rows,
            key=lambda row: float(row["margin"]),
        )
        worst = {
            "gate_id": worst_row["gate_id"],
            "margin": float(worst_row["margin"]),
            "bound": float(worst_row["simultaneous_bound"]),
            "tolerance": float(worst_row["tolerance"]),
        }
    else:
        worst = {
            "gate_id": "none_infrastructure_incomplete",
            "margin": None,
            "bound": None,
            "tolerance": None,
        }
    downstream = amendment["failure_and_release_policy"]
    if verdict == PASS_VERDICT:
        released = list(downstream["released_after_pass"])
        blocked: list[str] = []
    elif verdict == FAIL_VERDICT:
        released = list(downstream["released_after_scientific_fail"])
        blocked = list(downstream["blocked_after_scientific_fail"])
    else:
        released = list(downstream["released_after_infrastructure_fail"])
        blocked = sorted(
            set(
                downstream["blocked_after_scientific_fail"]
                + downstream["released_after_pass"]
            )
        )
    claim_state = {field: None for field in TYPED_NULL_FIELDS}
    report: dict[str, Any] = {
        "task_id": TASK_ID,
        "schema_version": "PHASE9-DUAL-BACKEND-QUALIFICATION-REPORT-V1",
        "verifier_id": VERIFIER_ID,
        "lineage": {
            "parent_preregistration_analysis_sha256": amendment[
                "parent_preregistration"
            ]["analysis_sha256"],
            "formal_child_seal_analysis_sha256": seal["analysis_sha256"],
            "execution_sha256": execution["execution_sha256"],
        },
        "row_accounting": {
            "expected": expected,
            "observed": len(rows),
            "unique_row_ids": len(set(row_ids)),
            "exception_rows": len(exception_rows),
            "conservation_pass_rows": sum(
                row["conservation_pass"] is True for row in rows
            ),
        },
        "mapping_diagnostics": {
            str(cutoff): {
                "captured_a": archive[
                    f"mapping_captured_a_cutoff_{cutoff}"
                ].tolist(),
                "captured_b": archive[
                    f"mapping_captured_b_cutoff_{cutoff}"
                ].tolist(),
            }
            for cutoff in (8, 12)
        },
        "bootstrap": bootstrap,
        "gate_summary": {
            "passed": passed_count,
            "failed": len(gate_rows) - passed_count,
            "total": len(gate_rows),
            "all_passed": bool(gate_rows)
            and passed_count == len(gate_rows),
        },
        "worst_gate": worst,
        "infrastructure_errors": infrastructure_errors,
        "failure_propagation": {
            "released_tasks": released,
            "blocked_tasks": blocked,
        },
        "claim_state": claim_state,
        "qualified_claim": (
            "dual_backend_agreement_for_prefrozen_synthetic_task"
            if verdict == PASS_VERDICT
            else None
        ),
        "verdict": verdict,
        "artifacts": {
            "execution_manifest": _binding(
                root,
                artifacts["execution_manifest"],
            ),
            "cell_ledger": _binding(root, artifacts["cell_ledger"]),
            "raw_state_archive": _binding(
                root,
                artifacts["raw_state_archive"],
            ),
            "source_data": _binding(root, artifacts["source_data"]),
        },
    }
    unsigned = dict(report)
    report["analysis_sha256"] = _sha_bytes(
        _canonical(unsigned).encode("utf-8")
    )
    report_path = root / artifacts["report"]
    _atomic_text(
        report_path,
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
    )
    markdown_path = root / artifacts["markdown"]
    _atomic_text(markdown_path, _markdown(report))
    release_pin = {
        "schema_version": "PHASE9-DUAL-BACKEND-QUALIFICATION-RELEASE-PIN-V1",
        "task_id": TASK_ID,
        "analysis_sha256": report["analysis_sha256"],
        "verdict": verdict,
        "lineage": {
            "parent_preregistration": _binding(
                root,
                amendment["parent_preregistration"]["path"],
            ),
            "formal_child_seal": _binding(
                root,
                artifacts["amendment_seal"],
            ),
        },
        "implementation": {
            "runner": _binding(root, artifacts["formal_runner"]),
            "verifier": _binding(root, artifacts["independent_verifier"]),
            "bridge": _binding(
                root,
                amendment["logical_bridge"]["implementation"],
            ),
            "runner_tests": _binding(root, artifacts["runner_tests"]),
            "verifier_tests": _binding(root, artifacts["verifier_tests"]),
        },
        "evidence": {
            "execution_manifest": _binding(
                root,
                artifacts["execution_manifest"],
            ),
            "cell_ledger": _binding(root, artifacts["cell_ledger"]),
            "raw_state_archive": _binding(
                root,
                artifacts["raw_state_archive"],
            ),
            "source_data": _binding(root, artifacts["source_data"]),
            "report": _binding(root, artifacts["report"]),
            "markdown": _binding(root, artifacts["markdown"]),
        },
        "claim_state": claim_state,
    }
    _atomic_text(
        root / artifacts["release_pin"],
        json.dumps(
            release_pin,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=_root())
    args = parser.parse_args(argv)
    report = verify_and_finalize(args.root.resolve())
    print(
        _canonical(
            {
                "task_id": TASK_ID,
                "verdict": report["verdict"],
                "analysis_sha256": report["analysis_sha256"],
            }
        )
    )
    return 0 if report["verdict"] != INCOMPLETE_VERDICT else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FAIL_VERDICT",
    "INCOMPLETE_VERDICT",
    "MetricSpec",
    "PASS_VERDICT",
    "RowGroup",
    "SOURCE_FIELDS",
    "SOURCE_SCHEMA",
    "TYPED_NULL_FIELDS",
    "VERIFIER_ID",
    "apply_simultaneous_bootstrap",
    "build_metric_specs",
    "load_archive",
    "load_ledger",
    "verify_and_finalize",
    "verify_lineage",
]
