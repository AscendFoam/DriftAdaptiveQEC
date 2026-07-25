"""Formal T9.2.4 dual-backend simulation and immutable raw-evidence writer.

The module performs physics execution only.  It never chooses thresholds and
does not issue a scientific verdict.  The independent
``phase9_dual_backend_verifier`` consumes the complete row ledger and raw state
archive, recomputes all frozen estimands, applies the pre-formal simultaneous
gate, and writes the report.

Formal execution is fail closed:

* the original preregistration and additive runner amendment must be
  byte-bound by the committed child seal;
* every expected row is emitted, including explicit exception rows;
* no formal interval may be shortened, expanded or substituted from the CLI;
* backend A/B use disjoint seed intervals and independent RNG implementations;
* the cutoff-12 confirmation rows reuse only the predeclared within-backend
  seed positions from cutoff 8.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import platform
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from physics.phase9_backend_a import (
    BACKEND_A_ID,
    BackendAConfig,
    BackendADriftState,
    Phase9BackendASimulator,
    backend_a_exogenous,
    diagnostic_action_word,
)
from physics.phase9_backend_b import (
    BACKEND_B_ID,
    BackendBConfig,
    BackendBDrift,
    backend_b_random_record,
    diagnostic_action_word_b,
)
from physics.phase9_backend_b_logical_bridge import (
    MehlerLogicalBridgeConfig,
    Phase9BackendBMehlerBridgeSimulator,
)
from physics.phase9_twin_contract import (
    ActionWord,
    NominalAction,
    execute_representative_probe,
    representative_action_probes,
)


TASK_ID = "T9.2.4"
RUNNER_ID = "PHASE9-DUAL-BACKEND-FORMAL-RUNNER-V1"
ROW_SCHEMA = "PHASE9-DUAL-BACKEND-ROUND-LEDGER-V1"
STATE_ARCHIVE_SCHEMA = "PHASE9-DUAL-BACKEND-STATE-ARCHIVE-V1"

LEDGER_FIELDS = (
    "row_id",
    "row_schema",
    "layer",
    "cell_base",
    "cell_id",
    "backend",
    "backend_id",
    "cutoff",
    "confirmation_cutoff",
    "convergence_member",
    "seed",
    "seed_position",
    "trajectory_id",
    "round_index",
    "terminal_round",
    "action",
    "probe_id",
    "scenario",
    "initial_state",
    "logical_label",
    "rng_namespace",
    "density_index",
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
    "reset_requested",
    "reset_hidden_success",
    "reset_ack",
    "leakage_resident",
    "leakage_age",
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
    "conservation_pass",
    "exception_type",
    "exception_message",
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


def verify_child_seal(
    root: Path,
) -> dict[str, Any]:
    """Verify the pre-formal child seal before importing an outcome path."""

    seal_path = root / "docs/t9_2_4_formal_runner_amendment_seal.json"
    if not seal_path.exists():
        raise RuntimeError("formal runner child seal is missing")
    seal = _load_json(seal_path)
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
    ):
        raise RuntimeError("formal runner child seal is invalid")
    unsigned = dict(seal)
    analysis = unsigned.pop("analysis_sha256", None)
    if analysis != _sha_bytes(_canonical(unsigned).encode("utf-8")):
        raise RuntimeError("formal child seal self-hash mismatch")
    expected = seal.get("live_bindings")
    if not isinstance(expected, dict):
        raise RuntimeError("formal child seal lacks live bindings")
    for name, binding in expected.items():
        if not isinstance(binding, dict) or "path" not in binding:
            raise RuntimeError(f"invalid child-seal binding {name}")
        if _binding(root, str(binding["path"])) != binding:
            raise RuntimeError(f"child-seal live binding drift: {name}")
    amendment = _load_json(
        root / "configs/phase9/t9_2_4_formal_runner_amendment.json"
    )
    artifacts = amendment["artifact_paths"]
    for output_name in (
        "execution_manifest",
        "report",
        "cell_ledger",
        "source_data",
        "raw_state_archive",
        "release_pin",
    ):
        if (root / artifacts[output_name]).exists():
            raise RuntimeError(
                f"formal output already exists before one-shot run: {output_name}"
            )
    return seal


def _common_config_kwargs(
    prereg: Mapping[str, Any],
    *,
    cutoff: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    common = dict(prereg["common_physics"])
    segment_steps = int(common.pop("segment_steps"))
    common["cutoff"] = int(cutoff)
    common["iq_centers"] = tuple(
        tuple(float(value) for value in row)
        for row in common["iq_centers"]
    )
    common["drift_retention"] = tuple(common["drift_retention"])
    common["drift_noise_std"] = tuple(common["drift_noise_std"])
    a_kwargs = dict(common)
    a_kwargs.update(
        {
            "substeps_per_segment": segment_steps,
            "logical_projector_delta": float(
                prereg["native_logical_mappings"]["backend_a"][
                    "projector_delta"
                ]
            ),
            "logical_grid_points": int(
                prereg["native_logical_mappings"]["backend_a"]["grid_points"]
            ),
        }
    )
    b_kwargs = dict(common)
    b_kwargs.update(
        {
            "split_steps_per_segment": segment_steps,
            "comb_squeezing": float(
                prereg["native_logical_mappings"]["backend_b"][
                    "comb_squeezing"
                ]
            ),
            "comb_envelope": float(
                prereg["native_logical_mappings"]["backend_b"][
                    "comb_envelope"
                ]
            ),
            "comb_half_width": int(
                prereg["native_logical_mappings"]["backend_b"][
                    "comb_half_width"
                ]
            ),
        }
    )
    return a_kwargs, b_kwargs


def build_simulators(
    prereg: Mapping[str, Any],
    amendment: Mapping[str, Any],
    cutoff: int,
) -> dict[str, object]:
    a_kwargs, b_kwargs = _common_config_kwargs(prereg, cutoff=cutoff)
    bridge_spec = amendment["logical_bridge"]
    bridge_config = MehlerLogicalBridgeConfig(
        projector_delta=float(bridge_spec["projector_delta"]),
        tail_tolerance=float(bridge_spec["tail_tolerance"]),
    )
    return {
        "A": Phase9BackendASimulator(BackendAConfig(**a_kwargs)),
        "B": Phase9BackendBMehlerBridgeSimulator(
            BackendBConfig(**b_kwargs),
            bridge_config=bridge_config,
        ),
    }


def _action_words() -> dict[str, ActionWord]:
    actions: dict[str, ActionWord] = {}
    for name in ("IDLE", "X", "Z", "XZ", "RESET", "HOLD", "LKG_HOLD"):
        from_a = diagnostic_action_word(name)
        from_b = diagnostic_action_word_b(name)
        if from_a.to_bytes() != from_b.to_bytes():
            raise RuntimeError(f"A/B action-word mismatch for {name}")
        if NominalAction(from_a.action_code).name != name:
            raise RuntimeError(f"action-word semantic mismatch for {name}")
        actions[name] = from_a
    return actions


def _initial_fock_ket(name: str, cutoff: int) -> tuple[np.ndarray, str]:
    ket = np.zeros(cutoff, dtype=np.complex128)
    if name in {"vacuum_g", "vacuum_e", "vacuum_f"}:
        ket[0] = 1.0
    elif name == "one_g":
        ket[1] = 1.0
    elif name == "zero_one_plus_g":
        ket[0] = 1.0 / np.sqrt(2.0)
        ket[1] = 1.0 / np.sqrt(2.0)
    else:
        raise ValueError(f"unknown shared initial state {name}")
    ancilla = {
        "vacuum_g": "g",
        "one_g": "g",
        "zero_one_plus_g": "g",
        "vacuum_e": "e",
        "vacuum_f": "f",
    }[name]
    return ket, ancilla


def _seed_for(
    prereg: Mapping[str, Any],
    backend: str,
    layer: str,
    position: int,
) -> int:
    if layer == "fault":
        key = (
            "trajectory_backend_a_seeds"
            if backend == "A"
            else "trajectory_backend_b_seeds"
        )
    else:
        key = (
            "formal_backend_a_seeds"
            if backend == "A"
            else "formal_backend_b_seeds"
        )
    split = prereg["splits"][key]
    if not 0 <= position < int(split["count"]):
        raise ValueError("seed position outside frozen interval")
    return int(split["start"]) + position


def _apply_intervention(
    state: object,
    backend: str,
    delta: Sequence[float],
) -> object:
    if backend == "A":
        vector = state.drift.vector() + np.asarray(delta, dtype=np.float64)
        return replace(state, drift=BackendADriftState.from_vector(vector))
    vector = state.drift.vector() + np.asarray(delta, dtype=np.float64)
    return replace(state, drift=BackendBDrift.from_vector(vector))


def _fault_delta_for_round(
    scenario: str,
    specification: Mapping[str, Any],
    round_index: int,
) -> np.ndarray:
    delta = np.asarray(specification["drift_delta"], dtype=np.float64)
    applied = np.zeros(5, dtype=np.float64)
    if scenario == "step":
        if round_index == int(specification["change_round"]):
            applied += delta
    elif scenario == "telegraph":
        period = int(specification["period"])
        if round_index % period == 0:
            block = round_index // period
            applied += delta if block % 2 == 0 else -delta
    elif scenario == "burst":
        if round_index == int(specification["start_round"]):
            applied += delta
        if round_index == (
            int(specification["start_round"])
            + int(specification["duration"])
        ):
            applied -= delta
    elif scenario == "compound":
        if round_index == int(specification["change_round"]):
            applied += delta
        if round_index == int(specification["burst_start"]):
            applied += delta
        if round_index == (
            int(specification["burst_start"])
            + int(specification["burst_duration"])
        ):
            applied -= delta
    else:
        raise ValueError(f"unknown fault scenario {scenario}")
    return applied


def _density_diagnostics(density: np.ndarray) -> tuple[float, float, float]:
    hermitian = 0.5 * (density + density.conj().T)
    return (
        abs(float(np.trace(density).real) - 1.0)
        + abs(float(np.trace(density).imag)),
        float(np.linalg.norm(density - density.conj().T, ord="fro")),
        float(np.min(np.linalg.eigvalsh(hermitian))),
    )


def _empty_row(**identity: object) -> dict[str, object]:
    row = {field: "" for field in LEDGER_FIELDS}
    row.update(identity)
    row["row_schema"] = ROW_SCHEMA
    row["density_index"] = -1
    row["confirmation_cutoff"] = identity.get(
        "confirmation_cutoff",
        int(identity.get("cutoff", 0)) == 12,
    )
    row["convergence_member"] = identity.get("convergence_member", False)
    row["terminal_round"] = identity.get("terminal_round", True)
    row["reset_requested"] = False
    row["reset_hidden_success"] = False
    row["leakage_resident"] = False
    row["conservation_pass"] = False
    row["exception_type"] = ""
    row["exception_message"] = ""
    return row


def _extract_row(
    *,
    identity: Mapping[str, object],
    backend: str,
    simulator: object,
    result: object,
    action: ActionWord,
    density_index: int,
) -> dict[str, object]:
    row = _empty_row(**identity)
    observation = result.observation
    truth = result.truth
    state = result.state
    density = np.asarray(state.joint_density, dtype=np.complex128)
    oscillator = simulator.oscillator_density(density)
    oscillator_matrix = (
        oscillator.matrix if hasattr(oscillator, "matrix") else oscillator
    )
    number = np.diag(np.arange(state.cutoff, dtype=np.float64))
    mean_photon = float(np.trace(number @ oscillator_matrix).real)
    levels = (
        truth.post_reset_level_probabilities
        if backend == "A"
        else truth.post_reset_levels
    )
    diagnostics = _density_diagnostics(density)
    posterior_error = abs(sum(observation.posterior_levels) - 1.0)
    level_error = abs(sum(levels) - 1.0)
    conservation = (
        np.all(np.isfinite(density))
        and diagnostics[0] <= 5.0e-8
        and diagnostics[1] <= 5.0e-8
        and diagnostics[2] >= -5.0e-8
        and posterior_error <= 5.0e-8
        and level_error <= 5.0e-8
    )
    hidden_reset = truth.reset_hidden_outcome
    row.update(
        {
            "density_index": density_index,
            "integrated_i": observation.integrated_i,
            "integrated_q": observation.integrated_q,
            "log_evidence": observation.log_evidence_density,
            "posterior_g": observation.posterior_levels[0],
            "posterior_e": observation.posterior_levels[1],
            "posterior_f": observation.posterior_levels[2],
            "level_g": levels[0],
            "level_e": levels[1],
            "level_f": levels[2],
            "mean_photon": mean_photon,
            "reset_requested": bool(action.reset_request),
            "reset_hidden_success": hidden_reset == "success",
            "reset_ack": observation.reset_ack,
            "leakage_resident": state.leakage_age > 0,
            "leakage_age": state.leakage_age,
            "drift_0": state.drift.vector()[0],
            "drift_1": state.drift.vector()[1],
            "drift_2": state.drift.vector()[2],
            "drift_3": state.drift.vector()[3],
            "drift_4": state.drift.vector()[4],
            "density_trace_error": diagnostics[0],
            "density_hermiticity_frobenius": diagnostics[1],
            "density_minimum_eigenvalue": diagnostics[2],
            "posterior_normalization_error": posterior_error,
            "level_normalization_error": level_error,
            "conservation_pass": conservation,
            "exception_type": "",
            "exception_message": "",
        }
    )
    logical = result.logical
    if logical is not None:
        survival = (
            logical.code_survival_probability
            if backend == "A"
            else logical.code_survival
        )
        corrected = (
            logical.frame_corrected_logical_density
            if backend == "A"
            else logical.corrected_density
        )
        block = survival * np.asarray(corrected, dtype=np.complex128)
        row.update(
            {
                "logical_survival": survival,
                "logical_block_00_real": block[0, 0].real,
                "logical_block_00_imag": block[0, 0].imag,
                "logical_block_01_real": block[0, 1].real,
                "logical_block_01_imag": block[0, 1].imag,
                "logical_block_10_real": block[1, 0].real,
                "logical_block_10_imag": block[1, 0].imag,
                "logical_block_11_real": block[1, 1].real,
                "logical_block_11_imag": block[1, 1].imag,
            }
        )
    return row


class EvidenceAccumulator:
    def __init__(self) -> None:
        self.rows: list[dict[str, object]] = []
        self.densities: dict[int, list[np.ndarray]] = {8: [], 12: []}
        self.density_row_ids: dict[int, list[str]] = {8: [], 12: []}
        self.mapping_arrays: dict[str, np.ndarray] = {}

    def append_success(
        self,
        *,
        identity: Mapping[str, object],
        backend: str,
        simulator: object,
        result: object,
        action: ActionWord,
        retain_density: bool,
    ) -> None:
        cutoff = int(identity["cutoff"])
        density_index = -1
        if retain_density:
            density_index = len(self.densities[cutoff])
            self.densities[cutoff].append(
                np.asarray(result.state.joint_density, dtype=np.complex128)
            )
            self.density_row_ids[cutoff].append(str(identity["row_id"]))
        self.rows.append(
            _extract_row(
                identity=identity,
                backend=backend,
                simulator=simulator,
                result=result,
                action=action,
                density_index=density_index,
            )
        )

    def append_exception(
        self,
        *,
        identity: Mapping[str, object],
        exception: BaseException,
    ) -> None:
        row = _empty_row(**identity)
        row["exception_type"] = type(exception).__name__
        row["exception_message"] = str(exception)[:1000]
        self.rows.append(row)

    def write_archive(self, path: Path) -> None:
        arrays: dict[str, np.ndarray] = {
            "schema": np.array([STATE_ARCHIVE_SCHEMA]),
        }
        arrays.update(self.mapping_arrays)
        for cutoff in (8, 12):
            dimension = cutoff * 3
            values = self.densities[cutoff]
            arrays[f"densities_cutoff_{cutoff}"] = (
                np.stack(values)
                if values
                else np.empty((0, dimension, dimension), dtype=np.complex128)
            )
            arrays[f"row_ids_cutoff_{cutoff}"] = np.asarray(
                self.density_row_ids[cutoff],
                dtype=f"<U{max([1] + [len(v) for v in self.density_row_ids[cutoff]])}",
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        temporary.replace(path)


def _identity(
    *,
    row_id: str,
    layer: str,
    cell_base: str,
    backend: str,
    cutoff: int,
    seed: int,
    seed_position: int,
    action: str,
    initial_state: str = "",
    logical_label: str = "",
    probe_id: str = "",
    scenario: str = "",
    trajectory_id: str = "",
    round_index: int = 0,
    terminal_round: bool = True,
    convergence_member: bool = False,
) -> dict[str, object]:
    return {
        "row_id": row_id,
        "layer": layer,
        "cell_base": cell_base,
        "cell_id": f"{cell_base}|cutoff={cutoff}",
        "backend": backend,
        "backend_id": BACKEND_A_ID if backend == "A" else BACKEND_B_ID,
        "cutoff": cutoff,
        "confirmation_cutoff": cutoff == 12,
        "convergence_member": convergence_member,
        "seed": seed,
        "seed_position": seed_position,
        "trajectory_id": trajectory_id,
        "round_index": round_index,
        "terminal_round": terminal_round,
        "action": action,
        "probe_id": probe_id,
        "scenario": scenario,
        "initial_state": initial_state,
        "logical_label": logical_label,
        "rng_namespace": (
            "NUMPY_SEEDSEQUENCE_ADDRESSED"
            if backend == "A"
            else "BLAKE2B_ADDRESS_PYTHON_RANDOM_BOX_MULLER"
        ),
    }


def _one_step(
    *,
    backend: str,
    simulator: object,
    state: object,
    evaluator: object | None,
    action: ActionWord,
    seed: int,
) -> object:
    if backend == "A":
        random_record = backend_a_exogenous(
            seed=seed,
            round_index=state.round_index,
            iq_samples=simulator.config.iq_samples,
        )
    else:
        random_record = backend_b_random_record(
            seed=seed,
            round_index=state.round_index,
            iq_samples=simulator.config.iq_samples,
        )
    return simulator.step(
        state,
        action,
        random_record,
        evaluator=evaluator,
    )


def _run_shared_layer(
    *,
    accumulator: EvidenceAccumulator,
    prereg: Mapping[str, Any],
    amendment: Mapping[str, Any],
    simulators: Mapping[int, Mapping[str, object]],
    actions: Mapping[str, ActionWord],
) -> None:
    matrix = prereg["formal_matrix"]
    convergence = amendment["cutoff_convergence_submatrix"]
    primary_count = int(matrix["samples_per_shared_state_action_backend"])
    confirmation_count = int(
        convergence["samples_per_state_action_backend_cutoff"]
    )
    for cutoff in (8, 12):
        state_names = (
            matrix["shared_fock_states"]
            if cutoff == 8
            else convergence["shared_states"]
        )
        action_names = (
            matrix["unique_nominal_actions"]
            if cutoff == 8
            else convergence["actions"]
        )
        count = primary_count if cutoff == 8 else confirmation_count
        for state_name in state_names:
            for action_name in action_names:
                cell_base = f"shared|{state_name}|{action_name}"
                for backend in ("A", "B"):
                    simulator = simulators[cutoff][backend]
                    for position in range(count):
                        seed = _seed_for(prereg, backend, "shared", position)
                        row_id = (
                            f"shared|c{cutoff}|{state_name}|{action_name}|"
                            f"{backend}|p{position:03d}"
                        )
                        identity = _identity(
                            row_id=row_id,
                            layer="shared",
                            cell_base=cell_base,
                            backend=backend,
                            cutoff=cutoff,
                            seed=seed,
                            seed_position=position,
                            action=action_name,
                            initial_state=state_name,
                            convergence_member=(
                                action_name in convergence["actions"]
                                and position < confirmation_count
                            ),
                        )
                        try:
                            ket, ancilla = _initial_fock_ket(
                                state_name,
                                cutoff,
                            )
                            state = simulator.initialize_fock(
                                oscillator_ket=ket,
                                ancilla_state=ancilla,
                            )
                            result = _one_step(
                                backend=backend,
                                simulator=simulator,
                                state=state,
                                evaluator=None,
                                action=actions[action_name],
                                seed=seed,
                            )
                            accumulator.append_success(
                                identity=identity,
                                backend=backend,
                                simulator=simulator,
                                result=result,
                                action=actions[action_name],
                                retain_density=True,
                            )
                        except BaseException as exc:
                            accumulator.append_exception(
                                identity=identity,
                                exception=exc,
                            )


def _run_probe_layer(
    *,
    accumulator: EvidenceAccumulator,
    prereg: Mapping[str, Any],
    simulators: Mapping[int, Mapping[str, object]],
) -> None:
    count = int(
        prereg["formal_matrix"]["samples_per_representative_probe_backend"]
    )
    cutoff = int(prereg["common_physics"]["cutoff"])
    probes = representative_action_probes()
    expected_ids = tuple(prereg["action_contract"]["probe_ids"])
    if tuple(probe.probe_id for probe in probes) != expected_ids:
        raise RuntimeError("live representative probe order drift")
    for probe in probes:
        receipts = execute_representative_probe(probe)
        terminal = receipts[-1]
        action = terminal.recurrence.action_word
        action_name = NominalAction(action.action_code).name
        if action_name != probe.expected_terminal:
            raise RuntimeError(f"probe terminal mismatch: {probe.probe_id}")
        cell_base = f"probe|{probe.probe_id}|{action_name}"
        for backend in ("A", "B"):
            simulator = simulators[cutoff][backend]
            for position in range(count):
                seed = _seed_for(prereg, backend, "probe", position)
                row_id = (
                    f"probe|c{cutoff}|{probe.probe_id}|{backend}|"
                    f"p{position:03d}"
                )
                identity = _identity(
                    row_id=row_id,
                    layer="probe",
                    cell_base=cell_base,
                    backend=backend,
                    cutoff=cutoff,
                    seed=seed,
                    seed_position=position,
                    action=action_name,
                    probe_id=probe.probe_id,
                    initial_state="vacuum_g",
                )
                try:
                    state = simulator.initialize_fock()
                    result = _one_step(
                        backend=backend,
                        simulator=simulator,
                        state=state,
                        evaluator=None,
                        action=action,
                        seed=seed,
                    )
                    accumulator.append_success(
                        identity=identity,
                        backend=backend,
                        simulator=simulator,
                        result=result,
                        action=action,
                        retain_density=True,
                    )
                except BaseException as exc:
                    accumulator.append_exception(
                        identity=identity,
                        exception=exc,
                    )


def _run_logical_layer(
    *,
    accumulator: EvidenceAccumulator,
    prereg: Mapping[str, Any],
    amendment: Mapping[str, Any],
    simulators: Mapping[int, Mapping[str, object]],
    actions: Mapping[str, ActionWord],
) -> None:
    matrix = prereg["formal_matrix"]
    convergence = amendment["cutoff_convergence_submatrix"]
    primary_count = int(matrix["samples_per_logical_state_action_backend"])
    confirmation_count = int(
        convergence["samples_per_state_action_backend_cutoff"]
    )
    for cutoff in (8, 12):
        labels = (
            matrix["logical_labels"]
            if cutoff == 8
            else convergence["logical_states"]
        )
        action_names = (
            matrix["unique_nominal_actions"]
            if cutoff == 8
            else convergence["actions"]
        )
        count = primary_count if cutoff == 8 else confirmation_count
        for label in labels:
            for action_name in action_names:
                cell_base = f"logical|{label}|{action_name}"
                for backend in ("A", "B"):
                    simulator = simulators[cutoff][backend]
                    for position in range(count):
                        seed = _seed_for(prereg, backend, "logical", position)
                        row_id = (
                            f"logical|c{cutoff}|{label}|{action_name}|"
                            f"{backend}|p{position:03d}"
                        )
                        identity = _identity(
                            row_id=row_id,
                            layer="logical",
                            cell_base=cell_base,
                            backend=backend,
                            cutoff=cutoff,
                            seed=seed,
                            seed_position=position,
                            action=action_name,
                            logical_label=label,
                            convergence_member=(
                                action_name in convergence["actions"]
                                and position < confirmation_count
                            ),
                        )
                        try:
                            state, evaluator = simulator.initialize_logical(
                                label
                            )
                            result = _one_step(
                                backend=backend,
                                simulator=simulator,
                                state=state,
                                evaluator=evaluator,
                                action=actions[action_name],
                                seed=seed,
                            )
                            accumulator.append_success(
                                identity=identity,
                                backend=backend,
                                simulator=simulator,
                                result=result,
                                action=actions[action_name],
                                retain_density=True,
                            )
                        except BaseException as exc:
                            accumulator.append_exception(
                                identity=identity,
                                exception=exc,
                            )


def _run_fault_layer(
    *,
    accumulator: EvidenceAccumulator,
    prereg: Mapping[str, Any],
    amendment: Mapping[str, Any],
    simulators: Mapping[int, Mapping[str, object]],
    actions: Mapping[str, ActionWord],
) -> None:
    matrix = prereg["formal_matrix"]
    convergence = amendment["cutoff_convergence_submatrix"]
    sequences = amendment["action_and_fault_semantics"][
        "fault_action_sequences"
    ]
    primary_count = int(matrix["trajectories_per_fault_backend"])
    confirmation_count = int(
        convergence["trajectories_per_fault_backend_cutoff"]
    )
    for cutoff in (8, 12):
        scenarios = (
            matrix["fault_scenarios"].keys()
            if cutoff == 8
            else convergence["fault_scenarios"]
        )
        count = primary_count if cutoff == 8 else confirmation_count
        for scenario in scenarios:
            specification = matrix["fault_scenarios"][scenario]
            horizon = int(specification["horizon"])
            sequence = list(sequences[scenario])
            cell_base = f"fault|{scenario}"
            for backend in ("A", "B"):
                simulator = simulators[cutoff][backend]
                for position in range(count):
                    seed = _seed_for(prereg, backend, "fault", position)
                    trajectory_id = (
                        f"fault|c{cutoff}|{scenario}|{backend}|"
                        f"p{position:03d}"
                    )
                    try:
                        state = simulator.initialize_fock()
                        evaluator = None
                        failed: BaseException | None = None
                        for round_index in range(horizon):
                            action_name = sequence[round_index % len(sequence)]
                            identity = _identity(
                                row_id=f"{trajectory_id}|r{round_index:03d}",
                                layer="fault",
                                cell_base=cell_base,
                                backend=backend,
                                cutoff=cutoff,
                                seed=seed,
                                seed_position=position,
                                action=action_name,
                                scenario=scenario,
                                trajectory_id=trajectory_id,
                                round_index=round_index,
                                terminal_round=round_index == horizon - 1,
                                convergence_member=position < confirmation_count,
                            )
                            if failed is not None:
                                accumulator.append_exception(
                                    identity=identity,
                                    exception=failed,
                                )
                                continue
                            try:
                                delta = _fault_delta_for_round(
                                    scenario,
                                    specification,
                                    round_index,
                                )
                                if np.any(delta):
                                    state = _apply_intervention(
                                        state,
                                        backend,
                                        delta,
                                    )
                                result = _one_step(
                                    backend=backend,
                                    simulator=simulator,
                                    state=state,
                                    evaluator=evaluator,
                                    action=actions[action_name],
                                    seed=seed,
                                )
                                accumulator.append_success(
                                    identity=identity,
                                    backend=backend,
                                    simulator=simulator,
                                    result=result,
                                    action=actions[action_name],
                                    retain_density=round_index == horizon - 1,
                                )
                                state = result.state
                            except BaseException as exc:
                                failed = exc
                                accumulator.append_exception(
                                    identity=identity,
                                    exception=exc,
                                )
                    except BaseException as exc:
                        for round_index in range(horizon):
                            action_name = sequence[round_index % len(sequence)]
                            identity = _identity(
                                row_id=f"{trajectory_id}|r{round_index:03d}",
                                layer="fault",
                                cell_base=cell_base,
                                backend=backend,
                                cutoff=cutoff,
                                seed=seed,
                                seed_position=position,
                                action=action_name,
                                scenario=scenario,
                                trajectory_id=trajectory_id,
                                round_index=round_index,
                                terminal_round=round_index == horizon - 1,
                                convergence_member=position < confirmation_count,
                            )
                            accumulator.append_exception(
                                identity=identity,
                                exception=exc,
                            )


def expected_row_count(
    prereg: Mapping[str, Any],
    amendment: Mapping[str, Any],
) -> int:
    return int(
        amendment["cutoff_convergence_submatrix"][
            "total_unique_formal_backend_rounds_after_amendment"
        ]
    )


def execute_matrix(
    prereg: Mapping[str, Any],
    amendment: Mapping[str, Any],
) -> EvidenceAccumulator:
    simulators = {
        cutoff: build_simulators(prereg, amendment, cutoff)
        for cutoff in (8, 12)
    }
    actions = _action_words()
    accumulator = EvidenceAccumulator()
    for cutoff in (8, 12):
        simulator_a = simulators[cutoff]["A"]
        simulator_b = simulators[cutoff]["B"]
        isometry_a = simulator_a._logical_engine().code_basis.isometry
        isometry_b = simulator_b._comb_isometry()
        accumulator.mapping_arrays.update(
            {
                f"mapping_isometry_a_cutoff_{cutoff}": np.asarray(
                    isometry_a,
                    dtype=np.complex128,
                ),
                f"mapping_isometry_b_cutoff_{cutoff}": np.asarray(
                    isometry_b,
                    dtype=np.complex128,
                ),
                f"mapping_projector_a_cutoff_{cutoff}": np.asarray(
                    isometry_a @ isometry_a.conj().T,
                    dtype=np.complex128,
                ),
                f"mapping_projector_b_cutoff_{cutoff}": np.asarray(
                    isometry_b @ isometry_b.conj().T,
                    dtype=np.complex128,
                ),
                f"mapping_captured_a_cutoff_{cutoff}": np.asarray(
                    simulator_a._logical_engine().code_basis.captured_probabilities,
                    dtype=np.float64,
                ),
                f"mapping_captured_b_cutoff_{cutoff}": np.asarray(
                    simulator_b.logical_bridge_diagnostics()[
                        "captured_probabilities"
                    ],
                    dtype=np.float64,
                ),
            }
        )
    _run_shared_layer(
        accumulator=accumulator,
        prereg=prereg,
        amendment=amendment,
        simulators=simulators,
        actions=actions,
    )
    _run_probe_layer(
        accumulator=accumulator,
        prereg=prereg,
        simulators=simulators,
    )
    _run_logical_layer(
        accumulator=accumulator,
        prereg=prereg,
        amendment=amendment,
        simulators=simulators,
        actions=actions,
    )
    _run_fault_layer(
        accumulator=accumulator,
        prereg=prereg,
        amendment=amendment,
        simulators=simulators,
        actions=actions,
    )
    expected = expected_row_count(prereg, amendment)
    if len(accumulator.rows) != expected:
        raise RuntimeError(
            f"runner row-accounting defect: {len(accumulator.rows)} != {expected}"
        )
    row_ids = [str(row["row_id"]) for row in accumulator.rows]
    if len(set(row_ids)) != len(row_ids):
        raise RuntimeError("runner produced duplicate row_id values")
    return accumulator


def run_raw_formal(root: Path) -> dict[str, object]:
    seal = verify_child_seal(root)
    prereg = _load_json(
        root / "configs/phase9/t9_2_4_twin_qualification.json"
    )
    amendment = _load_json(
        root / "configs/phase9/t9_2_4_formal_runner_amendment.json"
    )
    accumulator = execute_matrix(prereg, amendment)
    expected = expected_row_count(prereg, amendment)

    artifacts = amendment["artifact_paths"]
    ledger_path = root / artifacts["cell_ledger"]
    archive_path = root / artifacts["raw_state_archive"]
    _atomic_csv(ledger_path, LEDGER_FIELDS, accumulator.rows)
    accumulator.write_archive(archive_path)
    execution = {
        "task_id": TASK_ID,
        "runner_id": RUNNER_ID,
        "formal": True,
        "seal_analysis_sha256": seal.get("analysis_sha256"),
        "expected_rows": expected,
        "observed_rows": len(accumulator.rows),
        "exception_rows": sum(
            bool(row["exception_type"]) for row in accumulator.rows
        ),
        "conservation_pass_rows": sum(
            row["conservation_pass"] is True for row in accumulator.rows
        ),
        "ledger": _binding(root, artifacts["cell_ledger"]),
        "raw_state_archive": _binding(root, artifacts["raw_state_archive"]),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
    }
    execution["execution_sha256"] = _sha_bytes(
        _canonical(execution).encode("utf-8")
    )
    _atomic_text(
        root / artifacts["execution_manifest"],
        json.dumps(
            execution,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
    )
    return execution


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=_root(),
        help="repository root",
    )
    args = parser.parse_args(argv)
    execution = run_raw_formal(args.root.resolve())
    print(_canonical(execution))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "LEDGER_FIELDS",
    "ROW_SCHEMA",
    "RUNNER_ID",
    "STATE_ARCHIVE_SCHEMA",
    "EvidenceAccumulator",
    "build_simulators",
    "execute_matrix",
    "expected_row_count",
    "run_raw_formal",
    "verify_child_seal",
]
