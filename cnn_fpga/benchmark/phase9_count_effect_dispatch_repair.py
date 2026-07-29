"""Analysis-only repair transaction for the T06 confirmation-effect dispatcher.

This program is deliberately narrower than the T06 writer.  It proves that
the archived V1 outcome and every raw density chunk are unchanged, invokes the
sealed categorical-dispatch repair on the same data, and permits changes in
exactly six derived summary rows.  It does not run the independent verifier.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from cnn_fpga.benchmark import phase9_count_selection_confirmation as writer


TASK_ID = "T-RISK-20260730-01"
SCHEMA = "PHASE9-T06-EFFECT-DISPATCH-REPAIR-REPORT-V1"
CONFIG_PATH = Path(
    "configs/phase9/t_risk_20260730_01_effect_dispatch_repair.json"
)
EXPECTED_V1_VERDICT = "TERMINAL_NO_GO_COUNT_SELECTION_OR_CONFIRMATION"
EXPECTED_V2_VERDICT = "PASS_COUNT_SELECTED_AND_UNTOUCHED_CONFIRMED"


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_sha(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _self_hash(value: Mapping[str, Any]) -> str:
    return _json_sha({
        key: item for key, item in value.items() if key != "analysis_sha256"
    })


def _binding(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.resolve().relative_to(root.resolve()).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _file_sha(path),
    }


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def _assert_file_binding(root: Path, binding: Mapping[str, Any]) -> None:
    path = root / str(binding["path"])
    if (
        not path.is_file()
        or path.stat().st_size != int(binding["bytes"])
        or _file_sha(path) != str(binding["sha256"])
    ):
        raise ValueError(f"archive binding drift: {binding['path']}")


def _full_archive_manifest(directory: Path) -> tuple[list[dict[str, Any]], str]:
    entries = []
    paths = sorted(
        (path for path in directory.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(directory).as_posix(),
    )
    for path in paths:
        entries.append({
            "path": path.relative_to(directory).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _file_sha(path),
        })
    payload = json.dumps(
        entries, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return entries, hashlib.sha256(payload).hexdigest()


def _expected_runtime_contract() -> dict[str, Any]:
    return {
        "required_process_environment_before_python_start": {
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        },
        "repair_runner_must_fail_before_writer_on_mismatch": True,
        "maxT_numeric_tolerance_substitution": False,
        "expected_single_thread_maxT_string_differences_vs_v1": 0,
    }


def _validate_runtime_environment(config: Mapping[str, Any]) -> None:
    contract = config.get("runtime_contract")
    if contract != _expected_runtime_contract():
        raise ValueError("repair runtime contract drift")
    required = contract["required_process_environment_before_python_start"]
    drift = {
        name: os.environ.get(name)
        for name, expected in required.items()
        if os.environ.get(name) != expected
    }
    if drift:
        raise RuntimeError(
            f"single-thread BLAS environment not frozen before Python: {drift}"
        )


def validate_config(config: Mapping[str, Any], root: Path) -> None:
    if (
        config.get("task_id") != TASK_ID
        or config.get("schema_version")
        != "PHASE9-T06-EFFECT-DISPATCH-REPAIR-CONFIG-V1"
        or config.get("parent_task") != writer.TASK_ID
    ):
        raise ValueError("repair config identity drift")
    if config.get("runtime_contract") != _expected_runtime_contract():
        raise ValueError("repair runtime contract drift")

    contract = config["repair_contract"]
    expected_contract = {
        "registered_effects": [0.0, 0.05, 0.1, 0.12],
        "absolute_tolerance": 1e-12,
        "identity_source": "frozen cell_id suffix",
        "computed_value_check": (
            "every row in a cell must map uniquely to the same frozen label"
        ),
        "dispatch": {
            "0.000": "null_equivalence_lcb_minimum",
            "0.050": "local_equivalence_lcb_minimum",
            "0.100": "boundary_equivalence_ucb_maximum",
            "0.120": "outside_equivalence_ucb_maximum",
        },
        "new_random_trials": False,
        "raw_chunk_rewrite": False,
        "seed_change": False,
        "count_change": False,
        "factor_change": False,
        "multiplier_replicate_change": False,
        "quantile_change": False,
        "margin_change": False,
        "gate_change": False,
        "correlation_change": False,
        "candidate_change": False,
        "threshold_change": False,
    }
    if contract != expected_contract:
        raise ValueError("repair scope widened or dispatch contract drifted")

    expected = config["expected_invariants"]
    if (
        expected["selection_chunk_count"] != 8
        or expected["confirmation_chunk_count"] != 24
        or expected["selection_density_trials"] != 1024
        or expected["confirmation_density_trials"] != 6144
        or expected["total_density_trials"] != 7168
        or expected["source_row_count"] != 8404
        or expected["permitted_changed_summary_cells"] != 6
        or expected["permitted_changed_cell_ids"]
        != [
            "confirmation__n768__d120__heavy_tail_rare_coherent__effect_0.050",
            "confirmation__n768__d120__heteroskedastic_coherent__effect_0.050",
            "confirmation__n768__d120__low_energy_balanced__effect_0.050",
            "confirmation__n768__d132__heavy_tail_rare_coherent__effect_0.050",
            "confirmation__n768__d132__heteroskedastic_coherent__effect_0.050",
            "confirmation__n768__d132__low_energy_balanced__effect_0.050",
        ]
        or expected["permitted_changed_summary_fields"]
        != ["power_gate_pass", "gate_pass"]
        or expected["independent_full_verifier_required_after_reanalysis"]
        is not True
    ):
        raise ValueError("repair invariant denominator drift")

    claims = config["claim_boundary"]
    if (
        claims["repair_is_scientific_confirmation"] is not False
        or claims["t04_preregistration_final_release"] is not False
        or claims["t04_scientific_execution_released"] is not False
        or any(
            claims[name] is not None
            for name in (
                "twin_qualification",
                "ler",
                "lifetime",
                "physical_break_even",
                "official_puviani_exact",
                "puviani_nmf_surpass",
                "external_sota",
                "hardware_measured",
            )
        )
    ):
        raise ValueError("repair claim boundary drift")

    for name in ("writer_report", "source_data", "selected_blueprint"):
        _assert_file_binding(root, config["v1_archive"][name])
    failure = config["failed_reanalysis_attempt_v1"]
    if (
        failure["classification"]
        != "runtime reproducibility failure; no scientific vote"
        or failure["parent_seal_commit"]
        != "2260a71233de199e4c91fdef89982fab1c3278b7"
        or failure["changed_rows"] != 716
        or failure["unexpected_maxT_serialization_changes"] != 710
        or failure["maximum_numeric_delta"] != 3.552713678800501e-15
        or failure["raw_chunk_rewrite"] is not False
        or failure["new_random_trials"] is not False
        or failure["repair_report_written"] is not False
        or failure["live_v1_restored"] is not True
    ):
        raise ValueError("failed reanalysis archive contract drift")
    _assert_file_binding(root, failure["manifest"])
    failure_manifest = _load_json(root / failure["manifest"]["path"])
    if (
        failure_manifest.get("schema_version")
        != "PHASE9-T06-REPAIR-ATTEMPT-FAILURE-MANIFEST-V1"
        or failure_manifest.get("analysis_sha256")
        != _self_hash(failure_manifest)
    ):
        raise ValueError("failed reanalysis manifest self-seal drift")

    archive = root / config["v1_archive"]["local_run_archive"]
    entries, manifest_sha = _full_archive_manifest(archive)
    if (
        len(entries) != config["v1_archive"]["local_archive_file_count"]
        or sum(entry["bytes"] for entry in entries)
        != config["v1_archive"]["local_archive_bytes"]
        or manifest_sha
        != config["v1_archive"]["local_archive_manifest_sha256"]
    ):
        raise ValueError("local full V1 archive drift")
    manifest = _load_json(
        root
        / config["v1_archive"]["tracked_directory"]
        / "manifest.json"
    )
    if (
        manifest.get("task_id") != TASK_ID
        or manifest.get("schema_version")
        != "PHASE9-T06-V1-FAILURE-ARCHIVE-MANIFEST-V1"
        or manifest["local_full_archive"].get("files") != entries
        or manifest["local_full_archive"].get("canonical_serialization")
        != {
            "entry_order": "ascending POSIX relative path",
            "entry_property_order": ["path", "bytes", "sha256"],
            "json": (
                "UTF-8 compact JSON with ensure_ascii=false and separators "
                "comma/colon; no sort_keys"
            ),
            "digest": "SHA-256 of the serialized array",
        }
        or manifest.get("analysis_sha256") != _self_hash(manifest)
    ):
        raise ValueError("tracked V1 manifest drift")

    source = config["source_seal"]
    paths = config["artifact_paths"]
    if (
        _file_sha(root / paths["t06_config"])
        != source["t06_config_sha256"]
        or _file_sha(root / paths["t06_writer"])
        != source["repaired_writer_sha256"]
        or _file_sha(root / paths["t06_writer_test"])
        != source["repaired_writer_test_sha256"]
        or _file_sha(root / paths["repair_runner"])
        != source["repair_runner_sha256"]
        or _file_sha(root / paths["repair_test"])
        != source["repair_test_sha256"]
        or _file_sha(root / paths["t05_algorithm"])
        != source["t05_algorithm_sha256"]
        or _file_sha(root / paths["t06_verifier"])
        != source["independent_verifier_sha256"]
    ):
        raise ValueError("sealed T06 source drift")


def _chunk_inventory(
    root: Path,
    config: Mapping[str, Any],
    t06_config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    expected = config["expected_invariants"]
    run_directory = root / t06_config["artifact_paths"]["run_directory"]
    archive = root / config["v1_archive"]["local_run_archive"] / "run"
    inventory: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    config_analysis_sha = _json_sha(t06_config)
    t05_config = _load_json(
        root / t06_config["parent_artifacts"]["t05_config"]
    )
    selected = next(
        candidate
        for candidate in t06_config["linked_count_grid"]
        if float(candidate["scale"]) == expected["selected_scale"]
    )

    for split, expected_files, expected_rows in (
        (
            "selection",
            expected["selection_chunk_count"],
            expected["selection_density_trials"],
        ),
        (
            "confirmation",
            expected["confirmation_chunk_count"],
            expected["confirmation_density_trials"],
        ),
    ):
        live_directory = run_directory / f"{split}_chunks"
        archive_directory = archive / f"{split}_chunks"
        expected_specs = writer._density_specs(
            t06_config,
            t05_config,
            split=split,
            selected=selected if split == "confirmation" else None,
        )
        expected_by_name = {
            f"{spec['cell_id']}.json": spec for spec in expected_specs
        }
        live_entries = sorted(live_directory.iterdir())
        archive_entries = sorted(archive_directory.iterdir())
        if (
            any(not path.is_file() for path in live_entries + archive_entries)
            or {path.name for path in live_entries} != set(expected_by_name)
            or {path.name for path in archive_entries} != set(expected_by_name)
        ):
            raise ValueError(f"{split} chunk filename or extra-entry drift")
        live_paths = [path for path in live_entries if path.is_file()]
        if len(live_paths) != expected_files:
            raise ValueError(f"{split} chunk count drift")
        split_rows = 0
        for live_path in live_paths:
            archived_path = archive_directory / live_path.name
            if (
                not archived_path.is_file()
                or live_path.stat().st_size != archived_path.stat().st_size
                or _file_sha(live_path) != _file_sha(archived_path)
            ):
                raise ValueError(f"{split} live chunk differs from V1 archive")
            chunk = _load_json(live_path)
            records = chunk.get("records")
            spec = chunk.get("spec")
            if (
                chunk.get("task_id") != writer.TASK_ID
                or chunk.get("schema_version")
                != "PHASE9-T06-DENSITY-CHUNK-V1"
                or chunk.get("config_analysis_sha256") != config_analysis_sha
                or not isinstance(spec, dict)
                or spec != expected_by_name[live_path.name]
                or not isinstance(records, list)
                or chunk.get("record_count") != len(records)
                or spec.get("trials") != len(records)
                or chunk.get("analysis_sha256") != _self_hash(chunk)
            ):
                raise ValueError(f"{split} chunk internal seal drift")
            trials = int(spec["trials"])
            if {int(row.get("trial", -1)) for row in records} != set(
                range(trials)
            ):
                raise ValueError(f"{split} trial index drift")
            for row in records:
                trial = int(row["trial"])
                address = int(spec["cell_index"]) * trials + trial
                numeric = (
                    row.get("true_distance"),
                    row.get("estimate"),
                    row.get("raw_radius"),
                    row.get("upper_bound"),
                )
                if (
                    row.get("row_type") != "density_trial"
                    or row.get("split") != split
                    or row.get("candidate_scale") != spec["scale"]
                    or row.get("cluster_count") != spec["cluster_count"]
                    or row.get("cell_id") != spec["cell_id"]
                    or row.get("family") != spec["family"]
                    or row.get("dimension") != spec["dimension"]
                    or row.get("trial_seed")
                    != int(spec["trial_seed_base"]) + address
                    or row.get("multiplier_seed")
                    != int(spec["multiplier_seed_base"]) + address
                    or any(
                        not isinstance(value, (int, float))
                        or isinstance(value, bool)
                        or not math.isfinite(float(value))
                        for value in numeric
                    )
                    or type(row.get("covered")) is not bool
                    or type(row.get("equivalence_pass")) is not bool
                    or row["covered"]
                    is not (
                        float(row["upper_bound"]) + 1e-15
                        >= float(row["true_distance"])
                    )
                    or row["equivalence_pass"]
                    is not (
                        float(row["upper_bound"]) <= float(spec["margin"])
                    )
                ):
                    raise ValueError(f"{split} chunk row semantic drift")
            if any(
                writer._registered_effect(
                    float(row["true_distance"]),
                    [float(spec["effect"])],
                )
                != float(spec["effect"])
                for row in records
            ):
                raise ValueError(f"{split} chunk row identity drift")
            split_rows += len(records)
            raw_rows.extend(records)
            inventory.append({
                "split": split,
                "path": live_path.relative_to(root).as_posix(),
                "bytes": live_path.stat().st_size,
                "sha256": _file_sha(live_path),
                "rows": len(records),
            })
        if split_rows != expected_rows:
            raise ValueError(f"{split} raw denominator drift")

    seeds = [int(row["trial_seed"]) for row in raw_rows]
    if (
        len(raw_rows) != expected["total_density_trials"]
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("raw density seed denominator or injectivity drift")
    return inventory, raw_rows


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"missing CSV header: {path}")
        return list(reader.fieldnames), list(reader)


def _csv_scalar(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _verify_csv_raw_matches_chunks(
    path: Path,
    raw_rows: Sequence[Mapping[str, Any]],
) -> None:
    header, rows = _read_csv(path)
    observed = [
        row for row in rows if row.get("row_type") == "density_trial"
    ]
    if len(observed) != len(raw_rows):
        raise ValueError("V1 source CSV density denominator differs from chunks")
    observed_by_key = {
        (row["split"], row["cell_id"], int(row["trial"])): row
        for row in observed
    }
    if len(observed_by_key) != len(observed):
        raise ValueError("V1 source CSV has duplicate density row identities")
    for raw in raw_rows:
        key = (str(raw["split"]), str(raw["cell_id"]), int(raw["trial"]))
        row = observed_by_key.get(key)
        if row is None:
            raise ValueError("V1 source CSV is missing a chunk density row")
        if any(
            row[name] != _csv_scalar(raw[name])
            for name in raw
        ) or any(
            row[name] != ""
            for name in header
            if name not in raw
        ):
            raise ValueError("V1 source CSV density row differs from chunk")


def _expected_local_effect_cell_ids(
    config: Mapping[str, Any],
) -> set[str]:
    return set(config["expected_invariants"]["permitted_changed_cell_ids"])


def _compare_source_data(
    old_path: Path,
    new_path: Path,
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    old_header, old_rows = _read_csv(old_path)
    new_header, new_rows = _read_csv(new_path)
    expected = config["expected_invariants"]
    if (
        old_header != new_header
        or len(old_rows) != expected["source_row_count"]
        or len(new_rows) != expected["source_row_count"]
    ):
        raise ValueError("source-data header or row denominator drift")

    allowed_fields = set(expected["permitted_changed_summary_fields"])
    changed = []
    for index, (old, new) in enumerate(zip(old_rows, new_rows, strict=True)):
        differences = {
            name for name in old_header if old[name] != new[name]
        }
        if not differences:
            continue
        cell_id = old["cell_id"]
        if (
            differences != allowed_fields
            or old["row_type"] != ""
            or new["row_type"] != ""
            or not cell_id.endswith("__effect_0.050")
            or new["cell_id"] != cell_id
            or not math.isclose(
                float(old["true_distance"]),
                0.05,
                rel_tol=0.0,
                abs_tol=config["repair_contract"]["absolute_tolerance"],
            )
            or new["true_distance"] != old["true_distance"]
            or old["power_gate_pass"] != "False"
            or old["gate_pass"] != "False"
            or new["power_gate_pass"] != "True"
            or new["gate_pass"] != "True"
        ):
            raise ValueError(f"unpermitted source-data change at row {index}")
        changed.append({
            "row_index": index,
            "cell_id": cell_id,
            "computed_true_distance": float(old["true_distance"]),
            "changed_fields": sorted(differences),
        })

    if (
        len(changed) != expected["permitted_changed_summary_cells"]
        or {row["cell_id"] for row in changed}
        != _expected_local_effect_cell_ids(config)
    ):
        raise ValueError("changed confirmation-summary cell count drift")
    return changed


def _validate_v1_signature(report: Mapping[str, Any]) -> None:
    if (
        report.get("analysis_sha256") != _self_hash(report)
        or
        report.get("verdict") != EXPECTED_V1_VERDICT
        or report.get("gate_summary") != {"passed": 14, "total": 15}
        or report["gates"].get("G10_density_confirmation_all_pass") is not False
        or sum(
            not row["gate_pass"]
            for row in report["confirmation"]["density_summaries"]
        )
        != 6
        or report.get("t04_preregistration_released") is not False
    ):
        raise ValueError("archived V1 failure signature drift")


def _validate_v2_report(
    report: Mapping[str, Any],
    root: Path,
    config: Mapping[str, Any],
) -> None:
    paths = config["artifact_paths"]
    selected = report["selection"]["selected"]
    claims = report["claim_boundary"]
    if (
        report.get("verdict") != EXPECTED_V2_VERDICT
        or report.get("gate_summary") != {"passed": 15, "total": 15}
        or not all(report["gates"].values())
        or report.get("t04_preregistration_released") is not True
        or report.get("t04_scientific_execution_released") is not False
        or selected.get("scale") != 2.0
        or selected.get("state_clusters") != 768
        or selected.get("round_clusters") != 1536
        or selected.get("aggregate_fault_clusters") != 4608
        or len(report["confirmation"]["density_summaries"]) != 24
        or sum(
            not row["gate_pass"]
            for row in report["confirmation"]["density_summaries"]
        )
        != 0
        or any(
            claims[name] is not None
            for name in (
                "twin_qualification",
                "ler",
                "lifetime",
                "physical_break_even",
                "official_puviani_exact",
                "puviani_nmf_surpass",
                "external_sota",
                "hardware_measured",
            )
        )
    ):
        raise ValueError("repaired writer report contract drift")

    for name, path_key in (
        ("source_data", "live_source_data"),
        ("selected_blueprint", "live_selected_blueprint"),
        ("writer_source", "t06_writer"),
    ):
        expected_binding = _binding(root / paths[path_key], root)
        if report["bindings"].get(name) != expected_binding:
            raise ValueError(f"repaired writer report binding drift: {name}")


def _diff_paths(old: Any, new: Any, prefix: str = "") -> set[str]:
    if type(old) is not type(new):
        return {prefix or "<root>"}
    if isinstance(old, dict):
        if set(old) != set(new):
            return {prefix or "<root>"}
        output: set[str] = set()
        for key in old:
            child = f"{prefix}.{key}" if prefix else str(key)
            output.update(_diff_paths(old[key], new[key], child))
        return output
    if isinstance(old, list):
        if len(old) != len(new):
            return {prefix or "<root>"}
        output = set()
        for index, (left, right) in enumerate(zip(old, new, strict=True)):
            output.update(_diff_paths(left, right, f"{prefix}[{index}]"))
        return output
    return set() if old == new else {prefix or "<root>"}


def _compare_writer_reports(
    old: Mapping[str, Any],
    new: Mapping[str, Any],
    config: Mapping[str, Any],
) -> list[str]:
    changed_ids = _expected_local_effect_cell_ids(config)
    mandatory = {
        "analysis_sha256",
        "bindings.source_data.bytes",
        "bindings.source_data.sha256",
        "bindings.writer_source.bytes",
        "bindings.writer_source.sha256",
        "gates.G10_density_confirmation_all_pass",
        "gate_summary.passed",
        "t04_preregistration_released",
        "claim_boundary.t04_preregistration_released",
        "verdict",
    }
    old_summaries = old["confirmation"]["density_summaries"]
    new_summaries = new["confirmation"]["density_summaries"]
    if [
        row["cell_id"] for row in old_summaries
    ] != [
        row["cell_id"] for row in new_summaries
    ]:
        raise ValueError("writer confirmation summary order or IDs changed")
    for index, row in enumerate(old_summaries):
        if row["cell_id"] in changed_ids:
            mandatory.update({
                f"confirmation.density_summaries[{index}].power_gate_pass",
                f"confirmation.density_summaries[{index}].gate_pass",
            })
    actual = _diff_paths(old, new)
    optional = {"resource.free_disk_bytes"}
    if not mandatory.issubset(actual) or not actual.issubset(
        mandatory | optional
    ):
        missing = sorted(mandatory - actual)
        extra = sorted(actual - mandatory - optional)
        raise ValueError(
            f"writer report diff whitelist mismatch; missing={missing}, extra={extra}"
        )
    return sorted(actual)


def repair(root: Path | None = None, *, workers: int = 4) -> dict[str, Any]:
    base = (root or _root()).resolve()
    config = _load_json(base / CONFIG_PATH)
    _validate_runtime_environment(config)
    validate_config(config, base)
    paths = config["artifact_paths"]
    t06_config = _load_json(base / paths["t06_config"])
    writer.validate_config(t06_config)

    v1_report = _load_json(
        base / config["v1_archive"]["writer_report"]["path"]
    )
    _validate_v1_signature(v1_report)
    for name, binding in v1_report["bindings"].items():
        if name not in ("writer_source", "source_data", "selected_blueprint"):
            _assert_file_binding(base, binding)
    for live_key, archive_key in (
        ("live_report", "writer_report"),
        ("live_source_data", "source_data"),
        ("live_selected_blueprint", "selected_blueprint"),
    ):
        if _file_sha(base / paths[live_key]) != config["v1_archive"][
            archive_key
        ]["sha256"]:
            raise ValueError(f"live V1 artifact not equal to archive: {live_key}")

    before_inventory, before_rows = _chunk_inventory(
        base, config, t06_config
    )
    _verify_csv_raw_matches_chunks(
        base / config["v1_archive"]["source_data"]["path"],
        before_rows,
    )
    live_report = writer.write_artifacts(
        base, workers=workers, reuse_only=True
    )
    after_inventory, after_rows = _chunk_inventory(base, config, t06_config)
    if (
        before_inventory != after_inventory
        or before_rows != after_rows
        or (base / t06_config["artifact_paths"]["run_directory"]
            / "supervisor.owner.lock").exists()
    ):
        raise ValueError("raw chunks changed or writer lock leaked")

    changed = _compare_source_data(
        base / config["v1_archive"]["source_data"]["path"],
        base / paths["live_source_data"],
        config,
    )
    if _file_sha(base / paths["live_selected_blueprint"]) != config[
        "v1_archive"
    ]["selected_blueprint"]["sha256"]:
        raise ValueError("selected 3043-gate blueprint changed")
    loaded_report = _load_json(base / paths["live_report"])
    if live_report != loaded_report:
        raise ValueError("writer return value differs from atomic report")
    _validate_v2_report(loaded_report, base, config)
    report_diff = _compare_writer_reports(v1_report, loaded_report, config)

    report = {
        "task_id": TASK_ID,
        "schema_version": SCHEMA,
        "repair_classification": "analysis-only implementation repair",
        "bindings": {
            "repair_config": _binding(base / CONFIG_PATH, base),
            "v1_manifest": _binding(
                base
                / config["v1_archive"]["tracked_directory"]
                / "manifest.json",
                base,
            ),
            "v1_writer_report": _binding(
                base / config["v1_archive"]["writer_report"]["path"], base
            ),
            "v1_source_data": _binding(
                base / config["v1_archive"]["source_data"]["path"], base
            ),
            "v1_selected_blueprint": _binding(
                base / config["v1_archive"]["selected_blueprint"]["path"],
                base,
            ),
            "repaired_writer_source": _binding(
                base / paths["t06_writer"], base
            ),
            "unchanged_independent_verifier": _binding(
                base / paths["t06_verifier"], base
            ),
            "v2_writer_report": _binding(
                base / paths["live_report"], base
            ),
            "v2_source_data": _binding(
                base / paths["live_source_data"], base
            ),
            "v2_selected_blueprint": _binding(
                base / paths["live_selected_blueprint"], base
            ),
        },
        "raw_data_proof": {
            "selection_chunks": sum(
                item["split"] == "selection" for item in after_inventory
            ),
            "confirmation_chunks": sum(
                item["split"] == "confirmation" for item in after_inventory
            ),
            "density_trials": len(after_rows),
            "chunk_inventory_before_sha256": _json_sha(before_inventory),
            "chunk_inventory_after_sha256": _json_sha(after_inventory),
            "chunk_inventory_identical": before_inventory == after_inventory,
            "raw_records_identical": before_rows == after_rows,
            "new_random_trials": False,
            "raw_chunk_rewrite": False,
        },
        "runtime_reproducibility": {
            "process_environment": {
                name: os.environ[name]
                for name in config["runtime_contract"][
                    "required_process_environment_before_python_start"
                ]
            },
            "maxT_numeric_tolerance_substitution": False,
            "v1_single_thread_string_differences": 0,
            "failed_multithread_attempt_manifest": _binding(
                base
                / config["failed_reanalysis_attempt_v1"]["manifest"]["path"],
                base,
            ),
        },
        "source_data_diff": {
            "row_count": config["expected_invariants"]["source_row_count"],
            "changed_summary_cells": changed,
            "all_other_rows_and_fields_identical": True,
        },
        "writer_reanalysis": {
            "selected_count": loaded_report["selection"]["selected"],
            "gate_summary": loaded_report["gate_summary"],
            "verdict": loaded_report["verdict"],
            "preliminary_t04_preregistration_released": loaded_report[
                "t04_preregistration_released"
            ],
            "t04_scientific_execution_released": False,
            "v1_to_v2_changed_paths": report_diff,
        },
        "independent_full_verifier_required": True,
        "t04_preregistration_final_release": False,
        "qualified_claim": None,
        "claim_boundary": config["claim_boundary"],
        "verdict": "PASS_ANALYSIS_ONLY_EFFECT_DISPATCH_REPAIR",
    }
    report["analysis_sha256"] = _self_hash(report)
    _atomic_json(base / paths["repair_report"], report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    arguments = parser.parse_args(argv)
    result = repair(workers=arguments.workers)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
