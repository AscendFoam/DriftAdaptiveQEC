from __future__ import annotations

import json

import pytest

from cnn_fpga.benchmark.phase9_powered_twin_resource_forensics import (
    _read_samples,
    _sha,
    ideal_lpt_curve,
)


def test_ideal_lpt_curve_is_deterministic_and_includes_fixed_overhead() -> None:
    cells = [
        {"plan_index": 3, "projected_wall_seconds": 7.0},
        {"plan_index": 1, "projected_wall_seconds": 7.0},
        {"plan_index": 2, "projected_wall_seconds": 4.0},
        {"plan_index": 0, "projected_wall_seconds": 2.0},
    ]
    curve = ideal_lpt_curve(
        cells,
        worker_counts=(1, 2, 3),
        fixed_overhead_seconds=3.0,
        maximum_wall_seconds=13.0,
    )
    assert [item["ideal_raw_lpt_wall_seconds"] for item in curve] == [
        20.0,
        11.0,
        7.0,
    ]
    assert [item["ideal_total_wall_seconds"] for item in curve] == [
        23.0,
        14.0,
        10.0,
    ]
    assert [item["wall_gate_pass_if_no_contention"] for item in curve] == [
        False,
        False,
        True,
    ]
    # Equal-cost cells are ordered by plan index, and worker ties by index.
    assert curve[1]["worker_load_seconds"] == [11.0, 9.0]


@pytest.mark.parametrize("workers", [0, -1, True, 1.5])
def test_ideal_lpt_curve_rejects_invalid_worker_count(workers: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        ideal_lpt_curve(
            [{"plan_index": 0, "projected_wall_seconds": 1.0}],
            worker_counts=(workers,),  # type: ignore[arg-type]
            fixed_overhead_seconds=0.0,
            maximum_wall_seconds=1.0,
        )


def test_read_samples_accepts_zero_child_records(tmp_path) -> None:
    record = {
        "aggregate_rss_bytes": 123,
        "child_process_tree_pids": {},
        "child_rss_bytes": {},
        "live_child_count": 0,
        "monotonic_seconds": 0.0,
        "parent_pid": 10,
        "parent_rss_bytes": 123,
        "previous_sample_sha256": "0" * 64,
        "schema_version": "PHASE9-POWERED-TWIN-RESOURCE-SAMPLING-V1",
        "sequence": 0,
        "stage": "starting",
    }
    record["sample_sha256"] = _sha(record)
    path = tmp_path / "samples.jsonl"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    records, summary = _read_samples(
        path,
        wall_anchor={"process_creation_time": 10.0},
    )
    assert summary["peak_individual_child_rss_bytes"] == 0
    assert summary["sample_count"] == 1
    assert records[0]["forensic_wall_time_ns"] == 10_000_000_000
