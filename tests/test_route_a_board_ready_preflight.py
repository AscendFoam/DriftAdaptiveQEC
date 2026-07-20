from __future__ import annotations

import json
from pathlib import Path

from cnn_fpga.benchmark.route_a_board_ready_preflight import _pairs


ROOT = Path(__file__).resolve().parents[1]


def test_summary_parser_is_strictly_integer_key_value() -> None:
    assert _pairs("responses=6,latency=7,note=ignored") == {
        "responses": 6,
        "latency": 7,
    }


def test_committed_candidate_is_explicitly_nonphysical() -> None:
    report = json.loads(
        (ROOT / "docs/t6_9_2_preboard_bitstream_candidate.json").read_text(encoding="utf-8")
    )
    assert report["verdict"] == "PASS_PREBOARD_CANDIDATE_NOT_PHYSICAL_QUALIFICATION"
    assert report["candidate_is_physical_evidence"] is False
    assert report["board_programmed"] is False
    assert report["physical_measurements_collected"] is False
    assert report["physical_qualification_blocked"] is True
    assert all(gate["pass"] for gate in report["gates"])
    assert report["post_route"]["fmax_mhz"] >= report["post_route"]["constraint_mhz"]
    assert "full-cadence" in report["claim_boundary"]["long_sequence_limitation"]

