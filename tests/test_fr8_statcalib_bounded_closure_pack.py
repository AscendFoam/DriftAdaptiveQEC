import json
import tempfile
import unittest
from pathlib import Path

from cnn_fpga.benchmark.build_fr8_statcalib_bounded_closure_pack import (
    ClosureInputs,
    build_closure_pack,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_T69_SUMMARY = (
    REPO_ROOT
    / "runs"
    / "p4_benchmark"
    / "T69_statcalib_clean_winner_tiebreak_20260608_160358"
    / "statcalib_clean_winner_tiebreak_summary"
    / "summary.json"
)


class Fr8StatCalibBoundedClosurePackTests(unittest.TestCase):
    def test_current_artifacts_report_expected_gate_outcome(self) -> None:
        pack = build_closure_pack()

        self.assertEqual(
            pack["final_strongest_clean_answer_set_after_t69"],
            [
                "statcalib_window_variance_t001",
                "statcalib_window_variance_t003",
                "statcalib_window_variance_t005",
            ],
        )
        self.assertFalse(pack["unique_clean_reference_point_exists"])
        self.assertEqual(
            pack["promotion_gate"]["verdict"],
            "no_promotion_keep_extension_lane_only",
        )
        self.assertEqual(
            pack["unique_threshold_gate"]["verdict"],
            "future_selection_task_required",
        )
        self.assertEqual(
            pack["extension_lane_evidence"]["t69"]["final_clean_winner_classification"],
            "persistent_clean_tie_set",
        )
        self.assertTrue(pack["extension_lane_evidence"]["t66"]["best_variant_beats_ukf_all_scenarios"])
        self.assertTrue(
            pack["extension_lane_evidence"]["t67"][
                "any_non_ukf_variant_beats_both_frozen_anchors_all_scenarios"
            ]
        )

    def test_rejects_t69_pack_if_persistent_tie_no_longer_matches_triplet(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            mutated_path = Path(tmpdir) / "bad_t69_summary.json"
            payload = json.loads(DEFAULT_T69_SUMMARY.read_text(encoding="utf-8"))
            payload["current_clean_answer_set"]["modes"] = [
                "statcalib_window_variance_t001",
            ]
            mutated_path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "T69 current_clean_answer_set.modes"):
                build_closure_pack(
                    ClosureInputs(
                        t69_summary_json=mutated_path,
                    )
                )


if __name__ == "__main__":
    unittest.main()
