from __future__ import annotations

import math
import unittest

import numpy as np

from physics.oracle_gap import (
    OracleGapMetrics,
    OracleGapPointEstimate,
    compute_oracle_gap_metrics,
    oracle_gap_from_rates,
)


class OracleGapPointEstimateTest(unittest.TestCase):
    def test_canonical_remaining_and_closed_gap_are_both_reported(self) -> None:
        result = oracle_gap_from_rates(0.20, 0.12, 0.05)
        self.assertIsInstance(result, OracleGapPointEstimate)
        self.assertAlmostEqual(result.static_oracle_gap, 0.15, places=15)
        self.assertAlmostEqual(result.dual_oracle_gap, 0.07, places=15)
        self.assertAlmostEqual(result.absolute_improvement, 0.08, places=15)
        self.assertAlmostEqual(result.gap_remaining_ratio, 7.0 / 15.0, places=15)
        self.assertAlmostEqual(result.gap_closed_fraction, 8.0 / 15.0, places=15)
        self.assertAlmostEqual(
            result.gap_remaining_ratio + result.gap_closed_fraction,
            1.0,
            places=15,
        )
        self.assertEqual(result.denominator_status, "positive")
        self.assertEqual(result.bracket_status, "within_oracle_static_bracket")
        self.assertTrue(result.reference_order_valid)

    def test_out_of_bracket_ratios_are_not_clipped(self) -> None:
        worse = oracle_gap_from_rates(0.20, 0.25, 0.05)
        self.assertLess(worse.gap_closed_fraction, 0.0)
        self.assertGreater(worse.gap_remaining_ratio, 1.0)
        self.assertEqual(worse.bracket_status, "dual_worse_than_static")
        self.assertIn("dual_worse_than_static", worse.flags)

        better = oracle_gap_from_rates(0.20, 0.03, 0.05)
        self.assertGreater(better.gap_closed_fraction, 1.0)
        self.assertLess(better.gap_remaining_ratio, 0.0)
        self.assertEqual(better.bracket_status, "dual_better_than_oracle")
        self.assertIn("dual_better_than_oracle_point_estimate", better.flags)

    def test_zero_and_inverted_reference_gap_are_distinguished(self) -> None:
        zero = oracle_gap_from_rates(0.10, 0.08, 0.10)
        self.assertEqual(zero.denominator_status, "zero")
        self.assertEqual(zero.bracket_status, "zero_reference_gap")
        self.assertIsNone(zero.gap_remaining_ratio)
        self.assertIsNone(zero.gap_closed_fraction)
        self.assertFalse(zero.reference_order_valid)

        inverted = oracle_gap_from_rates(0.05, 0.08, 0.10)
        self.assertEqual(inverted.denominator_status, "inverted")
        self.assertEqual(inverted.bracket_status, "reference_order_inverted")
        self.assertAlmostEqual(inverted.gap_remaining_ratio, 0.4, places=15)
        self.assertAlmostEqual(inverted.gap_closed_fraction, 0.6, places=15)
        self.assertIn("oracle_worse_than_static", inverted.flags)
        self.assertFalse(inverted.reference_order_valid)

    def test_invalid_rates_and_epsilon_fail_closed(self) -> None:
        invalid_calls = [
            lambda: oracle_gap_from_rates(-0.1, 0.1, 0.0),
            lambda: oracle_gap_from_rates(0.1, 1.1, 0.0),
            lambda: oracle_gap_from_rates(0.1, 0.1, float("nan")),
            lambda: oracle_gap_from_rates(0.1, 0.1, 0.0, denominator_epsilon=-1.0),
        ]
        for call in invalid_calls:
            with self.subTest(call=call):
                with self.assertRaises(ValueError):
                    call()


class PairedOracleGapMetricsTest(unittest.TestCase):
    @staticmethod
    def _nested_outcomes() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        static = np.zeros(1_000, dtype=bool)
        dual = np.zeros(1_000, dtype=bool)
        oracle = np.zeros(1_000, dtype=bool)
        static[:200] = True
        dual[:120] = True
        oracle[:50] = True
        return static, dual, oracle

    def test_paired_metrics_match_exact_counts_intervals_and_mcnemar(self) -> None:
        static, dual, oracle = self._nested_outcomes()
        result = compute_oracle_gap_metrics(
            static,
            dual,
            oracle,
            bootstrap_replicates=5_000,
            seed=91,
        )
        self.assertIsInstance(result, OracleGapMetrics)
        self.assertEqual(result.n_samples, 1_000)
        self.assertAlmostEqual(result.point.static_error_rate, 0.20, places=15)
        self.assertAlmostEqual(result.point.dual_error_rate, 0.12, places=15)
        self.assertAlmostEqual(result.point.oracle_error_rate, 0.05, places=15)
        self.assertAlmostEqual(result.static_minus_dual.estimate, 0.08, places=15)
        self.assertAlmostEqual(result.static_minus_oracle.estimate, 0.15, places=15)
        self.assertAlmostEqual(result.dual_minus_oracle.estimate, 0.07, places=15)
        expected_se = math.sqrt(0.08 * 0.92 / 999.0)
        self.assertAlmostEqual(result.static_minus_dual.standard_error, expected_se, places=15)
        self.assertEqual(result.static_only_failure_count, 80)
        self.assertEqual(result.dual_only_failure_count, 0)
        self.assertAlmostEqual(result.mcnemar_z, math.sqrt(80.0), places=15)
        self.assertEqual(sum(result.joint_outcome_counts), 1_000)
        self.assertTrue(result.denominator_stable)
        self.assertTrue(result.ratio_ci_reliable)
        self.assertGreaterEqual(result.bootstrap_valid_replicates, 4_750)
        self.assertLess(result.gap_closed_ci[0], result.gap_closed_fraction)
        self.assertGreater(result.gap_closed_ci[1], result.gap_closed_fraction)
        self.assertLess(result.gap_remaining_ci[0], result.gap_remaining_ratio)
        self.assertGreater(result.gap_remaining_ci[1], result.gap_remaining_ratio)
        self.assertEqual(result.evidence_scope, "paired_logical_failure_metric")

    def test_fixed_seed_bootstrap_is_bit_reproducible(self) -> None:
        outcomes = self._nested_outcomes()
        first = compute_oracle_gap_metrics(*outcomes, bootstrap_replicates=2_000, seed=13)
        second = compute_oracle_gap_metrics(*outcomes, bootstrap_replicates=2_000, seed=13)
        self.assertEqual(first, second)

    def test_pairing_changes_uncertainty_even_when_all_three_rates_match(self) -> None:
        oracle = np.zeros(1_000, dtype=bool)
        static = np.zeros(1_000, dtype=bool)
        static[:200] = True

        nested_dual = np.zeros(1_000, dtype=bool)
        nested_dual[:100] = True
        disjoint_dual = np.zeros(1_000, dtype=bool)
        disjoint_dual[200:300] = True

        nested = compute_oracle_gap_metrics(
            static,
            nested_dual,
            oracle,
            bootstrap_replicates=0,
        )
        disjoint = compute_oracle_gap_metrics(
            static,
            disjoint_dual,
            oracle,
            bootstrap_replicates=0,
        )
        self.assertEqual(nested.point, disjoint.point)
        self.assertGreater(
            disjoint.static_minus_dual.standard_error,
            nested.static_minus_dual.standard_error,
        )
        self.assertEqual(nested.static_only_failure_count, 100)
        self.assertEqual(nested.dual_only_failure_count, 0)
        self.assertEqual(disjoint.static_only_failure_count, 200)
        self.assertEqual(disjoint.dual_only_failure_count, 100)

    def test_small_unstable_denominator_is_flagged_without_hiding_point_ratio(self) -> None:
        static = np.zeros(40, dtype=bool)
        dual = np.zeros(40, dtype=bool)
        oracle = np.zeros(40, dtype=bool)
        static[:5] = True
        dual[:4] = True
        oracle[:4] = True
        result = compute_oracle_gap_metrics(
            static,
            dual,
            oracle,
            bootstrap_replicates=4_000,
            seed=27,
        )
        self.assertEqual(result.gap_remaining_ratio, 0.0)
        self.assertEqual(result.gap_closed_fraction, 1.0)
        self.assertFalse(result.denominator_stable)
        self.assertFalse(result.ratio_ci_reliable)
        self.assertLess(result.bootstrap_valid_replicates, 3_800)
        self.assertLessEqual(result.static_minus_oracle.ci_low, 0.0)

    def test_zero_denominator_and_disabled_bootstrap_have_no_ratio_ci(self) -> None:
        static = np.array([0, 1, 0, 1, 0, 0], dtype=int)
        oracle = static.copy()
        dual = np.array([0, 0, 0, 1, 0, 0], dtype=int)
        zero = compute_oracle_gap_metrics(
            static,
            dual,
            oracle,
            bootstrap_replicates=2_000,
        )
        self.assertIsNone(zero.gap_remaining_ratio)
        self.assertIsNone(zero.gap_closed_fraction)
        self.assertIsNone(zero.gap_remaining_ci)
        self.assertIsNone(zero.gap_closed_ci)
        self.assertEqual(zero.bootstrap_valid_replicates, 0)

        nonzero = compute_oracle_gap_metrics(
            *self._nested_outcomes(),
            bootstrap_replicates=0,
        )
        self.assertIsNotNone(nonzero.gap_closed_fraction)
        self.assertIsNone(nonzero.gap_closed_ci)
        self.assertFalse(nonzero.ratio_ci_reliable)

    def test_invalid_paired_inputs_and_protocols_fail_closed(self) -> None:
        valid = [0, 1, 0]
        invalid_calls = [
            lambda: compute_oracle_gap_metrics([], [], []),
            lambda: compute_oracle_gap_metrics([0], [0], [0]),
            lambda: compute_oracle_gap_metrics([[0, 1]], valid, valid),
            lambda: compute_oracle_gap_metrics([0, 1], valid, valid),
            lambda: compute_oracle_gap_metrics([0, 0.5, 1], valid, valid),
            lambda: compute_oracle_gap_metrics([0, float("nan"), 1], valid, valid),
            lambda: compute_oracle_gap_metrics(["bad", "bad", "bad"], valid, valid),
            lambda: compute_oracle_gap_metrics(valid, valid, valid, confidence_level=0.0),
            lambda: compute_oracle_gap_metrics(valid, valid, valid, confidence_level=1.0),
            lambda: compute_oracle_gap_metrics(valid, valid, valid, bootstrap_replicates=-1),
            lambda: compute_oracle_gap_metrics(valid, valid, valid, bootstrap_replicates=True),
            lambda: compute_oracle_gap_metrics(valid, valid, valid, bootstrap_replicates=1_000_001),
            lambda: compute_oracle_gap_metrics(valid, valid, valid, seed=-1),
            lambda: compute_oracle_gap_metrics(valid, valid, valid, denominator_epsilon=float("nan")),
        ]
        for call in invalid_calls:
            with self.subTest(call=call):
                with self.assertRaises((TypeError, ValueError)):
                    call()


if __name__ == "__main__":
    unittest.main()
