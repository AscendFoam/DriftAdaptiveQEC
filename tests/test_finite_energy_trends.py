from __future__ import annotations

import math
import unittest

import numpy as np

from physics.constants import LATTICE_CONST
from physics.finite_energy_gkp import damped_projector_state
from physics.finite_energy_trends import (
    ShrinkageTrendConfig,
    run_finite_energy_shrinkage_trend,
)


class FiniteEnergyShrinkageTrendTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = run_finite_energy_shrinkage_trend()

    def test_default_protocol_reproduces_all_declared_trend_flags(self) -> None:
        result = self.result
        self.assertEqual(result.evidence_scope, "syndrome_level_effective_model")
        self.assertTrue(result.fitted_gain_increases_as_delta_decreases)
        self.assertTrue(result.logical_advantage_shrinks_as_delta_decreases)
        self.assertTrue(result.mse_advantage_shrinks_as_delta_decreases)
        self.assertTrue(result.all_shrinkage_mse_not_worse)
        self.assertEqual(len(result.points), len(result.config.delta_values))

    def test_intrinsic_sigma_is_derived_from_damped_projector_state(self) -> None:
        for point in self.result.points:
            state = damped_projector_state(
                "0",
                point.delta,
                lattice=self.result.config.lattice,
                tail_tolerance=1.0e-10,
            )
            expected = math.sqrt(state.amplitude_variance / 2.0)
            with self.subTest(delta=point.delta):
                self.assertAlmostEqual(point.intrinsic_sigma, expected, places=15)

    def test_fitted_shrinkage_gain_moves_toward_one_as_delta_decreases(self) -> None:
        fitted = np.array([point.fitted_gain for point in self.result.points])
        unwrapped = np.array(
            [point.unwrapped_mmse_gain for point in self.result.points]
        )

        self.assertTrue(np.all(np.diff(fitted) > 0.0))
        self.assertTrue(np.all(np.diff(unwrapped) > 0.0))
        self.assertTrue(np.all((fitted >= 0.0) & (fitted <= 1.0)))
        self.assertTrue(np.all((unwrapped > 0.0) & (unwrapped < 1.0)))
        # wrapped syndrome 有 alias ambiguity，因此 fitted gain 低于不 wrap 的 MMSE 值。
        self.assertTrue(np.all(fitted < unwrapped))

    def test_standard_gain_one_is_suboptimal_at_broad_finite_energy_points(self) -> None:
        broad_points = self.result.points[:3]
        for point in broad_points:
            with self.subTest(delta=point.delta):
                self.assertLess(point.shrinkage_mse, point.standard_mse)
                self.assertLess(
                    point.shrinkage_logical_error,
                    point.standard_logical_error,
                )
                self.assertGreater(point.gain_ci_low, 0.0)
                self.assertGreater(point.mcnemar_z, 10.0)

    def test_logical_advantage_shrinks_to_negligible_scale_at_better_squeezing(self) -> None:
        gains = np.array([point.absolute_logical_gain for point in self.result.points])
        relative = np.array(
            [point.relative_logical_reduction for point in self.result.points]
        )

        self.assertTrue(np.all(np.diff(gains) <= 0.0))
        self.assertTrue(np.all(np.diff(relative) <= 0.0))
        self.assertGreater(gains[0], 0.02)
        self.assertGreater(relative[0], 0.5)
        self.assertLess(gains[-1], 2.0e-5)
        self.assertLess(relative[-1], 2.0e-3)

    def test_mse_advantage_is_not_inferred_from_logical_error_only(self) -> None:
        mse_gains = np.array(
            [point.standard_mse - point.shrinkage_mse for point in self.result.points]
        )
        self.assertTrue(np.all(mse_gains > 0.0))
        self.assertTrue(np.all(np.diff(mse_gains) < 0.0))
        # 最后一点 logical gain 已降至万分之一级，但 continuous residual MSE
        # 仍保留可测改善，不能把两种 metric 混为一谈。
        self.assertLess(self.result.points[-1].absolute_logical_gain, 2.0e-5)
        self.assertGreater(mse_gains[-1], 0.0)

    def test_training_gain_and_heldout_metrics_match_independent_recalculation(self) -> None:
        config = ShrinkageTrendConfig(
            delta_values=(0.60, 0.45, 0.30),
            train_samples=30_000,
            eval_samples=50_000,
            seed=731,
        )
        result = run_finite_energy_shrinkage_trend(config)
        rng = np.random.default_rng(config.seed)
        train_x = rng.normal(0.0, config.channel_sigma, size=config.train_samples)
        train_z = rng.normal(0.0, 1.0, size=config.train_samples)
        eval_x = rng.normal(0.0, config.channel_sigma, size=config.eval_samples)
        eval_z = rng.normal(0.0, 1.0, size=config.eval_samples)
        first = result.points[0]
        train_s = train_x + first.intrinsic_sigma * train_z
        train_s -= np.floor(train_s / config.lattice + 0.5) * config.lattice
        expected_gain = float(np.clip(np.dot(train_x, train_s) / np.dot(train_s, train_s), 0, 1))
        eval_s = eval_x + first.intrinsic_sigma * eval_z
        eval_s -= np.floor(eval_s / config.lattice + 0.5) * config.lattice
        standard_residual = eval_x - eval_s
        shrinkage_residual = eval_x - expected_gain * eval_s
        expected_standard_mse = float(np.mean(standard_residual**2))
        expected_shrinkage_mse = float(np.mean(shrinkage_residual**2))
        standard_index = np.floor(standard_residual / config.lattice + 0.5).astype(np.int64)
        shrinkage_index = np.floor(shrinkage_residual / config.lattice + 0.5).astype(np.int64)
        expected_standard_error = float(np.mean(np.mod(standard_index, 2) != 0))
        expected_shrinkage_error = float(np.mean(np.mod(shrinkage_index, 2) != 0))

        self.assertAlmostEqual(first.fitted_gain, expected_gain, places=15)
        self.assertAlmostEqual(first.standard_mse, expected_standard_mse, places=15)
        self.assertAlmostEqual(first.shrinkage_mse, expected_shrinkage_mse, places=15)
        self.assertAlmostEqual(first.standard_logical_error, expected_standard_error, places=15)
        self.assertAlmostEqual(first.shrinkage_logical_error, expected_shrinkage_error, places=15)

    def test_protocol_is_bit_reproducible_for_fixed_seed(self) -> None:
        config = ShrinkageTrendConfig(
            delta_values=(0.55, 0.40, 0.25),
            train_samples=20_000,
            eval_samples=30_000,
            seed=91,
        )
        first = run_finite_energy_shrinkage_trend(config)
        second = run_finite_energy_shrinkage_trend(config)
        self.assertEqual(first, second)

    def test_invalid_protocols_fail_closed(self) -> None:
        invalid_calls = [
            lambda: ShrinkageTrendConfig(delta_values=(0.6, 0.5)),
            lambda: ShrinkageTrendConfig(delta_values=(0.4, 0.5, 0.3)),
            lambda: ShrinkageTrendConfig(delta_values=(0.5, 0.5, 0.3)),
            lambda: ShrinkageTrendConfig(delta_values=(0.5, 0.3, 0.0)),
            lambda: ShrinkageTrendConfig(channel_sigma=0.0),
            lambda: ShrinkageTrendConfig(channel_sigma=1.0e308),
            lambda: ShrinkageTrendConfig(train_samples=100),
            lambda: ShrinkageTrendConfig(eval_samples=100),
            lambda: ShrinkageTrendConfig(seed=-1),
            lambda: ShrinkageTrendConfig(lattice=0.0),
            lambda: run_finite_energy_shrinkage_trend("bad"),  # type: ignore[arg-type]
        ]
        for call in invalid_calls:
            with self.subTest(call=call):
                with self.assertRaises((ValueError, TypeError)):
                    call()


if __name__ == "__main__":
    unittest.main()
