"""Numerical and import characterization for the backend-A façade split."""

from __future__ import annotations

from hashlib import sha256

import numpy as np

import physics.phase9_backend_a as backend_a


def test_fixed_seed_three_round_characterization_is_unchanged() -> None:
    config = backend_a.BackendAConfig(
        cutoff=8,
        substeps_per_segment=2,
        iq_samples=3,
        logical_grid_points=1025,
    )
    simulator = backend_a.Phase9BackendASimulator(config)
    initial, evaluator = simulator.initialize_logical("+")
    actions = tuple(
        backend_a.diagnostic_action_word(name)
        for name in ("IDLE", "X", "RESET")
    )

    trajectory = simulator.simulate(
        initial,
        actions,
        seed=20260809,
        evaluator=evaluator,
    )

    assert config.semantic_sha256() == (
        "f3033486e7fa0f246b8e4f2b909d61cb5fb77216567c7f280e53318bf6bb30d8"
    )
    density_bytes = np.ascontiguousarray(
        trajectory.final_state.joint_density
    ).view(np.uint8)
    assert sha256(density_bytes).hexdigest() == (
        "7f20f908f29ea50a687ea6583077721f42128e51460eb9548f3d7107ea6da3e1"
    )
    np.testing.assert_allclose(
        [item for round_ in trajectory.rounds for item in round_.observation.iq_i],
        [
            -0.2875014902912928,
            -0.35060793592755124,
            -1.2371185074913582,
            -0.9656542851269678,
            -0.6124722369546969,
            0.1839026448932125,
            -1.196128952470687,
            -1.245650681933674,
            -1.3833650948998288,
        ],
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        trajectory.final_state.drift.vector(),
        [
            0.005901721835853373,
            0.00033602078308698916,
            0.0021586302227852737,
            -0.00035894765844034194,
            0.0019877536543677753,
        ],
        rtol=0.0,
        atol=0.0,
    )
    assert [round_.truth.reset_hidden_outcome for round_ in trajectory.rounds] == [
        "none",
        "none",
        "success",
    ]
    np.testing.assert_allclose(
        [round_.logical.target_fidelity for round_ in trajectory.rounds],
        [0.9996350684112958, 0.9989828212319561, 0.9980988738346193],
        rtol=0.0,
        atol=0.0,
    )
