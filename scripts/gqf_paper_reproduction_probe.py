from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf


LOGICAL_STATES = (
    ("plus_z", "zero logical", "Z", 1.0),
    ("minus_z", "one logical", "Z", -1.0),
    ("plus_x", "x logical", "X", 1.0),
    ("minus_x", "-x logical", "X", -1.0),
    ("plus_y", "y logical", "Y", 1.0),
    ("minus_y", "-y logical", "Y", -1.0),
)


def _real_flat(value: tf.Tensor) -> np.ndarray:
    return np.asarray(tf.math.real(value).numpy(), dtype=np.float64).reshape(-1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gqf-root", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--cutoff", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seeds", type=int, nargs="+", default=(68401, 68407, 68419))
    parser.add_argument("--max-steps", type=int, default=21)
    args = parser.parse_args()

    source_root = args.gqf_root.resolve()
    sys.path.insert(0, str(source_root))
    from GKP_environment import GKPEnv  # type: ignore[import-not-found]

    if args.cutoff != 8 or args.batch_size != 2 or args.max_steps != 21:
        raise ValueError("T6.8.4 diagnostic configuration is frozen at cutoff=8, batch=2, max_steps=21")
    if tuple(args.seeds) != (68401, 68407, 68419):
        raise ValueError("T6.8.4 diagnostic seed set is frozen")

    rows: list[dict[str, object]] = []
    started = datetime.now(timezone.utc)
    tic = time.perf_counter()
    for seed in args.seeds:
        np.random.seed(seed)
        tf.random.set_seed(seed)
        for state_id, initial_state, pauli_axis, eigenvalue in LOGICAL_STATES:
            env = GKPEnv(
                N=args.cutoff,
                Delta=0.34,
                noise_level="low",
                max_steps=args.max_steps,
                memory_length=1,
                protocol="sBs",
                initialized_values=True,
                batch_size=args.batch_size,
                dynamics=True,
                deterministic=False,
                mu=None,
                dynamics_type="Sivak2023",
                Hamiltonian="off",
                reward_type="final fidelity",
                evaluate=False,
                initial_state=initial_state,
                verbosity="low",
                TEST=True,
            )
            env.reset()
            done = False
            half_cycle = 0
            while not done:
                _, reward, done, info = env.step(actions=None)
                half_cycle += 1
                _, pauli_z, pauli_x = env.pauli(env.rho2)
                pauli_y = env.pauliy(env.rho2)
                axis_values = {
                    "X": _real_flat(pauli_x),
                    "Y": _real_flat(pauli_y),
                    "Z": _real_flat(pauli_z),
                }[pauli_axis]
                probabilities = np.asarray(info["prob"].numpy(), dtype=np.float64).reshape(-1)
                rho = np.asarray(env.rho.numpy(), dtype=np.complex128)
                traces = np.trace(rho, axis1=1, axis2=2)
                for batch_index in range(args.batch_size):
                    rows.append(
                        {
                            "seed": int(seed),
                            "state_id": state_id,
                            "initial_state": initial_state,
                            "pauli_axis": pauli_axis,
                            "expected_eigenvalue": eigenvalue,
                            "batch_index": batch_index,
                            "half_cycle": half_cycle,
                            "full_cycle": half_cycle / 2.0,
                            "signed_pauli": float(eigenvalue * axis_values[batch_index]),
                            "measurement_probability": float(probabilities[batch_index]),
                            "reward": float(reward),
                            "rho_trace_real": float(traces[batch_index].real),
                            "rho_trace_imag": float(traces[batch_index].imag),
                        }
                    )

    expected_rows = len(args.seeds) * len(LOGICAL_STATES) * args.batch_size * args.max_steps
    if len(rows) != expected_rows:
        raise ValueError(f"diagnostic row count mismatch: {len(rows)} != {expected_rows}")
    if not all(np.isfinite(row["signed_pauli"]) for row in rows):
        raise ValueError("non-finite logical observable in diagnostic")
    if not all(0.0 <= row["measurement_probability"] <= 1.0 for row in rows):
        raise ValueError("invalid measurement probability in diagnostic")
    if max(abs(row["rho_trace_real"] - 1.0) for row in rows) > 5.0e-5:
        raise ValueError("density trace drift in diagnostic")
    if max(abs(row["rho_trace_imag"]) for row in rows) > 5.0e-5:
        raise ValueError("complex density trace drift in diagnostic")

    paper_ten_cycle_rows = [row for row in rows if row["half_cycle"] == 20]
    terminal_rows = [row for row in rows if row["half_cycle"] == args.max_steps]
    payload = {
        "schema_version": "t6.8.4-gqf-paper-reproduction-probe-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "started_at_utc": started.isoformat(),
        "source_root": source_root.as_posix(),
        "scope": "REDUCED_STANDARD_PATH_DIAGNOSTIC_NOT_PAPER_REPRODUCTION",
        "configuration": {
            "cutoff": args.cutoff,
            "Delta": 0.34,
            "noise_level": "low",
            "dynamics_type": "Sivak2023",
            "max_steps": args.max_steps,
            "paper_ten_cycle_prefix_half_steps": 20,
            "official_terminal_half_steps": args.max_steps,
            "official_terminal_cycle_coordinate": args.max_steps / 2.0,
            "batch_size": args.batch_size,
            "seeds": list(args.seeds),
            "logical_states": [row[0] for row in LOGICAL_STATES],
            "strategy": "standard_sBs_TEST_true",
        },
        "coverage": {
            "rows": len(rows),
            "expected_rows": expected_rows,
            "trajectories": len(args.seeds) * len(LOGICAL_STATES) * args.batch_size,
            "environment_steps": len(args.seeds) * len(LOGICAL_STATES) * args.max_steps,
        },
        "checks": {
            "row_count": len(rows) == expected_rows,
            "six_states": len({row["state_id"] for row in rows}) == 6,
            "three_seeds": len({row["seed"] for row in rows}) == 3,
            "paper_ten_cycle_prefix_present": len(paper_ten_cycle_rows) == len(args.seeds) * len(LOGICAL_STATES) * args.batch_size,
            "official_max_steps21_executes21_steps": max(row["half_cycle"] for row in rows) == 21,
            "finite_observables": all(np.isfinite(row["signed_pauli"]) for row in rows),
            "probabilities_valid": all(0.0 <= row["measurement_probability"] <= 1.0 for row in rows),
            "trace_one": max(abs(row["rho_trace_real"] - 1.0) for row in rows) <= 5.0e-5,
            "trace_real": max(abs(row["rho_trace_imag"]) for row in rows) <= 5.0e-5,
        },
        "paper_ten_cycle_prefix_signed_pauli_by_state": {
            state_id: float(np.mean([row["signed_pauli"] for row in paper_ten_cycle_rows if row["state_id"] == state_id]))
            for state_id, *_ in LOGICAL_STATES
        },
        "official_terminal_signed_pauli_by_state": {
            state_id: float(np.mean([row["signed_pauli"] for row in terminal_rows if row["state_id"] == state_id]))
            for state_id, *_ in LOGICAL_STATES
        },
        "elapsed_s": time.perf_counter() - tic,
        "status": "PASS_REDUCED_STANDARD_PATH_DIAGNOSTIC",
        "rows": rows,
    }
    if not all(payload["checks"].values()):
        raise ValueError(f"diagnostic checks failed: {payload['checks']}")
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    args.artifact.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("status", "coverage", "elapsed_s")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
