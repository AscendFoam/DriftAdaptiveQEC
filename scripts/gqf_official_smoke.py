"""Execute a real, bounded smoke against a patched official GQF worktree.

This script intentionally lives outside ``third_party/GQF``.  It imports the
upstream modules by path, constructs a finite-energy GKP state, and (for the
CPU full mode) executes one actual sBs environment step.  The GPU state mode
is isolated in a subprocess because TensorFlow 2.10 can terminate the process
inside cuSolver on unsupported GPU/runtime combinations.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from time import perf_counter


def _write(path: Path | None, payload: dict) -> None:
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("GQF_SMOKE_RESULT=" + json.dumps(payload, sort_keys=True), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gqf-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("cpu_full", "gpu_state"), required=True)
    parser.add_argument("--artifact", type=Path)
    args = parser.parse_args()

    if args.mode == "cpu_full":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

    source_root = args.gqf_root.resolve()
    if not (source_root / "GKP_environment.py").is_file():
        raise FileNotFoundError(source_root / "GKP_environment.py")
    sys.path.insert(0, str(source_root))

    import numpy as np
    import tensorflow as tf

    from feedback_GRAPE import FeedbackGRAPE
    from GKP_environment import GKPEnv
    from mesolve import MESolve
    from operators import Operators
    from qutils import Qutils
    from states import States

    del MESolve, Operators, Qutils  # successful imports are part of the smoke
    seed = 6_803
    np.random.seed(seed)
    tf.random.set_seed(seed)
    physical_gpus = [device.name for device in tf.config.list_physical_devices("GPU")]
    print(
        "GQF_SMOKE_START="
        + json.dumps(
            {
                "mode": args.mode,
                "tensorflow": tf.__version__,
                "physical_gpus": physical_gpus,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    begin = perf_counter()
    states = States()
    cutoff = 8
    alpha = tf.reshape(tf.cast(tf.sqrt(np.pi / 2.0), tf.complex64), (1, 1, 1))
    beta = tf.reshape(tf.cast(1j * np.sqrt(np.pi / 2.0), tf.complex64), (1, 1, 1))
    psi = states.gkp(
        cutoff,
        alpha=alpha,
        beta=beta,
        mu=0,
        Delta=0.34,
        batch_size=1,
    )
    psi_norm = float(tf.linalg.norm(psi).numpy())
    if not np.isfinite(psi_norm) or abs(psi_norm - 1.0) > 2.0e-5:
        raise ValueError(f"GQF state norm failed: {psi_norm}")

    if args.mode == "gpu_state":
        if not physical_gpus:
            raise RuntimeError("GPU state smoke requested but TensorFlow registered no GPU")
        payload = {
            "schema_version": "t6.8.3-gqf-smoke-v1",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "mode": args.mode,
            "source_root": source_root.as_posix(),
            "seed": seed,
            "tensorflow": tf.__version__,
            "physical_gpus": physical_gpus,
            "cutoff": cutoff,
            "gkp_state_shape": list(psi.shape),
            "gkp_state_norm": psi_norm,
            "environment_steps": 0,
            "elapsed_s": perf_counter() - begin,
            "status": "PASS_GPU_STATE",
        }
        _write(args.artifact, payload)
        return 0

    controller = FeedbackGRAPE(
        N=cutoff,
        Delta=0.34,
        max_steps=2,
        batch_size=1,
        num_controls=15,
        num_training_episodes=1,
        num_evaluation_episodes=1,
        model="rNN",
        reward_type="final fidelity",
        dynamics_type="simple",
        noise_level="high",
        TEST=True,
    )
    if controller.N != cutoff or controller.num_controls != 15:
        raise ValueError("FeedbackGRAPE constructor did not preserve smoke contract")

    environment = GKPEnv(
        N=cutoff,
        Delta=0.34,
        noise_level="high",
        max_steps=2,
        memory_length=1,
        protocol="sBs",
        initialized_values=True,
        batch_size=1,
        dynamics=True,
        deterministic=True,
        mu=[1.0],
        dynamics_type="simple",
        Hamiltonian="off",
        reward_type="final fidelity",
        initial_state="zero logical",
        verbosity="low",
        TEST=True,
    )
    initial_observation = np.asarray(environment.reset())
    observation, reward, done, info = environment.step(
        tf.zeros((1, 1, 15), dtype=tf.float32)
    )
    rho = np.asarray(environment.rho[0].numpy(), dtype=np.complex128)
    rho_trace = complex(np.trace(rho))
    hermitian_residual = float(np.max(np.abs(rho - rho.conj().T)))
    minimum_eigenvalue = float(np.min(np.linalg.eigvalsh((rho + rho.conj().T) / 2.0)))
    probability = float(np.real(np.asarray(info["prob"]).reshape(-1)[0]))
    checks = {
        "initial_observation_shape": list(initial_observation.shape) == [1],
        "observation_shape": list(np.asarray(observation).shape) == [1],
        "observation_is_binary_syndrome": bool(
            np.all(np.isin(np.real(np.asarray(observation)), (-1.0, 1.0)))
        ),
        "reward_finite": bool(np.isfinite(reward)),
        "probability_valid": bool(0.0 <= probability <= 1.0),
        "trace_one": bool(abs(rho_trace - 1.0) <= 2.0e-5),
        "hermitian": hermitian_residual <= 2.0e-5,
        "positive_semidefinite_tolerance": minimum_eigenvalue >= -2.0e-5,
        "not_terminal_after_first_of_two_steps": done is False,
    }
    if not all(checks.values()):
        raise ValueError(f"GQF environment smoke checks failed: {checks}")
    payload = {
        "schema_version": "t6.8.3-gqf-smoke-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "source_root": source_root.as_posix(),
        "seed": seed,
        "tensorflow": tf.__version__,
        "physical_gpus": physical_gpus,
        "cutoff": cutoff,
        "gkp_state_shape": list(psi.shape),
        "gkp_state_norm": psi_norm,
        "environment_steps": 1,
        "reward": float(reward),
        "probability": probability,
        "rho_trace_real": float(rho_trace.real),
        "rho_trace_imag": float(rho_trace.imag),
        "rho_hermitian_residual": hermitian_residual,
        "rho_minimum_eigenvalue": minimum_eigenvalue,
        "checks": checks,
        "elapsed_s": perf_counter() - begin,
        "status": "PASS_CPU_FULL",
    }
    _write(args.artifact, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
