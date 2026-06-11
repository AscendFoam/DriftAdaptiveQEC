# P4 Benchmark Development Protocol

## 1. Purpose

This document fixes the Phase 2 development protocol for P4 frozen benchmark work.

It is intentionally narrower than the historical formal benchmark conclusion set. Its job is to tell a Worker:

1. what `T9` recovery smoke already proved,
2. what `T15` is allowed to run next,
3. what must remain unchanged while collecting stronger evidence.

## 2. Three Distinct Benchmark Layers

### 2.1 Recovery smoke

Recovery smoke is the already re-verified bounded path used in `T7` and `T9`.

- Config: `cnn_fpga/config/p4_multiscenario_recovery_smoke.yaml`
- Interpreter: `C:\ProgramData\anaconda3\python.exe`
- Backend: `mock`
- Inference service mode: `inproc`
- CNN artifact backend: `artifact_npz`
- Fixed artifact path: `artifacts/models/static_theta_v2/tiny_cnn_20260319_151717_b87c6c227b57.npz`
- Typical filters:
  - `static_bias_theta`
  - `static_linear / cnn_fpga`
  - or `static_linear / window_variance / ekf / cnn_fpga`
- Seed policy: `--paired-seeds`
- Repeats: `1`

This layer proves the wrapper and HIL stack can run on the current machine. It does not restore formal multi-scenario P4 evidence.

### 2.2 Development bounded run

Development bounded run is the next-step evidence layer for `T15`.

It must still stay within the same realism boundary as recovery smoke:

- P4 wrapper over the same software HIL stack
- `mock` backend only
- no `.tflite` runtime
- no `real_board`
- no benchmark code or config edits

The difference is only that `T15` may expand scenario/mode/repeat coverage in a controlled way, using the existing runner CLI and frozen config semantics.

### 2.3 Formal frozen benchmark

Formal frozen benchmark refers to the historical multi-scenario strong-baseline protocol used for formal comparison claims.

- Config family: `cnn_fpga/config/p4_multiscenario_strong_baselines.yaml`
- Base long config: `cnn_fpga/config/p4_multiscenario_hybrid_b_long.yaml`
- Inheritance note: `p4_multiscenario_strong_baselines.yaml` declares `base_config: p4_multiscenario_hybrid_b_long.yaml`, so the strong-baseline protocol inherits the long HIL/benchmark defaults and then overrides the frozen baseline set.
- Historical formal scenario set:
  - `static_bias_theta`
  - `linear_ramp`
  - `step_sigma_theta`
  - `periodic_drift`
- Historical frozen baseline set:
  - `ekf`
  - `ukf`
  - `constant_residual_mu`
  - `rls_residual_b`
  - `hybrid_residual_b`

This layer is not automatically re-opened by `T14`. It remains a later decision after bounded evidence collection.

## 3. Fixed Evidence Already Available

### 3.1 Recovery smoke evidence from `T9`

`T9` already re-verified the following bounded run:

- Run dir: `runs/p4_benchmark/p4multis_20260508_121828_0c12d7_46732`
- Config: `p4_multiscenario_recovery_smoke.yaml`
- Scenario: `static_bias_theta`
- Modes:
  - `static_linear`
  - `window_variance`
  - `ekf`
  - `cnn_fpga`
- Repeats: `1`
- Seed pairing: `paired`

Observed winner in that bounded run:

- winner: `window_variance`
- runner-up gap: `0.10509375`

This is enough to prove the current recovery path can exercise the frozen smoke baseline subset, but not enough to claim restored formal P4 frozen benchmark coverage.

### 3.2 Historical formal evidence referenced by development work

`docs/02_experiment_plan.md` still records the historical formal strong-baseline setting:

- scenario set: `static_bias_theta / linear_ramp / step_sigma_theta / periodic_drift`
- baseline set: `ekf / ukf / constant_residual_mu / rls_residual_b / hybrid_residual_b`
- historical conclusion: `Hybrid Residual-B` was the strongest formal mainline method, and `UKF` the strongest classical baseline

`T14` does not re-prove those results. It only uses them as the reference protocol that development runs must not silently redefine.

## 4. Frozen Protocol Items That Must Not Change In T15

The following items are frozen for bounded development evidence collection:

1. Do not modify `cnn_fpga/benchmark/run_p4_multiscenario_benchmark.py`.
2. Do not modify `cnn_fpga/config/p4_multiscenario_recovery_smoke.yaml`.
3. Do not modify `cnn_fpga/config/p4_multiscenario_strong_baselines.yaml`.
4. Do not modify baseline names, scenario names, or ParamMapper semantics.
5. Do not reinterpret `mock` results as `real_board` or `.tflite` evidence.
6. Do not introduce new benchmark modes into `T15`.
7. Do not start long-run formal sweeps outside the bounded matrix below.

## 5. Runner Capabilities Confirmed In Code

`cnn_fpga/benchmark/run_p4_multiscenario_benchmark.py` already supports the controls needed for bounded development work:

- `--scenario`
- `--mode`
- `--repeats`
- `--paired-seeds`
- `--run-dir`
- `--repeat-start`
- `--repeat-stop`
- `--resume-only`

That means `T15` can expand evidence without changing code, and can chunk or resume work if the bounded matrix is split across sessions.

## 6. Approved T15 Bounded Run Plan

### 6.1 Scope

`T15` should stay on the formal strong-baseline config family, but limit run size explicitly.

- Interpreter: `C:\ProgramData\anaconda3\python.exe`
- Entry: `python -m cnn_fpga.benchmark.run_p4_multiscenario_benchmark`
- Config: `cnn_fpga/config/p4_multiscenario_strong_baselines.yaml`
- Backend expectation: inherited software HIL path, still documented as `mock-backed P4 wrapper over HIL`
- Seed policy: `--paired-seeds`
- Repeat policy: explicit CLI override `--repeats 2`

### 6.2 Bounded matrix

Recommended `T15` bounded matrix:

- Scenarios:
  - `static_bias_theta`
  - `linear_ramp`
- Modes:
  - `ekf`
  - `ukf`
  - `constant_residual_mu`
  - `rls_residual_b`
  - `hybrid_residual_b`
- Repeats:
  - `2`

Why this matrix:

1. It upgrades from single-scenario smoke to multi-scenario evidence.
2. It uses the formal strong-baseline baseline set without redefining it.
3. It keeps runtime bounded by not immediately reopening all four formal scenarios.
4. It includes one static scenario and one dynamic scenario, which is enough to test whether the frozen formal ordering is directionally preserved before any larger run.

### 6.3 Scenarios intentionally deferred

The following are deferred beyond `T15` unless a later task package says otherwise:

- `step_sigma_theta`
- `periodic_drift`

Reason:

- They belong to the formal scenario set, but adding both now would turn `T15` from bounded evidence collection into a larger re-opened benchmark pass.

## 7. T15 Command Drafts

### 7.1 Single-command bounded run

```powershell
& 'C:\ProgramData\anaconda3\python.exe' -m cnn_fpga.benchmark.run_p4_multiscenario_benchmark --config cnn_fpga/config/p4_multiscenario_strong_baselines.yaml --scenario static_bias_theta --scenario linear_ramp --mode ekf --mode ukf --mode constant_residual_mu --mode rls_residual_b --mode hybrid_residual_b --paired-seeds --repeats 2
```

### 7.2 Chunked / resumable form

If `T15` needs bounded chunking, use the existing runner controls instead of editing code:

```powershell
& 'C:\ProgramData\anaconda3\python.exe' -m cnn_fpga.benchmark.run_p4_multiscenario_benchmark --config cnn_fpga/config/p4_multiscenario_strong_baselines.yaml --scenario static_bias_theta --scenario linear_ramp --mode ekf --mode ukf --mode constant_residual_mu --mode rls_residual_b --mode hybrid_residual_b --paired-seeds --repeats 2 --run-dir <fixed_run_dir> --repeat-start 0 --repeat-stop 1
```

```powershell
& 'C:\ProgramData\anaconda3\python.exe' -m cnn_fpga.benchmark.run_p4_multiscenario_benchmark --config cnn_fpga/config/p4_multiscenario_strong_baselines.yaml --scenario static_bias_theta --scenario linear_ramp --mode ekf --mode ukf --mode constant_residual_mu --mode rls_residual_b --mode hybrid_residual_b --paired-seeds --repeats 2 --run-dir <fixed_run_dir> --repeat-start 1 --repeat-stop 2
```

```powershell
& 'C:\ProgramData\anaconda3\python.exe' -m cnn_fpga.benchmark.run_p4_multiscenario_benchmark --config cnn_fpga/config/p4_multiscenario_strong_baselines.yaml --run-dir <fixed_run_dir> --resume-only
```

## 8. Required Reporting Labels For T15

Any `T15` report must explicitly state:

1. this is a `development bounded run`, not a restored formal benchmark;
2. the config file used;
3. the exact scenario filter;
4. the exact mode filter;
5. the repeat count;
6. whether `--paired-seeds` was enabled;
7. whether the execution was chunked or resumable;
8. that the path is still a P4 wrapper over software HIL;
9. that it is not `real_board`;
10. that it is not `.tflite` runtime evidence.

## 9. Exit Criteria For T15

`T15` should be considered successful only if it produces all of the following without changing benchmark semantics:

1. a bounded multi-scenario run under the matrix in Section 6;
2. `summary.json`, `comparison.csv`, `delta.csv`, `report.md`, and `progress.jsonl`;
3. explicit filter evidence showing the intended scenarios, modes, repeats, and paired seeds;
4. enough documentation for `T16` to judge whether the project should:
   - extend to the full four-scenario formal set,
   - stop at bounded evidence,
   - or redirect effort to manifests / training / `.tflite` / cleanup tasks.

## 10. T15 Execution Record

`T15` has now been executed under the bounded matrix defined above.

- Run dir: `runs/p4_benchmark/p4multis_20260508_221718_b82874_48280`
- Interpreter: `C:\ProgramData\anaconda3\python.exe`
- Config: `cnn_fpga/config/p4_multiscenario_strong_baselines.yaml`
- Scenarios:
  - `static_bias_theta`
  - `linear_ramp`
- Modes:
  - `ekf`
  - `ukf`
  - `constant_residual_mu`
  - `rls_residual_b`
  - `hybrid_residual_b`
- Repeats: `2`
- Seed policy: `--paired-seeds`

Execution note:

- the first full command exceeded the single shell timeout window;
- the run was then resumed on the same `run_dir`, which is allowed by Section 7.2 and did not change benchmark semantics;
- final `summary.json` reports `missing_runs = []` and full coverage for all scenario/mode pairs.

Key bounded results:

- `static_bias_theta`
  - winner: `hybrid_residual_b`
  - `final_ler_mean = 0.8109015277777778`
  - runner-up: `ukf`
  - `runner_up_gap = 0.014468888888888864`
- `linear_ramp`
  - winner: `hybrid_residual_b`
  - `final_ler_mean = 0.7877551388888888`
  - runner-up: `ukf`
  - `runner_up_gap = 0.023445694444444554`

Boundary checks observed in the generated evidence:

- all modes remained `backend = mock`
- all checked repeats remained `inference_service_mode = inproc`
- `hybrid_residual_b` used artifact:
  - `artifacts/models/runtime_b_residual_v1/tiny_cnn_20260401_083648_2fc740424c0d.npz`
- dominant overflow source stayed `histogram_input`
- `correction_saturation_rate_mean = 0.0`
- `aggressive_param_rate_mean = 0.0`

This execution upgrades the project from single-scenario recovery smoke to bounded multi-scenario development evidence. It still does not restore the full four-scenario formal frozen benchmark.

## 11. What T14 And T15 Still Do Not Claim

`T14` and `T15` do not claim:

1. formal P4 frozen benchmark has been restored;
2. historical strong-baseline conclusions have been re-run on this machine;
3. `real_board` HIL is ready;
4. `.tflite` runtime is restored;
5. the full four-scenario formal matrix has been re-opened;
6. later workers may expand beyond the bounded matrix above without a new task package.

## 12. Relationship To T23 Formal Protocol

`T23` adds `docs/protocols/benchmark/P4_benchmark_formal_protocol.md` as the locked next-step reference for the recovered four-scenario frozen-set revalidation path.

Use the two protocol documents as follows:

1. this document remains the boundary for `T15`-style development evidence
2. `docs/protocols/benchmark/P4_benchmark_formal_protocol.md` defines the later formal software benchmark revalidation scope
3. neither document upgrades the current project state to `.tflite` runtime or `real_board` validation
