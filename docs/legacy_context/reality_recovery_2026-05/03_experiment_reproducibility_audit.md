# Experiment Reproducibility Audit

Freeze date: `2026-05-19`

This document audits what is reproducible now and what is not, across environment, seed/config, benchmark/smoke, and runtime boundaries.

## 1. What Is Reproducible Now

### 1.1 P0 Smoke

- Command: `python benchmark/compare_full_vs_simplified_ler.py --n-rounds 10 --repeats 2 --no-plot --output-dir runs/smoke_test_anaconda`
- Interpreter: `C:\ProgramData\anaconda3\python.exe`
- Dependencies: `numpy + PyYAML` (`requirements-recovery.txt`)
- Rerunnable: yes, within recovery-scoped constraints
- Evidence: `runs/smoke_test_anaconda/n10_r2_s0.250_ler_curve_compare.csv`, `runs/smoke_test_anaconda/n10_r2_s0.250_summary.json`

### 1.2 P3 Software HIL Recovery Smoke

- Command: `python -m cnn_fpga.benchmark.run_hil_suite --config cnn_fpga/config/hardware_hil_recovery_smoke.yaml`
- Backend: `mock` + `model_artifact` + `artifact_npz` + `inproc`
- Deterministic: yes (T12 confirmed byte-identical SHA256 across two runs)
- Artifact: `artifacts/models/static_theta_v2/tiny_cnn_20260319_151717_b87c6c227b57.npz`
- Evidence: two run dirs with matching SHA256

### 1.3 P4 Frozen-Set Formal Software Revalidation

- Command: repeat-chunked CLI under `docs/protocols/benchmark/P4_benchmark_formal_protocol.md`
- Config: `p4_multiscenario_hybrid_b_long.yaml` (locked by T23)
- Matrix: 4 scenarios x 5 modes x repeats=2, paired seeds
- Evidence: `runs/p4_benchmark/T24_formal_software_revalidation_20260510_200743/`
- Protocol locked: yes (T23)
- Rerunnable: yes, within the frozen-set boundary

### 1.4 Clean CPU-Only Training Smoke

- Interpreter: `.venvs/t39_train_cpu_py312/` (Python 3.12, CPU-only)
- Dependencies: `requirements-train-cpu-win-py312.txt` (draft lock)
- Command: one real training run with `device=cpu`, `backend=numpy`
- Evidence: `artifacts/t40_train_smoke/` (task-scoped isolated outputs)
- Rerunnable: the clean-env setup is documented; the exact smoke can be repeated

## 2. What Is Partially Reproducible

### 2.1 Seed=20260429 Mechanism Diagnosis

- Trace evidence exists: `runs/T38_seed20260429_trace_probe_20260513/trace_export/trace_rows.csv` (4798 rows)
- Reproducible as a frozen artifact lane: the trace can be re-examined
- Not yet reproducible as a multi-seed or intervention-backed mechanism package: only one seed, no counterfactual
- Boundary: single-seed trace-supported diagnosis

### 2.2 Figure/Table Input Data

- Core tables (T1, T2, T3, T4 from paper ledger) can be reconstructed from frozen CSV/JSON sources
- Core figures lack stable regeneration scripts/paths (no frozen figure-generation pipeline)
- Boundary: data exists; regeneration scripts not frozen

### 2.3 Teacher Diagnostics

- T28 repaired missing-vs-zero semantics in the output writer
- Current output correctly distinguishes `not_applicable`, `not_generated`, and observed zero
- However, `hybrid_residual_b` still produces `not_generated` teacher diagnostics because it uses broadcast teacher features rather than scalar features
- Boundary: writer semantics repaired; mechanism evidence still not fully generated

## 3. What Is Not Reproducible Enough for Strong Submission

### 3.1 Training

- No cross-host reproducibility matrix
- No cross-OS (Linux/macOS) matrix
- No GPU/CUDA training evidence
- No repeated clean-env training series
- Blocker: R11

### 3.2 Runtime / Deployment

- No true `.tflite` runtime validation (tensorflow/tflite_runtime not installed)
- No real-board run pack
- Blocker: R12, R13, R14

### 3.3 Mechanism

- No multi-seed mechanism confirmation
- No targeted counterfactual or intervention evidence
- No correction saturation triggerability proof
- Blocker: R10, R20

### 3.4 Benchmark Breadth

- No declared paper-grade benchmark expansion beyond the frozen set
- No formal ablation/result pack
- Blocker: R5, R9

## 4. Seed / Run / Config Facts

| Item | Current status | Reproducible? |
| --- | --- | --- |
| T24 formal revalidation seeds | Paired seeds; `repeats=2`; locked protocol | Yes (within frozen set) |
| T12 deterministic HIL seeds | Explicit seed chain in fast loop and measurement noise | Yes (byte-identical) |
| T38 trace probe seed | `seed=20260429` only | Frozen artifact only |
| T40 training smoke config | Task-scoped derived config; isolated output paths | Yes (one clean env) |
| T15 development run seeds | `repeats=2`, paired seeds | Yes (but labeled `development_smoke`, not formal) |
| Multi-seed mechanism evidence | Does not exist | No |
| Broader benchmark seeds | Does not exist | No |

## 5. What Would Still Be Needed

For stronger reproducibility claims, the following would be needed:

1. Repeatability matrix for training: at least 3 seeds, ideally across 2+ host configurations
2. Multi-seed benchmark/mechanism pack: at least 3-5 seeds for mechanism diagnosis
3. Figure/table regeneration manifest: frozen scripts that regenerate each target figure from frozen data
4. True `.tflite` runtime pack: validated export + inference on a machine with TensorFlow
5. Bounded board smoke pack: if hardware-validation language remains in scope

## 6. Verdict

Current reproducibility is enough for:
- bounded recovery claims
- narrow software-HIL benchmark claims
- one clean-environment training smoke claim

Current reproducibility is not enough for:
- strong reproducibility positioning in the paper
- strong deployment or runtime positioning
- strong hardware validation positioning
- broad benchmark generalization claims
