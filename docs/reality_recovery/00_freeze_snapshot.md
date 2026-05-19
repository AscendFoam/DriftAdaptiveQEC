# Freeze Snapshot

Freeze date: `2026-05-19`

This document captures the current project posture in one place. It is a governance freeze, not an evidence upgrade.

## 1. Current Posture

- Phase: `Phase 2: Controlled Development`
- Decision state: `Go`
- Sub-mode: `Research Reality Recovery Mode`
- Current unique task: `T44: Research Reality Recovery Mode setup and evidence-gap ledger`
- T43 verdict: `PASS`; blocking issues: none; all warnings `accepted`
- Paper prose expansion: **paused** after T43
- Source of truth: `docs/04_task_board.md`, `docs/07_handoff.md`, `docs/08_risks_and_open_questions.md`, `docs/paper_claim_evidence_ledger.md`

## 2. What Has Been Verified

### 2.1 Software HIL Recovery (C1, `supported`)

- A bounded `mock-backed software HIL` path exists and was recovered, revalidated, and confirmed deterministic (T6, T12).
- Backend: `mock` + `model_artifact` + `artifact_npz` + `inproc`.
- Two consecutive runs produce byte-identical `hil_summary.json` and `hil_events.json` (SHA256 match).
- Evidence: `docs/P3_software_hil_bootstrap.md`, `docs/03_hil_p4_boundary_audit.md`.

### 2.2 Frozen-Set Formal Software Revalidation (C2, C3, `supported`)

- T24 executed `4 scenarios x 5 modes x repeats=2` under locked protocol.
- `missing_runs = []`; all 20 scenario/mode pairs `coverage = 1.0`; 40 repeat-runs completed.
- `hybrid_residual_b` wins all four scenarios; runner-up in all four is `ukf`.
- Evidence: `docs/P4_benchmark_formal_protocol.md`, `runs/p4_benchmark/T24_formal_software_revalidation_20260510_200743/`.
- Boundary: mock-backed software HIL only. Not `.tflite` runtime, not real-board, not paper-grade expanded benchmark.

### 2.3 Seed=20260429 Mechanism Diagnosis (C4, `partial`)

- T36 narrowed the failure to residual-amplitude / teacher-delta regime instability.
- T38 exported 4798 trace rows with per-window committed-`b`, teacher-`b`, and predicted `delta_b`.
- Trace evidence supports combined committed-`b` instability for `seed=20260429`.
- Boundary: single-seed trace-supported diagnosis. Not causal proof, not multi-seed confirmation, not mitigation success.

### 2.4 Clean-Environment Training Smoke (C5, `supported`)

- T39 established a clean CPU-only Windows/Python 3.12 environment with draft dependency lock.
- T40 completed one real isolated `tiny_cnn` training smoke in that environment.
- Boundary: one clean-environment CPU-only smoke. Not full reproducibility, not GPU/CUDA portability, not Linux portability.

### 2.5 Statcalib Interface Contract (C9, `supported`)

- T26 passed feasibility gate (`CONDITIONAL_GO`) for statcalib as separate comparator lane.
- T30 produced `cnn_fpga/decoder/statcalib.py` with typed `StatCalibInput` / `StatCalibOutput` and 6 focused tests.
- Boundary: interface-contract only. Not integrated into slow-loop runtime, formal benchmark, `.tflite` runtime, or real-board path.

### 2.6 Code Truth (verified by T44 audit)

- `board_backend.py`: header says "Placeholder" but file is a 308-line structurally complete implementation with `MemoryMappedRegisterIO`, `MemoryMappedDMARegion`, and `BoardFPGA`. The "placeholder" label is a legacy misnomer. However, no real hardware has been connected, so the evidence boundary remains at readiness-plan level.
- `export.py`: stub fallback path (`tflite_stub_v1`) still present. True `.tflite` export fails gracefully.
- `inference_service.py`: correctly distinguishes `tflite_service` vs `tflite_stub_service` in metadata.
- `statcalib.py`: purely interface/schema module. No algorithmic calibration logic. Not imported by `slow_loop_runtime.py`.
- `fast_loop_emulator.py`: `correction_saturation_rate` computation is structurally sound. Whether it is a genuine zero depends on `LinearRuntime.decode()` saturation logic and runtime parameters.
- `syndrome_measurement.py`: `RealisticSyndromeMeasurement` uses explicit `rng` injection. Fast-loop emulator passes dedicated `measurement_rng`.
- No TODO/FIXME/HACK/PLACEHOLDER marker comments in `cnn_fpga/` or `physics/`.

### 2.7 Paper Scaffolding

- Paper claim/evidence ledger exists: `docs/paper_claim_evidence_ledger.md` (C1-C11, F1-F3, T1-T5).
- Paper draft skeleton exists: `docs/paper_draft_skeleton.md`.
- Paper method-positioning calibration exists: `docs/paper_method_positioning_calibration.md`.
- Paper reviewer-risk audit exists: `docs/paper_reviewer_risk_audit.md`.
- Paper Background / Related Work prose draft exists: `docs/paper_background_related_work_draft.md`.
- All paper artifacts are evidence-bounded. None upgrade mock/stub/readiness evidence to stronger claims.

## 3. What Remains Unverified

| Item | Risk IDs | Evidence gap |
| --- | --- | --- |
| True `.tflite` runtime on current machine | R12 | `tensorflow` / `tflite_runtime` not installed; true runtime never executed |
| Real-board HIL validation | R13, R14 | No device connected; no board logs; AXI/DMA contract unverified with real hardware |
| Multi-seed mechanism closure | R10 | Only `seed=20260429` has trace-level evidence; no intervention or multi-seed confirmation |
| Correction saturation triggerability | R20 | Metric computation is structurally sound but observed value is identically 0.0 across all T24 runs |
| Training reproducibility beyond one smoke | R11 | Only one CPU-only clean-env run; no cross-host, cross-OS, GPU, or repeated-run evidence |
| Benchmark expansion beyond frozen set | R5, R9 | Four scenarios, five modes, two repeats only; no extra drift families, no CI-driven stopping |
| Integrated statcalib comparator evidence | R24 | Interface contract exists but is not wired into slow loop or formal benchmark |
| Paper-ready figure/material pack | — | F1 partial, F3 blocked; no stable regeneration path for most target figures |

## 4. What Must Not Be Claimed

The following statements must not appear in any paper section, review, or governance document as completed facts:

1. "hardware validated" or "real-board HIL complete"
2. "deployment-ready" or "TFLite deployed"
3. "training reproducible" or "training reproducibility established"
4. "mechanism proven" or "root cause found"
5. "comprehensive benchmark" or "broad evaluation"
6. "integrated calibrated comparator" or "statcalib benchmark advantage"
7. "generally superior decoder" or "state-of-the-art"
8. "nearly deployment-ready" or "effectively reproducible"

## 5. Mainline Blockers vs Boosters vs Extensions

### 5.1 Mainline Paper-Readiness Blockers

These block a strong paper submission if unresolved:

1. Paper-grade benchmark broadening (T45)
2. Multi-seed mechanism/intervention evidence (T46)
3. Paper ablation result-pack and material ledger (T47)

### 5.2 Strong-Quality Boosters

These materially raise paper quality but are not hard blockers for a minimum evidence-bounded paper:

1. True `.tflite` runtime validation (T48)
2. Bounded real-board smoke evidence (T49)
3. Training reproducibility and material-regeneration pack (T50)

### 5.3 Future Extension Lane

Items from `docs/reference/延伸改进思路.md` and similar reference documents remain:
- `future_extension_reference`
- Not current mainline truth
- Not prerequisites for the minimum evidence-bounded paper thesis

## 6. Roadmap Sufficiency Judgment

1. **T44 alone is insufficient** for strong-submission standard. It freezes truth but does not create missing evidence.
2. **Currently visible pending tasks (T45-T52)** are the correct next bounded tasks, but they are still proposed and unexecuted.
3. **Even if T48/T49 both pass**, the paper would still need T45-T47 for mainline evidence hardening before a strong submission is defensible.
4. **The current roadmap direction is correct** but the paper must remain evidence-bounded until mainline blockers are resolved through bounded execution tasks.
