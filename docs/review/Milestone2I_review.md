# Milestone 2I Review: Mechanism Evidence Hardening

## Review Metadata

- Reviewer: Captain milestone review
- Date: `2026-05-16`
- Milestone: `2I: Mechanism Evidence Hardening`
- Scope reviewed:
  - `T26`: Calibration/statcalib baseline feasibility gate
  - `T27`: Teacher diagnostics path audit
  - `T28`: Teacher diagnostics missing-vs-zero semantics repair
  - `T29`: P4 markdown report header cleanup
  - `T30`: Statcalib comparator interface contract
  - `T36`: `seed=20260429` failure-mechanism diagnosis
  - `T38`: `seed=20260429` single-seed trace-export probe

## Verdict

`Conditional Allow`

Milestone 2I may close as a mechanism-evidence hardening milestone. The project may enter the next milestone, but only into reproducibility / deployment-boundary work (`Milestone 2J`). It should not jump directly to paper claims, broad benchmark expansion, statcalib integration, or real-board execution.

## 1. 当前功能是否真的完成

Yes, within the bounded milestone scope.

- T26 established that statcalib is feasible only as a separate comparator lane, not as a silent addition to the frozen benchmark set.
- T27 located the teacher diagnostics path issue: broadcast teacher features do not emit scalar-branch diagnostics.
- T28 repaired missing-vs-zero writer semantics for current outputs.
- T29 fixed the duplicated P4 markdown report header.
- T30 added an interface-only statcalib contract with focused tests.
- T36 produced a bounded diagnosis for `seed=20260429`.
- T38 upgraded that diagnosis with per-window trace evidence from one bounded rerun.

Not completed:

- Full teacher-mechanism causality is not proven across seeds.
- Statcalib is not integrated into slow-loop runtime or formal benchmark evidence.
- No mitigation has been tested for the Gated-v5 amplitude issue.

## 2. 是否能从干净环境运行

Conditional.

The recovery smoke path has a minimal manifest (`requirements-recovery.txt`) and uses `C:\ProgramData\anaconda3\python.exe`. However, the full mechanism-evidence path depends on existing model artifacts and historical run structure. T38 also relies on the local run/artifact layout and the existing Anaconda environment.

Current clean-environment status:

- P0/P3/P4 recovery smoke: documented and partially portable via `requirements-recovery.txt`.
- Training chain: documented in `docs/evidence_packs/training_reproducibility/training_chain_bootstrap.md`, but no portable lockfile.
- TFLite runtime: not available on the current machine.
- Real-board path: not executable without hardware host, permissions, bitstream, and DMA/register evidence.
- T38 trace probe: reproducible only if the required artifacts and local run layout are present.

This does not block closing Milestone 2I, but it directly motivates Milestone 2J.

## 3. 是否有测试、demo 或实验结果

Yes.

Evidence available:

- T24 formal frozen-set software revalidation:
  - `runs/p4_benchmark/T24_formal_software_revalidation_20260510_200743`
  - `missing_runs = []`
  - `coverage = 1.0`
  - `4 scenarios x 5 modes x 2 repeats`
- T28 smoke:
  - verifies `not_applicable` / `not_generated` / observed-zero semantics.
- T29 validation:
  - `py_compile` passed.
  - static report shape check passed with one header row.
- T30 tests:
  - `unittest` passed, `Ran 6 tests`, `OK`.
  - `py_compile` passed.
- T36 diagnosis:
  - `docs/evidence_packs/mechanism_ablation/seed20260429_failure_diagnosis.md`
  - read-only script passed.
- T38 trace probe:
  - `runs/T38_seed20260429_trace_probe_20260513`
  - completed with `missing_runs = []`, `raw_rows = 16`, `comparison_rows = 8`
  - trace export contains `4798` rows with `100%` availability for required fields.

## 4. 是否存在伪完成

No blocking pseudo-completion found.

Boundary checks:

- T24 is still only mock-backed software HIL.
- T30 is only an interface contract, not validated statcalib runtime behavior.
- T36 is summary/final-snapshot diagnosis, not causal proof.
- T38 is single-seed diagnostic trace evidence, not formal benchmark evidence.
- `.tflite` runtime remains unavailable.
- Real-board HIL remains unvalidated.
- Historical `runs/` / `artifacts/` were not rewritten into new facts.

Remaining pseudo-completion risks are documented rather than hidden.

## 5. 是否允许进入下一里程碑

Allowed conditionally.

Allowed next milestone:

- `Milestone 2J: Reproducibility And Deployment Boundary`

Recommended next unique task:

- `T31: Training-chain portable dependency lock plan`

Why T31:

- The milestone review's weakest answer is clean-environment reproducibility.
- T38 closes the immediate observability gap enough to stop adding more trace-only tasks.
- The next high-value work is to make the training/dependency boundary portable and auditable before any mitigation, TFLite, paper, or hardware escalation.

Not allowed yet:

- Paper claim/evidence ledger.
- Statcalib slow-loop integration.
- New teacher-representation branch expansion.
- Broader formal benchmark or CI-driven stopping.
- Real-board execution.

## Evidence-Level Impact

Milestone 2I improves mechanism evidence from "diagnostics path ambiguous" to "single-seed trace-supported instability diagnosis".

It does not upgrade the project to:

- deployment evidence
- hardware evidence
- true `.tflite` runtime evidence
- paper-grade mechanism proof

## Residual Risks

- R10 remains open but substantially narrowed: T38 supports combined committed-`b` instability, but does not isolate whether teacher amplitude or CNN residual amplitude is the first upstream cause.
- R20 remains open: correction saturation remains structurally zero and still needs a separate edge/stress task if the project needs triggerability evidence.
- R11 remains open: training-chain portability is not locked.
- R12/R13/R14 remain open: TFLite and real-board paths are not validated.

## Captain Recommendation

Close Milestone 2I as `Conditional Allow`.

Set `Current Unique Task` to T31 and do not execute it in this Captain turn. T31 should produce a portable training-chain dependency lock plan without installing packages, changing runtime semantics, or launching training.
