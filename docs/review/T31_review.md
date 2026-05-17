# T31 Adversarial Review: Training-Chain Portable Dependency Lock Plan

**Reviewer:** Independent adversarial review
**Date:** 2026-05-17
**Task package:** `docs/tasks/Phase2/T31_training_chain_portable_dependency_lock_plan.md`

---

## Verdict: PASS

---

## 1. Blocking Issues

None.

## 2. Task Completion Assessment

T31 要求产出 8 类内容。逐项确认：

| # | Expected Output | Present | Verified |
|---|----------------|---------|----------|
| 1 | Current local training interpreter inventory | Yes | Section 2 of plan doc; three interpreters inventoried with versions |
| 2 | Package evidence from DLEnv and fallback interpreters | Yes | Section 3; DLEnv, base Anaconda, system Python probed |
| 3 | Training entrypoint dependency map (static theta, residual-b, Gated v5) | Yes | Sections 4.2–4.4; all three paths mapped with config facts |
| 4 | Proposed lock strategy (CPU-only, CUDA caveat, OS constraints) | Yes | Section 5; two-lane strategy with naming pattern |
| 5 | What can be committed now vs local evidence | Yes | Section 6; clean separation |
| 6 | Clean-environment bootstrap procedure proposal | Yes | Section 7; step-by-step procedure with dry-run verification |
| 7 | Explicit non-claims | Yes | Section 8; 7 explicit non-claims listed |
| 8 | Recommended next bounded task | Yes | Section 9; scope, allowed outputs, required verification, forbidden scope |

All 8 items present and substantive.

## 3. Allowed Files Check

Git diff and status show exactly 4 files touched:

| File | Status | Allowed by T31 |
|------|--------|---------------|
| `docs/tasks/Phase2/T31_training_chain_portable_dependency_lock_plan.md` | Modified | Yes |
| `docs/training_chain_portable_dependency_lock_plan.md` | New | Yes |
| `docs/review/T31_review.md` | New | Yes |
| `docs/for_human/T31_explanation.md` | New | Yes |

No other file was modified or created.

## 4. Forbidden Scope Check

| Forbidden Action | Verified |
|-----------------|----------|
| No package install/upgrade/remove | Confirmed: no pip/conda install commands in evidence |
| No training/benchmark/TFLite/hardware/cleanup execution | Confirmed: only read-only probes |
| No `runs/` or `artifacts/` creation | Confirmed: `git diff --name-only -- runs artifacts` = empty |
| No source code modification | Confirmed: only docs changed |
| No config modification | Confirmed: only docs changed |
| No `requirements-recovery.txt` modification | Confirmed: `git diff --name-only -- requirements-recovery.txt` = empty |
| No claim of clean environment verification | Confirmed: Section 8 explicitly non-claims |

## 5. Source-Code Claim Verification

The plan makes several claims about code behavior. I independently verified:

| Claim | Source Evidence | Verified |
|-------|----------------|----------|
| `experiment_static_theta_v2.yaml` does not force `backend=torch` or `device=cuda` | Config file inspected: `training.model_type: tiny_cnn`, no `backend` or `device` key | Yes |
| `experiment_runtime_b_residual.yaml` does not force `backend=torch` or `device=cuda` | Config file inspected: no `backend` or `device` key | Yes |
| `experiment_runtime_b_residual_norm_gated_teacher_v5.yaml` inherits from residual-b | Config line 10: `base_config: experiment_runtime_b_residual.yaml` | Yes |
| Gated v5 does not force `backend=torch` | Config inspected: only feature/layout changes, no backend override | Yes |
| `tiny_cnn.py` defaults to `backend="numpy"`, `device="auto"` | Source code line 29–30 in `TinyCNNConfig` dataclass | Yes |
| `train.py` does not import torch at module level | Source code inspected: only numpy + stdlib imports | Yes |
| `dataset_builder.py` only imports numpy externally | Source code inspected: confirmed | Yes |
| `runtime_dataset_builder.py` only imports numpy externally | Source code inspected: confirmed | Yes |
| `config.py` has a YAML fallback parser | `_load_yaml_fallback()` exists at lines 250–259 | Yes |

All code-level claims are accurate.

## 6. Non-Claims Verification

The plan explicitly states it does NOT claim (Section 8):

1. Clean environment rebuilt — confirmed not claimed
2. DLEnv as canonical environment — confirmed not claimed
3. Nightly torch as portable — confirmed not claimed
4. Linux portability — confirmed not claimed
5. `.tflite` coverage — confirmed not claimed
6. Real-board/benchmark coverage — confirmed not claimed
7. `requirements-recovery.txt` expansion — confirmed not claimed

Additionally, the plan consistently uses qualifier language: "local fact only", "portable CPU-lock candidate", "not yet verified". No instance of local evidence being overstated as portable guarantee was found.

## 7. Non-Blocking Issues

### N1: Section numbering formatting inconsistency

`docs/training_chain_portable_dependency_lock_plan.md` uses `## 4.1`, `## 4.2`, `## 4.3`, `## 5.1`–`## 5.4`, `## 6.1`–`## 6.2` as sub-headings under `## 4` / `## 5` / `## 6`. These should be `### 4.1` etc. for correct Markdown nesting. The content is unaffected; this is cosmetic.

**Classification:** accepted, cosmetic only.

### N2: Cross-document consistency with `training_chain_bootstrap.md`

`docs/training_chain_bootstrap.md` (from T17) currently writes the DLEnv interpreter as the single "recommended interpreter" for training. T31's plan document adds a more nuanced view: CPU-only lane is plausible without torch. The two documents are not contradictory, but the bootstrap doc could be updated later to reference the T31 plan's two-lane finding.

**Classification:** accepted, future alignment; not a T31 scope issue.

### N3: Worker self-review file structure

The worker wrote `docs/review/T31_review.md` as a "worker-side review input" with the note "independent adversarial review is still pending". This is consistent with the task package's allowed files, and the reviewer (me) is now overwriting it with the adversarial review. The self-check structure was reasonable scaffolding.

**Classification:** accepted.

## 8. Missing Validation

None beyond what T31's scope already acknowledges. T31 is explicitly a plan/boundary task. The clean-environment validation is correctly deferred to a follow-up task.

## 9. Suspicious Implementation Details

None found. The plan document is well-structured, honest about its boundaries, and its code-level claims are verified against source.

## 10. Summary Assessment

**Strengths:**

- Clean separation of local evidence vs portable guarantees throughout all 4 output files
- Code-level claims verified independently and found accurate
- The two-lane strategy (CPU-portable + GPU-local) is well-reasoned given the config semantics
- The recommended next task has concrete scope, verification steps, and forbidden scope
- Verification record is thorough and reproducible
- Non-claims section is comprehensive

**No weaknesses found that would warrant downgrading from PASS.**

## 11. Recommended Next Action

T31 can be accepted as complete. The recommended next task (clean-environment CPU-only draft lock + dry-run bootstrap) is well-scoped and should be the next bounded task in Milestone 2J.

Captain should:

1. Accept T31 as `PASS`
2. Update task board to mark T31 complete
3. Create the next bounded task per T31 Section 9 recommendation
4. Update handoff document
