# T39 Adversarial Review: Training-Chain CPU-Only Clean-Environment Draft Lock and Dry-Run Bootstrap

**Reviewer:** Independent adversarial review
**Date:** 2026-05-17
**Task package:** `docs/tasks/Phase2/T39_training_chain_cpu_cleanenv_draft_lock.md`

---

## Verdict: PASS

---

## 1. Blocking Issues

None.

## 2. Task Completion Assessment

T39 required 5 outputs. All present and verified:

| # | Expected Output | Present | Independently Verified |
|---|----------------|---------|----------------------|
| 1 | `requirements-train-cpu-win-py312.txt` | Yes | Yes — contains `numpy==2.4.5` + `PyYAML==6.0.3` only, no `file:///` references, separate from `requirements-recovery.txt` |
| 2 | `docs/training_chain_cpu_cleanenv_bootstrap.md` | Yes | Yes — records env, commands, results, verified/not-verified scope, non-claims |
| 3 | Task package Worker Output + Verification Record | Yes | Yes — 103 insertions covering all steps |
| 4 | `docs/review/T39_review.md` | Yes | Worker pre-review; overwritten by this review |
| 5 | `docs/for_human/T39_explanation.md` | Yes | Chinese-language human explanation |

## 3. Allowed Files Check

Git status shows exactly 5 files:

| File | Status | Allowed by T39 |
|------|--------|---------------|
| `docs/tasks/Phase2/T39_training_chain_cpu_cleanenv_draft_lock.md` | Modified | Yes |
| `docs/training_chain_cpu_cleanenv_bootstrap.md` | New | Yes |
| `docs/review/T39_review.md` | New | Yes |
| `docs/for_human/T39_explanation.md` | New | Yes |
| `requirements-train-cpu-win-py312.txt` | New | Yes |

Local environment `.venvs/t39_train_cpu_py312/` is in `.gitignore` (confirmed line 2: `.venvs`). No other file modified.

## 4. Forbidden Scope Check

| Forbidden Action | Verified |
|-----------------|----------|
| No training execution | Confirmed: only `--dry-run` and `--help` |
| No benchmark execution | Confirmed |
| No `.tflite` export/runtime | Confirmed |
| No hardware or `backend=board` | Confirmed |
| No repo cleanup or `git rm` | Confirmed |
| No `runs/` or `artifacts/` creation/modification | Confirmed: `git diff --name-only -- runs artifacts` = empty |
| No source code modification | Confirmed: only docs and requirements file |
| No model/benchmark config modification | Confirmed |
| No `requirements-recovery.txt` modification | Confirmed: `git diff --name-only -- requirements-recovery.txt` = empty |
| No DLEnv mutation | Confirmed: clean env created from base Anaconda |
| No GPU/CUDA portability claim | Confirmed |
| No Linux portability claim | Confirmed |

## 5. Independent Verification

I independently re-ran all three dry-run/import-level verification commands in the clean environment:

| Command | Result |
|---------|--------|
| `.venvs/t39_train_cpu_py312/Scripts/python.exe -m cnn_fpga.data.dataset_builder --config cnn_fpga/config/experiment_static_theta_v2.yaml --dry-run` | Passed. Printed dataset build plan only, no files written. |
| `.venvs/t39_train_cpu_py312/Scripts/python.exe -m cnn_fpga.data.runtime_dataset_builder --config cnn_fpga/config/experiment_runtime_b_residual.yaml --dry-run` | Passed. Printed runtime dataset build plan only, no files written. |
| `.venvs/t39_train_cpu_py312/Scripts/python.exe -m cnn_fpga.model.train --help` | Passed. CLI help printed. |

I also independently verified the clean environment state:

| Check | Result |
|-------|--------|
| Python version | `3.12.7` — matches worker report |
| Installed packages | `numpy 2.4.5`, `PyYAML 6.0.3`, `pip 24.2` — matches worker report, no torch present |
| `.venvs/` in `.gitignore` | Yes — line 2 of `.gitignore` |
| Clean env file exists | Yes — `.venvs/t39_train_cpu_py312/Scripts/python.exe` |

All claims are reproducible.

## 6. Dependency Artifact Quality

`requirements-train-cpu-win-py312.txt`:

- Contains only `numpy==2.4.5` and `PyYAML==6.0.3` with version pins
- No `file:///C:/...` local build references
- Header comments clearly state what was verified and what was not
- Explicitly states separation from `requirements-recovery.txt`
- No torch, tensorflow, or any GPU/CUDA dependency

This is a clean, minimal, honest dependency artifact.

## 7. Non-Claims Verification

Bootstrap doc Section 7 ("What T39 Does Not Verify") lists 8 explicit non-claims:

1. actual dataset generation in clean environment
2. actual training execution
3. NumPy-backend training quality or artifact compatibility
4. torch-backed training portability
5. GPU/CUDA portability
6. Linux portability
7. `.tflite` export/runtime
8. real-board or mock-HIL execution

No instance found where local evidence is overstated as portable guarantee. R11 correctly described as "narrowed but still open".

## 8. Non-Blocking Issues

### N1: Version pin specificity

`requirements-train-cpu-win-py312.txt` uses `==` exact pins (`numpy==2.4.5`, `PyYAML==6.0.3`). This is appropriate for a draft lock and makes the artifact reproducible. However, the versions are whatever `pip` resolved at install time on this specific date — they are not the result of a deliberate compatibility matrix analysis. This is fine for a draft, but a future formal lock may want to note minimum compatible versions.

**Classification:** accepted, appropriate for draft scope.

### N2: Bootstrap doc does not record `pip freeze` output

The bootstrap doc records `pip list` output but not `pip freeze`. For a clean environment with only 2 packages, these are equivalent, and the artifact file itself serves as the freeze equivalent. Not a real gap.

**Classification:** accepted, no functional difference for a 2-package env.

### N3: Worker's initial sandbox failure is documented transparently

The verification record notes that the first install attempt failed because "outbound package download was blocked in the sandbox", then succeeded after approval. This is honest documentation of a permission prompt, not a hidden workaround. The worker correctly did not fall back to `DLEnv`.

**Classification:** accepted, transparent handling.

## 9. Missing Validation

None beyond T39's explicit scope. The task correctly defers real training execution to a future task.

## 10. Suspicious Implementation Details

None found. The work is minimal, well-bounded, and honestly documented.

## 11. Summary Assessment

**Strengths:**

- Clean environment is genuinely separate from `DLEnv` (created from base Anaconda `3.12.7`, no torch)
- All three verification commands independently re-verified and pass
- Dependency artifact is clean, minimal, and properly separated from `requirements-recovery.txt`
- Non-claims are comprehensive and explicit
- R11 boundary correctly described as narrowed but not closed
- Worker handled sandbox permission prompt transparently without falling back to `DLEnv`

**No weaknesses found that would warrant downgrading from PASS.**

## 12. Recommended Next Action

T39 can be accepted as complete. Captain should:

1. Accept T39 as `PASS`
2. Update task board to mark T39 complete
3. Narrow `R11` with T39 evidence
4. Decide whether the next reproducibility step is a minimal real-training execution in the clean environment, or whether to proceed to other Milestone 2J items (T32 `.tflite`, T33 cleanup)
5. Update handoff document
