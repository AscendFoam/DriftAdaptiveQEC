# T33: Tracked cache physical cleanup execution, only within T19 manifest

## Status

- Created by Captain on `2026-05-17`
- Current phase: `Phase 2: Controlled Development`
- Decision state: `Go`
- Task type: bounded repo-hygiene execution

## Why This Task Exists

`T19` completed the read-only manifest for tracked cache cleanup and fixed the exact cleanup scope to 9 `__pycache__` directories containing 116 tracked `.pyc` files. The manifest already includes preflight commands, bounded untrack commands, rollback guidance, and acceptance criteria.

`T32` remains blocked by missing true `.tflite` runtime dependencies on the current machine, and `T37` remains blocked by hardware/bitstream readiness. `T33` is therefore the next bounded task that is both actionable and already supported by a validated manifest.

## Goal

Execute the tracked-cache cleanup physically, but only for the manifest-listed `__pycache__/` / `.pyc` targets. Remove them from Git tracking without widening scope into `runs/`, `artifacts/`, source changes, or any broader cleanup campaign.

## Allowed Files

Worker may modify:

- `docs/tasks/Phase2/T33_tracked_cache_physical_cleanup_execution.md`
- `docs/cleanup_tracked_cache_manifest.md`
- `docs/review/T33_review.md`
- `docs/for_human/T33_explanation.md`

Worker may stage/untrack only these manifest-listed directories:

- `cnn_fpga/__pycache__`
- `cnn_fpga/benchmark/__pycache__`
- `cnn_fpga/data/__pycache__`
- `cnn_fpga/decoder/__pycache__`
- `cnn_fpga/hwio/__pycache__`
- `cnn_fpga/model/__pycache__`
- `cnn_fpga/runtime/__pycache__`
- `cnn_fpga/utils/__pycache__`
- `physics/__pycache__`

## Required Inputs

Read at minimum:

- `README.md`
- `AGENTS.md`
- `docs/00_project_snapshot.md`
- `docs/01_legacy_audit.md`
- `docs/02_experiment_plan.md`
- `docs/04_task_board.md`
- `docs/07_handoff.md`
- `docs/08_risks_and_open_questions.md`
- `docs/06_repo_noise_governance.md`
- `docs/cleanup_tracked_cache_manifest.md`
- `docs/tasks/Phase2/T19_tracked_cache_cleanup_manifest.md`
- `.gitignore`

## Required Execution Boundary

T33 must treat `docs/cleanup_tracked_cache_manifest.md` as the only execution manifest.

That means:

1. do preflight against the listed 9 target directories
2. do not add new cleanup targets
3. do not infer adjacent cleanup opportunities
4. do not touch any path outside the manifest, even if it also looks like cache/noise

## Forbidden Scope

Do not:

- modify `docs/02_experiment_plan.md`
- touch `runs/`
- touch `artifacts/`
- touch source code, configs, tests, benchmark protocol, training outputs, `.tflite` paths, or hardware paths
- expand cleanup into `.pytest_cache/`, `.mypy_cache/`, temp files, logs, or any non-manifest noise
- use destructive reset/rewrite commands such as `git reset --hard`
- mix unrelated cleanup with this task

## Expected Output

Create or update:

1. this task package with Worker Output and Verification Record
2. `docs/review/T33_review.md`
3. `docs/for_human/T33_explanation.md`
4. if needed, a small execution note appended to `docs/cleanup_tracked_cache_manifest.md`

## Required Verification

Before cleanup, run the manifest preflight:

```powershell
git ls-files -- "cnn_fpga/__pycache__/*" "cnn_fpga/benchmark/__pycache__/*" "cnn_fpga/data/__pycache__/*" "cnn_fpga/decoder/__pycache__/*" "cnn_fpga/hwio/__pycache__/*" "cnn_fpga/model/__pycache__/*" "cnn_fpga/runtime/__pycache__/*" "cnn_fpga/utils/__pycache__/*" "physics/__pycache__/*"
```

Then execute bounded untrack only for those 9 directories:

```powershell
git rm --cached -r -- cnn_fpga/__pycache__ cnn_fpga/benchmark/__pycache__ cnn_fpga/data/__pycache__ cnn_fpga/decoder/__pycache__ cnn_fpga/hwio/__pycache__ cnn_fpga/model/__pycache__ cnn_fpga/runtime/__pycache__ cnn_fpga/utils/__pycache__ physics/__pycache__
```

After cleanup, verify:

```powershell
git ls-files | rg "__pycache__|\\.pyc$"
git diff --name-only -- runs artifacts
git diff --name-only -- . ":(exclude)cnn_fpga/__pycache__" ":(exclude)cnn_fpga/benchmark/__pycache__" ":(exclude)cnn_fpga/data/__pycache__" ":(exclude)cnn_fpga/decoder/__pycache__" ":(exclude)cnn_fpga/hwio/__pycache__" ":(exclude)cnn_fpga/model/__pycache__" ":(exclude)cnn_fpga/runtime/__pycache__" ":(exclude)cnn_fpga/utils/__pycache__" ":(exclude)physics/__pycache__"
```

Expected result:

- `git ls-files | rg "__pycache__|\\.pyc$"` returns `0` lines
- `git diff --name-only -- runs artifacts` returns empty output
- no non-manifest paths appear in the diff

If rollback is needed before commit, follow the manifest:

```powershell
git restore --staged -- cnn_fpga/__pycache__ cnn_fpga/benchmark/__pycache__ cnn_fpga/data/__pycache__ cnn_fpga/decoder/__pycache__ cnn_fpga/hwio/__pycache__ cnn_fpga/model/__pycache__ cnn_fpga/runtime/__pycache__ cnn_fpga/utils/__pycache__ physics/__pycache__
```

## Docs To Update

- `docs/tasks/Phase2/T33_tracked_cache_physical_cleanup_execution.md`
- `docs/review/T33_review.md`
- `docs/for_human/T33_explanation.md`

## Reviewer Type

Adversarial review.

Focus areas:

- cleanup stayed exactly inside the T19 manifest
- no `runs/` / `artifacts/` / source/config/benchmark paths were touched
- tracked `__pycache__/` / `.pyc` entries actually dropped to zero
- rollback path is still available if needed before commit

## Captain Notes

This is a repo-hygiene execution task, not a code or experiment task. Do not "improve" the manifest mid-flight by adding more cleanup scope. If preflight no longer matches the T19 manifest, stop and report the mismatch instead of improvising.

## Worker Output

- Read the required T33 inputs, including:
  - `README.md`
  - `AGENTS.md`
  - `docs/00_project_snapshot.md`
  - `docs/01_legacy_audit.md`
  - `docs/02_experiment_plan.md`
  - `docs/04_task_board.md`
  - `docs/07_handoff.md`
  - `docs/08_risks_and_open_questions.md`
  - `docs/06_repo_noise_governance.md`
  - `docs/cleanup_tracked_cache_manifest.md`
  - `docs/tasks/Phase2/T19_tracked_cache_cleanup_manifest.md`
  - `.gitignore`
- Confirmed the T19 manifest still limited cleanup to these 9 tracked `__pycache__` directories:
  - `cnn_fpga/__pycache__`
  - `cnn_fpga/benchmark/__pycache__`
  - `cnn_fpga/data/__pycache__`
  - `cnn_fpga/decoder/__pycache__`
  - `cnn_fpga/hwio/__pycache__`
  - `cnn_fpga/model/__pycache__`
  - `cnn_fpga/runtime/__pycache__`
  - `cnn_fpga/utils/__pycache__`
  - `physics/__pycache__`
- Ran the required preflight inventory:
  - `git ls-files -- "cnn_fpga/__pycache__/*" "cnn_fpga/benchmark/__pycache__/*" "cnn_fpga/data/__pycache__/*" "cnn_fpga/decoder/__pycache__/*" "cnn_fpga/hwio/__pycache__/*" "cnn_fpga/model/__pycache__/*" "cnn_fpga/runtime/__pycache__/*" "cnn_fpga/utils/__pycache__/*" "physics/__pycache__/*"`
  - confirmed 116 tracked `.pyc` files across the 9 manifest directories
- Executed the bounded cleanup exactly for the manifest-listed directories:
  - `git rm --cached -r -- cnn_fpga/__pycache__ cnn_fpga/benchmark/__pycache__ cnn_fpga/data/__pycache__ cnn_fpga/decoder/__pycache__ cnn_fpga/hwio/__pycache__ cnn_fpga/model/__pycache__ cnn_fpga/runtime/__pycache__ cnn_fpga/utils/__pycache__ physics/__pycache__`
- Did not touch:
  - `runs/`
  - `artifacts/`
  - source code
  - benchmark semantics
  - `.tflite` paths
  - hardware paths
  - any non-manifest cache directories
- Updated:
  - `docs/review/T33_review.md`
  - `docs/for_human/T33_explanation.md`
  - this task package
- Did not mark the task board as complete.

## Verification Record

1. Preflight inventory:
   - command:
     - `git ls-files -- "cnn_fpga/__pycache__/*" "cnn_fpga/benchmark/__pycache__/*" "cnn_fpga/data/__pycache__/*" "cnn_fpga/decoder/__pycache__/*" "cnn_fpga/hwio/__pycache__/*" "cnn_fpga/model/__pycache__/*" "cnn_fpga/runtime/__pycache__/*" "cnn_fpga/utils/__pycache__/*" "physics/__pycache__/*"`
   - result:
     - returned 116 tracked `.pyc` paths
2. Cleanup execution:
   - command:
     - `git rm --cached -r -- cnn_fpga/__pycache__ cnn_fpga/benchmark/__pycache__ cnn_fpga/data/__pycache__ cnn_fpga/decoder/__pycache__ cnn_fpga/hwio/__pycache__ cnn_fpga/model/__pycache__ cnn_fpga/runtime/__pycache__ cnn_fpga/utils/__pycache__ physics/__pycache__`
   - result:
     - passed
     - removed all manifest-listed tracked cache files from Git index
3. Post-cleanup verification:
   - command:
     - `git ls-files | rg "__pycache__|\\.pyc$"`
   - result:
     - returned 0 lines
   - command:
     - `git diff --name-only -- runs artifacts`
   - result:
     - empty output
   - command:
     - `git diff --name-only -- . ":(exclude)cnn_fpga/__pycache__" ":(exclude)cnn_fpga/benchmark/__pycache__" ":(exclude)cnn_fpga/data/__pycache__" ":(exclude)cnn_fpga/decoder/__pycache__" ":(exclude)cnn_fpga/hwio/__pycache__" ":(exclude)cnn_fpga/model/__pycache__" ":(exclude)cnn_fpga/runtime/__pycache__" ":(exclude)cnn_fpga/utils/__pycache__" ":(exclude)physics/__pycache__"`
   - result:
     - no non-manifest path changes were introduced
4. Scope check:
   - result:
     - only the 9 manifest-listed cache directories were cleaned
     - `runs/` and `artifacts/` were not touched
