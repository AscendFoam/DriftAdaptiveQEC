# T31: Training-chain portable dependency lock plan

## Status

- Created by Captain on `2026-05-16`
- Current phase: `Phase 2: Controlled Development`
- Decision state: `Go`
- Task type: reproducibility / environment-boundary planning

## Why This Task Exists

Milestone 2I closed with `Conditional Allow`. The weakest remaining gate answer is clean-environment reproducibility: recovery smoke has `requirements-recovery.txt`, but the training chain still only has local bootstrap notes and depends on a local `DLEnv` / dev torch build.

T31 should make the training-chain dependency boundary auditable before the project starts mitigation experiments, paper claim work, or runtime/deployment escalation.

## Goal

Produce a portable dependency-lock plan for the training chain.

This task should not install packages or claim that a clean environment has been successfully rebuilt. It should identify what can be locked now, what remains machine-specific, and what exact follow-up would be needed to create a real lockfile.

## Allowed Files

Worker may modify:

- `docs/tasks/Phase2/T31_training_chain_portable_dependency_lock_plan.md`
- `docs/training_chain_portable_dependency_lock_plan.md`
- `docs/review/T31_review.md`
- `docs/for_human/T31_explanation.md`

Worker may read environment metadata using non-mutating commands, for example:

- `C:\ProgramData\anaconda3\envs\DLEnv\python.exe --version`
- `C:\ProgramData\anaconda3\envs\DLEnv\python.exe -m pip freeze`
- import probes for `numpy`, `yaml`, `torch`

Do not write generated lockfiles unless the task report explicitly labels them as drafts and keeps them in docs.

## Required Inputs

Read at minimum:

- `README.md`
- `docs/00_project_snapshot.md`
- `docs/01_legacy_audit.md`
- `docs/02_experiment_plan.md`
- `docs/04_task_board.md`
- `docs/07_handoff.md`
- `docs/08_risks_and_open_questions.md`
- `docs/training_chain_bootstrap.md`
- `requirements-recovery.txt`
- `cnn_fpga/model/train.py`
- `cnn_fpga/config/experiment_static_theta_v2.yaml`
- `cnn_fpga/config/experiment_runtime_b_residual.yaml`
- `cnn_fpga/config/experiment_runtime_b_residual_norm_gated_teacher_v5.yaml`

## Forbidden Scope

Do not:

- modify `docs/02_experiment_plan.md`
- install, upgrade, or remove packages
- run training, benchmark, `.tflite`, hardware, or cleanup commands
- create or modify `runs/` or `artifacts/`
- change source code, model configs, benchmark semantics, formal protocol, baseline/scenario set, or seed/repeat policy
- claim that a clean environment is verified unless the task actually creates and verifies one, which is not expected in T31
- replace `requirements-recovery.txt` or broaden it beyond recovery smoke

## Expected Output

Create `docs/training_chain_portable_dependency_lock_plan.md` with:

1. Current local training interpreter inventory.
2. Package evidence from `DLEnv` and any relevant fallback interpreter.
3. Training entrypoint dependency map:
   - static theta training
   - residual-b training
   - Gated v5 / teacher representation training
4. Proposed lock strategy:
   - CPU-only lock candidate
   - CUDA/dev-torch lock caveat
   - Windows vs Linux constraints
5. What can be committed now vs what must stay as local evidence.
6. Clean-environment bootstrap procedure proposal.
7. Explicit non-claims.
8. Recommended next bounded task, but do not execute it.

## Verification

Required verification:

1. Record all environment/probe commands used.
2. Confirm no package install/update/remove occurred.
3. Confirm no `runs/` or `artifacts/` paths were created or modified.
4. Confirm `requirements-recovery.txt` was not repurposed.
5. Confirm the report distinguishes local `DLEnv` facts from portable guarantees.

## Docs To Update

- `docs/tasks/Phase2/T31_training_chain_portable_dependency_lock_plan.md`
- `docs/training_chain_portable_dependency_lock_plan.md`
- `docs/review/T31_review.md`
- `docs/for_human/T31_explanation.md`

## Reviewer Type

Adversarial review.

Focus areas:

- no package installation or hidden environment mutation
- no benchmark/training execution
- local dev torch facts are not written as portable guarantees
- recovery manifest remains scoped to recovery smoke
- clean-environment claims are not overstated

## Captain Notes

T31 is deliberately a plan/manifest-boundary task. It should make the next reproducibility step executable without pretending the clean environment already exists.

## Worker Output

- Read the required T31 inputs, including:
  - `README.md`
  - `docs/00_project_snapshot.md`
  - `docs/01_legacy_audit.md`
  - `docs/02_experiment_plan.md`
  - `docs/04_task_board.md`
  - `docs/07_handoff.md`
  - `docs/08_risks_and_open_questions.md`
  - `docs/training_chain_bootstrap.md`
  - `requirements-recovery.txt`
  - `cnn_fpga/model/train.py`
  - `cnn_fpga/config/experiment_static_theta_v2.yaml`
  - `cnn_fpga/config/experiment_runtime_b_residual.yaml`
  - `cnn_fpga/config/experiment_runtime_b_residual_norm_gated_teacher_v5.yaml`
- Collected read-only interpreter/package evidence from:
  - `C:\ProgramData\anaconda3\envs\DLEnv\python.exe`
  - `C:\ProgramData\anaconda3\python.exe`
  - `C:\Python313\python.exe`
- Added `docs/training_chain_portable_dependency_lock_plan.md`.
- Added `docs/review/T31_review.md`; independent adversarial review later overwrote this file with final verdict `PASS`.
- Added `docs/for_human/T31_explanation.md`.
- Updated this task package with worker output and verification records.
- Did not install, upgrade, or remove packages.
- Did not run training, benchmark, `.tflite`, hardware, or cleanup commands.
- Did not modify `runs/`, `artifacts/`, source code, configs, formal protocol, baseline/scenario set, or `requirements-recovery.txt`.
- Explicitly separated:
  - local `DLEnv` / CUDA / dev-torch facts
  - from portable CPU-lock candidate statements

## Verification Record

1. Interpreter version probes:
   - `C:\ProgramData\anaconda3\envs\DLEnv\python.exe --version`
   - `C:\ProgramData\anaconda3\python.exe --version`
   - `C:\Python313\python.exe --version`
   - Result:
     - `DLEnv = Python 3.12.9`
     - `base Anaconda = Python 3.12.7`
     - `system Python = Python 3.13.7`
2. Package evidence probes:
   - `C:\ProgramData\anaconda3\envs\DLEnv\python.exe -m pip freeze`
   - `C:\ProgramData\anaconda3\python.exe -m pip freeze`
   - `C:\ProgramData\anaconda3\envs\DLEnv\python.exe -c "import numpy, yaml, torch; ..."`
   - `C:\ProgramData\anaconda3\python.exe -c "import numpy, yaml; ..."`
   - `C:\Python313\python.exe -c "import importlib.util; ..."`
   - Result:
     - local `DLEnv` torch/CUDA facts captured
     - base Anaconda confirmed as `numpy + PyYAML` without `torch`
     - system Python confirmed insufficient for the training chain
3. Entrypoint/import probes:
   - `C:\ProgramData\anaconda3\envs\DLEnv\python.exe -m cnn_fpga.model.train --help`
   - `C:\ProgramData\anaconda3\python.exe -m cnn_fpga.model.train --help`
   - `C:\ProgramData\anaconda3\python.exe -m cnn_fpga.data.dataset_builder --help`
   - `C:\ProgramData\anaconda3\python.exe -m cnn_fpga.data.runtime_dataset_builder --help`
   - Result:
     - all four probes passed
4. No package mutation:
   - no install/update/remove command was used during T31
5. No `runs/` or `artifacts/` modification:
   - `git -c core.excludesFile=NUL diff --name-only -- runs artifacts`
   - Result:
     - empty
6. `requirements-recovery.txt` boundary preserved:
   - checked by inspection during T31; no edit was made and the T31 plan explicitly keeps recovery scope separate from training-chain lock scope
7. Documentation honesty:
   - `docs/training_chain_portable_dependency_lock_plan.md` explicitly distinguishes:
     - local `DLEnv` facts
     - portable CPU-lock candidates
     - non-claims / not-yet-verified areas
8. Closeout boundary:
   - independent adversarial review completed with verdict `PASS`
   - Captain accepted T31 as `PASS` on `2026-05-17`
   - T31 is now marked complete in `docs/04_task_board.md`
   - next unique task: `T39: Training-chain CPU-only clean-environment draft lock and dry-run bootstrap`
