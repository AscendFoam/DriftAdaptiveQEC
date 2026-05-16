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
