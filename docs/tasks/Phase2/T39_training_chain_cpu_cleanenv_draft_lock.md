# T39: Training-chain CPU-only clean-environment draft lock and dry-run bootstrap

## Status

- Created by Captain on `2026-05-17`
- Current phase: `Phase 2: Controlled Development`
- Decision state: `Go`
- Task type: bounded reproducibility / clean-environment dry-run

## Why This Task Exists

`T31` produced a portable dependency-lock plan and confirmed that the current training configs do not force `backend=torch` or `device=cuda`. That makes a CPU-only training dependency lane plausible, but it is still only a plan until a clean environment is created and dry-run entrypoints are verified.

T39 is the smallest next step: create a CPU-only draft lock and bootstrap record without running training, benchmark, `.tflite`, real-board, cleanup, or GPU/CUDA validation.

## Goal

Create a Windows-first Python `3.12` CPU-only draft training dependency spec/lock and verify only dry-run/import-level training-chain entrypoints in a clean environment.

This task must not claim full clean-environment training reproducibility unless actual training is run, which is explicitly out of scope for T39.

## Allowed Files

Worker may modify:

- `docs/tasks/Phase2/T39_training_chain_cpu_cleanenv_draft_lock.md`
- `docs/training_chain_cpu_cleanenv_bootstrap.md`
- `docs/review/T39_review.md`
- `docs/for_human/T39_explanation.md`
- `requirements-train-cpu-win-py312.txt`

Worker may create a local ignored clean environment only if needed:

- `.venvs/t39_train_cpu_py312/`

The `.venvs/` directory is ignored and must not be committed.

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
- `docs/training_chain_bootstrap.md`
- `docs/training_chain_portable_dependency_lock_plan.md`
- `requirements-recovery.txt`
- `cnn_fpga/model/train.py`
- `cnn_fpga/data/dataset_builder.py`
- `cnn_fpga/data/runtime_dataset_builder.py`
- `cnn_fpga/config/experiment_static_theta_v2.yaml`
- `cnn_fpga/config/experiment_runtime_b_residual.yaml`
- `cnn_fpga/config/experiment_runtime_b_residual_norm_gated_teacher_v5.yaml`

## Forbidden Scope

Do not:

- modify `docs/02_experiment_plan.md`
- run training
- run benchmark
- run `.tflite` export/runtime validation
- call hardware or run `backend=board`
- execute repo cleanup or `git rm`
- create or modify `runs/` or `artifacts/`
- modify source code, model configs, benchmark configs, formal protocol, baseline/scenario set, seed/repeat policy, or `requirements-recovery.txt`
- claim Linux portability, GPU/CUDA portability, dev-torch portability, full training reproducibility, `.tflite` runtime validation, or real-board validation

## Expected Output

Create or update:

1. `requirements-train-cpu-win-py312.txt`
   - CPU-only direct dependency draft for the current training-chain dry-run scope.
   - Must be clearly separate from `requirements-recovery.txt`.
   - Should avoid local `file:///C:/...` build references.
2. `docs/training_chain_cpu_cleanenv_bootstrap.md`
   - interpreter/environment used
   - dependency installation command
   - exact dry-run/import-level verification commands
   - results
   - what is verified vs not verified
   - known blockers if clean-env setup cannot complete
3. This task package with Worker Output and Verification Record.
4. `docs/review/T39_review.md` as review output.
5. `docs/for_human/T39_explanation.md` for human-readable explanation.

## Required Verification

If clean environment setup succeeds, run only:

```powershell
python -m cnn_fpga.data.dataset_builder --config cnn_fpga/config/experiment_static_theta_v2.yaml --dry-run
python -m cnn_fpga.data.runtime_dataset_builder --config cnn_fpga/config/experiment_runtime_b_residual.yaml --dry-run
python -m cnn_fpga.model.train --help
```

Also verify:

```powershell
git diff --name-only -- runs artifacts
git diff --name-only -- requirements-recovery.txt
```

Expected result for both git checks: empty output.

If `--dry-run` is not supported by an entrypoint, do not silently replace it with a real data/training run. Record the unsupported flag as a blocker or use an import/help-level check that cannot create `runs/` or `artifacts/`.

If package install or environment creation requires network/escalation and is unavailable, stop at a blocker report. Do not work around by mutating `DLEnv`.

## Docs To Update

- `docs/tasks/Phase2/T39_training_chain_cpu_cleanenv_draft_lock.md`
- `docs/training_chain_cpu_cleanenv_bootstrap.md`
- `docs/review/T39_review.md`
- `docs/for_human/T39_explanation.md`

## Reviewer Type

Adversarial review.

Focus areas:

- no training/benchmark/runtime/hardware/cleanup execution
- clean environment is not `DLEnv`
- dependency spec does not repurpose `requirements-recovery.txt`
- no local `DLEnv` / dev torch facts are written as portable guarantees
- no new `runs/` or `artifacts/` outputs
- dry-run commands did not silently become real data generation

## Captain Notes

T39 is still a bounded reproducibility task, not a model-development task. If the environment cannot be created cleanly, a precise blocker report is acceptable and better than mutating the known-good local environments.
