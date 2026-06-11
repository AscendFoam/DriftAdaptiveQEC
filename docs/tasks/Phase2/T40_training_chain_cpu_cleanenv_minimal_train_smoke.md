# T40: Training-chain CPU-only clean-environment minimal real-training smoke

## Status

- Created by Captain on `2026-05-17`
- Current phase: `Phase 2: Controlled Development`
- Decision state: `Go`
- Task type: bounded reproducibility / minimal real-training smoke

## Why This Task Exists

`T31` proved the training-chain dependency boundary can be described as a portable CPU-vs-GPU two-lane plan. `T39` then proved that a clean Windows/Python `3.12` CPU-only environment can be created, locked at draft level, and used for dry-run/import-level entrypoint verification.

`R11` is still open because no real clean-environment training execution has been completed yet. T40 is the smallest next step that closes part of that gap without silently expanding into benchmark, `.tflite`, real-board, or broad training work.

## Goal

Run exactly one bounded CPU-only clean-environment real-training smoke and capture task-scoped output/report evidence without mutating canonical historical training artifact directories.

This task must not claim:

- full training reproducibility
- GPU/CUDA portability
- Linux portability
- `.tflite` runtime validation
- benchmark evidence
- real-board validation

## Allowed Files

Worker may modify:

- `docs/tasks/Phase2/T40_training_chain_cpu_cleanenv_minimal_train_smoke.md`
- `docs/evidence_packs/training_reproducibility/training_chain_cpu_cleanenv_train_smoke.md`
- `docs/review/T40_review.md`
- `docs/for_human/T40_explanation.md`
- `cnn_fpga/config/task_tmp/T40_static_theta_train_smoke.yaml`

Worker may create task-scoped output directories only:

- `artifacts/t40_train_smoke/models/static_theta_v2/`
- `artifacts/t40_train_smoke/reports/static_theta_v2/`

Worker may reuse, but must not mutate into a new environment design:

- `.venvs/t39_train_cpu_py312/`

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
- `docs/evidence_packs/training_reproducibility/training_chain_portable_dependency_lock_plan.md`
- `docs/evidence_packs/training_reproducibility/training_chain_cpu_cleanenv_bootstrap.md`
- `docs/review/T39_review.md`
- `requirements-train-cpu-win-py312.txt`
- `cnn_fpga/model/train.py`
- `cnn_fpga/model/tiny_cnn.py`
- `cnn_fpga/utils/config.py`
- `cnn_fpga/config/experiment_static_theta_v2.yaml`

## Required Implementation Boundary

T40 must use a derived task-scoped config instead of editing canonical configs in place.

The derived config must:

1. inherit from `cnn_fpga/config/experiment_static_theta_v2.yaml`
2. keep the existing dataset/input path semantics unless a hard blocker is discovered
3. redirect `paths.model_dir` and `paths.report_dir` into the T40-isolated directories listed above
4. reduce training scale to a bounded smoke size

If the base config already exposes bounded knobs such as epochs, patience, `max_train_samples`, or `max_val_samples`, T40 should reduce them conservatively for smoke scope rather than inventing a new training regime.

## Forbidden Scope

Do not:

- modify `docs/02_experiment_plan.md`
- edit canonical historical configs such as `cnn_fpga/config/experiment_static_theta_v2.yaml`
- write to canonical historical training output paths such as:
  - `artifacts/models/static_theta_v2/`
  - `artifacts/reports/static_theta_v2/`
- regenerate datasets unless the entrypoint hard-requires it and the task package is explicitly updated by Captain
- run benchmark
- run `.tflite` export/runtime validation
- call hardware or run `backend=board`
- execute repo cleanup or `git rm`
- modify source code, formal protocol, benchmark configs, baseline/scenario set, seed/repeat policy, or `requirements-recovery.txt`
- expand to residual-b / Gated-v5 / teacher-representation branch work
- claim GPU/CUDA portability, Linux portability, full training reproducibility, benchmark readiness, `.tflite` runtime validation, or real-board validation

## Expected Output

Create or update:

1. `cnn_fpga/config/task_tmp/T40_static_theta_train_smoke.yaml`
   - derived from canonical static-theta config
   - task-scoped isolated output paths
   - bounded smoke-scale training settings
2. `docs/evidence_packs/training_reproducibility/training_chain_cpu_cleanenv_train_smoke.md`
   - interpreter/environment used
   - exact training command
   - exact derived-config path
   - exact output paths
   - what succeeded
   - what remains unverified
3. This task package with Worker Output and Verification Record
4. `docs/review/T40_review.md`
5. `docs/for_human/T40_explanation.md`

## Required Verification

Run exactly one real training smoke:

```powershell
python -m cnn_fpga.model.train --config cnn_fpga/config/task_tmp/T40_static_theta_train_smoke.yaml
```

Also verify:

```powershell
git diff --name-only -- artifacts/models artifacts/reports
git diff --name-only -- runs
git diff --name-only -- requirements-recovery.txt
```

Expected result:

- no diff under canonical historical `artifacts/models/*`
- no diff under canonical historical `artifacts/reports/*`
- no diff under `runs/`
- no diff under `requirements-recovery.txt`

The worker must also verify that T40-generated outputs, if any, exist only under:

- `artifacts/t40_train_smoke/models/static_theta_v2/`
- `artifacts/t40_train_smoke/reports/static_theta_v2/`

If training fails, record the exact blocker honestly. Do not work around by switching to `DLEnv`, GPU/CUDA, or canonical historical artifact paths.

## Docs To Update

- `docs/tasks/Phase2/T40_training_chain_cpu_cleanenv_minimal_train_smoke.md`
- `docs/evidence_packs/training_reproducibility/training_chain_cpu_cleanenv_train_smoke.md`
- `docs/review/T40_review.md`
- `docs/for_human/T40_explanation.md`

## Reviewer Type

Adversarial review.

Focus areas:

- clean environment really reuses the T39 CPU-only lane and does not mutate `DLEnv`
- exactly one bounded real-training smoke was attempted
- output isolation is correct
- canonical historical artifacts were not modified
- no benchmark / `.tflite` / hardware / cleanup scope creep
- no overclaim from a single smoke run

## Captain Notes

This is still a governance-bounded reproducibility task, not a model-development expansion task. Prefer the simplest viable static-theta path. If a blocker appears, keep the artifact trail precise and stop at the blocker rather than widening scope.

## Worker Output

- Read the required T40 task context and required inputs, including:
  - `README.md`
  - `AGENTS.md`
  - `docs/00_project_snapshot.md`
  - `docs/01_legacy_audit.md`
  - `docs/02_experiment_plan.md`
  - `docs/04_task_board.md`
  - `docs/07_handoff.md`
  - `docs/08_risks_and_open_questions.md`
  - `docs/evidence_packs/training_reproducibility/training_chain_portable_dependency_lock_plan.md`
  - `docs/evidence_packs/training_reproducibility/training_chain_cpu_cleanenv_bootstrap.md`
  - `docs/review/T39_review.md`
  - `requirements-train-cpu-win-py312.txt`
  - `cnn_fpga/model/train.py`
  - `cnn_fpga/model/tiny_cnn.py`
  - `cnn_fpga/utils/config.py`
  - `cnn_fpga/config/experiment_static_theta_v2.yaml`
- Reused the clean CPU-only environment created by T39:
  - `.venvs/t39_train_cpu_py312/`
- Created the task-scoped derived config:
  - `cnn_fpga/config/task_tmp/T40_static_theta_train_smoke.yaml`
- Kept dataset/input path semantics inherited from `cnn_fpga/config/experiment_static_theta_v2.yaml`.
- Redirected outputs only to:
  - `artifacts/t40_train_smoke/models/static_theta_v2/`
  - `artifacts/t40_train_smoke/reports/static_theta_v2/`
- Reduced smoke scale only through existing bounded knobs:
  - `training.max_train_samples = 1024`
  - `training.max_val_samples = 256`
  - `training.tiny_cnn.epochs = 3`
  - `training.tiny_cnn.patience = 2`
- Added:
  - `docs/evidence_packs/training_reproducibility/training_chain_cpu_cleanenv_train_smoke.md`
  - `docs/review/T40_review.md`
  - `docs/for_human/T40_explanation.md`
- Updated this task package with worker output and verification record.
- Did not modify:
  - `docs/02_experiment_plan.md`
  - `cnn_fpga/config/experiment_static_theta_v2.yaml`
  - `requirements-recovery.txt`
  - source code
  - benchmark configs
  - formal protocol
  - baseline/scenario/seed policy
- Did not run:
  - benchmark
  - `.tflite` export/runtime validation
  - hardware
  - cleanup
- Did not mark task board or handoff as completed.

## Verification Record

1. Clean environment reuse check:
   - commands:
     - `& '.venvs\t39_train_cpu_py312\Scripts\python.exe' --version`
     - `& '.venvs\t39_train_cpu_py312\Scripts\python.exe' -m pip list`
   - result:
     - `Python 3.12.7`
     - installed packages remained:
       - `numpy 2.4.5`
       - `PyYAML 6.0.3`
       - `pip 24.2`
2. Dataset input inspection before smoke:
   - command:
     - `& '.venvs\t39_train_cpu_py312\Scripts\python.exe' -c "import numpy as np; d=np.load(r'artifacts/datasets/static_theta_v2/train.npz'); v=np.load(r'artifacts/datasets/static_theta_v2/val.npz'); print('train_hist', d['histograms'].shape); print('train_scalar', d['scalar_features'].shape if 'scalar_features' in d.files else (d['histograms'].shape[0],0)); print('train_labels', d['labels'].shape); print('val_hist', v['histograms'].shape); print('val_scalar', v['scalar_features'].shape if 'scalar_features' in v.files else (v['histograms'].shape[0],0)); print('val_labels', v['labels'].shape)"`
   - result:
     - `train_hist (13107, 32, 32)`
     - `train_scalar (13107, 0)`
     - `train_labels (13107, 4)`
     - `val_hist (1638, 32, 32)`
     - `val_scalar (1638, 0)`
     - `val_labels (1638, 4)`
3. Required real training smoke:
   - command:
     - `& '.venvs\t39_train_cpu_py312\Scripts\python.exe' -m cnn_fpga.model.train --config cnn_fpga/config/task_tmp/T40_static_theta_train_smoke.yaml`
   - result:
     - passed
     - printed:
       - `Training complete.`
       - model path under `artifacts/t40_train_smoke/models/static_theta_v2/`
       - `Train MSE: 14.369050`
       - `Val MSE: 16.112364`
4. Produced report inspection:
   - inspected file:
     - `artifacts/t40_train_smoke/reports/static_theta_v2/tiny_cnn_20260517_144945_b0a63c413dbb_train_report.json`
   - result:
     - `run_name = tiny_cnn_20260517_144945_b0a63c413dbb`
     - `model_path = D:\Codes\Quantum\DriftAdaptiveQEC\artifacts\t40_train_smoke\models\static_theta_v2\tiny_cnn_20260517_144945_b0a63c413dbb.npz`
     - `n_train = 1024`
     - `n_val = 256`
     - `training_backend = numpy`
     - `training_device = cpu`
     - `epochs = 3`
     - `patience = 2`
     - `batch_size = 64`
5. Required git boundary checks:
   - commands:
     - `git diff --name-only -- artifacts/models artifacts/reports`
     - `git diff --name-only -- runs`
     - `git diff --name-only -- requirements-recovery.txt`
   - result:
     - all returned empty output
6. Output isolation check:
   - inspected path:
     - `artifacts/t40_train_smoke/`
   - result:
     - generated files exist only under:
       - `artifacts/t40_train_smoke/models/static_theta_v2/`
       - `artifacts/t40_train_smoke/reports/static_theta_v2/`
     - no canonical historical model/report directories were modified
