# Training-Chain CPU-Only Clean-Environment Minimal Train Smoke

## 1. Scope

This document records the bounded `T40` real-training smoke that reuses the clean CPU-only environment created in `T39`.

It does:

- reuse `.venvs/t39_train_cpu_py312`
- run exactly one real training command
- isolate all T40 outputs under `artifacts/t40_train_smoke/`

It does not:

- claim full training reproducibility
- claim GPU/CUDA portability
- claim Linux portability
- run benchmark
- run `.tflite` export/runtime validation
- run hardware or `backend=board`

## 2. Environment Used

- Environment path:
  - `D:\Codes\Quantum\DriftAdaptiveQEC\.venvs\t39_train_cpu_py312`
- Interpreter:
  - `D:\Codes\Quantum\DriftAdaptiveQEC\.venvs\t39_train_cpu_py312\Scripts\python.exe`
- Python version:
  - `Python 3.12.7`
- Installed packages reused from T39:
  - `numpy==2.4.5`
  - `PyYAML==6.0.3`

## 3. Derived Config Used

- Derived config:
  - `cnn_fpga/config/task_tmp/T40_static_theta_train_smoke.yaml`
- Base config:
  - `cnn_fpga/config/experiment_static_theta_v2.yaml`

T40 keeps the canonical dataset path from the base config and overrides only:

- `paths.model_dir = artifacts/t40_train_smoke/models/static_theta_v2`
- `paths.report_dir = artifacts/t40_train_smoke/reports/static_theta_v2`
- `training.max_train_samples = 1024`
- `training.max_val_samples = 256`
- `training.tiny_cnn.epochs = 3`
- `training.tiny_cnn.patience = 2`

## 4. Training Command

```powershell
& '.venvs\t39_train_cpu_py312\Scripts\python.exe' -m cnn_fpga.model.train --config cnn_fpga/config/task_tmp/T40_static_theta_train_smoke.yaml
```

## 5. Result

The command passed.

- Run name:
  - `tiny_cnn_20260517_144945_b0a63c413dbb`
- Backend recorded by the training report:
  - `numpy`
- Device recorded by the training report:
  - `cpu`
- Train sample count:
  - `1024`
- Validation sample count:
  - `256`
- Train MSE:
  - `14.369049842413549`
- Validation MSE:
  - `16.11236397832407`

Generated outputs:

- Model artifact:
  - `artifacts/t40_train_smoke/models/static_theta_v2/tiny_cnn_20260517_144945_b0a63c413dbb.npz`
- Train report:
  - `artifacts/t40_train_smoke/reports/static_theta_v2/tiny_cnn_20260517_144945_b0a63c413dbb_train_report.json`

## 6. Boundary Checks

Required git checks:

```powershell
git diff --name-only -- artifacts/models artifacts/reports
git diff --name-only -- runs
git diff --name-only -- requirements-recovery.txt
```

Results:

- all three commands returned empty output
- no canonical historical model artifacts were modified
- no canonical historical report artifacts were modified
- no `runs/` output was modified
- no `requirements-recovery.txt` change occurred

T40-generated outputs were verified to exist only under:

- `artifacts/t40_train_smoke/models/static_theta_v2/`
- `artifacts/t40_train_smoke/reports/static_theta_v2/`

## 7. What T40 Verifies

T40 verifies:

- the T39 clean Windows/Python `3.12` CPU-only environment can execute one real `tiny_cnn` training smoke
- `base_config` inheritance works for a task-scoped derived config
- canonical dataset inputs can be reused without redirecting training output into canonical historical artifact directories
- the current clean-env smoke path records `backend=numpy` and `device=cpu` in the produced report

## 8. What T40 Does Not Verify

T40 does not verify:

- full training reproducibility
- quality stability beyond one bounded smoke
- GPU/CUDA portability
- Linux portability
- torch-backed training portability
- `.tflite` export/runtime
- benchmark readiness
- hardware or real-board validation
