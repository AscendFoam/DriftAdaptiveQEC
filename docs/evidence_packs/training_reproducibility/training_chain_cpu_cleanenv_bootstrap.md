# Training-Chain CPU-Only Clean-Environment Bootstrap

## 1. Scope

This document records the bounded `T39` clean-environment bootstrap for the CPU-only training-chain lane.

It does:

- create a new clean Python `3.12` environment outside `DLEnv`
- install the minimal draft CPU-only dependencies
- run only dry-run/import-level verification

It does not:

- run training
- run benchmark
- run `.tflite` export/runtime validation
- run hardware or `backend=board`
- claim GPU/CUDA portability
- claim Linux portability
- claim full training reproducibility

## 2. Environment Used

### 2.1 Clean Environment

- Environment path:
  - `D:\Codes\Quantum\DriftAdaptiveQEC\.venvs\t39_train_cpu_py312`
- Interpreter:
  - `D:\Codes\Quantum\DriftAdaptiveQEC\.venvs\t39_train_cpu_py312\Scripts\python.exe`
- Python version:
  - `Python 3.12.7`

### 2.2 Creation Command

```powershell
& 'C:\ProgramData\anaconda3\python.exe' -m venv .venvs/t39_train_cpu_py312
```

Why this interpreter:

- `T31` established Python `3.12.x` as the practical Windows-first baseline.
- This environment is separate from `DLEnv`.
- `.venvs/` is ignored by repo policy and was not committed.

## 3. Installed Draft CPU Dependencies

### 3.1 Install Command

```powershell
& '.venvs\t39_train_cpu_py312\Scripts\python.exe' -m pip install numpy PyYAML
```

Note:

- the first sandboxed install attempt failed because outbound package download was blocked
- T39 then used an approved escalated install for this exact command
- no package was installed into `DLEnv`

### 3.2 Installed Versions

```text
numpy==2.4.5
PyYAML==6.0.3
pip==24.2
```

### 3.3 Draft Dependency Artifact

- `requirements-train-cpu-win-py312.txt`

This artifact is:

- Windows-first
- Python-3.12-specific
- CPU-only
- limited to the dry-run/import-level training-chain scope verified in T39

This artifact is not yet:

- a cross-OS lock
- a GPU/CUDA lock
- proof that full training execution succeeds in a clean environment

## 4. Verification Commands

T39 ran only the commands allowed by the task package.

### 4.1 Static Dataset Builder Dry Run

Command:

```powershell
& '.venvs\t39_train_cpu_py312\Scripts\python.exe' -m cnn_fpga.data.dataset_builder --config cnn_fpga/config/experiment_static_theta_v2.yaml --dry-run
```

Result:

- passed
- printed a dataset build plan for `artifacts/datasets/static_theta_v2`
- did not proceed into real dataset generation because `--dry-run` returned early

### 4.2 Runtime Dataset Builder Dry Run

Command:

```powershell
& '.venvs\t39_train_cpu_py312\Scripts\python.exe' -m cnn_fpga.data.runtime_dataset_builder --config cnn_fpga/config/experiment_runtime_b_residual.yaml --dry-run
```

Result:

- passed
- printed a runtime dataset build plan for `artifacts/datasets/runtime_b_residual_v1`
- did not proceed into scenario capture or dataset writing because `--dry-run` returned early

### 4.3 Training Entrypoint Help

Command:

```powershell
& '.venvs\t39_train_cpu_py312\Scripts\python.exe' -m cnn_fpga.model.train --help
```

Result:

- passed
- CLI help printed successfully with:
  - `--config`
  - `--train-split`
  - `--val-split`

## 5. Artifact And Boundary Checks

### 5.1 Required Git Checks

Commands:

```powershell
git diff --name-only -- runs artifacts
git diff --name-only -- requirements-recovery.txt
```

Results:

- both commands returned empty output

Implication:

- T39 did not modify `runs/`
- T39 did not modify `artifacts/`
- T39 did not repurpose or edit `requirements-recovery.txt`

### 5.2 Pre-Existing Dataset Directories

At verification time, these dataset directories already existed in the repo working tree:

- `artifacts/datasets/static_theta_v2`
- `artifacts/datasets/runtime_b_residual_v1`

T39 did not create or modify them. The dry-run commands only printed plans referencing those configured paths.

## 6. What T39 Verifies

T39 verifies:

- a clean Python `3.12` environment can be created outside `DLEnv`
- the CPU-only draft dependency set `numpy + PyYAML` is sufficient for:
  - `dataset_builder --dry-run`
  - `runtime_dataset_builder --dry-run`
  - `train --help`
- the current training-chain entrypoints are still importable in that clean CPU-only lane

## 7. What T39 Does Not Verify

T39 does not verify:

- actual dataset generation in the clean environment
- actual `python -m cnn_fpga.model.train --config ...` execution
- NumPy-backend training quality or artifact compatibility
- torch-backed training portability
- GPU/CUDA portability
- Linux portability
- `.tflite` export/runtime
- real-board or mock-HIL execution

## 8. Recommended Interpretation

`R11` is narrowed again:

- T31 proved the CPU-only lane was plausible by config and import evidence
- T39 proves a clean Windows Python `3.12` environment can install a minimal CPU-only dependency set and pass the bounded dry-run/import-level checks

`R11` is not closed:

- T39 still does not run real training in the clean environment
- T39 therefore must not be treated as full training reproducibility proof
