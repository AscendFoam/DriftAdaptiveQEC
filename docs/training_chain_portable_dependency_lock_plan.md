# Training-Chain Portable Dependency Lock Plan

## 1. Scope And Boundary

This document is a planning/evidence artifact for `T31`.

It does:

- inventory the current local training interpreters
- record package evidence from the machine that currently hosts the repo
- map the training-chain entrypoints used by the current configs
- propose a portable dependency-lock strategy

It does not:

- install, upgrade, or remove packages
- prove that a clean environment has already been rebuilt
- create a new lockfile
- repurpose `requirements-recovery.txt`
- validate `.tflite`, benchmark, or real-board paths

## 2. Current Local Interpreter Inventory

| Interpreter | Version | Relevant package evidence | Role in T31 |
| --- | --- | --- | --- |
| `C:\ProgramData\anaconda3\envs\DLEnv\python.exe` | `Python 3.12.9` | `numpy=2.2.4`, `PyYAML=6.0.2`, `torch=2.8.0.dev20250405+cu128`, `torch.cuda.is_available()=True` | Current local GPU-capable training interpreter; local fact only |
| `C:\ProgramData\anaconda3\python.exe` | `Python 3.12.7` | `numpy=1.26.4`, `PyYAML=6.0.1`, `torch=missing` | Relevant fallback / CPU-only evidence lane |
| `C:\Python313\python.exe` | `Python 3.13.7` | `yaml=present`, `numpy=missing`, `torch=missing` | Not currently sufficient for the training chain |

## 3. Package Evidence Summary

### 3.1 Local `DLEnv` Facts

- `DLEnv` contains a large mixed environment, not a training-only environment.
- `pip freeze` shows many unrelated packages and many `file:///C:/...` Conda-local build references.
- The installed `torch` is a nightly/dev CUDA build, not a stable release pin.

Implication:

- `DLEnv` is valid as local evidence for "this machine can currently run a GPU-capable torch lane".
- `DLEnv` is not suitable to copy directly into a portable canonical lockfile.

### 3.2 Base Anaconda Facts

- Base Anaconda has `numpy` and `PyYAML`.
- Base Anaconda does not have `torch`.
- The following entrypoints still resolve and print `--help` successfully in base Anaconda:
  - `python -m cnn_fpga.model.train --help`
  - `python -m cnn_fpga.data.dataset_builder --help`
  - `python -m cnn_fpga.data.runtime_dataset_builder --help`

Implication:

- The current training-chain code is not import-time hard-wired to `torch`.
- A CPU-only lock candidate is plausible for the current config family.

### 3.3 System Python Facts

- System Python `3.13.7` does not currently have `numpy`.
- It is therefore not a usable training-chain baseline on this machine.

Implication:

- The portable baseline should target Python `3.12.x`, not the current system `3.13.x`.

## 4. Training Entrypoint Dependency Map

## 4.1 Shared Entrypoints

Current training-chain entrypoints are:

- `python -m cnn_fpga.data.dataset_builder`
- `python -m cnn_fpga.data.runtime_dataset_builder`
- `python -m cnn_fpga.model.train`

Direct non-stdlib imports visible from the inspected files:

- `numpy`
- `yaml` through `cnn_fpga.utils.config.load_yaml_config()`
- optional `torch` inside `cnn_fpga.model.tiny_cnn` only when `backend=torch`

Important boundary:

- `cnn_fpga.utils.config` contains a limited YAML fallback parser, so `PyYAML` is not the only possible parser path.
- For portability and readability, `PyYAML` should still be treated as a real direct dependency for the training chain.

## 4.2 Static Theta Training Path

Primary config:

- `cnn_fpga/config/experiment_static_theta_v2.yaml`

Path shape:

1. dataset build:
   - `python -m cnn_fpga.data.dataset_builder --config cnn_fpga/config/experiment_static_theta_v2.yaml`
2. training:
   - `python -m cnn_fpga.model.train --config cnn_fpga/config/experiment_static_theta_v2.yaml`

External dependency observations:

- `dataset_builder.py` imports `numpy` and the YAML config loader.
- `train.py` imports `numpy`, config helpers, and `TinyCNNConfig` / `fit_tiny_cnn`.
- The config sets `training.model_type: tiny_cnn`.
- The config does not set `training.tiny_cnn.backend` or `training.tiny_cnn.device`.
- `tiny_cnn.py` defaults to `backend="numpy"` and `device="auto"`.

T31 conclusion for this path:

- Static-theta training is currently CPU-capable by config semantics.
- `torch` is optional for this path unless a future config explicitly selects `backend=torch`.

## 4.3 Residual-B Training Path

Primary config:

- `cnn_fpga/config/experiment_runtime_b_residual.yaml`

Path shape:

1. runtime-consistent dataset build:
   - `python -m cnn_fpga.data.runtime_dataset_builder --config cnn_fpga/config/experiment_runtime_b_residual.yaml`
2. training:
   - `python -m cnn_fpga.model.train --config cnn_fpga/config/experiment_runtime_b_residual.yaml`

Config facts:

- `runtime_dataset.reference_mode: static_linear`
- `runtime_dataset.teacher_mode: window_variance`
- `runtime_dataset.label_semantics: residual_b`
- `slow_loop.inference_service.backend: artifact_npz`

External dependency observations:

- `runtime_dataset_builder.py` imports many local runtime/benchmark modules, but its top-level external dependency footprint remains `numpy` + YAML in the inspected path.
- Base Anaconda can resolve `runtime_dataset_builder --help` without `torch`.
- The config still does not set `training.tiny_cnn.backend` or `device`.

T31 conclusion for this path:

- The residual-b dataset-and-train path is also CPU-capable by current config semantics.
- The local `DLEnv` torch lane is optional acceleration/local evidence, not the only valid path implied by the config.

## 4.4 Gated V5 / Teacher-Feature Training Path

Primary config:

- `cnn_fpga/config/experiment_runtime_b_residual_norm_gated_teacher_v5.yaml`

Inheritance / delta:

- this config inherits `experiment_runtime_b_residual.yaml`
- it changes teacher-feature layout and scalar fusion details
- it does not introduce a new external package family
- it does not force `backend=torch` or `device=cuda`

Config facts:

- `runtime_dataset.teacher_prediction_layout: scalar_branch`
- `runtime_dataset.teacher_params_layout: scalar_branch`
- `runtime_dataset.teacher_deltas_layout: scalar_branch`
- `training.tiny_cnn.scalar_fusion_mode: gated`

T31 conclusion for this path:

- Gated-v5 training stays on the same training-chain dependency family as residual-b.
- Its semantic delta is feature/layout/config-level, not dependency-level.

Important exclusion:

- `cnn_fpga.benchmark.run_p4_teacher_representation_paired` is a benchmark/evidence entrypoint, not part of the portable training lock scope for T31.

## 5. Proposed Lock Strategy

## 5.1 Recommended Structure

Use a two-lane lock strategy instead of one mixed file:

1. a portable CPU training lane
2. a local GPU/CUDA lane

Rationale:

- the current configs do not require `torch` by default
- the current local `torch` install is a dev CUDA build with machine-specific provenance
- mixing these into one canonical lock would overstate portability

## 5.2 CPU-Only Lock Candidate

Target:

- Python `3.12.x`
- Windows first
- `numpy` + `PyYAML` as the initial direct dependency floor

What this candidate is intended to cover:

- `cnn_fpga.data.dataset_builder`
- `cnn_fpga.data.runtime_dataset_builder`
- `cnn_fpga.model.train`
- current configs that remain on the default NumPy backend

What must be verified in the follow-up task before calling it real:

- `python -m cnn_fpga.data.dataset_builder --config ... --dry-run`
- `python -m cnn_fpga.data.runtime_dataset_builder --config ... --dry-run`
- `python -m cnn_fpga.model.train --help`

Why this is only a candidate:

- T31 did not execute a clean-environment rebuild
- T31 did not run dataset generation or training in a new environment

## 5.3 CUDA / Dev-Torch Caveat

Current local GPU facts:

- `torch=2.8.0.dev20250405+cu128`
- `cuda=True`

Portable-lock interpretation:

- this should remain a local evidence lane, not the canonical lock
- the nightly/dev build and CUDA provenance must be explicitly labeled as local and non-portable
- if a GPU lock is later needed, it should be generated as a separate local lane with explicit channel/index provenance

Recommended future naming pattern:

- canonical lane: `requirements-train-cpu-*.lock.txt`
- local lane: `requirements-train-gpu-local-*.lock.txt` or `environment-train-gpu-local-*.yml`

## 5.4 Windows vs Linux Constraints

Current evidence is Windows-only:

- interpreter paths are Windows paths
- `pip freeze` contains Windows/Conda-local `file:///C:/...` references
- no Linux interpreter/package evidence was collected in T31

Lock consequence:

- one lockfile should not be presented as cross-OS validated
- CPU lock generation should be per-platform if Linux support is later required
- the current T31 output only justifies a Windows-first plan

## 6. What Can Be Committed Now Vs Local Evidence Only

## 6.1 Safe To Commit Now

- this plan document
- the statement that current configs do not explicitly force `backend=torch`
- the statement that Python `3.12.x` is the current practical baseline
- the recommended split between CPU-portable lane and GPU-local lane
- the future lockfile naming/ownership strategy

## 6.2 Must Stay As Local Evidence

- full raw `pip freeze` output from `DLEnv`
- the exact nightly/dev `torch` build provenance
- local CUDA availability
- absolute interpreter paths
- Conda-local `file:///C:/...` wheel/build references

Reason:

- these facts describe one machine
- they are not portable guarantees
- directly freezing them would overfit the lock to the current workstation

## 7. Clean-Environment Bootstrap Procedure Proposal

Recommended next bounded task:

- create a CPU-only training lock in a fresh Python `3.12` environment
- validate only dry-run/import-level entrypoints first
- defer the GPU-local lane to a separate follow-up

Suggested procedure:

1. create a new clean Python `3.12` environment, not `DLEnv`
2. install only the draft CPU dependencies
3. run:
   - `python -m cnn_fpga.data.dataset_builder --config cnn_fpga/config/experiment_static_theta_v2.yaml --dry-run`
   - `python -m cnn_fpga.data.runtime_dataset_builder --config cnn_fpga/config/experiment_runtime_b_residual.yaml --dry-run`
   - `python -m cnn_fpga.model.train --help`
4. if those pass, generate a Windows CPU lock artifact from that clean environment
5. keep GPU/CUDA work in a separate explicitly local task

Why dry-run first:

- it stays closer to Phase 2 bounded verification
- it avoids silently creating new `artifacts/` during the first reproducibility pass

## 8. Explicit Non-Claims

T31 does not claim:

- that a clean environment has already been rebuilt successfully
- that `DLEnv` is the canonical or required training environment
- that the current `torch` nightly build is portable
- that Linux portability has been validated
- that `.tflite` export/runtime is covered
- that real-board or benchmark environments are covered
- that `requirements-recovery.txt` should be expanded or reused for training

## 9. Recommended Next Bounded Task

Recommended next task after T31:

- `Training-chain CPU-only clean-environment draft lock + dry-run bootstrap`

Recommended scope for that task:

- Allowed outputs:
  - one CPU-only training dependency spec/lock artifact
  - one short bootstrap document or update
- Required verification:
  - clean Python `3.12` environment
  - `dataset_builder --dry-run`
  - `runtime_dataset_builder --dry-run`
  - `train --help`
- Forbidden scope:
  - no benchmark
  - no `.tflite`
  - no real-board
  - no GPU portability claims
  - no expansion of `requirements-recovery.txt`

## 10. Evidence Commands Used In T31

Commands executed during T31:

```powershell
& 'C:\ProgramData\anaconda3\envs\DLEnv\python.exe' --version
& 'C:\ProgramData\anaconda3\python.exe' --version
& 'C:\Python313\python.exe' --version

& 'C:\ProgramData\anaconda3\envs\DLEnv\python.exe' -m pip freeze
& 'C:\ProgramData\anaconda3\python.exe' -m pip freeze

& 'C:\ProgramData\anaconda3\envs\DLEnv\python.exe' -c "import numpy, yaml, torch; print('numpy='+numpy.__version__); print('PyYAML='+yaml.__version__); print('torch='+torch.__version__); print('cuda='+str(torch.cuda.is_available()))"
& 'C:\ProgramData\anaconda3\python.exe' -c "import numpy, yaml; print('numpy='+numpy.__version__); print('PyYAML='+yaml.__version__); import importlib.util; print('torch=' + ('present' if importlib.util.find_spec('torch') else 'missing'))"
& 'C:\Python313\python.exe' -c "import importlib.util; print('numpy=' + ('present' if importlib.util.find_spec('numpy') else 'missing')); print('yaml=' + ('present' if importlib.util.find_spec('yaml') else 'missing')); print('torch=' + ('present' if importlib.util.find_spec('torch') else 'missing'))"

& 'C:\ProgramData\anaconda3\envs\DLEnv\python.exe' -m cnn_fpga.model.train --help
& 'C:\ProgramData\anaconda3\python.exe' -m cnn_fpga.model.train --help
& 'C:\ProgramData\anaconda3\python.exe' -m cnn_fpga.data.dataset_builder --help
& 'C:\ProgramData\anaconda3\python.exe' -m cnn_fpga.data.runtime_dataset_builder --help
```
