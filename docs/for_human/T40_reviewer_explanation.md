# T40 Reviewer Explanation: Review Rationale and Task Context

## 1. What T40 Is Trying To Accomplish (Plain Language)

T40 is a "can we really train?" smoke test.

Before T40, the project had two related tasks completed:

- **T31** mapped out what the training chain depends on and proposed a two-lane strategy (CPU-portable lane + GPU-local lane).
- **T39** created a clean Python 3.12 environment on Windows with only `numpy` and `PyYAML`, then verified that the training script could at least start up (import modules, parse config, do dry-runs).

But T39 stopped at dry-run level. It never actually ran a training loop. The risk register (R11) explicitly noted that "real clean-environment training execution remains unverified."

T40 closes that gap by running exactly one small training job in that clean environment. It uses a temporary config that:
- Keeps reading from the existing dataset
- Writes output to a throwaway directory (not the historical artifact paths)
- Runs only 3 epochs on 1024 training samples instead of the full training run

If this succeeds, we know the clean environment can actually train, not just import.

## 2. Detailed Implementation Explanation

### 2.1 Task Goal

Execute one bounded real-training smoke in the T39 clean CPU-only environment (`.venvs/t39_train_cpu_py312/`), with all output isolated to T40-specific directories, without touching canonical historical artifacts.

### 2.2 Task Flow

1. **Read required inputs**: The worker read all 16 documents listed in the task package, including the canonical training config, source code, and T39 bootstrap/review docs.

2. **Create derived config**: The worker created `cnn_fpga/config/task_tmp/T40_static_theta_train_smoke.yaml` with:
   ```yaml
   base_config: ../experiment_static_theta_v2.yaml
   paths:
     model_dir: artifacts/t40_train_smoke/models/static_theta_v2
     report_dir: artifacts/t40_train_smoke/reports/static_theta_v2
   training:
     max_train_samples: 1024
     max_val_samples: 256
     tiny_cnn:
       epochs: 3
       patience: 2
   ```
   This leverages the existing `base_config` inheritance mechanism in `config.py:load_yaml_config()`, which recursively loads the base config and then deep-merges the overrides. The derived config only overrides four values — everything else (conv_channels, kernel_size, hidden_dim, batch_size, learning_rate, weight_decay, label_weights, etc.) is inherited from the canonical config.

3. **Execute training**: The worker ran:
   ```powershell
   .venvs\t39_train_cpu_py312\Scripts\python.exe -m cnn_fpga.model.train --config cnn_fpga/config/task_tmp/T40_static_theta_train_smoke.yaml
   ```
   Training completed successfully with:
   - Backend: `numpy` (CPU-only, no torch/CUDA)
   - Train MSE: 14.37, Val MSE: 16.11
   - Per-label R2: sigma=0.73, mu_q=0.71, mu_p=0.88, theta_deg=0.24 (theta_deg is poor because only 3 epochs)

4. **Verify boundaries**: Three git checks confirmed no canonical paths were touched:
   - `git diff --name-only -- artifacts/models artifacts/reports` → empty
   - `git diff --name-only -- runs` → empty
   - `git diff --name-only -- requirements-recovery.txt` → empty

### 2.3 Code/Config Changes

| File | Change Type | Description |
|---|---|---|
| `cnn_fpga/config/task_tmp/T40_static_theta_train_smoke.yaml` | New | Derived config inheriting from canonical, overriding output paths and smoke-scale knobs |
| `docs/evidence_packs/training_reproducibility/training_chain_cpu_cleanenv_train_smoke.md` | New | Smoke execution record with environment, command, results, boundary checks |
| `docs/review/T40_review.md` | New (worker self-check, now overwritten by reviewer) | Worker pre-review documenting scope compliance |
| `docs/for_human/T40_explanation.md` | New | Human-facing task explanation |
| `docs/tasks/Phase2/T40_...md` | Modified | Worker Output and Verification Record sections filled in |

No source code was modified. No canonical config was modified. No historical artifacts were modified.

### 2.4 Significance for Future Development

T40's contribution to the project is narrow but important:

1. **R11 narrowing**: The risk register entry R11 ("training chain clean-environment execution unverified") is further narrowed. The clean CPU-only environment has now progressed from "can import and dry-run" to "can execute one real training smoke." This does not close R11 (full training reproducibility, GPU/CUDA, Linux, torch paths all remain unverified), but it is the strongest evidence so far that the CPU-only portable training lane is viable.

2. **Derived config pattern established**: The `base_config` inheritance pattern used here (`task_tmp/T40_*.yaml` inheriting from canonical config and overriding only output paths + scale) can be reused for future smoke/reproducibility tasks without creating config divergence.

3. **No regression risk**: Because T40 added no new code and wrote to isolated directories, it has zero risk of breaking existing benchmark, HIL, or training paths.

## 3. Why the Review Verdict Is PASS

The review verdict is **PASS** (not PASS_WITH_WARNINGS, not BLOCK) for the following reasons:

1. **Task goal fully met**: One bounded real CPU-only training smoke was executed in the clean environment. Output was isolated to T40-specific paths. Canonical artifacts were untouched.

2. **No forbidden scope violations**: The worker did not modify source code, canonical configs, `docs/02_experiment_plan.md`, `requirements-recovery.txt`, benchmark configs, or any file outside the allowed list. No benchmark, `.tflite`, hardware, or cleanup was executed.

3. **No fake completion or overclaim**: Documentation explicitly lists what T40 does NOT verify (full reproducibility, GPU/CUDA, Linux, torch, `.tflite`, benchmark, real-board). The worker did not mark the task board or handoff as completed. The per-label R2 values honestly show that `theta_deg` is poorly fit at smoke scale.

4. **Verification is adequate for a smoke task**: The worker ran all required verification commands, inspected the produced report JSON, and checked output isolation. The reviewer independently confirmed all git boundary checks and inspected the derived config, source code, and report.

5. **No mock, stub, or hardcoded output**: The training used real data (`artifacts/datasets/static_theta_v2/train.npz`, `val.npz`), real config inheritance, and produced a genuine model artifact and training report with per-epoch loss history.

The three non-blocking issues (N1 worker pre-review file overlap, N2 macOS paths in dataset manifest, N3 R11 update deferred to Captain) are all minor observations that do not affect the validity of the smoke evidence or the honesty of the documentation. None warranted a downgrade to PASS_WITH_WARNINGS.
