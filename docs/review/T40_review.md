# Review: T40

Verdict: PASS

## Summary

T40 ran exactly one bounded CPU-only real-training smoke in the T39 clean environment, created a properly isolated derived config, and produced output only under `artifacts/t40_train_smoke/`. Canonical historical artifacts, source code, and configs remain untouched. Documentation accurately scopes what T40 does and does not verify.

## Blocking Issues

None.

## Non-Blocking Issues

N1 Worker pre-review file overlap: The worker wrote a self-check document to `docs/review/T40_review.md`, which the task package lists as expected worker output. The content was transparently labeled "Worker Pre-Review ... pending independent adversarial review" and did not attempt to pass as adversarial review. The reviewer has now overwritten this file with the actual adversarial review. The overlap is a task-package design artifact, not a worker error. **Status: accepted.**

N2 Dataset manifest contains macOS paths: The training report at `artifacts/t40_train_smoke/reports/static_theta_v2/tiny_cnn_20260517_144945_b0a63c413dbb_train_report.json` includes `dataset_manifest.files` with macOS-style paths from the original data generation machine. These paths are historical metadata carried over from `manifest.json` and are not used for data loading at runtime. No functional impact, but a reader inspecting the report might be confused. **Status: accepted — this is an inherited artifact from legacy data generation, not introduced by T40.**

N3 R11 narrowing not yet recorded: T40 provides evidence that the clean CPU-only environment can execute a real training smoke, not just dry-run/import checks. This further narrows R11. However, updating `docs/08_risks_and_open_questions.md` is Captain's governance responsibility, not the worker's or reviewer's scope. **Status: deferred to Captain.**

## Missing Tests

None expected. T40 is a bounded reproducibility smoke, not a code-change task. The verification record covers:
- clean environment reuse (Python 3.12.7, numpy 2.4.5, PyYAML 6.0.3)
- dataset shape inspection before smoke
- one real training command execution
- produced report inspection (backend, device, sample counts, metrics)
- git boundary checks (all empty — canonical artifacts untouched)
- output isolation check (files only under `artifacts/t40_train_smoke/`)

## Suspicious Implementation Details

None found. Specific checks performed:

1. **Derived config correctness**: `cnn_fpga/config/task_tmp/T40_static_theta_train_smoke.yaml` uses `base_config: ../experiment_static_theta_v2.yaml`. Relative path resolves correctly from `cnn_fpga/config/task_tmp/` to `cnn_fpga/config/experiment_static_theta_v2.yaml`. The `_deep_merge` in `config.py` handles nested dict overrides, so `training.tiny_cnn.epochs` and `training.tiny_cnn.patience` override only those two fields while preserving the rest of `training.tiny_cnn` (conv_channels, kernel_size, hidden_dim, batch_size, learning_rate, weight_decay, label_weights, backend, device).

2. **Output path isolation**: Config overrides `paths.model_dir` and `paths.report_dir` to T40-isolated paths. The `train.py` entrypoint uses `get_path(config, "model_dir", ...)` and `get_path(config, "report_dir", ...)` which read from the merged config. Verified via `git diff --name-only` and filesystem inspection that canonical `artifacts/models/` and `artifacts/reports/` are untouched.

3. **Smoke scale reduction**: The derived config only touches `max_train_samples`, `max_val_samples`, `epochs`, and `patience` — all existing knobs from the canonical config. No new training regime was invented.

4. **Training report authenticity**: The report JSON at `artifacts/t40_train_smoke/reports/static_theta_v2/tiny_cnn_20260517_144945_b0a63c413dbb_train_report.json` contains genuine per-epoch loss history with 3 epochs of decreasing loss. Per-label metrics show `theta_deg` has poor R2 (0.24 train, 0.17 val), consistent with only 3 epochs of training on a small subset — this is expected for a smoke-scale run.

5. **Source code untouched**: `git diff --name-only` confirms no changes to `cnn_fpga/model/train.py`, `cnn_fpga/model/tiny_cnn.py`, `cnn_fpga/utils/config.py`, or `cnn_fpga/config/experiment_static_theta_v2.yaml`.

6. **No forbidden scope violations**: No benchmark, `.tflite`, hardware, cleanup, GPU/CUDA, Linux, or broader training activity detected.

## Scope Boundary Verification

| Forbidden item | Evidence |
|---|---|
| `docs/02_experiment_plan.md` modified | `git diff --name-only` shows no change |
| `cnn_fpga/config/experiment_static_theta_v2.yaml` modified | `git diff --name-only` shows no change |
| Canonical `artifacts/models/static_theta_v2/` written | `git diff --name-only -- artifacts/models` empty |
| Canonical `artifacts/reports/static_theta_v2/` written | `git diff --name-only -- artifacts/reports` empty |
| `runs/` modified | `git diff --name-only -- runs` empty |
| `requirements-recovery.txt` modified | `git diff --name-only -- requirements-recovery.txt` empty |
| Source code modified | `git diff --name-only` restricted to task package doc |
| Benchmark / `.tflite` / hardware / cleanup executed | No evidence in any changed or new file |
| Task board or handoff marked complete | `docs/04_task_board.md` and `docs/07_handoff.md` unchanged |

## Overclaim Check

Documentation correctly limits claims:
- `docs/training_chain_cpu_cleanenv_train_smoke.md` Section 7/8 explicitly lists what T40 does and does not verify
- `docs/for_human/T40_explanation.md` states this is not "full training reproducibility" proof
- Worker output in task package does not overclaim

No instance of planned work, mock, stub, placeholder, or hardcoded output being written as completed fact.

## Recommended Next Action

1. Captain should update `docs/08_risks_and_open_questions.md` R11 to reflect T40 further narrowing: clean CPU-only environment can now execute real training smoke, not just dry-run/import.
2. Captain should update `docs/04_task_board.md` and `docs/07_handoff.md` to mark T40 as complete and select the next bounded task.
3. The next task within Milestone 2J could address GPU/CUDA training, Linux portability, `.tflite` runtime smoke, or physical cleanup, but each requires its own bounded task package.
