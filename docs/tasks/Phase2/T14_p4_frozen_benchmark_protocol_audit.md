# T14: P4 Frozen Benchmark Protocol Audit

Task ID: `T14`

Goal: 审计 P4 frozen benchmark 的正式口径与 recovery smoke 口径，产出一份可指导 `T15` 的 bounded run plan。

Why now: `T9` 已复验单场景四模式 smoke，`T13` 已允许进入受控开发；但 `docs/02_experiment_plan.md` 仍禁止无准备地启动长跑正式 benchmark。因此先做 protocol audit，避免 Worker 直接改 benchmark 或跑过大任务。

Allowed files:

- `docs/tasks/Phase2/T14_p4_frozen_benchmark_protocol_audit.md`
- `docs/P4_benchmark_development_protocol.md`
- `docs/04_task_board.md`
- `docs/07_handoff.md`
- `docs/08_risks_and_open_questions.md`

Forbidden scope:

- 不修改 `cnn_fpga/benchmark/run_p4_multiscenario_benchmark.py`
- 不修改 `cnn_fpga/config/p4_multiscenario_strong_baselines.yaml`
- 不修改 `cnn_fpga/config/p4_multiscenario_recovery_smoke.yaml`
- 不修改 `cnn_fpga/decoder/param_mapper.py`
- 不改变 baseline 集合、场景定义、seed 口径或 ParamMapper 语义
- 不启动正式长跑 benchmark
- 不把 mock-backed P4 结果写成真板或 `.tflite` 验收

Inputs to read:

- `README.md`
- `AGENTS.md`
- `docs/02_experiment_plan.md`
- `docs/03_hil_p4_boundary_audit.md`
- `docs/P4_benchmark_recovery_bootstrap.md`
- `docs/04_task_board.md`
- `docs/07_handoff.md`
- `docs/08_risks_and_open_questions.md`
- `cnn_fpga/config/p4_multiscenario_recovery_smoke.yaml`
- `cnn_fpga/config/p4_multiscenario_strong_baselines.yaml`
- `cnn_fpga/benchmark/run_p4_multiscenario_benchmark.py`

Expected output:

- 新增或更新 `docs/P4_benchmark_development_protocol.md`
- 文档必须包含：
  - recovery smoke 与正式/development benchmark 的区别
  - 当前 frozen baseline set 的证据来源
  - 推荐给 `T15` 的 bounded run matrix
  - 明确的 run command 草案
  - backend / artifact type / interpreter / manifest 边界
  - 不应启动的长跑范围

Verification:

- 只读核查：
  - `Get-Content -Raw -Encoding UTF8 cnn_fpga/config/p4_multiscenario_recovery_smoke.yaml`
  - `Get-Content -Raw -Encoding UTF8 cnn_fpga/config/p4_multiscenario_strong_baselines.yaml`
  - `Select-String -Path cnn_fpga/benchmark/run_p4_multiscenario_benchmark.py -Pattern "add_argument|--scenario|--mode|--repeats|--paired-seeds"`
- 不要求生成新的 `runs/` 目录。

Docs to update:

- `docs/P4_benchmark_development_protocol.md`
- `docs/07_handoff.md`
- `docs/08_risks_and_open_questions.md`（如新增风险或关闭开放问题）
- `docs/04_task_board.md`（仅在 Captain 整合阶段标记完成并切换下一唯一任务）

Reviewer type: `normal`

## Worker Output Summary

Status: completed as a documentation-only bounded audit.

Produced:

- `docs/P4_benchmark_development_protocol.md`
- `docs/07_handoff.md`
- `docs/08_risks_and_open_questions.md`
- `docs/tasks/Phase2/T14_p4_frozen_benchmark_protocol_audit.md`

Key conclusions:

1. `T9` evidence remains recovery smoke only:
   - `static_bias_theta`
   - `static_linear / window_variance / ekf / cnn_fpga`
   - `repeats=1`
   - `mock + artifact_npz + inproc`
2. `T15` should not reopen the full formal four-scenario run immediately.
3. Recommended `T15` bounded matrix is:
   - scenarios: `static_bias_theta`, `linear_ramp`
   - modes: `ekf`, `ukf`, `constant_residual_mu`, `rls_residual_b`, `hybrid_residual_b`
   - repeats: `2`
   - paired seeds: `true`
   - interpreter: `C:\ProgramData\anaconda3\python.exe`
   - config: `cnn_fpga/config/p4_multiscenario_strong_baselines.yaml`
4. Existing runner CLI already supports the needed controls:
   - `--scenario`
   - `--mode`
   - `--repeats`
   - `--paired-seeds`
   - `--run-dir`
   - `--repeat-start`
   - `--repeat-stop`
   - `--resume-only`

Verification performed:

- read-only check of:
  - `cnn_fpga/config/p4_multiscenario_recovery_smoke.yaml`
  - `cnn_fpga/config/p4_multiscenario_strong_baselines.yaml`
  - `cnn_fpga/benchmark/run_p4_multiscenario_benchmark.py`
- no new benchmark run launched
- no code or benchmark semantics changed
