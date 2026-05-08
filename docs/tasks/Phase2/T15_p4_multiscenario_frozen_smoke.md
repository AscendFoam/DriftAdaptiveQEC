# T15: P4 Multi-Scenario Frozen Baseline Bounded Smoke

Task ID: `T15`

Goal: 基于 `T14` 产出的 protocol 文档，运行一个有边界的 P4 多场景 frozen baseline smoke，并记录证据。

Why now: 只有在 `T14` 明确 run matrix、命令与边界后，才能把 `T9` 的单场景证据扩展到更接近 development benchmark 的多场景证据。

Allowed files:

- `docs/tasks/Phase2/T15_p4_multiscenario_frozen_smoke.md`
- `docs/P4_benchmark_development_protocol.md`
- `docs/P4_benchmark_recovery_bootstrap.md`
- `docs/04_task_board.md`
- `docs/07_handoff.md`
- `docs/08_risks_and_open_questions.md`
- 新产生的 `runs/p4_benchmark/...` 输出目录

Forbidden scope:

- 不改 benchmark runner 代码
- 不改 frozen baseline set
- 不改场景定义
- 不改 ParamMapper
- 不改训练 artifact
- 不运行超出 `T14` run matrix 的长跑
- 不把结果写成正式 paper 结论

Inputs to read:

- `docs/tasks/Phase2/T14_p4_frozen_benchmark_protocol_audit.md`
- `docs/P4_benchmark_development_protocol.md`
- `docs/P4_benchmark_recovery_bootstrap.md`
- `docs/03_hil_p4_boundary_audit.md`
- `cnn_fpga/config/p4_multiscenario_recovery_smoke.yaml`

Expected output:

- 一个 bounded P4 smoke run 目录
- 文档记录：
  - 命令
  - 解释器
  - config
  - scenarios / modes / repeats / seed pairing
  - backend 与 inference artifact type
  - run dir
  - summary 与 comparison 的关键字段
  - 结论边界

Verification:

- 按 `T14` 指定命令运行。
- 检查新 run 中的 `summary.json`、`comparison.csv` 与各 mode repeat 的 `hil_summary.json`。

Docs to update:

- `docs/P4_benchmark_development_protocol.md`
- `docs/P4_benchmark_recovery_bootstrap.md`
- `docs/07_handoff.md`
- `docs/08_risks_and_open_questions.md`
- `docs/04_task_board.md`（Captain 整合阶段）

Reviewer type: `normal`

